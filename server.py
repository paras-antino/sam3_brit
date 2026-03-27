import base64
import csv
import os
import psutil
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image as PILImage
from pydantic import BaseModel
import supervision as sv
from supervision import ByteTrack
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/paras/sam3/sam3.pt"
OUTPUT_DIR = "/home/paras/sam3/outputs"
MAX_FRAMES = None

# YOLO model for person detection (COCO-trained, auto-downloads ~22 MB).
# Upgrade to "yolov8m.pt" / "yolov8l.pt" for better small-object recall.
YOLO_MODEL = "yolov8s.pt"

# YOLOWorld for open-vocabulary cap / hair-net detection (auto-downloads ~50 MB).
# YOLOWorld is trained on large text-image datasets and understands arbitrary
# object descriptions — far better calibrated for "hair net" than SAM3 text.
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"

# Text synonyms tried for hair-net / cap detection.
# YOLOWorld runs all of them simultaneously; any match = cap present.
CAP_SYNONYMS = [
    "hair net", "hairnet", "hair cap", "head net",
    "mesh cap", "hair covering", "head covering",
    "safety cap", "net cap", "hair snood",
]

# Head crops only for persons whose bbox height is >= this many pixels.
# Below this the person is too far and the full-frame tile result is used instead.
MIN_CROP_PERSON_HEIGHT = 80

# ── Session state ─────────────────────────────────────────────────────────────
session = {
    "running":    False,
    "frame":      None,
    "counts":     {},
    "fps":        0.0,
    "frame_idx":  0,
    "error":      None,
    "labels":     [],
    "rtsp_url":   None,
    "started_at": None,
    "saving":     False,
    "save_path":  None,
    "violations": 0,
    "compliant":  0,
    "cap_label":  "",
    "crops":      {},   # track_id → {img: base64_jpeg, status: "ok"|"violation"|"unknown"}
}
session_lock   = threading.Lock()
session_thread = None

# ── Global writer state ───────────────────────────────────────────────────────
writer      = None
csv_file    = None
csv_writer  = None
writer_lock = threading.Lock()

app = FastAPI()


# ── Request models ────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    rtsp_url:   str
    labels:     List[str]
    confidence: float = 0.30
    every_n:    int   = 5
    save:       bool  = False
    cap_label:  str   = ""


class SaveRequest(BaseModel):
    save: bool


# ── Pre-inference quality enhancement ────────────────────────────────────────
#
# Applied to every tile (SAHI) and every head crop before SAM3 sees them.
# Two stages:
#   1. CLAHE on the LAB luminance channel — normalises local contrast so the
#      subtle rim / texture of a transparent cap becomes visible without
#      blowing out highlights elsewhere in the tile.
#   2. Unsharp mask — sharpens edges so the cap boundary is crisp even when
#      the person is small or slightly motion-blurred.
#
# Neither stage changes the spatial resolution (no upscaling is done here
# because SAM3 will resize to imgsz= anyway), but both change *pixel values*
# in ways that make faint transparent-object features stand out.

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _enhance(img: np.ndarray) -> np.ndarray:
    """
    Enhance a BGR tile/crop for better SAM3 detection.
    Returns a new BGR array; never modifies the input.
    """
    # ── 1. CLAHE on luminance (LAB L-channel) ─────────────────────────────
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = _CLAHE.apply(l)
    out     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── 2. Unsharp mask (amount=0.6, radius=2 px) ─────────────────────────
    # addWeighted(original, 1+amount, blurred, -amount, 0)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
    out  = cv2.addWeighted(out, 1.6, blur, -0.6, 0)

    return out


# ── Real-ESRGAN super-resolution (optional quality layer) ────────────────────
#
# When the `realesrgan` package is installed and model weights are available,
# the full frame is upscaled 2× by RealESRGAN_x2plus before SAHI runs.
# This synthesises genuine detail (not just bicubic interpolation), which
# makes transparent-cap boundaries visible even at 50 m distance.
#
# Graceful fallback: if the package / weights are absent the code continues
# with CLAHE + unsharp masking only (no crash, just a one-time warning).
#
# Auto-download: on first use the ~67 MB RealESRGAN_x2plus.pth weights are
# downloaded from GitHub releases into MODEL_PATH's directory.

def _load_realesrgan(model_dir: str, scale: int = 2):
    """
    Load a RealESRGAN upsampler.  Returns (upsampler, scale) on success,
    or (None, 1) if the package / CUDA is unavailable.
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("[esrgan] `realesrgan` package not found — SR disabled. "
              "Install with: pip install realesrgan basicsr")
        return None, 1

    weight_name = f"RealESRGAN_x{scale}plus.pth"
    weight_path = os.path.join(model_dir, weight_name)

    if not os.path.exists(weight_path):
        url = (f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
               f"v0.2.1/{weight_name}")
        print(f"[esrgan] downloading {weight_name} from GitHub releases…")
        try:
            import urllib.request
            os.makedirs(model_dir, exist_ok=True)
            urllib.request.urlretrieve(url, weight_path)
            print(f"[esrgan] saved to {weight_path}")
        except Exception as exc:
            print(f"[esrgan] download failed: {exc}  — SR disabled")
            return None, 1

    try:
        num_feat   = 64
        num_block  = 23 if scale == 4 else 23
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3,
                             num_feat=num_feat, num_block=num_block,
                             num_grow_ch=32, scale=scale)
        upsampler  = RealESRGANer(
            scale=scale,
            model_path=weight_path,
            model=model_arch,
            tile=256,            # tile-based SR so VRAM stays bounded
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"[esrgan] RealESRGAN_x{scale}plus ready  "
              f"(device={'cuda' if torch.cuda.is_available() else 'cpu'})")
        return upsampler, scale
    except Exception as exc:
        print(f"[esrgan] init failed: {exc}  — SR disabled")
        return None, 1


def _apply_sr(frame: np.ndarray, upsampler) -> np.ndarray:
    """
    Run Real-ESRGAN on `frame` (BGR).  Returns the upscaled BGR image.
    Falls back to the original frame on any error.
    """
    try:
        # RealESRGANer expects RGB uint8
        rgb_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_rgb, _ = upsampler.enhance(rgb_in, outscale=None)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f"[esrgan] enhance error: {exc}")
        return frame


def _scale_detections(dets: sv.Detections, sx: float, sy: float) -> sv.Detections:
    """
    Scale detection boxes back from SR-space to original-frame space.
    sx = orig_w / sr_w,  sy = orig_h / sr_h.
    """
    if dets is None or len(dets) == 0:
        return dets
    scaled = dets.xyxy.copy()
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return sv.Detections(
        xyxy=scaled,
        confidence=dets.confidence,
        class_id=dets.class_id,
    )


# ── SAHI: tiled inference ─────────────────────────────────────────────────────

def _run_sahi(frame: np.ndarray, predictor: SAM3SemanticPredictor,
              labels: list, n_cols: int, n_rows: int,
              overlap: float = 0.20) -> sv.Detections:
    """
    Sliced Aided Hyper Inference (SAHI):
      1. Divide frame into n_cols × n_rows overlapping tiles.
      2. Run SAM3 on each tile independently.
      3. Translate tile-local boxes back to full-frame coordinates.
      4. Merge with NMS.

    Why it helps at distance: a person 50 m away may be only 20 px tall in a
    1080p frame processed at imgsz=1024.  With 2×2 tiling each tile covers half
    the frame width/height, so the same person is ~38 px in the tile inference —
    nearly 2× the resolution, making the transparent cap detectable.
    """
    fh, fw = frame.shape[:2]

    # Tile size that gives n_cols / n_rows coverage with the requested overlap
    tile_w = int(fw / (n_cols - overlap * (n_cols - 1)))
    tile_h = int(fh / (n_rows - overlap * (n_rows - 1)))
    stride_x = int(tile_w * (1 - overlap))
    stride_y = int(tile_h * (1 - overlap))

    all_boxes, all_confs, all_cls = [], [], []

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * stride_x
            y1 = row * stride_y
            x2 = min(x1 + tile_w, fw)
            y2 = min(y1 + tile_h, fh)

            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            # Enhance tile quality before SAM3 sees it
            tile = _enhance(tile)

            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            try:
                predictor.set_image(pil)
                results = predictor(text=labels)
            except Exception as exc:
                print(f"[sahi] tile({row},{col}) error: {exc}")
                continue

            if results and results[0].boxes is not None:
                r     = results[0]
                boxes = r.boxes.xyxy.cpu().numpy().copy()
                # Translate from tile-local to full-frame coordinates
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh)
                all_boxes.append(boxes)
                all_confs.append(r.boxes.conf.cpu().numpy())
                all_cls.append(r.boxes.cls.cpu().numpy().astype(int))

    if not all_boxes:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.vstack(all_boxes),
        confidence=np.hstack(all_confs),
        class_id=np.hstack(all_cls),
    )
    return merged.with_nms(threshold=0.45)


def _run_yolo_sahi(frame: np.ndarray, yolo_model,
                   n_cols: int, n_rows: int,
                   overlap: float = 0.20,
                   person_label_idx: int = 0) -> sv.Detections:
    """
    YOLO-based person detection with SAHI tiling.

    Why YOLO instead of SAM3 text for persons:
      SAM3 text-prompted detection is general-purpose and struggles with
      small/moving persons at distance.  YOLO is trained on 118k COCO images
      specifically for object detection — person recall at small scale is
      dramatically better.  SAM3 is kept only for cap segmentation where
      its open-vocabulary text understanding is genuinely needed.

    Returns sv.Detections where every box has class_id = person_label_idx.
    """
    fh, fw = frame.shape[:2]
    tile_w   = int(fw / (n_cols - overlap * (n_cols - 1)))
    tile_h   = int(fh / (n_rows - overlap * (n_rows - 1)))
    stride_x = int(tile_w * (1 - overlap))
    stride_y = int(tile_h * (1 - overlap))

    all_boxes, all_confs = [], []

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * stride_x
            y1 = row * stride_y
            x2 = min(x1 + tile_w, fw)
            y2 = min(y1 + tile_h, fh)
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            tile = _enhance(tile)   # same CLAHE + unsharp pass

            try:
                res = yolo_model(tile, verbose=False, classes=[0])   # 0 = person in COCO
            except Exception as exc:
                print(f"[yolo] tile({row},{col}) error: {exc}")
                continue

            if res and res[0].boxes is not None and len(res[0].boxes):
                boxes = res[0].boxes.xyxy.cpu().numpy().copy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                boxes[:, [0, 2]]  = np.clip(boxes[:, [0, 2]], 0, fw)
                boxes[:, [1, 3]]  = np.clip(boxes[:, [1, 3]], 0, fh)
                all_boxes.append(boxes)
                all_confs.append(res[0].boxes.conf.cpu().numpy())

    if not all_boxes:
        return sv.Detections.empty()

    boxes_np = np.vstack(all_boxes)
    merged   = sv.Detections(
        xyxy=boxes_np,
        confidence=np.hstack(all_confs),
        class_id=np.full(len(boxes_np), person_label_idx, dtype=int),
    )
    return merged.with_nms(threshold=0.45)


def _run_yolo_world_caps(frame: np.ndarray, cap_yolo,
                         conf: float = 0.12) -> sv.Detections:
    """
    Run YOLOWorld on `frame` to detect all cap/hair-net regions.
    cap_yolo must already have set_classes() called with CAP_SYNONYMS.
    Returns sv.Detections of all found cap boxes (class_id = 0 for all).
    Used for the full-frame distant-worker fallback.
    """
    if cap_yolo is None:
        return sv.Detections.empty()
    try:
        res = cap_yolo(frame, verbose=False, conf=conf)
        if res and res[0].boxes is not None and len(res[0].boxes):
            return sv.Detections(
                xyxy=res[0].boxes.xyxy.cpu().numpy().copy(),
                confidence=res[0].boxes.conf.cpu().numpy(),
                class_id=np.zeros(len(res[0].boxes), dtype=int),
            )
    except Exception as exc:
        print(f"[capyolo] error: {exc}")
    return sv.Detections.empty()


def _run_head_crops(frame: np.ndarray, dets: sv.Detections,
                    cap_yolo,
                    crop_predictor: SAM3SemanticPredictor,
                    labels: list, cap_label: str,
                    width: int, height: int) -> list:
    """
    For each YOLO-detected person, run cap detection on the head crop
    (top 38% of the person bbox).

    Detection priority:
      1. YOLOWorld on the head crop (primary — open-vocab, fast, accurate).
         cap_yolo already has set_classes([cap_label] + CAP_SYNONYMS) so it
         recognises "hair net", "hairnet", "head net", etc. simultaneously.
      2. SAM3 text-prompted on the head crop (fallback if YOLOWorld absent).

    Returns list of (center, has_cap, is_close):
      has_cap = True/False for close workers (bbox height >= MIN_CROP_PERSON_HEIGHT)
      has_cap = None for distant workers (caller falls back to full-frame bbox overlap)
    """
    result = []
    if not cap_label or dets is None or len(dets) == 0:
        return result
    if cap_yolo is None and crop_predictor is None:
        return result
    # Both models run when available; cap detected if either says yes.

    person_cls_ids = {i for i, l in enumerate(labels) if l == "person"}

    for box, cls_id in zip(dets.xyxy,
                            dets.class_id if dets.class_id is not None else []):
        if cls_id not in person_cls_ids:
            continue

        px1, py1, px2, py2 = box.astype(int)
        person_h = py2 - py1
        is_close = person_h >= MIN_CROP_PERSON_HEIGHT

        if is_close:
            head_h = max(int(0.38 * person_h), 20)
            cx1, cy1 = max(0, px1), max(0, py1)
            cx2, cy2 = min(width, px2), min(height, py1 + head_h)
            has_cap  = False
            crop_w   = cx2 - cx1
            crop_h   = cy2 - cy1

            if crop_w >= 10 and crop_h >= 10:
                head_crop = _enhance(frame[cy1:cy2, cx1:cx2])
                try:
                    # ── YOLOWorld on the head crop ────────────────────────────
                    yw_hit = False
                    if cap_yolo is not None:
                        res = cap_yolo(head_crop, verbose=False, conf=0.10)
                        yw_hit = bool(res and res[0].boxes is not None
                                      and len(res[0].boxes) > 0)

                    # ── SAM3 on the head crop ─────────────────────────────────
                    sam_hit = False
                    if crop_predictor is not None and not yw_hit:
                        # Only run SAM3 when YOLOWorld missed — saves time when
                        # YOLOWorld already found the cap.
                        head_crop_pil = PILImage.fromarray(
                            cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
                        crop_predictor.set_image(head_crop_pil)
                        try:
                            crop_res = crop_predictor(
                                text=[cap_label],
                                bboxes=[[0, 0, crop_w, crop_h]],
                            )
                        except TypeError:
                            crop_res = crop_predictor(text=[cap_label])
                        sam_hit = bool(crop_res and crop_res[0].boxes is not None
                                       and len(crop_res[0].boxes) > 0)

                    has_cap = yw_hit or sam_hit
                    print(f"[crop] h={person_h}px  yw={yw_hit}  sam={sam_hit}  cap={has_cap}")
                except Exception as exc:
                    print(f"[crop] error: {exc}")

            result.append((_box_center(box), has_cap, True))
        else:
            result.append((_box_center(box), None, False))

    return result


# ── Cap compliance helpers ────────────────────────────────────────────────────

def _box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _person_has_cap_bbox(person_box, cap_boxes) -> bool:
    px1, py1, px2, py2 = person_box
    head = (px1, py1, px2, py1 + 0.40 * (py2 - py1))
    return any(_boxes_overlap(head, cb) for cb in cap_boxes)


# ── Writer helpers ────────────────────────────────────────────────────────────

def open_writers(ts: str, labels_list: list, width: int, height: int, fps: float,
                 cap_mode: bool = False):
    global writer, csv_file, csv_writer

    scratch = os.path.join(OUTPUT_DIR, f"detection_{ts}.mkv")
    cpath   = os.path.join(OUTPUT_DIR, f"counts_{ts}.csv")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(int(fps)), "-i", "pipe:0",
        "-codec:v", "libx264", "-preset", "ultrafast",
        "-crf", "23", "-pix_fmt", "yuv420p", scratch,
    ]
    with writer_lock:
        new_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        new_csv = open(cpath, "w", newline="")
        new_cw  = csv.writer(new_csv)
        header  = ["frame", "timestamp"] + labels_list + ["total_tracks", "fps"]
        if cap_mode:
            header += ["violations", "compliant"]
        new_cw.writerow(header)
        writer = new_proc; csv_file = new_csv; csv_writer = new_cw

    with session_lock:
        session["save_path"] = scratch
    print(f"[writer] scratch={scratch}  csv={cpath}")


def _remux_to_mp4(scratch: str) -> str:
    mp4 = scratch.replace(".mkv", ".mp4")
    r   = subprocess.run(
        ["ffmpeg", "-y", "-i", scratch, "-codec", "copy", "-movflags", "+faststart", mp4],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if r.returncode == 0:
        os.remove(scratch)
        return mp4
    return scratch


def close_writers():
    global writer, csv_file, csv_writer
    scratch_path = None
    with writer_lock:
        if writer:
            try: writer.stdin.close(); writer.wait()
            except Exception: pass
            writer = None
        if csv_file:
            try: csv_file.close()
            except Exception: pass
            csv_file = None
        csv_writer = None
    with session_lock:
        scratch_path = session.get("save_path")
        session["save_path"] = None
    if scratch_path and scratch_path.endswith(".mkv") and os.path.exists(scratch_path):
        threading.Thread(target=_remux_to_mp4, args=(scratch_path,), daemon=True).start()


def cleanup_loop():
    while True:
        try:
            cutoff = time.time() - 30 * 60
            for fname in os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []:
                if not (fname.endswith(".mp4") or fname.endswith(".mkv") or fname.endswith(".csv")):
                    continue
                fpath = os.path.join(OUTPUT_DIR, fname)
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
        except Exception: pass
        time.sleep(300)


# ── Detection loop ────────────────────────────────────────────────────────────

def detection_loop(rtsp_url: str, labels: list, confidence: float,
                   every_n: int, save_on_start: bool, cap_label: str = ""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap_mode = bool(cap_label)

    # ── Models ────────────────────────────────────────────────────────────────
    #
    # Person detection  → YOLO (yolov8s.pt, COCO-trained)
    #   Purpose-built for detection, handles motion/distance well, fast.
    #
    # Cap / hair-net    → YOLOWorld (yolov8s-worldv2.pt, open-vocabulary)
    #   Set with CAP_SYNONYMS so it recognises "hair net", "hairnet",
    #   "head net", "mesh cap", etc. simultaneously.  Also used for the
    #   full-frame cap scan that handles distant workers.
    #
    # SAM3              → fallback cap predictor only if YOLOWorld absent.
    #
    print(f"[yolo] loading {YOLO_MODEL} for person detection…")
    yolo_model = YOLO(YOLO_MODEL)
    _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    yolo_model(_dummy, verbose=False)
    print("[yolo] ready")

    # Ensure "person" is always at index 0 so YOLO's class_id=0 maps correctly.
    if "person" not in labels:
        labels = ["person"] + labels
    person_label_idx = labels.index("person")

    cap_yolo = None
    crop_predictor = None
    if cap_mode:
        # ── YOLOWorld (optional, loaded if available) ──────────────────────────
        # Open-vocabulary detection; tries all CAP_SYNONYMS simultaneously.
        # Loaded alongside SAM3 (not instead of it) so both run and either
        # finding a cap counts as detected.
        try:
            from ultralytics import YOLOWorld
            print(f"[yoloworld] loading {YOLO_WORLD_MODEL} for cap detection…")
            cap_yolo = YOLOWorld(YOLO_WORLD_MODEL)
            classes = list(dict.fromkeys([cap_label] + CAP_SYNONYMS))
            cap_yolo.set_classes(classes)
            cap_yolo(_dummy, verbose=False)
            print(f"[yoloworld] ready  classes={classes}")
        except Exception as exc:
            print(f"[yoloworld] unavailable ({exc})")
            cap_yolo = None

        # ── SAM3 cap predictor (always loaded in cap mode) ─────────────────────
        # Kept regardless of whether YOLOWorld loaded — the two models run in
        # parallel and cap = YOLOWorld-hit OR SAM3-hit, maximising recall.
        crop_conf = max(confidence * 0.50, 0.12)
        print(f"[sam3] loading head-crop predictor (imgsz=512, conf={crop_conf:.2f})…")
        crop_predictor = SAM3SemanticPredictor(overrides=dict(
            conf=crop_conf, task="segment", mode="predict",
            model=MODEL_PATH, half=True, imgsz=512, verbose=False,
        ))
        print("[sam3] cap predictor ready")

    # ── Real-ESRGAN (optional) ─────────────────────────────────────────────────
    # Load once here so the weight file is downloaded before the stream starts.
    # upsampler=None means SR is unavailable; code falls back transparently.
    sr_model_dir = os.path.dirname(MODEL_PATH)
    upsampler, sr_scale = _load_realesrgan(sr_model_dir, scale=2)

    # ── Open RTSP ─────────────────────────────────────────────────────────────
    print(f"[rtsp] connecting -> {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        with session_lock:
            session["error"] = "Could not open RTSP stream"; session["running"] = False
        return

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[rtsp] {width}x{height} @ {stream_fps:.1f} fps")

    # Tile count: 2×2 for ≤1080p, 3×2 for wide/4K
    n_cols = 3 if width > 2560 else 2
    n_rows = 3 if height > 1440 else 2
    print(f"[sahi] using {n_cols}×{n_rows} tile grid")

    if save_on_start:
        open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"),
                     labels, width, height, stream_fps, cap_mode)
        with session_lock:
            session["saving"] = True

    # ── Tracker + annotators ──────────────────────────────────────────────────
    tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                          minimum_matching_threshold=0.70, frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=50)
    if cap_mode:
        viol_box_ann = sv.BoxAnnotator(thickness=3,
            color=sv.ColorPalette.from_hex(["#FF3333"]))
        viol_lbl_ann = sv.LabelAnnotator(text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#FF3333"]), text_color=sv.Color.WHITE)
        ok_box_ann = sv.BoxAnnotator(thickness=3,
            color=sv.ColorPalette.from_hex(["#00C864"]))
        ok_lbl_ann = sv.LabelAnnotator(text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#00C864"]), text_color=sv.Color.BLACK)

    # ── Async inference thread ─────────────────────────────────────────────────
    # YOLO person detection runs fast (~20-50 ms), but SAM3 cap detection on
    # head crops can take 200-500 ms.  Running everything in a background
    # thread keeps the annotation loop (and browser stream) smooth at full FPS
    # while ByteTrack uses the latest available detections.
    infer: dict = {
        "running":   True,
        "frame":     None,
        "dets":      sv.Detections.empty(),
        "cap_dets":  sv.Detections.empty(),   # full-frame cap boxes (distant workers)
        "head_caps": [],
        "lock":      threading.Lock(),
        "event":     threading.Event(),
    }

    def _inference_worker():
        while infer["running"]:
            infer["event"].wait(timeout=1.0)
            infer["event"].clear()
            if not infer["running"]:
                break
            with infer["lock"]:
                f = infer["frame"]
            if f is None:
                continue
            t0 = time.time()

            # ── YOLO person detection (with optional SR pre-pass) ─────────────
            if upsampler is not None:
                t_sr = time.time()
                f_sr = _apply_sr(f, upsampler)
                sr_h, sr_w = f_sr.shape[:2]
                sx = f.shape[1] / sr_w
                sy = f.shape[0] / sr_h
                print(f"[esrgan] {f.shape[1]}x{f.shape[0]} → {sr_w}x{sr_h}  "
                      f"{(time.time()-t_sr)*1000:.0f} ms")
                sr_n_cols = 3 if sr_w > 2560 else 2
                sr_n_rows = 3 if sr_h > 1440 else 2
                dets = _run_yolo_sahi(f_sr, yolo_model, sr_n_cols, sr_n_rows,
                                      person_label_idx=person_label_idx)
                dets = _scale_detections(dets, sx, sy)
            else:
                dets = _run_yolo_sahi(f, yolo_model, n_cols, n_rows,
                                      person_label_idx=person_label_idx)

            # ── YOLOWorld full-frame cap scan (distant-worker fallback) ────────
            # For workers whose bbox is too small for a reliable head crop,
            # the compliance check falls back to whether any cap box overlaps
            # the top of their person box.  Run YOLOWorld on the full frame
            # once to get those cap boxes.
            cap_dets = sv.Detections.empty()
            if cap_mode:
                cap_dets = _run_yolo_world_caps(f, cap_yolo)

            # ── Per-crop cap detection (close workers) ─────────────────────────
            hcaps = _run_head_crops(f, dets, cap_yolo, crop_predictor,
                                    labels, cap_label, width, height)

            print(f"[infer] persons={len(dets)}  caps_ff={len(cap_dets)}  "
                  f"crops={len(hcaps)}  {(time.time()-t0)*1000:.0f} ms")
            with infer["lock"]:
                infer["dets"]      = dets
                infer["cap_dets"]  = cap_dets
                infer["head_caps"] = hcaps

    infer_thread = threading.Thread(target=_inference_worker, daemon=True)
    infer_thread.start()

    # ── Loop state ────────────────────────────────────────────────────────────
    frame_idx   = 0
    t_start     = time.time()
    t_frame     = t_start
    fps_display = 0.0

    with session_lock:
        session["running"]    = True
        session["started_at"] = datetime.now().isoformat()
        session["error"]      = None

    try:
        while True:
            with session_lock:
                should_run  = session["running"]
                should_save = session["saving"]
            if not should_run:
                break

            with writer_lock:
                writer_active = writer is not None
            if should_save and not writer_active:
                open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"),
                             labels, width, height, stream_fps, cap_mode)
            elif not should_save and writer_active:
                close_writers()

            # ── Read frame ────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                with session_lock:
                    if not session["running"]: break
                cap.release(); time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened(): break
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # ── Trigger async inference every N frames ────────────────────────
            if frame_idx % every_n == 0:
                with infer["lock"]:
                    infer["frame"] = frame.copy()
                infer["event"].set()

            # ── Read latest inference results (non-blocking) ──────────────────
            with infer["lock"]:
                last_sv_dets         = infer["dets"]
                last_cap_dets        = infer["cap_dets"]    # full-frame cap boxes
                last_head_cap_status = infer["head_caps"]

            # ── ByteTrack every frame ─────────────────────────────────────────
            tracked     = tracker.update_with_detections(last_sv_dets)
            counts      = defaultdict(int)
            label_texts = []

            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            class_ids   = tracked.class_id   if tracked.class_id   is not None else []
            confidences = tracked.confidence  if tracked.confidence  is not None else []

            for tid, cls_id, conf in zip(tracker_ids, class_ids, confidences):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            # ── Annotate ──────────────────────────────────────────────────────
            annotated = frame.copy()
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            # ── Cap compliance ────────────────────────────────────────────────
            violations = 0
            compliant  = 0
            viol_set   = set()
            comply_set = set()
            crops      = {}

            if cap_mode and tracked.xyxy is not None and len(tracked.xyxy) > 0:
                n        = len(tracked.xyxy)
                cls_arr  = tracked.class_id   if tracked.class_id   is not None else np.zeros(n, int)
                tid_arr  = tracked.tracker_id if tracked.tracker_id is not None else np.arange(n)
                conf_arr = tracked.confidence if tracked.confidence is not None else np.ones(n)

                person_cls = {i for i, l in enumerate(labels) if l == "person"}

                # Cap boxes from YOLOWorld full-frame scan (distant-worker fallback).
                # last_cap_dets contains boxes detected by YOLOWorld on the whole
                # frame — these are used for workers too far away for a head crop.
                cap_boxes_ff = (list(last_cap_dets.xyxy)
                                if last_cap_dets is not None and len(last_cap_dets) > 0
                                else [])

                def _has_cap(tracked_box) -> bool:
                    tc    = _box_center(tracked_box)
                    max_d = (tracked_box[2] - tracked_box[0]) * 0.9

                    # 1. Head-crop YOLOWorld result (close workers)
                    if last_head_cap_status:
                        best_d, best_v, best_close = float('inf'), None, False
                        for center, hc, is_close in last_head_cap_status:
                            d = _center_dist(tc, center)
                            if d < best_d:
                                best_d, best_v, best_close = d, hc, is_close
                        if best_close and best_v is not None and best_d < max_d:
                            return best_v

                    # 2. Full-frame YOLOWorld cap bbox overlap (distant workers)
                    return _person_has_cap_bbox(tracked_box, cap_boxes_ff)

                for i in range(n):
                    if cls_arr[i] not in person_cls:
                        continue
                    if _has_cap(tracked.xyxy[i]):
                        comply_set.add(i)
                    else:
                        viol_set.add(i)

                violations = len(viol_set)
                compliant  = len(comply_set)

                def _make_dets(idx_set):
                    idx = np.array(list(idx_set))
                    return sv.Detections(xyxy=tracked.xyxy[idx],
                                         confidence=conf_arr[idx],
                                         class_id=cls_arr[idx],
                                         tracker_id=tid_arr[idx])

                if viol_set:
                    vd = _make_dets(viol_set)
                    vl = [f"#{tid_arr[i]} NO CAP" for i in viol_set]
                    annotated = viol_box_ann.annotate(annotated, vd)
                    annotated = viol_lbl_ann.annotate(annotated, vd, labels=vl)

                if comply_set:
                    cd = _make_dets(comply_set)
                    cl = [f"#{tid_arr[i]} CAP OK" for i in comply_set]
                    annotated = ok_box_ann.annotate(annotated, cd)
                    annotated = ok_lbl_ann.annotate(annotated, cd, labels=cl)

                if violations > 0:
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 36), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
                    cv2.putText(annotated,
                                f"  VIOLATION: {violations} worker(s) without head cap",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # Person crop thumbnails for the browser panel
                for i in range(n):
                    if cls_arr[i] not in person_cls:
                        continue
                    tid = int(tid_arr[i])
                    px1, py1, px2, py2 = tracked.xyxy[i].astype(int)
                    px1 = max(0, px1); py1 = max(0, py1)
                    px2 = min(width, px2); py2 = min(height, py2)
                    if (px2 - px1) < 8 or (py2 - py1) < 8:
                        continue
                    crop_img = annotated[py1:py2, px1:px2]
                    _, buf = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    status = ("violation" if i in viol_set
                              else "ok" if i in comply_set else "unknown")
                    crops[str(tid)] = {"img": base64.b64encode(buf).decode(), "status": status}

            # ── Timing ────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            elapsed     = now - t_start

            with session_lock:
                session["frame"]      = annotated.copy()
                session["counts"]     = dict(counts)
                session["fps"]        = fps_display
                session["frame_idx"]  = frame_idx
                session["violations"] = violations
                session["compliant"]  = compliant
                session["crops"]      = crops

            # ── Write to disk ─────────────────────────────────────────────────
            with session_lock:
                currently_saving = session["saving"]
            with writer_lock:
                current_writer = writer; current_cw = csv_writer

            if currently_saving and current_writer is not None:
                try:
                    current_writer.stdin.write(annotated.tobytes())
                    current_writer.stdin.flush()
                except BrokenPipeError:
                    close_writers()
                    with session_lock: session["saving"] = False

            if current_cw is not None:
                row  = [frame_idx, round(elapsed, 2)]
                row += [counts.get(lbl, 0) for lbl in labels]
                row += [len(tracked), round(fps_display, 2)]
                if cap_mode:
                    row += [violations, compliant]
                current_cw.writerow(row)

            if frame_idx % 30 == 0:
                print(f"[loop] frame={frame_idx:05d} fps={fps_display:.1f} "
                      f"counts={dict(counts)}"
                      + (f" viol={violations} ok={compliant}" if cap_mode else ""))

            frame_idx += 1

    except Exception as exc:
        with session_lock: session["error"] = str(exc)
        import traceback; traceback.print_exc()
    finally:
        infer["running"] = False
        infer["event"].set()
        infer_thread.join(timeout=5)
        cap.release()
        close_writers()
        with session_lock:
            session["running"] = False; session["saving"] = False
        print("[loop] stopped")


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/session/start")
def start_session(req: StartRequest):
    global session_thread
    with session_lock:
        if session["running"]:
            raise HTTPException(409, "Session already running.")
        if not req.labels:
            raise HTTPException(400, "Provide at least one label.")
        labels = list(req.labels)
        if req.cap_label:
            if req.cap_label not in labels:
                raise HTTPException(400, f"cap_label must be in labels list.")
            if "person" not in labels:
                labels.insert(0, "person")
        session.update({
            "labels": labels, "rtsp_url": req.rtsp_url,
            "frame": None, "counts": {}, "fps": 0.0, "frame_idx": 0,
            "error": None, "saving": req.save, "save_path": None,
            "violations": 0, "compliant": 0, "cap_label": req.cap_label, "crops": {},
        })
    session_thread = threading.Thread(
        target=detection_loop,
        args=(req.rtsp_url, labels, req.confidence, req.every_n, req.save, req.cap_label),
        daemon=True,
    )
    session_thread.start()
    return {"status": "started", "labels": labels, "cap_mode": bool(req.cap_label)}


@app.post("/session/stop")
def stop_session():
    with session_lock:
        if not session["running"]:
            return {"status": "not running"}
        session["running"] = False; session["saving"] = False

    def _clear():
        time.sleep(4)
        with session_lock:
            session.update({"frame": None, "counts": {}, "fps": 0.0,
                            "frame_idx": 0, "violations": 0, "compliant": 0, "crops": {}})
    threading.Thread(target=_clear, daemon=True).start()
    return {"status": "stopping"}


@app.post("/session/save")
def toggle_save(req: SaveRequest):
    with session_lock:
        if not session["running"]: raise HTTPException(400, "No session running.")
        session["saving"] = req.save
    return {"status": "saving" if req.save else "not saving"}


@app.get("/session/status")
def session_status():
    with session_lock:
        return JSONResponse({
            "running": session["running"], "labels": session["labels"],
            "rtsp_url": session["rtsp_url"], "counts": session["counts"],
            "fps": round(session["fps"], 2), "frame_idx": session["frame_idx"],
            "started_at": session["started_at"], "error": session["error"],
            "saving": session["saving"], "save_path": session["save_path"],
            "violations": session["violations"], "compliant": session["compliant"],
            "cap_label": session["cap_label"],
        })


@app.get("/crops")
def get_crops():
    """Per-person crop thumbnails with compliance status for the browser panel."""
    with session_lock:
        return JSONResponse(session.get("crops", {}))


@app.get("/recordings")
def list_recordings():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if not (fname.endswith(".mp4") or fname.endswith(".mkv")): continue
            fpath = os.path.join(OUTPUT_DIR, fname)
            stat  = os.stat(fpath)
            files.append({"name": fname,
                          "size_mb": round(stat.st_size / 1e6, 1),
                          "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
    return JSONResponse(files)


CHUNK_SIZE = 1024 * 256

@app.api_route("/recordings/{filename}", methods=["GET", "HEAD"])
async def stream_recording(filename: str, request: Request):
    filename = os.path.basename(filename)
    fpath    = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath): raise HTTPException(404, "File not found")
    file_size    = os.path.getsize(fpath)
    range_header = request.headers.get("range")
    mime_type    = "video/mp4" if filename.endswith(".mp4") else "video/x-matroska"
    if request.method == "HEAD":
        return Response(status_code=200, media_type=mime_type, headers={
            "Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Type": mime_type})
    if range_header:
        rv    = range_header.strip().lower().replace("bytes=", "").split("-")
        start = int(rv[0]) if rv[0] else 0
        end   = int(rv[1]) if (len(rv) > 1 and rv[1]) else min(start + CHUNK_SIZE * 8 - 1, file_size - 1)
        end   = min(end, file_size - 1)
        csz   = end - start + 1
        def _iter_r():
            rem = csz
            with open(fpath, "rb") as f:
                f.seek(start)
                while rem > 0:
                    d = f.read(min(CHUNK_SIZE, rem))
                    if not d: break
                    rem -= len(d); yield d
        return StreamingResponse(_iter_r(), status_code=206, media_type=mime_type, headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes", "Content-Length": str(csz), "Content-Type": mime_type})
    def _iter_f():
        with open(fpath, "rb") as f:
            while True:
                d = f.read(CHUNK_SIZE)
                if not d: break
                yield d
    return StreamingResponse(_iter_f(), status_code=200, media_type=mime_type, headers={
        "Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Type": mime_type})


@app.get("/video")
def video_feed():
    def _gen():
        while True:
            with session_lock: frame = session["frame"]
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for stream...", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/resources")
def resources():
    cpu = psutil.cpu_percent(interval=None); ram = psutil.virtual_memory()
    gpu_util = gpu_mem_used = gpu_mem_total = None
    try:
        raw = subprocess.check_output(["nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"]).decode().strip().split(",")
        gpu_util = int(raw[0].strip()); gpu_mem_used = int(raw[1].strip()); gpu_mem_total = int(raw[2].strip())
    except Exception: pass
    return JSONResponse({"cpu_percent": round(cpu, 1), "ram_percent": round(ram.percent, 1),
                         "ram_used_gb": round(ram.used/1e9, 1), "ram_total_gb": round(ram.total/1e9, 1),
                         "gpu_util": gpu_util, "gpu_mem_used": gpu_mem_used, "gpu_mem_total": gpu_mem_total})


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "frontend.html"), encoding="utf-8") as f:
        return f.read()


@app.on_event("startup")
def _startup():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    print("[server] SAM3 detection server ready")
