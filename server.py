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
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/paras/sam3/sam3.pt"
OUTPUT_DIR = "/home/paras/sam3/outputs"
MAX_FRAMES = None

# Motion-adaptive inference: when frame-diff mean exceeds this value,
# inference runs at every (every_n // 2) frames instead of every_n.
MOTION_THRESHOLD = 3.0   # tune if needed; typical values: static~0.5, walking~4-8

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


# ── Optical flow box propagation ──────────────────────────────────────────────

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
)


def _propagate_boxes_with_flow(prev_gray: np.ndarray,
                                curr_gray: np.ndarray,
                                dets: sv.Detections) -> sv.Detections:
    """
    Estimate how each detection box moved using sparse Lucas-Kanade optical flow.
    Returns a new Detections with translated boxes (masks dropped — stale after move).
    Called every non-inference frame so ByteTrack always sees boxes near the object.
    """
    if dets is None or len(dets) == 0:
        return dets

    h, w = prev_gray.shape[:2]
    new_xyxy = dets.xyxy.copy().astype(float)

    for i, box in enumerate(dets.xyxy):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        if (x2 - x1) < 6 or (y2 - y1) < 6:
            continue

        # 4×4 grid of sample points inside the box
        xs = np.linspace(x1 + 3, x2 - 3, 4, dtype=np.float32)
        ys = np.linspace(y1 + 3, y2 - 3, 4, dtype=np.float32)
        pts = np.array([[x, y] for y in ys for x in xs],
                       dtype=np.float32).reshape(-1, 1, 2)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **_LK_PARAMS
        )

        if next_pts is None or status is None:
            continue

        good_old = pts[status.flatten() == 1].reshape(-1, 2)
        good_new = next_pts[status.flatten() == 1].reshape(-1, 2)

        if len(good_old) < 3:
            continue

        motion   = good_new - good_old
        dx       = float(np.median(motion[:, 0]))
        dy       = float(np.median(motion[:, 1]))

        new_xyxy[i] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

    return sv.Detections(
        xyxy=new_xyxy.astype(np.float32),
        confidence=dets.confidence,
        class_id=dets.class_id,
        # masks are NOT propagated — they are position-specific to the inference frame
    )


# ── Cap compliance helpers ────────────────────────────────────────────────────

def _box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _person_has_cap_bbox(person_box, cap_boxes) -> bool:
    """Fallback: does any cap box overlap the top-40% head region of the person?"""
    px1, py1, px2, py2 = person_box
    head = (px1, py1, px2, py1 + 0.40 * (py2 - py1))
    return any(_boxes_overlap(head, cb) for cb in cap_boxes)


def _mask_in_region(mask_hw: np.ndarray, region, frame_shape) -> bool:
    """True if any mask pixel falls inside region (x1,y1,x2,y2)."""
    fh, fw = frame_shape[:2]
    x1 = max(0, int(region[0])); y1 = max(0, int(region[1]))
    x2 = min(fw, int(region[2])); y2 = min(fh, int(region[3]))
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(mask_hw[y1:y2, x1:x2].any())


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
        print(f"[ffmpeg] started PID={new_proc.pid}  scratch={scratch}")

        new_csv = open(cpath, "w", newline="")
        new_cw  = csv.writer(new_csv)
        header  = ["frame", "timestamp"] + labels_list + ["total_tracks", "fps"]
        if cap_mode:
            header += ["violations", "compliant"]
        new_cw.writerow(header)

        writer     = new_proc
        csv_file   = new_csv
        csv_writer = new_cw

    with session_lock:
        session["save_path"] = scratch

    print(f"[writer] scratch → {scratch}  CSV → {cpath}")


def _remux_to_mp4(scratch: str) -> str:
    mp4_path = scratch.replace(".mkv", ".mp4")
    cmd = ["ffmpeg", "-y", "-i", scratch, "-codec", "copy", "-movflags", "+faststart", mp4_path]
    print(f"[remux] {scratch} → {mp4_path}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode == 0:
        os.remove(scratch)
        print(f"[remux] done → {mp4_path}")
        return mp4_path
    else:
        print(f"[remux] FAILED: {result.stderr.decode(errors='replace')}")
        return scratch


def close_writers():
    global writer, csv_file, csv_writer
    scratch_path = None

    with writer_lock:
        if writer is not None:
            try:
                writer.stdin.close(); writer.wait()
            except Exception as exc:
                print(f"[ffmpeg] close error: {exc}")
            writer = None
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception as exc:
                print(f"[csv] close error: {exc}")
            csv_file = None
        csv_writer = None

    with session_lock:
        scratch_path = session.get("save_path")
        session["save_path"] = None

    print("[writer] closed")
    if scratch_path and scratch_path.endswith(".mkv") and os.path.exists(scratch_path):
        threading.Thread(target=_remux_to_mp4, args=(scratch_path,), daemon=True).start()


# ── Background cleanup ────────────────────────────────────────────────────────
def cleanup_loop():
    while True:
        try:
            cutoff = time.time() - 30 * 60
            if os.path.exists(OUTPUT_DIR):
                for fname in os.listdir(OUTPUT_DIR):
                    if not (fname.endswith(".mp4") or fname.endswith(".mkv") or fname.endswith(".csv")):
                        continue
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        print(f"[cleanup] deleted {fname}")
        except Exception as exc:
            print(f"[cleanup] error: {exc}")
        time.sleep(300)


# ── Detection loop ────────────────────────────────────────────────────────────
def detection_loop(rtsp_url: str, labels: list, confidence: float,
                   every_n: int, save_on_start: bool, cap_label: str = ""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap_mode = bool(cap_label)

    # ── Load predictors ───────────────────────────────────────────────────────
    print("[sam3] loading full-frame predictor...")
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=confidence, task="segment", mode="predict",
        model=MODEL_PATH, half=True, imgsz=644, verbose=False,
    ))

    crop_predictor = None
    if cap_mode:
        crop_conf = max(confidence * 0.7, 0.15)
        print(f"[sam3] loading head-crop predictor (imgsz=320, conf={crop_conf:.2f})...")
        crop_predictor = SAM3SemanticPredictor(overrides=dict(
            conf=crop_conf, task="segment", mode="predict",
            model=MODEL_PATH, half=True, imgsz=320, verbose=False,
        ))

    print("[sam3] predictors ready")

    # ── Tracker + annotators ──────────────────────────────────────────────────
    tracker = ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=20,
        minimum_matching_threshold=0.55,   # lenient IOU — handles moderate movement
        frame_rate=15,
    )
    mask_ann  = sv.MaskAnnotator(opacity=0.38)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=50)

    if cap_mode:
        viol_box_ann = sv.BoxAnnotator(
            thickness=3, color=sv.ColorPalette.from_hex(["#FF3333"]))
        viol_lbl_ann = sv.LabelAnnotator(
            text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#FF3333"]),
            text_color=sv.Color.WHITE)
        ok_box_ann = sv.BoxAnnotator(
            thickness=3, color=sv.ColorPalette.from_hex(["#00C864"]))
        ok_lbl_ann = sv.LabelAnnotator(
            text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#00C864"]),
            text_color=sv.Color.BLACK)

    # ── Open RTSP ─────────────────────────────────────────────────────────────
    print(f"[rtsp] connecting -> {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        with session_lock:
            session["error"]   = "Could not open RTSP stream"
            session["running"] = False
        print("[rtsp] ERROR: could not open stream")
        return

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[rtsp] {width}x{height} @ {stream_fps:.1f} fps")

    if save_on_start:
        open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"),
                     labels, width, height, stream_fps, cap_mode)
        with session_lock:
            session["saving"] = True

    # ── Loop state ────────────────────────────────────────────────────────────
    frame_idx   = 0
    t_start     = time.time()
    t_frame     = t_start
    fps_display = 0.0

    # Detection state — two copies so optical flow propagation doesn't corrupt masks
    last_sv_dets_tracked  = sv.Detections.empty()  # propagated positions (for ByteTrack)
    last_inference_masks  = sv.Detections.empty()  # latest raw inference with masks (for display)

    # head-crop compliance: list of (person_box_center, has_cap)
    last_head_cap_status: list = []

    # Optical flow state
    prev_gray: np.ndarray | None = None

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
                print("[loop] stop signal received")
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
                    if not session["running"]:
                        break
                print("[rtsp] stream lost -- reconnecting...")
                cap.release()
                time.sleep(2)
                with session_lock:
                    if not session["running"]:
                        break
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("[rtsp] reconnect failed")
                    break
                prev_gray = None   # reset flow on reconnect
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # ── Grayscale + motion score ──────────────────────────────────────
            curr_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            if prev_gray is not None:
                motion_score = float(cv2.absdiff(curr_gray, prev_gray).mean())

            # Motion-adaptive inference interval:
            # If there's significant motion, run SAM3 twice as often.
            effective_n  = max(1, every_n // 2) if motion_score > MOTION_THRESHOLD else every_n
            run_inference = (frame_idx % effective_n == 0)

            # ── SAM3 full-frame inference ─────────────────────────────────────
            if run_inference:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = PILImage.fromarray(rgb)
                predictor.set_image(pil)
                results = predictor(text=labels)

                if results and results[0].boxes is not None:
                    r      = results[0]
                    boxes  = r.boxes.xyxy.cpu().numpy()
                    confs  = r.boxes.conf.cpu().numpy()
                    clsids = r.boxes.cls.cpu().numpy().astype(int)

                    masks = None
                    if r.masks is not None:
                        try:
                            masks = r.masks.data.cpu().numpy().astype(bool)
                        except Exception:
                            pass

                    fresh = sv.Detections(xyxy=boxes, confidence=confs,
                                          class_id=clsids, mask=masks)
                    last_sv_dets_tracked = sv.Detections(
                        xyxy=boxes.copy(), confidence=confs, class_id=clsids)
                    last_inference_masks = fresh
                else:
                    last_sv_dets_tracked = sv.Detections.empty()
                    last_inference_masks = sv.Detections.empty()

            else:
                # ── Optical flow: propagate boxes to current frame ────────────
                # This keeps ByteTrack fed with boxes near the object's CURRENT
                # position even when SAM3 hasn't run yet, so IOU matching succeeds
                # even for fast-walking workers.
                if prev_gray is not None and len(last_sv_dets_tracked) > 0:
                    last_sv_dets_tracked = _propagate_boxes_with_flow(
                        prev_gray, curr_gray, last_sv_dets_tracked)

            # ── ByteTrack: update every frame ─────────────────────────────────
            tracked     = tracker.update_with_detections(last_sv_dets_tracked)
            counts      = defaultdict(int)
            label_texts = []

            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            class_ids   = tracked.class_id   if tracked.class_id   is not None else []
            confidences = tracked.confidence  if tracked.confidence  is not None else []

            for tid, cls_id, conf in zip(tracker_ids, class_ids, confidences):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            # ── Head-crop re-detection ────────────────────────────────────────
            # Runs on inference frames, using TRACKED positions (Kalman-predicted
            # current location) rather than raw detection boxes.  This ensures the
            # crop is centred on where the person actually IS now, not N frames ago.
            if run_inference and cap_mode and crop_predictor is not None:
                last_head_cap_status = []
                person_cls_ids = {i for i, l in enumerate(labels) if l == "person"}

                for box, cls_id in zip(
                        tracked.xyxy,
                        tracked.class_id if tracked.class_id is not None else []):
                    if cls_id not in person_cls_ids:
                        continue

                    px1, py1, px2, py2 = box.astype(int)
                    person_h = max(py2 - py1, 1)

                    # Head region: top 38%, minimum 20 px tall
                    head_h = max(int(0.38 * person_h), 20)
                    cx1 = max(0, px1)
                    cy1 = max(0, py1)
                    cx2 = min(width,  px2)
                    cy2 = min(height, py1 + head_h)

                    if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                        continue

                    head_crop = frame[cy1:cy2, cx1:cx2]
                    try:
                        crop_predictor.set_image(
                            PILImage.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)))
                        crop_res = crop_predictor(text=[cap_label])
                        has_cap  = bool(crop_res and crop_res[0].boxes is not None
                                        and len(crop_res[0].boxes) > 0)
                    except Exception as exc:
                        print(f"[crop] inference error: {exc}")
                        has_cap = False

                    last_head_cap_status.append((_box_center(box), has_cap))
                    print(f"[crop] person({px1},{py1}) crop={cx2-cx1}x{cy2-cy1} "
                          f"motion={motion_score:.1f} has_cap={has_cap}")

            # ── Annotate ──────────────────────────────────────────────────────
            annotated = frame.copy()

            # Segmentation masks from the most recent inference (semi-transparent)
            if last_inference_masks.mask is not None and len(last_inference_masks) > 0:
                try:
                    annotated = mask_ann.annotate(annotated, last_inference_masks)
                except Exception:
                    pass

            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            # ── Cap compliance overlay ────────────────────────────────────────
            violations = 0
            compliant  = 0

            if cap_mode and tracked.xyxy is not None and len(tracked.xyxy) > 0:
                n        = len(tracked.xyxy)
                cls_arr  = tracked.class_id   if tracked.class_id   is not None else np.zeros(n, int)
                tid_arr  = tracked.tracker_id if tracked.tracker_id is not None else np.arange(n)
                conf_arr = tracked.confidence if tracked.confidence is not None else np.ones(n)

                person_cls = {i for i, l in enumerate(labels) if l == "person"}
                cap_cls    = {i for i, l in enumerate(labels) if l == cap_label}
                cap_boxes_fb = [tracked.xyxy[i] for i in range(n) if cls_arr[i] in cap_cls]

                cap_masks_fb = []
                if (last_inference_masks.mask is not None
                        and last_inference_masks.class_id is not None):
                    cap_cls_global = {i for i, l in enumerate(labels) if l == cap_label}
                    for mi, mc in enumerate(last_inference_masks.class_id):
                        if mc in cap_cls_global and mi < len(last_inference_masks.mask):
                            cap_masks_fb.append(last_inference_masks.mask[mi])

                def _has_cap(tracked_box) -> bool:
                    """
                    Priority order:
                    1. Head-crop result  — closest person match (most reliable)
                    2. Cap mask overlap  — precise even without crop
                    3. Cap bbox overlap  — last resort
                    """
                    tc = _box_center(tracked_box)

                    # 1. Head-crop result
                    if last_head_cap_status:
                        max_d = (tracked_box[2] - tracked_box[0]) * 0.9
                        best_d, best_v = float('inf'), None
                        for center, has_cap in last_head_cap_status:
                            d = _center_dist(tc, center)
                            if d < best_d:
                                best_d, best_v = d, has_cap
                        if best_v is not None and best_d < max_d:
                            return best_v

                    px1, py1, px2, py2 = tracked_box
                    head_region = (px1, py1, px2, py1 + 0.40 * (py2 - py1))

                    # 2. Segmentation mask overlap
                    if cap_masks_fb:
                        return any(_mask_in_region(m, head_region, frame.shape)
                                   for m in cap_masks_fb)

                    # 3. Bbox overlap fallback
                    return _person_has_cap_bbox(tracked_box, cap_boxes_fb)

                viol_idx   = [i for i in range(n)
                              if cls_arr[i] in person_cls and not _has_cap(tracked.xyxy[i])]
                comply_idx = [i for i in range(n)
                              if cls_arr[i] in person_cls and _has_cap(tracked.xyxy[i])]

                violations = len(viol_idx)
                compliant  = len(comply_idx)

                def _make_dets(idx_list):
                    idx = np.array(idx_list)
                    return sv.Detections(xyxy=tracked.xyxy[idx],
                                         confidence=conf_arr[idx],
                                         class_id=cls_arr[idx],
                                         tracker_id=tid_arr[idx])

                if viol_idx:
                    vd = _make_dets(viol_idx)
                    vl = [f"#{tid_arr[i]} NO CAP" for i in viol_idx]
                    annotated = viol_box_ann.annotate(annotated, vd)
                    annotated = viol_lbl_ann.annotate(annotated, vd, labels=vl)

                if comply_idx:
                    cd = _make_dets(comply_idx)
                    cl = [f"#{tid_arr[i]} CAP OK" for i in comply_idx]
                    annotated = ok_box_ann.annotate(annotated, cd)
                    annotated = ok_lbl_ann.annotate(annotated, cd, labels=cl)

                if violations > 0:
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (width, 36), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
                    cv2.putText(annotated,
                                f"  VIOLATION: {violations} worker(s) without head cap",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Motion indicator (small overlay, useful during tuning)
            if motion_score > MOTION_THRESHOLD:
                cv2.putText(annotated, f"MOTION {motion_score:.1f}",
                            (width - 130, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

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

            # ── Write to disk ─────────────────────────────────────────────────
            with session_lock:
                currently_saving = session["saving"]
            with writer_lock:
                current_writer = writer
                current_cw     = csv_writer

            if currently_saving and current_writer is not None:
                try:
                    current_writer.stdin.write(annotated.tobytes())
                    current_writer.stdin.flush()
                except BrokenPipeError:
                    print("[ffmpeg] broken pipe -- stopping save")
                    close_writers()
                    with session_lock:
                        session["saving"] = False
            elif currently_saving and current_writer is None:
                print(f"[debug] saving=True but writer is None at frame={frame_idx}")

            if current_cw is not None:
                row  = [frame_idx, round(elapsed, 2)]
                row += [counts.get(lbl, 0) for lbl in labels]
                row += [len(tracked), round(fps_display, 2)]
                if cap_mode:
                    row += [violations, compliant]
                current_cw.writerow(row)

            if frame_idx % 30 == 0:
                print(f"[loop] frame={frame_idx:05d} fps={fps_display:.1f} "
                      f"motion={motion_score:.1f} n={effective_n} "
                      f"counts={dict(counts)}"
                      + (f" viol={violations} ok={compliant}" if cap_mode else ""))

            # ── Advance state ─────────────────────────────────────────────────
            prev_gray  = curr_gray
            frame_idx += 1

    except Exception as exc:
        with session_lock:
            session["error"] = str(exc)
        print(f"[loop] ERROR: {exc}")
        import traceback; traceback.print_exc()
    finally:
        cap.release()
        close_writers()
        with session_lock:
            session["running"] = False
            session["saving"]  = False
        print("[loop] stopped")


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/session/start")
def start_session(req: StartRequest):
    global session_thread
    with session_lock:
        if session["running"]:
            raise HTTPException(409, "Session already running. Stop it first.")
        if not req.labels:
            raise HTTPException(400, "Provide at least one label.")

        labels = list(req.labels)
        if req.cap_label:
            if req.cap_label not in labels:
                raise HTTPException(400, f"cap_label '{req.cap_label}' must be in labels list.")
            if "person" not in labels:
                labels.insert(0, "person")

        session.update({
            "labels": labels, "rtsp_url": req.rtsp_url,
            "frame": None, "counts": {}, "fps": 0.0, "frame_idx": 0,
            "error": None, "saving": req.save, "save_path": None,
            "violations": 0, "compliant": 0, "cap_label": req.cap_label,
        })

    session_thread = threading.Thread(
        target=detection_loop,
        args=(req.rtsp_url, labels, req.confidence, req.every_n, req.save, req.cap_label),
        daemon=True,
    )
    session_thread.start()
    return {"status": "started", "labels": labels, "rtsp_url": req.rtsp_url,
            "cap_mode": bool(req.cap_label)}


@app.post("/session/stop")
def stop_session():
    with session_lock:
        if not session["running"]:
            return {"status": "not running"}
        session["running"] = False
        session["saving"]  = False

    def _clear_after_delay():
        time.sleep(4)
        with session_lock:
            session["frame"] = None; session["counts"] = {}
            session["fps"] = 0.0; session["frame_idx"] = 0
            session["violations"] = 0; session["compliant"] = 0

    threading.Thread(target=_clear_after_delay, daemon=True).start()
    return {"status": "stopping"}


@app.post("/session/save")
def toggle_save(req: SaveRequest):
    with session_lock:
        if not session["running"]:
            raise HTTPException(400, "No session running.")
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


@app.get("/recordings")
def list_recordings():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if not (fname.endswith(".mp4") or fname.endswith(".mkv")):
                continue
            fpath = os.path.join(OUTPUT_DIR, fname)
            stat  = os.stat(fpath)
            files.append({
                "name":     fname,
                "size_mb":  round(stat.st_size / 1e6, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return JSONResponse(files)


CHUNK_SIZE = 1024 * 256

@app.api_route("/recordings/{filename}", methods=["GET", "HEAD"])
async def stream_recording(filename: str, request: Request):
    filename = os.path.basename(filename)
    fpath    = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")

    file_size    = os.path.getsize(fpath)
    range_header = request.headers.get("range")
    mime_type    = "video/mp4" if filename.endswith(".mp4") else "video/x-matroska"

    if request.method == "HEAD":
        return Response(status_code=200, media_type=mime_type, headers={
            "Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Type": mime_type,
        })

    if range_header:
        range_val  = range_header.strip().lower().replace("bytes=", "")
        parts      = range_val.split("-")
        start      = int(parts[0]) if parts[0] else 0
        end        = int(parts[1]) if (len(parts) > 1 and parts[1]) else min(start + CHUNK_SIZE * 8 - 1, file_size - 1)
        end        = min(end, file_size - 1)
        chunk_size = end - start + 1

        def _iter_range():
            remaining = chunk_size
            with open(fpath, "rb") as f:
                f.seek(start)
                while remaining > 0:
                    data = f.read(min(CHUNK_SIZE, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(_iter_range(), status_code=206, media_type=mime_type, headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes", "Content-Length": str(chunk_size), "Content-Type": mime_type,
        })

    def _iter_full():
        with open(fpath, "rb") as f:
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    return StreamingResponse(_iter_full(), status_code=200, media_type=mime_type, headers={
        "Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Type": mime_type,
    })


@app.get("/video")
def video_feed():
    def _generate():
        while True:
            with session_lock:
                frame = session["frame"]
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for stream...", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.033)

    return StreamingResponse(_generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/resources")
def resources():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    gpu_util = gpu_mem_used = gpu_mem_total = None
    try:
        raw   = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]).decode().strip()
        parts         = raw.split(",")
        gpu_util      = int(parts[0].strip())
        gpu_mem_used  = int(parts[1].strip())
        gpu_mem_total = int(parts[2].strip())
    except Exception:
        pass
    return JSONResponse({
        "cpu_percent": round(cpu, 1), "ram_percent": round(ram.percent, 1),
        "ram_used_gb": round(ram.used / 1e9, 1), "ram_total_gb": round(ram.total / 1e9, 1),
        "gpu_util": gpu_util, "gpu_mem_used": gpu_mem_used, "gpu_mem_total": gpu_mem_total,
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu":    torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.on_event("startup")
def _startup():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    print("[server] SAM3 detection server ready")
