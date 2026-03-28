import csv
import math
import os
import psutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Optional

# Reduce CUDA memory fragmentation — must be set before torch is imported.
# expandable_segments allows PyTorch to grow/shrink segments instead of
# reserving a fixed block, which is the main cause of "reserved but
# unallocated" OOM errors when running multiple large inference passes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
MAX_FRAMES = None   # set to an int to cap frames per session

# Stream is delayed by this many seconds so inference results are ready before
# the frame appears on screen.  Raise if inference is still lagging.
STREAM_DELAY_S = 3.0

# ── Multi-session state ───────────────────────────────────────────────────────
# sessions maps session_id (str UUID) → session dict.
# Each session dict is fully self-contained: own lock, own writer state,
# own thread.  sessions_lock only guards the top-level dict itself.
sessions: dict       = {}
sessions_lock        = threading.Lock()

# ── Shared Real-ESRGAN (loaded once, all sessions share the model weights) ────
# GPU forward passes are serialized by _sr_call_lock so threads don't race on
# the CUDA context.  Initialization is lazy and guarded by _sr_init_lock.
_sr_upsampler        = None
_sr_outscale: int    = 1
_sr_init_lock        = threading.Lock()
_sr_initialized      = False
_sr_call_lock        = threading.Lock()   # one SR call at a time across all streams

app = FastAPI()


# ── Request models ────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    rtsp_url:    str
    labels:      List[str]
    confidence:  float          = 0.30
    every_n:     int            = 5
    save:        bool           = False
    session_id:  Optional[str]  = None
    quality:     str            = "balanced"   # "fast" | "balanced" | "maximum"
    compliance:  bool           = False        # highlight persons missing required items


class SaveRequest(BaseModel):
    save: bool


class EnhanceRequest(BaseModel):
    filename:   str
    labels:     List[str]
    confidence: float = 0.25
    quality:    str   = "maximum"   # "mild" | "strong" | "maximum"


# ── Enhance-job state ─────────────────────────────────────────────────────────
enhance_jobs: dict      = {}   # job_id → job dict
enhance_jobs_lock       = threading.Lock()
_enhance_sem            = threading.Semaphore(1)   # one job at a time (VRAM)


# ── Session factory ───────────────────────────────────────────────────────────
def _make_session(sid: str, rtsp_url: str, labels: list, saving: bool) -> dict:
    return {
        "id":          sid,
        "running":     False,
        "frame":       None,
        "counts":      {},
        "fps":         0.0,
        "frame_idx":   0,
        "error":       None,
        "labels":      labels,
        "rtsp_url":    rtsp_url,
        "started_at":  None,
        "saving":      saving,
        "save_path":   None,
        # Per-session writer state (replaces old module-level globals)
        "_writer":     None,
        "_csv_file":   None,
        "_csv_writer": None,
        "_wlock":        threading.Lock(),
        "_lock":         threading.Lock(),
        "_thread":       None,
        "show_lost":     False,   # ghost tracks hidden by default
        "compliance":    False,   # PPE compliance mode
        "violations":    0,       # count of non-compliant persons in last frame
    }


def _session_public(sess: dict) -> dict:
    """Return only the public-facing keys of a session dict."""
    return {
        "id":          sess["id"],
        "running":     sess["running"],
        "labels":      sess["labels"],
        "rtsp_url":    sess["rtsp_url"],
        "counts":      sess["counts"],
        "fps":         round(sess["fps"], 2),
        "frame_idx":   sess["frame_idx"],
        "started_at":  sess["started_at"],
        "error":       sess["error"],
        "saving":      sess["saving"],
        "save_path":   sess["save_path"],
        "show_lost":   sess.get("show_lost", False),
        "compliance":  sess.get("compliance", False),
        "violations":  sess.get("violations", 0),
    }


def _get_session(sid: str) -> dict:
    with sessions_lock:
        sess = sessions.get(sid)
    if sess is None:
        raise HTTPException(404, f"Session '{sid}' not found.")
    return sess


# ── Writer helpers (per-session) ──────────────────────────────────────────────
#
# Strategy:
#   1. Record into a .mkv scratch file.  MKV is a streamable container — a
#      crash or pipe-break still leaves a playable file.
#   2. On close, remux the .mkv → .mp4 with "-movflags +faststart" so the
#      moov atom lands at the front and browsers can seek immediately.
#
def open_writers(sess: dict, ts: str, labels_list: list,
                 width: int, height: int, fps: float):
    """Start ffmpeg process + CSV file for a session."""
    scratch = os.path.join(OUTPUT_DIR, f"detection_{ts}.mkv")
    cpath   = os.path.join(OUTPUT_DIR, f"counts_{ts}.csv")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-pix_fmt", "bgr24",
        "-s",       f"{width}x{height}",
        "-r",       str(int(fps)),
        "-i",       "pipe:0",
        "-codec:v", "libx264",
        "-preset",  "ultrafast",
        "-crf",     "23",
        "-pix_fmt", "yuv420p",
        scratch,
    ]

    with sess["_wlock"]:
        new_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"[ffmpeg][{sess['id'][:8]}] started PID={new_proc.pid}  scratch={scratch}")

        new_csv = open(cpath, "w", newline="")
        new_cw  = csv.writer(new_csv)
        new_cw.writerow(["frame", "timestamp"] + labels_list + ["total_tracks", "fps"])

        sess["_writer"]     = new_proc
        sess["_csv_file"]   = new_csv
        sess["_csv_writer"] = new_cw

    with sess["_lock"]:
        sess["save_path"] = scratch

    print(f"[writer][{sess['id'][:8]}] scratch → {scratch}")
    print(f"[writer][{sess['id'][:8]}] CSV     → {cpath}")


def _remux_to_mp4(scratch: str, sid_prefix: str) -> str:
    mp4_path = scratch.replace(".mkv", ".mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i",        scratch,
        "-codec",    "copy",
        "-movflags", "+faststart",
        mp4_path,
    ]
    print(f"[remux][{sid_prefix}] {scratch} → {mp4_path}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode == 0:
        os.remove(scratch)
        print(f"[remux][{sid_prefix}] done → {mp4_path}")
        return mp4_path
    else:
        err = result.stderr.decode(errors="replace")
        print(f"[remux][{sid_prefix}] FAILED: {err}")
        return scratch


def close_writers(sess: dict):
    """Flush and close ffmpeg + CSV for a session, then remux to MP4."""
    scratch_path  = None
    sid_prefix    = sess["id"][:8]

    with sess["_wlock"]:
        if sess["_writer"] is not None:
            try:
                sess["_writer"].stdin.close()
                sess["_writer"].wait()
            except Exception as exc:
                print(f"[ffmpeg][{sid_prefix}] close error: {exc}")
            sess["_writer"] = None

        if sess["_csv_file"] is not None:
            try:
                sess["_csv_file"].close()
            except Exception as exc:
                print(f"[csv][{sid_prefix}] close error: {exc}")
            sess["_csv_file"]   = None
            sess["_csv_writer"] = None

    with sess["_lock"]:
        scratch_path    = sess.get("save_path")
        sess["save_path"] = None

    print(f"[writer][{sid_prefix}] closed")

    if scratch_path and scratch_path.endswith(".mkv") and os.path.exists(scratch_path):
        threading.Thread(
            target=_remux_to_mp4, args=(scratch_path, sid_prefix), daemon=True
        ).start()


# ── Background cleanup (delete output files older than 30 min) ───────────────
def cleanup_loop():
    while True:
        try:
            cutoff = time.time() - 30 * 60
            if os.path.exists(OUTPUT_DIR):
                for fname in os.listdir(OUTPUT_DIR):
                    if not (fname.endswith(".mp4") or fname.endswith(".mkv")
                            or fname.endswith(".csv")):
                        continue
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        print(f"[cleanup] deleted {fname}")
        except Exception as exc:
            print(f"[cleanup] error: {exc}")
        time.sleep(300)


# ── Real-ESRGAN super-resolution (shared, loaded once) ───────────────────────
def _load_realesrgan(model_dir: str):
    """
    Load RealESRGAN_x4plus.  Called at most once; result cached in module-level
    globals.  Returns (upsampler, outscale) or (None, 1) if unavailable.
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("[esrgan] package not found — SR disabled. "
              "Install with: pip install realesrgan basicsr")
        return None, 1

    weight_name = "RealESRGAN_x4plus.pth"
    weight_path = os.path.join(model_dir, weight_name)

    if not os.path.exists(weight_path):
        url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               f"v0.1.0/{weight_name}")
        print(f"[esrgan] downloading {weight_name} (~67 MB)…")
        try:
            import urllib.request
            os.makedirs(model_dir, exist_ok=True)
            urllib.request.urlretrieve(url, weight_path)
            print(f"[esrgan] saved → {weight_path}")
        except Exception as exc:
            print(f"[esrgan] download failed: {exc} — SR disabled")
            return None, 1

    try:
        arch = RRDBNet(num_in_ch=3, num_out_ch=3,
                       num_feat=64, num_block=23,
                       num_grow_ch=32, scale=4)
        up   = RealESRGANer(
            scale=4,
            model_path=weight_path,
            model=arch,
            tile=256,       # reduced from 512 — leaves more VRAM for SAM3 passes
            tile_pad=16,
            pre_pad=0,
            half=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("[esrgan] RealESRGAN_x4plus ready (outscale=2, tile=256, fp16)")
        return up, 2
    except Exception as exc:
        print(f"[esrgan] init failed: {exc} — SR disabled")
        return None, 1


def _get_realesrgan():
    """Lazy-init Real-ESRGAN once for the whole process."""
    global _sr_upsampler, _sr_outscale, _sr_initialized
    with _sr_init_lock:
        if not _sr_initialized:
            _sr_upsampler, _sr_outscale = _load_realesrgan(
                os.path.dirname(MODEL_PATH))
            _sr_initialized = True
    return _sr_upsampler, _sr_outscale


def _apply_sr(frame: np.ndarray, upsampler, outscale: int = 2) -> np.ndarray:
    try:
        with _sr_call_lock:   # serialize GPU SR calls across all sessions
            out_rgb, _ = upsampler.enhance(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), outscale=outscale)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f"[esrgan] enhance error: {exc}")
        return frame


def _scale_boxes(xyxy: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = xyxy.copy()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


# ── Lighting analysis ─────────────────────────────────────────────────────────
# Four lighting tiers drive the entire adaptive pipeline.
# Thresholds are on mean L* (perceptual luminance, 0-255):
#   normal   ≥ 100   — daylight / well-lit factory
#   low      ≥  55   — overcast, dusk, dim artificial light
#   very_low ≥  25   — very dark room, single lamp, heavy shadow
#   night    <  25   — near-total darkness, IR cameras, night-shift

def _brightness_level(frame: np.ndarray) -> str:
    """
    Classify scene lighting robustly against HDR scenes.
    A dark room with a bright doorway has a misleadingly high MEAN but the
    relevant objects (workers) are in the dark area.
    Strategy: use the 20th percentile of luminance (the darker half of the
    scene) as the primary signal, not the mean.  This correctly classifies
    "dark room + bright exit" as night/very_low rather than normal.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # p20 = brightness of the darkest 20% of pixels (where workers typically are)
    p20  = float(np.percentile(gray, 20))
    mean = float(gray.mean())
    # If there's a large gap between mean and p20, the scene is HDR-backlit:
    # use the darker metric to drive enhancement.
    score = p20 * 0.7 + mean * 0.3   # weight darker regions more heavily
    if score >= 90: return "normal"
    if score >= 50: return "low"
    if score >= 22: return "very_low"
    return "night"


# ── Adaptive gamma correction ─────────────────────────────────────────────────
# Computes the gamma exponent needed to move the mean brightness toward `target`.
# Applied before Retinex so the network sees a reasonably exposed image.

def _auto_gamma(img: np.ndarray, target: float = 115.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = max(float(gray.mean()), 1.0)
    if mean >= target:
        return img
    gamma     = math.log(target / 255.0) / math.log(mean / 255.0)
    gamma     = float(np.clip(gamma, 0.25, 5.0))
    inv_gamma = 1.0 / gamma
    lut       = np.array([((i / 255.0) ** inv_gamma) * 255
                           for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


# ── Multi-Scale Retinex with Color Restoration (MSRCR) ────────────────────────
# Decomposes the scene into reflectance (what objects actually look like) and
# illumination (the lighting itself), then keeps only reflectance.
# This is the standard technique used in surveillance low-light research.
# Sigmas [15, 80, 200] cover detail, mid-range, and large-scale illumination.

def _msrcr(img: np.ndarray) -> np.ndarray:
    sigmas  = [15.0, 80.0, 200.0]
    img_f   = img.astype(np.float32) + 1.0          # avoid log(0)
    log_img = np.log(img_f)
    msr     = np.zeros_like(img_f)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
        msr += log_img - np.log(blur + 1.0)
    msr /= len(sigmas)

    # Color Restoration: re-introduce local colour balance
    img_sum    = img_f.sum(axis=2, keepdims=True) + 1e-6
    color_coef = np.log(125.0 * img_f / img_sum)
    result     = color_coef * msr

    # Normalize per-channel to 0-255
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        ch   = result[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            ch = (ch - lo) / (hi - lo) * 255.0
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


# ── Edge-preserving denoising ─────────────────────────────────────────────────
# Bilateral filter: smooths sensor noise while keeping object edges sharp.
# Critical for night frames where Real-ESRGAN would otherwise amplify grain.

def _denoise(img: np.ndarray, strength: int = 7) -> np.ndarray:
    sigma = strength * 6
    return cv2.bilateralFilter(img, d=7, sigmaColor=sigma, sigmaSpace=sigma)


# ── Gray-world white balance ──────────────────────────────────────────────────
# Corrects colour casts from sodium/LED/fluorescent factory lighting by
# scaling each channel so the mean of all channels becomes equal.

def _white_balance(img: np.ndarray) -> np.ndarray:
    f      = img.astype(np.float32)
    means  = f.mean(axis=(0, 1))           # per-channel mean
    global_mean = means.mean()
    scale  = np.where(means > 1.0, global_mean / means, 1.0)
    f     *= scale[np.newaxis, np.newaxis, :]
    return np.clip(f, 0, 255).astype(np.uint8)


# ── Temporal blending ─────────────────────────────────────────────────────────
# Averages the last N frames with exponential decay weighting.
# Reduces temporal shot noise without adding spatial blur.
# Most effective when the scene is mostly static (factory floor).

class _TemporalBlender:
    def __init__(self, maxlen: int = 4, decay: float = 0.55):
        self._buf   = deque(maxlen=maxlen)
        self._decay = decay

    def update(self, frame: np.ndarray) -> np.ndarray:
        self._buf.append(frame.astype(np.float32))
        n       = len(self._buf)
        weights = [self._decay ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        blended = sum(w * f for w, f in zip(weights, self._buf)) / total_w
        return np.clip(blended, 0, 255).astype(np.uint8)


# ── Reinhard tone mapping ─────────────────────────────────────────────────────
# Critical for backlit / HDR scenes (e.g. dark room + bright doorway).
# Standard gamma lifts shadows but simultaneously saturates highlights.
# Reinhard compresses the log-luminance globally: bright areas are pulled
# down, dark areas are pulled up — both in the same pass.
# light_adapt=0.8 → strong local adaptation (each region normalised separately)
# This is the correct operator for CCTV footage where the camera AGC hasn't
# balanced the exposure between dark interior and bright exits/windows.

_TONEMAP = cv2.createTonemapReinhard(
    gamma=1.0, intensity=0.0, light_adapt=0.8, color_adapt=0.0)

def _tone_map(img: np.ndarray) -> np.ndarray:
    """Reinhard tone mapping: handles bright-door + dark-room HDR."""
    f   = img.astype(np.float32) / 255.0
    out = _TONEMAP.process(f)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ── Adaptive CLAHE ────────────────────────────────────────────────────────────
_CLAHE_NORMAL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_LOW    = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
_CLAHE_NIGHT  = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))


# ── Master enhancement pipeline ───────────────────────────────────────────────
# Called on every tile and on the full frame before SAM3.
# Unsharp masking is intentionally omitted — ringing artifacts break SAM3's
# text-visual similarity scoring for surgical caps and gloves.
#
# Pipeline per level:
#   normal   → CLAHE only (unchanged from original — no regression)
#   low      → tone_map → gamma → stronger CLAHE
#   very_low → denoise → tone_map → white_balance → gamma → aggressive CLAHE
#   night    → denoise → MSRCR → tone_map → white_balance → gamma → aggressive CLAHE
#
# Tone mapping is added to ALL non-normal levels because factory CCTV footage
# almost always has HDR situations (windows, lamps, exit signs) that pure
# gamma cannot handle.  MSRCR is reserved for near-total darkness only.

def _enhance(img: np.ndarray, level: str = "normal") -> np.ndarray:
    if img.shape[0] < 8 or img.shape[1] < 8:
        return img

    if level == "normal":
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = _CLAHE_NORMAL.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    elif level == "low":
        img     = _tone_map(img)            # compress highlights, lift shadows
        img     = _auto_gamma(img, target=108.0)
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = _CLAHE_LOW.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    elif level == "very_low":
        img     = _denoise(img, strength=6)
        img     = _tone_map(img)
        img     = _white_balance(img)
        img     = _auto_gamma(img, target=115.0)
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = _CLAHE_NIGHT.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    else:  # night
        img     = _denoise(img, strength=9)
        img     = _msrcr(img)               # Retinex: strip illumination, keep reflectance
        img     = _tone_map(img)            # flatten remaining HDR
        img     = _white_balance(img)
        img     = _auto_gamma(img, target=120.0)
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = _CLAHE_NIGHT.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Per-frame dynamic confidence for dark conditions ──────────────────────────
# SAM3's text-visual similarity scores are naturally depressed when the input
# is dark or noisy, so the same hard threshold misses real detections.
# We lower the predictor's threshold before running on dark frames (each session
# owns its predictor, so this is thread-safe) and restore it afterwards.
# NOTE: we do NOT use a globally permissive base_conf at init time because that
# causes SAM3's internal NMS to see far more candidate boxes, allowing low-conf
# false positives to suppress real detections via IoU overlap.

# Scale factors per lighting level: user conf × factor = actual threshold used
_CONF_SCALE = {"normal": 1.00, "low": 0.70, "very_low": 0.55, "night": 0.40}


def _set_predictor_conf(predictor, conf: float):
    """Best-effort in-place update of the SAM3 predictor's confidence threshold."""
    try:
        if hasattr(predictor, "args"):
            predictor.args.conf = conf
        if hasattr(predictor, "overrides"):
            predictor.overrides["conf"] = conf
        if hasattr(predictor, "model") and hasattr(predictor.model, "predictor"):
            if hasattr(predictor.model.predictor, "args"):
                predictor.model.predictor.args.conf = conf
    except Exception:
        pass  # best-effort; original conf remains if anything fails


# ── SAHI: tiled inference ─────────────────────────────────────────────────────
def _run_sahi(frame: np.ndarray, predictor, labels: list,
              n_cols: int, n_rows: int,
              overlap: float = 0.20,
              level: str = "normal") -> sv.Detections:
    if not labels:
        return sv.Detections.empty()
    fh, fw   = frame.shape[:2]
    tile_w   = int(fw / (n_cols - overlap * (n_cols - 1)))
    tile_h   = int(fh / (n_rows - overlap * (n_rows - 1)))
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
            # Skip degenerate tiles that would crash CLAHE (needs >= tileGridSize=8)
            if tile.size == 0 or tile.shape[0] < 32 or tile.shape[1] < 32:
                continue

            tile = _enhance(tile, level)
            pil  = PILImage.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            try:
                predictor.set_image(pil)
                results = predictor(text=labels)
            except Exception as exc:
                print(f"[sahi] tile({row},{col}) error: {exc}")
                continue

            if results and results[0].boxes is not None:
                r     = results[0]
                boxes = r.boxes.xyxy.cpu().numpy().copy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                boxes[:, [0, 2]]  = np.clip(boxes[:, [0, 2]], 0, fw)
                boxes[:, [1, 3]]  = np.clip(boxes[:, [1, 3]], 0, fh)
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


def _infer_full_frame(frame: np.ndarray, predictor,
                      labels: list, level: str = "normal") -> sv.Detections:
    """Single SAM3 inference on the entire frame — catches objects at tile boundaries."""
    if not labels:
        return sv.Detections.empty()
    pil = PILImage.fromarray(
        cv2.cvtColor(_enhance(frame, level), cv2.COLOR_BGR2RGB))
    try:
        predictor.set_image(pil)
        results = predictor(text=labels)
    except torch.cuda.OutOfMemoryError:
        # OOM on full-frame is non-fatal — SAHI tiles already covered the frame.
        # Free cache so subsequent passes are not also affected.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[full] OOM — skipped (SAHI tiles still used)")
        return sv.Detections.empty()
    except Exception as exc:
        print(f"[full] error: {exc}")
        return sv.Detections.empty()
    if results and results[0].boxes is not None:
        r = results[0]
        return sv.Detections(
            xyxy=r.boxes.xyxy.cpu().numpy().copy(),
            confidence=r.boxes.conf.cpu().numpy(),
            class_id=r.boxes.cls.cpu().numpy().astype(int),
        )
    return sv.Detections.empty()


def _merge(*dets_list) -> sv.Detections:
    """Merge any number of sv.Detections and deduplicate with NMS."""
    boxes, confs, cls = [], [], []
    for d in dets_list:
        if d is not None and len(d) > 0:
            boxes.append(d.xyxy)
            confs.append(d.confidence)
            cls.append(d.class_id)
    if not boxes:
        return sv.Detections.empty()
    merged = sv.Detections(
        xyxy=np.vstack(boxes),
        confidence=np.hstack(confs),
        class_id=np.hstack(cls),
    )
    return merged.with_nms(threshold=0.45)


def _compliance_check(dets: sv.Detections, all_labels: list,
                      required_labels: list,
                      frame_shape: tuple = (1080, 1920, 3)) -> list:
    """
    Returns a list of violation dicts for persons not wearing every required item.
    Each violation: {"xyxy": [x1,y1,x2,y2], "missing": ["cap","gloves"]}

    Person boxes are filtered strictly before association to eliminate false positives:
      - Must be portrait-shaped (height >= width * 0.8)
      - Must cover at least 2% of frame area
      - Must have confidence >= 0.38
      - Duplicate/overlapping person boxes collapsed with tight NMS (0.30)
    """
    if len(dets) == 0:
        return []

    person_idx = [i for i, l in enumerate(all_labels) if l == "person"]
    if not person_idx:
        return []
    person_cls = set(person_idx)

    if not any(lbl in all_labels for lbl in required_labels):
        return []

    fh, fw = frame_shape[:2]
    frame_area = max(fh * fw, 1)

    class_ids   = dets.class_id   if dets.class_id   is not None else []
    confidences = dets.confidence if dets.confidence is not None else [1.0] * len(dets)
    xyxys       = dets.xyxy

    # ── Collect & filter person boxes ────────────────────────────────────────
    raw_persons = []
    for i, c in enumerate(class_ids):
        if c not in person_cls:
            continue
        pb = xyxys[i]
        x1, y1, x2, y2 = pb
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

        # 1. Must be portrait or at least squarish (not landscape)
        if bh < bw * 0.8:
            continue
        # 2. Must cover ≥2% of frame (eliminates distant/tiny false detections)
        if (bw * bh) / frame_area < 0.02:
            continue
        # 3. Confidence floor — persons need 0.38 minimum
        if float(confidences[i]) < 0.38:
            continue

        raw_persons.append((pb, float(confidences[i])))

    if not raw_persons:
        return []

    # ── Collapse duplicate person boxes (same person from multiple SAHI tiles) ─
    # Tight NMS at 0.30 IoU ensures we keep one box per real person
    p_xyxy = np.array([p[0] for p in raw_persons])
    p_conf = np.array([p[1] for p in raw_persons])
    keep = sv.box_non_max_suppression(p_xyxy, p_conf, 0.30)
    person_boxes = [raw_persons[k][0] for k in keep]

    # ── Target boxes ──────────────────────────────────────────────────────────
    target_boxes = [(xyxys[i], all_labels[c] if c < len(all_labels) else "?")
                    for i, c in enumerate(class_ids)
                    if c not in person_cls]

    # ── Association ───────────────────────────────────────────────────────────
    violations = []
    for pb in person_boxes:
        px1, py1, px2, py2 = pb
        pw, ph = px2 - px1, py2 - py1
        # Expand person box 20% outward — cap extends above head, gloves beside hip
        mx1 = px1 - pw * 0.20
        my1 = py1 - ph * 0.20
        mx2 = px2 + pw * 0.20
        my2 = py2 + ph * 0.20

        found_labels = set()
        for tb, tlbl in target_boxes:
            cx = (tb[0] + tb[2]) / 2
            cy = (tb[1] + tb[3]) / 2
            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                found_labels.add(tlbl)

        missing = [lbl for lbl in required_labels if lbl not in found_labels]
        if missing:
            violations.append({
                "xyxy":    [int(px1), int(py1), int(px2), int(py2)],
                "missing": missing,
            })

    return violations


def _draw_compliance(img: np.ndarray, violations: list) -> np.ndarray:
    """Draw red violation boxes with MISSING label on the frame."""
    for v in violations:
        x1, y1, x2, y2 = v["xyxy"]
        miss_txt = "MISSING: " + ", ".join(v["missing"])
        # Red filled border
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 3)
        # Corner ticks for emphasis
        dash = 16
        for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (sx, sy), (sx+dx*dash, sy), (0,0,255), 3)
            cv2.line(img, (sx, sy), (sx, sy+dy*dash), (0,0,255), 3)
        # Label background
        (tw, th), _ = cv2.getTextSize(miss_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 0, 180), -1)
        cv2.putText(img, miss_txt, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _filter_false_positives(dets: sv.Detections, frame: np.ndarray,
                             labels: list) -> sv.Detections:
    """
    Remove detections that are almost certainly background/black-space false positives.

    Rules applied per detection box:
      1. Too large  — box covers >40% of frame area → likely background blob.
      2. Too small  — box is <0.03% of frame area   → noise speck.
      3. Zero texture — std-dev of grayscale crop <6 → pure flat/black region.
         (Gloves/caps have edges; an unlit black wall does not.)
      4. Aspect ratio guard — width or height >8× the other → degenerate sliver.
    """
    if len(dets) == 0:
        return dets

    fh, fw = frame.shape[:2]
    frame_area = fh * fw
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keep = []
    for i, box in enumerate(dets.xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        box_area = bw * bh

        # Rule 1: too large
        if box_area / frame_area > 0.40:
            continue

        # Rule 2: too small
        if box_area / frame_area < 0.0003:
            continue

        # Rule 3: aspect ratio
        if bw / bh > 8 or bh / bw > 8:
            continue

        # Rule 4: texture check
        crop = gray[max(y1,0):min(y2,fh), max(x1,0):min(x2,fw)]
        if crop.size > 0 and float(crop.std()) < 6.0:
            continue

        keep.append(i)

    if not keep:
        return sv.Detections.empty()
    return dets[keep]


# ── Detection loop (per-session) ──────────────────────────────────────────────
def detection_loop(sess: dict, confidence: float, every_n: int,
                   quality: str = "balanced"):
    sid = sess["id"]
    labels    = list(sess["labels"])            # user-specified targets
    compliance_mode = sess.get("compliance", False)
    # In compliance mode we always detect "person" so we can cross-reference
    if compliance_mode and "person" not in labels:
        labels = labels + ["person"]
    # required_labels = the original user labels (not "person")
    required_labels = [l for l in labels if l != "person"]
    rtsp_url = sess["rtsp_url"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load SAM3 (one instance per session) ──────────────────────────────────
    print(f"[sam3][{sid[:8]}] loading model...")
    # imgsz=896 = 14×64 — exact multiple of SAM3's stride (avoids rounding
    # warning) and uses ~50% less VRAM than 1280 on full-frame inference while
    # still resolving objects in each SAHI tile at adequate resolution.
    overrides = dict(
        conf=confidence, task="segment", mode="predict",
        model=MODEL_PATH, half=True, imgsz=896, verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"[sam3][{sid[:8]}] model loaded  (conf={confidence:.2f})")

    # ── Quality preset — each level is meaningfully different ────────────────
    # fast:     2×2 SAHI, no SR, no TTA,  overlap=0.10  → lowest GPU, fastest
    # balanced: 3×3 SAHI, SR,   TTA dark, overlap=0.20  → default
    # maximum:  4×4 SAHI, SR,   TTA always, overlap=0.30,
    #           confidence*0.85 to surface weak detections → best recall
    _q_use_sr      = quality != "fast"
    _q_force_tta   = quality == "maximum"
    _q_force_blend = quality == "maximum"
    _Q_GRID  = {"fast": (2, 2), "balanced": (3, 3), "maximum": (4, 4)}
    _Q_OVL   = {"fast": 0.10,   "balanced": 0.20,   "maximum": 0.30}
    _Q_CSCALE= {"fast": 1.00,   "balanced": 1.00,   "maximum": 0.85}
    _q_base_grid  = _Q_GRID.get(quality, (3, 3))
    _q_overlap    = _Q_OVL.get(quality, 0.20)
    _q_conf_scale = _Q_CSCALE.get(quality, 1.00)

    # Shared Real-ESRGAN (lazy init, loaded once for all sessions)
    upsampler, sr_outscale = _get_realesrgan()

    sr_status = "disabled (not installed)" if upsampler is None else ("enabled" if _q_use_sr else "skipped (fast mode)")
    print(f"[quality][{sid[:8]}] preset={quality}  grid={_q_base_grid}  "
          f"overlap={_q_overlap}  SR={sr_status}  "
          f"TTA={'always' if _q_force_tta else 'dark-only'}  "
          f"conf_scale={_q_conf_scale}")

    # ── Tracker + annotators ──────────────────────────────────────────────────
    # lost_track_buffer=300 → Kalman-predicts position for ~20s before ByteTrack gives up
    tracker   = ByteTrack(track_activation_threshold=0.20, lost_track_buffer=300,
                          minimum_matching_threshold=0.8, frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40)

    # Ghost-track cache: survives ByteTrack drops, shown with an amber box
    # {tid: {"xyxy": [...], "class_id": int, "label": str, "conf": float, "ttl": int}}
    _GHOST_TTL    = 900   # frames to keep ghost alive (~60s at 15fps)
    ghost_tracks: dict = {}
    # track_memory: last known state for every active track_id
    track_memory: dict = {}

    # ── Open RTSP ─────────────────────────────────────────────────────────────
    print(f"[rtsp][{sid[:8]}] connecting -> {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        with sess["_lock"]:
            sess["error"]   = "Could not open RTSP stream"
            sess["running"] = False
        print(f"[rtsp][{sid[:8]}] ERROR: could not open stream")
        return

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        with sess["_lock"]:
            sess["error"]   = "Stream reported invalid resolution (0×0)"
            sess["running"] = False
        print(f"[rtsp][{sid[:8]}] ERROR: invalid resolution")
        cap.release()
        return
    print(f"[rtsp][{sid[:8]}] {width}x{height} @ {stream_fps:.1f} fps")

    print(f"[sahi][{sid[:8]}] base grid={_q_base_grid} (dark frames always upgrade to 4×4)")

    # ── Open writers if save was requested at start ────────────────────────────
    with sess["_lock"]:
        save_on_start = sess["saving"]
    if save_on_start:
        open_writers(sess, datetime.now().strftime("%Y%m%d_%H%M%S"),
                     labels, width, height, stream_fps)

    # ── Delay buffer + inference thread ───────────────────────────────────────
    _buf_maxlen = max(200, int(stream_fps * (STREAM_DELAY_S + 10)))
    display_buf    = deque(maxlen=_buf_maxlen)
    det_store      = {}
    det_store_lock = threading.Lock()

    infer = {
        "running":   True,
        "frame_idx": 0,
        "frame":     None,
        "lock":      threading.Lock(),
        "event":     threading.Event(),
    }

    # Temporal blender: averages the last 4 frames before SR to suppress
    # sensor noise.  On dark/night frames this dramatically improves SNR.
    _blender = _TemporalBlender(maxlen=4, decay=0.55)

    def _inference_worker():
        while infer["running"]:
            infer["event"].wait(timeout=1.0)
            infer["event"].clear()
            if not infer["running"]:
                break
            with infer["lock"]:
                fid = infer["frame_idx"]
                f   = infer["frame"]
            if f is None:
                continue
            t0 = time.time()
            try:
                # ── 1. Lighting analysis ───────────────────────────────────────
                level = _brightness_level(f)

                # ── 2. Temporal blending for noise suppression ─────────────────
                if _q_force_blend or level in ("very_low", "night"):
                    f_proc = _blender.update(f)
                else:
                    f_proc = f

                # ── 3. Adaptive SAHI grid ──────────────────────────────────────
                # Quality sets the base; dark scenes always upgrade to at least 4×4
                if level in ("very_low", "night"):
                    n_cols, n_rows = 4, 4
                else:
                    n_cols, n_rows = _q_base_grid

                # ── 4. Dynamic confidence ──────────────────────────────────────
                # Dark scenes lower conf to surface weak detections.
                # Maximum quality also lowers it to improve recall in good light.
                eff_conf = confidence * _CONF_SCALE.get(level, 1.0) * _q_conf_scale
                if level != "normal":
                    _set_predictor_conf(predictor, eff_conf)

                # ── 5. Super-resolution ────────────────────────────────────────
                if _q_use_sr and upsampler is not None:
                    f_sr = _apply_sr(f_proc, upsampler, outscale=sr_outscale)
                    sx   = f_proc.shape[1] / f_sr.shape[1]
                    sy   = f_proc.shape[0] / f_sr.shape[0]
                else:
                    f_sr = f_proc
                    sx   = sy = 1.0

                # ── 6. SAHI on SR frame ────────────────────────────────────────
                sahi_dets = _run_sahi(f_sr, predictor, labels,
                                      n_cols, n_rows,
                                      overlap=_q_overlap, level=level)
                if len(sahi_dets) > 0 and upsampler is not None:
                    sahi_dets = sv.Detections(
                        xyxy=_scale_boxes(sahi_dets.xyxy, sx, sy),
                        confidence=sahi_dets.confidence,
                        class_id=sahi_dets.class_id,
                    )

                # Free VRAM fragments before full-frame inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ── 7. Full-frame pass on original ────────────────────────────
                full_dets = _infer_full_frame(f_proc, predictor, labels, level)

                # ── 8. TTA: horizontal flip (skipped on OOM) ──────────────────
                if _q_force_tta or level in ("low", "very_low", "night"):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        fw_  = f_proc.shape[1]
                        f_fl = cv2.flip(f_proc, 1)
                        fl_dets = _infer_full_frame(f_fl, predictor, labels, level)
                        if len(fl_dets) > 0:
                            xyxy_fl = fl_dets.xyxy.copy()
                            xyxy_fl[:, [0, 2]] = fw_ - xyxy_fl[:, [2, 0]]
                            fl_dets = sv.Detections(
                                xyxy=xyxy_fl,
                                confidence=fl_dets.confidence,
                                class_id=fl_dets.class_id,
                            )
                    except torch.cuda.OutOfMemoryError:
                        print(f"[infer][{sid[:8]}] TTA skipped — OOM")
                        fl_dets = sv.Detections.empty()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    fl_dets = sv.Detections.empty()

                # ── 9. Restore original confidence + merge ─────────────────────
                if level != "normal":
                    _set_predictor_conf(predictor, confidence)

                dets = _merge(sahi_dets, full_dets, fl_dets)
                dets = _filter_false_positives(dets, f_proc, labels)

                print(f"[infer][{sid[:8]}] fid={fid} level={level} "
                      f"conf={eff_conf:.2f} grid={n_cols}×{n_rows} "
                      f"sahi={len(sahi_dets)} full={len(full_dets)} "
                      f"flip={len(fl_dets)} final={len(dets)} "
                      f"{(time.time()-t0)*1000:.0f} ms")
            except Exception as exc:
                print(f"[infer][{sid[:8]}] error: {exc}")
                import traceback; traceback.print_exc()
                dets = sv.Detections.empty()

            with det_store_lock:
                det_store[fid] = dets
                cutoff = fid - int(stream_fps * (STREAM_DELAY_S + 5))
                for k in [k for k in det_store if k < cutoff]:
                    del det_store[k]

    infer_thread = threading.Thread(target=_inference_worker, daemon=True)
    infer_thread.start()

    # ── Loop state ────────────────────────────────────────────────────────────
    frame_idx       = 0
    reconnect_count = 0
    MAX_RECONNECTS  = 10
    t_start         = time.time()
    t_frame         = t_start
    fps_display     = 0.0
    last_sv_dets    = sv.Detections.empty()

    # Buffering placeholder — resolution-safe text placement
    buf_frame  = np.zeros((height, width, 3), dtype=np.uint8)
    _buf_text  = "Buffering..."
    _buf_scale = max(0.4, min(1.0, width / 640))
    _buf_thick = max(1, int(_buf_scale * 2))
    (_tw, _th), _ = cv2.getTextSize(_buf_text, cv2.FONT_HERSHEY_SIMPLEX,
                                    _buf_scale, _buf_thick)
    _bx = max(5, (width  - _tw) // 2)
    _by = max(20, height // 2)
    cv2.putText(buf_frame, _buf_text, (_bx, _by),
                cv2.FONT_HERSHEY_SIMPLEX, _buf_scale, (200, 200, 200), _buf_thick)

    with sess["_lock"]:
        sess["running"]    = True
        sess["started_at"] = datetime.now().isoformat()
        sess["error"]      = None
        sess["frame"]      = buf_frame.copy()

    try:
        while True:
            # ── Stop / save-toggle signals ─────────────────────────────────────
            with sess["_lock"]:
                should_run  = sess["running"]
                should_save = sess["saving"]

            if not should_run:
                print(f"[loop][{sid[:8]}] stop signal received")
                break

            with sess["_wlock"]:
                writer_active = sess["_writer"] is not None

            if should_save and not writer_active:
                open_writers(sess, datetime.now().strftime("%Y%m%d_%H%M%S"),
                             labels, width, height, stream_fps)
            elif not should_save and writer_active:
                close_writers(sess)

            # ── Read frame ─────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                with sess["_lock"]:
                    if not sess["running"]:
                        break
                reconnect_count += 1
                if reconnect_count > MAX_RECONNECTS:
                    print(f"[rtsp][{sid[:8]}] {MAX_RECONNECTS} reconnect attempts exhausted")
                    with sess["_lock"]:
                        sess["error"] = "RTSP stream lost after max reconnect attempts"
                    break
                backoff = min(2 ** (reconnect_count - 1), 30)
                print(f"[rtsp][{sid[:8]}] stream lost — reconnecting "
                      f"({reconnect_count}/{MAX_RECONNECTS}) in {backoff}s…")
                cap.release()
                display_buf.clear()
                with det_store_lock:
                    det_store.clear()
                last_sv_dets = sv.Detections.empty()
                ghost_tracks.clear()
                track_memory.clear()
                time.sleep(backoff)
                with sess["_lock"]:
                    if not sess["running"]:
                        break
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print(f"[rtsp][{sid[:8]}] reconnect failed")
                    continue
                reconnect_count = 0
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # ── Buffer current frame ───────────────────────────────────────────
            display_buf.append((time.time(), frame_idx, frame.copy()))

            # ── Trigger inference ──────────────────────────────────────────────
            if frame_idx % every_n == 0:
                with infer["lock"]:
                    infer["frame_idx"] = frame_idx
                    infer["frame"]     = frame.copy()
                infer["event"].set()

            # ── Pull delayed display frame ─────────────────────────────────────
            now = time.time()
            display_frame = None
            display_fid   = None
            while display_buf and (now - display_buf[0][0]) >= STREAM_DELAY_S:
                _, fid_d, f_d = display_buf.popleft()
                display_frame = f_d
                display_fid   = fid_d

            if display_frame is None:
                frame_idx += 1
                time.sleep(1.0 / stream_fps)
                continue

            # ── Match best inference result ────────────────────────────────────
            with det_store_lock:
                candidates = [k for k in det_store if k <= display_fid]
                if candidates:
                    last_sv_dets = det_store[max(candidates)]

            # ── ByteTrack ──────────────────────────────────────────────────────
            tracked     = tracker.update_with_detections(last_sv_dets)
            counts      = defaultdict(int)
            label_texts = []

            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            class_ids   = tracked.class_id   if tracked.class_id   is not None else []
            confidences = tracked.confidence  if tracked.confidence  is not None else []
            xyxys       = tracked.xyxy        if tracked.xyxy        is not None else []

            active_tids = set()
            for tid, cls_id, conf, xyxy in zip(tracker_ids, class_ids, confidences, xyxys):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")
                active_tids.add(tid)
                # Update track memory
                track_memory[tid] = {
                    "xyxy": xyxy.tolist(), "class_id": cls_id,
                    "label": name, "conf": float(conf),
                }
                # If it re-appeared, remove from ghost
                ghost_tracks.pop(tid, None)

            # Promote newly-lost tracks to ghost cache
            for tid, mem in list(track_memory.items()):
                if tid not in active_tids and tid not in ghost_tracks:
                    ghost_tracks[tid] = {**mem, "ttl": _GHOST_TTL}

            # Tick ghost TTLs and count them in totals
            for tid in list(ghost_tracks.keys()):
                g = ghost_tracks[tid]
                g["ttl"] -= 1
                if g["ttl"] <= 0:
                    ghost_tracks.pop(tid)
                    track_memory.pop(tid, None)
                else:
                    counts[g["label"]] += 1   # keep label count alive

            # ── Annotate ───────────────────────────────────────────────────────
            annotated = display_frame.copy()
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            # ── Compliance check ───────────────────────────────────────────────
            _violations = []
            if compliance_mode and len(tracked) > 0:
                _violations = _compliance_check(tracked, labels, required_labels,
                                                frame_shape=display_frame.shape)
                if _violations:
                    annotated = _draw_compliance(annotated, _violations)
            with sess["_lock"]:
                sess["violations"] = len(_violations)

            # Draw ghost tracks — amber dashed-style box (only if enabled)
            _show_lost = sess.get("show_lost", False)
            for tid, g in (ghost_tracks.items() if _show_lost else []):
                x1, y1, x2, y2 = [int(v) for v in g["xyxy"]]
                # Fade opacity based on remaining TTL
                alpha = min(1.0, g["ttl"] / (_GHOST_TTL * 0.3))
                color = (0, int(165 * alpha), int(255 * alpha))  # amber fades out
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                # Dashed effect: draw corner ticks
                dash = 12
                for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(annotated, (sx, sy), (sx + dx*dash, sy), color, 2)
                    cv2.line(annotated, (sx, sy), (sx, sy + dy*dash), color, 2)
                txt = f"#{tid} {g['label']} [LOST]"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                ty = max(y1 - 4, th + 4)
                cv2.rectangle(annotated, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
                cv2.putText(annotated, txt, (x1 + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # ── Timing ─────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            elapsed     = now - t_start

            # ── Push to session ────────────────────────────────────────────────
            with sess["_lock"]:
                sess["frame"]     = annotated.copy()
                sess["counts"]    = dict(counts)
                sess["fps"]       = fps_display
                sess["frame_idx"] = frame_idx

            # ── Write to disk ──────────────────────────────────────────────────
            with sess["_lock"]:
                currently_saving = sess["saving"]
            with sess["_wlock"]:
                current_writer = sess["_writer"]
                current_cw     = sess["_csv_writer"]

            if currently_saving and current_writer is not None:
                try:
                    current_writer.stdin.write(annotated.tobytes())
                    current_writer.stdin.flush()
                except BrokenPipeError:
                    print(f"[ffmpeg][{sid[:8]}] broken pipe — stopping save")
                    close_writers(sess)
                    with sess["_lock"]:
                        sess["saving"] = False

            if current_cw is not None:
                row  = [display_fid, round(elapsed, 2)]
                row += [counts.get(lbl, 0) for lbl in labels]
                row += [len(tracked), round(fps_display, 2)]
                current_cw.writerow(row)

            if frame_idx % 30 == 0:
                print(f"[loop][{sid[:8]}] frame={frame_idx:05d} "
                      f"fps={fps_display:.1f} counts={dict(counts)}")

            frame_idx += 1

    except Exception as exc:
        with sess["_lock"]:
            sess["error"] = str(exc)
        print(f"[loop][{sid[:8]}] ERROR: {exc}")
    finally:
        infer["running"] = False
        infer["event"].set()
        infer_thread.join(timeout=5)
        cap.release()
        close_writers(sess)
        with sess["_lock"]:
            sess["running"] = False
            sess["saving"]  = False
        print(f"[loop][{sid[:8]}] stopped")


# ── Enhance job runner ────────────────────────────────────────────────────────

def _run_enhance_job(job_id: str, filename: str, labels: list,
                     confidence: float, quality: str):
    """Offline enhancement + detection on a saved recording."""
    job = enhance_jobs[job_id]

    def _upd(**kw):
        with enhance_jobs_lock:
            enhance_jobs[job_id].update(kw)

    in_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(in_path):
        _upd(status="failed", error="Input file not found")
        return   # finally block releases semaphore

    out_name = f"enhanced_{job_id[:8]}_{os.path.basename(filename)}"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            _upd(status="failed", error="Cannot open video file")
            return   # finally block releases semaphore

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _upd(status="running", total_frames=total_frames, processed=0)

        # Quality flags
        use_sr    = quality != "mild"
        force_tta = quality == "maximum"

        # Load SAM3 for this job
        predictor = SAM3SemanticPredictor(MODEL_PATH)
        predictor.set_classes(labels)
        _set_predictor_conf(predictor, confidence)

        # Shared SR
        upsampler, sr_outscale = _get_realesrgan()

        # Output writer (start after first SR'd frame so we know final size)
        writer_obj  = None
        out_w = out_h = None

        blender = _TemporalBlender()
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            level = _brightness_level(frame)

            # 1. Enhancement
            f = _enhance(frame, level)

            # 2. Temporal blend
            if force_tta or level in ("very_low", "night"):
                f = blender.update(f)

            # 3. Grid
            q_grid = {"fast": (2,2), "balanced": (3,3), "maximum": (4,4)}
            q_ovl  = {"fast": 0.10,  "balanced": 0.20,  "maximum": 0.30}
            if level in ("very_low", "night"):
                n_cols, n_rows = 4, 4
            else:
                n_cols, n_rows = q_grid.get(quality, (3, 3))
            _job_overlap = q_ovl.get(quality, 0.20)

            # 4. Dynamic confidence
            eff_conf = confidence * _CONF_SCALE.get(level, 1.0)
            _set_predictor_conf(predictor, eff_conf)

            # 5. SR
            if use_sr and upsampler is not None:
                f_sr = _apply_sr(f, upsampler, outscale=sr_outscale)
                sx   = f.shape[1] / f_sr.shape[1]
                sy   = f.shape[0] / f_sr.shape[0]
            else:
                f_sr = f
                sx = sy = 1.0

            # 6. SAHI
            sahi_dets = _run_sahi(f_sr, predictor, labels, n_cols, n_rows,
                                  overlap=_job_overlap)
            if (sx != 1.0 or sy != 1.0) and len(sahi_dets) > 0:
                sahi_dets = sv.Detections(
                    xyxy=_scale_boxes(sahi_dets.xyxy, sx, sy),
                    confidence=sahi_dets.confidence,
                    class_id=sahi_dets.class_id,
                )

            # 7. Full-frame
            full_dets = _infer_full_frame(f, predictor, labels, level)

            # 8. TTA
            fl_dets = sv.Detections.empty()
            if force_tta or level in ("low", "very_low", "night"):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fw_ = f.shape[1]
                    f_fl = cv2.flip(f, 1)
                    fl_dets = _infer_full_frame(f_fl, predictor, labels, level)
                    if len(fl_dets) > 0:
                        xyxy_fl = fl_dets.xyxy.copy()
                        xyxy_fl[:, [0, 2]] = fw_ - xyxy_fl[:, [2, 0]]
                        fl_dets = sv.Detections(
                            xyxy=xyxy_fl,
                            confidence=fl_dets.confidence,
                            class_id=fl_dets.class_id,
                        )
                except (torch.cuda.OutOfMemoryError, Exception):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fl_dets = sv.Detections.empty()

            # 9. Merge + NMS
            _set_predictor_conf(predictor, confidence)
            dets = sv.Detections.merge([sahi_dets, full_dets, fl_dets])
            if len(dets) > 0:
                keep = sv.box_non_max_suppression(dets.xyxy, dets.confidence or np.ones(len(dets)), 0.45)
                dets = dets[keep]

            # Draw on original-res frame (not SR'd)
            out_frame = frame.copy()
            if len(dets) > 0:
                annotator = sv.BoxAnnotator(thickness=2)
                labeler   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
                class_names = [labels[i] if i is not None and i < len(labels) else "?"
                               for i in (dets.class_id if dets.class_id is not None
                                         else [None]*len(dets))]
                det_labels  = [f"{n} {c:.2f}" for n, c in
                               zip(class_names, dets.confidence if dets.confidence is not None
                                   else [0.0]*len(dets))]
                out_frame = annotator.annotate(out_frame.copy(), dets)
                out_frame = labeler.annotate(out_frame, dets, det_labels)

            # Init writer on first frame
            if writer_obj is None:
                out_w, out_h = out_frame.shape[1], out_frame.shape[0]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_obj = cv2.VideoWriter(out_path, fourcc, fps_in, (out_w, out_h))

            writer_obj.write(out_frame)
            processed += 1
            if processed % 30 == 0:
                pct = int(processed / max(total_frames, 1) * 100)
                _upd(processed=processed, progress_pct=pct)

        cap.release()
        if writer_obj is not None:
            writer_obj.release()

        _upd(status="done", processed=processed, progress_pct=100,
             output_filename=out_name)

    except Exception as exc:
        _upd(status="failed", error=str(exc))
        print(f"[enhance][{job_id[:8]}] error: {exc}")
    finally:
        _enhance_sem.release()


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/enhance")
def start_enhance(req: EnhanceRequest):
    if not req.labels:
        raise HTTPException(400, "Provide at least one label.")
    fpath = os.path.join(OUTPUT_DIR, os.path.basename(req.filename))
    if not os.path.exists(fpath):
        raise HTTPException(404, f"Recording '{req.filename}' not found.")

    job_id = str(uuid.uuid4())
    job = {
        "job_id":        job_id,
        "filename":      req.filename,
        "labels":        req.labels,
        "confidence":    req.confidence,
        "quality":       req.quality,
        "status":        "queued",
        "processed":     0,
        "total_frames":  0,
        "progress_pct":  0,
        "output_filename": None,
        "error":         None,
        "created_at":    datetime.now().isoformat(),
    }
    with enhance_jobs_lock:
        enhance_jobs[job_id] = job

    def _run():
        _enhance_sem.acquire()
        _run_enhance_job(job_id, req.filename, req.labels, req.confidence, req.quality)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "queued", "job_id": job_id}


@app.get("/enhance/jobs")
def list_enhance_jobs():
    with enhance_jobs_lock:
        return JSONResponse(list(enhance_jobs.values()))


@app.get("/enhance/{job_id}")
def get_enhance_job(job_id: str):
    with enhance_jobs_lock:
        job = enhance_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


@app.post("/session/start")
def start_session(req: StartRequest):
    if not req.labels:
        raise HTTPException(400, "Provide at least one label.")

    sid = req.session_id or str(uuid.uuid4())

    with sessions_lock:
        if sid in sessions and sessions[sid]["running"]:
            raise HTTPException(409, f"Session '{sid}' is already running.")
        sess = _make_session(sid, req.rtsp_url, req.labels, req.save)
        sess["compliance"] = req.compliance
        sessions[sid] = sess

    t = threading.Thread(
        target=detection_loop,
        args=(sess, req.confidence, req.every_n, req.quality),
        daemon=True,
    )
    sess["_thread"] = t
    t.start()
    return {"status": "started", "session_id": sid,
            "labels": req.labels, "rtsp_url": req.rtsp_url}


@app.post("/session/{session_id}/stop")
def stop_session(session_id: str):
    sess = _get_session(session_id)
    with sess["_lock"]:
        if not sess["running"]:
            return {"status": "not running"}
        sess["running"] = False
        sess["saving"]  = False
    thread = sess.get("_thread")

    def _cleanup():
        # Wait for detection_loop to exit cleanly before removing session
        if thread is not None:
            thread.join(timeout=8)
        with sessions_lock:
            sessions.pop(session_id, None)

    threading.Thread(target=_cleanup, daemon=True).start()
    return {"status": "stopping", "session_id": session_id}


@app.post("/session/{session_id}/save")
def toggle_save(session_id: str, req: SaveRequest):
    sess = _get_session(session_id)
    with sess["_lock"]:
        if not sess["running"]:
            raise HTTPException(400, "Session is not running.")
        sess["saving"] = req.save
    return {"status": "saving" if req.save else "not saving", "session_id": session_id}


@app.post("/session/{session_id}/show_lost")
def toggle_show_lost(session_id: str):
    sess = _get_session(session_id)
    with sess["_lock"]:
        sess["show_lost"] = not sess.get("show_lost", False)
        current = sess["show_lost"]
    return {"show_lost": current, "session_id": session_id}


@app.get("/session/{session_id}/status")
def session_status(session_id: str):
    sess = _get_session(session_id)
    with sess["_lock"]:
        return JSONResponse(_session_public(sess))


@app.get("/sessions")
def list_sessions():
    with sessions_lock:
        sids = list(sessions.keys())
    result = []
    for sid in sids:
        with sessions_lock:
            sess = sessions.get(sid)
        if sess is None:
            continue
        with sess["_lock"]:
            result.append(_session_public(sess))
    return JSONResponse(result)


@app.get("/stream/{session_id}")
def stream_session(session_id: str):
    """MJPEG stream for a specific session."""
    sess = _get_session(session_id)

    def _generate():
        while True:
            with sess["_lock"]:
                running = sess["running"]
                frame   = sess["frame"]
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for stream...", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            if not running:
                break
            time.sleep(0.033)

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video")
def video_feed():
    """Backward-compatible single-stream endpoint — serves the first active session."""
    def _generate():
        while True:
            frame = None
            with sessions_lock:
                for sess in sessions.values():
                    with sess["_lock"]:
                        if sess["running"] and sess["frame"] is not None:
                            frame = sess["frame"].copy()
                            break
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No active session", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.033)

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


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


CHUNK_SIZE = 1024 * 256  # 256 KB per read

@app.api_route("/recordings/{filename}", methods=["GET", "HEAD"])
async def stream_recording(filename: str, request: Request):
    filename = os.path.basename(filename)   # prevent path traversal
    fpath    = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")

    file_size    = os.path.getsize(fpath)
    range_header = request.headers.get("range")
    mime_type    = "video/mp4" if filename.endswith(".mp4") else "video/x-matroska"

    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type=mime_type,
            headers={
                "Accept-Ranges":  "bytes",
                "Content-Length": str(file_size),
                "Content-Type":   mime_type,
            },
        )

    if range_header:
        range_val = range_header.strip().lower().replace("bytes=", "")
        parts     = range_val.split("-")
        start     = int(parts[0]) if parts[0] else 0
        if len(parts) > 1 and parts[1]:
            end = int(parts[1])
        else:
            end = min(start + CHUNK_SIZE * 8 - 1, file_size - 1)
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

        return StreamingResponse(
            _iter_range(),
            status_code=206,
            media_type=mime_type,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type":   mime_type,
            },
        )

    def _iter_full():
        with open(fpath, "rb") as f:
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    return StreamingResponse(
        _iter_full(),
        status_code=200,
        media_type=mime_type,
        headers={
            "Accept-Ranges":  "bytes",
            "Content-Length": str(file_size),
            "Content-Type":   mime_type,
        },
    )


@app.get("/resources")
def resources():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    gpu_util = gpu_mem_used = gpu_mem_total = None
    try:
        raw = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]).decode().strip()
        parts         = raw.split(",")
        gpu_util      = int(parts[0].strip())
        gpu_mem_used  = int(parts[1].strip())
        gpu_mem_total = int(parts[2].strip())
    except Exception:
        pass
    return JSONResponse({
        "cpu_percent":   round(cpu, 1),
        "ram_percent":   round(ram.percent, 1),
        "ram_used_gb":   round(ram.used  / 1e9, 1),
        "ram_total_gb":  round(ram.total / 1e9, 1),
        "gpu_util":      gpu_util,
        "gpu_mem_used":  gpu_mem_used,
        "gpu_mem_total": gpu_mem_total,
    })


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "gpu":     torch.cuda.is_available(),
        "device":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sessions": len(sessions),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.on_event("startup")
def _startup():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    print("[server] SAM3 multi-stream detection server ready")
