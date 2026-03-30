"""
server1.py — Simple backup detection server.
No SAHI, no SR, no complex enhancement. Just:
  RTSP → CLAHE → SAM3 full-frame → ByteTrack → MJPEG stream
Runs reliably with minimal GPU memory (~2 GB).
"""
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import supervision as sv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from supervision import ByteTrack
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/paras/sam3/sam3.pt"
OUTPUT_DIR = "/home/paras/sam3/outputs"

app = FastAPI()

# ── State ──────────────────────────────────────────────────────────────────────
sessions: dict = {}
sessions_lock  = threading.Lock()


# ── Models ─────────────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    rtsp_url:   str
    labels:     List[str]
    confidence: float        = 0.30
    every_n:    int          = 3
    session_id: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_session(sid, rtsp_url, labels):
    return {
        "id":        sid,
        "running":   False,
        "frame":     None,
        "counts":    {},
        "fps":       0.0,
        "error":     None,
        "labels":    labels,
        "rtsp_url":  rtsp_url,
        "_lock":     threading.Lock(),
        "_thread":   None,
    }


def _get_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
    if s is None:
        raise HTTPException(404, f"Session '{sid}' not found.")
    return s


def _clahe(frame):
    """Basic CLAHE on L channel — fast, reliable contrast boost."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Detection loop ─────────────────────────────────────────────────────────────
def detection_loop(sess: dict, confidence: float, every_n: int):
    sid    = sess["id"]
    labels = sess["labels"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load SAM3
    try:
        predictor = SAM3SemanticPredictor(overrides=dict(
            conf=confidence, task="segment", mode="predict",
            model=MODEL_PATH, half=True, imgsz=640, verbose=False,
        ))
        predictor.set_classes(labels)
        print(f"[s1][{sid[:8]}] SAM3 loaded  conf={confidence}")
    except Exception as exc:
        with sess["_lock"]:
            sess["error"]   = f"Model load failed: {exc}"
            sess["running"] = False
        print(f"[s1][{sid[:8]}] model error: {exc}")
        return

    tracker   = ByteTrack(track_activation_threshold=0.25,
                          lost_track_buffer=60,
                          minimum_matching_threshold=0.8,
                          frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Open RTSP
    cap = cv2.VideoCapture(sess["rtsp_url"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        with sess["_lock"]:
            sess["error"]   = "Could not open RTSP stream"
            sess["running"] = False
        print(f"[s1][{sid[:8]}] ERROR: cannot open stream")
        return

    with sess["_lock"]:
        sess["running"] = True

    print(f"[s1][{sid[:8]}] stream open")

    frame_idx     = 0
    last_dets     = sv.Detections.empty()
    t_frame       = time.time()
    reconnects    = 0
    MAX_RECONNECTS = 10

    try:
        while True:
            with sess["_lock"]:
                if not sess["running"]:
                    break

            ret, frame = cap.read()
            if not ret:
                reconnects += 1
                if reconnects > MAX_RECONNECTS:
                    with sess["_lock"]:
                        sess["error"] = "Stream lost — max reconnects reached"
                    break
                backoff = min(2 ** (reconnects - 1), 30)
                print(f"[s1][{sid[:8]}] reconnecting in {backoff}s ({reconnects}/{MAX_RECONNECTS})")
                cap.release()
                time.sleep(backoff)
                cap = cv2.VideoCapture(sess["rtsp_url"])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                last_dets = sv.Detections.empty()
                continue
            reconnects = 0

            # Apply CLAHE
            frame = _clahe(frame)

            # Run inference every N frames
            if frame_idx % every_n == 0:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results = predictor(frame)
                    if results and results[0].boxes is not None:
                        r = results[0]
                        last_dets = sv.Detections(
                            xyxy=r.boxes.xyxy.cpu().numpy(),
                            confidence=r.boxes.conf.cpu().numpy(),
                            class_id=r.boxes.cls.cpu().numpy().astype(int),
                        )
                    else:
                        last_dets = sv.Detections.empty()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[s1][{sid[:8]}] OOM — skipped frame")
                    last_dets = sv.Detections.empty()
                except Exception as exc:
                    print(f"[s1][{sid[:8]}] infer error: {exc}")
                    last_dets = sv.Detections.empty()

            # Track
            tracked = tracker.update_with_detections(last_dets)
            counts  = defaultdict(int)
            label_texts = []

            tids  = tracked.tracker_id if tracked.tracker_id is not None else []
            cids  = tracked.class_id   if tracked.class_id   is not None else []
            confs = tracked.confidence if tracked.confidence  is not None else []

            for tid, cid, conf in zip(tids, cids, confs):
                name = labels[cid] if cid < len(labels) else f"cls_{cid}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            # Annotate
            annotated = frame.copy()
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            # FPS
            now       = time.time()
            fps       = 1.0 / max(now - t_frame, 1e-6)
            t_frame   = now

            with sess["_lock"]:
                sess["frame"]  = annotated.copy()
                sess["counts"] = dict(counts)
                sess["fps"]    = round(fps, 1)

            frame_idx += 1

    except Exception as exc:
        import traceback
        print(f"[s1][{sid[:8]}] ERROR: {exc}")
        traceback.print_exc()
        with sess["_lock"]:
            sess["error"] = str(exc)
    finally:
        cap.release()
        with sess["_lock"]:
            sess["running"] = False
        print(f"[s1][{sid[:8]}] stopped")


# ── API ────────────────────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session(req: StartRequest):
    if not req.labels:
        raise HTTPException(400, "Provide at least one label.")
    sid = req.session_id or str(uuid.uuid4())
    with sessions_lock:
        if sid in sessions and sessions[sid]["running"]:
            raise HTTPException(409, f"Session '{sid}' already running.")
        sess = _make_session(sid, req.rtsp_url, req.labels)
        sessions[sid] = sess
    t = threading.Thread(
        target=detection_loop,
        args=(sess, req.confidence, req.every_n),
        daemon=True,
    )
    sess["_thread"] = t
    t.start()
    return {"status": "started", "session_id": sid}


@app.post("/session/{session_id}/stop")
def stop_session(session_id: str):
    sess = _get_session(session_id)
    with sess["_lock"]:
        sess["running"] = False
    thread = sess.get("_thread")
    def _cleanup():
        if thread:
            thread.join(timeout=8)
        with sessions_lock:
            sessions.pop(session_id, None)
    threading.Thread(target=_cleanup, daemon=True).start()
    return {"status": "stopping", "session_id": session_id}


@app.get("/session/{session_id}/status")
def session_status(session_id: str):
    sess = _get_session(session_id)
    with sess["_lock"]:
        return JSONResponse({
            "id":      sess["id"],
            "running": sess["running"],
            "labels":  sess["labels"],
            "counts":  sess["counts"],
            "fps":     sess["fps"],
            "error":   sess["error"],
        })


@app.get("/sessions")
def list_sessions():
    with sessions_lock:
        sids = list(sessions.keys())
    out = []
    for sid in sids:
        with sessions_lock:
            s = sessions.get(sid)
        if s is None:
            continue
        with s["_lock"]:
            out.append({
                "id":      s["id"],
                "running": s["running"],
                "labels":  s["labels"],
                "counts":  s["counts"],
                "fps":     s["fps"],
                "error":   s["error"],
                "rtsp_url": s["rtsp_url"],
            })
    return JSONResponse(out)


@app.get("/stream/{session_id}")
def stream_session(session_id: str):
    sess = _get_session(session_id)

    def _gen():
        while True:
            with sess["_lock"]:
                running = sess["running"]
                frame   = sess["frame"]
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Connecting...", (200, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            if not running:
                break
            time.sleep(0.033)

    return StreamingResponse(_gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "gpu":      torch.cuda.is_available(),
        "sessions": len(sessions),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(__file__), "frontend1.html")
    with open(path, encoding="utf-8") as f:
        return f.read()
