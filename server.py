import cv2, csv, time, threading, numpy as np, os
from PIL import Image as PILImage
from collections import defaultdict
from datetime import datetime
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import supervision as sv
from supervision import ByteTrack
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/paras/sam3/sam3.pt"
OUTPUT_DIR = "/home/paras/sam3/outputs"
SAVE_VIDEO = True
SAVE_CSV   = True

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
}
session_lock = threading.Lock()
session_thread = None

app = FastAPI()

# ── Request model ─────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    rtsp_url:   str
    labels:     List[str]
    confidence: float = 0.30
    every_n:    int   = 5

# ── Drawing helper ────────────────────────────────────────────────────────────
def draw_panel(frame, counts, track_count, fps_val, labels):
    panel_h = 50 + 28 * max(len(labels), 1)
    cv2.rectangle(frame, (8,8), (280,panel_h), (15,15,15), -1)
    cv2.rectangle(frame, (8,8), (280,panel_h), (55,55,55),  1)
    cv2.putText(frame, f"SAM3+ByteTrack {fps_val:.1f}fps",
                (14,26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    cv2.putText(frame, f"Tracked: {track_count}",
                (14,42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,190,190), 1)
    palette = [
        (0,200,100),(255,165,0),(100,149,237),(200,50,200),
        (0,200,200),(200,200,0),(255,80,80),(80,255,80),
    ]
    for idx, label in enumerate(labels):
        color = palette[idx % len(palette)]
        cnt   = counts.get(label, 0)
        y     = 62 + idx * 28
        cv2.rectangle(frame, (14,y-12), (26,y), color, -1)
        cv2.putText(frame, f"{label}: {cnt}", (32,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235,235,235), 1)
    return frame

# ── Cleanup old output files ──────────────────────────────────────────────────
def cleanup_loop():
    while True:
        try:
            cutoff = time.time() - 30 * 60
            if os.path.exists(OUTPUT_DIR):
                for fname in os.listdir(OUTPUT_DIR):
                    if not (fname.endswith(".mp4") or fname.endswith(".csv")):
                        continue
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        print(f"Deleted old file: {fname}")
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(300)

# ── Detection loop ────────────────────────────────────────────────────────────
def detection_loop(rtsp_url, labels, confidence, every_n):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading SAM3...")
    overrides = dict(conf=confidence, task="segment", mode="predict",
                     model=MODEL_PATH, half=True, imgsz=644, verbose=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("SAM3 loaded")

    tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                          minimum_matching_threshold=0.8, frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40)

    print(f"Connecting RTSP: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        with session_lock:
            session["error"]   = "Could not open RTSP stream"
            session["running"] = False
        print("ERROR: Could not open RTSP stream")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream: {width}x{height} @ {fps}fps")

    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer   = None
    csv_file = None

    if SAVE_VIDEO:
        vpath  = f"{OUTPUT_DIR}/detection_{ts_str}.mp4"
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"Saving video: {vpath}")

    if SAVE_CSV:
        cpath      = f"{OUTPUT_DIR}/counts_{ts_str}.csv"
        csv_file   = open(cpath, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame","timestamp"] + labels + ["total_tracks","fps"])
        print(f"Saving CSV: {cpath}")

    frame_idx    = 0
    t_start      = time.time()
    t_frame      = time.time()
    fps_display  = 0.0
    last_sv_dets = sv.Detections.empty()

    with session_lock:
        session["running"]    = True
        session["started_at"] = datetime.now().isoformat()
        session["error"]      = None

    try:
        while True:
            # Stop signal from /session/stop
            with session_lock:
                if not session["running"]:
                    print("Stop signal received")
                    break

            ret, frame = cap.read()
            if not ret:
                print("Stream lost, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Reconnect failed")
                    break
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # SAM3 every N frames
            if frame_idx % every_n == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = PILImage.fromarray(rgb)
                predictor.set_image(pil)
                results = predictor(text=labels)
                if results and results[0].boxes is not None:
                    r = results[0]
                    last_sv_dets = sv.Detections(
                        xyxy=r.boxes.xyxy.cpu().numpy(),
                        confidence=r.boxes.conf.cpu().numpy(),
                        class_id=r.boxes.cls.cpu().numpy().astype(int))
                else:
                    last_sv_dets = sv.Detections.empty()

            # ByteTrack every frame
            tracked     = tracker.update_with_detections(last_sv_dets)
            counts      = defaultdict(int)
            label_texts = []
            palette     = [(0,200,100),(255,165,0),(100,149,237),(200,50,200),
                           (0,200,200),(200,200,0),(255,80,80),(80,255,80)]

            for tid, cls_id, conf in zip(
                tracked.tracker_id if tracked.tracker_id is not None else [],
                tracked.class_id   if tracked.class_id   is not None else [],
                tracked.confidence if tracked.confidence is not None else []):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            annotated = frame.copy()
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            annotated   = draw_panel(annotated, counts, len(tracked), fps_display, labels)
            elapsed     = now - t_start
            cv2.putText(annotated,
                        f"Frame {frame_idx:05d} | {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                        (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

            with session_lock:
                session["frame"]     = annotated.copy()
                session["counts"]    = dict(counts)
                session["fps"]       = fps_display
                session["frame_idx"] = frame_idx

            if writer:
                writer.write(annotated)

            if csv_file:
                row = [frame_idx, round(elapsed,2)]
                row += [counts.get(l,0) for l in labels]
                row += [len(tracked), round(fps_display,2)]
                csv_writer.writerow(row)

            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx:05d} | {fps_display:.1f}fps | {dict(counts)}")

            frame_idx += 1

    except Exception as e:
        with session_lock:
            session["error"] = str(e)
        print(f"ERROR: {e}")
    finally:
        cap.release()
        if writer:   writer.release()
        if csv_file: csv_file.close()
        with session_lock:
            session["running"] = False
        print("Detection loop stopped")

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session(req: StartRequest):
    global session_thread
    with session_lock:
        if session["running"]:
            raise HTTPException(status_code=409, detail="A session is already running. Stop it first.")
        if not req.labels:
            raise HTTPException(status_code=400, detail="Provide at least one label.")
        session["labels"]   = req.labels
        session["rtsp_url"] = req.rtsp_url
        session["frame"]    = None
        session["counts"]   = {}
        session["fps"]      = 0.0
        session["frame_idx"]= 0
        session["error"]    = None

    session_thread = threading.Thread(
        target=detection_loop,
        args=(req.rtsp_url, req.labels, req.confidence, req.every_n),
        daemon=True
    )
    session_thread.start()
    return {"status": "started", "labels": req.labels, "rtsp_url": req.rtsp_url}


@app.post("/session/stop")
def stop_session():
    with session_lock:
        if not session["running"]:
            return {"status": "not running"}
        session["running"] = False
    return {"status": "stopping"}


@app.get("/session/status")
def session_status():
    with session_lock:
        return JSONResponse({
            "running":    session["running"],
            "labels":     session["labels"],
            "rtsp_url":   session["rtsp_url"],
            "counts":     session["counts"],
            "fps":        round(session["fps"], 2),
            "frame_idx":  session["frame_idx"],
            "started_at": session["started_at"],
            "error":      session["error"],
        })


@app.get("/video")
def video_feed():
    def generate():
        while True:
            with session_lock:
                frame = session["frame"]
            if frame is None:
                blank = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for stream...", (120,240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)
                frame = blank
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu":    torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "frontend.html")) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    print("SAM3 server ready — waiting for session start")