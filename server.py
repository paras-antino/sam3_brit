import cv2, csv, time, threading, numpy as np, os, psutil, subprocess
from PIL import Image as PILImage
from collections import defaultdict
from datetime import datetime
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import supervision as sv
from supervision import ByteTrack
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/paras/sam3/sam3.pt"
OUTPUT_DIR = "/home/paras/sam3/outputs"
MAX_FRAMES = None

# ── Session state ─────────────────────────────────────────────────────────────
session = {
    "running":      False,
    "frame":        None,
    "counts":       {},
    "fps":          0.0,
    "frame_idx":    0,
    "error":        None,
    "labels":       [],
    "rtsp_url":     None,
    "started_at":   None,
    "saving":       False,   # whether we are currently saving to disk
    "save_path":    None,    # current video save path
}
session_lock   = threading.Lock()
session_thread = None

app = FastAPI()

# ── Request models ────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    rtsp_url:   str
    labels:     List[str]
    confidence: float = 0.30
    every_n:    int   = 5
    save:       bool  = False   # start saving immediately

class SaveRequest(BaseModel):
    save: bool   # True = start saving, False = stop saving

# ── Cleanup old output files (30 min) ────────────────────────────────────────
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
def detection_loop(rtsp_url, labels, confidence, every_n, save_on_start):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load SAM3
    print("Loading SAM3...")
    overrides = dict(
        conf=confidence, task="segment", mode="predict",
        model=MODEL_PATH, half=True, imgsz=644, verbose=False
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("SAM3 loaded")

    # ByteTrack + annotators
    tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                          minimum_matching_threshold=0.8, frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40)

    # Open RTSP
    print(f"Connecting RTSP: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    # Save state (can be toggled at runtime)
    writer     = None
    csv_file   = None
    csv_writer = None

    def open_writers(ts):
        nonlocal writer, csv_file, csv_writer
        vpath = f"{OUTPUT_DIR}/detection_{ts}.mp4"
        cpath = f"{OUTPUT_DIR}/counts_{ts}.csv"
        writer     = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        csv_file   = open(cpath, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame","timestamp"] + labels + ["total_tracks","fps"])
        with session_lock:
            session["save_path"] = vpath
        print(f"Saving video : {vpath}")
        print(f"Saving CSV   : {cpath}")

    def close_writers():
        nonlocal writer, csv_file, csv_writer
        if writer:   writer.release();   writer = None
        if csv_file: csv_file.close();   csv_file = None
        csv_writer = None
        with session_lock:
            session["save_path"] = None
        print("Stopped saving")

    if save_on_start:
        open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"))
        with session_lock:
            session["saving"] = True

    frame_idx    = 0
    t_start      = time.time()
    t_frame      = time.time()
    fps_display  = 0.0
    last_sv_dets = sv.Detections.empty()
    palette      = [(0,200,100),(255,165,0),(100,149,237),(200,50,200),
                    (0,200,200),(200,200,0),(255,80,80),(80,255,80)]

    with session_lock:
        session["running"]    = True
        session["started_at"] = datetime.now().isoformat()
        session["error"]      = None

    try:
        while True:
            # ── Check stop / save-toggle signals ─────────────────────────────
            with session_lock:
                should_run  = session["running"]
                should_save = session["saving"]

            if not should_run:
                print("Stop signal received")
                break

            # Toggle saving on/off at runtime
            if should_save and writer is None:
                open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"))
            elif not should_save and writer is not None:
                close_writers()

            # ── Read frame ───────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                with session_lock:
                    if not session["running"]:
                        break
                print("Stream lost, reconnecting...")
                cap.release()
                time.sleep(2)
                with session_lock:
                    if not session["running"]:
                        break
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("Reconnect failed")
                    break
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # ── SAM3 every N frames ──────────────────────────────────────────
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

            # ── ByteTrack every frame ────────────────────────────────────────
            tracked     = tracker.update_with_detections(last_sv_dets)
            counts      = defaultdict(int)
            label_texts = []

            for tid, cls_id, conf in zip(
                tracked.tracker_id if tracked.tracker_id is not None else [],
                tracked.class_id   if tracked.class_id   is not None else [],
                tracked.confidence if tracked.confidence is not None else []):
                name = labels[cls_id] if cls_id < len(labels) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            # ── Annotate (no panel overlay — stats shown in UI) ──────────────
            annotated = frame.copy()
            annotated = trace_ann.annotate(annotated, tracked)
            annotated = box_ann.annotate(annotated, tracked)
            if label_texts:
                annotated = lbl_ann.annotate(annotated, tracked, labels=label_texts)

            # ── FPS ──────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            elapsed     = now - t_start

            # ── Push to browser ──────────────────────────────────────────────
            with session_lock:
                session["frame"]     = annotated.copy()
                session["counts"]    = dict(counts)
                session["fps"]       = fps_display
                session["frame_idx"] = frame_idx

            # ── Write to disk if saving ──────────────────────────────────────
            if writer:
                writer.write(annotated)
            if csv_writer:
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
        close_writers()
        with session_lock:
            session["running"] = False
            session["saving"]  = False
        print("Detection loop stopped")

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session(req: StartRequest):
    global session_thread
    with session_lock:
        if session["running"]:
            raise HTTPException(status_code=409, detail="Session already running. Stop it first.")
        if not req.labels:
            raise HTTPException(status_code=400, detail="Provide at least one label.")
        session["labels"]    = req.labels
        session["rtsp_url"]  = req.rtsp_url
        session["frame"]     = None
        session["counts"]    = {}
        session["fps"]       = 0.0
        session["frame_idx"] = 0
        session["error"]     = None
        session["saving"]    = req.save
        session["save_path"] = None

    session_thread = threading.Thread(
        target=detection_loop,
        args=(req.rtsp_url, req.labels, req.confidence, req.every_n, req.save),
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
        session["saving"]  = False

    def force_clear():
        time.sleep(4)
        with session_lock:
            session["frame"]     = None
            session["counts"]    = {}
            session["fps"]       = 0.0
            session["frame_idx"] = 0
    threading.Thread(target=force_clear, daemon=True).start()
    return {"status": "stopping"}


@app.post("/session/save")
def toggle_save(req: SaveRequest):
    with session_lock:
        if not session["running"]:
            raise HTTPException(status_code=400, detail="No session running.")
        session["saving"] = req.save
    return {"status": "saving" if req.save else "not saving"}


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
            "saving":     session["saving"],
            "save_path":  session["save_path"],
        })


@app.get("/recordings")
def list_recordings():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for fname in sorted(os.listdir(OUTPUT_DIR), reverse=True):
            if not fname.endswith(".mp4"):
                continue
            fpath = os.path.join(OUTPUT_DIR, fname)
            stat  = os.stat(fpath)
            files.append({
                "name":     fname,
                "size_mb":  round(stat.st_size / 1e6, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return JSONResponse(files)


@app.get("/recordings/{filename}")
def stream_recording(filename: str):
    # Prevent path traversal
    filename = os.path.basename(filename)
    fpath    = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")

    def iter_file():
        with open(fpath, "rb") as f:
            while chunk := f.read(1024 * 256):
                yield chunk

    return StreamingResponse(iter_file(), media_type="video/mp4",
                             headers={"Content-Disposition": f"inline; filename={filename}"})


@app.get("/video")
def video_feed():
    def generate():
        while True:
            with session_lock:
                frame = session["frame"]
            if frame is None:
                blank = np.zeros((360,640,3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for stream...", (140,180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,60,60), 2)
                frame = blank
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/resources")
def resources():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    gpu_util = gpu_mem_used = gpu_mem_total = None
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        parts         = result.split(",")
        gpu_util      = int(parts[0].strip())
        gpu_mem_used  = int(parts[1].strip())
        gpu_mem_total = int(parts[2].strip())
    except Exception:
        pass
    return JSONResponse({
        "cpu_percent":  round(cpu, 1),
        "ram_percent":  round(ram.percent, 1),
        "ram_used_gb":  round(ram.used / 1e9, 1),
        "ram_total_gb": round(ram.total / 1e9, 1),
        "gpu_util":     gpu_util,
        "gpu_mem_used": gpu_mem_used,
        "gpu_mem_total":gpu_mem_total,
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
    with open(os.path.join(os.path.dirname(__file__), "frontend.html")) as f:
        return f.read()


@app.on_event("startup")
def startup():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    print("SAM3 server ready")