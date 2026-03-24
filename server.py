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
MAX_FRAMES = None   # set to an int to cap frames per session

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
}
session_lock   = threading.Lock()
session_thread = None

# ── Global writer state ───────────────────────────────────────────────────────
# These globals are ONLY ever assigned inside open_writers() / close_writers(),
# never directly inside detection_loop — that was the "referenced before
# assignment" bug (Python treats any name assigned anywhere in a function as
# local for the whole function).
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


class SaveRequest(BaseModel):
    save: bool


# ── Writer helpers ────────────────────────────────────────────────────────────
def open_writers(ts: str, labels_list: list, width: int, height: int, fps: float):
    """Start ffmpeg process + CSV file.  Must NOT be called while holding session_lock."""
    global writer, csv_file, csv_writer

    vpath = os.path.join(OUTPUT_DIR, f"detection_{ts}.mp4")
    cpath = os.path.join(OUTPUT_DIR, f"counts_{ts}.csv")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(int(fps)),
        "-i", "pipe:0",
        "-codec:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        vpath,
    ]

    with writer_lock:
        new_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"[ffmpeg] started PID={new_proc.pid}")

        new_csv = open(cpath, "w", newline="")
        new_cw  = csv.writer(new_csv)
        new_cw.writerow(["frame", "timestamp"] + labels_list + ["total_tracks", "fps"])

        writer     = new_proc
        csv_file   = new_csv
        csv_writer = new_cw

    # Update session path separately to keep lock ordering consistent
    with session_lock:
        session["save_path"] = vpath

    print(f"[writer] video → {vpath}")
    print(f"[writer] CSV   → {cpath}")


def close_writers():
    """Flush and close ffmpeg + CSV.  Safe to call even if already closed."""
    global writer, csv_file, csv_writer

    with writer_lock:
        if writer is not None:
            try:
                writer.stdin.close()
                writer.wait()
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
        session["save_path"] = None

    print("[writer] closed")


# ── Background cleanup (delete files older than 30 min) ──────────────────────
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
                        print(f"[cleanup] deleted {fname}")
        except Exception as exc:
            print(f"[cleanup] error: {exc}")
        time.sleep(300)


# ── Detection loop ────────────────────────────────────────────────────────────
def detection_loop(rtsp_url: str, labels: list, confidence: float,
                   every_n: int, save_on_start: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("[sam3] loading model...")
    overrides = dict(
        conf=confidence, task="segment", mode="predict",
        model=MODEL_PATH, half=True, imgsz=644, verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("[sam3] model loaded")

    # ── Tracker + annotators ──────────────────────────────────────────────────
    tracker   = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30,
                          minimum_matching_threshold=0.8, frame_rate=15)
    box_ann   = sv.BoxAnnotator(thickness=2)
    lbl_ann   = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=40)

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

    # ── Open writers if requested at start ────────────────────────────────────
    if save_on_start:
        open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"),
                     labels, width, height, stream_fps)
        with session_lock:
            session["saving"] = True   # keep flag in sync

    # ── Loop state ────────────────────────────────────────────────────────────
    frame_idx    = 0
    t_start      = time.time()
    t_frame      = t_start
    fps_display  = 0.0
    last_sv_dets = sv.Detections.empty()

    with session_lock:
        session["running"]    = True
        session["started_at"] = datetime.now().isoformat()
        session["error"]      = None

    try:
        while True:
            # ── Stop / save-toggle signals ────────────────────────────────────
            with session_lock:
                should_run  = session["running"]
                should_save = session["saving"]

            if not should_run:
                print("[loop] stop signal received")
                break

            # Read writer state under lock — no direct global access in this function
            with writer_lock:
                writer_active = writer is not None

            if should_save and not writer_active:
                open_writers(datetime.now().strftime("%Y%m%d_%H%M%S"),
                             labels, width, height, stream_fps)
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
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            # ── SAM3 inference every N frames ─────────────────────────────────
            if frame_idx % every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = PILImage.fromarray(rgb)
                predictor.set_image(pil)
                results = predictor(text=labels)
                if results and results[0].boxes is not None:
                    r = results[0]
                    last_sv_dets = sv.Detections(
                        xyxy       = r.boxes.xyxy.cpu().numpy(),
                        confidence = r.boxes.conf.cpu().numpy(),
                        class_id   = r.boxes.cls.cpu().numpy().astype(int),
                    )
                else:
                    last_sv_dets = sv.Detections.empty()

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

            # ── Timing ────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            elapsed     = now - t_start

            # ── Push annotated frame to browser stream ────────────────────────
            with session_lock:
                session["frame"]     = annotated.copy()
                session["counts"]    = dict(counts)
                session["fps"]       = fps_display
                session["frame_idx"] = frame_idx

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
                current_cw.writerow(row)

            if frame_idx % 30 == 0:
                print(f"[loop] frame={frame_idx:05d} fps={fps_display:.1f} counts={dict(counts)}")

            frame_idx += 1

    except Exception as exc:
        with session_lock:
            session["error"] = str(exc)
        print(f"[loop] ERROR: {exc}")
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
        session.update({
            "labels":    req.labels,
            "rtsp_url":  req.rtsp_url,
            "frame":     None,
            "counts":    {},
            "fps":       0.0,
            "frame_idx": 0,
            "error":     None,
            "saving":    req.save,
            "save_path": None,
        })

    session_thread = threading.Thread(
        target=detection_loop,
        args=(req.rtsp_url, req.labels, req.confidence, req.every_n, req.save),
        daemon=True,
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

    def _clear_after_delay():
        time.sleep(4)
        with session_lock:
            session["frame"]     = None
            session["counts"]    = {}
            session["fps"]       = 0.0
            session["frame_idx"] = 0

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


@app.api_route("/recordings/{filename}", methods=["GET", "HEAD"])
def stream_recording(filename: str, request: Request):
    filename = os.path.basename(filename)   # prevent path traversal
    fpath    = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(404, "File not found")

    file_size    = os.path.getsize(fpath)
    range_header = request.headers.get("range")

    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    if range_header:
        range_val  = range_header.strip().lower().replace("bytes=", "")
        parts      = range_val.split("-")
        start      = int(parts[0]) if parts[0] else 0
        end        = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end        = min(end, file_size - 1)
        chunk_size = end - start + 1
        with open(fpath, "rb") as f:
            f.seek(start)
            data = f.read(chunk_size)
        return Response(
            content=data,
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    with open(fpath, "rb") as f:
        data = f.read()
    return Response(
        content=data,
        status_code=200,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


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
            time.sleep(0.033)   # ~30 fps cap for the browser stream

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


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