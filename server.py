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
    # cap compliance
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
    cap_label:  str   = ""   # if set, enables head-cap compliance mode


class SaveRequest(BaseModel):
    save: bool


# ── Cap compliance helpers ────────────────────────────────────────────────────

def _boxes_overlap(box_a, box_b) -> bool:
    """Return True if two (x1,y1,x2,y2) boxes overlap."""
    return (box_a[0] < box_b[2] and box_a[2] > box_b[0] and
            box_a[1] < box_b[3] and box_a[3] > box_b[1])


def _person_has_cap(person_box, cap_boxes) -> bool:
    """True if any cap box overlaps the top-40% (head) region of the person box."""
    px1, py1, px2, py2 = person_box
    head_region = (px1, py1, px2, py1 + 0.40 * (py2 - py1))
    return any(_boxes_overlap(head_region, cb) for cb in cap_boxes)


# ── Writer helpers ────────────────────────────────────────────────────────────
def open_writers(ts: str, labels_list: list, width: int, height: int, fps: float,
                 cap_mode: bool = False):
    """Start ffmpeg process + CSV file.  Must NOT be called while holding session_lock."""
    global writer, csv_file, csv_writer

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

    with writer_lock:
        new_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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

    print(f"[writer] scratch → {scratch}")
    print(f"[writer] CSV     → {cpath}")


def _remux_to_mp4(scratch: str) -> str:
    mp4_path = scratch.replace(".mkv", ".mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i",        scratch,
        "-codec",    "copy",
        "-movflags", "+faststart",
        mp4_path,
    ]
    print(f"[remux] {scratch} → {mp4_path}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode == 0:
        os.remove(scratch)
        print(f"[remux] done → {mp4_path}")
        return mp4_path
    else:
        err = result.stderr.decode(errors="replace")
        print(f"[remux] FAILED: {err}")
        return scratch


def close_writers():
    global writer, csv_file, csv_writer

    scratch_path = None

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

    # Compliance-specific annotators (only used when cap_mode is True)
    if cap_mode:
        viol_box_ann = sv.BoxAnnotator(
            thickness=3,
            color=sv.ColorPalette.from_hex(["#FF3333"]),
        )
        viol_lbl_ann = sv.LabelAnnotator(
            text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#FF3333"]),
            text_color=sv.Color.WHITE,
        )
        ok_box_ann = sv.BoxAnnotator(
            thickness=3,
            color=sv.ColorPalette.from_hex(["#00C864"]),
        )
        ok_lbl_ann = sv.LabelAnnotator(
            text_scale=0.55, text_thickness=2,
            color=sv.ColorPalette.from_hex(["#00C864"]),
            text_color=sv.Color.BLACK,
        )

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
                     labels, width, height, stream_fps, cap_mode)
        with session_lock:
            session["saving"] = True

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

                cap_boxes = [tracked.xyxy[i] for i in range(n) if cls_arr[i] in cap_cls]

                viol_idx   = [i for i in range(n)
                              if cls_arr[i] in person_cls
                              and not _person_has_cap(tracked.xyxy[i], cap_boxes)]
                comply_idx = [i for i in range(n)
                              if cls_arr[i] in person_cls
                              and _person_has_cap(tracked.xyxy[i], cap_boxes)]

                violations = len(viol_idx)
                compliant  = len(comply_idx)

                def _make_dets(idx_list):
                    idx = np.array(idx_list)
                    return sv.Detections(
                        xyxy=tracked.xyxy[idx],
                        confidence=conf_arr[idx],
                        class_id=cls_arr[idx],
                        tracker_id=tid_arr[idx],
                    )

                if viol_idx:
                    vd = _make_dets(viol_idx)
                    vl = [f"#{tid_arr[i]} NO CAP" for i in viol_idx]
                    annotated = viol_box_ann.annotate(annotated, vd)
                    annotated = viol_lbl_ann.annotate(annotated, vd, labels=vl)

                if comply_idx:
                    cd = _make_dets(comply_idx)
                    cl_labels = [f"#{tid_arr[i]} CAP OK" for i in comply_idx]
                    annotated = ok_box_ann.annotate(annotated, cd)
                    annotated = ok_lbl_ann.annotate(annotated, cd, labels=cl_labels)

                # Violation flash banner at top of frame
                if violations > 0:
                    banner_h = 36
                    overlay  = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
                    msg = f"  VIOLATION: {violations} worker(s) without head cap"
                    cv2.putText(annotated, msg, (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ── Timing ────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now
            elapsed     = now - t_start

            # ── Push annotated frame to browser stream ────────────────────────
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
                      f"counts={dict(counts)}"
                      + (f" violations={violations} compliant={compliant}" if cap_mode else ""))

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

        # In cap-compliance mode, ensure "person" is in labels so SAM3 detects people
        labels = list(req.labels)
        if req.cap_label:
            if req.cap_label not in labels:
                raise HTTPException(400, f"cap_label '{req.cap_label}' must be in labels list.")
            if "person" not in labels:
                labels.insert(0, "person")

        session.update({
            "labels":     labels,
            "rtsp_url":   req.rtsp_url,
            "frame":      None,
            "counts":     {},
            "fps":        0.0,
            "frame_idx":  0,
            "error":      None,
            "saving":     req.save,
            "save_path":  None,
            "violations": 0,
            "compliant":  0,
            "cap_label":  req.cap_label,
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
            session["frame"]      = None
            session["counts"]     = {}
            session["fps"]        = 0.0
            session["frame_idx"]  = 0
            session["violations"] = 0
            session["compliant"]  = 0

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
            "violations": session["violations"],
            "compliant":  session["compliant"],
            "cap_label":  session["cap_label"],
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


CHUNK_SIZE = 1024 * 256  # 256 KB

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
