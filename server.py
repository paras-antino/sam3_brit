# server.py — SAM3 + ByteTrack + MJPEG browser stream + MP4/CSV save
# Run: uvicorn server:app --host 0.0.0.0 --port 8000

import io
import cv2
import csv
import time
import threading
import numpy as np
from PIL import Image as PILImage
from collections import defaultdict
from datetime import datetime

import torch
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

import supervision as sv
from supervision import ByteTrack
from ultralytics.models.sam import SAM3SemanticPredictor

# ── Config — edit these ───────────────────────────────────────────────────────
RTSP_URL        = "rtsp://your-stream-url-here"   # ← your RTSP stream
MODEL_PATH      = "/home/paras/sam3/sam3.pt"
DETECT_PROMPTS  = ["person", "laptop", "mobile phone"]
CONFIDENCE      = 0.30
SAM3_EVERY_N    = 5          # run SAM3 every N frames, ByteTrack fills the rest
MAX_FRAMES      = None       # None = run forever
SAVE_VIDEO      = True
SAVE_CSV        = True
OUTPUT_DIR      = "/home/paras/sam3/outputs"
API_KEY         = "changeme123"   # ← change this, add ?key=yourkey to browser URL

LABEL_COLORS = {
    "person":       (0,   200, 100),
    "laptop":       (255, 165,   0),
    "mobile phone": (100, 149, 237),
}

# ── State shared between threads ──────────────────────────────────────────────
state = {
    "running":      False,
    "frame":        None,       # latest annotated frame (BGR numpy array)
    "counts":       {},         # current detection counts
    "fps":          0.0,
    "frame_idx":    0,
    "started_at":   None,
    "error":        None,
}
state_lock  = threading.Lock()
stream_lock = threading.Lock()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


# ── HTML viewer page ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>SAM3 Live Detection</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: #0d0d0d;
          color: #e0e0e0;
          font-family: 'Courier New', monospace;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 24px;
          min-height: 100vh;
        }
        h1 { color: #00c864; font-size: 1.3rem; margin-bottom: 6px; letter-spacing: 2px; }
        #subtitle { color: #555; font-size: 0.75rem; margin-bottom: 20px; }
        #stream {
          border: 1px solid #222;
          border-radius: 6px;
          max-width: 100%;
          width: 960px;
        }
        #stats {
          margin-top: 16px;
          display: flex;
          gap: 24px;
          flex-wrap: wrap;
          justify-content: center;
        }
        .stat-box {
          background: #161616;
          border: 1px solid #2a2a2a;
          border-radius: 6px;
          padding: 12px 20px;
          text-align: center;
          min-width: 120px;
        }
        .stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { font-size: 1.5rem; font-weight: bold; margin-top: 4px; }
        .person       { color: #00c864; }
        .laptop       { color: #ffa500; }
        .mobile-phone { color: #6495ed; }
        .fps          { color: #cc44cc; }
        .frames       { color: #aaaaaa; }
        footer { margin-top: 24px; color: #333; font-size: 0.7rem; }
      </style>
    </head>
    <body>
      <h1>🎯 SAM3 LIVE DETECTION</h1>
      <div id="subtitle">RTX 5060 Ti · SAM3 + ByteTrack · MJPEG</div>
      <img id="stream" src="/video" />
      <div id="stats">
        <div class="stat-box">
          <div class="stat-label">Person</div>
          <div class="stat-value person" id="s-person">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Laptop</div>
          <div class="stat-value laptop" id="s-laptop">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Mobile Phone</div>
          <div class="stat-value mobile-phone" id="s-mobile-phone">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">FPS</div>
          <div class="stat-value fps" id="s-fps">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Frames</div>
          <div class="stat-value frames" id="s-frames">—</div>
        </div>
      </div>
      <footer>Auto-refreshing stats every 2s</footer>
      <script>
        async function refreshStats() {
          try {
            const r = await fetch('/stats');
            const d = await r.json();
            document.getElementById('s-person').textContent       = d.counts['person'] ?? 0;
            document.getElementById('s-laptop').textContent       = d.counts['laptop'] ?? 0;
            document.getElementById('s-mobile-phone').textContent = d.counts['mobile phone'] ?? 0;
            document.getElementById('s-fps').textContent          = d.fps.toFixed(1);
            document.getElementById('s-frames').textContent       = d.frame_idx;
          } catch(e) {}
        }
        setInterval(refreshStats, 2000);
        refreshStats();
      </script>
    </body>
    </html>
    """


# ── MJPEG stream endpoint ─────────────────────────────────────────────────────
def mjpeg_generator():
    """Yield MJPEG frames to the browser."""
    while True:
        with state_lock:
            frame = state["frame"]

        if frame is None:
            # Send a blank frame while warming up
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for stream...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            frame = blank

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )
        time.sleep(0.033)   # ~30fps cap on the stream endpoint


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Stats endpoint ────────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    with state_lock:
        return JSONResponse({
            "running":   state["running"],
            "counts":    state["counts"],
            "fps":       round(state["fps"], 2),
            "frame_idx": state["frame_idx"],
            "error":     state["error"],
        })


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu":    torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_free_gb": round(
            (torch.cuda.get_device_properties(0).total_memory -
             torch.cuda.memory_allocated(0)) / 1e9, 2
        ) if torch.cuda.is_available() else None,
    }


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_panel(frame, counts, track_count, fps_val):
    labels = list(counts.keys())
    panel_h = 50 + 28 * max(len(labels), 1)
    cv2.rectangle(frame, (8, 8), (260, panel_h), (15, 15, 15), -1)
    cv2.rectangle(frame, (8, 8), (260, panel_h), (55, 55, 55), 1)
    cv2.putText(frame, f"SAM3 + ByteTrack  {fps_val:.1f} fps",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(frame, f"Tracked: {track_count}",
                (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
    for idx, (label, cnt) in enumerate(counts.items()):
        color = LABEL_COLORS.get(label, (200, 50, 200))
        y = 62 + idx * 28
        cv2.rectangle(frame, (14, y - 12), (26, y), color, -1)
        cv2.putText(frame, f"{label}: {cnt}", (32, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    return frame


# ── Main detection loop (runs in background thread) ───────────────────────────
def detection_loop():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load SAM3 ─────────────────────────────────────────────────────────────
    print("Loading SAM3...")
    overrides = dict(
        conf=CONFIDENCE,
        task="segment",
        mode="predict",
        model=MODEL_PATH,
        half=True,
        imgsz=644,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("✅ SAM3 loaded")

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    tracker = ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=15,
    )
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

    # ── Open RTSP ─────────────────────────────────────────────────────────────
    print(f"Connecting to RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        with state_lock:
            state["error"] = "Could not open RTSP stream"
        print("❌ Could not open RTSP stream")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Stream: {width}x{height} @ {fps}fps")

    # ── Output files ──────────────────────────────────────────────────────────
    ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer  = None
    csv_file = None

    if SAVE_VIDEO:
        video_path = f"{OUTPUT_DIR}/detection_{ts_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        print(f"📹 Saving video to: {video_path}")

    if SAVE_CSV:
        csv_path = f"{OUTPUT_DIR}/counts_{ts_str}.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp"] + DETECT_PROMPTS + ["total_tracks", "fps"])
        print(f"📄 Saving CSV to: {csv_path}")

    # ── Loop ──────────────────────────────────────────────────────────────────
    frame_idx     = 0
    t_start       = time.time()
    t_frame       = time.time()
    fps_display   = 0.0
    last_sv_dets  = sv.Detections.empty()

    with state_lock:
        state["running"]    = True
        state["started_at"] = datetime.now().isoformat()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Stream lost — reconnecting in 2s...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(RTSP_URL)
                if not cap.isOpened():
                    print("❌ Reconnect failed.")
                    break
                continue

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                print(f"✅ Reached MAX_FRAMES ({MAX_FRAMES})")
                break

            # ── SAM3 every N frames ───────────────────────────────────────────
            if frame_idx % SAM3_EVERY_N == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb)
                predictor.set_image(pil_img)
                results = predictor(text=DETECT_PROMPTS)

                if results and results[0].boxes is not None:
                    r       = results[0]
                    boxes   = r.boxes.xyxy.cpu().numpy()
                    scores  = r.boxes.conf.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    last_sv_dets = sv.Detections(
                        xyxy=boxes,
                        confidence=scores,
                        class_id=cls_ids,
                    )
                else:
                    last_sv_dets = sv.Detections.empty()

            # ── ByteTrack every frame ─────────────────────────────────────────
            tracked = tracker.update_with_detections(last_sv_dets)

            # ── Build labels and counts ───────────────────────────────────────
            counts      = defaultdict(int)
            label_texts = []
            for tid, cls_id, conf in zip(
                tracked.tracker_id if tracked.tracker_id is not None else [],
                tracked.class_id   if tracked.class_id   is not None else [],
                tracked.confidence if tracked.confidence is not None else [],
            ):
                name = DETECT_PROMPTS[cls_id] if cls_id < len(DETECT_PROMPTS) else f"cls_{cls_id}"
                counts[name] += 1
                label_texts.append(f"#{tid} {name} {conf:.2f}")

            # ── Annotate ──────────────────────────────────────────────────────
            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, tracked)
            annotated = box_annotator.annotate(annotated, tracked)
            if label_texts:
                annotated = label_annotator.annotate(annotated, tracked, labels=label_texts)

            # ── FPS ───────────────────────────────────────────────────────────
            now         = time.time()
            fps_display = 1.0 / max(now - t_frame, 1e-6)
            t_frame     = now

            annotated = draw_panel(annotated, counts, len(tracked), fps_display)

            elapsed = now - t_start
            cv2.putText(
                annotated,
                f"Frame {frame_idx:05d} | {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
            )

            # ── Push to browser stream ────────────────────────────────────────
            with state_lock:
                state["frame"]     = annotated.copy()
                state["counts"]    = dict(counts)
                state["fps"]       = fps_display
                state["frame_idx"] = frame_idx

            # ── Save ──────────────────────────────────────────────────────────
            if writer:
                writer.write(annotated)

            if csv_file:
                row = [frame_idx, round(elapsed, 2)]
                row += [counts.get(p, 0) for p in DETECT_PROMPTS]
                row += [len(tracked), round(fps_display, 2)]
                csv_writer.writerow(row)

            if frame_idx % 30 == 0:
                count_str = " | ".join(f"{k}: {v}" for k, v in counts.items())
                print(f"Frame {frame_idx:05d} | {fps_display:.1f}fps — {count_str}")

            frame_idx += 1

    except Exception as e:
        with state_lock:
            state["error"] = str(e)
        print(f"❌ Detection loop error: {e}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        with state_lock:
            state["running"] = False
        print("🛑 Detection loop stopped.")


# ── Start detection loop when server boots ────────────────────────────────────
@app.on_event("startup")
def startup():
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("🚀 Detection thread started")