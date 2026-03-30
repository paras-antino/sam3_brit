# SAM3 Factory Safety Monitor

A real-time multi-stream object detection and tracking system for industrial safety and PPE compliance monitoring. Built on SAM3 (Segment Anything Model 3) with open-vocabulary text-guided detection, enhanced for night/low-light environments, and deployable on any RTSP camera network.

---

## What It Does

Point it at any IP camera, type what you want to find (e.g. `surgical cap`, `gloves`, `helmet`, `forklift`), and the system detects, tracks, and records them in real time — even at night, even 5 metres away, across multiple camera feeds simultaneously.

---

## Key Features

### Detection & Tracking
- **Open-vocabulary detection** — no retraining needed; describe objects in plain English
- **SAM3 (Segment Anything Model 3)** — state-of-the-art text-guided segmentation as the detection backbone
- **SAHI (Sliced Aided Hyper Inference)** — splits each frame into overlapping tiles (2×2 / 3×3 / 4×4) so small or distant objects that would be missed in full-frame inference are caught
- **ByteTrack** — multi-object Kalman-filter tracker; assigns persistent IDs across frames even through occlusion
- **Ghost track cache** — when ByteTrack loses a track, the last known bounding box is held for up to 60 seconds and re-associated if the object reappears; shown as an amber `[LOST]` box (toggle-able per stream)
- **Test-Time Augmentation (TTA)** — horizontal flip pass added and merged for harder-to-detect orientations
- **False positive filter** — post-detection filter removes detections that are too large, too small, wrong aspect ratio, or have near-zero texture (e.g. black walls mistaken for gloves)

### Night & Low-Light Pipeline
The system automatically classifies each frame into one of four lighting levels (`normal / low / very_low / night`) using a weighted brightness score (`p20 × 0.7 + mean × 0.3`) that correctly handles HDR / backlit scenes. Each level triggers a progressively stronger enhancement stack:

| Level | Score | Enhancement Pipeline |
|-------|-------|----------------------|
| Normal | ≥ 90 | CLAHE on L channel |
| Low | ≥ 50 | Reinhard tone map → gamma → CLAHE |
| Very Low | ≥ 22 | Denoise → tone map → white balance → gamma → CLAHE |
| Night | < 22 | Denoise → MSRCR → tone map → white balance → gamma → CLAHE + temporal blend |

- **CLAHE** — Contrast Limited Adaptive Histogram Equalisation on the LAB L-channel
- **MSRCR** — Multi-Scale Retinex with Colour Restoration (sigmas 15/80/200); restores colour and contrast lost in dark environments
- **Reinhard Tone Mapping** — compresses HDR luminance range; critical for backlit CCTV where bright doorway + dark room confuses standard gamma
- **Adaptive Gamma** — computed per-frame from mean brightness so the lift is always proportional
- **Bilateral Denoising** — edge-preserving noise reduction before super-resolution on dark frames
- **Temporal Blending** — exponential-decay average of last 4 frames (decay=0.55) to suppress temporal shot noise
- **Dynamic Confidence Scaling** — confidence threshold automatically lowered for darker frames (night → ×0.40 of base) so weak but real detections aren't discarded

### Super-Resolution (Real-ESRGAN) 
- **RealESRGAN x4plus** loaded once, shared across all sessions (GPU serialised via lock)
- Runs on the inference frame before SAHI tiling; detections are scaled back to original resolution
- tile=256, fp16, outscale=2 — tuned to avoid CUDA OOM on 16 GB GPU
- Skipped on `fast` quality preset; falls back gracefully if model weights not present

### Quality Presets

| Preset | Grid | Overlap | SR | TTA | Conf Scale |
|--------|------|---------|----|----|------------|
| Fast | 2×2 | 10% | No | Dark only | 1.00 |
| Balanced | 3×3 | 20% | Yes | Dark only | 1.00 |
| Maximum | 4×4 | 30% | Yes | Always | 0.85 |

Dark frames always upgrade to 4×4 regardless of preset.

### Multi-Stream Architecture
- Unlimited concurrent RTSP sessions; each session is fully isolated (separate SAM3 instance, ByteTrack, writers, locks)
- Real-ESRGAN shared across sessions (one GPU load, serialised calls)
- Sessions survive page reloads — reconnects automatically on browser open
- 3-second display buffer decouples inference latency from display framerate

### Recording
- FFmpeg pipeline: raw BGR frames → libx264 ultrafast CRF 23 → MKV scratch → remux to MP4
- Per-session CSV with per-label counts, total tracks, FPS logged every frame
- Start/stop recording independently from detection
- HTTP range-request support for in-browser video playback of recordings

### Offline Enhancement Jobs
- Apply the full enhancement + detection pipeline to any saved recording
- Progress tracked per-job; result saved as new annotated MP4
- One job at a time (semaphore) to avoid VRAM contention

---

## System Architecture

```
Browser (frontend.html)
    │
    │  REST + MJPEG + SSE
    ▼
FastAPI Server (server.py)
    │
    ├── /session/start  ──► detection_loop (thread per session)
    │                           │
    │                           ├── RTSP → OpenCV cap
    │                           ├── display_buf (deque, 3s delay)
    │                           │
    │                           ├── _inference_worker (background thread)
    │                           │       ├── _enhance()      ← lighting fix
    │                           │       ├── _TemporalBlender
    │                           │       ├── _apply_sr()     ← RealESRGAN
    │                           │       ├── _run_sahi()     ← tiled SAM3
    │                           │       ├── _infer_full_frame()
    │                           │       ├── TTA flip pass
    │                           │       ├── _merge() + NMS
    │                           │       └── _filter_false_positives()
    │                           │
    │                           ├── ByteTrack
    │                           ├── Ghost track cache
    │                           ├── FFmpeg writer
    │                           └── CSV writer
    │
    ├── /stream/{id}    ──► MJPEG generator (30 fps)
    ├── /recordings     ──► list / stream saved videos
    ├── /enhance        ──► offline job queue
    └── /resources      ──► CPU / RAM / GPU metrics
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection model | SAM3 (Segment Anything Model 3) via Ultralytics |
| Tracker | ByteTrack (supervision library) |
| Tiled inference | SAHI |
| Super-resolution | Real-ESRGAN x4plus (basicsr + realesrgan) |
| Image enhancement | OpenCV (CLAHE, bilateral, Reinhard, gamma) |
| Server | FastAPI + Uvicorn |
| Video encoding | FFmpeg (subprocess pipe) |
| Frontend | Vanilla JS + CSS custom properties (no framework) |
| GPU | CUDA via PyTorch (fp16 throughout) |

---

## Installation

```bash
# 1. Clone and enter
git clone <repo-url>
cd sam3_brit

# 2. Install Python dependencies
pip install fastapi uvicorn opencv-python-headless supervision ultralytics \
            torch torchvision numpy Pillow psutil realesrgan basicsr

# 3. Download SAM3 model weights
# Place sam3.pt at /home/paras/sam3/sam3.pt (or edit MODEL_PATH in server.py)

# 4. (Optional) Download Real-ESRGAN weights
# Place RealESRGAN_x4plus.pth in the same directory as sam3.pt
# Without it, SR is disabled and detection still works normally

# 5. Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://<server-ip>:8000` in any browser.

---

## Configuration

Edit the top of `server.py`:

```python
MODEL_PATH = "/home/paras/sam3/sam3.pt"   # SAM3 weights path
OUTPUT_DIR = "/home/paras/sam3/outputs"    # recordings + enhanced videos
MAX_FRAMES = None                          # cap frames per session (None = unlimited)
```

---

## Usage

### Starting a Stream
1. Click **+ Add Stream** (top right)
2. Enter RTSP URL (e.g. `rtsp://192.168.1.10:554/stream`)
3. Type detection targets and press Enter (e.g. `surgical cap`, `gloves`)
4. Choose quality preset (Fast / Balanced / Maximum)
5. Click **▶ Start Stream**

### Controls Per Stream
| Control | Action |
|---------|--------|
| Scroll wheel | Zoom in / out |
| Click + drag | Pan when zoomed |
| Double-click | Reset zoom |
| `⛶` button | Full screen |
| `◌ Lost` button | Toggle ghost track visibility |
| `⏺ Record` | Start saving to disk |
| `■ Stop` | Stop the stream |

### API (direct use)
```bash
# Start a session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url":"rtsp://...","labels":["helmet","vest"],"quality":"balanced"}'

# List active sessions
curl http://localhost:8000/sessions

# Stop a session
curl -X POST http://localhost:8000/session/{id}/stop

# List recordings
curl http://localhost:8000/recordings

# Enhance a recording with full pipeline + detection
curl -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{"filename":"detection_20260330_120000.mp4","labels":["helmet"],"quality":"maximum"}'
```

---

## Design Decisions & Trade-offs

**Why SAHI instead of just full-frame inference?**
At 5 metres, a surgical cap covers roughly 40×40 pixels on a 1080p feed. Full-frame SAM3 at imgsz=896 misses objects this small. SAHI tiles the frame and runs inference on each tile independently, then merges results — effectively giving the model a higher-resolution view of small regions.

**Why RealESRGAN before tiling, not after?**
SR before tiling means each tile is fed upscaled content, so SAM3 sees sharper edges and textures. SR after tiling would require running SR on N² tiles per frame — prohibitively slow.

**Why a 3-second display buffer?**
SAM3 + SAHI inference takes 1–4 seconds. Without the buffer, detections would appear 2–3 frames behind the displayed image, making tracking look jittery and wrong. The buffer holds 3 seconds of raw frames and pairs each one with the closest inference result by frame index, so boxes always appear on the frame they were computed for.

**Why Reinhard tone mapping for backlit scenes?**
Standard gamma correction lifts all pixels uniformly. In a backlit scene (bright doorway, dark room), gamma over-exposes the bright region while still under-exposing the dark one. Reinhard operates in log-luminance space and compresses both ends toward the midtone simultaneously — the correct operator for this class of CCTV footage.

**Why ByteTrack over SORT or DeepSORT?**
ByteTrack uses a two-stage association: high-confidence detections first, then low-confidence detections matched against existing tracks. This dramatically improves tracking through brief occlusions and at the detection confidence boundary — common with partial PPE views (e.g. glove partially behind equipment).

---

## Performance

Tested on Ubuntu 22.04, NVIDIA RTX 4080 (16 GB), Python 3.10:

| Configuration | Inference time | GPU VRAM |
|---------------|---------------|----------|
| Fast, 1 stream | ~300 ms/frame | ~3 GB |
| Balanced, 1 stream | ~800 ms/frame | ~5 GB |
| Maximum, 1 stream | ~2–4 s/frame | ~8 GB |
| Balanced, 2 streams | ~1.2 s/frame | ~7 GB |

Inference runs asynchronously — display always runs at full stream FPS regardless of inference speed.

---

## Project Structure

```
sam3_brit/
├── server.py          # FastAPI backend — all detection, tracking, recording logic
├── frontend.html      # Single-file frontend — no build step, no framework
└── outputs/           # Saved recordings and enhanced videos (auto-created)
```

---

## Interview Notes

### What problem does this solve?
Manual safety monitoring of factory/hospital floors is labour-intensive and error-prone. This system automates real-time PPE compliance checking across multiple camera feeds with no camera-specific training — operators describe what to look for in plain English.

### What was the hardest part?
Making detection reliable in night/low-light conditions. The naive approach (just lower the confidence threshold) generates too many false positives. The solution required a layered pipeline: classify the lighting level first, then apply the appropriate combination of MSRCR, Reinhard tone mapping, adaptive gamma, and dynamic confidence scaling — each chosen for a specific physical reason.

### What would you do differently?
Fine-tune SAM3 on rear-view PPE images. The biggest remaining gap is detecting surgical caps and gloves from behind — the CLIP text embedding for these labels was learned mostly from front-facing images. A small fine-tuning dataset of rear-view shots would close this gap without any pipeline changes.

### What did you learn?
That inference latency and display latency are separate problems that need separate solutions. Decoupling them with the 3-second frame buffer was non-obvious but essential — without it the system works but feels broken because boxes lag behind the video.
