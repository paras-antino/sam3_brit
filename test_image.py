"""
Diagnostic: run the full detection pipeline on image.png and save annotated output.
Usage:  python test_image.py
Shows what each stage of the enhancement does and what SAM3 returns at each stage.
"""
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image as PILImage
from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "/home/paras/sam3/sam3.pt"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.png")

# ── Labels to test (edit these to match what you actually use) ─────────────
LABELS = ["person", "surgical cap", "gloves", "hair net", "hard hat"]
CONF   = 0.20   # deliberately low for diagnosis

# ── Load image ────────────────────────────────────────────────────────────────
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("ERROR: could not read image.png"); sys.exit(1)

h, w = img.shape[:2]
gray_mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
print(f"Image: {w}×{h}  mean-brightness={gray_mean:.1f}")

# ── Enhancement stages ────────────────────────────────────────────────────────
import math

def gamma(img, target=115.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = max(float(gray.mean()), 1.0)
    if mean >= target:
        return img
    g   = math.log(target / 255.0) / math.log(mean / 255.0)
    g   = float(np.clip(g, 0.25, 5.0))
    lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def tone_map(img):
    """Reinhard tone mapping — handles HDR (bright door + dark room)."""
    f    = img.astype(np.float32) / 255.0
    tm   = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0,
                                      light_adapt=0.8, color_adapt=0.0)
    out  = tm.process(f)
    return np.clip(out * 255, 0, 255).astype(np.uint8)

def clahe(img, clip=4.0, grid=4):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = c.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.bilateralFilter(img, d=7, sigmaColor=60, sigmaSpace=60)

def msrcr(img):
    sigmas = [15.0, 80.0, 200.0]
    f  = img.astype(np.float32) + 1.0
    lf = np.log(f)
    msr = np.zeros_like(f)
    for s in sigmas:
        msr += lf - np.log(cv2.GaussianBlur(f, (0,0), s) + 1.0)
    msr /= len(sigmas)
    s2  = f.sum(2, keepdims=True) + 1e-6
    cr  = np.log(125.0 * f / s2) * msr
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        ch = cr[:,:,c]; lo, hi = ch.min(), ch.max()
        if hi > lo:
            ch = (ch - lo)/(hi - lo)*255.0
        out[:,:,c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out

# Build stages
stages = {
    "0_original":      img,
    "1_gamma":         gamma(img),
    "2_tone_map":      tone_map(img),
    "3_denoise_gamma": gamma(denoise(img)),
    "4_msrcr":         msrcr(img),
    "5_msrcr_gamma":   gamma(msrcr(img)),
    "6_full_pipeline": clahe(gamma(tone_map(denoise(img))), clip=5.0, grid=4),
}

# Save all stages
out_dir = os.path.dirname(IMAGE_PATH)
for name, out_img in stages.items():
    path = os.path.join(out_dir, f"dbg_{name}.jpg")
    cv2.imwrite(path, out_img)
    m = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY).mean()
    print(f"  saved {path}  mean={m:.1f}")

# ── Load SAM3 ─────────────────────────────────────────────────────────────────
print(f"\nLoading SAM3 from {MODEL_PATH} ...")
overrides = dict(conf=CONF, task="segment", mode="predict",
                 model=MODEL_PATH, half=True, imgsz=1280, verbose=False)
predictor = SAM3SemanticPredictor(overrides=overrides)
print("Model loaded.\n")

# ── Run inference on each stage ───────────────────────────────────────────────
best_stage = None
best_count = 0

for name, enh in stages.items():
    pil = PILImage.fromarray(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
    predictor.set_image(pil)
    results = predictor(text=LABELS)
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        n     = len(boxes)
        confs = boxes.conf.cpu().numpy()
        cls   = boxes.cls.cpu().numpy().astype(int)
        names = [LABELS[c] if c < len(LABELS) else f"cls{c}" for c in cls]
        print(f"[{name}]  {n} detections:")
        for lbl, conf in zip(names, confs):
            print(f"    {lbl:20s}  conf={conf:.3f}")
        # Save annotated
        annotated = enh.copy()
        for box, lbl, conf in zip(boxes.xyxy.cpu().numpy(), names, confs):
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,217,126), 2)
            cv2.putText(annotated, f"{lbl} {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,217,126), 1)
        cv2.imwrite(os.path.join(out_dir, f"dbg_{name}_ann.jpg"), annotated)
        if n > best_count:
            best_count = n; best_stage = name
    else:
        print(f"[{name}]  0 detections")

print(f"\nBest stage: {best_stage} ({best_count} detections)")
print(f"\nAll debug images saved to {out_dir}")
print("Open dbg_*.jpg files to visually compare enhancement stages.")
