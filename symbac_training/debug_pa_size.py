#!/usr/bin/env python3
"""Direct, from-scratch sanity check of real vs synthetic PA cell sizes,
fixed version: real cells measured from actual GT annotation polygons
(not noisy Otsu thresholding, which picked up background speckle as false
blobs), synthetic cells measured with cv2.minAreaRect (rotation-aware --
the first attempt used axis-aligned boundingRect, which for a rotated
capsule gives neither the true length nor width).
"""
import json
from pathlib import Path
import numpy as np
import cv2

ANN_PATH = str(Path.home() / "data/pa_standardized/annotations/instances_train_3class.json")
TARGET_FILE = "AMK_PAO1_O23_p05_t00_crop_r000_c000.tiff"
SYNTH_DIR = str(Path.home() / "data/synth_pa_procedural_v1/movie_000")

print("=== REAL (from GT annotation polygons) ===")
with open(ANN_PATH) as f:
    coco = json.load(f)
img_meta = next((im for im in coco["images"] if im["file_name"] == TARGET_FILE), None)
if img_meta is None:
    print(f"'{TARGET_FILE}' not found in {ANN_PATH} -- listing a few real filenames instead:")
    for im in coco["images"][:5]:
        print(" ", im["file_name"])
else:
    anns = [a for a in coco["annotations"] if a["image_id"] == img_meta["id"] and a["category_id"] == 1]
    print(f"n single-cell GT polygons: {len(anns)}")
    real_boxes = []
    for a in anns:
        pts = np.array(a["segmentation"][0], dtype=np.float32).reshape(-1, 2)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        real_boxes.append((max(w, h), min(w, h), a["area"]))
    real_boxes.sort(key=lambda x: -x[2])
    for w, h, ar in real_boxes[:15]:
        print(f"  length={w:.1f} width={h:.1f} area={ar:.0f}")
    rw = np.mean([b[0] for b in real_boxes])
    rh = np.mean([b[1] for b in real_boxes])
    print(f"real mean: length={rw:.2f} width={rh:.2f}")

print("\n=== SYNTHETIC (movie_000, t=19, minAreaRect) ===")
masks = np.load(f"{SYNTH_DIR}/masks.npz")["data"]
mask = masks[19]
synth_boxes = []
for lbl in np.unique(mask):
    if lbl == 0:
        continue
    m = (mask == lbl).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    synth_boxes.append((max(w, h), min(w, h), int(m.sum())))
synth_boxes.sort(key=lambda x: -x[2])
for w, h, ar in synth_boxes[:15]:
    print(f"  length={w:.1f} width={h:.1f} area={ar}")
sw = np.mean([b[0] for b in synth_boxes])
sh = np.mean([b[1] for b in synth_boxes])
print(f"synth mean: length={sw:.2f} width={sh:.2f}")

if img_meta is not None:
    print("\n=== RATIO ===")
    print(f"length ratio: {sw/rw:.2f}x   width ratio: {sh/rh:.2f}x")
