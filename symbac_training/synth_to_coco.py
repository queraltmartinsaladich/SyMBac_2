#!/usr/bin/env python3
"""Convert calibrated SyMBac_2 synthetic movies (images.npz + masks.npz) into
detector-training-shaped COCO annotations, all labeled single-cell (the only
class SyMBac's rasterizer can produce -- it has no clump/joint-cell/debris
model).

Output layout mirrors the two real-data conventions this project already
uses, so results can be handed straight to the existing dataset builders as
train-only augmentation:

  --format mtb_crops : one <stem>_coco.json + <stem>.tiff per frame, matching
                        data/bacdetr_mtb/annotation_crops/ -- feed the output
                        dir to build_annotations_mtb.py's --extra_crop_dir.
  --format pa_records : a single records.json (list of {file_name, anns,
                         movie}) + images/*.tiff, matching the record shape
                         build_pa_standardized_dataset.py's load_ibs() /
                         load_e013_tiled() already produce -- consumed by
                         that script's load_synthetic().

Usage:
  python synth_to_coco.py --movies_dir <symbac output dir> --output_dir <out> \\
      --format mtb_crops --category_id 1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

SINGLE_CELL_CATEGORIES_MTB = [
    {"id": 1, "name": "single-cell", "supercategory": "cell"},
    {"id": 2, "name": "clump",       "supercategory": "cell"},
    {"id": 3, "name": "joint-cell",  "supercategory": "cell"},
]


def _mask_to_polygons(mask: np.ndarray):
    """Instance-labeled mask -> list of (bbox, area, polygon) per label,
    mirroring prep_training/preprocess_training_data_mtb.py's contour approach."""
    out = []
    for label in np.unique(mask):
        if label == 0:
            continue
        obj_mask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        area = float(cv2.contourArea(contour))
        if area < 4:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        poly = contour.reshape(-1).astype(float).tolist()
        out.append({"bbox": [float(x), float(y), float(w), float(h)],
                     "area": area, "segmentation": [poly]})
    return out


def _iter_frames(movies_dir):
    movies_dir = Path(movies_dir)
    for movie_dir in sorted(movies_dir.glob("movie_*")):
        images_path = movie_dir / "images.npz"
        masks_path = movie_dir / "masks.npz"
        if not (images_path.exists() and masks_path.exists()):
            continue
        images = np.load(images_path)["data"]
        masks = np.load(masks_path)["data"]
        for t in range(images.shape[0]):
            yield movie_dir.name, t, images[t], masks[t]


def write_mtb_crops(movies_dir, output_dir, category_id):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for movie_name, t, img, mask in _iter_frames(movies_dir):
        objs = _mask_to_polygons(mask)
        if not objs:
            continue
        stem = f"synth_{movie_name}_t{t:03d}"
        H, W = img.shape
        tifffile.imwrite(output_dir / f"{stem}.tiff", img.astype(np.float32))

        annotations = []
        for i, obj in enumerate(objs, start=1):
            annotations.append({
                "id": i, "image_id": 1, "category_id": category_id,
                "segmentation": obj["segmentation"], "bbox": obj["bbox"],
                "area": obj["area"], "iscrowd": 0,
                "attributes": {"source": "synthetic_symbac", "movie": movie_name, "frame": t},
            })
        coco = {
            "images": [{"id": 1, "file_name": f"{stem}.tiff", "height": H, "width": W}],
            "annotations": annotations,
            "categories": SINGLE_CELL_CATEGORIES_MTB,
        }
        with open(output_dir / f"{stem}_coco.json", "w") as f:
            json.dump(coco, f)
        n_written += 1
    print(f"Wrote {n_written} synthetic crops -> {output_dir}")


def write_pa_records(movies_dir, output_dir, category_id):
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records_meta = []
    for movie_name, t, img, mask in _iter_frames(movies_dir):
        objs = _mask_to_polygons(mask)
        if not objs:
            continue
        file_name = f"synth_{movie_name}_t{t:03d}.tiff"
        tifffile.imwrite(img_dir / file_name, img.astype(np.float32))
        anns = [(category_id, obj["segmentation"]) for obj in objs]
        records_meta.append({
            "file_name": file_name,
            "anns": anns,
            "movie": f"synthetic_{movie_name}",
            "source": "synthetic_symbac",
        })
    with open(output_dir / "records.json", "w") as f:
        json.dump(records_meta, f)
    print(f"Wrote {len(records_meta)} synthetic records -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies_dir", required=True, help="SyMBac generate_dataset.py --output_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["mtb_crops", "pa_records"], required=True)
    parser.add_argument("--category_id", type=int, default=1,
                         help="COCO category_id for single-cell (1 in both mtb 3-class and rod_3class schemes)")
    args = parser.parse_args()

    if args.format == "mtb_crops":
        write_mtb_crops(args.movies_dir, args.output_dir, args.category_id)
    else:
        write_pa_records(args.movies_dir, args.output_dir, args.category_id)


if __name__ == "__main__":
    main()
