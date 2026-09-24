#!/usr/bin/env python3
"""Fold calibrated SyMBac synthetic crops (synth_to_coco.py --format mtb_crops
output) into a species' real train/val COCO set as TRAIN-ONLY augmentation.
Species-agnostic: works for any species whose real dataset is a single merged
COCO json + a flat images/{train,val} dir (E. coli, M. abscessus -- the
bacdetr_rod_nafnet_only_perspecies-derived per-species slices), as opposed to
TB's per-crop-json layout (build_annotations_mtb.py) or PA's own multi-source
merge (build_pa_standardized_dataset.py), which already have their own
dedicated synth-aug wiring.

Writes a standalone output_dir (real val copied through byte-identical, never
touched by synthetic data) rather than mutating the shared pooled dataset
in place.

Usage:
  python build_rod_synthaug.py \\
      --real_train_ann data/bacdetr_rod_nafnet_only_perspecies/annotations/instances_train_3class_ecoli_ibsonly.json \\
      --real_val_ann   data/bacdetr_rod_nafnet_only_perspecies/annotations/instances_val_3class_ecoli_ibsonly.json \\
      --real_images_dir data/bacdetr_rod_nafnet_only_perspecies/images \\
      --extra_crop_dir  data/synth_ecoli_coco_synth1x_subset \\
      --output_dir      data/ecoli_synthaug_1x
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_train_ann", required=True)
    parser.add_argument("--real_val_ann", required=True)
    parser.add_argument("--real_images_dir", required=True,
                         help="Dir containing train/ and val/ subdirs of real crop tiffs")
    parser.add_argument("--extra_crop_dir", required=True,
                         help="synth_to_coco.py --format mtb_crops output -- folded into TRAIN only")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    real_images_dir = Path(args.real_images_dir)
    extra_dir = Path(args.extra_crop_dir)
    out = Path(args.output_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    with open(args.real_train_ann) as f:
        train_coco = json.load(f)
    with open(args.real_val_ann) as f:
        val_coco = json.load(f)

    categories = train_coco["categories"]
    single_cell_id = next(c["id"] for c in categories if c["name"] == "single-cell")

    # Copy through the real images this species' train/val jsons actually reference.
    for im in train_coco["images"]:
        shutil.copy2(real_images_dir / "train" / im["file_name"], out / "images" / "train" / im["file_name"])
    for im in val_coco["images"]:
        shutil.copy2(real_images_dir / "val" / im["file_name"], out / "images" / "val" / im["file_name"])

    img_id = max(im["id"] for im in train_coco["images"]) + 1
    ann_id = max(a["id"] for a in train_coco["annotations"]) + 1

    n_synthetic = 0
    for jf in sorted(f for f in extra_dir.iterdir() if f.suffix == ".json" and "_coco" in f.name):
        stem = jf.stem.replace("_coco", "")
        src_img = extra_dir / f"{stem}.tiff"
        if not src_img.exists():
            continue
        with open(jf) as f:
            synth = json.load(f)
        if not synth["annotations"]:
            continue

        shutil.copy2(src_img, out / "images" / "train" / f"{stem}.tiff")
        src_meta = synth["images"][0]
        train_coco["images"].append({"id": img_id, "file_name": f"{stem}.tiff",
                                      "width": src_meta["width"], "height": src_meta["height"]})
        for a in synth["annotations"]:
            train_coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": single_cell_id,   # synthetic is single-cell only, regardless of its own id=1
                "segmentation": a["segmentation"], "bbox": a["bbox"],
                "area": a["area"], "iscrowd": 0,
                "attributes": {"source": "synthetic_symbac"},
            })
            ann_id += 1
        img_id += 1
        n_synthetic += 1

    with open(out / "annotations" / "instances_train.json", "w") as f:
        json.dump(train_coco, f)
    with open(out / "annotations" / "instances_val.json", "w") as f:
        json.dump(val_coco, f)

    split_doc = {
        "real_train_ann": str(args.real_train_ann),
        "real_val_ann": str(args.real_val_ann),
        "n_real_train": len(train_coco["images"]) - n_synthetic,
        "n_synthetic_train": n_synthetic,
        "n_val": len(val_coco["images"]),
        "synthetic_source_dir": str(args.extra_crop_dir),
        "split": "synthetic is train only -- val is a byte-identical copy of real_val_ann",
    }
    with open(out / "annotations" / "split.json", "w") as f:
        json.dump(split_doc, f, indent=2)

    print(f"train: {len(train_coco['images'])} images "
          f"({split_doc['n_real_train']} real + {n_synthetic} synthetic), "
          f"{len(train_coco['annotations'])} anns")
    print(f"val:   {len(val_coco['images'])} images (real only, untouched), "
          f"{len(val_coco['annotations'])} anns")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
