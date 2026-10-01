#!/usr/bin/env python3
"""One-off validation script for generate_coc_dataset.py output: renders
early/mid/late frames to PNG and measures synthetic geometry/shape stats
against the real calibration profile -- same methodology as
check_rod_procedural.py, for an apples-to-apples 2% match bar across species.
Meant to be run via SLURM (slurm/check_coc_procedural.sh), not directly on
the login node.

Usage:
  python check_coc_procedural.py --pool_dir data/synth_coc_v2 --output_dir <png output>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.measure import regionprops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--movie", default="movie_000")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profile_path = Path(__file__).parent / "calibration_profiles" / "coc.json"
    profile = json.loads(profile_path.read_text())

    movie_dir = Path(args.pool_dir) / args.movie
    images = np.load(movie_dir / "images.npz")["data"]
    masks = np.load(movie_dir / "masks.npz")["data"]
    n_frames = images.shape[0]
    frame_idxs = sorted(set([0, n_frames // 2, n_frames - 1]))

    rng = np.random.default_rng(0)
    for t in frame_idxs:
        img8 = (np.clip(images[t], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img8).save(out / f"t{t:02d}_raw.png")
        m = masks[t]
        colors = rng.integers(50, 255, size=(int(m.max()) + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        blend = (0.5 * np.stack([img8] * 3, axis=-1) + 0.5 * colors[m]).astype(np.uint8)
        Image.fromarray(blend).save(out / f"t{t:02d}_overlay.png")

    lengths, widths, eccs, sols, areas = [], [], [], [], []
    cells_per_frame = []
    for t in range(n_frames):
        m = masks[t]
        labels = np.unique(m)
        cells_per_frame.append(int(len(labels) - 1))
        for lbl in labels:
            if lbl == 0:
                continue
            cm = (m == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if len(c) < 5:
                continue
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            lengths.append(max(w, h))
            widths.append(min(w, h))
            props = regionprops(cm)
            if props:
                eccs.append(props[0].eccentricity)
                sols.append(props[0].solidity)
                areas.append(props[0].area)

    lengths, widths = np.array(lengths), np.array(widths)
    eccs, sols, areas = np.array(eccs), np.array(sols), np.array(areas)

    report = {
        "species": "coc",
        "cells_per_frame": cells_per_frame,
        "synthetic": {
            "width_px_mean": float(widths.mean()), "width_px_std": float(widths.std()),
            "length_px_mean": float(lengths.mean()),
            "aspect_mean": float((lengths / widths).mean()),
            "eccentricity_mean": float(eccs.mean()) if eccs.size else None,
            "solidity_mean": float(sols.mean()) if sols.size else None,
            "area_px_mean": float(areas.mean()) if areas.size else None,
        },
        "real": {
            "width_px_mean": profile["geometry"]["width_px_mean"],
            "length_px_mean": profile["geometry"]["length_px_mean"],
            "aspect_mean": profile["geometry"]["aspect_ratio_mean"],
            "eccentricity_mean": profile["shape"]["eccentricity_mean"],
            "solidity_mean": profile["shape"]["solidity_mean"],
            "area_px_mean": profile["shape"]["area_px_mean"],
            "cells_per_crop_mean": profile["density"]["cells_per_crop_mean"],
        },
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
