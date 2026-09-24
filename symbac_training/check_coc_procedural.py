#!/usr/bin/env python3
"""One-off validation script for generate_coc_dataset.py output: renders
t=0/9/19 frames to PNG and reports cell counts. Meant to be run via SLURM
(slurm/check_coc_procedural.sh), not directly on the login node.

Usage:
  python check_coc_procedural.py --pool_dir data/synth_coc_v2 --output_dir <png output>
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--movie", default="movie_000")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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
        n_cells = int(np.count_nonzero(np.unique(m)))
        print(f"t={t}: {n_cells} cells")


if __name__ == "__main__":
    main()
