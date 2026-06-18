#!/usr/bin/env python3
"""
visualize_synthetic_data.py — show example SyMBac_2 synthetic movies.

Selects one sparse, one medium, and one dense movie from synthetic_data/,
then renders 3 evenly-spaced frames per movie: brightfield (left) and
ground-truth labelled mask as a coloured overlay with division arrows (right).

Usage (from symbac_training/):
    python plots/visualize_synthetic_data.py
    python plots/visualize_synthetic_data.py --data_dir synthetic_data/ --output figures/synthetic_examples.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

# ── colour palette (matches LaTeX document) ───────────────────────────────────
C_TEAL   = (0,    76/255, 76/255)
C_TEAL2  = (0,   128/255, 128/255)
C_PURPLE = (85/255, 0,    75/255)
C_GOLD   = "#e8c531"


def load_movie(movie_dir):
    masks  = np.load(os.path.join(movie_dir, "masks.npz"))["data"]
    images = np.load(os.path.join(movie_dir, "images.npz"))["data"]
    with open(os.path.join(movie_dir, "lineage.json")) as f:
        lineage = json.load(f)
    return images, masks, lineage


def median_cells(masks):
    return float(np.median([m.max() for m in masks]))


def pick_movies(data_dir):
    """Return (sparse, medium, dense) movie paths by median cell count."""
    dirs = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("movie_")
    ])
    scored = []
    for d in dirs:
        try:
            masks = np.load(os.path.join(d, "masks.npz"))["data"]
            scored.append((median_cells(masks), d))
        except Exception:
            pass
    scored.sort()
    n = len(scored)
    return [scored[n // 8][1], scored[n // 2][1], scored[7 * n // 8][1]]


def coloured_overlay(image, mask_frame):
    """Return (rgb_bg, rgba_overlay) ready for sequential imshow."""
    img_n = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb_bg = np.stack([img_n, img_n, img_n], axis=-1)

    labels = np.unique(mask_frame)
    labels = labels[labels > 0]
    rng    = np.random.default_rng(seed=0)
    n      = int(labels.max()) + 1
    hues   = rng.uniform(0.42, 0.58, n)     # teal family
    sats   = rng.uniform(0.5,  0.9,  n)
    vals   = rng.uniform(0.6,  1.0,  n)

    rgba = np.zeros((*mask_frame.shape, 4))
    for lab in labels:
        rgb = hsv_to_rgb([hues[lab], sats[lab], vals[lab]])
        rgba[mask_frame == lab, :3] = rgb
        rgba[mask_frame == lab,  3] = 0.45

    bounds = find_boundaries(mask_frame, mode="outer")
    rgba[bounds] = [1, 1, 1, 0.85]
    return rgb_bg, rgba


def draw_division_arrows(ax, mask_t, mask_t1, lineage, t):
    """Draw yellow arrows: parent centroid → each daughter centroid."""
    t1_key = str(t + 1)
    if t1_key not in lineage:
        return
    parent_to_daughters = {}
    for d_str, p in lineage[t1_key].items():
        d = int(d_str)
        if d != p:
            parent_to_daughters.setdefault(p, []).append(d)
    if not parent_to_daughters:
        return

    props_t  = {r.label: r.centroid for r in regionprops(mask_t)}
    props_t1 = {r.label: r.centroid for r in regionprops(mask_t1)}

    for parent, daughters in parent_to_daughters.items():
        if parent not in props_t:
            continue
        py, px = props_t[parent]
        for d in daughters:
            if d not in props_t1:
                continue
            dy, dx = props_t1[d]
            ax.annotate("",
                        xy=(dx, dy), xytext=(px, py),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=C_GOLD,
                            lw=1.4,
                            mutation_scale=12,
                        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="synthetic_data")
    ap.add_argument("--output",   default="figures/synthetic_examples.png")
    ap.add_argument("--dpi",      type=int, default=150)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    picks  = pick_movies(args.data_dir)
    labels = ["Sparse", "Medium", "Dense"]

    N_MOVIES = 3
    N_FRAMES = 3
    fig, axes = plt.subplots(
        N_MOVIES, N_FRAMES * 2,
        figsize=(N_FRAMES * 2 * 2.6, N_MOVIES * 2.8),
        gridspec_kw={"wspace": 0.04, "hspace": 0.28},
    )
    fig.patch.set_facecolor("#f5f7fa")

    for row, (mdir, density_label) in enumerate(zip(picks, labels)):
        images, masks, lineage = load_movie(mdir)
        T        = images.shape[0]
        idxs     = [T // 6, T // 2, 5 * T // 6]
        med_c    = int(median_cells(masks))
        mname    = os.path.basename(mdir)

        for col, t in enumerate(idxs):
            ax_bf  = axes[row, col * 2]
            ax_seg = axes[row, col * 2 + 1]

            img_n = (images[t] - images[t].min()) / (images[t].max() - images[t].min() + 1e-8)
            ax_bf.imshow(img_n, cmap="gray", interpolation="nearest")
            ax_bf.axis("off")

            rgb_bg, rgba = coloured_overlay(images[t], masks[t])
            ax_seg.imshow(rgb_bg, interpolation="nearest")
            ax_seg.imshow(rgba,   interpolation="nearest")
            if t + 1 < T:
                draw_division_arrows(ax_seg, masks[t], masks[t + 1], lineage, t)
            ax_seg.axis("off")

            if row == 0:
                ax_bf.set_title(f"t = {t}  |  Brightfield",
                                fontsize=7, color=C_TEAL, pad=3, fontweight="bold")
                ax_seg.set_title(f"t = {t}  |  Masks + divisions",
                                 fontsize=7, color=C_TEAL, pad=3, fontweight="bold")

        # Left-side row annotation
        axes[row, 0].text(
            -0.18, 0.5,
            f"{density_label}\n{mname}\n(median {med_c} cells/frame)",
            transform=axes[row, 0].transAxes,
            fontsize=7, color="#333333", va="center", ha="right",
            rotation=90,
        )

    arrow_patch = mpatches.Patch(color=C_GOLD, label="Division event (arrow: parent→daughters)")
    fig.legend(handles=[arrow_patch], loc="lower center", fontsize=8,
               framealpha=0.9, ncol=1, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("SyMBac² — Synthetic Training Data Examples",
                 fontsize=12, fontweight="bold", color=C_TEAL, y=1.02)

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
