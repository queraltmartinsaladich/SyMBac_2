#!/usr/bin/env python3
"""
plot_hard_negative_mining.py — illustrate the hard-negative mining strategy.

Picks a dense test movie, finds a division event, and shows on a single frame:
  - Parent cell (red outline)
  - True daughters (green outline)
  - Old strategy: 3 random non-daughter cells sampled from anywhere (blue)
  - New strategy: up to 8 non-daughter cells sampled within max_dist_norm (orange)

The side-by-side comparison makes clear why random negatives are too easy:
they are almost always far from the parent in dense movies.

Usage (from symbac_training/):
    python plots/plot_hard_negative_mining.py
    python plots/plot_hard_negative_mining.py --movie_dir synthetic_data/movie_018 --output figures/hard_negative_mining.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

C_TEAL   = (0,    76/255,  76/255)
C_TEAL2  = (0,   128/255, 128/255)
C_PURPLE = (85/255, 0,     75/255)
BG       = "#f5f7fa"

C_PARENT   = "#e53935"   # red
C_DAUGHTER = "#43a047"   # green
C_OLD_NEG  = "#1e88e5"   # blue
C_NEW_NEG  = "#fb8c00"   # orange
C_PROXIMITY = "#fb8c00"


def load_movie(movie_dir):
    masks = np.load(os.path.join(movie_dir, "masks.npz"))["data"]
    with open(os.path.join(movie_dir, "lineage.json")) as f:
        lineage = json.load(f)
    return masks, lineage


def find_dense_division(masks, lineage, max_dist_norm=5.0):
    """Return (t, parent_label, daughter_labels) from the densest eligible frame."""
    best = None
    best_score = -1

    for t_str, cells in lineage.items():
        t = int(t_str)
        if t == 0 or t >= len(masks):
            continue
        t_prev = t - 1
        parent_to_daughters = {}
        for d_str, p in cells.items():
            d = int(d_str)
            if d != p:
                parent_to_daughters.setdefault(p, []).append(d)
        if not parent_to_daughters:
            continue

        n_cells_prev = masks[t_prev].max()
        for parent, daughters in parent_to_daughters.items():
            if len(daughters) < 2:
                continue
            # Score = cells in previous frame (prefer dense)
            score = n_cells_prev
            if score > best_score:
                best_score = score
                best = (t_prev, parent, daughters[:2])

    return best


def compute_median_major(masks_frame):
    rps = regionprops(masks_frame)
    if not rps:
        return 1.0
    return float(np.median([r.major_axis_length for r in rps if r.major_axis_length > 0]) or 1.0)


def get_centroid(masks_frame, label):
    rps = {r.label: r for r in regionprops(masks_frame)}
    if label not in rps:
        return None
    return rps[label].centroid


def find_nearby(masks_frame, parent_label, daughter_labels, max_dist_norm):
    """Return (all_non_daughters, nearby_non_daughters) as label lists."""
    rps = {r.label: r for r in regionprops(masks_frame)}
    if parent_label not in rps:
        return [], []

    med_major = compute_median_major(masks_frame)
    py, px    = rps[parent_label].centroid
    daughters_set = set(daughter_labels)

    non_daughters = [l for l in rps if l not in daughters_set and l != parent_label]
    nearby = [
        l for l in non_daughters
        if (((rps[l].centroid[0] - py) ** 2 +
             (rps[l].centroid[1] - px) ** 2) ** 0.5
            / med_major) <= max_dist_norm
    ]
    return non_daughters, nearby


def draw_frame(ax, masks_frame, masks_next, parent_label, daughter_labels,
               highlight_labels, highlight_color, highlight_name,
               max_dist_norm, title):
    """Render one panel: gray mask + highlighted cells + proximity circle."""
    rps = {r.label: r for r in regionprops(masks_frame)}
    rps_next = {r.label: r for r in regionprops(masks_next) if daughter_labels and r.label in daughter_labels}

    # Gray background: all cell outlines
    bounds_all = find_boundaries(masks_frame, mode="outer")
    gray = np.zeros((*masks_frame.shape, 3))
    gray[bounds_all] = [0.7, 0.7, 0.7]

    # RGBA overlay for highlighted cells
    rgba = np.zeros((*masks_frame.shape, 4))

    def fill(label, frame_masks, color_hex, alpha=0.45):
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color_hex)
        m   = frame_masks == label
        rgba[m, :3] = rgb
        rgba[m,  3] = alpha
        bounds = find_boundaries(m.astype(np.uint8), mode="outer")
        rgba[bounds, :3] = rgb
        rgba[bounds,  3] = 0.95

    fill(parent_label, masks_frame, C_PARENT, alpha=0.50)
    for d in daughter_labels:
        fill(d, masks_next if d in rps_next else masks_frame, C_DAUGHTER, alpha=0.50)
    for l in highlight_labels:
        fill(l, masks_frame, highlight_color, alpha=0.35)

    ax.imshow(gray, interpolation="nearest")
    ax.imshow(rgba, interpolation="nearest")

    # Proximity circle around parent
    if parent_label in rps:
        py, px = rps[parent_label].centroid
        med_major = compute_median_major(masks_frame)
        radius    = max_dist_norm * med_major
        circle    = plt.Circle((px, py), radius,
                                color=C_PROXIMITY, fill=False, lw=1.4, ls="--", alpha=0.7)
        ax.add_patch(circle)
        ax.plot(px, py, "+", color=C_PARENT, ms=6, mew=1.5)

    ax.set_title(title, fontsize=9, color=C_TEAL, fontweight="bold", pad=4)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      default="synthetic_data")
    ap.add_argument("--movie_dir",     default=None,
                    help="Specific movie dir; auto-selects densest if omitted")
    ap.add_argument("--max_dist_norm", type=float, default=5.0)
    ap.add_argument("--output",        default="figures/hard_negative_mining.png")
    ap.add_argument("--dpi",           type=int, default=150)
    args = ap.parse_args()

    if args.movie_dir:
        movie_dir = args.movie_dir
    else:
        # Pick densest movie from data_dir
        dirs = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("movie_")
        ])
        scored = []
        for d in dirs:
            try:
                masks = np.load(os.path.join(d, "masks.npz"))["data"]
                scored.append((float(np.median([m.max() for m in masks])), d))
            except Exception:
                pass
        scored.sort(reverse=True)
        movie_dir = scored[0][1]

    print(f"Using movie: {movie_dir}")
    masks, lineage = load_movie(movie_dir)

    event = find_dense_division(masks, lineage, args.max_dist_norm)
    if event is None:
        print("ERROR: no division event found in this movie.")
        return

    t, parent_label, daughter_labels = event
    print(f"  Division at frame t={t}, parent={parent_label}, daughters={daughter_labels}")
    print(f"  Cells in frame: {masks[t].max()}")

    non_daughters, nearby = find_nearby(masks[t], parent_label, daughter_labels,
                                        args.max_dist_norm)
    rng = np.random.default_rng(seed=42)

    # Old: 3 random from all non-daughters
    old_neg = list(rng.choice(non_daughters,
                               size=min(3, len(non_daughters)),
                               replace=False).astype(int))
    # New: up to 8 from nearby (proximity-filtered)
    new_neg = list(rng.choice(nearby,
                               size=min(8, len(nearby)),
                               replace=False).astype(int)) if len(nearby) >= 2 else old_neg

    print(f"  Non-daughters total: {len(non_daughters)}")
    print(f"  Nearby (within {args.max_dist_norm}×major): {len(nearby)}")
    print(f"  Old negatives (random): {old_neg}")
    print(f"  New negatives (proximity): {new_neg}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                              gridspec_kw={"wspace": 0.06})
    fig.patch.set_facecolor(BG)

    masks_next = masks[t + 1] if t + 1 < len(masks) else masks[t]

    draw_frame(axes[0], masks[t], masks_next,
               parent_label, daughter_labels, old_neg, C_OLD_NEG,
               "Old negatives (random)",
               args.max_dist_norm,
               f"v1 — Random negatives  ({len(old_neg)} sampled)\n"
               f"Frame t={t}  |  {masks[t].max()} cells")

    draw_frame(axes[1], masks[t], masks_next,
               parent_label, daughter_labels, new_neg, C_NEW_NEG,
               "New negatives (proximity-filtered)",
               args.max_dist_norm,
               f"v2 — Hard negatives within {args.max_dist_norm}× major axis  ({len(new_neg)} sampled)\n"
               f"Dashed circle = proximity threshold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_PARENT,   label="Parent cell"),
        mpatches.Patch(color=C_DAUGHTER, label="True daughters (t+1)"),
        mpatches.Patch(color=C_OLD_NEG,  label="Old negatives: random from all cells"),
        mpatches.Patch(color=C_NEW_NEG,  label="New negatives: nearby non-daughters"),
        mpatches.Patch(color=C_PROXIMITY, fill=False, label=f"Proximity threshold ({args.max_dist_norm}× median major axis)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Hard-Negative Mining — Sampling Strategy Comparison\n"
        f"({os.path.basename(movie_dir)}, {masks[t].max()} cells/frame)",
        fontsize=11, fontweight="bold", color=C_TEAL, y=1.01,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
