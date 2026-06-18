#!/usr/bin/env python3
"""
plot_feature_distributions.py — KDE overlay plots for triplet and pair features.

Loads the training HDF5 split and shows the per-feature density for the
positive class (true assignment / true division) vs the negative class.
Useful for assessing feature discriminability and the effect of hard-negative
mining (run before and after rebuilding the dataset to compare).

Usage (from symbac_training/):
    python plots/plot_feature_distributions.py
    python plots/plot_feature_distributions.py --dataset_dir dataset/ --output_dir figures/
"""
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction import PAIR_FEATURE_NAMES, TRIPLET_FEATURE_NAMES

C_POS  = (0,    128/255, 128/255)   # teal — positive class
C_NEG  = (85/255, 0,     75/255)    # purple — negative class
C_FILL_POS = (*C_POS,  0.25)
C_FILL_NEG = (*C_NEG,  0.25)


def load_split(path, key_feats, key_labels):
    with h5py.File(path, "r") as f:
        feats  = f[key_feats][:]
        labels = f[key_labels][:].astype(int)
    return feats, labels


def kde_plot(ax, values_pos, values_neg, feature_name, clip_percentile=99):
    """One KDE density plot: positive (teal) vs negative (purple)."""
    lo = min(np.percentile(values_pos, 1), np.percentile(values_neg, 1))
    hi = max(np.percentile(values_pos, clip_percentile),
             np.percentile(values_neg, clip_percentile))
    xs = np.linspace(lo, hi, 300)

    for vals, color, fill_color, label in [
        (values_pos, C_POS, C_FILL_POS, "Positive"),
        (values_neg, C_NEG, C_FILL_NEG, "Negative"),
    ]:
        try:
            kde  = gaussian_kde(vals, bw_method="silverman")
            dens = kde(xs)
            ax.plot(xs, dens, color=color, lw=1.6, label=label)
            ax.fill_between(xs, dens, color=fill_color)
        except Exception:
            ax.hist(vals, bins=40, density=True, alpha=0.4, color=color, label=label)

    ax.set_title(feature_name.replace("_", " "), fontsize=7.5,
                 color=(0, 76/255, 76/255), fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)


def make_figure(feats, labels, feature_names, title, output_path, dpi):
    n = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.0, nrows * 2.4),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    fig.patch.set_facecolor("#f5f7fa")
    axes_flat = axes.flat

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    # Subsample negatives for speed if very unbalanced
    if n_neg > 10 * n_pos:
        rng      = np.random.default_rng(seed=42)
        neg_idx  = rng.choice(np.where(neg_mask)[0], size=min(n_neg, 10 * n_pos), replace=False)
        neg_mask = np.zeros(len(labels), dtype=bool)
        neg_mask[neg_idx] = True

    for i, fname in enumerate(feature_names):
        ax = axes_flat[i]
        kde_plot(ax, feats[pos_mask, i], feats[neg_mask, i], fname)
        if i == 0:
            ax.legend(fontsize=6.5, framealpha=0.8, loc="best")

    for j in range(i + 1, nrows * ncols):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{title}\n"
        f"({pos_mask.sum():,} positive  |  {neg_mask.sum():,} negative shown)",
        fontsize=10, fontweight="bold", color=(0, 76/255, 76/255), y=1.01,
    )

    if "triplet" in output_path.lower() or "division" in title.lower():
        caption = (
            "Kernel density estimates (KDE) for the 8 triplet features used by the DivisionClassifier. "
            "Teal = positive class (confirmed division events); purple = negative class (non-division triplets). "
            "Negatives are subsampled to 10× the positive count for visual clarity. "
            "Features with strong teal/purple separation carry the most discriminative information for the MLP."
        )
    else:
        caption = (
            "Kernel density estimates (KDE) for the 10 pair features used by the AssignmentScorer. "
            "Teal = correct cell–cell links across frames; purple = incorrect links. "
            "Features such as centroid_dist_norm and iou show strong separation and are the dominant "
            "predictors; the MLP combines all 10 to compute a learned assignment cost."
        )
    fig.text(0.02, -0.01, caption, fontsize=7, color="#444444", style="italic",
             va="top", ha="left", transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--output_dir",  default="figures")
    ap.add_argument("--split",       default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--dpi",         type=int, default=300)
    args = ap.parse_args()

    h5_path = os.path.join(args.dataset_dir, f"{args.split}_assignments.h5")
    print(f"Loading {h5_path} …")

    # ── Triplet features (DivisionClassifier) ─────────────────────────────────
    trip_feats, trip_labels = load_split(h5_path, "triplet_features", "triplet_labels")
    print(f"  Triplet:  {trip_feats.shape[0]:,} samples  "
          f"({(trip_labels==1).sum():,} pos / {(trip_labels==0).sum():,} neg)")
    make_figure(
        trip_feats, trip_labels, TRIPLET_FEATURE_NAMES,
        "Triplet features — DivisionClassifier",
        os.path.join(args.output_dir, "feature_distributions_triplets.png"),
        args.dpi,
    )

    # ── Pair features (AssignmentScorer) ──────────────────────────────────────
    pair_feats, pair_labels = load_split(h5_path, "pair_features", "pair_labels")
    print(f"  Pair:     {pair_feats.shape[0]:,} samples  "
          f"({(pair_labels==1).sum():,} pos / {(pair_labels==0).sum():,} neg)")
    make_figure(
        pair_feats, pair_labels, PAIR_FEATURE_NAMES,
        "Pair features — AssignmentScorer",
        os.path.join(args.output_dir, "feature_distributions_pairs.png"),
        args.dpi,
    )


if __name__ == "__main__":
    main()
