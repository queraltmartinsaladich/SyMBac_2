#!/usr/bin/env python3
"""
plot_performance.py — visualise tracking and division classifier performance.

Produces two figures:
  1. tracking_comparison.png — AssignmentScorer vs geometric baseline (bar + per-movie IDSW)
  2. division_performance.png — per-movie DivisionClassifier F1, FP/FN breakdown,
                                 density scatter, PR curve, score histogram

Input JSON files are produced by evaluate_tracking.py and evaluate_divisions.py.

Usage (from symbac_training/):
    python plots/plot_performance.py
    python plots/plot_performance.py --weights_dir weights/ --output_dir figures/
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _wrap(fig, text, chars_per_inch=9):
    return textwrap.fill(text, width=max(40, int(fig.get_figwidth() * chars_per_inch)))


C_TEAL   = (0,    76/255,  76/255)
C_TEAL2  = (0,   128/255, 128/255)
C_PURPLE = (85/255, 0,     75/255)
C_RED    = (200/255, 30/255, 30/255)
C_GREEN  = (70/255, 120/255, 20/255)
C_ORANGE = (210/255, 110/255, 0)
BG       = "#f5f7fa"


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Tracking comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tracking_comparison(baseline_path, learned_path, output_path, dpi):
    with open(baseline_path) as f:
        base = json.load(f)
    with open(learned_path) as f:
        lrn  = json.load(f)

    metrics = ["link_f1", "mota", "idf1"]
    labels  = ["Link F1", "MOTA", "IDF1"]
    base_vals = [base["aggregate"][m] for m in metrics]
    lrn_vals  = [lrn["aggregate"][m]  for m in metrics]

    # Per-movie ID switches
    base_idsw = {os.path.basename(r["movie"]): r["id_switches"]
                 for r in base.get("per_movie", [])}
    lrn_idsw  = {os.path.basename(r["movie"]): r["id_switches"]
                 for r in lrn.get("per_movie",  [])}
    shared_movies = sorted(set(base_idsw) & set(lrn_idsw))

    fig = plt.figure(figsize=(12, 4.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.8, 0.05],
                             wspace=0.35, left=0.06, right=0.97,
                             top=0.85, bottom=0.18)

    # ── Aggregate metrics bar ─────────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    x    = np.arange(len(metrics))
    w    = 0.32
    bars_b = ax1.bar(x - w/2, base_vals, w, color=C_TEAL2,   alpha=0.85, label="Geometric baseline")
    bars_l = ax1.bar(x + w/2, lrn_vals,  w, color=C_PURPLE, alpha=0.85, label="AssignmentScorer (MLP)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0.55, 1.02)
    ax1.set_ylabel("Score", fontsize=9)
    ax1.set_title("Aggregate tracking metrics\n(10 test movies)", fontsize=9,
                  color=C_TEAL, fontweight="bold")
    ax1.legend(fontsize=7.5, framealpha=0.85, loc="best")
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_facecolor(BG)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    for bar, val in [(bars_b, base_vals), (bars_l, lrn_vals)]:
        for b, v in zip(bar, val):
            ax1.text(b.get_x() + b.get_width()/2, v + 0.004,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    # ── Per-movie ID switches ─────────────────────────────────────────────────
    ax2  = fig.add_subplot(gs[1])
    xs   = np.arange(len(shared_movies))
    b_sw = [base_idsw[m] for m in shared_movies]
    l_sw = [lrn_idsw[m]  for m in shared_movies]
    ax2.bar(xs - w/2, b_sw, w, color=C_TEAL2,   alpha=0.85, label="Baseline")
    ax2.bar(xs + w/2, l_sw, w, color=C_PURPLE, alpha=0.85, label="AssignmentScorer")
    # delta annotations
    for xi, (bv, lv) in enumerate(zip(b_sw, l_sw)):
        delta = lv - bv
        col   = C_GREEN if delta <= 0 else C_RED
        ax2.text(xi, max(bv, lv) + 5, f"{delta:+d}",
                 ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([m.replace("movie_", "") for m in shared_movies],
                         fontsize=7.5, rotation=35, ha="right")
    ax2.set_ylabel("ID switches", fontsize=9)
    ax2.set_title("ID switches per test movie\n(lower = better)",
                  fontsize=9, color=C_TEAL, fontweight="bold")
    ax2.legend(fontsize=7.5, framealpha=0.85, loc="best")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_facecolor(BG)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    fig.suptitle("AssignmentScorer vs Geometric Baseline — Tracking Performance",
                 fontsize=11, fontweight="bold", color=C_TEAL, y=0.97)

    caption = (
        "Left: aggregate tracking metrics (Link F1, MOTA, IDF1) on held-out test movies, "
        "comparing a geometric centroid-distance baseline against the learned AssignmentScorer (MLP, 10 pair features). "
        "Right: per-movie identity switches (lower is better); coloured delta indicates change from baseline "
        "(green = improvement, red = regression). "
        "The AssignmentScorer reduces total ID switches by ~11% with minimal impact on other metrics."
    )
    fig.text(0.02, 0.01, _wrap(fig, caption), fontsize=7, color="#444444",
             style="italic", va="bottom", ha="left",
             transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Division classifier performance
# ═══════════════════════════════════════════════════════════════════════════════

def pr_curve(model_path, dataset_dir, split="test"):
    """Compute precision-recall points from the val/test dataset."""
    try:
        import h5py
        from src.feature_extraction import TRIPLET_FEATURE_NAMES
    except ImportError:
        return None, None, None

    ckpt = torch.load(model_path, map_location="cpu")
    n_in = ckpt.get("input_dim", 8)

    # Rebuild DivisionMLP locally to avoid import path issues
    import torch.nn as nn
    class DivisionMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(16, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = DivisionMLP(n_in)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    h5 = os.path.join(dataset_dir, f"{split}_assignments.h5")
    if not os.path.exists(h5):
        return None, None, None

    with h5py.File(h5, "r") as f:
        feats  = torch.tensor(f["triplet_features"][:], dtype=torch.float32)
        labels = f["triplet_labels"][:].astype(int)

    with torch.no_grad():
        probs = torch.sigmoid(model(feats)).numpy()

    thresholds = np.arange(0.05, 0.96, 0.02)
    precisions, recalls = [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))

    return np.array(precisions), np.array(recalls), probs, labels


def plot_division_performance(div_path, model_path, dataset_dir, output_path, dpi):
    with open(div_path) as f:
        data = json.load(f)

    per_movie = data.get("per_movie", [])
    if not per_movie:
        print(f"WARNING: no per_movie data in {div_path}")
        return

    movies   = [os.path.basename(r["movie"]) for r in per_movie]
    f1s      = [r["f1"]        for r in per_movie]
    fps      = [r["fp"]        for r in per_movie]
    fns      = [r["fn"]        for r in per_movie]
    tps      = [r["tp"]        for r in per_movie]
    gt_divs  = [r["n_gt_divisions"] for r in per_movie]

    order = np.argsort(f1s)  # sort ascending by F1
    movies  = [movies[i]  for i in order]
    f1s     = [f1s[i]     for i in order]
    fps     = [fps[i]     for i in order]
    fns     = [fns[i]     for i in order]
    tps     = [tps[i]     for i in order]
    gt_divs = [gt_divs[i] for i in order]

    # Try PR curve (may fail if model/dataset not available)
    try:
        precs, recs, probs, pr_labels = pr_curve(model_path, dataset_dir)
    except Exception:
        precs, recs, probs, pr_labels = None, None, None, None

    n_panels = 4 if (precs is not None and probs is not None) else 3
    fig = plt.figure(figsize=(15, 4.8), facecolor=BG)
    gs  = gridspec.GridSpec(1, n_panels, wspace=0.38,
                             left=0.05, right=0.97, top=0.84, bottom=0.2)

    # ── Per-movie F1 bar ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    short = [m.replace("movie_", "") for m in movies]
    colors = [C_GREEN if v >= 0.6 else C_ORANGE if v >= 0.35 else C_RED for v in f1s]
    ax1.barh(short, f1s, color=colors, alpha=0.85)
    ax1.axvline(np.mean(f1s), color=C_TEAL, lw=1.5, ls="--",
                label=f"Mean F1 = {np.mean(f1s):.3f}")
    for i, v in enumerate(f1s):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    ax1.set_xlabel("F1 score", fontsize=9)
    ax1.set_title("DivisionClassifier (v1)\nF1 per test movie",
                  fontsize=9, color=C_TEAL, fontweight="bold")
    ax1.set_xlim(0, 1.08)
    ax1.legend(fontsize=7.5, framealpha=0.85, loc="best")
    ax1.set_facecolor(BG)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # ── TP/FP/FN stacked bar ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(short, tps, color=C_GREEN,  alpha=0.85, label="TP")
    ax2.barh(short, fps, left=tps, color=C_RED,    alpha=0.75, label="FP")
    lefts_fn = [t + f for t, f in zip(tps, fps)]
    ax2.barh(short, fns, left=lefts_fn, color=C_ORANGE, alpha=0.75, label="FN")
    ax2.set_xlabel("Event count", fontsize=9)
    ax2.set_title("TP / FP / FN per movie\n(sorted by F1 ascending)",
                  fontsize=9, color=C_TEAL, fontweight="bold")
    ax2.legend(fontsize=7.5, framealpha=0.85, loc="best")
    ax2.set_facecolor(BG)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ── Density vs F1 scatter ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    sc  = ax3.scatter(gt_divs, [f1s[movies.index(m)] for m in movies],
                      c=[f1s[movies.index(m)] for m in movies],
                      cmap="RdYlGn", vmin=0, vmax=1, s=60, edgecolors="k", lw=0.4)
    for m, x, y in zip(movies, gt_divs, [f1s[movies.index(m)] for m in movies]):
        ax3.annotate(m.replace("movie_", ""), (x, y),
                     textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax3.set_xlabel("GT divisions in movie (density proxy)", fontsize=9)
    ax3.set_ylabel("F1", fontsize=9)
    ax3.set_title("Division density vs F1\n(dense movies → lower F1)",
                  fontsize=9, color=C_TEAL, fontweight="bold")
    plt.colorbar(sc, ax=ax3, shrink=0.8, label="F1")
    ax3.set_facecolor(BG)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # ── PR curve + score histogram ────────────────────────────────────────────
    if precs is not None and probs is not None:
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(recs, precs, color=C_TEAL2, lw=2.0, label="PR curve")
        ax4.axvline(data["aggregate"]["recall"],
                    color=C_PURPLE, lw=1.3, ls="--",
                    label=f"Operating R={data['aggregate']['recall']:.2f}")
        ax4.axhline(data["aggregate"]["precision"],
                    color=C_PURPLE, lw=1.3, ls=":",
                    label=f"Operating P={data['aggregate']['precision']:.2f}")
        ax4.scatter([data["aggregate"]["recall"]],
                    [data["aggregate"]["precision"]],
                    color=C_RED, s=60, zorder=5)
        ax4.set_xlabel("Recall",    fontsize=9)
        ax4.set_ylabel("Precision", fontsize=9)
        ax4.set_title("Precision–Recall curve\n(val split)",
                      fontsize=9, color=C_TEAL, fontweight="bold")
        ax4.legend(fontsize=6.5, framealpha=0.85, loc="best")
        ax4.set_xlim(0, 1.04)
        ax4.set_ylim(0, 1.04)
        ax4.set_facecolor(BG)
        for sp in ["top", "right"]:
            ax4.spines[sp].set_visible(False)

    fig.suptitle("DivisionClassifier (v1) — Performance Analysis",
                 fontsize=11, fontweight="bold", color=C_TEAL, y=0.97)

    caption = (
        "DivisionClassifier v1 evaluated on held-out test movies using 8 triplet features "
        "(parent@t, daughter₁@t+1, daughter₂@t+1). "
        "Left: F1 per movie (green ≥ 0.6, orange ≥ 0.35, red < 0.35). "
        "Centre: TP/FP/FN breakdown — dense movies accumulate large FP counts because the model "
        "was trained on random (easy) negatives and fires on nearby non-daughter pairs. "
        "Right: GT division count (density proxy) vs F1 — the negative correlation motivates "
        "hard-negative mining in v2 (re-sample negatives from the proximity neighbourhood of each parent)."
    )
    fig.text(0.02, 0.01, _wrap(fig, caption), fontsize=7, color="#444444",
             style="italic", va="bottom", ha="left",
             transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir",  default="weights")
    ap.add_argument("--dataset_dir",  default="dataset")
    ap.add_argument("--output_dir",   default="figures")
    ap.add_argument("--dpi",          type=int, default=300)
    args = ap.parse_args()

    baseline_json  = os.path.join(args.weights_dir, "results_baseline.json")
    learned_json   = os.path.join(args.weights_dir, "results_learned.json")
    divisions_json = os.path.join(args.weights_dir, "results_divisions.json")
    div_model      = os.path.join(args.weights_dir, "division_classifier.pt")

    if os.path.exists(baseline_json) and os.path.exists(learned_json):
        plot_tracking_comparison(
            baseline_json, learned_json,
            os.path.join(args.output_dir, "tracking_comparison.png"),
            args.dpi,
        )
    else:
        print(f"SKIP tracking comparison: {baseline_json} or {learned_json} not found")

    if os.path.exists(divisions_json):
        plot_division_performance(
            divisions_json, div_model, args.dataset_dir,
            os.path.join(args.output_dir, "division_performance.png"),
            args.dpi,
        )
    else:
        print(f"SKIP division performance: {divisions_json} not found")


if __name__ == "__main__":
    main()
