#!/usr/bin/env python3
"""
plot_training_curves.py — plot epoch-level training dynamics.

Reads the training log JSON files saved by train_assignment.py and
train_division.py (requires --save_log flag or the logging added after June 2026).

Each log file is a JSON list:
  [{"epoch": 1, "train_loss": 0.42, "val_f1": 0.61, "val_p": 0.70, "val_r": 0.55}, ...]

If both an old and new log exist for the DivisionClassifier (e.g. pre/post
hard-negative mining), both are plotted on the same axes for comparison.

Usage (from symbac_training/):
    python plots/plot_training_curves.py
    python plots/plot_training_curves.py --weights_dir weights/ --output_dir figures/
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

C_TEAL   = (0,    76/255,  76/255)
C_TEAL2  = (0,   128/255, 128/255)
C_PURPLE = (85/255, 0,     75/255)
C_ORANGE = (210/255, 110/255, 0)
BG       = "#f5f7fa"


def load_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_one_run(axes, log, color, label_prefix, lw=1.8, ls="-"):
    """Plot train_loss, val_f1, val_p, val_r onto 4 axes."""
    epochs = [r["epoch"]      for r in log]
    loss   = [r["train_loss"] for r in log]
    f1     = [r["val_f1"]     for r in log]
    prec   = [r["val_p"]      for r in log]
    rec    = [r["val_r"]      for r in log]

    kw = dict(color=color, lw=lw, ls=ls)
    axes[0].plot(epochs, loss,  **kw, label=f"{label_prefix} train loss")
    axes[1].plot(epochs, f1,    **kw, label=f"{label_prefix} val F1")
    axes[2].plot(epochs, prec,  **kw, label=f"{label_prefix} val precision")
    axes[3].plot(epochs, rec,   **kw, label=f"{label_prefix} val recall")

    # Mark best epoch (highest val_f1)
    best_epoch = epochs[int(np.argmax(f1))]
    for ax, vals in zip(axes, [loss, f1, prec, rec]):
        ax.axvline(best_epoch, color=color, lw=0.8, ls=":", alpha=0.7)


def make_figure(logs_and_meta, title, output_path, dpi, caption=""):
    """
    logs_and_meta: list of (log, color, label_prefix) tuples.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2),
                              gridspec_kw={"wspace": 0.32,
                                           "left": 0.05, "right": 0.97,
                                           "top": 0.84, "bottom": 0.20})
    fig.patch.set_facecolor(BG)

    ylabels = ["Train loss", "Val F1", "Val precision", "Val recall"]
    y_ranges = [None, (0, 1.05), (0, 1.05), (0, 1.05)]

    for log, color, label_prefix in logs_and_meta:
        plot_one_run(axes, log, color, label_prefix)

    for ax, ylabel, ylim in zip(axes, ylabels, y_ranges):
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7.5, framealpha=0.85, loc="best")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_facecolor(BG)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", color=C_TEAL, y=0.97)

    if caption:
        fig.text(0.02, 0.01, caption, fontsize=7, color="#444444", style="italic",
                 va="bottom", ha="left", transform=fig.transFigure)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--output_dir",  default="figures")
    ap.add_argument("--dpi",         type=int, default=300)
    args = ap.parse_args()

    wdir = args.weights_dir
    odir = args.output_dir

    # ── AssignmentScorer ──────────────────────────────────────────────────────
    asgn_log = load_log(os.path.join(wdir, "assignment_scorer_training_log.json"))
    if asgn_log:
        make_figure(
            [(asgn_log, C_TEAL2, "AssignmentMLP")],
            "AssignmentScorer — Training Curves",
            os.path.join(odir, "training_curves_assignment.png"),
            args.dpi,
            caption=(
                "Epoch-level training dynamics for the AssignmentScorer (MLP, 10 pair features → assignment cost). "
                "Vertical dotted line marks the epoch with the highest validation F1 "
                "(the checkpoint that is saved). "
                "Loss uses FocalLoss(α=0.25, γ=2.0); threshold is tuned post-training to maximise F1 on the validation split."
            ),
        )
    else:
        print(f"SKIP AssignmentScorer curves: assignment_scorer_training_log.json not found in {wdir}")
        print("  Re-run train_assignment.py — log is written automatically to <output_dir>/")

    # ── DivisionClassifier — optionally compare v1 and v2 ────────────────────
    div_log_v1  = load_log(os.path.join(wdir, "division_classifier_training_log_v1.json"))
    div_log_new = load_log(os.path.join(wdir, "division_classifier_training_log.json"))

    runs = []
    if div_log_v1:
        runs.append((div_log_v1, C_ORANGE, "v1 (random neg)"))
    if div_log_new:
        runs.append((div_log_new, C_PURPLE, "v2 (hard-neg mining)"))

    if runs:
        title = "DivisionClassifier — Training Curves"
        if len(runs) == 2:
            title += "\n(v1 = random negatives | v2 = hard-negative mining)"
        div_caption = (
            "Epoch-level training dynamics for the DivisionClassifier (MLP, 8 triplet features → division probability). "
            "Vertical dotted line marks the epoch with highest validation F1 (saved checkpoint). "
            "Loss uses FocalLoss(α=0.75, γ=2.0) to up-weight the rare positive (division) class; "
            "threshold is tuned post-training to maximise recall while keeping precision ≥ 0.60."
        )
        if len(runs) == 2:
            div_caption += (
                " Orange = v1 (random negatives, easy negatives dominate training). "
                "Purple = v2 (proximity-filtered hard negatives, negatives sampled within 5× median major axis of parent)."
            )
        make_figure(
            runs,
            title,
            os.path.join(odir, "training_curves_division.png"),
            args.dpi,
            caption=div_caption,
        )
    else:
        print(f"SKIP DivisionClassifier curves: division_classifier_training_log.json not found in {wdir}")
        print("  Re-run train_division.py — log is written automatically to <output_dir>/")

    if not asgn_log and not runs:
        print("\nNo training logs found. Logs are written to weights/ at the end of each training run.")


if __name__ == "__main__":
    main()
