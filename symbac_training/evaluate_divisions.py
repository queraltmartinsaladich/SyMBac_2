#!/usr/bin/env python3
"""
evaluate_divisions.py — evaluate DivisionClassifier precision/recall on held-out movies.

Runs HungarianTracker on GT masks (perfect detection), then DivisionClassifier
to detect division events, and compares to ground-truth divisions from lineage.json.

A predicted division is a TP when the correct parent track is identified at the
correct frame transition. FP = predicted but not in GT. FN = GT but not predicted.

Usage:
    python evaluate_divisions.py \\
        --split_file /path/to/split.json --split test \\
        --model_path /path/to/division_classifier.pt

    # With assignment scorer for better upstream tracking:
    python evaluate_divisions.py \\
        --split_file split.json --split test \\
        --model_path division_classifier.pt \\
        --assignment_model assignment_scorer.pt \\
        --output results_divisions.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from skimage.measure import regionprops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from HiTMicTools.tracking.hungarian_tracker import HungarianTracker
    from HiTMicTools.tracking.division_classifier import DivisionClassifier
except ImportError:
    raise ImportError(
        "HiTMicTools not found. Install with:\n"
        "  pip install -e /path/to/HiTMicTools"
    )


# ── data loading ──────────────────────────────────────────────────────────────

def load_movie(movie_dir):
    masks = np.load(os.path.join(movie_dir, "masks.npz"))["data"]
    with open(os.path.join(movie_dir, "lineage.json")) as f:
        lineage = json.load(f)
    return masks, lineage


def extract_measurements(masks):
    rows = []
    for t, frame in enumerate(masks):
        for rp in regionprops(frame):
            cy, cx = rp.centroid
            rows.append({
                "frame": t,
                "label": rp.label,
                "centroid_0": cy,
                "centroid_1": cx,
                "area": rp.area,
                "solidity": rp.solidity,
                "major_axis_length": rp.major_axis_length,
                "minor_axis_length": rp.minor_axis_length,
                "orientation": rp.orientation,
            })
    if not rows:
        return pd.DataFrame(
            columns=["frame", "label", "centroid_0", "centroid_1",
                     "area", "solidity", "major_axis_length",
                     "minor_axis_length", "orientation"]
        )
    return pd.DataFrame(rows)


# ── ground truth ──────────────────────────────────────────────────────────────

def build_gt_divisions(lineage):
    """
    Returns set of (parent_label_at_frame_t_minus_1, frame_t) for each division.

    SyMBac_2 lineage: {frame: {cell_id: parent_cell_id}}.
    Non-self entries (cell_id != parent_cell_id) mark new daughters.
    Each such entry implies a division of parent_cell_id at frame_t - 1.
    """
    events = set()
    for t_str, cells in lineage.items():
        t = int(t_str)
        for d_str, p in cells.items():
            if int(d_str) != int(p):
                events.add((int(p), t))
    return events


# ── tracker ───────────────────────────────────────────────────────────────────

def run_tracker(masks, df, tracker, assignment_scorer=None, learned_max_cost=0.5):
    cost_overrides = None
    if assignment_scorer is not None:
        from HiTMicTools.tracking.feature_extraction import compute_movie_stats
        stats = compute_movie_stats(masks)
        sorted_frames = sorted(df["frame"].unique())
        cost_overrides = {}
        for fi, fval in enumerate(sorted_frames[:-1]):
            fval_next = sorted_frames[fi + 1]
            cm, lt, lt1 = assignment_scorer.predict_cost_matrix(
                masks[fval], masks[fval_next], stats,
                masks_t_prev=masks[fval - 1] if fval > 0 else None,
            )
            cost_overrides[fval] = (cm, lt, lt1)

    return tracker.track_objects(
        df.copy(),
        cost_overrides=cost_overrides,
        learned_max_cost=learned_max_cost,
    )


# ── per-movie evaluation ──────────────────────────────────────────────────────

def evaluate_movie(masks, lineage, tracker, division_clf,
                   assignment_scorer=None, learned_max_cost=0.5):
    df = extract_measurements(masks)
    if df.empty:
        return None

    gt_div_labels = build_gt_divisions(lineage)
    if not gt_div_labels:
        return None  # movie has no divisions — skip

    # Run tracker to assign track IDs
    result_df = run_tracker(masks, df, tracker, assignment_scorer, learned_max_cost)

    # Resolve GT division labels → track IDs
    label_to_track = {
        (int(r["frame"]), int(r["label"])): int(r["trackid"])
        for _, r in result_df.iterrows()
    }

    gt_div_track = set()
    for (parent_label, frame_t) in gt_div_labels:
        parent_trackid = label_to_track.get((frame_t - 1, parent_label))
        if parent_trackid is not None and parent_trackid >= 0:
            gt_div_track.add((parent_trackid, frame_t))

    # Run DivisionClassifier
    result_df_div, _ = division_clf.predict_divisions(result_df.copy(), masks)

    # Predicted divisions: unique (parent_trackid, frame_t) where column is set
    pred_div_track = set()
    div_rows = result_df_div[result_df_div["division_parent_trackid"].notna()]
    for _, row in div_rows.iterrows():
        pred_div_track.add((int(row["division_parent_trackid"]), int(row["frame"])))

    tp = len(gt_div_track & pred_div_track)
    fp = len(pred_div_track - gt_div_track)
    fn = len(gt_div_track - pred_div_track)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "n_gt_divisions":   len(gt_div_track),
        "n_pred_divisions": len(pred_div_track),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(metrics_list):
    if not metrics_list:
        return {}
    count_keys = {"n_gt_divisions", "n_pred_divisions", "tp", "fp", "fn"}
    agg = {}
    for k in metrics_list[0]:
        vals = [m[k] for m in metrics_list]
        agg[k] = sum(vals) if k in count_keys else float(np.mean(vals))
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    agg["precision"] = tp / max(tp + fp, 1)
    agg["recall"]    = tp / max(tp + fn, 1)
    p, r = agg["precision"], agg["recall"]
    agg["f1"] = 2 * p * r / max(p + r, 1e-9)
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DivisionClassifier on synthetic movies."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split_file", metavar="PATH",
                        help="split.json written by build_dataset.py")
    source.add_argument("--data_dir", metavar="PATH",
                        help="folder containing movie_XXX sub-folders")

    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--model_path", required=True, metavar="PATH",
                        help="path to division_classifier.pt")
    parser.add_argument("--assignment_model", default=None, metavar="PATH",
                        help="optional path to assignment_scorer.pt for better upstream tracking")
    parser.add_argument("--max_distance", type=float, default=25.0)
    parser.add_argument("--gap_bridge_frames", type=int, default=2)
    parser.add_argument("--learned_max_cost", type=float, default=0.5)
    parser.add_argument("--output", metavar="PATH", default=None)
    args = parser.parse_args()

    # Resolve movie list
    if args.split_file:
        with open(args.split_file) as f:
            split_data = json.load(f)
        if args.split == "all":
            movie_dirs = split_data["train"] + split_data["val"] + split_data["test"]
        else:
            movie_dirs = split_data[args.split]
        print(f"Split [{args.split}]: {len(movie_dirs)} movies  "
              f"(from {args.split_file})")
    else:
        movie_dirs = sorted([
            os.path.join(args.data_dir, d)
            for d in os.listdir(args.data_dir)
            if d.startswith("movie_") and os.path.isdir(os.path.join(args.data_dir, d))
        ])
        print(f"Data dir: {args.data_dir}  ({len(movie_dirs)} movies found)")

    if not movie_dirs:
        print("ERROR: no movies found.")
        sys.exit(1)

    # Load models
    division_clf = DivisionClassifier(args.model_path)
    print(f"DivisionClassifier : {args.model_path}")
    print(f"  threshold        : {division_clf.threshold:.4f}")

    assignment_scorer = None
    if args.assignment_model:
        from HiTMicTools.tracking.assignment_scorer import AssignmentScorer
        assignment_scorer = AssignmentScorer(args.assignment_model)
        print(f"AssignmentScorer   : {args.assignment_model}")

    tracker = HungarianTracker(
        max_distance=args.max_distance,
        gap_bridge_frames=args.gap_bridge_frames,
    )
    tracker.set_features(
        ["area", "major_axis_length", "minor_axis_length", "solidity", "orientation"]
    )
    print(f"Tracker            : max_distance={args.max_distance} px, "
          f"gap_bridge_frames={args.gap_bridge_frames}")
    print()

    all_metrics = []
    per_movie_records = []

    for movie_dir in movie_dirs:
        name = os.path.basename(movie_dir)
        try:
            masks, lineage = load_movie(movie_dir)
        except Exception as e:
            print(f"  SKIP {name}: load error — {e}")
            continue

        try:
            m = evaluate_movie(
                masks, lineage, tracker, division_clf,
                assignment_scorer=assignment_scorer,
                learned_max_cost=args.learned_max_cost,
            )
        except Exception as e:
            print(f"  SKIP {name}: error — {e}")
            import traceback; traceback.print_exc()
            continue

        if m is None:
            print(f"  SKIP {name}: no GT divisions")
            continue

        all_metrics.append(m)
        per_movie_records.append({"movie": movie_dir, **m})
        print(
            f"  {name:20s}  "
            f"GT={m['n_gt_divisions']:3d}  Pred={m['n_pred_divisions']:3d}  "
            f"TP={m['tp']:3d}  FP={m['fp']:3d}  FN={m['fn']:3d}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )

    if not all_metrics:
        print("\nNo movies evaluated successfully.")
        sys.exit(1)

    agg = aggregate(all_metrics)
    w = 58
    print(f"\n{'─' * w}")
    print(f"  AGGREGATE  ({len(all_metrics)} / {len(movie_dirs)} movies)")
    print(f"{'─' * w}")
    print(f"  GT divisions   : {agg['n_gt_divisions']:5d}    "
          f"Predicted : {agg['n_pred_divisions']:5d}")
    print(f"  TP / FP / FN   : {agg['tp']:4d} / {agg['fp']:4d} / {agg['fn']:4d}")
    print(f"  Precision      : {agg['precision']:.4f}")
    print(f"  Recall         : {agg['recall']:.4f}")
    print(f"  F1             : {agg['f1']:.4f}")
    print(f"{'─' * w}")

    if args.output:
        out_data = {
            "config": {
                "model_path":        args.model_path,
                "assignment_model":  args.assignment_model,
                "max_distance":      args.max_distance,
                "gap_bridge_frames": args.gap_bridge_frames,
                "learned_max_cost":  args.learned_max_cost,
                "split":             args.split,
                "n_movies":          len(all_metrics),
            },
            "aggregate": agg,
            "per_movie": per_movie_records,
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
