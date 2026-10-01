#!/usr/bin/env python3
"""Check whether PA's two real data sources (IBS vs e013) have consistent
cell size scales, or whether pooling them is averaging over two genuinely
different physical scales -- would explain why a single real crop can look
nothing like the pooled calibration average."""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path.home() / "prep_training"))
import build_pa_standardized_dataset as pa_builder


def measure(records, name):
    lengths, widths = [], []
    for rec in records:
        for cat_id, poly in rec["anns"]:
            if cat_id != 1:
                continue
            pts = np.array(poly[0], dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            lengths.append(max(w, h))
            widths.append(min(w, h))
    lengths, widths = np.array(lengths), np.array(widths)
    print(f"{name}: n={len(lengths)} length={lengths.mean():.2f}+/-{lengths.std():.2f} "
          f"width={widths.mean():.2f}+/-{widths.std():.2f}")


print("Loading IBS...")
ibs = pa_builder.load_ibs()
measure(ibs, "IBS (all movies, incl val)")

print("Loading e013...")
e013 = pa_builder.load_e013_tiled()
measure(e013, "e013 (all frames, incl val)")
