#!/usr/bin/env python3
"""Measure real single-cell geometry/photometry stats to calibrate SyMBac_2
synthetic generation (see generate_dataset.py's --species dispatch).

Only ever reads the TRAIN split of each species' real dataset -- val must
never leak into calibration, since calibrated synthetic data is folded back
into training.

TB source: data/bacdetr_mtb/{annotation_crops/*_coco.json, images/train/*.tiff}
  (frames 0..23, i.e. args.val_start_frame, matching build_annotations_mtb.py)
PA source: prep_training/build_pa_standardized_dataset.py's load_ibs() /
  load_e013_tiled(), filtered to exclude VAL_IBS_MOVIES / VAL_E013_MOVIES.

Output: calibration_profiles/{species}.json with geometry (length_px,
width_px, aspect_ratio) and photometry (snr_db, psf_sigma_px) distributions.

Usage:
  python calibrate_from_real.py --species tb --output calibration_profiles/tb.json
  python calibrate_from_real.py --species pa --output calibration_profiles/pa.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.measure import regionprops

SINGLE_CELL_CAT_ID = 1  # consistent across mtb 3-class and rod_3class schemes

MTB_ROOT = Path("/scicore/home/boeluc00/martin0088/data/bacdetr_mtb")
PA_BUILDER_DIR = Path("/scicore/home/boeluc00/martin0088/prep_training")


# ---------------------------------------------------------------------------
# Per-image stat extraction (shared by both species)
# ---------------------------------------------------------------------------

def _polygon_to_contour(seg_flat):
    pts = np.asarray(seg_flat, dtype=np.float32).reshape(-1, 2)
    return pts


def _length_width_px(contour):
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    length, width = max(w, h), min(w, h)
    return length, width


def _fill_mask(shape_hw, contours):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for c in contours:
        cv2.fillPoly(mask, [c.astype(np.int32)], 1)
    return mask


def _edge_sigma_estimate(img, all_mask):
    """Best-effort PSF-sigma proxy: for a Gaussian-blurred step edge of
    amplitude A, max slope ~= A / (sigma * sqrt(2*pi)) -> sigma ~= A / (grad * sqrt(2*pi)).
    Approximated using a boundary ring around all cell masks in this crop.
    Coarse by construction (no real optical model available to calibrate
    against more rigorously) -- treat as an order-of-magnitude guide only.
    """
    if all_mask.sum() == 0:
        return None
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(all_mask, kernel, iterations=1)
    eroded = cv2.erode(all_mask, kernel, iterations=1)
    ring = (dilated - eroded).astype(bool)
    if ring.sum() < 10:
        return None

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_at_ring = grad_mag[ring]
    if grad_at_ring.size == 0:
        return None
    grad_p95 = np.percentile(grad_at_ring, 95)
    if grad_p95 <= 0:
        return None

    fg_mean = img[eroded.astype(bool)].mean() if eroded.sum() > 0 else img[all_mask.astype(bool)].mean()
    bg_mean = img[(all_mask == 0)].mean()
    amplitude = abs(fg_mean - bg_mean)
    if amplitude <= 0:
        return None
    return float(amplitude / (grad_p95 * np.sqrt(2 * np.pi)))


def _cell_shape_stats(contour, shape_hw):
    """Per-cell eccentricity/solidity/area via skimage regionprops on that
    cell's own filled mask -- shape descriptors that a pure length/width
    aspect ratio misses (a real cell can have aspect~1 but still not be a
    perfect circle, e.g. cocci: mildly elongated *and* not fully convex)."""
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(shape_hw[1], x + w + pad)
    y1 = min(shape_hw[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_contour = contour.copy()
    local_contour[:, 0] -= x0
    local_contour[:, 1] -= y0
    cv2.fillPoly(local_mask, [local_contour.astype(np.int32)], 1)
    if local_mask.sum() < 9:
        return None
    props = regionprops(local_mask)
    if not props:
        return None
    p = props[0]
    return {"eccentricity": float(p.eccentricity), "solidity": float(p.solidity),
            "area_px": float(p.area)}


def _crop_stats(img, single_cell_contours, all_contours):
    """Returns dict of per-crop measurements, or None if nothing usable."""
    if not single_cell_contours:
        return None

    lengths, widths = [], []
    eccentricities, solidities, areas = [], [], []
    for c in single_cell_contours:
        length, width = _length_width_px(c)
        if width <= 0:
            continue
        lengths.append(length)
        widths.append(width)
        shape = _cell_shape_stats(c, img.shape)
        if shape is not None:
            eccentricities.append(shape["eccentricity"])
            solidities.append(shape["solidity"])
            areas.append(shape["area_px"])
    if not lengths:
        return None

    all_mask = _fill_mask(img.shape, all_contours)
    bg_pixels = img[all_mask == 0]
    if bg_pixels.size < 100:
        return None
    bg_mean, bg_std = float(bg_pixels.mean()), float(bg_pixels.std())

    sc_mask = _fill_mask(img.shape, single_cell_contours)
    cell_pixels = img[sc_mask == 1]
    cell_mean = float(cell_pixels.mean()) if cell_pixels.size else bg_mean

    sigma_est = _edge_sigma_estimate(img, all_mask)

    return {
        "length_px": lengths,
        "width_px": widths,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "cell_mean": cell_mean,
        "psf_sigma_px": sigma_est,
        "eccentricity": eccentricities,
        "solidity": solidities,
        "area_px": areas,
    }


# ---------------------------------------------------------------------------
# TB loader
# ---------------------------------------------------------------------------

def _frame_of_crop(crop_stem: str) -> int:
    tpart = crop_stem.split("_")[2]
    assert tpart.startswith("t")
    return int(tpart[1:])


def load_tb_crops(val_start_frame=24):
    ann_crop_dir = MTB_ROOT / "annotation_crops"
    img_dir = MTB_ROOT / "images" / "train"
    crops = []
    for jf in sorted(ann_crop_dir.glob("*_coco.json")):
        stem = jf.stem.replace("_coco", "")
        if _frame_of_crop(stem) >= val_start_frame:
            continue
        img_path = img_dir / f"{stem}.tiff"
        if not img_path.exists():
            continue
        with open(jf) as f:
            data = json.load(f)
        anns = [a for a in data.get("annotations", []) if not a.get("iscrowd", 0)]
        if not anns:
            continue
        img = tifffile.imread(img_path).astype(np.float32)
        single_cell = [_polygon_to_contour(a["segmentation"][0])
                       for a in anns if a["category_id"] == SINGLE_CELL_CAT_ID]
        all_contours = [_polygon_to_contour(a["segmentation"][0]) for a in anns]
        crops.append((img, single_cell, all_contours))
    return crops


# ---------------------------------------------------------------------------
# Generic single-COCO-json loader (E. coli, M. abscessus -- any species whose
# real train set is already one merged COCO json + a flat images dir, as
# opposed to TB's per-crop-json layout or PA's own multi-source merge)
# ---------------------------------------------------------------------------

ROD_PERSPECIES_ROOT = Path("/scicore/home/boeluc00/martin0088/data/bacdetr_rod_nafnet_only_perspecies")
COC_MERGED_ROOT = Path("/scicore/home/boeluc00/martin0088/data/bacdetr_coc_merged")
GENERIC_SPECIES_ROOT = {
    "ecoli": ROD_PERSPECIES_ROOT,
    "mabs":  ROD_PERSPECIES_ROOT,
    "coc":   COC_MERGED_ROOT,
}
GENERIC_SPECIES_ANN = {
    "ecoli": "instances_train_3class_ecoli_ibsonly.json",
    "mabs":  "instances_train_2class_m_abscessus.json",
    "coc":   "instances_train_coc_3class.json",
}


def load_generic_coco_crops(ann_path, images_dir):
    with open(ann_path) as f:
        data = json.load(f)
    anns_by_img = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    crops = []
    for im in data["images"]:
        anns = anns_by_img.get(im["id"], [])
        if not anns:
            continue
        img_path = Path(images_dir) / im["file_name"]
        if not img_path.exists():
            continue
        img = tifffile.imread(img_path).astype(np.float32)
        single_cell = [_polygon_to_contour(a["segmentation"][0])
                       for a in anns if a["category_id"] == SINGLE_CELL_CAT_ID]
        all_contours = [_polygon_to_contour(a["segmentation"][0]) for a in anns]
        crops.append((img, single_cell, all_contours))
    return crops


# ---------------------------------------------------------------------------
# PA loader (reuses build_pa_standardized_dataset.py's real-data loaders)
# ---------------------------------------------------------------------------

def load_pa_crops():
    sys.path.insert(0, str(PA_BUILDER_DIR))
    import build_pa_standardized_dataset as pa_builder  # noqa: E402

    records = pa_builder.load_ibs() + pa_builder.load_e013_tiled()
    val_movies = pa_builder.VAL_IBS_MOVIES | pa_builder.VAL_E013_MOVIES
    crops = []
    for rec in records:
        if rec["movie"] in val_movies:
            continue
        img = rec["arr"].astype(np.float32)
        single_cell, all_contours = [], []
        for cat_id, poly in rec["anns"]:
            contour = _polygon_to_contour(poly[0] if isinstance(poly[0], list) else poly)
            all_contours.append(contour)
            if cat_id == SINGLE_CELL_CAT_ID:
                single_cell.append(contour)
        crops.append((img, single_cell, all_contours))
    return crops


# ---------------------------------------------------------------------------
# Aggregation + profile output
# ---------------------------------------------------------------------------

def build_profile(crops, species):
    all_lengths, all_widths = [], []
    all_eccentricities, all_solidities, all_areas = [], [], []
    bg_means, bg_stds, cell_means, sigmas = [], [], [], []

    for img, single_cell, all_contours in crops:
        stats = _crop_stats(img, single_cell, all_contours)
        if stats is None:
            continue
        all_lengths.extend(stats["length_px"])
        all_widths.extend(stats["width_px"])
        all_eccentricities.extend(stats["eccentricity"])
        all_solidities.extend(stats["solidity"])
        all_areas.extend(stats["area_px"])
        bg_means.append(stats["bg_mean"])
        bg_stds.append(stats["bg_std"])
        cell_means.append(stats["cell_mean"])
        if stats["psf_sigma_px"] is not None:
            sigmas.append(stats["psf_sigma_px"])

    if not all_lengths:
        raise RuntimeError(f"No usable single-cell annotations found for species={species}")

    lengths = np.array(all_lengths)
    widths = np.array(all_widths)
    aspect = lengths / widths
    eccentricities = np.array(all_eccentricities) if all_eccentricities else np.array([])
    solidities = np.array(all_solidities) if all_solidities else np.array([])
    areas = np.array(all_areas) if all_areas else np.array([])

    bg_mean_overall = float(np.mean(bg_means))
    bg_std_overall = float(np.mean(bg_stds))
    cell_mean_overall = float(np.mean(cell_means))
    amplitude = max(cell_mean_overall - bg_mean_overall, 1e-6)
    snr_db = 20.0 * np.log10(amplitude / bg_std_overall) if bg_std_overall > 0 else None

    return {
        "species": species,
        "n_crops_used": len(bg_means),
        "n_single_cells": int(len(all_lengths)),
        "geometry": {
            "width_px_mean": float(widths.mean()),
            "width_px_std": float(widths.std()),
            "length_px_mean": float(lengths.mean()),
            "length_px_std": float(lengths.std()),
            "aspect_ratio_mean": float(aspect.mean()),
            "aspect_ratio_std": float(aspect.std()),
        },
        "shape": {
            "eccentricity_mean": float(eccentricities.mean()) if eccentricities.size else None,
            "eccentricity_std": float(eccentricities.std()) if eccentricities.size else None,
            "solidity_mean": float(solidities.mean()) if solidities.size else None,
            "solidity_std": float(solidities.std()) if solidities.size else None,
            "area_px_mean": float(areas.mean()) if areas.size else None,
            "area_px_std": float(areas.std()) if areas.size else None,
            "note": ("skimage.measure.regionprops on each single-cell polygon's own "
                     "filled mask -- eccentricity=0/solidity=1 is a perfect circle; "
                     "useful mainly for round cells (coc) where length/width aspect "
                     "ratio alone doesn't capture non-circularity."),
        },
        "photometry": {
            "snr_db_estimate": float(snr_db) if snr_db is not None else None,
            "psf_sigma_px_median": float(np.median(sigmas)) if sigmas else None,
            "psf_sigma_px_n": len(sigmas),
            "background_mean": bg_mean_overall,
            "background_std": bg_std_overall,
            "cell_mean": cell_mean_overall,
        },
        "notes": (
            "psf_sigma is a coarse edge-spread proxy (amplitude / (p95 boundary "
            "gradient * sqrt(2*pi))), not a fitted optical model -- treat as "
            "order-of-magnitude guidance only. snr_db uses generate_dataset.py's "
            "own definition (20*log10(peak_signal/noise_std)) so it plugs "
            "directly into --species sampling ranges."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=["tb", "pa", "ecoli", "mabs", "coc"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_start_frame", type=int, default=24,
                         help="TB only -- matches build_annotations_mtb.py")
    args = parser.parse_args()

    if args.species == "tb":
        crops = load_tb_crops(args.val_start_frame)
    elif args.species == "pa":
        crops = load_pa_crops()
    else:
        root = GENERIC_SPECIES_ROOT[args.species]
        ann_path = root / "annotations" / GENERIC_SPECIES_ANN[args.species]
        images_dir = root / "images" / "train"
        crops = load_generic_coco_crops(ann_path, images_dir)

    profile = build_profile(crops, args.species)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"[{args.species}] {profile['n_crops_used']} crops, "
          f"{profile['n_single_cells']} single-cell polygons")
    print(f"  width_px  = {profile['geometry']['width_px_mean']:.2f} +/- {profile['geometry']['width_px_std']:.2f}")
    print(f"  length_px = {profile['geometry']['length_px_mean']:.2f} +/- {profile['geometry']['length_px_std']:.2f}")
    print(f"  aspect    = {profile['geometry']['aspect_ratio_mean']:.2f} +/- {profile['geometry']['aspect_ratio_std']:.2f}")
    print(f"  eccentricity = {profile['shape']['eccentricity_mean']} +/- {profile['shape']['eccentricity_std']}")
    print(f"  solidity     = {profile['shape']['solidity_mean']} +/- {profile['shape']['solidity_std']}")
    print(f"  area_px      = {profile['shape']['area_px_mean']} +/- {profile['shape']['area_px_std']}")
    print(f"  snr_db    = {profile['photometry']['snr_db_estimate']}")
    print(f"  psf_sigma = {profile['photometry']['psf_sigma_px_median']} (n={profile['photometry']['psf_sigma_px_n']})")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
