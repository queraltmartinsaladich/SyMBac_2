#!/usr/bin/env python3
"""Generate synthetic rod-shaped bacteria crops via procedural cell
placement -- NOT SyMBac's pymunk physics engine (generate_dataset.py).
That engine produces two real, verified defects: cells seeded in one tight
grid at the canvas center (never populating the full field of view like
real crops do) and, worse, visible curvature/bending in higher-aspect-ratio
species (physics joint-bend accumulation, worst in PA) that real crops
don't show at all. This is a from-scratch procedural replacement, built
and validated the same way generate_coc_dataset.py was for cocci: measure
real shape/orientation/density stats first, then generate directly from
those instead of simulating physics.

Cell shape: an ellipse sized from real area + eccentricity (calibrate_from_
real.py's "shape" section -- rods are just very eccentric ellipses in this
framework, eccentricity ~0.86-0.93 across TB/PA/ecoli/mabs vs coc's ~0.42).
No curvature is possible by construction (a straight ellipse), which is the
main defect this replaces.

Division: splits one cell into two smaller tangent daughters end-to-end
along a chosen axis (center separation = r_a + r_b, exact, never
overlapping). That axis is NOT assumed colinear with the parent -- measured
real crops (calibrate_from_real.py's "orientation" section) show nearest-
neighbor cell orientation is close to statistically independent for TB/
M. abscessus, and only mildly correlated for PA/E. coli. Each species'
measured near-neighbor vs. random-pair angular gap sets an
`orientation_align_prob` blend between "inherit parent's axis" and "fully
independent," rather than assuming real chains stay visually aligned.

Temporal growth, same structure as generate_coc_dataset.py: movies start as
scattered separate single cells (no pre-formed groups) across the whole
canvas -- real crops show cells as separate individuals/small groups
spread across the full field of view, not one growing central colony
(SyMBac's physics engine only ever populated the canvas center). Each
later frame, existing cells have a chance to divide in place and a few
fresh singles enter, until density approaches the real per-crop average.

Usage:
  python generate_rod_dataset.py --species tb --output_dir data/synth_tb_pool2 --n_movies 20 --seed 1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

FRAMES_PER_MOVIE = 20
IMAGE_SIZE = 512          # matches real crop size

CELL_SIZE_CV = 0.15       # per-cell area jitter (coefficient of variation) around the real mean
SEPARATION = 1.4          # fresh singles placed with an explicit gap (>1 = gap, in combined radii)
# IMPORTANT: the 512x512 frame is a CROP, not a full field of view --
# calibrate_from_real.py's density.cells_per_crop_mean (~44.4 for TB) is the
# real per-crop target, not a floor to exceed. An earlier pass raised this
# past the real mean on a mistaken "denser" request that assumed full-FOV
# images -- reverted back to matching the real target.
# Lower than before now that init_population already starts near the real
# target -- only mild further drift needed, not another 1.7x growth arc.
# Trimmed again per feedback that the final frame still ran slightly denser
# than real (e.g. PA t=19 34 cells vs 30.4 target, TB 51 vs 44.4).
DIVISION_PROB_PER_FRAME = 0.005
NEW_SINGLES_PER_FRAME_RANGE = (0, 1)
MAX_CELLS = 300


def load_profile(species):
    profile_path = Path(__file__).parent / "calibration_profiles" / f"{species}.json"
    with open(profile_path) as f:
        return json.load(f)


def _orientation_align_prob(profile):
    """0 = new cell's orientation is fully independent of its parent/neighbor
    (real near-neighbor angle gap == the ~45deg random-pair baseline);
    1 = always inherits. Derived directly from calibrate_from_real.py's
    measured gap, not assumed."""
    orient = profile.get("orientation", {})
    near = orient.get("near_neighbor_angdiff_deg_mean")
    rand = orient.get("random_pair_angdiff_deg_mean")
    if near is None or rand is None or rand <= 0:
        return 0.0
    return float(np.clip((rand - near) / rand, 0.0, 1.0))


def _target_density(profile):
    d = profile.get("density", {})
    mean = d.get("cells_per_crop_mean")
    return float(mean) if mean else 40.0


def _solve_capsule_halfwidth(a, area):
    """Given a capsule's half-length a and target fill area, solve for the
    half-width b such that 4*b*(a-b) + pi*b^2 == area exactly (the real
    capsule area formula: a rectangle of width 2b, length 2(a-b), plus a
    full circle of radius b from the two end-caps)."""
    A, B, C = (np.pi - 4.0), 4.0 * a, -area
    roots = np.roots([A, B, C])
    valid = [r.real for r in roots if abs(r.imag) < 1e-6 and 0 < r.real <= a]
    if valid:
        return min(valid)
    return max(1.0, area / (4.0 * a))   # fallback: rectangle-dominated approximation


def _sample_cell_ellipse(rng, shape_profile):
    """Semi-major/minor axes (px). Length is sampled directly from the real
    measured length_px (geometry section) -- more reliable than width_px
    alone, since for some species (PA) width_px and area_px/eccentricity
    imply different cell sizes (real cells deviate from a clean geometric
    model enough that not all measured stats agree simultaneously). Width
    is then DERIVED by solving for the half-width that makes this cell's
    capsule area match the real area_px target exactly, rather than
    independently sampling width_px_mean and letting area fall out however
    it lands (that was overshooting real area by ~14-28% after switching
    from ellipses to capsules -- capsules have strictly more area than an
    ellipse at the same length/width, so matching length+width exactly no
    longer means matching area)."""
    length_mean, area_mean = shape_profile["length_px_mean"], shape_profile["area_px_mean"]
    width_mean = shape_profile["width_px_mean"]
    length = float(np.clip(rng.normal(length_mean, length_mean * CELL_SIZE_CV),
                            length_mean * 0.5, length_mean * 1.8))
    area = float(np.clip(rng.normal(area_mean, area_mean * CELL_SIZE_CV),
                          area_mean * 0.5, area_mean * 1.5))
    a = length / 2.0
    b_from_area = _solve_capsule_halfwidth(a, area)
    # Blend toward the directly-measured width_px too: b_from_area alone
    # slightly undershoots real width (small-capsule rasterization measures
    # ~9-12% more area than the continuous formula predicts, so solving
    # purely for area pulls b down further than warranted).
    b = min(0.6 * b_from_area + 0.4 * (width_mean / 2.0), a)
    return a, b   # semi-major, semi-minor


def _cell_radius(cell):
    return sum(cell["axes"]) / 2.0


def _new_cell(rng, shape_profile, pos, orientation_deg):
    a, b = _sample_cell_ellipse(rng, shape_profile)
    return {"pos": pos, "axes": (a, b), "angle_deg": float(orientation_deg)}


def _sample_new_orientation(rng, parent_angle_deg, align_prob):
    if rng.uniform() < align_prob:
        return parent_angle_deg + float(rng.normal(0, 12))   # inherit, with modest jitter
    return float(rng.uniform(0, 180))                         # independent


def _place_free(rng, image_size, shape_profile, existing_cells, sep_factor, max_attempts=150):
    candidate = _new_cell(rng, shape_profile, (0.0, 0.0), rng.uniform(0, 180))
    r_new = _cell_radius(candidate)
    for _ in range(max_attempts):
        x = rng.uniform(r_new, image_size - r_new)
        y = rng.uniform(r_new, image_size - r_new)
        ok = all(
            np.hypot(x - c["pos"][0], y - c["pos"][1]) > sep_factor * (r_new + _cell_radius(c))
            for c in existing_cells
        )
        if ok:
            candidate["pos"] = (x, y)
            return candidate
    return None


def _divide_cell(rng, cell, shape_profile, align_prob):
    """Splits one cell into 2 tangent daughters end-to-end along a chosen
    axis (parent-inherited or independent, per align_prob)."""
    new_angle_deg = _sample_new_orientation(rng, cell["angle_deg"], align_prob)
    axis = np.radians(new_angle_deg)
    dir_x, dir_y = np.cos(axis), np.sin(axis)

    daughters_shape = [_sample_cell_ellipse(rng, shape_profile) for _ in range(2)]
    radii = [sum(ax) / 2.0 for ax in daughters_shape]

    out = []
    for (a, b), sign, r_self in zip(daughters_shape, (1.0, -1.0), radii):
        out.append({
            "pos": (cell["pos"][0] + sign * r_self * dir_x, cell["pos"][1] + sign * r_self * dir_y),
            "axes": (a, b),
            "angle_deg": new_angle_deg,
        })
    return out


def init_population(rng, image_size, shape_profile):
    # Real crops are independent snapshots, not a single colony's early
    # timepoint -- t=0 should already sit AT the real per-crop mean density,
    # not below it (real crops have ~44 cells regardless of when in a
    # hypothetical timeline they were taken).
    target = shape_profile["_target_density"]
    n_initial = max(4, int(rng.integers(int(target * 0.95), int(target * 1.05) + 1)))
    cells = []
    for _ in range(n_initial):
        c = _place_free(rng, image_size, shape_profile, cells, SEPARATION)
        if c is not None:
            cells.append(c)
    return cells


def step_population(rng, cells, image_size, shape_profile, align_prob):
    if len(cells) < MAX_CELLS:
        survivors, daughters = [], []
        for cell in cells:
            if rng.uniform() < DIVISION_PROB_PER_FRAME:
                daughters.extend(_divide_cell(rng, cell, shape_profile, align_prob))
            else:
                survivors.append(cell)
        cells = survivors + daughters

    if len(cells) < MAX_CELLS:
        n_new = int(rng.integers(*NEW_SINGLES_PER_FRAME_RANGE))
        for _ in range(n_new):
            if len(cells) >= MAX_CELLS:
                break
            c = _place_free(rng, image_size, shape_profile, cells, SEPARATION)
            if c is not None:
                cells.append(c)
    return cells


def _draw_capsule(canvas, center, a, b, angle_deg, color):
    """Real rods are a capsule/stadium shape (parallel sides, rounded end-
    caps) -- NOT an ellipse, which continuously tapers and bulges in the
    middle. cv2 has no native capsule primitive, so this composites two
    filled circles (radius b, the half-width) at each end-cap center plus a
    filled rotated rectangle (width 2b) connecting them, exactly matching a
    real rod's silhouette for a given length (2a) and width (2b)."""
    theta = np.radians(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    half_body = max(0.0, a - b)   # distance from center to each cap center
    cx, cy = center
    cap1 = (cx + half_body * dx, cy + half_body * dy)
    cap2 = (cx - half_body * dx, cy - half_body * dy)
    r = max(1, int(round(b)))

    cv2.circle(canvas, (int(round(cap1[0])), int(round(cap1[1]))), r, color, -1)
    cv2.circle(canvas, (int(round(cap2[0])), int(round(cap2[1]))), r, color, -1)
    if half_body > 0:
        pdx, pdy = -dy * b, dx * b   # perpendicular offset, length b
        corners = np.array([
            [cap1[0] + pdx, cap1[1] + pdy], [cap1[0] - pdx, cap1[1] - pdy],
            [cap2[0] - pdx, cap2[1] - pdy], [cap2[0] + pdx, cap2[1] + pdy],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [corners], color)


def render_cells(rng, cells, profile, image_size=IMAGE_SIZE):
    phot = profile["photometry"]
    psf_sigma = phot["psf_sigma_px_median"] or 1.0
    snr_db = phot["snr_db_estimate"] if phot["snr_db_estimate"] is not None else 20.0

    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=np.int32)

    for label, cell in enumerate(cells, start=1):
        px, py = cell["pos"]
        a, b = cell["axes"]
        angle = float(cell["angle_deg"])
        _draw_capsule(img, (px, py), a, b, angle, 1.0)
        _draw_capsule(mask, (px, py), a, b, angle, int(label))
    mask = mask.astype(np.uint16)

    blurred = gaussian_filter(img, sigma=psf_sigma)
    peak = blurred.max() if blurred.max() > 0.0 else 1.0
    noise_std = peak / (10.0 ** (snr_db / 20.0))
    noisy = blurred + rng.normal(0.0, noise_std, (image_size, image_size)).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0), mask


def generate_movie(rng, profile, n_frames=FRAMES_PER_MOVIE):
    shape_profile = dict(profile["shape"])
    shape_profile.update(profile["geometry"])   # length_px_mean/width_px_mean -- see _sample_cell_ellipse
    shape_profile["_target_density"] = _target_density(profile)
    align_prob = _orientation_align_prob(profile)

    images = np.zeros((n_frames, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    masks = np.zeros((n_frames, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    lineage = {}

    cells = init_population(rng, IMAGE_SIZE, shape_profile)
    for t in range(n_frames):
        if t > 0:
            cells = step_population(rng, cells, IMAGE_SIZE, shape_profile, align_prob)
        img, mask = render_cells(rng, cells, profile)
        images[t] = img
        masks[t] = mask
        lineage[str(t)] = {str(int(lbl)): int(lbl) for lbl in np.unique(mask) if lbl != 0}
    return images, masks, lineage, align_prob


def save_movie(output_dir, movie_idx, images, masks, lineage):
    movie_dir = Path(output_dir) / f"movie_{movie_idx:03d}"
    movie_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(movie_dir / "images.npz", data=images)
    np.savez_compressed(movie_dir / "masks.npz", data=masks)
    with open(movie_dir / "lineage.json", "w") as f:
        json.dump(lineage, f)


def movie_is_complete(output_dir, movie_idx):
    movie_dir = Path(output_dir) / f"movie_{movie_idx:03d}"
    return all((movie_dir / fname).exists() for fname in ("images.npz", "masks.npz", "lineage.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=["tb", "pa", "ecoli", "mabs"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_movies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.species)

    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(args.n_movies)

    for i in range(args.n_movies):
        if movie_is_complete(output_dir, i):
            print(f"movie {i+1:03d}/{args.n_movies} skipped (already complete)")
            continue
        rng = np.random.default_rng(child_seeds[i])
        images, masks, lineage, align_prob = generate_movie(rng, profile)
        save_movie(output_dir, i, images, masks, lineage)
        cells_per_frame = [int(np.count_nonzero(np.unique(masks[t]))) for t in range(masks.shape[0])]
        print(f"movie {i+1:03d}/{args.n_movies} done (align_prob={align_prob:.2f}, "
              f"cells: {cells_per_frame[0]} -> {cells_per_frame[-1]})")

    print(f"\nDone. movie_* dirs in {output_dir}/")


if __name__ == "__main__":
    main()
