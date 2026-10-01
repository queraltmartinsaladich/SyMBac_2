#!/usr/bin/env python3
"""Generate synthetic coccus (S. aureus) crops via procedural cluster
growth -- NOT SyMBac's physics engine, which only models rod-shaped
elongation/division (see generate_dataset.py). Cocci divide by alternating
across ~3 roughly-perpendicular planes, producing tetrad/octad "grape
cluster" arrangements -- approximated by a recursive split where each
division axis rotates ~90 degrees (+jitter) from its parent's axis.
Daughters are placed exactly tangent (center separation = r_a + r_b, each
cell's own radius) so cells touch but never overlap.

Each "movie" is a genuine time series, not independent per-frame samples:
frame 0 starts with scattered, separate single cells (no pre-formed
clusters); each subsequent frame, existing cells independently have a
chance to divide in place (growing into clusters over time, per real
division mechanics) and a few fresh singles enter the field. Output uses
the same movie_NNN/{images.npz,masks.npz,lineage.json} shape as
generate_dataset.py (lineage.json here is a trivial per-frame identity map,
unused by synth_to_coco.py) so the existing synth_to_coco.py
--format mtb_crops converter works unchanged on each frame independently.

Usage:
  python generate_coc_dataset.py --output_dir data/synth_coc_pool --n_movies 20 --seed 1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

FRAMES_PER_MOVIE = 20
IMAGE_SIZE = 512          # matches real coc crop size

N_DIVISIONS_WEIGHTS = {0: 0.35, 1: 0.30, 2: 0.20, 3: 0.15}   # for a division event's own sub-splits (rare, usually 1 split -> 2 cells)
# Cocci are famously size-uniform within a population -- the raw annotation
# area_std (~55% of mean) mixes in measurement/segmentation noise and
# cross-image variability, not true cell-to-cell size difference. Cap the
# per-cell size jitter to a tight coefficient of variation instead of using
# that std directly.
CELL_SIZE_CV = 0.10

# Temporal growth parameters. IMPORTANT: the 512x512 frame is a CROP, not a
# full field of view -- calibrate_from_real.py's density.cells_per_crop_mean
# (~57.6 for coc) is the real per-crop target, not a floor to exceed. An
# earlier version of this file grew density to an arbitrary MAX_CELLS=260
# cap (~4.5x real) based on "looks like good progression" rather than the
# measured target -- fixed: final-frame density now approaches the real
# mean, not blows past it.
N_INITIAL_RANGE = (28, 40)          # separate singles at frame 0 (~0.6x real target)
DIVISION_PROB_PER_FRAME = 0.045     # each existing cell, each frame -- clustering forms a bit faster
NEW_SINGLES_PER_FRAME_RANGE = (0, 1)
MAX_CELLS = 75                      # soft safety cap, not a target -- real mean is ~57.6
SINGLE_SEPARATION = 1.4             # singles start visibly apart, not touching (>1 = gap, in radii)


def load_profile(species="coc"):
    profile_path = Path(__file__).parent / "calibration_profiles" / f"{species}.json"
    with open(profile_path) as f:
        return json.load(f)


def _sample_cell_ellipse(rng, shape_profile):
    """Semi-major/minor axes (px). Previously derived from real area +
    eccentricity (area = pi*a*b) -- switched to sampling length_px/width_px
    directly (same approach as generate_rod_dataset.py's rod species)
    because the area+eccentricity route doesn't reproduce the real
    calibration's own width/length means: even before any rasterization,
    that formula averaged ~10-16% over real length_px_mean/width_px_mean
    (real cocci deviate from a clean geometric model enough that area,
    eccentricity, and width/length don't all agree simultaneously -- same
    issue documented for PA elsewhere in this project). Sampling length and
    width directly targets the stat that actually gets measured and
    compared against real data."""
    length_mean = shape_profile["length_px_mean"]
    width_mean = shape_profile["width_px_mean"]
    length = float(np.clip(rng.normal(length_mean, length_mean * CELL_SIZE_CV),
                            length_mean * 0.7, length_mean * 1.3))
    width = float(np.clip(rng.normal(width_mean, width_mean * CELL_SIZE_CV),
                           width_mean * 0.7, width_mean * 1.3))
    width = min(width, length)   # semi-minor can't exceed semi-major
    a, b = length / 2.0, width / 2.0
    # Rasterization measurement bias correction -- see generate_rod_dataset.py's
    # _calibrate_rasterization_bias for the full rationale: a small ellipse
    # drawn with cv2.ellipse's integer-rounded axes, then measured back via
    # cv2.minAreaRect the same way both check_coc_procedural.py and the real
    # calibration do, systematically over/under-reads relative to the (a, b)
    # actually used to draw it. Pre-divide by the measured bias so the
    # MEASURED output lands on the real calibration target.
    a /= shape_profile.get("_length_bias_correction", 1.0)
    b /= shape_profile.get("_width_bias_correction", 1.0)
    return a, b


def _calibrate_rasterization_bias(rng, shape_profile, n_measure=300, n_iters=5):
    """Coc's own version of generate_rod_dataset.py's bias calibrator, using
    cv2.ellipse instead of the capsule renderer. Same fixed-point iteration
    (the two axis biases are coupled through eccentricity, same as rod's
    width/length coupling through the a/b ratio) -- but measures bias over
    the ACTUAL sampled (a, b) population each round (varying size and
    angle), not at one representative mean-sized shape: cocci sit near
    aspect ratio 1 (mildly elongated, not rod-like), where bias-at-the-mean
    turned out not to represent the population average well enough to
    reach a 2% match on width specifically -- averaging the real per-sample
    bias directly is more robust to that nonlinearity."""
    length_corr, width_corr = 1.0, 1.0
    for _ in range(n_iters):
        probe_profile = dict(shape_profile)
        probe_profile["_length_bias_correction"] = length_corr
        probe_profile["_width_bias_correction"] = width_corr
        len_meas, wid_meas, a_used, b_used = [], [], [], []
        for _ in range(n_measure):
            a, b = _sample_cell_ellipse(rng, probe_profile)
            angle = rng.uniform(0, 180)
            canvas_size = int(4 * a) + 40
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.int32)
            c = canvas_size // 2
            axes = (max(1, int(round(a))), max(1, int(round(b))))
            cv2.ellipse(canvas, (c, c), axes, angle, 0.0, 360.0, 1, -1)
            cm = (canvas == 1).astype(np.uint8)
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
            (_, _), (w, h), _ = cv2.minAreaRect(contour)
            len_meas.append(max(w, h))
            wid_meas.append(min(w, h))
            a_used.append(a)
            b_used.append(b)

        length_corr = float(np.mean(len_meas)) / (2.0 * np.mean(a_used)) if len_meas else length_corr
        width_corr = float(np.mean(wid_meas)) / (2.0 * np.mean(b_used)) if wid_meas else width_corr
    return length_corr, width_corr


def _cell_radius(cell):
    return sum(cell["axes"]) / 2.0


def _new_cell(rng, shape_profile, pos, split_axis):
    a, b = _sample_cell_ellipse(rng, shape_profile)
    return {"pos": pos, "axes": (a, b), "angle_deg": float(rng.uniform(0, 180)), "split_axis": split_axis}


def _place_free(rng, image_size, shape_profile, existing_cells, sep_factor, max_attempts=150):
    """Sample a new cell and a position that doesn't collide with any
    existing cell (separation = sep_factor * (r_new + r_existing))."""
    candidate = _new_cell(rng, shape_profile, (0.0, 0.0), float(rng.uniform(0, np.pi)))
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
    return None   # gave up -- canvas too crowded, skip this one


def _divide_cell(rng, cell, shape_profile):
    """Splits one cell into 2 tangent daughters at its position, rotating
    the division plane ~90 degrees from the parent's own split axis (real
    cocci alternate division planes -> tetrad/octad clusters)."""
    new_axis = cell["split_axis"] + np.pi / 2 + float(rng.normal(0, 0.3))
    dir_x, dir_y = np.cos(new_axis), np.sin(new_axis)
    daughters_shape = [_sample_cell_ellipse(rng, shape_profile) for _ in range(2)]
    radii = [sum(ax) / 2.0 for ax in daughters_shape]

    out = []
    for (a, b), sign, r_self in zip(daughters_shape, (1.0, -1.0), radii):
        # each daughter sits r_self from the split point -> center separation = r_a + r_b (tangent)
        out.append({
            "pos": (cell["pos"][0] + sign * r_self * dir_x, cell["pos"][1] + sign * r_self * dir_y),
            "axes": (a, b),
            "angle_deg": float(np.degrees(new_axis)) + float(rng.normal(0, 15)),
            "split_axis": new_axis,
        })
    return out


def init_population(rng, image_size, shape_profile):
    n_initial = int(rng.integers(*N_INITIAL_RANGE))
    cells = []
    for _ in range(n_initial):
        c = _place_free(rng, image_size, shape_profile, cells, SINGLE_SEPARATION)
        if c is not None:
            cells.append(c)
    return cells


def step_population(rng, cells, image_size, shape_profile):
    if len(cells) < MAX_CELLS:
        survivors = []
        daughters = []
        for cell in cells:
            if rng.uniform() < DIVISION_PROB_PER_FRAME:
                daughters.extend(_divide_cell(rng, cell, shape_profile))
            else:
                survivors.append(cell)
        cells = survivors + daughters

    if len(cells) < MAX_CELLS:
        n_new = int(rng.integers(*NEW_SINGLES_PER_FRAME_RANGE))
        for _ in range(n_new):
            if len(cells) >= MAX_CELLS:
                break
            c = _place_free(rng, image_size, shape_profile, cells, SINGLE_SEPARATION)
            if c is not None:
                cells.append(c)
    return cells


RIM_BOOST_SIGMA = 4.0   # rim brightness = cell_mean + this many background_std above fill -- see generate_rod_dataset.py's render_cells for the same fix, same rationale

def render_cells(rng, cells, profile, image_size=IMAGE_SIZE):
    phot = profile["photometry"]
    psf_sigma = phot["psf_sigma_px_median"] or 1.0
    background_mean = phot.get("background_mean", 0.1)
    background_std = phot.get("background_std", 0.02)
    cell_mean = phot.get("cell_mean", 1.0)
    rim_color = float(np.clip(cell_mean + RIM_BOOST_SIGMA * background_std, 0.0, 1.0))

    # Same fix as generate_rod_dataset.py's render_cells: a black canvas with
    # cells at full white looked like a different imaging modality entirely
    # from real phase-contrast/brightfield crops (mid-gray background, cells
    # only slightly brighter). Render on the real background level with a
    # thin bright rim per cell (phase-contrast edge halo a uniform fill can't
    # produce), and draw noise from the real background_std directly instead
    # of a peak-derived guess that assumed the canvas was black.
    img = np.full((image_size, image_size), background_mean, dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=np.int32)   # cv2 needs signed for fillable draw target

    for label, cell in enumerate(cells, start=1):
        px, py = cell["pos"]
        a, b = cell["axes"]
        center = (int(round(px)), int(round(py)))
        axes = (max(1, int(round(a))), max(1, int(round(b))))
        angle = float(cell["angle_deg"])
        cv2.ellipse(img, center, axes, angle, 0.0, 360.0, cell_mean, -1)
        cv2.ellipse(img, center, axes, angle, 0.0, 360.0, rim_color, 1)
        cv2.ellipse(mask, center, axes, angle, 0.0, 360.0, label, -1)
    mask = mask.astype(np.uint16)

    blurred = gaussian_filter(img, sigma=psf_sigma)
    noisy = blurred + rng.normal(0.0, background_std, (image_size, image_size)).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0), mask


def generate_movie(rng, profile, n_frames=FRAMES_PER_MOVIE):
    shape_profile = dict(profile["shape"])
    shape_profile.update(profile["geometry"])   # length_px_mean/width_px_mean -- see _sample_cell_ellipse
    shape_profile["_length_bias_correction"], shape_profile["_width_bias_correction"] = \
        _calibrate_rasterization_bias(rng, shape_profile)
    images = np.zeros((n_frames, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    masks = np.zeros((n_frames, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    lineage = {}

    cells = init_population(rng, IMAGE_SIZE, shape_profile)
    for t in range(n_frames):
        if t > 0:
            cells = step_population(rng, cells, IMAGE_SIZE, shape_profile)
        img, mask = render_cells(rng, cells, profile)
        images[t] = img
        masks[t] = mask
        lineage[str(t)] = {str(int(lbl)): int(lbl) for lbl in np.unique(mask) if lbl != 0}
    return images, masks, lineage


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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_movies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile("coc")

    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(args.n_movies)

    for i in range(args.n_movies):
        if movie_is_complete(output_dir, i):
            print(f"movie {i+1:03d}/{args.n_movies} skipped (already complete)")
            continue
        rng = np.random.default_rng(child_seeds[i])
        images, masks, lineage = generate_movie(rng, profile)
        save_movie(output_dir, i, images, masks, lineage)
        cells_per_frame = [int(np.count_nonzero(np.unique(masks[t]))) for t in range(masks.shape[0])]
        print(f"movie {i+1:03d}/{args.n_movies} done (frames={FRAMES_PER_MOVIE}, "
              f"cells: {cells_per_frame[0]} -> {cells_per_frame[-1]})")

    print(f"\nDone. movie_* dirs in {output_dir}/")


if __name__ == "__main__":
    main()
