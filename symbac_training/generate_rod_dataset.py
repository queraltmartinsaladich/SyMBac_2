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

# _sample_cell_ellipse's area-solve/direct-width blend systematically
# overshoots width for some species (measured on check_rod_procedural.py
# output, e.g. ecoli: synth width_px_mean=11.48 vs real=9.76, ~118%).
# Per-species multiplier applied to the derived half-width as a direct,
# measured correction rather than re-deriving the blend formula per
# species. 1.0 = no correction (TB/PA/mabs untouched, not yet audited
# for the same overshoot).
SPECIES_WIDTH_SCALE = {
    "ecoli": 0.72,   # feedback: synthetic cells read as much too thick --
                      # corrects the ~18% mean overshoot and lands visibly
                      # thinner than the real mean, not just at parity with it
}

# Explicit request to shrink cells well below the real calibration target
# (not a correction -- a deliberate departure from it), applied uniformly
# to both length and width on top of any SPECIES_WIDTH_SCALE correction
# above. 1.0 = no shrink.
SPECIES_SIZE_SCALE = {
    "ecoli": 2.0 / 3.0,   # reduce both width and length by 1/3
    "pa":    2.0 / 3.0,   # reduce both width and length by 1/3
}

# Species using patchy (Gaussian-cluster) spatial placement instead of
# uniform-random -- feedback that PA isn't spatially even, some regions
# sparse, others tightly packed with cells close together. Off by default
# (uniform placement) for species not listed.
SPECIES_CLUSTERED_PLACEMENT = {"pa"}

# Looser minimum-gap requirement for species using clustered placement, so
# cells inside a dense cluster can actually sit close together rather than
# being spread out by the same gap enforced everywhere. Falls back to the
# global SEPARATION for species not listed.
SPECIES_SEPARATION = {"pa": 1.05}

# Per-species (area_weight, width_weight) blend for _sample_cell_ellipse's
# derived half-width. Species not listed keep the original 0.6/0.4 default.
# TB/mabs have no deliberate size override (unlike PA/ecoli's intentional
# shrink), so they're set to sample width directly for a tight match to the
# real calibration mean instead of letting the area-solve pull it down.
SPECIES_WIDTH_BLEND = {
    "tb":   (0.0, 1.0),
    "mabs": (0.0, 1.0),
}

# Feedback: PA and E. coli rods aren't always straight -- some visibly bow
# along their length -- and length was too uniform cell-to-cell. Both
# enabled together for these two species only (not TB/mabs/coc, not
# reported as an issue there).
SPECIES_CURVATURE = {"pa", "ecoli"}
SPECIES_REAL_LENGTH_STD = {"pa", "ecoli"}

# Per-species multipliers on the base curvature/length-std magnitude below.
# Follow-up feedback: PA specifically should curve MORE but vary LESS in
# size than the initial pass (which used the same amount of each for both
# species) -- E. coli's amounts are left at the original 1.0 baseline.
SPECIES_CURVATURE_SCALE = {"pa": 1.8}
SPECIES_LENGTH_STD_SCALE = {"pa": 0.5}


def _sample_curvature_deg(rng, scale=1.0):
    """Most rods render nearly straight; a minority bow noticeably --
    right-skewed (exponential) magnitude capped at a visibly-curved but not
    coiled maximum, random bend direction. scale multiplies both the
    typical magnitude and the cap together (a species curving "more"
    should see both a higher average bend and a higher ceiling, not just
    one or the other)."""
    bend = min(float(rng.exponential(scale=6.0 * scale)), 35.0 * scale)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    return sign * bend


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
    # Length jitter: default is a tight generic CV (CELL_SIZE_CV) around the
    # mean. Species in SPECIES_REAL_LENGTH_STD instead use the real
    # calibrated length_px_std directly -- feedback that cells all looked
    # the same length, which the tight generic CV was doing by construction;
    # real PA/ecoli length_px_std is 66-83% of the mean (not 15%), so this
    # is a real, measured amount of variability, not an exaggeration. Wider
    # clip bounds (mean +/- 3 real std) since the old (0.5, 1.8)*mean bounds
    # would clip most of that wider distribution away.
    if shape_profile.get("_use_real_length_std") and shape_profile.get("length_px_std"):
        length_std = shape_profile["length_px_std"] * shape_profile.get("_length_std_scale", 1.0)
        length = float(np.clip(rng.normal(length_mean, length_std),
                                max(1.0, length_mean - 3 * length_std), length_mean + 3 * length_std))
    else:
        length = float(np.clip(rng.normal(length_mean, length_mean * CELL_SIZE_CV),
                                length_mean * 0.5, length_mean * 1.8))
    area = float(np.clip(rng.normal(area_mean, area_mean * CELL_SIZE_CV),
                          area_mean * 0.5, area_mean * 1.5))
    a = length / 2.0
    b_from_area = _solve_capsule_halfwidth(a, area)
    # Blend toward the directly-measured width_px too: b_from_area alone
    # slightly undershoots real width (small-capsule rasterization measures
    # ~9-12% more area than the continuous formula predicts, so solving
    # purely for area pulls b down further than warranted). Blend weight is
    # per-species tunable (area_weight, width_weight) -- default keeps the
    # original 0.6/0.4 split; species targeting a tight (~2%) width/length
    # match set this closer to (0, 1) to sample width directly, accepting
    # area/solidity drift as the cost (capsules structurally hold more area
    # than a real cell at matched length+width -- can't hit all of
    # length+width+area+solidity at once with one shape family).
    area_w, width_w = shape_profile.get("_width_blend", (0.6, 0.4))
    b = min(area_w * b_from_area + width_w * (width_mean / 2.0), a)
    b *= shape_profile.get("_width_scale", 1.0)
    size_scale = shape_profile.get("_size_scale", 1.0)
    a *= size_scale
    b *= size_scale
    # Rasterization measurement bias correction (see _calibrate_rasterization_bias):
    # a small filled capsule's contour, measured back via cv2.minAreaRect the
    # same way both check_rod_procedural.py and the real-data calibration do,
    # systematically reads ~10% wider and ~7% shorter than the (a, b) actually
    # used to draw it -- a pixel-discretization artifact of drawing/rasterizing
    # small shapes, not a sampling error. Dividing by the measured bias here
    # pre-compensates so the MEASURED output matches the intended target.
    a /= shape_profile.get("_length_bias_correction", 1.0)
    b /= shape_profile.get("_width_bias_correction", 1.0)
    return a, b   # semi-major, semi-minor


def _calibrate_rasterization_bias(rng, shape_profile, n_probe=60, n_measure=80, n_iters=4):
    """Empirically measures how much cv2.minAreaRect over/under-reads this
    species' typical capsule size once rasterized, by drawing and measuring
    test capsules at the species' own representative (a, b) across random
    angles -- self-calibrating rather than a hardcoded per-species constant,
    consistent with this project's calibrate-from-real-data approach.

    The length and width biases aren't independent: correcting b shifts the
    a/b ratio, which shifts the length bias too (and vice versa), so a
    single measurement at the UNcorrected (a, b) doesn't land on a
    self-consistent answer once both corrections are applied together
    (confirmed empirically -- one-shot correction left length ~7% short).
    Fixed-point iterate instead: measure bias at the current best-guess
    corrected (a, b), set the correction to that measured bias, repeat --
    converges in a few rounds since the coupling is a small effect."""
    curved = bool(shape_profile.get("_curvature_enabled"))
    length_corr, width_corr = 1.0, 1.0
    for _ in range(n_iters):
        probe_profile = dict(shape_profile)
        probe_profile["_length_bias_correction"] = length_corr
        probe_profile["_width_bias_correction"] = width_corr

        if curved:
            # Curvature reduces the minAreaRect-measured length relative to
            # the true arc length (a bowed rod's bounding box is shorter
            # than its straight arc), and varies per cell -- like coc's
            # near-circular shapes, a single representative-size probe point
            # doesn't represent the population average here. Measure over
            # the actual sampled (a, b, curvature) population directly.
            len_meas, wid_meas, a_used, b_used = [], [], [], []
            for _ in range(n_measure):
                a, b = _sample_cell_ellipse(rng, probe_profile)
                curvature = _sample_curvature_deg(rng, scale=shape_profile.get("_curvature_scale", 1.0))
                angle = rng.uniform(0, 180)
                canvas_size = int(4 * a) + 40
                canvas = np.zeros((canvas_size, canvas_size), dtype=np.int32)
                c = canvas_size / 2.0
                _draw_curved_rod(canvas, (c, c), a, b, angle, curvature, 1)
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
            continue

        a_list, b_list = [], []
        for _ in range(n_probe):
            a, b = _sample_cell_ellipse(rng, probe_profile)
            a_list.append(a)
            b_list.append(b)
        a_mean, b_mean = float(np.mean(a_list)), float(np.mean(b_list))

        canvas_size = int(4 * a_mean) + 40
        len_meas, wid_meas = [], []
        for _ in range(n_measure):
            angle = rng.uniform(0, 180)
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.int32)
            c = canvas_size / 2.0
            _draw_capsule(canvas, (c, c), a_mean, b_mean, angle, 1)
            cm = (canvas == 1).astype(np.uint8)
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
            (_, _), (w, h), _ = cv2.minAreaRect(contour)
            len_meas.append(max(w, h))
            wid_meas.append(min(w, h))

        # measured/(2*mean) is the bias AT this corrected operating point --
        # since rendered_a = target_a / length_corr, wanting
        # measured_length == target_length works out to length_corr should
        # equal this ratio exactly (not compound with the previous value).
        length_corr = float(np.mean(len_meas)) / (2.0 * a_mean) if len_meas else length_corr
        width_corr = float(np.mean(wid_meas)) / (2.0 * b_mean) if wid_meas else width_corr
    return length_corr, width_corr


def _cell_radius(cell):
    return sum(cell["axes"]) / 2.0


def _new_cell(rng, shape_profile, pos, orientation_deg):
    a, b = _sample_cell_ellipse(rng, shape_profile)
    curvature = (_sample_curvature_deg(rng, scale=shape_profile.get("_curvature_scale", 1.0))
                 if shape_profile.get("_curvature_enabled") else 0.0)
    return {"pos": pos, "axes": (a, b), "angle_deg": float(orientation_deg), "curvature_deg": curvature}


def _sample_new_orientation(rng, parent_angle_deg, align_prob):
    if rng.uniform() < align_prob:
        return parent_angle_deg + float(rng.normal(0, 12))   # inherit, with modest jitter
    return float(rng.uniform(0, 180))                         # independent


def _make_density_field(rng, image_size, n_clusters_range=(3, 6), sigma_range=(60, 140), floor=0.15, grid_res=64):
    """Coarse spatial density field (sum of a few Gaussian blobs, normalized
    to [floor, 1]) used to bias where new cells land -- real PA crops show
    patchy density (some regions sparse, others tightly packed), not the
    spatially-uniform placement a plain rng.uniform(0, image_size) gives.
    floor keeps sparse regions non-empty rather than completely excluded."""
    xs = np.linspace(0, image_size, grid_res)
    ys = np.linspace(0, image_size, grid_res)
    X, Y = np.meshgrid(xs, ys)
    field = np.zeros_like(X)
    for _ in range(int(rng.integers(*n_clusters_range))):
        cx, cy = rng.uniform(0, image_size, size=2)
        sigma = rng.uniform(*sigma_range)
        field += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    field = field / field.max()
    field = floor + (1.0 - floor) * field
    return field, grid_res


def _sample_density_position(rng, density_field, image_size, r_new):
    """Grid-cell-weighted sample from a density field, then uniform jitter
    within the chosen cell -- clusters candidate positions in high-density
    regions while still covering the whole canvas."""
    field, grid_res = density_field
    flat = field.ravel()
    idx = rng.choice(flat.size, p=flat / flat.sum())
    gy, gx = divmod(idx, grid_res)
    cell_w = image_size / grid_res
    x = np.clip(rng.uniform(gx * cell_w, (gx + 1) * cell_w), r_new, image_size - r_new)
    y = np.clip(rng.uniform(gy * cell_w, (gy + 1) * cell_w), r_new, image_size - r_new)
    return float(x), float(y)


def _place_free(rng, image_size, shape_profile, existing_cells, sep_factor, max_attempts=150):
    candidate = _new_cell(rng, shape_profile, (0.0, 0.0), rng.uniform(0, 180))
    r_new = _cell_radius(candidate)
    density_field = shape_profile.get("_density_field")
    for _ in range(max_attempts):
        if density_field is not None:
            x, y = _sample_density_position(rng, density_field, image_size, r_new)
        else:
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
        curvature = (_sample_curvature_deg(rng, scale=shape_profile.get("_curvature_scale", 1.0))
                     if shape_profile.get("_curvature_enabled") else 0.0)
        out.append({
            "pos": (cell["pos"][0] + sign * r_self * dir_x, cell["pos"][1] + sign * r_self * dir_y),
            "axes": (a, b),
            "angle_deg": new_angle_deg,
            "curvature_deg": curvature,
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
        c = _place_free(rng, image_size, shape_profile, cells, shape_profile.get("_separation", SEPARATION))
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
            c = _place_free(rng, image_size, shape_profile, cells, shape_profile.get("_separation", SEPARATION))
            if c is not None:
                cells.append(c)
    return cells


def _draw_capsule(canvas, center, a, b, angle_deg, color, rim_color=None):
    """Real rods are a capsule/stadium shape (parallel sides, rounded end-
    caps) -- NOT an ellipse, which continuously tapers and bulges in the
    middle. cv2 has no native capsule primitive, so this composites two
    filled circles (radius b, the half-width) at each end-cap center plus a
    filled rotated rectangle (width 2b) connecting them, exactly matching a
    real rod's silhouette for a given length (2a) and width (2b).

    Optional rim_color draws a thin bright outline along the capsule
    perimeter after the fill -- real phase-contrast/brightfield crops carry
    a bright edge halo (measured directly: real cell-interior pixels span
    up to the 99th percentile well above the mean interior fill value used
    for `color`, an effect a uniform-fill capsule can't produce on its own).
    Without it, synthetic frames only match real ones under independent
    per-image contrast stretching -- under a shared, real-anchored display
    window they read as much flatter/lower-contrast than real crops."""
    theta = np.radians(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    half_body = max(0.0, a - b)   # distance from center to each cap center
    cx, cy = center
    cap1 = (cx + half_body * dx, cy + half_body * dy)
    cap2 = (cx - half_body * dx, cy - half_body * dy)
    r = max(1, int(round(b)))
    cap1_i = (int(round(cap1[0])), int(round(cap1[1])))
    cap2_i = (int(round(cap2[0])), int(round(cap2[1])))

    cv2.circle(canvas, cap1_i, r, color, -1)
    cv2.circle(canvas, cap2_i, r, color, -1)
    corners = None
    if half_body > 0:
        pdx, pdy = -dy * b, dx * b   # perpendicular offset, length b
        corners = np.array([
            [cap1[0] + pdx, cap1[1] + pdy], [cap1[0] - pdx, cap1[1] - pdy],
            [cap2[0] - pdx, cap2[1] - pdy], [cap2[0] + pdx, cap2[1] + pdy],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [corners], color)

    if rim_color is not None:
        cv2.circle(canvas, cap1_i, r, rim_color, 1)
        cv2.circle(canvas, cap2_i, r, rim_color, 1)
        if corners is not None:
            cv2.polylines(canvas, [corners], isClosed=True, color=rim_color, thickness=1)


def _draw_curved_rod(canvas, center, a, b, angle_deg, curvature_deg, color, rim_color=None):
    """Real rods (PA/E. coli especially, per feedback) aren't always
    perfectly straight -- some visibly bow along their length. Models the
    centerline as a circular arc of length 2a bent through curvature_deg
    total, then stamps overlapping filled circles of radius b along it
    (spacing < b so they merge into one continuous curved body) -- there's
    no cv2 primitive for a bent capsule, so this is the same "build it from
    circles" idea as the straight capsule's end-caps, just applied along
    the whole centerline instead of only at the two ends."""
    if abs(curvature_deg) < 1e-3:
        return _draw_capsule(canvas, center, a, b, angle_deg, color, rim_color=rim_color)

    theta_total = np.radians(curvature_deg)
    length = 2.0 * a
    radius = length / abs(theta_total)
    n_steps = max(8, int(np.ceil(length / max(1.0, b * 0.4))))
    s_vals = np.linspace(-a, a, n_steps)
    sign = 1.0 if theta_total >= 0 else -1.0
    phi = (s_vals / radius) * sign
    local_x = radius * np.sin(phi)
    local_y = radius * (1.0 - np.cos(phi)) * sign

    theta = np.radians(angle_deg)
    cosA, sinA = np.cos(theta), np.sin(theta)
    cx, cy = center
    r = max(1, int(round(b)))
    pts = []
    for lx, ly in zip(local_x, local_y):
        rx, ry = lx * cosA - ly * sinA, lx * sinA + ly * cosA
        pts.append((int(round(cx + rx)), int(round(cy + ry))))

    for p in pts:
        cv2.circle(canvas, p, r, color, -1)
    if rim_color is not None:
        for p in pts:
            cv2.circle(canvas, p, r, rim_color, 1)


RIM_BOOST_SIGMA = 4.0   # rim brightness = cell_mean + this many background_std above fill

def render_cells(rng, cells, profile, image_size=IMAGE_SIZE):
    phot = profile["photometry"]
    psf_sigma = phot["psf_sigma_px_median"] or 1.0
    background_mean = phot.get("background_mean", 0.1)
    background_std = phot.get("background_std", 0.02)
    cell_mean = phot.get("cell_mean", 1.0)
    rim_color = float(np.clip(cell_mean + RIM_BOOST_SIGMA * background_std, 0.0, 1.0))

    # Real crops are phase-contrast/brightfield: a mid-intensity, low-contrast
    # background with cells only slightly brighter (or darker) than it -- NOT
    # a black/fluorescence-style field with cells at full white. Rendering on
    # a zero background with color=1.0 cells made every synthetic frame look
    # like a different imaging modality entirely (near-binary black/white
    # speckle) regardless of how well cell geometry was calibrated.
    img = np.full((image_size, image_size), background_mean, dtype=np.float32)
    mask = np.zeros((image_size, image_size), dtype=np.int32)

    for label, cell in enumerate(cells, start=1):
        px, py = cell["pos"]
        a, b = cell["axes"]
        angle = float(cell["angle_deg"])
        curvature = float(cell.get("curvature_deg", 0.0))
        _draw_curved_rod(img, (px, py), a, b, angle, curvature, cell_mean, rim_color=rim_color)
        _draw_curved_rod(mask, (px, py), a, b, angle, curvature, int(label))
    mask = mask.astype(np.uint16)

    blurred = gaussian_filter(img, sigma=psf_sigma)
    noisy = blurred + rng.normal(0.0, background_std, (image_size, image_size)).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0), mask


def generate_movie(rng, profile, n_frames=FRAMES_PER_MOVIE):
    shape_profile = dict(profile["shape"])
    shape_profile.update(profile["geometry"])   # length_px_mean/width_px_mean -- see _sample_cell_ellipse
    shape_profile["_target_density"] = _target_density(profile)
    shape_profile["_width_scale"] = SPECIES_WIDTH_SCALE.get(profile.get("species"), 1.0)
    shape_profile["_size_scale"] = SPECIES_SIZE_SCALE.get(profile.get("species"), 1.0)
    shape_profile["_separation"] = SPECIES_SEPARATION.get(profile.get("species"), SEPARATION)
    shape_profile["_width_blend"] = SPECIES_WIDTH_BLEND.get(profile.get("species"), (0.6, 0.4))
    shape_profile["_curvature_enabled"] = profile.get("species") in SPECIES_CURVATURE
    shape_profile["_use_real_length_std"] = profile.get("species") in SPECIES_REAL_LENGTH_STD
    shape_profile["_curvature_scale"] = SPECIES_CURVATURE_SCALE.get(profile.get("species"), 1.0)
    shape_profile["_length_std_scale"] = SPECIES_LENGTH_STD_SCALE.get(profile.get("species"), 1.0)
    shape_profile["_length_bias_correction"], shape_profile["_width_bias_correction"] = \
        _calibrate_rasterization_bias(rng, shape_profile)
    if profile.get("species") in SPECIES_CLUSTERED_PLACEMENT:
        shape_profile["_density_field"] = _make_density_field(rng, IMAGE_SIZE)
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
