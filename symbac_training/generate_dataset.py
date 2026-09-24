#!/usr/bin/env python3
"""Generate synthetic bacterial microscopy dataset using SyMBac_2."""

import argparse
import json
import numpy as np
from pathlib import Path


GEOMETRY_GROUPS = [
    "open_well_sparse",
    "open_well_dense",
    "trench",
    "box",
]


def load_species_profile(species: str | None) -> dict | None:
    """Load a calibration profile written by calibrate_from_real.py, or None
    for the default (uncalibrated, blind-random) behavior."""
    if species is None:
        return None
    profile_path = Path(__file__).parent / "calibration_profiles" / f"{species}.json"
    with open(profile_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_params(group_name: str, rng: np.random.Generator, profile: dict | None = None) -> dict:
    """Sample random simulation parameters for a given geometry group.

    If `profile` (a calibration_profiles/*.json dict) is given, geometry and
    photometry are sampled around the real-measured distribution instead of
    the blind default ranges -- see calibrate_from_real.py.
    """
    cell_width_mean  = float(rng.uniform(16, 26))   # full diameter, arbitrary sim units
    mean_div_time    = int(rng.integers(10, 31))     # faster: [10,30] frames
    n_initial_cells  = int(rng.integers(8, 26))      # pre-populate: [8,25] cells

    if profile is None:
        cell_length_mean = float(rng.uniform(80, 150))
        psf_sigma        = float(rng.uniform(0.8, 2.5))
        snr_db           = float(rng.uniform(15, 35))
        pixels_per_unit  = 12.0 / cell_width_mean        # target ~12 px cell diameter
        image_size       = [256, 256]
    else:
        geom, phot = profile["geometry"], profile["photometry"]
        # Only the width_px target and the length/width ratio matter for
        # final rendered geometry -- pixels_per_unit cancels cell_width_mean
        # out exactly (rendered_width_px = cell_width_mean * pixels_per_unit).
        aspect_ratio = max(1.1, float(rng.normal(geom["aspect_ratio_mean"], geom["aspect_ratio_std"])))
        cell_length_mean = cell_width_mean * aspect_ratio
        target_width_px = max(4.0, float(rng.normal(geom["width_px_mean"], geom["width_px_std"])))
        pixels_per_unit = target_width_px / cell_width_mean

        snr_center = phot["snr_db_estimate"] if phot["snr_db_estimate"] is not None else 25.0
        snr_db = float(rng.uniform(snr_center * 0.85, snr_center * 1.15))
        sigma_center = phot["psf_sigma_px_median"] if phot["psf_sigma_px_median"] is not None else 1.5
        psf_sigma = float(rng.uniform(sigma_center * 0.85, sigma_center * 1.15))

        image_size = [512, 512]   # match real crop size -- see synth_to_coco.py

    IW, IH = image_size

    base = dict(
        cell_length_mean   = cell_length_mean,
        cell_width_mean    = cell_width_mean,
        mean_division_time = mean_div_time,
        n_initial_cells    = n_initial_cells,
        warmup_steps       = 30,
        psf_sigma          = psf_sigma,
        snr_db             = snr_db,
        pixels_per_unit    = pixels_per_unit,
        image_size         = image_size,
        granularity        = 4,
        length_variation   = float(rng.uniform(0.1, 0.3)),
        max_bend_angle     = 0.005,
        stiffness          = 300_000,
        pivot_stiffness    = 5000.0,
        noise_strength     = float(rng.uniform(0.01, 0.05)),
        start_angle        = float(np.pi / 2),
        geometry_group     = group_name,
    )

    if group_name == "open_well_sparse":
        base.update(
            geometry  = "open",
            n_frames  = int(rng.integers(40, 71)),
            origin_px = [IW // 2, IH // 2],
        )
    elif group_name == "open_well_dense":
        base.update(
            geometry  = "open",
            n_frames  = int(rng.integers(70, 101)),
            origin_px = [IW // 2, IH // 2],
        )
    elif group_name == "trench":
        base.update(
            geometry      = "trench",
            n_frames      = int(rng.integers(50, 91)),
            trench_width  = cell_width_mean * float(rng.uniform(1.3, 1.8)),
            trench_length = cell_length_mean * float(rng.uniform(4.0, 7.0)),
            origin_px     = [IW // 2, int(IH * 230 / 256)],   # bottom-centre; sim y increases upward
        )
    elif group_name == "box":
        base.update(
            geometry   = "box",
            n_frames   = int(rng.integers(50, 91)),
            box_width  = cell_width_mean * float(rng.uniform(4.0, 8.0)),
            box_height = cell_length_mean * float(rng.uniform(3.0, 6.0)),
            origin_px  = [IW // 2, IH // 2],
        )

    return base


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def generate_movie(params: dict):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import pymunk
    from symbac.simulation.simulator import Simulator
    from symbac.simulation.config import CellConfig, PhysicsConfig
    from symbac.simulation.microfluidic_geometry import trench_creator, box_creator

    n_frames   = params['n_frames']
    H, W       = params['image_size']
    psf_sigma  = params.get('psf_sigma', 1.5)
    snr_db     = params.get('snr_db', 20.0)
    cell_len   = params['cell_length_mean']
    cell_diam  = params['cell_width_mean']
    div_time   = params['mean_division_time']
    geometry   = params.get('geometry', 'open')
    scale      = params.get('pixels_per_unit', 1.0)
    rng        = np.random.default_rng(params.get('seed', 0))

    granularity      = params.get('granularity', 4)
    seg_radius       = cell_diam / 2.0
    joint_dist       = seg_radius / granularity      # JOINT_DISTANCE = SEGMENT_RADIUS / GRANULARITY
    seed_segs        = max(5, int(cell_len / (2.0 * joint_dist)))
    start_len        = seed_segs * joint_dist
    dt               = PhysicsConfig.DT              # ≈ 1/60
    growth_rate      = (cell_len - start_len) / (div_time * dt)
    n_initial_cells  = max(1, params.get('n_initial_cells', 1))
    warmup_steps     = params.get('warmup_steps', 30)

    # --- Starting positions for all initial cells ---
    # Spacing: enough to avoid hard overlaps; warmup physics will separate them further
    dx = cell_diam * 2.5                       # horizontal spacing (sim units)
    dy = (start_len + cell_diam) * 1.2         # vertical spacing (sim units)

    if geometry == 'trench':
        # Single column stacked upward inside the trench (real physical confinement)
        grid_pos = [(0.0, k * dy) for k in range(n_initial_cells)]
    elif geometry == 'box':
        # 2-D grid centred at origin (real physical confinement -- a microfluidic box)
        n_cols = max(1, int(np.ceil(np.sqrt(n_initial_cells))))
        n_rows = int(np.ceil(n_initial_cells / n_cols))
        grid_pos = []
        for k in range(n_initial_cells):
            row = k // n_cols
            col = k % n_cols
            grid_pos.append((
                (col - (n_cols - 1) / 2.0) * dx,
                (row - (n_rows - 1) / 2.0) * dy,
            ))
    else:
        # Open field-of-view: real crops (see calibrate_from_real.py's source
        # images) show cells scattered as separate individuals/small groups
        # across the WHOLE frame, not one growing central microcolony -- a
        # tight grid-at-origin seed (the old behavior here) only ever
        # populates the canvas center. Scatter seeds across the visible
        # canvas instead, in sim-unit coordinates (inverting the same
        # px = ox + sx*scale, py = oy - sy*scale mapping _rasterise uses).
        _ox, _oy = params.get('origin_px', [W // 2, H // 2])
        _margin_px = max(0.04 * min(H, W), 1.0 * cell_len * scale)   # cell_len is in sim units -- convert to px
        sx_lo, sx_hi = (_margin_px - _ox) / scale, (W - _margin_px - _ox) / scale
        sy_lo, sy_hi = (_oy - (H - _margin_px)) / scale, (_oy - _margin_px) / scale
        min_sep = dx   # reuse the old grid spacing as the minimum center-to-center separation

        grid_pos = []
        for _ in range(n_initial_cells):
            for _attempt in range(200):
                cand = (float(rng.uniform(sx_lo, sx_hi)), float(rng.uniform(sy_lo, sy_hi)))
                if all(np.hypot(cand[0] - p[0], cand[1] - p[1]) > min_sep for p in grid_pos):
                    grid_pos.append(cand)
                    break
            else:
                grid_pos.append((float(rng.uniform(sx_lo, sx_hi)), float(rng.uniform(sy_lo, sy_hi))))

    cell_cfg = CellConfig(
        GRANULARITY           = granularity,
        SEGMENT_RADIUS        = seg_radius,
        SEGMENT_MASS          = params.get('segment_mass', 1.0),
        GROWTH_RATE           = growth_rate,
        BASE_MAX_LENGTH       = cell_len,
        MAX_LENGTH_VARIATION  = params.get('length_variation', 0.2),
        MIN_LENGTH_AFTER_DIVISION = max(4, seed_segs // 2),
        SEED_CELL_SEGMENTS    = seed_segs,
        NOISE_STRENGTH        = params.get('noise_strength', 0.02),
        PIVOT_JOINT_STIFFNESS = params.get('pivot_stiffness', 5000.0),
        ROTARY_LIMIT_JOINT    = True,
        MAX_BEND_ANGLE        = params.get('max_bend_angle', 0.005),
        STIFFNESS             = params.get('stiffness', 300_000),
        START_ANGLE           = params.get('start_angle', np.pi / 2),
        SEPTUM_DURATION       = params.get('septum_duration', div_time * dt * 0.1),
        START_POS             = pymunk.Vec2d(*grid_pos[0]),
    )
    phys_cfg = PhysicsConfig(
        ITERATIONS = params.get('physics_iterations', 100),
        DAMPING    = params.get('damping', 0.5),
    )

    _div_log       = []
    _frame_counter = [0]

    def _on_division(mother, daughter):
        _div_log.append((_frame_counter[0], mother.group_id, daughter.group_id))

    # Create simulator WITHOUT division hook so warmup divisions don't enter lineage
    sim = Simulator(physics_config=phys_cfg, initial_cell_config=cell_cfg)

    def _add_geometry(simulator):
        origin = pymunk.Vec2d(0.0, 0.0)
        if geometry == 'trench':
            trench_creator(
                params.get('trench_width',  cell_diam * 1.5),
                params.get('trench_length', cell_len  * 5.0),
                origin, simulator.space,
            )
        elif geometry == 'box':
            box_creator(
                params.get('box_width',  cell_diam * 6.0),
                params.get('box_height', cell_len  * 4.0),
                origin, simulator.space,
            )

    sim.add_and_run_post_init_hook(_add_geometry)

    # Seed extra cells at their grid positions
    if n_initial_cells > 1:
        from symbac.simulation.simcell import SimCell as _SimCell
        for pos in grid_pos[1:]:
            new_cell = _SimCell(
                space     = sim.space,
                config    = cell_cfg,
                start_pos = pymunk.Vec2d(*pos),
                group_id  = sim.next_group_id,
            )
            sim.colony.add_cell(new_cell)
            sim.next_group_id += 1

    # Warmup: let physics separate overlapping cells; divisions not tracked
    for _ in range(warmup_steps):
        sim.step()

    # Attach division hook only after warmup
    sim.add_post_division_hook(_on_division)

    images  = np.zeros((n_frames, H, W), dtype=np.float32)
    masks   = np.zeros((n_frames, H, W), dtype=np.uint16)
    lineage: dict = {}

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    default_ox = W // 2
    default_oy = int(H * 0.85) if geometry == 'trench' else H // 2
    ox, oy = params.get('origin_px', [default_ox, default_oy])

    def _rasterise(t: int) -> None:
        img  = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.uint16)
        for cell in sim.cells:
            lbl = int(cell.group_id) + 1
            for seg in cell.physics_representation.segments:
                sx = seg.body.position.x
                sy = seg.body.position.y
                px = ox + sx * scale
                py = oy - sy * scale
                r  = seg.shape.radius * scale
                disk = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
                img[disk]  = 1.0
                mask[disk] = lbl
        blurred   = gaussian_filter(img, sigma=psf_sigma)
        peak      = blurred.max() if blurred.max() > 0.0 else 1.0
        noise_std = peak / (10.0 ** (snr_db / 20.0))
        noisy     = blurred + rng.normal(0.0, noise_std, (H, W)).astype(np.float32)
        images[t] = np.clip(noisy, 0.0, 1.0)
        masks[t]  = mask

    def _record_lineage(t: int, divs: list) -> None:
        daughter_to_mother = {d: m for _, m, d in divs}
        entry: dict = {}
        for cell in sim.cells:
            lbl = int(cell.group_id) + 1
            if cell.group_id in daughter_to_mother:
                entry[str(lbl)] = daughter_to_mother[cell.group_id] + 1
            else:
                entry[str(lbl)] = lbl
        lineage[str(t)] = entry

    _rasterise(0)
    for t in range(1, n_frames):
        _frame_counter[0] = t
        log_before = len(_div_log)
        sim.step()
        _rasterise(t)
        _record_lineage(t, _div_log[log_before:])

    return images, masks, lineage


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_movie(output_dir, movie_idx, images, masks, lineage, params):
    movie_dir = Path(output_dir) / f"movie_{movie_idx:03d}"
    movie_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(movie_dir / "images.npz", data=images)
    np.savez_compressed(movie_dir / "masks.npz",  data=masks)

    with open(movie_dir / "lineage.json", "w") as f:
        json.dump(lineage, f)


def movie_is_complete(output_dir: Path, movie_idx: int) -> bool:
    movie_dir = output_dir / f"movie_{movie_idx:03d}"
    return all(
        (movie_dir / fname).exists()
        for fname in ("images.npz", "masks.npz", "lineage.json")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bacterial microscopy dataset with SyMBac_2"
    )
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for dataset output")
    parser.add_argument("--n_movies",   type=int, default=100,
                        help="Total number of movies to generate (default: 100)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Master random seed (default: 42)")
    parser.add_argument("--species",    choices=["tb", "pa", "ecoli", "mabs"], default=None,
                        help="Calibrate geometry/photometry against a real-data "
                             "profile from calibration_profiles/ (default: "
                             "uncalibrated blind-random ranges)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_movies = args.n_movies
    profile = load_species_profile(args.species)

    # Derive one deterministic child seed per movie, independent of n_movies,
    # so that resuming a partial run gives identical params.
    ss          = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(n_movies)

    movies_per_group = max(1, n_movies // len(GEOMETRY_GROUPS))

    all_params = []
    for i in range(n_movies):
        group_idx  = min(i // movies_per_group, len(GEOMETRY_GROUPS) - 1)
        group_name = GEOMETRY_GROUPS[group_idx]
        movie_rng  = np.random.default_rng(child_seeds[i])
        p = sample_params(group_name, movie_rng, profile=profile)
        p['seed'] = int(child_seeds[i].entropy)   # store for reproducibility
        all_params.append(p)

    MAX_RETRIES = 3   # a dense/small-cell packing can occasionally exceed a fixed-size
                       # internal array in SyMBac's division_manager.py (IndexError on
                       # _mother_septum_segments) -- resample and retry rather than abort.

    generation_log = {}

    for i, p in enumerate(all_params):
        tag = f"movie_{i:03d}"

        if movie_is_complete(output_dir, i):
            print(f"movie {i+1:03d}/{n_movies} skipped "
                  f"(already complete, geometry={p['geometry_group']})")
            generation_log[tag] = p
            continue

        retry_seed_seq = child_seeds[i]
        images = masks = lineage = None
        for attempt in range(MAX_RETRIES):
            try:
                images, masks, lineage = generate_movie(p)
                break
            except IndexError as e:
                print(f"movie {i+1:03d}/{n_movies} attempt {attempt+1}/{MAX_RETRIES} "
                      f"failed ({e}); resampling and retrying")
                retry_seed_seq = retry_seed_seq.spawn(1)[0]
                retry_rng = np.random.default_rng(retry_seed_seq)
                p = sample_params(p['geometry_group'], retry_rng, profile=profile)
        else:
            print(f"movie {i+1:03d}/{n_movies} FAILED after {MAX_RETRIES} attempts "
                  f"(geometry={p['geometry_group']}), skipping")
            generation_log[tag] = {**p, "failed": True}
            continue

        save_movie(output_dir, i, images, masks, lineage, p)

        cells_per_frame = [
            int(np.count_nonzero(np.unique(masks[t])))   # count non-zero labels (background=0 excluded)
            for t in range(masks.shape[0])
        ]
        cells_avg = int(round(np.mean(cells_per_frame)))

        print(f"movie {i+1:03d}/{n_movies} done "
              f"(geometry={p['geometry_group']}, "
              f"frames={p['n_frames']}, "
              f"cells_avg={cells_avg})")

        generation_log[tag] = p

    with open(output_dir / "generation_params.json", "w") as f:
        json.dump(generation_log, f, indent=2, default=str)

    print(f"\nDone. generation_params.json saved to {output_dir}/")


if __name__ == "__main__":
    main()
