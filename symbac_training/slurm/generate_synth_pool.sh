#!/usr/bin/env bash
# SLURM job — generate a calibrated SyMBac synthetic movie pool for one
# species (TB or PA), to be converted (synth_to_coco.py) and folded into
# that species' real training set as train-only augmentation. See
# ~/.claude/plans/playful-popping-allen.md for the full pilot plan.
#
# CPU-only (pymunk physics + numpy/scipy rendering, no GPU) -- capped at the
# "scicore" partition's 6h qos, the longest non-GPU qos on this cluster.
# generate_dataset.py resumes automatically (skips already-complete movies),
# so if this times out before finishing --n_movies, just resubmit.
#
# Submit from symbac_training/:
#     sbatch slurm/generate_synth_pool.sh tb   # or: pa

#SBATCH --job-name=generate_synth_pool
#SBATCH --output=logs/generate_synth_%j.out
#SBATCH --error=logs/generate_synth_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scicore
#SBATCH --qos=6hours

set -euo pipefail

SPECIES="${1:?Usage: sbatch generate_synth_pool.sh tb-or-pa}"
case "$SPECIES" in
  # Sized against calibrate_from_real.py's own real-train crop counts (TB:
  # 2400, PA: 118, ecoli: 635, mabs: 699), targeting a synthetic:real ratio --
  # subset_synth_pool.py then samples the exact ratio needed downstream.
  # TB capped at ~2x (not 3x): full-scale timing came in at ~12min/movie,
  # not the ~15s/movie a small 4-movie smoke test suggested (dense/box
  # groups + crash-retry overhead dominate) -- 120 movies would need several
  # more 6h job resubmissions. 75 movies (~4800 frames at the observed
  # ~64 frames/movie average) reaches ~2x within one more resumed run.
  tb)    N_MOVIES=75 ;;
  pa)    N_MOVIES=20 ;;   # PA movies are much heavier (larger cells, more divisions/segments)
  # ecoli/mabs raised 20->35 (real=635/699, 3x targets=1905/2097; observed
  # ~65 frames/movie from the first 8-movie batch) -- resumable, so this
  # picks up from the existing 8 movies rather than restarting.
  ecoli) N_MOVIES=35 ;;
  mabs)  N_MOVIES=35 ;;
  *) echo "Unknown species: $SPECIES (expected tb, pa, ecoli, or mabs)"; exit 1 ;;
esac

VENV="$HOME/venv"
TRAINING_ROOT="$HOME/SyMBac_2/symbac_training"
OUTPUT_DIR="$HOME/data/synth_${SPECIES}_pool"

mkdir -p logs "${OUTPUT_DIR}"
source "${VENV}/bin/activate"

echo "==> Node     : $(hostname)"
echo "==> Species  : ${SPECIES}"
echo "==> N movies : ${N_MOVIES}"
echo "==> Output   : ${OUTPUT_DIR}"

python -u "${TRAINING_ROOT}/generate_dataset.py" \
    --output_dir "${OUTPUT_DIR}" \
    --n_movies   "${N_MOVIES}" \
    --species    "${SPECIES}" \
    --seed       1

echo "Done. generation_params.json + movie_* dirs in ${OUTPUT_DIR}"
