#!/usr/bin/env bash
# SLURM job — generate a calibrated procedural rod-species movie pool
# (generate_rod_dataset.py -- replaces the pymunk physics engine for
# rod species, see that script's docstring for why). CPU-only.
#
# Submit from symbac_training/:
#     sbatch slurm/generate_rod_dataset.sh tb 5 test_v1   # species n_movies output_tag

#SBATCH --job-name=generate_rod_dataset
#SBATCH --output=logs/generate_rod_dataset_%j.out
#SBATCH --error=logs/generate_rod_dataset_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail

SPECIES="${1:?Usage: sbatch generate_rod_dataset.sh SPECIES N_MOVIES OUTPUT_TAG}"
N_MOVIES="${2:?Usage: sbatch generate_rod_dataset.sh SPECIES N_MOVIES OUTPUT_TAG}"
OUTPUT_TAG="${3:?Usage: sbatch generate_rod_dataset.sh SPECIES N_MOVIES OUTPUT_TAG}"

VENV="/scicore/home/boeluc00/martin0088/venv"
TRAINING_ROOT="/scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training"
OUTPUT_DIR="/scicore/home/boeluc00/martin0088/data/synth_${SPECIES}_${OUTPUT_TAG}"

mkdir -p logs "${OUTPUT_DIR}"
source "${VENV}/bin/activate"

echo "==> Species  : ${SPECIES}"
echo "==> N movies : ${N_MOVIES}"
echo "==> Output   : ${OUTPUT_DIR}"

python -u "${TRAINING_ROOT}/generate_rod_dataset.py" \
    --species    "${SPECIES}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_movies   "${N_MOVIES}" \
    --seed       1

echo "Done."
