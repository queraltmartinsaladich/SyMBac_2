#!/usr/bin/env bash
# SLURM job — generate a calibrated procedural coc (S. aureus) movie pool
# (generate_coc_dataset.py). CPU-only.
#
# Submit from symbac_training/:
#     sbatch slurm/generate_coc_dataset.sh 5 v2   # n_movies output_tag

#SBATCH --job-name=generate_coc_dataset
#SBATCH --output=logs/generate_coc_dataset_%j.out
#SBATCH --error=logs/generate_coc_dataset_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail

N_MOVIES="${1:?Usage: sbatch generate_coc_dataset.sh N_MOVIES OUTPUT_TAG}"
OUTPUT_TAG="${2:?Usage: sbatch generate_coc_dataset.sh N_MOVIES OUTPUT_TAG}"

VENV="$HOME/venv"
TRAINING_ROOT="$HOME/SyMBac_2/symbac_training"
OUTPUT_DIR="$HOME/data/synth_coc_${OUTPUT_TAG}"

mkdir -p logs "${OUTPUT_DIR}"
source "${VENV}/bin/activate"

echo "==> N movies : ${N_MOVIES}"
echo "==> Output   : ${OUTPUT_DIR}"

python -u "${TRAINING_ROOT}/generate_coc_dataset.py" \
    --output_dir "${OUTPUT_DIR}" \
    --n_movies   "${N_MOVIES}" \
    --seed       1

echo "Done."
