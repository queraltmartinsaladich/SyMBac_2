#!/usr/bin/env bash
# SLURM job — render generate_coc_dataset.py output for visual QA. CPU-only.
#
# Submit from symbac_training/:
#     sbatch slurm/check_coc_procedural.sh data/synth_coc_v2

#SBATCH --job-name=check_coc_procedural
#SBATCH --output=logs/check_coc_procedural_%j.out
#SBATCH --error=logs/check_coc_procedural_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail

POOL_DIR="${1:?Usage: sbatch check_coc_procedural.sh POOL_DIR}"

VENV="$HOME/venv"
TRAINING_ROOT="$HOME/SyMBac_2/symbac_training"
OUTPUT_DIR="$HOME/data/coc_procedural_check"

mkdir -p logs
source "${VENV}/bin/activate"

python -u "${TRAINING_ROOT}/check_coc_procedural.py" \
    --pool_dir   "${POOL_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done -> ${OUTPUT_DIR}"
