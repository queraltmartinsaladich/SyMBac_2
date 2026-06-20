#!/usr/bin/env bash
# Build the HDF5 training dataset from SyMBac_2 synthetic movies.
# CPU-only — no GPU needed. Run this first, then submit train_division_only.sh.
#
# Submit from inside symbac_training/:
#     cd ~/SyMBac_2/symbac_training
#     mkdir -p logs
#     sbatch slurm/build_dataset.sh

#SBATCH --job-name=build_dataset
#SBATCH --output=logs/build_dataset.out
#SBATCH --error=logs/build_dataset.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TRAINING_ROOT="/scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training"
VENV="/scicore/home/boeluc00/martin0088/venv"
DATASET_DIR="${TRAINING_ROOT}/dataset"
# ── END CONFIG ─────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p logs "${DATASET_DIR}"

echo "==> Node    : $(hostname)"
echo "==> Dataset : ${DATASET_DIR}"
echo "==> Started : $(date)"

source "${VENV}/bin/activate"

echo ""
echo "==> Building dataset (hard-negative mining active)..."
python "${TRAINING_ROOT}/build_dataset.py" \
    --data_dir   "${TRAINING_ROOT}/synthetic_data" \
    --output_dir "${DATASET_DIR}"

echo ""
echo "==> Done: $(date)"
ls -lh "${DATASET_DIR}/"
