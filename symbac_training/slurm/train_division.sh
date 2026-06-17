#!/usr/bin/env bash
# SLURM job — train DivisionMLP only on sciCORE.
# Submit from inside symbac_training/:
#     cd ~/SyMBac_2/symbac_training
#     sbatch slurm/train_division.sh

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=symbac_division
#SBATCH --output=logs/train_division.out
#SBATCH --error=logs/train_division.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TRAINING_ROOT="/scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training"
VENV="/scicore/home/boeluc00/martin0088/venv"
DATASET_DIR="${TRAINING_ROOT}/dataset"
OUTPUT_DIR="${TRAINING_ROOT}/weights"
# ── END CONFIG ─────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p logs "${OUTPUT_DIR}"

echo "==> Node      : $(hostname)"
echo "==> GPU       : ${CUDA_VISIBLE_DEVICES:-auto}"
echo "==> Dataset   : ${DATASET_DIR}"
echo "==> Output    : ${OUTPUT_DIR}"

module load CUDA/12.1.0 2>/dev/null || true

source "${VENV}/bin/activate"

echo ""
echo "==> Training DivisionMLP..."
python "${TRAINING_ROOT}/train_division.py" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch_size 512 \
    --lr 5e-4 \
    --patience 20

echo ""
echo "==> DivisionMLP saved to ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
