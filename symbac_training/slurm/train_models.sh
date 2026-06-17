#!/usr/bin/env bash
# SLURM job — train AssignmentMLP and DivisionMLP on sciCORE.
# Submit from inside symbac_training/:
#     cd ~/SyMBac_2/symbac_training
#     mkdir -p logs
#     sbatch slurm/train_symbac.sh

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=symbac_train
#SBATCH --output=logs/train_symbac.out
#SBATCH --error=logs/train_symbac.err
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
echo "==> Training AssignmentMLP..."
python "${TRAINING_ROOT}/train_assignment.py" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch_size 4096 \
    --lr 1e-3 \
    --patience 15

echo ""
echo "==> Training DivisionMLP..."
python "${TRAINING_ROOT}/train_division.py" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3 \
    --patience 15

echo ""
echo "==> Both models saved to ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
