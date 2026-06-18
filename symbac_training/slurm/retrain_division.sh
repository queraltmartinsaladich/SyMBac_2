#!/usr/bin/env bash
# Retrain DivisionMLP only — assumes dataset already exists in dataset/.
# Use this when you only want to improve the division classifier without
# touching the already-good AssignmentMLP.
#
# Workflow:
#     sbatch slurm/build_dataset.sh      # (re)build dataset first if needed
#     sbatch slurm/retrain_division.sh   # then retrain division classifier
#
# Submit from inside symbac_training/:
#     cd ~/SyMBac_2/symbac_training
#     mkdir -p logs
#     sbatch slurm/retrain_division.sh

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=retrain_div
#SBATCH --output=logs/retrain_div.out
#SBATCH --error=logs/retrain_div.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
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

# Fail fast if dataset is missing
if [[ ! -f "${DATASET_DIR}/train_assignments.h5" ]]; then
    echo "ERROR: dataset not found at ${DATASET_DIR}."
    echo "       Run sbatch slurm/build_dataset.sh first."
    exit 1
fi

echo "==> Node    : $(hostname)"
echo "==> GPU     : ${CUDA_VISIBLE_DEVICES:-auto}"
echo "==> Dataset : ${DATASET_DIR}"
echo "==> Output  : ${OUTPUT_DIR}"

module load CUDA/12.1.0 2>/dev/null || true

source "${VENV}/bin/activate"

echo ""
echo "==> Training DivisionMLP (FocalLoss, min_precision=0.60)..."
python "${TRAINING_ROOT}/train_division.py" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3 \
    --patience 15 \
    --min_precision 0.60

echo ""
echo "==> DivisionClassifier saved to ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/division_classifier.pt"
echo ""
echo "==> Done: $(date)"
