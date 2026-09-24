#!/usr/bin/env bash
# SLURM job — render + geometry-check generate_rod_dataset.py output
# against its real calibration target. CPU-only.
#
# Submit from symbac_training/:
#     sbatch slurm/check_rod_procedural.sh tb data/synth_tb_procedural_v1

#SBATCH --job-name=check_rod_procedural
#SBATCH --output=logs/check_rod_procedural_%j.out
#SBATCH --error=logs/check_rod_procedural_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail

SPECIES="${1:?Usage: sbatch check_rod_procedural.sh SPECIES POOL_DIR}"
POOL_DIR="${2:?Usage: sbatch check_rod_procedural.sh SPECIES POOL_DIR}"

VENV="/scicore/home/boeluc00/martin0088/venv"
TRAINING_ROOT="/scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training"
OUTPUT_DIR="/scicore/home/boeluc00/martin0088/data/rod_procedural_check_${SPECIES}"

mkdir -p logs
source "${VENV}/bin/activate"

python -u "${TRAINING_ROOT}/check_rod_procedural.py" \
    --species    "${SPECIES}" \
    --pool_dir   "${POOL_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done -> ${OUTPUT_DIR}"
