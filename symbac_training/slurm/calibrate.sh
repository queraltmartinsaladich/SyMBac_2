#!/usr/bin/env bash
# SLURM job — run calibrate_from_real.py for one species. CPU-only, but
# real-data crop loops (TB: 2400 crops) can run several minutes.
#
# Submit from symbac_training/:
#     sbatch slurm/calibrate.sh pa

#SBATCH --job-name=calibrate
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail

SPECIES="${1:?Usage: sbatch calibrate.sh SPECIES}"

VENV="/scicore/home/boeluc00/martin0088/venv"
TRAINING_ROOT="/scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training"

mkdir -p logs
source "${VENV}/bin/activate"
cd "${TRAINING_ROOT}"

python -u calibrate_from_real.py --species "${SPECIES}" --output "calibration_profiles/${SPECIES}.json"
