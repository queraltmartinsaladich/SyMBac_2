#!/usr/bin/env bash
# SLURM job — convert a generated synthetic movie pool (generate_synth_pool.sh)
# into the COCO shape each species' real dataset builder expects. CPU-only,
# lighter than generation (contour extraction only).
#
# Submit from symbac_training/ AFTER generate_synth_pool.sh has completed:
#     sbatch slurm/convert_synth_pool.sh tb   # or: pa

#SBATCH --job-name=convert_synth_pool
#SBATCH --output=logs/convert_synth_%j.out
#SBATCH --error=logs/convert_synth_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=6hours

set -euo pipefail

SPECIES="${1:?Usage: sbatch convert_synth_pool.sh tb-or-pa-or-ecoli-or-mabs}"
case "$SPECIES" in
  tb)    FORMAT="mtb_crops"  ;;
  pa)    FORMAT="pa_records" ;;
  ecoli) FORMAT="mtb_crops"  ;;   # generic per-crop-json+tiff shape, consumed by build_rod_synthaug.py
  mabs)  FORMAT="mtb_crops"  ;;
  *) echo "Unknown species: $SPECIES (expected tb, pa, ecoli, or mabs)"; exit 1 ;;
esac

VENV="$HOME/venv"
TRAINING_ROOT="$HOME/SyMBac_2/symbac_training"
MOVIES_DIR="$HOME/data/synth_${SPECIES}_pool"
OUTPUT_DIR="$HOME/data/synth_${SPECIES}_coco"

mkdir -p logs
source "${VENV}/bin/activate"

echo "==> Species    : ${SPECIES} (format=${FORMAT})"
echo "==> Movies dir : ${MOVIES_DIR}"
echo "==> Output dir : ${OUTPUT_DIR}"

python -u "${TRAINING_ROOT}/synth_to_coco.py" \
    --movies_dir  "${MOVIES_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --format      "${FORMAT}" \
    --category_id 1

echo "Done -> ${OUTPUT_DIR}"
