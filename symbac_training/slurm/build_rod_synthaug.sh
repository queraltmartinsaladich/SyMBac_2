#!/usr/bin/env bash
# SLURM job — build a 1x synthetic:real train set for E. coli or M. abscessus
# (see ~/.claude/plans/playful-popping-allen.md). Lower priority than TB/PA
# (real data already flagged "sufficient" for both in the project's own
# fine-tune data audit) -- 1x only, no 2x/3x arm, just checking whether
# synthetic augmentation helps at all when real data isn't the bottleneck.
#
# Prereqs: generate_synth_pool.sh {ecoli,mabs} + convert_synth_pool.sh
# {ecoli,mabs} already ran (data/synth_{species}_coco exists).
#
# Submit from symbac_training/:
#     sbatch slurm/build_rod_synthaug.sh ecoli   # or: mabs

#SBATCH --job-name=build_rod_synthaug
#SBATCH --output=logs/build_rod_synthaug_%j.out
#SBATCH --error=logs/build_rod_synthaug_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=6hours

set -euo pipefail

SPECIES="${1:?Usage: sbatch build_rod_synthaug.sh ecoli-or-mabs}"
case "$SPECIES" in
  ecoli) TRAIN_ANN="instances_train_3class_ecoli_ibsonly.json"
         VAL_ANN="instances_val_3class_ecoli_ibsonly.json"
         N=635 ;;   # real train image count -- 1x target
  mabs)  TRAIN_ANN="instances_train_2class_m_abscessus.json"
         VAL_ANN="instances_val_2class_m_abscessus.json"
         N=699 ;;
  *) echo "Unknown species: $SPECIES (expected ecoli or mabs)"; exit 1 ;;
esac

VENV="$HOME/venv"
SYMBAC_ROOT="$HOME/SyMBac_2/symbac_training"
REAL_ROOT="$HOME/data/bacdetr_rod_nafnet_only_perspecies"
SYNTH_COCO="$HOME/data/synth_${SPECIES}_coco"
SUBSET_DIR="$HOME/data/synth_${SPECIES}_coco_synth1x_subset"
OUTPUT_DIR="$HOME/data/${SPECIES}_synthaug_1x"

source "${VENV}/bin/activate"
cd "${SYMBAC_ROOT}"

echo "==> Species : ${SPECIES} (n=${N})"
python -u subset_synth_pool.py \
    --format mtb_crops --input_dir "${SYNTH_COCO}" \
    --n "${N}" --output_dir "${SUBSET_DIR}" --seed 0

python -u build_rod_synthaug.py \
    --real_train_ann  "${REAL_ROOT}/annotations/${TRAIN_ANN}" \
    --real_val_ann    "${REAL_ROOT}/annotations/${VAL_ANN}" \
    --real_images_dir "${REAL_ROOT}/images" \
    --extra_crop_dir  "${SUBSET_DIR}" \
    --output_dir      "${OUTPUT_DIR}"

echo "Done -> ${OUTPUT_DIR}"
