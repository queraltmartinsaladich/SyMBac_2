#!/usr/bin/env bash
# Generate all visualization figures for SyMBac_2 / HiTMicTools.
# Run from symbac_training/:
#   bash generate_all_figures.sh
#   bash generate_all_figures.sh --dpi 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /scicore/home/boeluc00/martin0088/venv/bin/activate

DPI="${DPI:-300}"
EXTRA_ARGS="$*"

mkdir -p figures

echo "==> [1/5] Synthetic data examples"
python plots/visualize_synthetic_data.py \
    --data_dir  synthetic_data \
    --output    figures/synthetic_examples.png \
    --dpi "$DPI" $EXTRA_ARGS

echo "==> [2/5] Feature distributions"
python plots/plot_feature_distributions.py \
    --dataset_dir dataset \
    --output_dir  figures \
    --dpi "$DPI"

echo "==> [3/5] Tracking + division performance"
python plots/plot_performance.py \
    --weights_dir weights \
    --dataset_dir dataset \
    --output_dir  figures \
    --dpi "$DPI"

echo "==> [4/5] Hard-negative mining illustration"
python plots/plot_hard_negative_mining.py \
    --data_dir  synthetic_data \
    --output    figures/hard_negative_mining.png \
    --dpi "$DPI"

echo "==> [5/5] Training curves"
python plots/plot_training_curves.py \
    --weights_dir weights \
    --output_dir  figures \
    --dpi "$DPI"

echo ""
echo "All figures written to figures/:"
ls -lh figures/*.png 2>/dev/null || echo "  (none produced — check output above)"
