#!/usr/bin/env bash
#SBATCH --job-name=debug_pa_sources
#SBATCH --output=logs/debug_pa_sources_%j.out
#SBATCH --error=logs/debug_pa_sources_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail
source /scicore/home/boeluc00/martin0088/venv/bin/activate
cd /scicore/home/boeluc00/martin0088/SyMBac_2/symbac_training
python -u debug_pa_sources.py
