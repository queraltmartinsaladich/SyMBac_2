#!/usr/bin/env bash
#SBATCH --job-name=debug_pa_size
#SBATCH --output=logs/debug_pa_size_%j.out
#SBATCH --error=logs/debug_pa_size_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=scicore
#SBATCH --qos=30min

set -euo pipefail
source $HOME/venv/bin/activate
cd $HOME/SyMBac_2/symbac_training
python -u debug_pa_size.py
