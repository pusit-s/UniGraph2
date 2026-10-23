#!/bin/bash
#SBATCH --job-name=unigraph2-all-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/all-report-%j.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -u all_datasets_report.py "$@"
