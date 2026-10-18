#!/bin/bash
#SBATCH --job-name=unigraph2-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/report-%j.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" diagnose.py "${1:?Pass a completed results directory}"
"$PYTHON_BIN" summarize.py "$1"
