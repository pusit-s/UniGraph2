#!/bin/bash
#SBATCH --job-name=unigraph2-download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/download-%j.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -u download_mmgraph.py --output "results/download-${SLURM_JOB_ID}.json" "$@"
