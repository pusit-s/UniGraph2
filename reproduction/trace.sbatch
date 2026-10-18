#!/bin/bash
#SBATCH --job-name=unigraph2-trace
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/trace-%j.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -u trace_collapse.py --output "results/trace-${SLURM_JOB_ID}.json" "$@"
