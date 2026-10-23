#!/bin/bash
#SBATCH --job-name=unigraph2-roma
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/roma-%A_%a.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -m unittest -v test_core
"$PYTHON_BIN" -u compare_roma.py --seed "$SLURM_ARRAY_TASK_ID" \
  --output "results/roma-${SLURM_ARRAY_JOB_ID}/seed-${SLURM_ARRAY_TASK_ID}" "$@"
