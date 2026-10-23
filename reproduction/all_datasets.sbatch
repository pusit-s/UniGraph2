#!/bin/bash
#SBATCH --job-name=unigraph2-all-datasets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/all-datasets-%A_%a.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -m unittest -v test_core
"$PYTHON_BIN" -u all_datasets.py --index "$SLURM_ARRAY_TASK_ID" \
  --output "results/all-datasets-${SLURM_ARRAY_JOB_ID}" "$@"
