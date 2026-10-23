#!/bin/bash
#SBATCH --job-name=unigraph2-shared
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/shared-%A_%a.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -m unittest -v test_core
# Three seed tasks fit the site's submit quota; each visits all five targets.
for DATASET_INDEX in 0 1 2 3 4; do
  RUN_INDEX=$((DATASET_INDEX * 3 + SLURM_ARRAY_TASK_ID))
  "$PYTHON_BIN" -u shared_datasets.py --index "$RUN_INDEX" \
    --output "results/shared-${SLURM_ARRAY_JOB_ID}" "$@"
done
