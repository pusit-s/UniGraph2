#!/bin/bash
#SBATCH --job-name=unigraph2-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/run-%j.log
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -m unittest -v test_core
"$PYTHON_BIN" -u run.py --output "results/run-${SLURM_JOB_ID}" "$@"
"$PYTHON_BIN" summarize.py "results/run-${SLURM_JOB_ID}"
