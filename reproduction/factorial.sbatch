#!/bin/bash
#SBATCH --job-name=unigraph2-factorial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/factorial-%j.log
# 2x2x2 factorial over the three collapse fixes, 3 encoder seeds each, run sequentially.
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
"$PYTHON_BIN" -m unittest test_core
for lr in 1e-3 1e-4; do for w in 0 0.1; do for st in no-standardize standardize; do for s in 0 1 2; do
  out="results/factorial-${SLURM_JOB_ID}/lr${lr}_w${w}_${st}_s${s}"
  "$PYTHON_BIN" -u run.py --lr $lr --warmup $w --$st --seed $s --output "$out" | grep -E "^(COMPLETE|TRAIN \{'epoch': 5)" | cut -c1-200
  "$PYTHON_BIN" diagnose.py "$out" > /dev/null
done; done; done; done
