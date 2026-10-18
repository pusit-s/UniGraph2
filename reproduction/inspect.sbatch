#!/bin/bash
#SBATCH --job-name=unigraph2-inspect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/inspect-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
/home/pusit.see/.conda/envs/mllm-xai/bin/python -u /home/pusit.see/UniGraph2/reproduction/inspect_data.py
