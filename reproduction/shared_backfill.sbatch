#!/bin/bash
#SBATCH --job-name=unigraph2-shared-backfill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pusit.see/UniGraph2/reproduction/logs/shared-backfill-%j.log
# Work on the final seed's last targets while the main task visits earlier ones.
# compare_shared.py locks each target/seed directory and reuses only exact cases.
set -euo pipefail
cd /home/pusit.see/UniGraph2/reproduction
export OMP_NUM_THREADS=4
PYTHON_BIN="${UNIGRAPH_PYTHON:-/home/pusit.see/.conda/envs/mllm-xai/bin/python}"
OUTPUT_ROOT="${1:?provide the existing shared sweep output directory}"
shift
for RUN_INDEX in 14 11; do
  "$PYTHON_BIN" -u shared_datasets.py --index "$RUN_INDEX" --output "$OUTPUT_ROOT" "$@"
done
