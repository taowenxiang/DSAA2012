#!/bin/bash
#SBATCH --job-name=member-b-qwen-image
#SBATCH --array=0-0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "$ROOT_DIR"

SHARD_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "outputs/intermediate/local_4gpu_jobs/shards.txt")
python scripts/run_hpc_generation.py \
  --config "configs/member_b_generation_config.local_4gpu.json" \
  --shard "$SHARD_FILE" \
  --status-path "outputs/intermediate/generation_status.local_4gpu.json"
