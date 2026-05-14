#!/bin/bash
#SBATCH --job-name=sd3-story-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=%x-%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "$ROOT_DIR"

source "${VENV_PATH:-/data/home/$USER/code/DSAA2012/venvs/sd3story/bin/activate}"
export HF_HOME="${HF_HOME:-/data/scratch/$USER/dsaa2012_sd3_assets/hf_cache}"

mkdir -p outputs/sd3_story/logs

export DIFFUSERS_DIR="${DIFFUSERS_DIR:-diffusers}"
export MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-3-medium-diffusers}"
export INSTANCE_DIR="${INSTANCE_DIR:-data/sd3_story/train_storyboard}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/loras/sd3/storyboard_lora}"
export RANK="${RANK:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"

bash scripts/sd3_train_storyboard_lora.sh
