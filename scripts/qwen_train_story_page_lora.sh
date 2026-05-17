#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-artifacts/models/Qwen-Image-2512}"
MUSUBI_ROOT="${MUSUBI_ROOT:-$HOME/code/musubi-tuner}"
DATASET_CONFIG="${DATASET_CONFIG:-train_configs/qwen_story_page.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/loras/qwen-image-2512/qwen-story-page-lora-v1}"
OUTPUT_NAME="${OUTPUT_NAME:-pytorch_lora_weights}"

NETWORK_DIM="${NETWORK_DIM:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-2012}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [ ! -d "$MUSUBI_ROOT" ]; then
  echo "Missing MUSUBI_ROOT: $MUSUBI_ROOT"
  exit 1
fi
if [ ! -d "$MODEL_ROOT" ]; then
  echo "Missing MODEL_ROOT: $MODEL_ROOT"
  exit 1
fi
if [ ! -f "$DATASET_CONFIG" ]; then
  echo "Missing DATASET_CONFIG: $DATASET_CONFIG"
  exit 1
fi

resolve_model_file() {
  local input_path="$1"
  local label="$2"

  if [ -f "$input_path" ]; then
    echo "$input_path"
    return 0
  fi

  if [ ! -d "$input_path" ]; then
    echo "Missing ${label}: $input_path" >&2
    return 1
  fi

  local shard_first
  shard_first=$(find "$input_path" -maxdepth 1 -type f \( \
      -name 'diffusion_pytorch_model-00001-of-*.safetensors' -o \
      -name 'model-00001-of-*.safetensors' \
    \) | sort | head -n 1)
  if [ -n "$shard_first" ]; then
    echo "$shard_first"
    return 0
  fi

  local single_file
  single_file=$(find "$input_path" -maxdepth 1 -type f -name '*.safetensors' | sort | head -n 1)
  if [ -n "$single_file" ]; then
    echo "$single_file"
    return 0
  fi

  echo "Could not resolve ${label} file from directory: $input_path" >&2
  return 1
}

DIT_PATH=$(resolve_model_file "$MODEL_ROOT/transformer" "transformer") || exit 1
VAE_PATH=$(resolve_model_file "$MODEL_ROOT/vae" "vae") || exit 1
TEXT_ENCODER_PATH=$(resolve_model_file "$MODEL_ROOT/text_encoder" "text_encoder") || exit 1

mkdir -p "$OUTPUT_DIR"
cd "$MUSUBI_ROOT"

accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision "$MIXED_PRECISION" \
  src/musubi_tuner/qwen_image_train_network.py \
  --dit "$DIT_PATH" \
  --vae "$VAE_PATH" \
  --text_encoder "$TEXT_ENCODER_PATH" \
  --model_version original \
  --dataset_config "$DATASET_CONFIG" \
  --sdpa --mixed_precision "$MIXED_PRECISION" \
  --timestep_sampling shift \
  --weighting_scheme none --discrete_flow_shift 2.2 \
  --optimizer_type adamw8bit --learning_rate "$LEARNING_RATE" --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_qwen_image \
  --network_dim "$NETWORK_DIM" \
  --max_train_epochs "$MAX_TRAIN_EPOCHS" --save_every_n_epochs 1 --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME" \
  $EXTRA_TRAIN_ARGS
