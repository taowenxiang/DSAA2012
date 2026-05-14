#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-3-medium-diffusers}"
INSTANCE_DIR="${INSTANCE_DIR:-data/sd3_story/train_character}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/loras/sd3/character_lora}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-diffusers}"

RESOLUTION="${RESOLUTION:-768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-600}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-200}"
RANK="${RANK:-8}"
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"

TRAIN_SCRIPT="$DIFFUSERS_DIR/examples/dreambooth/train_dreambooth_lora_sd3.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Missing $TRAIN_SCRIPT. Clone diffusers first:"
  echo "  git clone https://github.com/huggingface/diffusers.git"
  echo "  pip install -e \"./diffusers[torch]\""
  exit 1
fi

RANK_ARGS=()
if python "$TRAIN_SCRIPT" --help 2>/dev/null | grep -q -- "--rank"; then
  RANK_ARGS+=(--rank="$RANK")
else
  echo "Warning: $TRAIN_SCRIPT help does not list --rank. If training fails, update Diffusers or remove RANK usage."
fi

EXTRA_ARGS=()
if [ "$USE_8BIT_ADAM" = "1" ]; then
  EXTRA_ARGS+=(--use_8bit_adam)
fi

accelerate launch "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="a sks_storyhero character, short black hair, yellow raincoat, red backpack" \
  --resolution="$RESOLUTION" \
  --center_crop \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate="$LEARNING_RATE" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  "${RANK_ARGS[@]}" \
  --seed="$SEED" \
  --mixed_precision="$MIXED_PRECISION" \
  --gradient_checkpointing \
  "${EXTRA_ARGS[@]}"
