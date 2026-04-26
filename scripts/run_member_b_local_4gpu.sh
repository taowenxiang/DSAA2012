#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/member_b_generation_config.local_4gpu.json}"
RUN_DIR="${RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then
  echo "Please set RUN_DIR to a numbered run folder, for example outputs/runs/run_0001_storybook" >&2
  exit 1
fi
MANIFEST="${MANIFEST:-$RUN_DIR/intermediate/generation_manifest.json}"
STATUS_PATH="${STATUS_PATH:-$RUN_DIR/intermediate/generation_status.local_4gpu.json}"
JOB_DIR="${JOB_DIR:-$RUN_DIR/intermediate/local_4gpu_jobs}"
QWEN_ENV="${QWEN_ENV:-cosplay}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$QWEN_ENV" python)
fi

"${PYTHON_CMD[@]}" scripts/generate_images.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --manifest "$MANIFEST" \
  --job-dir "$JOB_DIR" \
  --status-path "$STATUS_PATH" \
  --run-model

"${PYTHON_CMD[@]}" scripts/run_local_generation_batch.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --manifest "$MANIFEST" \
  --status-path "$STATUS_PATH" \
  "$@"
