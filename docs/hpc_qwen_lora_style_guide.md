# HPC Guide: Qwen-Image-2512 LoRA Style Training and Integration

This guide is for the following target change:

- keep the current default style unchanged: `storybook`
- add two optional styles: `baroque` and `shinkai`
- train both styles locally on HPC with `sbatch`
- use the trained LoRA weights inside the existing pipeline

The repository now supports per-style LoRA loading through `configs/style_presets.json`.

## 1. Recommended HPC layout

Keep code and large assets separate.

```text
/data/home/$USER/code/DSAA2012/
├── Project/                                  # git clone of this repo
└── venvs/
    └── dsaa2012/

/data/scratch/$USER/dsaa2012_assets/
├── hf_cache/
├── models/
│   └── Qwen-Image-2512/
├── datasets/
│   ├── raw/
│   │   ├── latin-american-baroque-18k-multimodal/
│   │   └── makoto-shinkai-picture/
│   └── prepared/
│       ├── baroque_v1/
│       │   ├── images/
│       │   └── metadata.jsonl
│       └── shinkai_v1/
│           ├── images/
│           └── metadata.jsonl
├── loras/
│   └── qwen-image-2512/
│       ├── baroque-lora-v1/
│       └── shinkai-lora-v1/
└── logs/
    ├── prep/
    └── train/
```

Inside the repo, use symlinks so config paths stay short:

```bash
cd /data/home/$USER/code/DSAA2012/Project
mkdir -p artifacts
ln -sfn /data/scratch/$USER/dsaa2012_assets/models artifacts/models
ln -sfn /data/scratch/$USER/dsaa2012_assets/datasets artifacts/datasets
ln -sfn /data/scratch/$USER/dsaa2012_assets/loras artifacts/loras
```

With this layout:

- base model path: `artifacts/models/Qwen-Image-2512`
- baroque LoRA path: `artifacts/loras/qwen-image-2512/baroque-lora-v1`
- shinkai LoRA path: `artifacts/loras/qwen-image-2512/shinkai-lora-v1`

Those are the paths already wired into `configs/style_presets.json`.

## 2. Clone the repo on HPC

```bash
mkdir -p /data/home/$USER/code/DSAA2012
cd /data/home/$USER/code/DSAA2012
git clone <YOUR_REPO_URL> Project
cd Project
```

If your cluster requires modules:

```bash
module purge
module load cuda/12.1
module load python/3.10
```

Then create a local environment:

```bash
python -m venv /data/home/$USER/code/DSAA2012/venvs/dsaa2012
source /data/home/$USER/code/DSAA2012/venvs/dsaa2012/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets huggingface_hub peft torchvision
```

## 3. Download the base model locally

Run this on a login node if compute nodes do not allow outbound network.

```bash
export HF_HOME=/data/scratch/$USER/dsaa2012_assets/hf_cache
mkdir -p /data/scratch/$USER/dsaa2012_assets/models

huggingface-cli download Qwen/Qwen-Image-2512 \
  --local-dir /data/scratch/$USER/dsaa2012_assets/models/Qwen-Image-2512
```

After that, copy the HPC generation template and point it at the local model:

```bash
cp configs/member_b_generation_config.hpc.template.json configs/member_b_generation_config.hpc.json
```

The template already uses:

- `model_path = artifacts/models/Qwen-Image-2512`

You only need to fill:

- `scheduler.partition`
- `scheduler.account`
- optionally `gpus_per_task`, `mem_gb`, `time_limit`

## 4. Download the two datasets locally

Target datasets:

- `FulcoPin/latin-american-baroque-18k-multimodal`
- `Fung804/makoto-shinkai-picture`

Recommended raw dataset directories:

- `artifacts/datasets/raw/latin-american-baroque-18k-multimodal`
- `artifacts/datasets/raw/makoto-shinkai-picture`

Suggested approach:

1. download the full raw dataset once
2. create a smaller curated training subset
3. export it into `images/ + metadata.jsonl`

Why subset first:

- baroque set has about 18.2k rows, which is larger than you likely need for a style-only LoRA
- shinkai set has about 1.35k rows, which is already compact

Good first-pass subset sizes:

- `baroque`: 2,000 to 4,000 images
- `shinkai`: use the full 1,347 images, then prune duplicates or weak examples manually if needed

## 5. Prepare the local training format

Use one JSONL row per image:

```json
{"file_name": "000001.png", "text": "baroque style painting, dramatic chiaroscuro, ornate gilded detail"}
```

Recommended export targets:

- `artifacts/datasets/prepared/baroque_v1/images`
- `artifacts/datasets/prepared/baroque_v1/metadata.jsonl`
- `artifacts/datasets/prepared/shinkai_v1/images`
- `artifacts/datasets/prepared/shinkai_v1/metadata.jsonl`

Preparation rules:

- resize or bucket to the same resolution you plan to train on, usually `512` first
- remove corrupt files
- remove exact duplicates
- keep captions short and style-focused
- for the shinkai-like set, avoid overly literal copyrighted title strings in the caption text

Recommended caption style:

- baroque: mention oil painting, ornate detail, religious tableau, theatrical light, gold ornament, colonial baroque
- shinkai-like: mention luminous sky, reflective light, cinematic clouds, saturated sunset, anime film background, rain glow

## 6. Training strategy

Train two separate LoRAs:

- `baroque-lora-v1`
- `shinkai-lora-v1`

Recommended first-pass hyperparameters:

```text
resolution: 512
rank: 32
lora_alpha: 32
batch_size_per_gpu: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
max_train_steps:
  - baroque: 3000 to 5000
  - shinkai: 2000 to 3500
mixed_precision: bf16
checkpoint_every: 500
validation_every: 500
```

Important note:

- the official Qwen-Image repo supports local inference well
- official Diffusers support for Qwen-Image LoRA and finetuning is still evolving
- for training, use a local Diffusers-compatible LoRA trainer you vendor into HPC, then keep all checkpoints fully local

Recommended trainer location:

```text
/data/home/$USER/code/DSAA2012/Project/third_party/qwen_image_lora_trainer/
```

Your trainer only needs to satisfy this contract:

- input base model: `artifacts/models/Qwen-Image-2512`
- input dataset: `artifacts/datasets/prepared/<style>_v1`
- output checkpoint dir: `artifacts/loras/qwen-image-2512/<style>-lora-v1`
- final weight file name: `pytorch_lora_weights.safetensors`

## 7. `sbatch` template for LoRA training

Save this as `scripts/train_style_lora.sbatch` on HPC:

```bash
#!/bin/bash
#SBATCH --job-name=qwen-lora-baroque
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=/data/scratch/%u/dsaa2012_assets/logs/train/%x-%j.out

set -euo pipefail

STYLE_NAME="${STYLE_NAME:-baroque}"
DATASET_DIR="${DATASET_DIR:-/data/scratch/$USER/dsaa2012_assets/datasets/prepared/baroque_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/scratch/$USER/dsaa2012_assets/loras/qwen-image-2512/baroque-lora-v1}"
MODEL_DIR="${MODEL_DIR:-/data/scratch/$USER/dsaa2012_assets/models/Qwen-Image-2512}"
TRAIN_ENTRY="${TRAIN_ENTRY:-/data/home/$USER/code/DSAA2012/Project/third_party/qwen_image_lora_trainer/train_lora.py}"

source /data/home/$USER/code/DSAA2012/venvs/dsaa2012/bin/activate
cd /data/home/$USER/code/DSAA2012/Project

export HF_HOME=/data/scratch/$USER/dsaa2012_assets/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p "$OUTPUT_DIR"

accelerate launch "$TRAIN_ENTRY" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --train_data_dir "$DATASET_DIR" \
  --image_column image \
  --caption_column text \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --rank 32 \
  --max_train_steps 4000 \
  --checkpointing_steps 500 \
  --mixed_precision bf16 \
  --output_dir "$OUTPUT_DIR" \
  --seed 2012
```

For the shinkai-like LoRA, submit:

```bash
sbatch \
  --export=ALL,STYLE_NAME=shinkai,DATASET_DIR=/data/scratch/$USER/dsaa2012_assets/datasets/prepared/shinkai_v1,OUTPUT_DIR=/data/scratch/$USER/dsaa2012_assets/loras/qwen-image-2512/shinkai-lora-v1 \
  scripts/train_style_lora.sbatch
```

## 8. Hook the trained LoRA into this repo

This repo now already supports LoRA-aware style presets.

Current preset ids:

- `storybook`
- `baroque`
- `shinkai`

Current LoRA target paths in `configs/style_presets.json`:

- `artifacts/loras/qwen-image-2512/baroque-lora-v1`
- `artifacts/loras/qwen-image-2512/shinkai-lora-v1`

Expected final file layout:

```text
artifacts/loras/qwen-image-2512/baroque-lora-v1/
└── pytorch_lora_weights.safetensors

artifacts/loras/qwen-image-2512/shinkai-lora-v1/
└── pytorch_lora_weights.safetensors
```

If your trainer writes a different filename, update:

- `lora_weight_name` in `configs/style_presets.json`

## 9. Run inference on HPC

First create a numbered run and prompt bundle:

```bash
python scripts/run_story_pipeline.py --style baroque --placeholder-images
```

Assume the new run is:

```text
outputs/runs/run_0004_baroque
```

Then rebuild the manifest for real HPC generation:

```bash
python scripts/generate_images.py \
  --style baroque \
  --run-dir outputs/runs/run_0004_baroque \
  --config configs/member_b_generation_config.hpc.json \
  --run-model
```

Then submit the generated Slurm array script:

```bash
sbatch outputs/runs/run_0004_baroque/intermediate/hpc_jobs/submit_member_b_array.slurm
```

For the other style:

```bash
python scripts/run_story_pipeline.py --style shinkai --placeholder-images

python scripts/generate_images.py \
  --style shinkai \
  --run-dir outputs/runs/run_0005_shinkai \
  --config configs/member_b_generation_config.hpc.json \
  --run-model
```

## 10. Practical recommendations

- Start with one style first, preferably `shinkai`, because the dataset is smaller.
- Do a 100-step smoke test before the full training job.
- Generate 10 to 20 held-out validation prompts after each checkpoint interval.
- Keep `storybook` as the no-LoRA baseline for comparison.
- If GPU memory is tight, reduce resolution before reducing caption quality.
- Keep code in home storage and large model/data/checkpoints in scratch storage.

## 11. Licensing and risk notes

- The baroque dataset viewer exposes a `cc-by-4.0` license on Hugging Face.
- The makoto-shinkai-picture viewer exposes image/text rows and size information, but the license is not obvious in the viewer page snapshot. Verify that dataset usage is acceptable for your course submission and local training before release or publication.
- For a course project, I recommend describing the second style as `shinkai-inspired` rather than implying official affiliation.
