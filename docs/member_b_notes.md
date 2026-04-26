# Member B: Candidate Generation Interface

This module now supports two layers:

1. a stable local interface that reads Member A prompt JSON files and writes a
   deterministic generation manifest
2. an HPC execution bundle for real candidate generation with a Qwen-Image
   style large model

The manifest schema and candidate output directory stay stable for Member C.

## Input

Member B reads:

```text
outputs/intermediate/prompts/*.prompts.json
```

Each file contains three `panel_prompts`, with a positive prompt, negative
prompt, scene id, and seed offset from Member A.

## Candidate Counts

Current project settings:

- 16 story cases
- 3 panels per case
- 2 candidates per panel

So Member B prepares **96 candidate images** in total.

If Member C later selects one image per panel, the final submission set is
**48 images**.

## Commands

Run from the project root.

### 1. Dry run only

```powershell
python scripts/generate_images.py --dry-run
```

Writes:

```text
outputs/intermediate/generation_manifest.json
```

### 2. Dry run with placeholder PNGs

```powershell
python scripts/generate_images.py --dry-run --placeholder-images
```

Writes placeholder images under:

```text
outputs/candidates/{case_id}/scene_{scene_id}/candidate_{candidate_id}.png
```

### 3. Prepare the HPC bundle

```powershell
python scripts/generate_images.py --run-model
```

Writes:

```text
outputs/intermediate/generation_manifest.json
outputs/intermediate/generation_status.json
outputs/intermediate/hpc_jobs/shards/job_*.json
outputs/intermediate/hpc_jobs/shards.txt
outputs/intermediate/hpc_jobs/submit_member_b_array.slurm
```

The default sharding strategy is one case per shard, so the current config
creates 16 shards, each with 6 candidate records.

### 4. Run one shard locally or on HPC

```powershell
python scripts/run_hpc_generation.py --config configs/member_b_generation_config.json --shard outputs/intermediate/hpc_jobs/shards/job_000.json
```

The current config uses the `python` adapter and calls
`scripts/qwen_image_infer.py` directly for generation.

### 5. Real Qwen-Image integration point

The scaffold script is:

```text
scripts/qwen_image_infer.py
```

If you need to launch inference through an external shell command instead of the
in-process Python adapter, switch the config from:

```json
"adapter": { "type": "python", ... }
```

to:

```json
"adapter": {
  "type": "command",
  "command": [
    "python",
    "scripts/qwen_image_infer.py",
    "--prompt-payload",
    "{prompt_payload_path}"
  ]
}
```

Then replace the placeholder logic in `scripts/qwen_image_infer.py` with your
real Qwen-Image inference code on HPC.

## Manifest Interface for Member C

Member C should continue reading:

```text
outputs/intermediate/generation_manifest.json
```

Fields that remain stable:

- `case_id`
- `scene_id`
- `candidate_id`
- `seed`
- `prompt`
- `negative_prompt`
- `output_path`
- `status`

The seed remains deterministic:

```text
seed = base_seed + case_index * 1000 + scene_id * 100 + candidate_id
```

## HPC Status and Retry

The worker writes:

```text
outputs/intermediate/generation_status.json
outputs/logs/member_b/*.log
outputs/logs/member_b/*.summary.json
```

Each record is tracked independently, so failed items can be rerun without
restarting the full 96-image batch.

Example rerun of failed records only:

```powershell
python scripts/run_hpc_generation.py --config configs/member_b_generation_config.json --shard outputs/intermediate/hpc_jobs/shards/job_000.json --only-failed
```
