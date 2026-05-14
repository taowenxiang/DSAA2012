# DSAA2012 Final Project 2
## Task 1: Story (Multi-Shot / Multi-Panel Image Generation)

Current integrated project status is summarized in `PROJECT_STATUS.md`.
Use the root-level `configs/`, `data/`, `docs/`, `outputs/`, and `scripts/` folders as the main working copy.

### Team Information
- Team Name: [Your Team Name]
- Course: DSAA2012
- Project Option: **Task 1 – Story**
- Members:
  - Member 1: [Name] ([Student ID])
  - Member 2: [Name] ([Student ID])
  - Member 3: [Name] ([Student ID])

---

## 1. Project Overview

This project builds an **automatic multi-image story generation pipeline** for **Task 1: Story**.  
Given a sequence of text descriptions (shots/panels), our system generates a corresponding sequence of images such that:

- each image matches its panel description,
- recurring characters, objects, backgrounds, and visual style remain consistent,
- the story progresses naturally across panels.

According to the project specification, the system must be:

- **fixed**,
- **automatic**,
- **reproducible**,

and must **not** use:

- manual per-case editing,
- hard-coded outputs for specific test cases,
- agent-based systems,
- external API usage. :contentReference[oaicite:1]{index=1}

---

## 2. Task Definition

### Input
A multi-panel text description for a story case.

### Output
A sequence of generated images that:

1. satisfy **per-panel correctness**,
2. preserve **cross-panel consistency**,
3. maintain **narrative continuity**,
4. preserve **style consistency**,
5. achieve strong **visual quality**. :contentReference[oaicite:2]{index=2}

---

## 3. Our Method

We focus on **pipeline design** rather than full model training.  
Our method uses a local open-source image generation model and builds a structured automatic system around it.

### 3.1 Pipeline Overview

Our pipeline consists of five main stages:

#### Stage 1: Story Parsing
We parse the input story into structured information:

- recurring characters
- recurring objects
- global scene/background
- global style
- panel-specific actions and details

This helps separate:
- **global consistent information**
- **panel-varying information**

#### Stage 2: Prompt Construction
For each panel, we construct a prompt using:

- global character description
- global scene description
- style description
- panel-specific action description
- continuity hints from previous panels

This ensures the generated images are not independent single-image outputs, but part of the same story.

#### Stage 3: Candidate Generation
For each panel, the system generates multiple candidate images using a local image generation model.

#### Stage 4: Automatic Reranking and Selection
Candidates are automatically scored and selected based on:

- panel-level prompt adherence
- consistency with previous panels
- style consistency
- image quality

#### Stage 5: Packaging
The selected final images are copied into the final output directory together
with a manifest for checking counts, paths, and image sizes.

### 3.2 Style System

The pipeline now supports style presets through configuration-driven style ids.

- default submission style: `storybook`
- demo styles: `watercolor`, `anime`, `paper_cutout`
- every pipeline execution creates a numbered run folder like
  `outputs/runs/run_0001_storybook/`
- all prompts, manifests, candidate images, final outputs, and metadata for that
  experiment are stored inside the run folder

For future extension, the generation payload also carries an optional
IP-Adapter backend request. If IP-Adapter is unavailable, styles configured as
`auto_ip_adapter` fall back to `prompt_only`, while `require_ip_adapter` fails
explicitly.

---

## 4. Repository Structure

```text
project_root/
├── README.md
├── PROJECT_STATUS.md
├── .gitignore
├── configs/
│   ├── member_a_prompt_config.json
│   ├── member_b_generation_config.json
│   └── member_b_generation_config.local_4gpu.json
├── data/
│   ├── task_a/
│   └── task_b/
├── docs/
│   ├── labor_division.md
│   ├── member_a_notes.md
│   ├── member_b_notes.md
│   ├── project_goal.md
│   ├── project_progress.md
│   └── project_spec.pdf
├── scripts/
│   ├── parse_story.py
│   ├── build_prompts.py
│   ├── generate_images.py
│   ├── run_hpc_generation.py
│   ├── run_local_generation_batch.py
│   ├── qwen_image_infer.py
│   ├── rerank_candidates.py
│   ├── package_outputs.py
│   ├── run_story_pipeline.py
│   └── validate_member_a.py
└── outputs/
    ├── intermediate/
    ├── candidates/
    ├── final/
    ├── logs/
    └── runs/
```

## 5. Quick Start

Run the integrated post-generation pipeline from the project root:

```bash
python3 scripts/run_story_pipeline.py
```

This will:

1. parse the Task A stories
2. build prompts
3. validate the parsed/prompt outputs
4. rerank the existing candidate images
5. package the final selected images into a new numbered run directory

Example result:

```text
outputs/runs/run_0001_storybook/
```

Key files inside a run:

- `metadata/run_metadata.json`: this run's style, CLI args, and config snapshot
- `intermediate/parsed/`: parsed story JSON
- `intermediate/prompts/`: prompt JSON
- `intermediate/generation_manifest.json`: candidate manifest
- `intermediate/selection_results.json`: rerank result
- `final/submission_manifest.json`: final packaged output manifest

Run a different style preset:

```bash
python3 scripts/run_story_pipeline.py --style watercolor --placeholder-images
```

Important behavior:

- each new run gets a new `run_000x_<style>` folder and never overwrites an old run
- `storybook` will reuse the legacy candidate image seed if needed
- `--placeholder-images` is useful for smoke-testing a style path even before
  real candidate images are generated for that style

---

## 6. SD3 Medium Storyboard LoRA Workflow

This repository also includes a parallel SD3 Medium workflow for a heavier
training-based project direction:

- SD3 Medium one-shot 2-panel / 3-panel storyboard generation
- self-trained Storyboard LoRA
- required Character LoRA
- public LoRA comparison
- independent generation ablation
- CLIPScore, DreamSim, human evaluation, and report asset collection

The SD3 workflow is intentionally separate from the existing Qwen independent
panel pipeline. See:

```text
docs/sd3_story_hpc_runbook.md
```

Local no-model preparation checks:

```bash
python scripts/sd3_build_story_prompts.py --input data/task_a --out-dir data/sd3_story/validation_prompts
python scripts/sd3_prepare_storyboard_data.py --sources local --out-dir data/sd3_story/train_storyboard --resolution 768 --clean
python scripts/sd3_check_resolution.py --dir data/sd3_story/train_storyboard --min-short 768 --fail-on-bad
```

Note: the current SD3 DreamBooth LoRA training path uses a single
`instance_prompt`. The generated `.txt` captions and `metadata.jsonl` files are
kept for dataset auditing, report evidence, and future trainer extensions; they
are not treated as per-image captions by the current training shell scripts.

Model inference and LoRA training are expected to run on HPC after Hugging Face
access for `stabilityai/stable-diffusion-3-medium-diffusers` is accepted.
