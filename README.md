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
    ├── candidates/
    ├── final/
    ├── intermediate/
    └── logs/
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
5. package the final selected images into `outputs/final/`
