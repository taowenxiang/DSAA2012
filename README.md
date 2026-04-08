# DSAA2012 Final Project 2
## Task 1: Story (Multi-Shot / Multi-Panel Image Generation)

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

Our pipeline consists of four main stages:

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

The final selected sequence is saved in the official output format.

---

## 4. Repository Structure

```text
project_root/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── testA/
│   ├── testB/
│   └── examples/
├── scripts/
│   ├── parse_story.py
│   ├── build_prompts.py
│   ├── generate_images.py
│   ├── rerank_candidates.py
│   └── package_outputs.py
├── models/
│   └── [local model files or checkpoints]
├── outputs/
│   ├── intermediate/
│   └── final/
└── report/
    └── final_report.pdf