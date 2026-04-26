DSAA2012 Task 1（Story）项目开发计划
一、目标与约束（先对齐验收）
目标：多格文本输入 → 多格图像输出；每格贴合描述，跨格人物/物体/背景/风格一致，叙事连贯。
硬性约束：流程固定、全自动、可复现；禁止：逐案例手工改图、针对测试用例硬编码、基于 Agent 的编排、外部 API（模型与推理须本地化）。
计划里要时刻体现：配置驱动、随机种子固定、依赖与模型版本可记录。
二、阶段划分（与 README 四阶段对应）
阶段 0：环境与仓库基座（1–3 天）
补齐 requirements.txt、Python 版本说明；选定本地开源文生图方案（如 Diffusers + 指定 checkpoint，具体以课程允许范围为准）。
建立 configs/default.yaml：模型路径、分辨率、采样步数、随机种子、每格候选数 K、设备（CPU/GPU）等。
约定 data/ 下测试/示例的输入格式；outputs/intermediate/ 与 outputs/final/ 的目录与命名规则（与官方提交格式对齐，需在课程文档中核实）。
阶段 1：Story Parsing（scripts/parse_story.py）
定义解析输出结构：全局（角色、物体、场景、风格）+ 每格（动作与细节）。
在禁止 Agent / 外部 API 前提下：用规则 + 模板 + 轻量本地 NLP（若课程允许）或完全规则化分段；关键是输出稳定、可复现。
单元测试或小样例回归：同一输入多次运行结构一致。
阶段 2：Prompt Construction（scripts/build_prompts.py）
将解析结果组装为每格完整 prompt：全局块 + 当格动作 + 来自前一格（或前两格）的连续性短语（衣着、姿态、镜头等用文字约束）。
输出每格的 prompt 中间文件（JSON/YAML），便于调试与报告截图。
阶段 3：Candidate Generation（scripts/generate_images.py）
按配置对每格生成 K 张候选图；固定 seed 策略（可“每格 base_seed + panel_index”避免完全雷同又可复现）。
若使用 IP-Adapter / Reference Image 等本地一致性手段，在本阶段接入并写入配置；若无，则靠 prompt 与后续打分迭代。
阶段 阶段 4：Rerank & Selection（scripts/rerank_candidates.py）
设计可自动打分的指标（全部本地）：
与当格 prompt 的语义对齐（本地 CLIP 或同类嵌入相似度）；
与已选上一格图像的视觉一致性（同一嵌入空间或感知哈希等轻量方案，按实现难度取舍）；
风格/质量 proxy（清晰度、饱和度等启发式或轻量模型，视时间而定）。
贪心或动态规划式选序列：逐格在候选中选分最高且兼顾与前一张一致的一张（先实现贪心，时间允许再优化）。
阶段 5：打包交付（scripts/package_outputs.py）
将最终序列整理为官方要求的 outputs/final/ 格式；校验张数、顺序、分辨率。
三、里程碑建议（可按周推进）
里程碑	内容	产出
M1
环境 + 单格文本→单图跑通
可复现的最小 demo
M2
解析 + prompt 全流程（多格）
中间 JSON + 多格 baseline 图
M3
每格 K 候选 + 简单 rerank
一致性主观明显提升
M4
打包脚本 + 在 testA/testB 上批量跑
完整流水线
M5
报告与消融
report/final_report.pdf：各模块作用与失败样例
四、风险与对策
一致性仅靠文本：优先级提高“前情摘要”进 prompt；提高 K 与 rerank 权重。
算力与时间：降低分辨率/步数做开发，提交前再拉到配置中的“正式”档位；可缓存解析与 prompt。
可复现性：所有随机性收敛到 configs/default.yaml 中的 seed；记录模型文件哈希或版本。
五、分工示意（三人队）
成员 A：解析 + prompt 流水线与数据格式。
成员 B：本地生成与 generate_images.py，效率与显存。
成员 C：rerank 指标、打包与批量评测、报告图表。
如果你希望计划更“落地”，可以补充两点我会按课程要求帮你把里程碑改成按天排期：（1）官方提交的图片格式与目录规范；（2）是否必须指定某一款本地模型。


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