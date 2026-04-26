# DSAA2012 Task 1 (Story) 项目开发计划书

## 一、 项目目标与核心约束
* **核心目标**：实现从“多格文本描述”到“多格图像序列”的全自动生成。确保每格图像准确传达文本语义，且在角色、物体、背景及艺术风格上保持跨格高度一致。
* **硬性约束**：
    * **全自动化**：流程固定，禁止针对特定案例的人工干预或硬编码。
    * **本地推理**：禁止调用任何外部 API（如 OpenAI, Midjourney），所有模型及推理逻辑必须在本地部署。
    * **禁止 Agent 编排**：不使用基于 LLM Agent 的动态决策流。
    * **可复现性**：全过程由 `configs/*.yaml` 驱动，固定随机种子（Seed），支持版本化复现。

---

## 二、 阶段划分与实施路径

### 阶段 0：环境搭建与工程基座 (Day 1–3)
* **环境规范**：
    * 编写 `requirements.txt` 或 `environment.yml`，锁定 Python、PyTorch、Diffusers 版本。
    * **选型**：选定本地开源方案（如 SDXL-Turbo 或 SD1.5 + 特定 LoRA），确保显存兼容性。
* **配置驱动管理**：
    * 建立 `configs/default.yaml`：包含模型路径、分辨率、采样器类型、步数、全局 Seed、每格候选数量 $K$、推理设备（GPU/CPU）等参数。
* **目录结构规范**：
    * `data/`：存放测试输入文本。
    * `outputs/intermediate/`：存解析出的结构化 JSON 和中间 Prompt。
    * `outputs/final/`：严格对齐官方提交格式（文件名、分辨率、索引）。

### 阶段 1：故事结构解析 (Story Parsing)
* **脚本**：`scripts/parse_story.py`
* **实现方案**：在无外部 API 前提下，采用“规则引擎 + 模板匹配 + 轻量化本地 NLP（如句子切割）”提取：
    * **全局属性**：核心角色描述、场景主色调、艺术风格词。
    * **局部属性**：每格的动作（Action）、特有物体、镜头语言（中景/特写）。
* **产出**：结构化 `story_structure.json`，确保同一文本多次运行产出的结构一致。

### 阶段 2：提示词工程化 (Prompt Construction)
* **脚本**：`scripts/build_prompts.py`
* **策略**：构建“全局一致性块 + 局部动态块”的复合 Prompt。
    * **全局块**：始终置于 Prompt 头部，锚定画风与角色特征。
    * **局部块**：描述当前帧的具体变化。
    * **连续性约束**：引入前一帧的关键描述词作为微弱引导，增强叙事逻辑。
* **产出**：`prompts_list.json`，用于后续生成。

### 阶段 3：候选生成 (Candidate Generation)
* **脚本**：`scripts/generate_images.py`
* **一致性保障方案**：
    * **Seed 策略**：采用 `base_seed + panel_index` 模式，既保证可复现，又避免多格图像过于雷同。
    * **技术接入**：若允许，接入本地 **IP-Adapter** 或 **ControlNet** 保持角色/构图一致性，相关权重写入配置。
* **并行/批量**：对每格生成 $K$ 张候选图（Candidate set），为后续 Rerank 提供筛选空间。

### 阶段 4：重排与筛选 (Rerank & Selection)
* **脚本**：`scripts/rerank_candidates.py`
* **自动评分指标（全本地实现）**：
    1.  **语义对齐度**：使用本地 CLIP 计算 Image 与 Prompt 的 Cosine Similarity。
    2.  **视觉一致性**：计算当前帧与上一帧已选图像的视觉特征向量距离（Perceptual Loss 或 CLIP Image Embedding）。
    3.  **质量评估**：启发式过滤（如过暗、噪点过大）。
* **决策算法**：采用**贪心搜索（Greedy Search）**或简单的**动态规划（DP）**，寻找整条故事线综合得分最高的路径。

### 阶段 5：打包交付与校验 (Packaging)
* **脚本**：`scripts/package_outputs.py`
* **功能**：自动化重命名、调整分辨率、生成最终序列，并进行合规性自检（张数、格式）。

---

## 三、 里程碑 (Milestones)

| 阶段 | 交付物 | 目标 |
| :--- | :--- | :--- |
| **M1** | Minimal Demo | 环境跑通，实现“单文本 -> 单图”可复现生成 |
| **M2** | Pipeline Baseline | 多格解析 + Prompt 自动构建，生成基础故事图序列 |
| **M3** | Consistency V1 | 引入 $K$ 候选与 CLIP Rerank，主观一致性明显提升 |
| **M4** | Batch Processing | 自动化脚本完成 testA/testB 批量生成，格式完全合规 |
| **M5** | Final Report | 完成报告，包含消融实验数据与一致性对比截图 |

---

## 四、 风险预估与应对

* **风险：跨格角色“走形”（Consistency Loss）**
    * **对策**：增强 Rerank 阶段视觉特征权重；在 Prompt 中强化角色特征词（如特定发色、配饰）；尝试本地 IP-Adapter。
* **风险：显存不足（OOM）**
    * **对策**：开发阶段使用较低采样步数（Steps）和 512x512 分辨率，最终导出时再切换至高精度配置。
* **风险：可复现性失效**
    * **对策**：严格执行 `torch.use_deterministic_algorithms(True)`，并将所有模型文件的 MD5 哈希记录在 README 中。

---

## 五、 团队分工

* **成员 A (解析与逻辑)**：负责 Story Parsing、Prompt Construction、数据流结构定义及项目主控脚本。
* **成员 B (生成与性能)**：负责本地模型部署、Diffusion 推理优化、显存调优、IP-Adapter/ControlNet 集成。
* **成员 C (评价与报告)**：负责 Rerank 算法实现（CLIP 评分）、批量自动化测试、数据可视化及最终报告撰写。