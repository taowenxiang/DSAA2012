# DSAA2012 Qwen Story Page Pipeline

测试结果在仓库根目录answer文件夹里

本仓库同步在canvas上上传了（canvas版本里有环境文件夹venvs和lora权重文件）

这个仓库当前推荐复现的是新的 **Qwen page-native 生图管线**：  

输入 `data/task_a/*.txt` 中的故事文本，直接生成一整张纵向 story page，而不是先单 panel 生成再拼接。

当前主线固定为：

- 只处理 `2-scene` 和 `3-scene`
- 固定 `vertical` 多格版式
- 生成单位是整页
- 使用 `Qwen-Image-2512 + hybrid page LoRA`
- 选择单位是 page-level rerank

## 1. 先准备好这些

1. 使用项目当前的 `venv` 环境

```bash
export VENV_ROOT=$HOME/code/DSAA2012/venvs/sd3story-py310
source "$VENV_ROOT/bin/activate"
```

仓库里已经保存了当前可工作的环境快照：

- [requirements.current.txt](/Users/mount/Desktop/Programming/DSAA2012/requirements.current.txt)：最适合直接复现安装
- [python-version.txt](/Users/mount/Desktop/Programming/DSAA2012/python-version.txt)：记录 Python 版本
- [pip-list.json](/Users/mount/Desktop/Programming/DSAA2012/pip-list.json)：方便排查包版本差异
- [python-path.txt](/Users/mount/Desktop/Programming/DSAA2012/python-path.txt)：记录当时使用的解释器路径

如果你要在新机器上尽量对齐当前环境，推荐：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.current.txt
```

`environment.yml` 和 `requirements.txt` 还保留在仓库里，但当前主线更建议以这四份导出的环境文件为准。

2. 准备本地模型和 LoRA

- `Qwen-Image-2512` 模型目录，请使用`huggingface-cli download Qwen/Qwen-Image-2512`下载
- 训练好的lora `qwen-story-page-lora-hybrid-v1` 目录（仓库已有）

建议放在下面这两个位置，和脚本默认值一致：

- `artifacts/models/Qwen-Image-2512`
- `artifacts/loras/qwen-image-2512/qwen-story-page-lora-hybrid-v1`

3. 确认输入故事存在

- 输入目录：`data/task_a`

## 2. 跑管线

在仓库根目录执行：

```bash
export PROJ_ROOT=$(pwd)
export MODEL_ROOT=$PROJ_ROOT/artifacts/models/Qwen-Image-2512
export PAGE_LORA_ROOT=$PROJ_ROOT/artifacts/loras/qwen-image-2512/qwen-story-page-lora-hybrid-v1
```
跑单个txt（以02.txt为例，需要修改两处02以换成其他的txt）

建议使用绝对路径
```bash
python scripts/run_qwen_story_page_pipeline.py \
  --stories /hpc2hdd/home/dsaa2012_032/code/DSAA2012/data/task_a/02.txt \
  --out-root /hpc2hdd/home/dsaa2012_032/code/DSAA2012/outputs/qwen_story_page_runs/02 \
  --scene-settings 3 \
  --layout vertical \
  --model-path /hpc2hdd/home/dsaa2012_032/code/DSAA2012/artifacts/models/Qwen-Image-2512 \
  --page-lora-path /hpc2hdd/home/dsaa2012_032/code/DSAA2012/artifacts/loras/qwen-image-2512/qwen-story-page-lora-hybrid-v1 \
  --page-lora-weight-name pytorch_lora_weights.safetensors \
  --page-lora-scale 0.55 \
  --num-candidates 4 \
  --steps 28 \
  --guidance 4.5 \
  --dtype bfloat16 \
  --device cuda \
  --cpu-offload \
  --dreamsim-cache-dir /hpc2hdd/home/dsaa2012_032/code/DSAA2012/artifacts/metrics/dreamsim_ckpts \
  --torch-hub-dir /hpc2hdd/home/dsaa2012_032/code/DSAA2012/artifacts/metrics/torch_hub \
  --require-dreamsim \
  --require-clipscore \
  --skip-existing
```
全部生成（较耗时）
```bash
python scripts/run_qwen_story_page_pipeline.py \
  --stories data/task_a \
  --out-root outputs/qwen_story_page \
  --scene-settings 2 3 \
  --layout vertical \
  --model-path "$MODEL_ROOT" \
  --page-lora-path "$PAGE_LORA_ROOT" \
  --page-lora-weight-name "pytorch_lora_weights.safetensors" \
  --page-lora-scale 0.55 \
  --num-candidates 8 \
  --steps 28 \
  --guidance 4.5 \
  --dtype bfloat16 \
  --device cuda \
  --cpu-offload
```

这条命令会自动串起来：

1. `scripts/qwen_story_build_page_prompts.py`
2. `scripts/qwen_story_infer.py`
3. `scripts/qwen_story_rank_candidates.py`
4. `scripts/qwen_story_export_top_candidates.py`

## 3. 跑完后应该看到什么

主要输出会在 `outputs/qwen_story_page/`（或`outputs/qwen_story_page_runs/`由你刚刚的输出路径决定）：

```text
outputs/qwen_story_page/
├── prompts/
├── candidates/
├── ranking/
└── final/
```

最重要的是：

- `outputs/qwen_story_page/final/2scene_top1`
- `outputs/qwen_story_page/final/3scene_top1`
- `outputs/qwen_story_page/final/2scene_top1_manifest.json`
- `outputs/qwen_story_page/final/3scene_top1_manifest.json`

快速检查是否成功：

```bash
find outputs/qwen_story_page/final -name "*.png" | wc -l
```

如果流程正常，你还会看到：

- `candidates/*_manifest.json`：候选页清单
- `ranking/*.csv`：page-level 排序结果
- `ranking/*.summary.json`：排序摘要

## 4. 最常见的失败点

- `--model-path` 或 `--page-lora-path` 写错：先确认目录真实存在。
- 显存不够：保留 `--cpu-offload`，必要时把 `--num-candidates` 从 `8` 降到 `4`。
- 只想断点续跑：加上 `--skip-existing`。
- 只想跑一种 setting：把 `--scene-settings 2 3` 改成 `--scene-settings 2` 或 `--scene-settings 3`。

