# Qwen Story Dual-LoRA HPC Runbook

这份手册对应的是新的 **Qwen 一次成故事页** 路线：

- 基座：`Qwen/Qwen-Image-2512`
- 方法：**全局版式 LoRA + 故事级角色 LoRA**
- 输出：**竖向 2/3 scene story page**
- 评估：`CLIPScore + DreamSim + human eval`

## 0. 重要说明

这条流程里的训练和双 LoRA 推理，当前仓库实现的是：

- **训练**：标准 Qwen checkpoint + `musubi-tuner`
- **推理**：标准 Qwen checkpoint + Diffusers + 多 LoRA 叠加

也就是说：

- 这份 runbook 的**主流程不使用 GGUF 作为训练 backbone**
- 如果你想额外做 GGUF 推理试验，可以单独做
- 但**不要把 GGUF 当成这条 LoRA 训练链的主路径**

原因很简单：这套仓库代码现在接的是可训练、可叠加 LoRA、可批量评估的 Qwen 全精度/标准 checkpoint 路线，而不是 `stable-diffusion.cpp` / `ComfyUI-GGUF` 后端。

## 1. 推荐目录

```text
$HOME/code/DSAA2012/
├── Project/
└── venvs/
    └── dsaa2012/

$HOME/dsaa2012_qwen_story_assets/
├── hf_cache/
├── models/
│   └── Qwen-Image-2512/
├── loras/
│   └── qwen-image-2512/
│       ├── qwen-story-page-lora-v1/
│       └── qwen-story-cast/
├── train_configs/
└── logs/
```

下面都假设：

```bash
export PROJ_ROOT=$HOME/code/DSAA2012/Project
export VENV_ROOT=$HOME/code/DSAA2012/venvs/dsaa2012
export ASSET_ROOT=$HOME/dsaa2012_qwen_story_assets
export MODEL_ROOT=$ASSET_ROOT/models/Qwen-Image-2512
export LORA_ROOT=$ASSET_ROOT/loras/qwen-image-2512
export TRAIN_CONFIG_ROOT=$ASSET_ROOT/train_configs
export HF_HOME=$ASSET_ROOT/hf_cache
```

## 2. 一次性环境准备

```bash
mkdir -p "$HOME/code/DSAA2012" "$ASSET_ROOT" "$TRAIN_CONFIG_ROOT"
cd "$HOME/code/DSAA2012"

git clone <YOUR_REPO_URL> Project
cd "$PROJ_ROOT"

python3.10 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets huggingface_hub peft torchvision
```

下载 Qwen 基座：

```bash
huggingface-cli download Qwen/Qwen-Image-2512 \
  --local-dir "$MODEL_ROOT"
```

安装 `musubi-tuner`：

```bash
cd "$HOME/code/DSAA2012"
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
source "$VENV_ROOT/bin/activate"
pip install -e .
```

给仓库挂软链接：

```bash
cd "$PROJ_ROOT"
mkdir -p artifacts
ln -sfn "$ASSET_ROOT/models" artifacts/models
ln -sfn "$ASSET_ROOT/loras" artifacts/loras
```

如果你用 `screen`：

```bash
screen -S qwen-story
# 离开：Ctrl-A 然后 D
# 回来：screen -r qwen-story
```

## 3. 先准备全局 page LoRA 训练数据

这个数据来自你们现有的 `storybook` 单格输出，把它们拼成**竖向** 2/3-panel 页。

```bash
cd "$PROJ_ROOT"
source "$VENV_ROOT/bin/activate"

python scripts/qwen_story_prepare_page_lora_data.py \
  --stories data/task_a \
  --run-dirs outputs/runs/run_0001_storybook/final outputs/runs/run_0003_storybook/final \
  --out-dir data/qwen_story/page_lora_train \
  --layout vertical \
  --panel-size 512 \
  --clean
```

检查：

```bash
ls data/qwen_story/page_lora_train | head
```

应该能看到：

- `.png`
- `.txt`
- `metadata.jsonl`

## 4. 为 page LoRA 写 musubi dataset config

```bash
python scripts/qwen_story_write_dataset_toml.py \
  --image-dir data/qwen_story/page_lora_train \
  --out "$TRAIN_CONFIG_ROOT/qwen_story_page.toml" \
  --resolution 512
```

## 5. 先做 page LoRA smoke test

先赋执行权限：

```bash
chmod +x scripts/qwen_train_story_page_lora.sh
chmod +x scripts/qwen_train_story_cast_lora.sh
```

Smoke test：

```bash
cd "$PROJ_ROOT"
source "$VENV_ROOT/bin/activate"

MODEL_ROOT="$MODEL_ROOT" \
MUSUBI_ROOT="$HOME/code/DSAA2012/musubi-tuner" \
DATASET_CONFIG="$TRAIN_CONFIG_ROOT/qwen_story_page.toml" \
OUTPUT_DIR="$LORA_ROOT/qwen-story-page-lora-v1-smoke" \
OUTPUT_NAME="pytorch_lora_weights" \
MAX_TRAIN_EPOCHS=1 \
NETWORK_DIM=16 \
bash scripts/qwen_train_story_page_lora.sh 2>&1 | tee "$ASSET_ROOT/logs/page_lora_smoke.log"
```

如果你显存比较紧，可以加：

```bash
EXTRA_TRAIN_ARGS="--fp8_base --fp8_scaled --blocks_to_swap 16"
```

## 6. 跑正式 page LoRA

推荐先试：

```bash
MODEL_ROOT="$MODEL_ROOT" \
MUSUBI_ROOT="$HOME/code/DSAA2012/musubi-tuner" \
DATASET_CONFIG="$TRAIN_CONFIG_ROOT/qwen_story_page.toml" \
OUTPUT_DIR="$LORA_ROOT/qwen-story-page-lora-v1" \
OUTPUT_NAME="pytorch_lora_weights" \
MAX_TRAIN_EPOCHS=8 \
NETWORK_DIM=16 \
bash scripts/qwen_train_story_page_lora.sh 2>&1 | tee "$ASSET_ROOT/logs/page_lora_full.log"
```

建议 checkpoint 看一次效果后，再决定要不要上到 `10-12` epochs。

## 7. 生成故事级 cast seed prompts

```bash
cd "$PROJ_ROOT"
source "$VENV_ROOT/bin/activate"

python scripts/qwen_story_build_cast_seed_prompts.py \
  --stories data/task_a \
  --out-dir data/qwen_story/cast_seed_prompts \
  --num-images 12 \
  --width 768 \
  --height 768
```

每个 story case 会得到一个 `.jsonl`。

## 8. 用 Qwen 生成 cast seed images

这里建议**带着已经训练好的 page LoRA** 去生成角色参考图，这样 cast LoRA 会更接近最终页风格。

```bash
python scripts/qwen_story_infer.py \
  --prompts data/qwen_story/cast_seed_prompts \
  --out-dir data/qwen_story/cast_seed_images \
  --model "$MODEL_ROOT" \
  --dtype bfloat16 \
  --device cuda \
  --steps 28 \
  --guidance 4.5 \
  --num-candidates 1 \
  --global-lora-path "$LORA_ROOT/qwen-story-page-lora-v1" \
  --global-lora-weight-name "pytorch_lora_weights.safetensors" \
  --global-lora-scale 0.55
```

如果你的 page LoRA 实际输出文件名不是上面这个，就改成真实文件名。

## 9. 把 cast seed images 整理成 per-story LoRA 数据集

```bash
python scripts/qwen_story_prepare_cast_lora_data.py \
  --prompt-dir data/qwen_story/cast_seed_prompts \
  --image-dir data/qwen_story/cast_seed_images \
  --out-root data/qwen_story/cast_lora_datasets \
  --clean
```

结果会长成：

```text
data/qwen_story/cast_lora_datasets/
├── 01/
├── 02/
├── ...
```

每个 case 目录里会有：

- `000001.png`
- `000001.txt`
- `metadata.jsonl`

## 10. 为每个 case 写 dataset TOML

```bash
mkdir -p "$TRAIN_CONFIG_ROOT/cast"

for case_dir in data/qwen_story/cast_lora_datasets/*; do
  case_id=$(basename "$case_dir")
  python scripts/qwen_story_write_dataset_toml.py \
    --image-dir "$case_dir" \
    --out "$TRAIN_CONFIG_ROOT/cast/${case_id}.toml" \
    --resolution 512
done
```

## 11. 先 smoke test 一个 cast LoRA

先挑一个 case，例如 `01`：

```bash
MODEL_ROOT="$MODEL_ROOT" \
MUSUBI_ROOT="$HOME/code/DSAA2012/musubi-tuner" \
DATASET_CONFIG="$TRAIN_CONFIG_ROOT/cast/01.toml" \
OUTPUT_DIR="$LORA_ROOT/qwen-story-cast/01" \
OUTPUT_NAME="pytorch_lora_weights" \
MAX_TRAIN_EPOCHS=1 \
NETWORK_DIM=16 \
bash scripts/qwen_train_story_cast_lora.sh 2>&1 | tee "$ASSET_ROOT/logs/cast_lora_01_smoke.log"
```

## 12. 批量训练 cast LoRA

如果 smoke 正常，再批量跑。

```bash
for toml in "$TRAIN_CONFIG_ROOT"/cast/*.toml; do
  case_id=$(basename "$toml" .toml)
  MODEL_ROOT="$MODEL_ROOT" \
  MUSUBI_ROOT="$HOME/code/DSAA2012/musubi-tuner" \
  DATASET_CONFIG="$toml" \
  OUTPUT_DIR="$LORA_ROOT/qwen-story-cast/${case_id}" \
  OUTPUT_NAME="pytorch_lora_weights" \
  MAX_TRAIN_EPOCHS=4 \
  NETWORK_DIM=16 \
  bash scripts/qwen_train_story_cast_lora.sh 2>&1 | tee "$ASSET_ROOT/logs/cast_lora_${case_id}.log"
done
```

如果时间紧，建议先只跑：

- `01`
- `02`
- `06`
- `08`

先做一批 showcase case。

## 13. 构建最终 one-shot page prompts

### 13.1 page-LoRA-only 版本

```bash
python scripts/qwen_story_build_page_prompts.py \
  --stories data/task_a \
  --out-dir data/qwen_story/page_prompts_page_only \
  --layout vertical \
  --global-lora-path "$LORA_ROOT/qwen-story-page-lora-v1" \
  --global-lora-weight-name "pytorch_lora_weights.safetensors" \
  --global-lora-scale 0.55
```

### 13.2 dual-LoRA 版本

```bash
python scripts/qwen_story_build_page_prompts.py \
  --stories data/task_a \
  --out-dir data/qwen_story/page_prompts_dual \
  --layout vertical \
  --global-lora-path "$LORA_ROOT/qwen-story-page-lora-v1" \
  --global-lora-weight-name "pytorch_lora_weights.safetensors" \
  --global-lora-scale 0.45 \
  --cast-lora-root "$LORA_ROOT/qwen-story-cast" \
  --cast-lora-weight-name "pytorch_lora_weights.safetensors" \
  --cast-lora-scale 0.85
```

说明：

- 这里默认 page LoRA 和 cast LoRA 都输出为 `pytorch_lora_weights.safetensors`
- 如果你的训练器实际写出的是别的文件名，就把这里的 `weight-name` 改成真实文件名

### 13.3 base Qwen 对照

```bash
python scripts/qwen_story_build_page_prompts.py \
  --stories data/task_a \
  --out-dir data/qwen_story/page_prompts_base \
  --layout vertical
```

## 14. 跑最终生成

### 14.1 base Qwen

```bash
python scripts/qwen_story_infer.py \
  --prompts data/qwen_story/page_prompts_base/page_prompts_3scene.jsonl \
  --out-dir outputs/qwen_story/base_qwen \
  --model "$MODEL_ROOT" \
  --dtype bfloat16 \
  --device cuda \
  --steps 28 \
  --guidance 4.5 \
  --num-candidates 4
```

### 14.2 page-LoRA-only

```bash
python scripts/qwen_story_infer.py \
  --prompts data/qwen_story/page_prompts_page_only/page_prompts_3scene.jsonl \
  --out-dir outputs/qwen_story/page_lora_only \
  --model "$MODEL_ROOT" \
  --dtype bfloat16 \
  --device cuda \
  --steps 28 \
  --guidance 4.5 \
  --num-candidates 4
```

### 14.3 dual-LoRA

```bash
python scripts/qwen_story_infer.py \
  --prompts data/qwen_story/page_prompts_dual/page_prompts_3scene.jsonl \
  --out-dir outputs/qwen_story/dual_lora \
  --model "$MODEL_ROOT" \
  --dtype bfloat16 \
  --device cuda \
  --steps 28 \
  --guidance 4.5 \
  --num-candidates 4
```

## 15. 自动排序并导出最佳候选

### 15.1 对 dual-LoRA 排序

```bash
python scripts/qwen_story_rank_candidates.py \
  --image-dir outputs/qwen_story/dual_lora \
  --prompts data/qwen_story/page_prompts_dual/page_prompts_3scene.jsonl \
  --layout vertical \
  --num-panels 3 \
  --out outputs/qwen_story/metrics/dual_lora_ranked.csv
```

### 15.2 导出每个 prompt 的 top-1

```bash
python scripts/qwen_story_export_top_candidates.py \
  --ranking outputs/qwen_story/metrics/dual_lora_ranked.csv \
  --out-dir outputs/qwen_story/dual_lora_top1 \
  --clean
```

也建议对 `base_qwen` 和 `page_lora_only` 各跑一遍 ranking。

## 16. 评估计划

### 16.1 分辨率/目录检查

```bash
find outputs/qwen_story/dual_lora_top1 -name "*.png" | wc -l
```

### 16.2 CLIPScore

```bash
python scripts/sd3_eval_clipscore.py \
  --image-dir outputs/qwen_story/dual_lora_top1 \
  --prompts data/qwen_story/page_prompts_dual/page_prompts_3scene.jsonl \
  --out outputs/qwen_story/metrics/dual_lora_top1_clipscore.csv
```

### 16.3 DreamSim 跨 panel 一致性

```bash
python scripts/sd3_split_panels.py \
  --image-dir outputs/qwen_story/dual_lora_top1 \
  --out-dir outputs/qwen_story/panels_split/dual_lora_top1 \
  --num-panels 3 \
  --layout vertical

python scripts/sd3_eval_dreamsim.py \
  --panel-dir outputs/qwen_story/panels_split/dual_lora_top1 \
  --out outputs/qwen_story/metrics/dual_lora_top1_dreamsim.csv
```

### 16.4 human eval

建议比较这三组：

- `outputs/qwen_story/base_qwen_top1`
- `outputs/qwen_story/page_lora_only_top1`
- `outputs/qwen_story/dual_lora_top1`

如果你还保留了 SD3 结果，也可以一起放进人工评测里作为失败对照。

## 17. 推荐实验顺序

最省时间的顺序：

1. page LoRA smoke
2. page LoRA full
3. cast seed prompts
4. 只做 `01/02/06/08` 四个 case 的 cast LoRA
5. 先跑这四个 case 的 dual-LoRA one-shot 结果
6. 如果明显优于 page-LoRA-only，再扩到全部 case

## 18. 经验值级别的时间预算

以 `A800 40GB / 80GB` 为参考：

- page LoRA smoke：`0.5 - 1.5` 小时
- page LoRA full：`6 - 12` 小时
- 12 张 cast seed 生成 / case：`10 - 30` 分钟
- cast LoRA smoke / case：`10 - 30` 分钟
- cast LoRA full / case：`20 - 60` 分钟
- one-shot 推理 4 candidates / case：分钟级到十几分钟级

如果只有 40GB 切片：

- 优先 `resolution = 512`
- 保留 `batch_size = 1`
- 必要时加 `--fp8_base --fp8_scaled --blocks_to_swap 16`

## 19. 最终建议

如果时间真的很紧：

- 不要一上来全量 16 个 case 都训 cast LoRA
- 先做 4 个代表性 case
- 先证明 dual-LoRA 比 page-LoRA-only 更稳
- 再决定是否扩全量

这条路线的卖点是：

- 最终结果确实依赖**训练后的参数适配**
- 一致性不是只靠 prompt，而是靠：
  - 全局 page LoRA 学版式/风格
  - story-specific cast LoRA 学角色身份
- 最终一次成图时同时加载两个 LoRA
