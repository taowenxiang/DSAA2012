# HPC 终端执行手册：Qwen-Image-2512 LoRA 风格训练与接入

这份文档给队友使用，目标是在 **纯终端** 环境下，从 SSH 登录 HPC 之后，把下面这件事做完：

- 保持默认风格 `storybook` 不变
- 新增两个可选风格：`baroque` 和 `shinkai`
- 基于 `Qwen-Image-2512` 做两个独立的 LoRA 微调
- 将 LoRA 权重接入当前仓库的生成管线
- 最后通过 `sbatch` 提交推理任务，用 GPU 跑图

仓库里相关配置已经准备好：

- `configs/style_presets.json`
- `configs/member_b_generation_config.hpc.template.json`
- `docs/hpc_qwen_lora_style_guide.md`

其中：

- `baroque` 的 LoRA 目标路径已经配置为 `artifacts/loras/qwen-image-2512/baroque-lora-v1`
- `shinkai` 的 LoRA 目标路径已经配置为 `artifacts/loras/qwen-image-2512/shinkai-lora-v1`

## 1. 先决条件

默认假设：

1. HPC 使用 Slurm，可以用 `sbatch`
2. 登录节点能联网访问 Hugging Face
3. 队友有自己的 HPC 用户名
4. 如果 HPC 有 `module` 系统，则可加载 Python/CUDA

如果第 2 条不满足，需要先在别的机器下载模型和数据集，再传到 HPC。

## 2. 登录 HPC 后，先建目录

先 SSH 上去：

```bash
ssh YOUR_USER@YOUR_HPC
```

进入之后，统一设置路径变量：

```bash
export PROJ_ROOT=/data/home/$USER/code/DSAA2012/Project
export VENV_ROOT=/data/home/$USER/code/DSAA2012/venvs/dsaa2012
export ASSET_ROOT=/data/scratch/$USER/dsaa2012_assets
export HF_HOME=$ASSET_ROOT/hf_cache
```

创建目录：

```bash
mkdir -p /data/home/$USER/code/DSAA2012
mkdir -p /data/home/$USER/code/DSAA2012/venvs
mkdir -p $ASSET_ROOT/hf_cache
mkdir -p $ASSET_ROOT/models
mkdir -p $ASSET_ROOT/datasets/raw
mkdir -p $ASSET_ROOT/datasets/prepared
mkdir -p $ASSET_ROOT/loras/qwen-image-2512
mkdir -p $ASSET_ROOT/logs/train
```

## 3. clone 仓库

```bash
cd /data/home/$USER/code/DSAA2012
git clone <仓库地址> Project
cd "$PROJ_ROOT"
```

如果仓库已经 clone 过，就直接：

```bash
cd "$PROJ_ROOT"
git pull
```

## 4. 准备 Python 环境

如果集群有 module：

```bash
module purge
module load python/3.10
module load cuda/12.1
```

然后建立虚拟环境：

```bash
python -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
pip install --upgrade pip
```

安装依赖：

```bash
pip install -r requirements.txt
pip install "datasets>=2.20.0" huggingface_hub peft torchvision sentencepiece
pip install git+https://github.com/huggingface/diffusers
pip install "transformers>=4.51.3"
pip install bitsandbytes
```

## 5. 给仓库挂软链接

仓库里使用 `artifacts/` 指向大文件目录。

```bash
cd "$PROJ_ROOT"
mkdir -p artifacts
ln -sfn $ASSET_ROOT/models artifacts/models
ln -sfn $ASSET_ROOT/datasets artifacts/datasets
ln -sfn $ASSET_ROOT/loras artifacts/loras
```

检查：

```bash
ls -l artifacts
```

应该能看到：

- `artifacts/models -> /data/scratch/.../models`
- `artifacts/datasets -> /data/scratch/.../datasets`
- `artifacts/loras -> /data/scratch/.../loras`

## 6. 下载 Qwen-Image-2512 基座模型

```bash
export HF_HOME=$ASSET_ROOT/hf_cache

huggingface-cli download Qwen/Qwen-Image-2512 \
  --local-dir $ASSET_ROOT/models/Qwen-Image-2512
```

下载后检查：

```bash
ls $ASSET_ROOT/models/Qwen-Image-2512
```

应至少看到类似目录：

- `scheduler`
- `text_encoder`
- `tokenizer`
- `transformer`
- `vae`

## 7. 准备 HPC 推理配置

复制模板：

```bash
cd "$PROJ_ROOT"
cp configs/member_b_generation_config.hpc.template.json configs/member_b_generation_config.hpc.json
```

编辑：

```bash
nano configs/member_b_generation_config.hpc.json
```

需要重点修改：

- `scheduler.partition`
- `scheduler.account`
- `scheduler.gpus_per_task`
- `scheduler.mem_gb`
- `scheduler.time_limit`

通常建议先设：

- `gpus_per_task = 1`
- `mem_gb = 96`
- `time_limit = 08:00:00`

`model_path` 保持为：

```text
artifacts/models/Qwen-Image-2512
```

## 8. 下载两个训练数据集

目标数据集：

- `FulcoPin/latin-american-baroque-18k-multimodal`
- `Fung804/makoto-shinkai-picture`

下载命令：

```bash
huggingface-cli download --repo-type dataset FulcoPin/latin-american-baroque-18k-multimodal \
  --local-dir $ASSET_ROOT/datasets/raw/latin-american-baroque-18k-multimodal

huggingface-cli download --repo-type dataset Fung804/makoto-shinkai-picture \
  --local-dir $ASSET_ROOT/datasets/raw/makoto-shinkai-picture
```

## 9. 导出训练数据为“图片 + 同名 txt caption”

为了兼容 Qwen-Image LoRA 训练，建议把数据处理成：

```text
000001.png
000001.txt
000002.png
000002.txt
...
```

### 9.1 先写一个导出脚本

```bash
cat > /tmp/export_style_dataset.py <<'PY'
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    image_col = "image" if "image" in ds.column_names else None
    if image_col is None:
        for k, v in ds.features.items():
            if v.__class__.__name__ == "Image":
                image_col = k
                break
    if image_col is None:
        raise SystemExit(f"No image column found in {args.dataset}: {ds.column_names}")

    text_col = None
    for k in ["text", "caption", "prompt", "description"]:
        if k in ds.column_names:
            text_col = k
            break

    if args.max_rows > 0:
        ds = ds.shuffle(seed=2012).select(range(min(args.max_rows, len(ds))))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        img = row[image_col]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        else:
            img = Image.open(img).convert("RGB")
        stem = f"{i:06d}"
        img.save(outdir / f"{stem}.png")
        base = str(row[text_col]).strip() if text_col else ""
        text = args.prefix if not base else f"{args.prefix}, {base}"
        (outdir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")

    print(f"exported {len(ds)} items to {outdir}")

if __name__ == "__main__":
    main()
PY
```

### 9.2 导出 baroque 数据

建议先抽样 3000 张：

```bash
source "$VENV_ROOT/bin/activate"

python /tmp/export_style_dataset.py \
  --dataset FulcoPin/latin-american-baroque-18k-multimodal \
  --outdir $ASSET_ROOT/datasets/prepared/baroque_v1 \
  --max-rows 3000 \
  --prefix "latin american baroque painting, ornate gilded detail, dramatic chiaroscuro, theatrical composition"
```

### 9.3 导出 shinkai 数据

建议先全量导出：

```bash
python /tmp/export_style_dataset.py \
  --dataset Fung804/makoto-shinkai-picture \
  --outdir $ASSET_ROOT/datasets/prepared/shinkai_v1 \
  --max-rows 1347 \
  --prefix "shinkai-inspired anime film scene, luminous sky, cinematic clouds, emotional lighting, reflective color transitions"
```

### 9.4 检查导出结果

```bash
ls $ASSET_ROOT/datasets/prepared/baroque_v1 | head
ls $ASSET_ROOT/datasets/prepared/shinkai_v1 | head
```

应该同时能看到 `.png` 和 `.txt` 文件。

## 10. 安装 LoRA trainer

推荐用 `musubi-tuner`。

```bash
cd /data/home/$USER/code/DSAA2012
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
source "$VENV_ROOT/bin/activate"
pip install -e .
```

## 11. 写 dataset TOML

回到项目目录：

```bash
cd "$PROJ_ROOT"
mkdir -p train_configs
```

### 11.1 `baroque.toml`

```bash
cat > $PROJ_ROOT/train_configs/baroque.toml <<EOF
[general]
enable_bucket = true
batch_size = 1
num_repeats = 1

[[datasets]]
resolution = 512
caption_extension = ".txt"
image_directory = "$ASSET_ROOT/datasets/prepared/baroque_v1"
EOF
```

### 11.2 `shinkai.toml`

```bash
cat > $PROJ_ROOT/train_configs/shinkai.toml <<EOF
[general]
enable_bucket = true
batch_size = 1
num_repeats = 1

[[datasets]]
resolution = 512
caption_extension = ".txt"
image_directory = "$ASSET_ROOT/datasets/prepared/shinkai_v1"
EOF
```

## 12. 写 `sbatch` 训练脚本

```bash
cat > $PROJ_ROOT/scripts/train_style_lora.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=qwen-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=/data/scratch/%u/dsaa2012_assets/logs/train/%x-%j.out

set -euo pipefail

source /data/home/$USER/code/DSAA2012/venvs/dsaa2012/bin/activate
export PROJ_ROOT=/data/home/$USER/code/DSAA2012/Project
export ASSET_ROOT=/data/scratch/$USER/dsaa2012_assets
export MODEL_ROOT=$ASSET_ROOT/models/Qwen-Image-2512
export MUSUBI_ROOT=/data/home/$USER/code/DSAA2012/musubi-tuner
export HF_HOME=$ASSET_ROOT/hf_cache

STYLE=${STYLE:?STYLE not set}

if [ "$STYLE" = "baroque" ]; then
  DATASET_TOML=$PROJ_ROOT/train_configs/baroque.toml
  OUTPUT_DIR=$ASSET_ROOT/loras/qwen-image-2512/baroque-lora-v1
  OUTPUT_NAME=baroque-lora
elif [ "$STYLE" = "shinkai" ]; then
  DATASET_TOML=$PROJ_ROOT/train_configs/shinkai.toml
  OUTPUT_DIR=$ASSET_ROOT/loras/qwen-image-2512/shinkai-lora-v1
  OUTPUT_NAME=shinkai-lora
else
  echo "unknown STYLE=$STYLE"
  exit 1
fi

cd "$MUSUBI_ROOT"
accelerate launch --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
  --dit "$MODEL_ROOT/transformer" \
  --vae "$MODEL_ROOT/vae" \
  --text_encoder "$MODEL_ROOT/text_encoder" \
  --model_version original \
  --dataset_config "$DATASET_TOML" \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift \
  --weighting_scheme none --discrete_flow_shift 2.2 \
  --optimizer_type adamw8bit --learning_rate 5e-5 --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_qwen_image \
  --network_dim 16 \
  --max_train_epochs 12 --save_every_n_epochs 1 --seed 2012 \
  --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME"
EOF
```

赋予执行权限：

```bash
chmod +x $PROJ_ROOT/scripts/train_style_lora.sbatch
```

### 12.1 重要：改 partition 和 account

打开：

```bash
nano $PROJ_ROOT/scripts/train_style_lora.sbatch
```

把下面两行改成 HPC 实际参数：

```text
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
```

## 13. 先做一次 smoke test

先不要直接完整训练，先把 `max_train_epochs 12` 改成 `1`，测是否能正常产出 checkpoint。

如果 GPU 显存小于 40GB，建议在训练命令后再追加：

```text
--fp8_base --fp8_scaled
```

如果仍然 OOM，再追加：

```text
--blocks_to_swap 16
```

## 14. 提交训练任务

### 14.1 训练 `baroque`

```bash
sbatch --export=ALL,STYLE=baroque $PROJ_ROOT/scripts/train_style_lora.sbatch
```

### 14.2 训练 `shinkai`

```bash
sbatch --export=ALL,STYLE=shinkai $PROJ_ROOT/scripts/train_style_lora.sbatch
```

## 15. 查看任务状态和日志

查看队列：

```bash
squeue -u $USER
```

查看输出日志：

```bash
ls $ASSET_ROOT/logs/train
tail -f $ASSET_ROOT/logs/train/qwen-lora-<jobid>.out
```

## 16. 训练完成后，整理 LoRA 权重文件

仓库当前默认希望看到：

```text
artifacts/loras/qwen-image-2512/baroque-lora-v1/pytorch_lora_weights.safetensors
artifacts/loras/qwen-image-2512/shinkai-lora-v1/pytorch_lora_weights.safetensors
```

先找实际输出文件：

```bash
find $ASSET_ROOT/loras/qwen-image-2512/baroque-lora-v1 -name "*.safetensors"
find $ASSET_ROOT/loras/qwen-image-2512/shinkai-lora-v1 -name "*.safetensors"
```

如果 trainer 输出的是：

- `baroque-lora.safetensors`
- `shinkai-lora.safetensors`

则复制成仓库预期文件名：

```bash
cp $ASSET_ROOT/loras/qwen-image-2512/baroque-lora-v1/baroque-lora.safetensors \
   $ASSET_ROOT/loras/qwen-image-2512/baroque-lora-v1/pytorch_lora_weights.safetensors

cp $ASSET_ROOT/loras/qwen-image-2512/shinkai-lora-v1/shinkai-lora.safetensors \
   $ASSET_ROOT/loras/qwen-image-2512/shinkai-lora-v1/pytorch_lora_weights.safetensors
```

如果实际输出名不一样，也可以不复制，改：

```text
configs/style_presets.json
```

里的 `lora_weight_name`。

## 17. 用仓库现有管线生成 baroque 风格结果

### 17.1 先生成一次 run

```bash
cd "$PROJ_ROOT"
source "$VENV_ROOT/bin/activate"

python scripts/run_story_pipeline.py --style baroque --placeholder-images
RUN_DIR=$(ls -dt outputs/runs/run_*_baroque | head -1)
echo "$RUN_DIR"
```

### 17.2 把 placeholder manifest 改成真实 HPC 推理任务

```bash
python scripts/generate_images.py \
  --style baroque \
  --run-dir "$RUN_DIR" \
  --config configs/member_b_generation_config.hpc.json \
  --run-model
```

### 17.3 提交数组任务

```bash
sbatch "$RUN_DIR/intermediate/hpc_jobs/submit_member_b_array.slurm"
```

### 17.4 推理完成后 rerank 和打包

```bash
python scripts/rerank_candidates.py --style baroque --run-dir "$RUN_DIR" --backend auto
python scripts/package_outputs.py --style baroque --run-dir "$RUN_DIR" --clean
```

## 18. 生成 shinkai 风格结果

步骤完全一样，只把 `baroque` 改成 `shinkai`：

```bash
python scripts/run_story_pipeline.py --style shinkai --placeholder-images
RUN_DIR=$(ls -dt outputs/runs/run_*_shinkai | head -1)
echo "$RUN_DIR"

python scripts/generate_images.py \
  --style shinkai \
  --run-dir "$RUN_DIR" \
  --config configs/member_b_generation_config.hpc.json \
  --run-model

sbatch "$RUN_DIR/intermediate/hpc_jobs/submit_member_b_array.slurm"

python scripts/rerank_candidates.py --style shinkai --run-dir "$RUN_DIR" --backend auto
python scripts/package_outputs.py --style shinkai --run-dir "$RUN_DIR" --clean
```

## 19. 推荐执行顺序

建议按下面顺序做，失败时更容易定位问题：

1. clone 仓库并装环境
2. 下载 Qwen-Image-2512 基座模型
3. 配 `configs/member_b_generation_config.hpc.json`
4. 下载并导出 `shinkai` 数据集
5. 先对 `shinkai` 做 1 epoch smoke test
6. 确认 LoRA 文件能生成
7. 再跑完整 `shinkai` 训练
8. 再跑 `baroque`
9. 最后分别做 `baroque` / `shinkai` 的 HPC 推理

## 20. 常见问题

### 20.1 `huggingface-cli` 下载失败

可能原因：

- 登录节点不能联网
- 没有登录 Hugging Face
- 某些模型或数据集访问受限

可尝试：

```bash
huggingface-cli login
```

### 20.2 训练时 OOM

优先级如下：

1. 加 `--fp8_base --fp8_scaled`
2. 加 `--blocks_to_swap 16`
3. 把 `network_dim` 从 `16` 降到 `8`
4. 减少分辨率

### 20.3 LoRA 权重路径对不上

检查：

```bash
cat configs/style_presets.json
```

重点确认：

- `lora_path`
- `lora_weight_name`

### 20.4 推理时加载不到 LoRA

检查：

```bash
ls artifacts/loras/qwen-image-2512/baroque-lora-v1
ls artifacts/loras/qwen-image-2512/shinkai-lora-v1
```

以及：

```bash
python -m compileall scripts
```

## 21. 参考

- Qwen-Image 官方仓库: <https://github.com/QwenLM/Qwen-Image>
- Qwen-Image-2512 模型页: <https://huggingface.co/Qwen/Qwen-Image-2512>
- musubi-tuner Qwen-Image 文档: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md>
- Baroque 数据集: <https://huggingface.co/datasets/FulcoPin/latin-american-baroque-18k-multimodal>
- Makoto Shinkai 数据集: <https://huggingface.co/datasets/Fung804/makoto-shinkai-picture>

