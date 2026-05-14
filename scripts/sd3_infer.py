"""Run SD3 Medium one-shot or independent prompt inference with optional LoRA adapters."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from sd3_story_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--lora-path", action="append", default=[], help="LoRA directory; repeat for multiple adapters.")
    parser.add_argument("--lora-weight-name", action="append", default=[])
    parser.add_argument("--lora-scale", action="append", type=float, default=[])
    parser.add_argument("--trigger-prefix", default="")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    load_lora_parameters = inspect.signature(pipe.load_lora_weights).parameters
    for index, lora_path in enumerate(args.lora_path):
        path = Path(lora_path)
        if not path.exists():
            raise SystemExit(f"LoRA path does not exist: {path}")
        adapter_name = f"adapter_{index}"
        kwargs: dict[str, Any] = {}
        if "adapter_name" in load_lora_parameters:
            kwargs["adapter_name"] = adapter_name
        if (
            "weight_name" in load_lora_parameters
            and index < len(args.lora_weight_name)
            and args.lora_weight_name[index]
        ):
            kwargs["weight_name"] = args.lora_weight_name[index]
        pipe.load_lora_weights(str(path), **kwargs)
        adapter_names.append(adapter_name)
        adapter_weights.append(args.lora_scale[index] if index < len(args.lora_scale) else 1.0)
    if len(adapter_names) > 1 and not hasattr(pipe, "set_adapters"):
        raise SystemExit(
            "Multiple LoRA adapters require a Diffusers pipeline with set_adapters(). "
            "Please upgrade Diffusers on HPC."
        )
    if adapter_names and hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except TypeError:
            pipe.set_adapters(adapter_names, adapter_weights)
    elif adapter_names and hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora(lora_scale=adapter_weights[0])

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def main() -> int:
    import torch

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.prompts)
    if not rows:
        raise SystemExit(f"No prompt rows found: {args.prompts}")

    pipe = load_pipeline(args)
    generator_device = "cuda" if torch.cuda.is_available() and not args.cpu_offload else "cpu"

    for item in rows:
        prompt_id = item["id"]
        prompt = item["prompt"]
        if args.trigger_prefix and args.trigger_prefix not in prompt:
            prompt = f"{args.trigger_prefix}, {prompt}"
        negative_prompt = item.get("negative_prompt", "")

        for candidate in range(args.num_candidates):
            seed = args.seed + candidate
            out_path = args.out_dir / f"{prompt_id}_cand{candidate}_seed{seed}.png"
            if args.skip_existing and out_path.exists():
                print(f"Skip existing {out_path}")
                continue
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(item.get("width", args.width)),
                height=int(item.get("height", args.height)),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
            image.save(out_path)
            print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
