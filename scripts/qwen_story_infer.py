"""Run Qwen story page inference with optional global and per-story LoRAs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_image_infer import generate_from_payload
from qwen_story_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen-Image-2512")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-lora-path", default="")
    parser.add_argument("--global-lora-weight-name", default="")
    parser.add_argument("--global-lora-scale", type=float, default=0.55)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def discover_prompt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def output_path_for_row(out_dir: Path, row: dict[str, Any], candidate: int, seed: int) -> Path:
    subdir = str(row.get("output_subdir") or "").strip()
    target_dir = out_dir / subdir if subdir else out_dir
    return target_dir / f"{row['id']}_cand{candidate}_seed{seed}.png"


def build_payload(
    row: dict[str, Any],
    args: argparse.Namespace,
    candidate: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    payload = {
        "prompt": row["prompt"],
        "negative_prompt": row.get("negative_prompt", ""),
        "seed": seed,
        "output_path": output_path.as_posix(),
        "width": int(row.get("width") or args.width or 1024),
        "height": int(row.get("height") or args.height or 1536),
        "model_path": args.model,
        "model_family": "qwen_image",
        "dtype": args.dtype,
        "device": args.device,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "temperature": 1.0,
        "tensor_parallel_size": 1,
        "style_id": "qwen_story",
        "style_prompt": "",
        "style_display_name": "Qwen Story",
        "style_reference_image_path": None,
        "style_backend_requested": "prompt_only",
        "style_lora_path": row.get("global_lora_path") or args.global_lora_path or "",
        "style_lora_weight_name": (
            row.get("global_lora_weight_name")
            or args.global_lora_weight_name
            or ""
        ),
        "style_lora_scale": float(
            row.get("global_lora_scale", args.global_lora_scale)
        ),
        "extra_loras": row.get("extra_loras", []),
        "case_id": row.get("case_id", row["id"]),
        "scene_id": row.get("num_panels", 0),
        "candidate_id": candidate,
        "ip_adapter": {
            "enabled": False,
            "model_path": "",
            "image_encoder_path": "",
            "scale": 1.0,
        },
    }
    return payload


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_files = discover_prompt_files(args.prompts)
    if not prompt_files:
        raise SystemExit(f"No prompt jsonl files found at {args.prompts}")

    generated = 0
    for prompt_file in prompt_files:
        rows = read_jsonl(prompt_file)
        for row_index, row in enumerate(rows):
            for candidate in range(args.num_candidates):
                if "seed" in row:
                    seed = int(row["seed"]) + candidate
                else:
                    seed = args.seed + row_index * 1000 + candidate
                output_path = output_path_for_row(args.out_dir, row, candidate, seed)
                if args.skip_existing and output_path.exists():
                    print(f"Skip existing {output_path}")
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = build_payload(row, args, candidate, seed, output_path)
                result = generate_from_payload(payload)
                print(
                    f"Saved {result['output_path']} "
                    f"(backend={result['style_backend_effective']})"
                )
                generated += 1
    print(f"Generated {generated} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
