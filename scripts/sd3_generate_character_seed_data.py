"""Generate fixed-character seed images for SD3 Character LoRA training."""

from __future__ import annotations

import argparse
from pathlib import Path

from sd3_story_utils import TRIGGER_CHARACTER, write_jsonl


POSES = [
    "standing and smiling in a cozy kitchen",
    "walking under light rain with a red backpack",
    "holding a small book beside a window",
    "running toward a bus stop",
    "sitting at a table with warm morning light",
    "waving gently in a quiet garden",
    "kneeling to help a small puppy",
    "looking curious in a city street",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sd3_story/train_character"))
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--num-images", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from diffusers import StableDiffusion3Pipeline
    from sd3_infer import torch_dtype

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")

    rows: list[dict] = []
    generator_device = "cuda" if torch.cuda.is_available() and not args.cpu_offload else "cpu"
    negative = "low quality, blurry, distorted face, bad hands, text, watermark, extra limbs"
    for index in range(args.num_images):
        pose = POSES[index % len(POSES)]
        prompt = (
            f"a {TRIGGER_CHARACTER} character, young girl with short black hair, yellow raincoat, "
            f"red backpack, {pose}, consistent storybook illustration, clean full-body character reference, no text"
        )
        seed = args.seed + index
        out_path = args.out_dir / f"{index + 1:06d}.png"
        txt_path = args.out_dir / f"{index + 1:06d}.txt"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        image.save(out_path)
        txt_path.write_text(prompt + "\n", encoding="utf-8")
        rows.append(
            {
                "file_name": out_path.name,
                "text": prompt,
                "source": "sd3_generated_character_seed",
                "seed": seed,
                "license_note": "Generated locally from SD3 Medium for course Character LoRA training.",
            }
        )
        print(f"Saved {out_path}")
    write_jsonl(args.out_dir / "metadata.jsonl", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
