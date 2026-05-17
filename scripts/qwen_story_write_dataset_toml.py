"""Write a musubi-tuner dataset TOML for Qwen LoRA training."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--caption-extension", default=".txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[general]\n"
        "enable_bucket = true\n"
        f"batch_size = {args.batch_size}\n"
        f"num_repeats = {args.num_repeats}\n\n"
        "[[datasets]]\n"
        f"resolution = {args.resolution}\n"
        f'caption_extension = "{args.caption_extension}"\n'
        f'image_directory = "{args.image_dir.resolve().as_posix()}"\n'
    )
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
