"""Split one-shot storyboard outputs into panels for consistency metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(args.image_dir.glob("*.png")):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        for index in range(args.num_panels):
            if args.layout == "horizontal":
                left = int(index * w / args.num_panels)
                right = int((index + 1) * w / args.num_panels)
                crop = image.crop((left, 0, right, h))
            else:
                top = int(index * h / args.num_panels)
                bottom = int((index + 1) * h / args.num_panels)
                crop = image.crop((0, top, w, bottom))
            out_path = args.out_dir / f"{path.stem}_panel{index + 1}.png"
            crop.save(out_path)
            count += 1
    print(f"Saved {count} panel crop(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
