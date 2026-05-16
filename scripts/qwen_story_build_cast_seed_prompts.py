"""Build per-story cast seed prompts for Qwen cast LoRA training."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_story_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PAGE_STYLE_PROMPT,
    build_cast_seed_rows,
    load_stories,
    normalize_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, default=Path("data") / "task_a")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "cast_seed_prompts",
    )
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--style-prompt", default=DEFAULT_PAGE_STYLE_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stories = sorted(load_stories(args.stories), key=lambda item: str(item["case_id"]))
    written = 0
    for story in stories:
        rows = build_cast_seed_rows(
            story=story,
            num_images=args.num_images,
            width=args.width,
            height=args.height,
            style_prompt=args.style_prompt,
            negative_prompt=args.negative_prompt,
        )
        if not rows:
            continue
        out_path = args.out_dir / f"{normalize_id(story['case_id'])}.jsonl"
        written += write_jsonl(out_path, rows)
        print(f"Wrote {out_path}")
    print(f"Prepared {written} cast seed prompt rows in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
