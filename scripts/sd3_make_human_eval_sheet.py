"""Create a CSV rubric sheet for SD3 story human evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "method",
    "image_path",
    "character_consistency_1_5",
    "color_consistency_1_5",
    "story_coherence_1_5",
    "prompt_alignment_1_5",
    "visual_naturalness_1_5",
    "panel_clarity_1_5",
    "overall_preference_1_5",
    "comments",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/sd3_story/human_eval/human_eval_sheet.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    for image_dir in args.image_dirs:
        method = image_dir.name
        for path in sorted(image_dir.glob("*.png")):
            rows.append({field: "" for field in FIELDS} | {"method": method, "image_path": path.as_posix()})
    if not rows:
        raise SystemExit("No images found for human evaluation sheet.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} row(s): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
