"""Copy the top-ranked candidate for each prompt into a final directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ranking)
    if "rank_within_prompt" not in df.columns:
        raise SystemExit("ranking CSV is missing rank_within_prompt")

    exported = 0
    for _, row in df[df["rank_within_prompt"] == 1].iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        target = args.out_dir / image_path.name
        shutil.copy2(image_path, target)
        exported += 1

    print(f"Exported {exported} top candidate image(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
