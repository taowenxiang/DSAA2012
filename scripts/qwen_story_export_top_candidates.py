"""Copy the top-ranked candidate for each prompt into a final directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
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
    manifest_rows: list[dict[str, object]] = []
    for _, row in df[df["rank_within_prompt"] == 1].iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        target = args.out_dir / image_path.name
        shutil.copy2(image_path, target)
        manifest_rows.append(
            {
                "prompt_id": str(row.get("prompt_id", "")),
                "case_id": str(row.get("case_id", "")),
                "candidate": int(row.get("candidate", -1)),
                "source_path": image_path.as_posix(),
                "exported_path": target.as_posix(),
                "combined_score": float(row.get("combined_score", 0.0)),
                "layout_score": float(row.get("layout_score", 0.0)),
                "perceptual_score": float(row.get("perceptual_score", 0.0)),
                "color_score": float(row.get("color_score", 0.0)),
                "quality_score": float(row.get("quality_score", 0.0)),
            }
        )
        exported += 1

    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_version": "qwen-story-page-top1-v1",
            "exported_count": exported,
            "records": manifest_rows,
        }
        args.manifest_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Exported {exported} top candidate image(s) to {args.out_dir}")
    if args.manifest_out is not None:
        print(f"Wrote export manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
