"""Summarize completed SD3 human evaluation CSV scores by method."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = [
    "character_consistency_1_5",
    "color_consistency_1_5",
    "story_coherence_1_5",
    "prompt_alignment_1_5",
    "visual_naturalness_1_5",
    "panel_clarity_1_5",
    "overall_preference_1_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=Path("data/sd3_story/human_eval/human_eval_sheet.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sd3_story/metrics/human_eval_summary.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.sheet)
    missing = [column for column in ["method", *SCORE_COLUMNS] if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s): {', '.join(missing)}")
    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    summary = df.groupby("method", dropna=False)[SCORE_COLUMNS].mean().reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
