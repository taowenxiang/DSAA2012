"""Collect selected SD3 result images and metric CSVs for report/presentation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/sd3_story"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sd3_story/report_assets"))
    parser.add_argument("--max-per-method", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for method_dir in sorted([item for item in args.outputs_root.iterdir() if item.is_dir()]):
        if method_dir.name in {"metrics", "panels_split", "report_assets"}:
            continue
        target_method_dir = args.out_dir / method_dir.name
        target_method_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(method_dir.glob("*.png"))[: args.max_per_method]:
            shutil.copy2(path, target_method_dir / path.name)
            copied += 1
    metrics_dir = args.outputs_root / "metrics"
    if metrics_dir.exists():
        target_metrics = args.out_dir / "metrics"
        target_metrics.mkdir(parents=True, exist_ok=True)
        for path in sorted(metrics_dir.glob("*.csv")):
            shutil.copy2(path, target_metrics / path.name)
            copied += 1
    print(f"Copied {copied} asset file(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
