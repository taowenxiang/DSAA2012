"""Evaluate cross-panel visual similarity with DreamSim distance."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def group_panels(panel_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(panel_dir.glob("*.png")):
        base = path.stem.rsplit("_panel", 1)[0]
        groups.setdefault(base, []).append(path)
    return groups


def main() -> int:
    args = parse_args()

    import torch
    from dreamsim import dreamsim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device)
    rows: list[dict] = []
    for base, paths in group_panels(args.panel_dir).items():
        paths = sorted(paths)
        if len(paths) < 2:
            continue
        tensors = {path: preprocess(Image.open(path).convert("RGB")).to(device) for path in paths}
        distances: list[float] = []
        for left, right in itertools.combinations(paths, 2):
            with torch.no_grad():
                distance = model(tensors[left], tensors[right]).detach().cpu().item()
            distances.append(distance)
            rows.append(
                {
                    "base": base,
                    "panel_a": left.as_posix(),
                    "panel_b": right.as_posix(),
                    "dreamsim_distance": distance,
                }
            )
        rows.append(
            {
                "base": base,
                "panel_a": "MEAN",
                "panel_b": "MEAN",
                "dreamsim_distance": sum(distances) / len(distances),
            }
        )
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
