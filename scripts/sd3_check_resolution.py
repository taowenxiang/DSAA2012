"""Check generated image resolutions against a minimum short side."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--min-short", type=int, default=512)
    parser.add_argument("--fail-on-bad", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bad: list[Path] = []
    for path in sorted(args.dir.glob("*.png")):
        image = Image.open(path)
        w, h = image.size
        ok = min(w, h) >= args.min_short
        print(f"{path.name}: {w}x{h}, short={min(w, h)}, ok={ok}")
        if not ok:
            bad.append(path)
    if bad and args.fail_on_bad:
        raise SystemExit(f"{len(bad)} image(s) failed minimum short side {args.min_short}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
