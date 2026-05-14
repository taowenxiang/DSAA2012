"""Compose independently generated scene images into storyboard pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SCENE_RE = re.compile(r"^(?P<story>.+)_scene(?P<panel>\d+)_cand(?P<candidate>\d+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--panel-height", type=int, default=512)
    parser.add_argument("--candidate", type=int, default=0)
    return parser.parse_args()


def group_images(image_dir: Path, num_panels: int, candidate: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(image_dir.glob("*.png")):
        match = SCENE_RE.match(path.stem)
        if not match:
            continue
        if int(match.group("candidate")) != candidate:
            continue
        story_id = match.group("story")
        panel = int(match.group("panel"))
        if panel <= num_panels:
            groups.setdefault(story_id, []).append(path)
    return groups


def compose(paths: list[Path], out_path: Path, layout: str, panel_width: int, panel_height: int) -> None:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(ImageOps.fit(image, (panel_width, panel_height), method=Image.Resampling.LANCZOS))

    if layout == "horizontal":
        canvas = Image.new("RGB", (panel_width * len(images), panel_height), "white")
        for index, image in enumerate(images):
            canvas.paste(image, (index * panel_width, 0))
        draw = ImageDraw.Draw(canvas)
        for index in range(1, len(images)):
            x = index * panel_width
            draw.line((x, 0, x, panel_height), fill=(0, 0, 0), width=5)
    else:
        canvas = Image.new("RGB", (panel_width, panel_height * len(images)), "white")
        for index, image in enumerate(images):
            canvas.paste(image, (0, index * panel_height))
        draw = ImageDraw.Draw(canvas)
        for index in range(1, len(images)):
            y = index * panel_height
            draw.line((0, y, panel_width, y), fill=(0, 0, 0), width=5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    groups = group_images(args.image_dir, args.num_panels, args.candidate)
    count = 0
    for story_id, paths in sorted(groups.items()):
        paths = sorted(paths, key=lambda path: int(SCENE_RE.match(path.stem).group("panel")))  # type: ignore[union-attr]
        if len(paths) != args.num_panels:
            continue
        out_path = args.out_dir / f"{story_id}_{args.num_panels}scene_cand{args.candidate}_composed.png"
        compose(paths, out_path, args.layout, args.panel_width, args.panel_height)
        print(f"Saved {out_path}")
        count += 1
    print(f"Composed {count} storyboard image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
