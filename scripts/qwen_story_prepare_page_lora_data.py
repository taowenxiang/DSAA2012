"""Prepare vertical multi-panel Qwen page LoRA training data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from qwen_story_utils import (
    DEFAULT_PAGE_STYLE_PROMPT,
    build_page_training_caption,
    clean_text,
    load_stories,
    normalize_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, default=Path("data") / "task_a")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs") / "runs" / "run_0001_storybook" / "final",
            Path("outputs") / "runs" / "run_0003_storybook" / "final",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "page_lora_train",
    )
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="vertical")
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--style-prompt", default=DEFAULT_PAGE_STYLE_PROMPT)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def collect_case_panels(run_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not run_dir.exists():
        return result
    for case_dir in sorted(item for item in run_dir.iterdir() if item.is_dir()):
        panels = sorted(case_dir.glob("scene_*.png"))
        if len(panels) >= 2:
            result[case_dir.name] = panels
    return result


def compose_page(paths: list[Path], layout: str, panel_size: int) -> Image.Image:
    panels = [
        ImageOps.fit(Image.open(path).convert("RGB"), (panel_size, panel_size), method=Image.Resampling.LANCZOS)
        for path in paths
    ]
    if layout == "vertical":
        canvas = Image.new("RGB", (panel_size, panel_size * len(panels)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, panel in enumerate(panels):
            canvas.paste(panel, (0, index * panel_size))
            if index:
                y = index * panel_size
                draw.line((0, y, panel_size, y), fill=(0, 0, 0), width=6)
        return canvas

    canvas = Image.new("RGB", (panel_size * len(panels), panel_size), "white")
    draw = ImageDraw.Draw(canvas)
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel_size, 0))
        if index:
            x = index * panel_size
            draw.line((x, 0, x, panel_size), fill=(0, 0, 0), width=6)
    return canvas


def main() -> int:
    args = parse_args()
    story_map = {str(item["case_id"]): item for item in load_stories(args.stories)}

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    index = 1
    for run_dir in args.run_dirs:
        for case_id, panels in collect_case_panels(run_dir).items():
            story = story_map.get(case_id)
            if story is None:
                continue
            for num_panels in [2, 3]:
                if len(panels) < num_panels:
                    continue
                image = compose_page(panels[:num_panels], args.layout, args.panel_size)
                stem = f"{index:06d}_{normalize_id(case_id)}_{num_panels}panel"
                image_path = args.out_dir / f"{stem}.png"
                caption = clean_text(
                    build_page_training_caption(
                        story=story,
                        num_panels=num_panels,
                        layout=args.layout,
                        style_prompt=args.style_prompt,
                    )
                )
                image.save(image_path)
                (args.out_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
                rows.append(
                    {
                        "file_name": image_path.name,
                        "text": caption,
                        "case_id": case_id,
                        "num_panels": num_panels,
                        "layout": args.layout,
                        "source_run_dir": run_dir.as_posix(),
                    }
                )
                index += 1

    if not rows:
        raise SystemExit("No page LoRA training rows were prepared.")
    manifest_path = args.out_dir / "metadata.jsonl"
    count = write_jsonl(manifest_path, rows)
    print(f"Wrote {count} page training rows to {manifest_path}")
    print(f"Images are in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
