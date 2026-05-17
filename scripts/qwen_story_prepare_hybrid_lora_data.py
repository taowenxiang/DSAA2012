"""Prepare a hybrid page + panel-crop LoRA dataset from raw three-panel pages."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_STYLE_PROMPT = (
    "consistent cinematic storybook illustration, painterly detail, coherent warm color palette, "
    "clear narrative composition, expressive faces, clean panel storytelling"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data") / "raw" / "pictures")
    parser.add_argument("--story-text", type=Path, default=Path("data") / "raw" / "text" / "raw.txt")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "page_lora_hybrid_train",
    )
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument(
        "--paired-text",
        action="store_true",
        help="Read one txt prompt per image from image_dir instead of a shared raw story text file.",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append metadata to an existing dataset directory instead of overwriting metadata.jsonl.",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def clean_scene_text(text: str) -> str:
    cleaned = re.sub(r"\[SCENE-\d+\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lstrip(" ,")
    if cleaned and cleaned[0].islower():
        cleaned = f"a recurring character {cleaned}"
    return cleaned


def parse_raw_story_blocks(raw_text: str) -> list[list[str]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw_text) if block.strip()]
    stories: list[list[str]] = []
    for block in blocks:
        parts = [clean_scene_text(part) for part in block.split("[SEP]")]
        scenes = [part for part in parts if part]
        if scenes:
            stories.append(scenes)
    return stories


def parse_page_prompt_text(prompt_text: str) -> list[str]:
    marker = prompt_text.find("[SCENE-1]")
    if marker != -1:
        prompt_text = prompt_text[marker:]
    blocks = parse_raw_story_blocks(prompt_text)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one story block in page prompt text, found {len(blocks)}")
    return blocks[0]


def numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def split_vertical_panels(image: Image.Image, num_panels: int = 3) -> list[Image.Image]:
    width, height = image.size
    panels: list[Image.Image] = []
    for index in range(num_panels):
        top = int(index * height / num_panels)
        bottom = int((index + 1) * height / num_panels)
        panels.append(image.crop((0, top, width, bottom)))
    return panels


def build_page_caption(scenes: list[str], style_prompt: str = DEFAULT_STYLE_PROMPT) -> str:
    panel_labels = ["top panel", "middle panel", "bottom panel"]
    scene_parts = [f"{panel_labels[index]}: {scene}" for index, scene in enumerate(scenes)]
    return (
        "three-panel storybook page, stacked vertical panels, clear gutters, top-to-bottom narrative flow, "
        "same recurring cast across all panels, "
        f"{style_prompt}, "
        + ", ".join(scene_parts)
    )


def build_panel_caption(
    scene: str,
    panel_index: int,
    num_panels: int = 3,
    style_prompt: str = DEFAULT_STYLE_PROMPT,
) -> str:
    panel_count_phrase = "three-panel" if num_panels == 3 else f"{num_panels}-panel"
    return (
        f"storybook illustration, panel {panel_index} of a continuous {panel_count_phrase} story, "
        "same recurring cast from the source page, "
        f"{style_prompt}, "
        f"{scene}"
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_story_scene_lists(
    image_paths: list[Path],
    story_text_path: Path,
    paired_text: bool,
) -> list[list[str]]:
    if paired_text:
        stories: list[list[str]] = []
        for image_path in image_paths:
            txt_path = image_path.with_suffix(".txt")
            if not txt_path.exists():
                raise SystemExit(f"Missing paired txt for image: {image_path}")
            stories.append(parse_page_prompt_text(txt_path.read_text(encoding="utf-8")))
        return stories

    return parse_raw_story_blocks(story_text_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.image_dir.glob("*.png"), key=numeric_sort_key)
    stories = load_story_scene_lists(
        image_paths=image_paths,
        story_text_path=args.story_text,
        paired_text=args.paired_text,
    )

    if len(stories) != len(image_paths):
        raise SystemExit(
            f"Story count ({len(stories)}) does not match image count ({len(image_paths)})."
        )

    metadata_path = args.out_dir / "metadata.jsonl"
    metadata_rows: list[dict] = read_jsonl(metadata_path) if args.append else []
    sample_index = args.start_index

    for story_index, (story_scenes, image_path) in enumerate(zip(stories, image_paths, strict=True), start=1):
        if len(story_scenes) != args.num_panels:
            raise SystemExit(
                f"Story {story_index} has {len(story_scenes)} scene(s); expected {args.num_panels}."
            )

        image = Image.open(image_path).convert("RGB")
        page_stem = f"{sample_index:06d}_page"
        page_caption = build_page_caption(story_scenes)
        page_image_path = args.out_dir / f"{page_stem}.png"
        page_text_path = args.out_dir / f"{page_stem}.txt"
        image.save(page_image_path)
        page_text_path.write_text(page_caption + "\n", encoding="utf-8")
        metadata_rows.append(
            {
                "sample_id": page_stem,
                "sample_type": "page",
                "source_image": image_path.as_posix(),
                "file_name": page_image_path.name,
                "text": page_caption,
                "story_index": story_index,
                "num_panels": args.num_panels,
            }
        )

        for panel_number, (scene_text, panel_image) in enumerate(
            zip(story_scenes, split_vertical_panels(image, args.num_panels), strict=True),
            start=1,
        ):
            panel_stem = f"{sample_index:06d}_panel{panel_number}"
            panel_caption = build_panel_caption(
                scene_text,
                panel_index=panel_number,
                num_panels=args.num_panels,
            )
            panel_image_path = args.out_dir / f"{panel_stem}.png"
            panel_text_path = args.out_dir / f"{panel_stem}.txt"
            panel_image.save(panel_image_path)
            panel_text_path.write_text(panel_caption + "\n", encoding="utf-8")
            metadata_rows.append(
                {
                    "sample_id": panel_stem,
                    "sample_type": "panel",
                    "source_image": image_path.as_posix(),
                    "file_name": panel_image_path.name,
                    "text": panel_caption,
                    "story_index": story_index,
                    "panel_index": panel_number,
                    "num_panels": args.num_panels,
                }
            )

        sample_index += 1

    write_jsonl(metadata_path, metadata_rows)
    print(
        f"Wrote {len(image_paths)} page sample(s) and {len(image_paths) * args.num_panels} panel sample(s) "
        f"to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
