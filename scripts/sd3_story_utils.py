"""Shared helpers for the SD3 storyboard training workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted face, deformed hands, extra limbs, "
    "inconsistent character, different clothing in each panel, inconsistent art style, "
    "messy panel layout, unreadable composition, text, watermark, logo"
)

DEFAULT_STYLE_PROMPT = (
    "consistent cinematic storybook illustration, coherent warm color palette, "
    "natural lighting, expressive faces, clean composition"
)

TRIGGER_STORYBOARD = "sks_storyboard_style"
TRIGGER_CHARACTER = "sks_storyhero"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean = line.strip()
            if clean:
                rows.append(json.loads(clean))
    return rows


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "item"


def format_character_reference(characters: list[str]) -> str:
    if not characters:
        return "the same main character"
    if len(characters) == 1:
        return characters[0]
    if len(characters) == 2:
        return f"{characters[0]} and {characters[1]}"
    return ", ".join(characters[:-1]) + f", and {characters[-1]}"


def storyboard_layout_phrase(num_panels: int, layout: str) -> str:
    panel_word = "two-panel" if num_panels == 2 else "three-panel"
    flow = "left-to-right story flow" if layout == "horizontal" else "top-to-bottom story flow"
    return f"a {panel_word} storyboard illustration, clear panel boundaries, {flow}"


def make_story_prompt(
    case_id: str,
    panels: list[dict[str, Any]],
    characters: list[str],
    num_panels: int,
    layout: str,
    style_prompt: str = DEFAULT_STYLE_PROMPT,
    trigger_prefix: str | None = None,
) -> dict[str, Any]:
    selected = panels[:num_panels]
    character_ref = format_character_reference(characters)
    setting = "same story world"
    global_context = ""
    scene_chunks: list[str] = []
    panel_prompts: list[dict[str, Any]] = []

    for index, panel in enumerate(selected, start=1):
        action = clean_text(panel.get("action") or panel.get("resolved_action") or panel.get("raw_text", ""))
        action = action.rstrip(".")
        setting_hint = clean_text(panel.get("setting_hint", "same story world"))
        if setting == "same story world" and setting_hint != "same story world":
            setting = setting_hint
        scene_chunks.append(f"panel {index}: {action}")
        panel_prompts.append(
            {
                "id": f"{case_id}_scene{index}",
                "story_id": case_id,
                "panel": index,
                "prompt": clean_text(
                    f"storybook illustration, same main character: {character_ref}, "
                    f"{action}, setting: {setting_hint}, {style_prompt}, no text, no watermark"
                ),
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            }
        )

    if characters:
        global_context = (
            f"same main character across all panels: {character_ref}, consistent face, "
            "hairstyle, clothing, and color palette across panels"
        )
    else:
        global_context = (
            "same story world across all panels, consistent color palette and visual style"
        )

    prefix = f"{trigger_prefix}, " if trigger_prefix else ""
    prompt = clean_text(
        f"{prefix}{storyboard_layout_phrase(num_panels, layout)}, {global_context}, "
        f"shared setting: {setting}, {style_prompt}, "
        + ", ".join(scene_chunks)
        + ", coherent lighting, natural poses, expressive faces, high visual quality, no text, no watermark"
    )
    setting_name = "2scene" if num_panels == 2 else "3scene"
    return {
        "id": f"{case_id}_{setting_name}",
        "setting": setting_name,
        "num_panels": num_panels,
        "layout": layout,
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "panel_prompts": panel_prompts,
    }


def iter_image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([item for item in path.rglob("*") if item.suffix.lower() in suffixes])


def infer_prompt_id(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.split("_cand", 1)[0]
    stem = stem.split("_scale", 1)[0]
    return stem
