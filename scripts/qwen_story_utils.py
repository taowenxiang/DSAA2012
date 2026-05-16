"""Shared helpers for the Qwen dual-LoRA story workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from story_utils import discover_json_inputs, discover_text_inputs, parse_story_file, read_json


DEFAULT_PAGE_STYLE_PROMPT = (
    "consistent cinematic storybook illustration, painterly detail, coherent warm color palette, "
    "clear narrative composition, expressive faces, clean panel storytelling"
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted anatomy, deformed hands, extra limbs, "
    "inconsistent character identity, inconsistent clothing, broken panel layout, "
    "messy composition, text, watermark, logo"
)

DEFAULT_PAGE_WIDTH = 1024
DEFAULT_TWO_PANEL_HEIGHT = 1024
DEFAULT_THREE_PANEL_HEIGHT = 1536

SINGLE_CAST_TEMPLATES = [
    "full-body character turnaround, front-facing, clean silhouette, simple background",
    "three-quarter portrait, gentle expression, soft window light, clean background",
    "walking pose, full body, readable clothing design, simple setting",
    "sitting at a table with a book, calm expression, warm indoor light",
    "standing outdoors, natural daylight, full body, clear costume details",
    "close portrait, expressive face, soft painterly shading, minimal background",
    "holding a small personal object, full body, clean storybook rendering",
    "side profile portrait, consistent hairstyle, soft atmosphere, simple background",
    "light rain scene, standing pose, readable outfit details, cinematic light",
    "casual walking pose, full body, same identity, simple city background",
    "seated portrait, hands visible, clear face, warm ambient light",
    "storybook character reference sheet feel, full body, same person, simple backdrop",
]

DUO_CAST_TEMPLATES = [
    "full-body duo character sheet, both standing side by side, clear costume separation, simple background",
    "two-character portrait, both facing camera, warm indoor light, clean background",
    "sitting across a table and talking, readable faces, simple park cafe background",
    "walking together, full body, same recurring duo, natural daylight",
    "standing together, one slightly turned, one front-facing, clean silhouette design",
    "close duo portrait, both identities readable, soft painterly shading",
    "storybook duo reference sheet feel, both characters visible from head to toe, simple backdrop",
    "quiet conversation pose, seated side by side, warm storybook atmosphere",
    "outdoor bench scene, both characters interacting, readable costume details",
    "gallery visit pose, both looking at the same direction, elegant composition",
    "rainy-day shelter scene, both characters under soft light, same recurring duo",
    "friendly full-body portrait, both standing naturally, simple background",
]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "item"


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


def load_stories(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and path.suffix.lower() == ".json":
        return [read_json(path)]
    if path.is_dir():
        json_files = discover_json_inputs(path, suffix=".parsed.json")
        if json_files:
            return [read_json(item) for item in json_files]
        text_files = discover_text_inputs(path)
        return [parse_story_file(item) for item in text_files]
    return [parse_story_file(path)]


def format_character_reference(characters: list[str]) -> str:
    if not characters:
        return "the same recurring characters"
    if len(characters) == 1:
        return characters[0]
    if len(characters) == 2:
        return f"{characters[0]} and {characters[1]}"
    return ", ".join(characters[:-1]) + f", and {characters[-1]}"


def page_layout_phrase(num_panels: int, layout: str) -> str:
    panel_word = "two-panel" if num_panels == 2 else "three-panel"
    if layout == "vertical":
        flow = "top-to-bottom narrative flow"
        panel_shape = "stacked vertical panels"
    else:
        flow = "left-to-right narrative flow"
        panel_shape = "side-by-side panels"
    return f"{panel_word} storybook page, {panel_shape}, clear gutters, {flow}"


def page_dimensions(num_panels: int, layout: str) -> tuple[int, int]:
    if layout == "vertical":
        if num_panels == 2:
            return DEFAULT_PAGE_WIDTH, DEFAULT_TWO_PANEL_HEIGHT
        return DEFAULT_PAGE_WIDTH, DEFAULT_THREE_PANEL_HEIGHT
    if num_panels == 2:
        return DEFAULT_TWO_PANEL_HEIGHT, DEFAULT_PAGE_WIDTH
    return DEFAULT_THREE_PANEL_HEIGHT, DEFAULT_PAGE_WIDTH


def panel_slot_label(index: int, num_panels: int, layout: str) -> str:
    if layout == "vertical":
        if num_panels == 2:
            return "top panel" if index == 1 else "bottom panel"
        return ["top panel", "middle panel", "bottom panel"][index - 1]
    if num_panels == 2:
        return "left panel" if index == 1 else "right panel"
    return ["left panel", "center panel", "right panel"][index - 1]


def default_cast_phrase(characters: list[str]) -> str:
    if not characters:
        return "the same recurring cast from this story"
    if len(characters) == 1:
        return f"{characters[0]}, the recurring protagonist"
    if len(characters) == 2:
        return f"{characters[0]} and {characters[1]}, the recurring duo"
    return f"{format_character_reference(characters)}, the recurring cast"


def build_page_prompt_row(
    story: dict[str, Any],
    num_panels: int,
    layout: str = "vertical",
    style_prompt: str = DEFAULT_PAGE_STYLE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    global_lora_path: str | None = None,
    global_lora_weight_name: str | None = None,
    global_lora_scale: float = 1.0,
    cast_lora_path: str | None = None,
    cast_lora_weight_name: str | None = None,
    cast_lora_scale: float = 0.85,
    cast_phrase: str | None = None,
) -> dict[str, Any]:
    selected = list(story.get("panels", []))[:num_panels]
    characters = list(story.get("characters", []))
    if len(selected) < num_panels:
        raise ValueError(f"Story {story.get('case_id')} does not have {num_panels} panel(s)")

    panel_chunks: list[str] = []
    for index, panel in enumerate(selected, start=1):
        action = clean_text(panel.get("action") or panel.get("resolved_action") or panel.get("raw_text", ""))
        action = action.rstrip(".")
        panel_chunks.append(f"{panel_slot_label(index, num_panels, layout)}: {action}")

    shared_setting = "same story world"
    for panel in selected:
        hint = clean_text(panel.get("setting_hint", "same story world"))
        if hint and hint != "same story world":
            shared_setting = hint
            break

    cast_reference = cast_phrase or default_cast_phrase(characters)
    width, height = page_dimensions(num_panels, layout)
    prompt = clean_text(
        f"{page_layout_phrase(num_panels, layout)}, same recurring cast across all panels: {cast_reference}, "
        f"shared setting: {shared_setting}, {style_prompt}, "
        + ", ".join(panel_chunks)
        + ", keep faces, hairstyles, clothing, and color palette consistent across panels, high visual quality, no text, no watermark"
    )

    extra_loras: list[dict[str, Any]] = []
    if cast_lora_path:
        extra_loras.append(
            {
                "role": "cast",
                "path": cast_lora_path,
                "weight_name": cast_lora_weight_name,
                "scale": cast_lora_scale,
            }
        )

    return {
        "id": f"{story['case_id']}_{num_panels}scene",
        "case_id": str(story["case_id"]),
        "num_panels": num_panels,
        "layout": layout,
        "width": width,
        "height": height,
        "characters": characters,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "global_lora_path": global_lora_path,
        "global_lora_weight_name": global_lora_weight_name,
        "global_lora_scale": global_lora_scale,
        "extra_loras": extra_loras,
    }


def build_page_training_caption(
    story: dict[str, Any],
    num_panels: int,
    layout: str = "vertical",
    style_prompt: str = DEFAULT_PAGE_STYLE_PROMPT,
) -> str:
    selected = list(story.get("panels", []))[:num_panels]
    characters = list(story.get("characters", []))
    cast_reference = default_cast_phrase(characters)
    panel_chunks: list[str] = []
    for index, panel in enumerate(selected, start=1):
        action = clean_text(panel.get("action") or panel.get("resolved_action") or panel.get("raw_text", ""))
        action = action.rstrip(".")
        panel_chunks.append(f"{panel_slot_label(index, num_panels, layout)}: {action}")
    return clean_text(
        f"{page_layout_phrase(num_panels, layout)}, same recurring cast: {cast_reference}, "
        f"{style_prompt}, " + ", ".join(panel_chunks)
    )


def build_cast_seed_rows(
    story: dict[str, Any],
    num_images: int = 12,
    width: int = 768,
    height: int = 768,
    style_prompt: str = DEFAULT_PAGE_STYLE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed_base: int = 7000,
) -> list[dict[str, Any]]:
    characters = list(story.get("characters", []))
    case_id = str(story["case_id"])
    cast_reference = default_cast_phrase(characters)
    templates = SINGLE_CAST_TEMPLATES if len(characters) <= 1 else DUO_CAST_TEMPLATES
    rows: list[dict[str, Any]] = []

    for index in range(num_images):
        template = templates[index % len(templates)]
        prompt = clean_text(
            f"storybook character reference illustration of {cast_reference}, "
            f"same identity across images, {template}, {style_prompt}, no text, no watermark"
        )
        row_id = f"{normalize_id(case_id)}_castref_{index + 1:02d}"
        rows.append(
            {
                "id": row_id,
                "case_id": case_id,
                "width": width,
                "height": height,
                "seed": seed_base + index,
                "output_subdir": normalize_id(case_id),
                "prompt": prompt,
                "caption": prompt,
                "negative_prompt": negative_prompt,
                "characters": characters,
            }
        )
    return rows
