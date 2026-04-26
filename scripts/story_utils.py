"""Utilities for Task A story parsing and prompt construction.

The implementation is intentionally deterministic and dependency-free so the
TA team can reproduce the same intermediate JSON files from the same inputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


SCENE_RE = re.compile(r"^\s*\[SCENE-(\d+)\]\s*(.*?)\s*$", re.DOTALL)
TAG_RE = re.compile(r"<([^<>]+)>")
SEP_RE = re.compile(r"\s*\[SEP\]\s*", re.IGNORECASE)
LEADING_PRONOUN_RE = re.compile(
    r"^(She|He|It|They|she|he|it|they)\b", re.IGNORECASE
)

DEFAULT_STYLE_PROMPT = (
    "consistent cinematic storybook illustration, natural lighting, coherent "
    "color palette, clear composition"
)
DEFAULT_NEGATIVE_PROMPT = (
    "different character, inconsistent clothing, extra characters, distorted "
    "face, bad hands, blurry, low quality, text, watermark, logo"
)
DEFAULT_CONTINUITY_NOTES = [
    "Use the same visual identity for each recurring character.",
    "Keep clothing, colors, lighting, and illustration style consistent across panels.",
    "Do not introduce extra named characters unless the scene text requires them.",
]

SETTING_PREPOSITIONS = (
    "in front of",
    "inside",
    "under",
    "across",
    "toward",
    "through",
    "along",
    "beside",
    "outside",
    "inside",
    "around",
    "with",
    "from",
    "into",
    "onto",
    "near",
    "over",
    "down",
    "by",
    "at",
    "in",
    "on",
)

TRAILING_NOISE_WORDS = {
    "quietly",
    "happily",
    "together",
    "ahead",
    "inside",
    "outside",
    "away",
    "down",
    "around",
    "quickly",
    "slowly",
}

PHRASE_STOP_WORDS = {
    "and",
    "arrive",
    "arrives",
    "check",
    "checks",
    "chase",
    "chases",
    "come",
    "comes",
    "continue",
    "continues",
    "cut",
    "cuts",
    "drive",
    "drives",
    "fly",
    "flies",
    "get",
    "gets",
    "leave",
    "leaves",
    "look",
    "looks",
    "meet",
    "meets",
    "move",
    "moves",
    "paint",
    "paints",
    "pause",
    "pauses",
    "return",
    "returns",
    "run",
    "runs",
    "serve",
    "serves",
    "show",
    "shows",
    "sit",
    "sits",
    "stand",
    "stands",
    "walk",
    "walks",
    "write",
    "writes",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def case_id_from_path(path: Path) -> str:
    return path.stem


def format_character_list(characters: list[str]) -> str:
    if not characters:
        return "the same story world"
    if len(characters) == 1:
        return characters[0]
    if len(characters) == 2:
        return f"{characters[0]} and {characters[1]}"
    return ", ".join(characters[:-1]) + f", and {characters[-1]}"


def strip_character_tags(text: str) -> str:
    return TAG_RE.sub(lambda match: match.group(1).strip(), text)


def split_scene_blocks(raw_text: str) -> list[str]:
    return [block.strip() for block in SEP_RE.split(raw_text.strip()) if block.strip()]


def parse_scene_block(block: str) -> tuple[int, str]:
    match = SCENE_RE.match(block)
    if not match:
        raise ValueError(f"Scene block does not match [SCENE-n] format: {block!r}")
    scene_id = int(match.group(1))
    scene_text = match.group(2).strip()
    if not scene_text:
        raise ValueError(f"Scene {scene_id} is empty")
    return scene_id, scene_text


def extract_tags(text: str) -> list[str]:
    return unique_in_order(match.group(1).strip() for match in TAG_RE.finditer(text))


def starts_with_pronoun(text: str) -> str | None:
    clean = strip_character_tags(text).strip()
    match = LEADING_PRONOUN_RE.match(clean)
    return match.group(1).lower() if match else None


def infer_pronoun_subject(
    scene_text: str,
    known_characters: list[str],
    previous_active: list[str],
) -> list[str]:
    pronoun = starts_with_pronoun(scene_text)
    if pronoun == "they":
        return known_characters[:] if len(known_characters) > 1 else previous_active[:]
    if pronoun in {"she", "he", "it"}:
        if len(previous_active) == 1:
            return previous_active[:]
        if len(known_characters) == 1:
            return known_characters[:]
        return previous_active[:1]

    return []


def infer_active_characters(
    explicit_tags: list[str],
    pronoun_subject: list[str],
) -> list[str]:
    if explicit_tags and pronoun_subject:
        return unique_in_order([*pronoun_subject, *explicit_tags])
    if explicit_tags:
        return explicit_tags
    return pronoun_subject


def resolve_action_text(scene_text: str, subject_characters: list[str]) -> str:
    clean = strip_character_tags(scene_text).strip()
    if not subject_characters:
        return clean

    replacement = format_character_list(subject_characters)

    def replace_pronoun(match: re.Match[str]) -> str:
        pronoun = match.group(1)
        if pronoun[0].isupper():
            return replacement
        return replacement[0].lower() + replacement[1:]

    return LEADING_PRONOUN_RE.sub(replace_pronoun, clean, count=1)


def trim_phrase(phrase: str) -> str:
    clean = re.sub(r"\s+", " ", phrase.strip(" .,\n\t"))
    words = clean.split()
    clipped: list[str] = []
    for word in words:
        if clipped and word.lower() in PHRASE_STOP_WORDS:
            break
        clipped.append(word)
    words = clipped
    while words and words[-1].lower() in TRAILING_NOISE_WORDS:
        words.pop()
    return " ".join(words)


def extract_setting_hint(text: str) -> str:
    clean = strip_character_tags(text)
    phrases: list[str] = []
    for prep in SETTING_PREPOSITIONS:
        pattern = re.compile(
            rf"\b{re.escape(prep)}\s+([^.,;]+?)(?:\s+and\b|\.|,|;|$)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(clean):
            phrase = trim_phrase(f"{prep} {match.group(1)}")
            if phrase:
                phrases.append(phrase)

    if phrases:
        return unique_in_order(phrases)[0]
    return "same story world"


def extract_object_hints(text: str) -> list[str]:
    clean = strip_character_tags(text).lower()
    candidates: list[str] = []
    determiner_pattern = re.compile(
        r"\b(?:a|an|the|his|her|their)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})"
    )
    for match in determiner_pattern.finditer(clean):
        phrase = trim_phrase(match.group(1))
        if phrase and phrase not in {"same story", "story world"}:
            candidates.append(phrase)
    return unique_in_order(candidates)


def parse_story_text(raw_text: str, case_id: str, source_path: str | None = None) -> dict[str, Any]:
    known_characters: list[str] = []
    previous_active: list[str] = []
    panels: list[dict[str, Any]] = []
    setting_hints: list[str] = []
    object_hints: list[str] = []

    for block in split_scene_blocks(raw_text):
        scene_id, scene_text = parse_scene_block(block)
        explicit_tags = extract_tags(scene_text)
        known_characters = unique_in_order([*known_characters, *explicit_tags])
        pronoun_subject = infer_pronoun_subject(
            scene_text, known_characters, previous_active
        )
        active_characters = infer_active_characters(explicit_tags, pronoun_subject)
        action = resolve_action_text(scene_text, pronoun_subject)
        setting_hint = extract_setting_hint(scene_text)

        if active_characters:
            previous_active = active_characters

        setting_hints.append(setting_hint)
        object_hints.extend(extract_object_hints(scene_text))

        panels.append(
            {
                "scene_id": scene_id,
                "raw_text": scene_text,
                "active_characters": active_characters,
                "subject_characters": pronoun_subject,
                "action": action,
                "setting_hint": setting_hint,
            }
        )

    return {
        "case_id": case_id,
        "source_path": source_path,
        "characters": known_characters,
        "global_context": {
            "setting_hints": unique_in_order(
                hint for hint in setting_hints if hint != "same story world"
            ),
            "object_hints": unique_in_order(object_hints),
            "style_prompt": DEFAULT_STYLE_PROMPT,
        },
        "panels": panels,
        "continuity_notes": DEFAULT_CONTINUITY_NOTES,
    }


def parse_story_file(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    return parse_story_text(raw_text, case_id_from_path(path), str(path))


def discover_text_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.txt"), key=lambda item: item.name.lower())


def discover_json_inputs(path: Path, suffix: str = ".parsed.json") -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob(f"*{suffix}"), key=lambda item: item.name.lower())


def build_panel_prompt(
    parsed_story: dict[str, Any],
    panel: dict[str, Any],
    style_prompt: str = DEFAULT_STYLE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict[str, Any]:
    all_characters = parsed_story.get("characters", [])
    active_characters = panel.get("active_characters", [])
    featured = format_character_list(active_characters)
    recurring = format_character_list(all_characters)
    scene_id = panel["scene_id"]
    action = panel["action"].rstrip(".")
    setting = panel["setting_hint"]

    if active_characters:
        opening = f"A consistent story panel illustration of {featured}."
    else:
        opening = "A consistent story panel illustration in the same story world."

    global_identity = (
        f"Recurring character reference: {recurring}. "
        "Keep the same identity whenever each recurring character appears."
        if all_characters
        else "No named recurring character is specified."
    )

    prompt = (
        f"{opening} {global_identity} Scene {scene_id}: {action}. "
        f"Setting: {setting}. Keep the same character appearance, clothing, "
        f"color palette, and visual style across all panels. {style_prompt}, "
        "cinematic composition, clear subject, high quality."
    )

    return {
        "panel_id": f"scene_{scene_id}",
        "scene_id": scene_id,
        "source_text": panel["raw_text"],
        "active_characters": active_characters,
        "resolved_action": action,
        "setting_hint": setting,
        "prompt": re.sub(r"\s+", " ", prompt).strip(),
        "negative_prompt": negative_prompt,
        "seed_offset": scene_id,
    }


def build_prompt_package(
    parsed_story: dict[str, Any],
    style_prompt: str = DEFAULT_STYLE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    prompt_version: str = "member-a-v1",
) -> dict[str, Any]:
    return {
        "case_id": parsed_story["case_id"],
        "prompt_version": prompt_version,
        "characters": parsed_story.get("characters", []),
        "global_context": parsed_story.get("global_context", {}),
        "continuity_notes": parsed_story.get("continuity_notes", []),
        "negative_prompt": negative_prompt,
        "panel_prompts": [
            build_panel_prompt(parsed_story, panel, style_prompt, negative_prompt)
            for panel in parsed_story.get("panels", [])
        ],
    }


def load_prompt_config(config_path: Path | None) -> dict[str, str]:
    if not config_path or not config_path.exists():
        return {
            "style_prompt": DEFAULT_STYLE_PROMPT,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }

    config = read_json(config_path)
    return {
        "style_prompt": config.get("style_prompt", DEFAULT_STYLE_PROMPT),
        "negative_prompt": config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
    }
