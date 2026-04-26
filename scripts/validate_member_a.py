"""Validate Member A parsing and prompt-construction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_utils import (
    discover_text_inputs,
    extract_tags,
    read_json,
    split_scene_blocks,
    starts_with_pronoun,
)


DEFAULT_INPUT_DIR = Path("data") / "task_a"
DEFAULT_PARSED_DIR = Path("outputs") / "intermediate" / "parsed"
DEFAULT_PROMPTS_DIR = Path("outputs") / "intermediate" / "prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--expected-panels", type=int, default=3)
    return parser.parse_args()


def validate_case(
    source_path: Path,
    parsed_dir: Path,
    prompts_dir: Path,
    expected_panels: int,
) -> list[str]:
    errors: list[str] = []
    case_id = source_path.stem
    raw_text = source_path.read_text(encoding="utf-8")
    source_tags = set(extract_tags(raw_text))
    source_blocks = split_scene_blocks(raw_text)

    parsed_path = parsed_dir / f"{case_id}.parsed.json"
    prompts_path = prompts_dir / f"{case_id}.prompts.json"
    if not parsed_path.exists():
        return [f"{case_id}: missing parsed JSON {parsed_path}"]
    if not prompts_path.exists():
        return [f"{case_id}: missing prompt JSON {prompts_path}"]

    parsed = read_json(parsed_path)
    prompts = read_json(prompts_path)

    panels = parsed.get("panels", [])
    if len(panels) != expected_panels:
        errors.append(f"{case_id}: expected {expected_panels} panels, got {len(panels)}")
    if len(source_blocks) != len(panels):
        errors.append(
            f"{case_id}: raw block count {len(source_blocks)} != parsed panels {len(panels)}"
        )

    scene_ids = [panel.get("scene_id") for panel in panels]
    if scene_ids != list(range(1, expected_panels + 1)):
        errors.append(f"{case_id}: scene ids are {scene_ids}")

    parsed_characters = set(parsed.get("characters", []))
    missing_tags = source_tags - parsed_characters
    if missing_tags:
        errors.append(f"{case_id}: missing character tags {sorted(missing_tags)}")

    previous_active: list[str] = []
    for panel in panels:
        if not panel.get("raw_text", "").strip():
            errors.append(f"{case_id}: scene {panel.get('scene_id')} has empty raw_text")
        if not panel.get("action", "").strip():
            errors.append(f"{case_id}: scene {panel.get('scene_id')} has empty action")

        explicit_tags = set(extract_tags(panel.get("raw_text", "")))
        if starts_with_pronoun(panel.get("raw_text", "")) and explicit_tags and previous_active:
            expected_active = set(previous_active) | explicit_tags
            actual_active = set(panel.get("active_characters", []))
            if not expected_active.issubset(actual_active):
                errors.append(
                    f"{case_id}: scene {panel.get('scene_id')} should keep prior "
                    f"active characters {sorted(previous_active)} with tags "
                    f"{sorted(explicit_tags)}, got {sorted(actual_active)}"
                )

        if panel.get("active_characters"):
            previous_active = panel.get("active_characters", [])

    panel_prompts = prompts.get("panel_prompts", [])
    if len(panel_prompts) != expected_panels:
        errors.append(
            f"{case_id}: expected {expected_panels} panel prompts, got {len(panel_prompts)}"
        )
    for panel_prompt in panel_prompts:
        if not panel_prompt.get("prompt", "").strip():
            errors.append(
                f"{case_id}: scene {panel_prompt.get('scene_id')} has empty prompt"
            )
        if not panel_prompt.get("negative_prompt", "").strip():
            errors.append(
                f"{case_id}: scene {panel_prompt.get('scene_id')} has empty negative prompt"
            )
        if ".." in panel_prompt.get("prompt", ""):
            errors.append(
                f"{case_id}: scene {panel_prompt.get('scene_id')} prompt contains double period"
            )

    return errors


def main() -> int:
    args = parse_args()
    inputs = discover_text_inputs(args.input_dir)
    if not inputs:
        raise SystemExit(f"No .txt story files found at {args.input_dir}")

    all_errors: list[str] = []
    for source_path in inputs:
        all_errors.extend(
            validate_case(
                source_path,
                args.parsed_dir,
                args.prompts_dir,
                args.expected_panels,
            )
        )

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"OK: validated {len(inputs)} story case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
