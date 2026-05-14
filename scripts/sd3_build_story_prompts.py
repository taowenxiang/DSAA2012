"""Build SD3 one-shot and independent story prompt JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_utils import discover_json_inputs, discover_text_inputs, parse_story_file, read_json
from sd3_story_utils import DEFAULT_STYLE_PROMPT, make_story_prompt, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "task_a",
        help="Directory of Task A txt files or parsed JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "sd3_story" / "validation_prompts",
    )
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--style-prompt", default=DEFAULT_STYLE_PROMPT)
    parser.add_argument(
        "--trigger-prefix",
        default="",
        help="Optional prefix such as sks_storyboard_style for self-LoRA inference.",
    )
    return parser.parse_args()


def load_stories(path: Path) -> list[dict]:
    if path.is_file() and path.suffix.lower() == ".json":
        return [read_json(path)]
    if path.is_dir():
        json_files = discover_json_inputs(path, suffix=".parsed.json")
        if json_files:
            return [read_json(item) for item in json_files]
        text_files = discover_text_inputs(path)
        return [parse_story_file(item) for item in text_files]
    return [parse_story_file(path)]


def main() -> int:
    args = parse_args()
    stories = load_stories(args.input)
    if not stories:
        raise SystemExit(f"No stories found at {args.input}")

    rows_2scene: list[dict] = []
    rows_3scene: list[dict] = []
    independent_2scene: list[dict] = []
    independent_3scene: list[dict] = []
    for story in sorted(stories, key=lambda item: str(item["case_id"])):
        panels = list(story.get("panels", []))
        if len(panels) < 2:
            continue
        characters = list(story.get("characters", []))
        row2 = make_story_prompt(
            case_id=str(story["case_id"]),
            panels=panels,
            characters=characters,
            num_panels=2,
            layout=args.layout,
            style_prompt=args.style_prompt,
            trigger_prefix=args.trigger_prefix or None,
        )
        rows_2scene.append(row2)
        independent_2scene.extend(row2["panel_prompts"])

        if len(panels) >= 3:
            row3 = make_story_prompt(
                case_id=str(story["case_id"]),
                panels=panels,
                characters=characters,
                num_panels=3,
                layout=args.layout,
                style_prompt=args.style_prompt,
                trigger_prefix=args.trigger_prefix or None,
            )
            rows_3scene.append(row3)
            independent_3scene.extend(row3["panel_prompts"])

    counts = {
        "prompts_2scene.jsonl": write_jsonl(args.out_dir / "prompts_2scene.jsonl", rows_2scene),
        "prompts_3scene.jsonl": write_jsonl(args.out_dir / "prompts_3scene.jsonl", rows_3scene),
        "prompts_independent_2scene.jsonl": write_jsonl(
            args.out_dir / "prompts_independent_2scene.jsonl", independent_2scene
        ),
        "prompts_independent_3scene.jsonl": write_jsonl(
            args.out_dir / "prompts_independent_3scene.jsonl", independent_3scene
        ),
    }
    for name, count in counts.items():
        print(f"Wrote {count} row(s): {args.out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
