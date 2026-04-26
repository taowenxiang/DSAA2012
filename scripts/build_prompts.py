"""Build per-panel image generation prompts from parsed Story JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_utils import (
    build_prompt_package,
    discover_json_inputs,
    load_prompt_config,
    read_json,
    write_json,
)


DEFAULT_PARSED_DIR = Path("outputs") / "intermediate" / "parsed"
DEFAULT_OUTPUT_DIR = Path("outputs") / "intermediate" / "prompts"
DEFAULT_CONFIG = Path("configs") / "member_a_prompt_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="Parsed JSON file or directory containing *.parsed.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for *.prompts.json files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Optional JSON config for style_prompt and negative_prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_prompt_config(args.config)
    parsed_files = discover_json_inputs(args.parsed, suffix=".parsed.json")
    if not parsed_files:
        raise SystemExit(f"No *.parsed.json files found at {args.parsed}")

    written: list[Path] = []
    for parsed_path in parsed_files:
        parsed_story = read_json(parsed_path)
        prompt_package = build_prompt_package(
            parsed_story,
            style_prompt=config["style_prompt"],
            negative_prompt=config["negative_prompt"],
        )
        output_path = args.output_dir / f"{parsed_story['case_id']}.prompts.json"
        write_json(output_path, prompt_package)
        written.append(output_path)

    print(f"Built {len(written)} prompt file(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
