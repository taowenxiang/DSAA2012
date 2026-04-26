"""Build per-panel image generation prompts from parsed Story JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_utils import resolve_run_paths
from style_utils import (
    DEFAULT_STYLE_CONFIG,
    resolve_style_preset,
    merge_negative_prompts,
)
from story_utils import (
    build_prompt_package,
    discover_json_inputs,
    load_prompt_config,
    read_json,
    write_json,
)


DEFAULT_PARSED_DIR = Path("outputs") / "intermediate" / "parsed"
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
        default=None,
        help="Directory for *.prompts.json files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Optional JSON config for prompt defaults.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Style preset id. Defaults to the config's default_style_id.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional numbered run directory. If provided, prompts are written into that run.",
    )
    parser.add_argument(
        "--style-config",
        type=Path,
        default=DEFAULT_STYLE_CONFIG,
        help="Style preset configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_prompt_config(args.config)
    style_id = args.style or config["default_style_id"]
    preset = resolve_style_preset(style_id, args.style_config)
    parsed_files = discover_json_inputs(args.parsed, suffix=".parsed.json")
    if not parsed_files:
        raise SystemExit(f"No *.parsed.json files found at {args.parsed}")

    run_paths = resolve_run_paths(args.run_dir) if args.run_dir else None
    output_dir = args.output_dir or (run_paths.prompts_dir if run_paths else None)
    if output_dir is None:
        raise SystemExit("Either --output-dir or --run-dir is required for build_prompts.py")
    negative_prompt = merge_negative_prompts(
        config["base_negative_prompt"], preset.negative_prompt_append
    )
    written: list[Path] = []
    for parsed_path in parsed_files:
        parsed_story = read_json(parsed_path)
        prompt_package = build_prompt_package(
            parsed_story,
            style_prompt=preset.style_prompt,
            negative_prompt=negative_prompt,
            prompt_version=config["prompt_version"],
            style_id=preset.style_id,
            style_display_name=preset.display_name,
            style_backend_preference=preset.backend_preference,
            style_reference_image_path=preset.reference_image_path,
        )
        output_path = output_dir / f"{parsed_story['case_id']}.prompts.json"
        write_json(output_path, prompt_package)
        written.append(output_path)

    print(
        f"Built {len(written)} prompt file(s) into {output_dir} "
        f"for style={preset.style_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
