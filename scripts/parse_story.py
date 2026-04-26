"""Parse Task A story txt files into deterministic intermediate JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_utils import discover_text_inputs, parse_story_file, write_json


DEFAULT_INPUT_DIR = Path("data") / "task_a"
DEFAULT_OUTPUT_DIR = Path("outputs") / "intermediate" / "parsed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="TaskA txt file or directory. Defaults to the provided TaskA folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for *.parsed.json files.",
    )
    parser.add_argument(
        "--expected-panels",
        type=int,
        default=3,
        help="Expected panel count for current TaskA data. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = discover_text_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No .txt story files found at {args.input}")

    written: list[Path] = []
    for input_path in inputs:
        parsed = parse_story_file(input_path)
        if args.expected_panels and len(parsed["panels"]) != args.expected_panels:
            raise SystemExit(
                f"{input_path} has {len(parsed['panels'])} panels; "
                f"expected {args.expected_panels}"
            )

        scene_ids = [panel["scene_id"] for panel in parsed["panels"]]
        if scene_ids != sorted(scene_ids):
            raise SystemExit(f"{input_path} scene ids are not ordered: {scene_ids}")

        output_path = args.output_dir / f"{parsed['case_id']}.parsed.json"
        write_json(output_path, parsed)
        written.append(output_path)

    print(f"Parsed {len(written)} story file(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
