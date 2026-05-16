"""Build one-shot vertical page prompts for Qwen story generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_story_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PAGE_STYLE_PROMPT,
    build_page_prompt_row,
    load_stories,
    normalize_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, default=Path("data") / "task_a")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "page_prompts",
    )
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="vertical")
    parser.add_argument("--style-prompt", default=DEFAULT_PAGE_STYLE_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--global-lora-path", default="")
    parser.add_argument("--global-lora-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--global-lora-scale", type=float, default=0.55)
    parser.add_argument("--cast-lora-root", default="")
    parser.add_argument("--cast-lora-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--cast-lora-scale", type=float, default=0.85)
    return parser.parse_args()


def resolve_case_cast_lora(root: str, case_id: str) -> str | None:
    if not root:
        return None
    path = Path(root) / normalize_id(case_id)
    return path.as_posix() if path.exists() else None


def main() -> int:
    args = parse_args()
    stories = sorted(load_stories(args.stories), key=lambda item: str(item["case_id"]))
    rows_2scene: list[dict] = []
    rows_3scene: list[dict] = []
    for story in stories:
        panels = list(story.get("panels", []))
        if len(panels) < 2:
            continue
        case_id = str(story["case_id"])
        cast_lora_path = resolve_case_cast_lora(args.cast_lora_root, case_id)
        global_lora_path = args.global_lora_path or None
        rows_2scene.append(
            build_page_prompt_row(
                story=story,
                num_panels=2,
                layout=args.layout,
                style_prompt=args.style_prompt,
                negative_prompt=args.negative_prompt,
                global_lora_path=global_lora_path,
                global_lora_weight_name=args.global_lora_weight_name if global_lora_path else None,
                global_lora_scale=args.global_lora_scale,
                cast_lora_path=cast_lora_path,
                cast_lora_weight_name=args.cast_lora_weight_name if cast_lora_path else None,
                cast_lora_scale=args.cast_lora_scale,
            )
        )
        if len(panels) >= 3:
            rows_3scene.append(
                build_page_prompt_row(
                    story=story,
                    num_panels=3,
                    layout=args.layout,
                    style_prompt=args.style_prompt,
                    negative_prompt=args.negative_prompt,
                    global_lora_path=global_lora_path,
                    global_lora_weight_name=args.global_lora_weight_name if global_lora_path else None,
                    global_lora_scale=args.global_lora_scale,
                    cast_lora_path=cast_lora_path,
                    cast_lora_weight_name=args.cast_lora_weight_name if cast_lora_path else None,
                    cast_lora_scale=args.cast_lora_scale,
                )
            )

    count_2 = write_jsonl(args.out_dir / "page_prompts_2scene.jsonl", rows_2scene)
    count_3 = write_jsonl(args.out_dir / "page_prompts_3scene.jsonl", rows_3scene)
    print(f"Wrote {count_2} two-scene page prompt row(s)")
    print(f"Wrote {count_3} three-scene page prompt row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
