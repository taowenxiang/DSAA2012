"""Run the Story pipeline from text parsing to final packaging."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from run_utils import create_run_paths, snapshot_json_configs, utc_now_iso, write_json
from style_utils import DEFAULT_STYLE_CONFIG, DEFAULT_STYLE_ID, resolve_style_preset


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE_ID,
        help="Style preset id. Defaults to storybook.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Regenerate the style-specific generation manifest from prompts.",
    )
    parser.add_argument(
        "--rerank-backend",
        choices=["auto", "pillow", "bytes"],
        default="auto",
        help="Backend passed to scripts/rerank_candidates.py.",
    )
    parser.add_argument(
        "--placeholder-images",
        action="store_true",
        help="When refreshing the manifest, also create placeholder candidate images.",
    )
    return parser.parse_args()


def run_step(name: str, args: list[str]) -> None:
    print(f"[pipeline] {name}")
    subprocess.run(args, cwd=ROOT, check=True)


def seed_storybook_run_from_legacy(style_id: str, run_dir: Path) -> None:
    if style_id != DEFAULT_STYLE_ID:
        return

    candidates_dir = run_dir / "candidates"
    legacy_candidates = ROOT / "outputs" / "legacy" / "storybook_seed" / "candidates"
    if legacy_candidates.exists() and not candidates_dir.exists():
        candidates_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_candidates, candidates_dir)


def write_run_metadata(
    run_paths,
    args: argparse.Namespace,
) -> None:
    style_preset = resolve_style_preset(args.style, DEFAULT_STYLE_CONFIG)
    config_paths = [
        ROOT / "configs" / "member_a_prompt_config.json",
        ROOT / "configs" / "member_b_generation_config.json",
        ROOT / "configs" / "member_b_generation_config.local_4gpu.json",
        ROOT / DEFAULT_STYLE_CONFIG,
    ]
    copied_configs = snapshot_json_configs(config_paths, run_paths.config_snapshot_dir)
    metadata = {
        "run_name": run_paths.run_name,
        "run_number": run_paths.run_number,
        "style_id": args.style,
        "created_at_utc": utc_now_iso(),
        "pipeline_args": {
            "style": args.style,
            "refresh_manifest": bool(args.refresh_manifest),
            "rerank_backend": args.rerank_backend,
            "placeholder_images": bool(args.placeholder_images),
        },
        "style_preset": {
            "style_id": style_preset.style_id,
            "display_name": style_preset.display_name,
            "style_prompt": style_preset.style_prompt,
            "negative_prompt_append": style_preset.negative_prompt_append,
            "reference_image_path": style_preset.reference_image_path,
            "backend_preference": style_preset.backend_preference,
        },
        "paths": {
            "run_root": run_paths.run_root.as_posix(),
            "parsed_dir": run_paths.parsed_dir.as_posix(),
            "prompts_dir": run_paths.prompts_dir.as_posix(),
            "candidates_dir": run_paths.candidates_dir.as_posix(),
            "manifest_path": run_paths.manifest_path.as_posix(),
            "selection_path": run_paths.selection_path.as_posix(),
            "final_dir": run_paths.final_dir.as_posix(),
            "final_manifest_path": run_paths.final_manifest_path.as_posix(),
        },
        "config_snapshots": copied_configs,
    }
    write_json(run_paths.run_metadata_path, metadata)


def main() -> int:
    args = parse_args()
    run_paths = create_run_paths(args.style)
    run_paths.run_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_paths, args)

    run_step(
        "parse stories",
        [
            sys.executable,
            "scripts/parse_story.py",
            "--output-dir",
            run_paths.parsed_dir.as_posix(),
        ],
    )
    run_step(
        "build prompts",
        [
            sys.executable,
            "scripts/build_prompts.py",
            "--style",
            args.style,
            "--run-dir",
            run_paths.run_root.as_posix(),
            "--parsed",
            run_paths.parsed_dir.as_posix(),
        ],
    )
    run_step(
        "validate member A outputs",
        [
            sys.executable,
            "scripts/validate_member_a.py",
            "--style",
            args.style,
            "--run-dir",
            run_paths.run_root.as_posix(),
            "--parsed-dir",
            run_paths.parsed_dir.as_posix(),
        ],
    )

    seed_storybook_run_from_legacy(args.style, run_paths.run_root)

    generate_args = [
        sys.executable,
        "scripts/generate_images.py",
        "--style",
        args.style,
        "--run-dir",
        run_paths.run_root.as_posix(),
        "--dry-run",
    ]
    if args.placeholder_images:
        generate_args.append("--placeholder-images")
    run_step("refresh generation manifest", generate_args)

    run_step(
        "rerank candidates",
        [
            sys.executable,
            "scripts/rerank_candidates.py",
            "--style",
            args.style,
            "--run-dir",
            run_paths.run_root.as_posix(),
            "--backend",
            args.rerank_backend,
        ],
    )
    run_step(
        "package final outputs",
        [
            sys.executable,
            "scripts/package_outputs.py",
            "--style",
            args.style,
            "--run-dir",
            run_paths.run_root.as_posix(),
            "--clean",
        ],
    )

    print(f"[pipeline] run directory: {run_paths.run_root}")
    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
