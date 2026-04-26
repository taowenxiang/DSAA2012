"""Run the Story pipeline from text parsing to final packaging."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Regenerate outputs/intermediate/generation_manifest.json from prompts.",
    )
    parser.add_argument(
        "--rerank-backend",
        choices=["auto", "pillow", "bytes"],
        default="auto",
        help="Backend passed to scripts/rerank_candidates.py.",
    )
    return parser.parse_args()


def run_step(name: str, args: list[str]) -> None:
    print(f"[pipeline] {name}")
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()

    run_step("parse stories", [sys.executable, "scripts/parse_story.py"])
    run_step("build prompts", [sys.executable, "scripts/build_prompts.py"])
    run_step("validate member A outputs", [sys.executable, "scripts/validate_member_a.py"])

    if args.refresh_manifest:
        run_step(
            "refresh generation manifest",
            [sys.executable, "scripts/generate_images.py", "--dry-run"],
        )

    run_step(
        "rerank candidates",
        [sys.executable, "scripts/rerank_candidates.py", "--backend", args.rerank_backend],
    )
    run_step(
        "package final outputs",
        [sys.executable, "scripts/package_outputs.py", "--clean"],
    )

    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
