"""Generate one 3-scene candidate page for every story text under data/test_taskA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, default=Path("data") / "test_taskA")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs") / "qwen_story_page_test_taskA_candidates",
    )
    parser.add_argument("--layout", choices=["vertical"], default="vertical")
    parser.add_argument("--model-path", default="artifacts/models/Qwen-Image-2512")
    parser.add_argument("--page-lora-path", default="")
    parser.add_argument(
        "--page-lora-weight-name",
        default="pytorch_lora_weights.safetensors",
    )
    parser.add_argument("--page-lora-scale", type=float, default=0.55)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print(f"[test-taskA-candidates] {name}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    prompt_dir = args.out_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        "scripts/qwen_story_build_page_prompts.py",
        "--stories",
        args.stories.as_posix(),
        "--out-dir",
        prompt_dir.as_posix(),
        "--layout",
        args.layout,
        "--scene-settings",
        "3",
    ]
    if args.page_lora_path:
        build_cmd.extend(
            [
                "--global-lora-path",
                args.page_lora_path,
                "--global-lora-weight-name",
                args.page_lora_weight_name,
                "--global-lora-scale",
                str(args.page_lora_scale),
            ]
        )
    run_step("build 3-scene prompts", build_cmd)

    infer_cmd = [
        sys.executable,
        "scripts/qwen_story_infer.py",
        "--prompts",
        (prompt_dir / "page_prompts_3scene.jsonl").as_posix(),
        "--out-dir",
        (args.out_root / "candidates" / "3scene").as_posix(),
        "--model",
        args.model_path,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--guidance",
        str(args.guidance),
        "--seed",
        str(args.seed),
        "--num-candidates",
        "1",
        "--manifest-out",
        (args.out_root / "candidates" / "3scene_manifest.json").as_posix(),
    ]
    if args.cpu_offload:
        infer_cmd.append("--cpu-offload")
    if args.page_lora_path:
        infer_cmd.extend(
            [
                "--global-lora-path",
                args.page_lora_path,
                "--global-lora-weight-name",
                args.page_lora_weight_name,
                "--global-lora-scale",
                str(args.page_lora_scale),
            ]
        )
    if args.skip_existing:
        infer_cmd.append("--skip-existing")
    run_step("generate 1 candidate for each 3-scene story", infer_cmd)

    print(f"[test-taskA-candidates] output root: {args.out_root}")
    print("[test-taskA-candidates] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
