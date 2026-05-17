"""Run the page-native Qwen story pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SceneSettingPaths:
    scene_count: int
    prompt_file: Path
    candidate_dir: Path
    infer_manifest: Path
    ranking_csv: Path
    ranking_summary: Path
    export_dir: Path
    export_manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, default=Path("data") / "task_a")
    parser.add_argument("--out-root", type=Path, default=Path("outputs") / "qwen_story_page")
    parser.add_argument("--scene-settings", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--layout", choices=["vertical"], default="vertical")
    parser.add_argument("--model-path", default="artifacts/models/Qwen-Image-2512")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--page-lora-path", default="")
    parser.add_argument("--page-lora-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--page-lora-scale", type=float, default=0.55)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--disable-clipscore", action="store_true")
    parser.add_argument("--require-dreamsim", action="store_true")
    parser.add_argument("--require-clipscore", action="store_true")
    parser.add_argument("--dreamsim-cache-dir", type=Path, default=None)
    parser.add_argument("--torch-hub-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_scene_setting_paths(out_root: Path, scene_count: int) -> SceneSettingPaths:
    label = f"{scene_count}scene"
    return SceneSettingPaths(
        scene_count=scene_count,
        prompt_file=out_root / "prompts" / f"page_prompts_{label}.jsonl",
        candidate_dir=out_root / "candidates" / label,
        infer_manifest=out_root / "candidates" / f"{label}_manifest.json",
        ranking_csv=out_root / "ranking" / f"{label}.csv",
        ranking_summary=out_root / "ranking" / f"{label}.summary.json",
        export_dir=out_root / "final" / f"{label}_top1",
        export_manifest=out_root / "final" / f"{label}_top1_manifest.json",
    )


def run_step(name: str, command: list[str]) -> None:
    print(f"[page-pipeline] {name}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    scene_settings = sorted({value for value in args.scene_settings if value in {2, 3}})
    if not scene_settings:
        raise SystemExit("scene-settings must include 2 and/or 3")

    prompt_dir = args.out_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    build_prompt_cmd = [
        sys.executable,
        "scripts/qwen_story_build_page_prompts.py",
        "--stories",
        args.stories.as_posix(),
        "--out-dir",
        prompt_dir.as_posix(),
        "--layout",
        args.layout,
        "--scene-settings",
        *[str(value) for value in scene_settings],
    ]
    if args.page_lora_path:
        build_prompt_cmd.extend(
            [
                "--global-lora-path",
                args.page_lora_path,
                "--global-lora-weight-name",
                args.page_lora_weight_name,
                "--global-lora-scale",
                str(args.page_lora_scale),
            ]
        )
    run_step("build page prompts", build_prompt_cmd)

    for scene_count in scene_settings:
        paths = build_scene_setting_paths(args.out_root, scene_count)
        infer_cmd = [
            sys.executable,
            "scripts/qwen_story_infer.py",
            "--prompts",
            paths.prompt_file.as_posix(),
            "--out-dir",
            paths.candidate_dir.as_posix(),
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
            "--num-candidates",
            str(args.num_candidates),
            "--manifest-out",
            paths.infer_manifest.as_posix(),
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
        run_step(f"infer {scene_count}-scene page candidates", infer_cmd)

        rank_cmd = [
            sys.executable,
            "scripts/qwen_story_rank_candidates.py",
            "--image-dir",
            paths.candidate_dir.as_posix(),
            "--prompts",
            paths.prompt_file.as_posix(),
            "--layout",
            args.layout,
            "--num-panels",
            str(scene_count),
            "--out",
            paths.ranking_csv.as_posix(),
            "--summary-out",
            paths.ranking_summary.as_posix(),
            "--clip-model",
            args.clip_model,
        ]
        if args.disable_clipscore:
            rank_cmd.append("--disable-clipscore")
        if args.require_dreamsim:
            rank_cmd.append("--require-dreamsim")
        if args.require_clipscore:
            rank_cmd.append("--require-clipscore")
        if args.dreamsim_cache_dir is not None:
            rank_cmd.extend(["--dreamsim-cache-dir", args.dreamsim_cache_dir.as_posix()])
        if args.torch_hub_dir is not None:
            rank_cmd.extend(["--torch-hub-dir", args.torch_hub_dir.as_posix()])
        run_step(f"rank {scene_count}-scene page candidates", rank_cmd)

        export_cmd = [
            sys.executable,
            "scripts/qwen_story_export_top_candidates.py",
            "--ranking",
            paths.ranking_csv.as_posix(),
            "--out-dir",
            paths.export_dir.as_posix(),
            "--manifest-out",
            paths.export_manifest.as_posix(),
            "--clean",
        ]
        run_step(f"export top1 {scene_count}-scene pages", export_cmd)

    print(f"[page-pipeline] output root: {args.out_root}")
    print("[page-pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
