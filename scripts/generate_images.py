"""Prepare candidate image generation manifests and HPC job bundles.

Member B keeps ownership of candidate generation, but real image synthesis is
expected to happen on HPC. This script remains the stable interface layer
between Member A prompt JSON files and downstream candidate selection.
"""

from __future__ import annotations

import argparse
import json
import struct
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS_DIR = Path("outputs") / "intermediate" / "prompts"
DEFAULT_CONFIG = Path("configs") / "member_b_generation_config.json"
DEFAULT_CANDIDATES_DIR = Path("outputs") / "candidates"
DEFAULT_MANIFEST = Path("outputs") / "intermediate" / "generation_manifest.json"
DEFAULT_JOB_DIR = Path("outputs") / "intermediate" / "hpc_jobs"
DEFAULT_STATUS = Path("outputs") / "intermediate" / "generation_status.json"


@dataclass(frozen=True)
class GenerationConfig:
    base_seed: int
    candidates_per_panel: int
    width: int
    height: int
    device: str
    dtype: str
    model_family: str
    model_path: str
    adapter_type: str
    adapter_command: list[str]
    num_inference_steps: int
    guidance_scale: float
    temperature: float
    tensor_parallel_size: int
    gpus_per_task: int
    cpus_per_task: int
    mem_gb: int
    time_limit: str
    partition: str
    account: str
    records_per_shard: int
    shard_strategy: str
    retries: int
    logs_dir: str
    status_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_DIR,
        help="Prompt JSON file or directory containing *.prompts.json files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Member B generation config JSON.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help="Root directory for candidate images.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path for the generation manifest JSON.",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=DEFAULT_JOB_DIR,
        help="Directory for HPC shard files and batch scripts.",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=DEFAULT_STATUS,
        help="Path for shared generation status JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Write the manifest without preparing HPC execution. Enabled by default.",
    )
    parser.add_argument(
        "--run-model",
        dest="dry_run",
        action="store_false",
        help="Prepare manifest and HPC job bundle for real model generation.",
    )
    parser.add_argument(
        "--placeholder-images",
        action="store_true",
        help="In dry-run mode, also write labeled placeholder PNGs.",
    )
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--candidates-per-panel", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--records-per-shard", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SystemExit("adapter.command must be a JSON array of strings")
    return value


def load_config(path: Path, args: argparse.Namespace) -> GenerationConfig:
    if not path.exists():
        raise SystemExit(f"Generation config not found: {path}")

    raw = read_json(path)
    adapter = raw.get("adapter", {})
    scheduler = raw.get("scheduler", {})
    config = GenerationConfig(
        base_seed=args.base_seed if args.base_seed is not None else int(raw["base_seed"]),
        candidates_per_panel=(
            args.candidates_per_panel
            if args.candidates_per_panel is not None
            else int(raw["candidates_per_panel"])
        ),
        width=args.width if args.width is not None else int(raw["width"]),
        height=args.height if args.height is not None else int(raw["height"]),
        device=str(raw.get("device", "cuda")),
        dtype=str(raw.get("dtype", "float16")),
        model_family=str(raw.get("model_family", "qwen_image")),
        model_path=str(raw.get("model_path", "")),
        adapter_type=str(adapter.get("type", "mock")),
        adapter_command=_string_list(adapter.get("command")),
        num_inference_steps=int(raw.get("num_inference_steps", 30)),
        guidance_scale=float(raw.get("guidance_scale", 4.5)),
        temperature=float(raw.get("temperature", 1.0)),
        tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
        gpus_per_task=int(scheduler.get("gpus_per_task", 1)),
        cpus_per_task=int(scheduler.get("cpus_per_task", 8)),
        mem_gb=int(scheduler.get("mem_gb", 64)),
        time_limit=str(scheduler.get("time_limit", "08:00:00")),
        partition=str(scheduler.get("partition", "")),
        account=str(scheduler.get("account", "")),
        records_per_shard=(
            args.records_per_shard
            if args.records_per_shard is not None
            else int(raw.get("records_per_shard", 6))
        ),
        shard_strategy=str(raw.get("shard_strategy", "case")),
        retries=int(raw.get("retries", 2)),
        logs_dir=str(raw.get("logs_dir", "outputs/logs/member_b")),
        status_path=str(raw.get("status_path", DEFAULT_STATUS.as_posix())),
    )

    if config.candidates_per_panel <= 0:
        raise SystemExit("candidates_per_panel must be positive")
    if config.width <= 0 or config.height <= 0:
        raise SystemExit("width and height must be positive")
    if config.records_per_shard <= 0:
        raise SystemExit("records_per_shard must be positive")
    if config.shard_strategy not in {"case", "fixed"}:
        raise SystemExit("shard_strategy must be 'case' or 'fixed'")
    if config.adapter_type == "command" and not config.adapter_command:
        raise SystemExit("adapter.command is required when adapter.type is 'command'")
    return config


def discover_prompt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.prompts.json"), key=lambda item: item.name.lower())


def candidate_seed(
    base_seed: int,
    case_index: int,
    scene_id: int,
    candidate_id: int,
) -> int:
    return base_seed + case_index * 1000 + scene_id * 100 + candidate_id


def relative_posix(path: Path) -> str:
    return path.as_posix()


def build_manifest(
    prompt_files: list[Path],
    config: GenerationConfig,
    candidates_dir: Path,
    dry_run: bool,
    placeholder_images: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    for case_index, prompt_path in enumerate(prompt_files):
        prompt_package = read_json(prompt_path)
        case_id = prompt_package["case_id"]

        for panel in prompt_package.get("panel_prompts", []):
            scene_id = int(panel["scene_id"])
            for candidate_id in range(1, config.candidates_per_panel + 1):
                output_path = (
                    candidates_dir
                    / case_id
                    / f"scene_{scene_id}"
                    / f"candidate_{candidate_id}.png"
                )
                status = "placeholder" if placeholder_images else "dry_run"
                if not dry_run:
                    status = "pending"
                records.append(
                    {
                        "case_id": case_id,
                        "scene_id": scene_id,
                        "candidate_id": candidate_id,
                        "seed": candidate_seed(
                            config.base_seed, case_index, scene_id, candidate_id
                        ),
                        "prompt": panel["prompt"],
                        "negative_prompt": panel["negative_prompt"],
                        "output_path": relative_posix(output_path),
                        "status": status,
                    }
                )

    return {
        "manifest_version": "member-b-v2",
        "mode": "dry_run" if dry_run else "hpc_batch",
        "placeholder_images": placeholder_images,
        "config": {
            "base_seed": config.base_seed,
            "candidates_per_panel": config.candidates_per_panel,
            "width": config.width,
            "height": config.height,
            "device": config.device,
            "dtype": config.dtype,
            "model_family": config.model_family,
            "model_path": config.model_path,
            "adapter_type": config.adapter_type,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "temperature": config.temperature,
            "tensor_parallel_size": config.tensor_parallel_size,
            "records_per_shard": config.records_per_shard,
            "retries": config.retries,
            "logs_dir": config.logs_dir,
            "status_path": config.status_path,
        },
        "source_prompt_count": len(prompt_files),
        "candidate_count": len(records),
        "candidates": records,
    }


def draw_placeholder(path: Path, record: dict[str, Any], width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        _write_minimal_png(path, record, width, height)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), color=(235, 238, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title_lines = [
        "PLACEHOLDER ONLY",
        f"case: {record['case_id']}",
        f"scene: {record['scene_id']}",
        f"candidate: {record['candidate_id']}",
        f"seed: {record['seed']}",
    ]
    prompt_lines = textwrap.wrap(record["prompt"], width=58)[:9]
    lines = title_lines + ["", "Prompt:"] + prompt_lines

    x = 24
    y = 24
    line_height = 18
    for index, line in enumerate(lines):
        fill = (160, 20, 20) if index == 0 else (20, 30, 38)
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height

    image.save(path)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_minimal_png(path: Path, record: dict[str, Any], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(record["seed"])
    r = 80 + (seed * 17) % 120
    g = 90 + (seed * 31) % 100
    b = 110 + (seed * 47) % 90
    row = bytes([0] + [r, g, b] * width)
    raw = row * height
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def write_placeholder_images(manifest: dict[str, Any], width: int, height: int) -> int:
    count = 0
    for record in manifest["candidates"]:
        draw_placeholder(Path(record["output_path"]), record, width, height)
        count += 1
    return count


def chunked(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


def build_shards(
    manifest: dict[str, Any],
    config: GenerationConfig,
    status_path: Path,
) -> list[dict[str, Any]]:
    records = manifest["candidates"]
    shards: list[list[dict[str, Any]]] = []

    if config.shard_strategy == "case":
        by_case: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_case.setdefault(str(record["case_id"]), []).append(record)
        for case_id in sorted(by_case):
            shards.append(by_case[case_id])
    else:
        shards = chunked(records, config.records_per_shard)

    shard_payloads: list[dict[str, Any]] = []
    for index, shard_records in enumerate(shards):
        case_ids = sorted({str(record["case_id"]) for record in shard_records})
        shard_payloads.append(
            {
                "job_index": index,
                "job_name": f"member_b_{'_'.join(case_ids)}",
                "case_ids": case_ids,
                "record_count": len(shard_records),
                "status_path": relative_posix(status_path),
                "records": shard_records,
            }
        )
    return shard_payloads


def _slurm_line(flag: str, value: str) -> str:
    return f"#SBATCH {flag}={value}\n" if value else ""


def render_slurm_array_script(
    shard_list_path: str,
    config_path: str,
    status_path: str,
    shard_count: int,
    config: GenerationConfig,
) -> str:
    partition_line = _slurm_line("--partition", config.partition)
    account_line = _slurm_line("--account", config.account)
    return f"""#!/bin/bash
#SBATCH --job-name=member-b-qwen-image
#SBATCH --array=0-{shard_count - 1}
#SBATCH --gres=gpu:{config.gpus_per_task}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem_gb}G
#SBATCH --time={config.time_limit}
{partition_line}{account_line}set -euo pipefail

ROOT_DIR="${{ROOT_DIR:-$PWD}}"
cd "$ROOT_DIR"

SHARD_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{shard_list_path}")
python scripts/run_hpc_generation.py \\
  --config "{config_path}" \\
  --shard "$SHARD_FILE" \\
  --status-path "{status_path}"
"""


def write_hpc_bundle(
    manifest: dict[str, Any],
    config: GenerationConfig,
    job_dir: Path,
    config_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = job_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_payloads = build_shards(manifest, config, status_path)
    shard_paths: list[str] = []
    for payload in shard_payloads:
        shard_path = shard_dir / f"job_{payload['job_index']:03d}.json"
        write_json(shard_path, payload)
        shard_paths.append(relative_posix(shard_path))

    shard_list = job_dir / "shards.txt"
    shard_list.write_text("\n".join(shard_paths) + "\n", encoding="utf-8")

    submit_script = job_dir / "submit_member_b_array.slurm"
    submit_script.write_text(
        render_slurm_array_script(
            shard_list_path=relative_posix(shard_list),
            config_path=relative_posix(config_path),
            status_path=relative_posix(status_path),
            shard_count=len(shard_payloads),
            config=config,
        ),
        encoding="utf-8",
        newline="\n",
    )

    status_template = {
        "status_version": "member-b-status-v1",
        "model_family": config.model_family,
        "status_path": relative_posix(status_path),
        "records": [],
    }
    write_json(status_path, status_template)

    return {
        "job_dir": relative_posix(job_dir),
        "shard_count": len(shard_payloads),
        "shard_list_path": relative_posix(shard_list),
        "submit_script": relative_posix(submit_script),
        "status_path": relative_posix(status_path),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args)

    prompt_files = discover_prompt_files(args.prompts)
    if not prompt_files:
        raise SystemExit(f"No *.prompts.json files found at {args.prompts}")

    manifest = build_manifest(
        prompt_files=prompt_files,
        config=config,
        candidates_dir=args.candidates_dir,
        dry_run=args.dry_run,
        placeholder_images=args.placeholder_images,
    )
    write_json(args.manifest, manifest)

    image_count = 0
    if args.placeholder_images:
        image_count = write_placeholder_images(manifest, config.width, config.height)

    print(
        "Prepared "
        f"{manifest['candidate_count']} candidate record(s) from "
        f"{manifest['source_prompt_count']} prompt file(s)."
    )
    print(f"Manifest: {args.manifest}")

    if args.placeholder_images:
        print(f"Placeholder images: {image_count} written under {args.candidates_dir}")

    if not args.dry_run:
        bundle = write_hpc_bundle(
            manifest=manifest,
            config=config,
            job_dir=args.job_dir,
            config_path=args.config,
            status_path=args.status_path,
        )
        print(
            "HPC bundle: "
            f"{bundle['shard_count']} shard(s), submit with sbatch {bundle['submit_script']}"
        )
        print(f"Status file: {bundle['status_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
