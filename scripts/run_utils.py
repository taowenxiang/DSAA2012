"""Helpers for numbered experiment run directories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS_ROOT = Path("outputs") / "runs"
RUN_DIR_RE = re.compile(r"^run_(\d{4})_(.+)$")


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    run_name: str
    run_number: int
    style_id: str
    metadata_dir: Path
    config_snapshot_dir: Path
    intermediate_dir: Path
    parsed_dir: Path
    prompts_dir: Path
    candidates_dir: Path
    logs_dir: Path
    final_dir: Path
    manifest_path: Path
    status_path: Path
    local_status_path: Path
    selection_path: Path
    final_manifest_path: Path
    hpc_job_dir: Path
    local_job_dir: Path
    run_metadata_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sanitize_style_id(style_id: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(style_id).strip()
    )
    return cleaned or "style"


def next_run_number(runs_root: Path | None = None) -> int:
    root = runs_root or RUNS_ROOT
    if not root.exists():
        return 1
    numbers: list[int] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def create_run_paths(style_id: str, runs_root: Path | None = None) -> RunPaths:
    root = runs_root or RUNS_ROOT
    number = next_run_number(root)
    safe_style_id = _sanitize_style_id(style_id)
    run_name = f"run_{number:04d}_{safe_style_id}"
    run_root = root / run_name
    metadata_dir = run_root / "metadata"
    config_snapshot_dir = metadata_dir / "config_snapshot"
    intermediate_dir = run_root / "intermediate"
    logs_dir = run_root / "logs"
    final_dir = run_root / "final"
    return RunPaths(
        run_root=run_root,
        run_name=run_name,
        run_number=number,
        style_id=safe_style_id,
        metadata_dir=metadata_dir,
        config_snapshot_dir=config_snapshot_dir,
        intermediate_dir=intermediate_dir,
        parsed_dir=intermediate_dir / "parsed",
        prompts_dir=intermediate_dir / "prompts",
        candidates_dir=run_root / "candidates",
        logs_dir=logs_dir,
        final_dir=final_dir,
        manifest_path=intermediate_dir / "generation_manifest.json",
        status_path=intermediate_dir / "generation_status.json",
        local_status_path=intermediate_dir / "generation_status.local_4gpu.json",
        selection_path=intermediate_dir / "selection_results.json",
        final_manifest_path=final_dir / "submission_manifest.json",
        hpc_job_dir=intermediate_dir / "hpc_jobs",
        local_job_dir=intermediate_dir / "local_4gpu_jobs",
        run_metadata_path=metadata_dir / "run_metadata.json",
    )


def resolve_run_paths(run_dir: Path) -> RunPaths:
    run_root = Path(run_dir)
    name = run_root.name
    match = RUN_DIR_RE.match(name)
    if match:
        number = int(match.group(1))
        style_id = match.group(2)
    else:
        number = 0
        style_id = "style"
    metadata_dir = run_root / "metadata"
    config_snapshot_dir = metadata_dir / "config_snapshot"
    intermediate_dir = run_root / "intermediate"
    logs_dir = run_root / "logs"
    final_dir = run_root / "final"
    return RunPaths(
        run_root=run_root,
        run_name=name,
        run_number=number,
        style_id=style_id,
        metadata_dir=metadata_dir,
        config_snapshot_dir=config_snapshot_dir,
        intermediate_dir=intermediate_dir,
        parsed_dir=intermediate_dir / "parsed",
        prompts_dir=intermediate_dir / "prompts",
        candidates_dir=run_root / "candidates",
        logs_dir=logs_dir,
        final_dir=final_dir,
        manifest_path=intermediate_dir / "generation_manifest.json",
        status_path=intermediate_dir / "generation_status.json",
        local_status_path=intermediate_dir / "generation_status.local_4gpu.json",
        selection_path=intermediate_dir / "selection_results.json",
        final_manifest_path=final_dir / "submission_manifest.json",
        hpc_job_dir=intermediate_dir / "hpc_jobs",
        local_job_dir=intermediate_dir / "local_4gpu_jobs",
        run_metadata_path=metadata_dir / "run_metadata.json",
    )


def snapshot_json_configs(config_paths: list[Path], destination_dir: Path) -> list[str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for config_path in config_paths:
        if not config_path.exists():
            continue
        target = destination_dir / config_path.name
        target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(target.as_posix())
    return written
