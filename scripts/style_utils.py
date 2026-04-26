"""Helpers for style presets and style-aware output paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STYLE_CONFIG = Path("configs") / "style_presets.json"
DEFAULT_STYLE_ID = "storybook"


@dataclass(frozen=True)
class StylePreset:
    style_id: str
    display_name: str
    style_prompt: str
    negative_prompt_append: str
    reference_image_path: str | None
    backend_preference: str
    lora_path: str | None
    lora_weight_name: str | None
    lora_scale: float


@dataclass(frozen=True)
class StyleRunPaths:
    style_id: str
    run_root: Path
    intermediate_dir: Path
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
    member_b_logs_dir: Path
    member_b_local_logs_dir: Path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_style_id(style_id: str | None) -> str:
    if not style_id:
        return DEFAULT_STYLE_ID
    return str(style_id).strip() or DEFAULT_STYLE_ID


def load_style_presets(config_path: Path | None = None) -> dict[str, StylePreset]:
    path = config_path or DEFAULT_STYLE_CONFIG
    raw = read_json(path)
    presets = raw.get("presets", [])
    if not isinstance(presets, list) or not presets:
        raise SystemExit(f"No style presets found in {path}")

    result: dict[str, StylePreset] = {}
    for item in presets:
        style_id = normalize_style_id(item.get("style_id"))
        preset = StylePreset(
            style_id=style_id,
            display_name=str(item.get("display_name", style_id.title())),
            style_prompt=str(item.get("style_prompt", "")).strip(),
            negative_prompt_append=str(item.get("negative_prompt_append", "")).strip(),
            reference_image_path=(
                str(item["reference_image_path"]).strip()
                if item.get("reference_image_path")
                else None
            ),
            backend_preference=str(item.get("backend_preference", "prompt_only")).strip(),
            lora_path=str(item["lora_path"]).strip() if item.get("lora_path") else None,
            lora_weight_name=(
                str(item["lora_weight_name"]).strip()
                if item.get("lora_weight_name")
                else None
            ),
            lora_scale=float(item.get("lora_scale", 1.0)),
        )
        if preset.backend_preference not in {
            "prompt_only",
            "auto_ip_adapter",
            "require_ip_adapter",
        }:
            raise SystemExit(
                f"Unsupported backend_preference for style {style_id}: "
                f"{preset.backend_preference}"
            )
        if preset.lora_scale <= 0:
            raise SystemExit(
                f"Unsupported lora_scale for style {style_id}: {preset.lora_scale}"
            )
        result[style_id] = preset
    return result


def resolve_style_preset(
    style_id: str | None,
    config_path: Path | None = None,
) -> StylePreset:
    normalized = normalize_style_id(style_id)
    presets = load_style_presets(config_path)
    if normalized not in presets:
        raise SystemExit(
            f"Unknown style_id '{normalized}'. Available: {', '.join(sorted(presets))}"
        )
    return presets[normalized]


def resolve_style_run_paths(style_id: str | None) -> StyleRunPaths:
    normalized = normalize_style_id(style_id)
    run_root = Path("outputs") / "runs" / normalized
    intermediate_dir = run_root / "intermediate"
    logs_dir = run_root / "logs"
    final_dir = run_root / "final"
    return StyleRunPaths(
        style_id=normalized,
        run_root=run_root,
        intermediate_dir=intermediate_dir,
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
        member_b_logs_dir=logs_dir / "member_b",
        member_b_local_logs_dir=logs_dir / "member_b_local_4gpu",
    )


def merge_negative_prompts(base_prompt: str, append_prompt: str) -> str:
    parts = [part.strip(" ,") for part in [base_prompt, append_prompt] if part and part.strip()]
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        for token in [item.strip() for item in part.split(",") if item.strip()]:
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(token)
    return ", ".join(merged)


def format_style_template(value: str, style_id: str) -> str:
    return str(value).format(style_id=normalize_style_id(style_id))
