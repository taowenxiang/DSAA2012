"""Run one style-aware HPC shard for Member B candidate generation."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from generate_images import read_json, write_json
from style_utils import DEFAULT_STYLE_ID, format_style_template, resolve_style_run_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE_ID,
        help="Style id used to resolve logs and status paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "member_b_generation_config.json",
        help="Member B generation config JSON.",
    )
    parser.add_argument(
        "--shard",
        type=Path,
        required=True,
        help="Shard JSON produced by scripts/generate_images.py --run-model.",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Optional override for shared generation status JSON.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only rerun records already marked failed in the status file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    return parser.parse_args()


def _resolve_style_path(raw: Any, style_id: str, fallback: str) -> str:
    return format_style_template(str(raw or fallback), style_id)


def load_runtime_config(path: Path, style_id: str) -> dict[str, Any]:
    raw = read_json(path)
    adapter = raw.get("adapter", {})
    scheduler = raw.get("scheduler", {})
    ip_adapter = raw.get("ip_adapter", {})
    return {
        "style_id": style_id,
        "model_family": str(raw.get("model_family", "qwen_image")),
        "model_path": str(raw.get("model_path", "")),
        "cpu_offload": bool(raw.get("cpu_offload", False)),
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "dtype": str(raw.get("dtype", "float16")),
        "device": str(raw.get("device", "cuda")),
        "num_inference_steps": int(raw.get("num_inference_steps", 30)),
        "guidance_scale": float(raw.get("guidance_scale", 4.5)),
        "temperature": float(raw.get("temperature", 1.0)),
        "tensor_parallel_size": int(raw.get("tensor_parallel_size", 1)),
        "retries": int(raw.get("retries", 2)),
        "logs_dir": _resolve_style_path(
            raw.get("logs_dir"),
            style_id,
            "outputs/runs/{style_id}/logs/member_b",
        ),
        "status_path": _resolve_style_path(
            raw.get("status_path"),
            style_id,
            "outputs/runs/{style_id}/intermediate/generation_status.json",
        ),
        "local_status_path": _resolve_style_path(
            raw.get("local_status_path"),
            style_id,
            "outputs/runs/{style_id}/intermediate/generation_status.local_4gpu.json",
        ),
        "adapter_type": str(adapter.get("type", "mock")),
        "adapter_command": adapter.get("command", []),
        "gpus_per_task": int(scheduler.get("gpus_per_task", 1)),
        "ip_adapter": {
            "enabled": bool(ip_adapter.get("enabled", False)),
            "model_path": str(ip_adapter.get("model_path", "")),
            "image_encoder_path": str(ip_adapter.get("image_encoder_path", "")),
            "scale": float(ip_adapter.get("scale", 1.0)),
        },
    }


def load_status(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)

    payload = {
        "status_version": "member-b-status-v2-style",
        "style_id": config["style_id"],
        "model_family": config["model_family"],
        "records": [],
    }
    write_json(path, payload)
    return payload


def save_status(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def record_key(record: dict[str, Any]) -> str:
    return f"{record['style_id']}|{record['case_id']}|{record['scene_id']}|{record['candidate_id']}"


def index_status_records(status_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in status_data.get("records", []):
        indexed[str(item["record_key"])] = item
    return indexed


def upsert_status_record(
    status_data: dict[str, Any],
    record: dict[str, Any],
    state: str,
    attempt: int,
    message: str,
    backend_effective: str | None = None,
) -> None:
    key = record_key(record)
    indexed = index_status_records(status_data)
    payload = {
        "record_key": key,
        "style_id": record["style_id"],
        "case_id": record["case_id"],
        "scene_id": record["scene_id"],
        "candidate_id": record["candidate_id"],
        "output_path": record["output_path"],
        "seed": record["seed"],
        "state": state,
        "attempt": attempt,
        "message": message,
        "style_backend_requested": record.get("style_backend_requested"),
        "style_backend_effective": backend_effective,
        "updated_at_epoch": int(time.time()),
    }
    indexed[key] = payload
    status_data["records"] = sorted(indexed.values(), key=lambda item: item["record_key"])


def should_run_record(
    record: dict[str, Any],
    status_index: dict[str, dict[str, Any]],
    only_failed: bool,
) -> bool:
    entry = status_index.get(record_key(record))
    if entry is None:
        return not only_failed
    if only_failed:
        return entry.get("state") == "failed"
    return entry.get("state") != "success"


def apply_template(values: dict[str, Any], command: list[str]) -> list[str]:
    return [token.format(**values) for token in command]


def build_template_values(record: dict[str, Any], config: dict[str, Any], prompt_path: Path) -> dict[str, Any]:
    return {
        "prompt": record["prompt"],
        "negative_prompt": record["negative_prompt"],
        "seed": record["seed"],
        "output_path": record["output_path"],
        "width": config["width"],
        "height": config["height"],
        "model_path": config["model_path"],
        "cpu_offload": config["cpu_offload"],
        "model_family": config["model_family"],
        "dtype": config["dtype"],
        "device": config["device"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "temperature": config["temperature"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "prompt_payload_path": prompt_path.as_posix(),
        "style_id": record["style_id"],
        "case_id": record["case_id"],
        "scene_id": record["scene_id"],
        "candidate_id": record["candidate_id"],
    }


def build_payload(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": record["prompt"],
        "negative_prompt": record["negative_prompt"],
        "seed": record["seed"],
        "output_path": record["output_path"],
        "width": config["width"],
        "height": config["height"],
        "model_path": config["model_path"],
        "cpu_offload": config["cpu_offload"],
        "model_family": config["model_family"],
        "dtype": config["dtype"],
        "device": config["device"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "temperature": config["temperature"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "style_id": record["style_id"],
        "style_prompt": record.get("style_prompt", ""),
        "style_display_name": record.get("style_display_name", ""),
        "style_reference_image_path": record.get("style_reference_image_path"),
        "style_lora_path": record.get("style_lora_path"),
        "style_lora_weight_name": record.get("style_lora_weight_name"),
        "style_lora_scale": float(record.get("style_lora_scale", 1.0)),
        "style_backend_requested": record.get("style_backend_requested", "prompt_only"),
        "style_backend_effective": record.get("style_backend_effective", "pending"),
        "ip_adapter": config["ip_adapter"],
        "case_id": record["case_id"],
        "scene_id": record["scene_id"],
        "candidate_id": record["candidate_id"],
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_mock_adapter(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from generate_images import draw_placeholder

    output_path = Path(record["output_path"])
    ensure_parent(output_path)
    draw_placeholder(output_path, record, config["width"], config["height"])
    return {
        "output_path": output_path.as_posix(),
        "style_backend_effective": "mock",
        "style_backend_message": "generated by mock adapter",
    }


def run_command_adapter(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(record["output_path"])
    ensure_parent(output_path)

    with tempfile.TemporaryDirectory(prefix="member_b_prompt_") as temp_dir:
        payload_path = Path(temp_dir) / "prompt_payload.json"
        write_json(payload_path, build_payload(record, config))
        command = apply_template(
            build_template_values(record, config, payload_path),
            list(config["adapter_command"]),
        )
        subprocess.run(command, check=True)

    if not output_path.exists():
        raise RuntimeError(
            "Adapter command finished without writing the expected output file: "
            f"{output_path}"
        )
    return {
        "output_path": output_path.as_posix(),
        "style_backend_effective": "command",
        "style_backend_message": "generated by command adapter",
    }


def run_python_adapter(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from qwen_image_infer import generate_from_payload

    return generate_from_payload(build_payload(record, config))


def run_single_record(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    adapter_type = config["adapter_type"]
    if adapter_type == "mock":
        return run_mock_adapter(record, config)
    if adapter_type == "python":
        return run_python_adapter(record, config)
    if adapter_type == "command":
        return run_command_adapter(record, config)
    raise RuntimeError(f"Unsupported adapter type: {adapter_type}")


def append_job_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line.rstrip() + "\n")


def main() -> int:
    args = parse_args()
    style_id = str(args.style or DEFAULT_STYLE_ID)
    config = load_runtime_config(args.config, style_id)
    shard = read_json(args.shard)
    status_path = args.status_path or Path(shard.get("status_path") or config["status_path"])
    status_data = load_status(status_path, config)
    status_index = index_status_records(status_data)

    logs_dir = Path(config["logs_dir"])
    job_log = logs_dir / f"{shard['job_name']}.log"
    summary_path = logs_dir / f"{shard['job_name']}.summary.json"

    selected_records = [
        record
        for record in shard["records"]
        if should_run_record(record, status_index, args.only_failed)
    ]
    if args.max_records is not None:
        selected_records = selected_records[: args.max_records]

    summary = {
        "job_name": shard["job_name"],
        "job_index": shard["job_index"],
        "style_id": style_id,
        "model_family": config["model_family"],
        "adapter_type": config["adapter_type"],
        "selected_record_count": len(selected_records),
        "success_count": 0,
        "failed_count": 0,
        "records": [],
    }

    append_job_log(
        job_log,
        f"Starting {shard['job_name']} style={style_id} with {len(selected_records)} record(s)",
    )

    for record in selected_records:
        message = ""
        success = False
        backend_effective = None
        for attempt in range(1, config["retries"] + 2):
            upsert_status_record(status_data, record, "running", attempt, "started")
            save_status(status_path, status_data)
            try:
                result = run_single_record(record, config)
                backend_effective = result.get("style_backend_effective")
                upsert_status_record(
                    status_data,
                    record,
                    "success",
                    attempt,
                    result.get("style_backend_message", "generated"),
                    backend_effective=backend_effective,
                )
                save_status(status_path, status_data)
                append_job_log(
                    job_log,
                    "SUCCESS "
                    f"{record_key(record)} seed={record['seed']} "
                    f"backend={backend_effective} output={record['output_path']}",
                )
                summary["success_count"] += 1
                summary["records"].append(
                    {
                        "record_key": record_key(record),
                        "state": "success",
                        "attempt": attempt,
                        "output_path": record["output_path"],
                        "style_backend_requested": record.get("style_backend_requested"),
                        "style_backend_effective": backend_effective,
                    }
                )
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                upsert_status_record(
                    status_data,
                    record,
                    "failed",
                    attempt,
                    message,
                    backend_effective=backend_effective,
                )
                save_status(status_path, status_data)
                append_job_log(
                    job_log,
                    f"FAILED {record_key(record)} attempt={attempt} error={message}",
                )

        if not success:
            summary["failed_count"] += 1
            summary["records"].append(
                {
                    "record_key": record_key(record),
                    "state": "failed",
                    "message": message,
                    "output_path": record["output_path"],
                    "style_backend_requested": record.get("style_backend_requested"),
                    "style_backend_effective": backend_effective,
                }
            )

    summary["completed_at_epoch"] = int(time.time())
    write_json(summary_path, summary)

    print(
        f"Completed {shard['job_name']}: "
        f"{summary['success_count']} success, {summary['failed_count']} failed"
    )
    print(f"Job log: {job_log}")
    print(f"Summary: {summary_path}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
