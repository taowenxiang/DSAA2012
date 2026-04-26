"""Run one HPC shard for Member B candidate generation.

The worker reads a shard file produced by generate_images.py and generates
candidate images record-by-record. The default adapter is a local mock so the
pipeline can be tested without the real 30B model; HPC users can switch to the
command adapter and point it at their Qwen-Image inference entrypoint.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from generate_images import draw_placeholder, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def load_runtime_config(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    adapter = raw.get("adapter", {})
    scheduler = raw.get("scheduler", {})
    return {
        "model_family": str(raw.get("model_family", "qwen_image")),
        "model_path": str(raw.get("model_path", "")),
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "dtype": str(raw.get("dtype", "float16")),
        "device": str(raw.get("device", "cuda")),
        "num_inference_steps": int(raw.get("num_inference_steps", 30)),
        "guidance_scale": float(raw.get("guidance_scale", 4.5)),
        "temperature": float(raw.get("temperature", 1.0)),
        "tensor_parallel_size": int(raw.get("tensor_parallel_size", 1)),
        "retries": int(raw.get("retries", 2)),
        "logs_dir": str(raw.get("logs_dir", "outputs/logs/member_b")),
        "status_path": str(raw.get("status_path", "outputs/intermediate/generation_status.json")),
        "adapter_type": str(adapter.get("type", "mock")),
        "adapter_command": adapter.get("command", []),
        "gpus_per_task": int(scheduler.get("gpus_per_task", 1)),
    }


def load_status(path: Path, shard: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)

    payload = {
        "status_version": "member-b-status-v1",
        "model_family": config["model_family"],
        "records": [],
    }
    write_json(path, payload)
    return payload


def save_status(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def record_key(record: dict[str, Any]) -> str:
    return f"{record['case_id']}|{record['scene_id']}|{record['candidate_id']}"


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
) -> None:
    key = record_key(record)
    indexed = index_status_records(status_data)
    payload = {
        "record_key": key,
        "case_id": record["case_id"],
        "scene_id": record["scene_id"],
        "candidate_id": record["candidate_id"],
        "output_path": record["output_path"],
        "seed": record["seed"],
        "state": state,
        "attempt": attempt,
        "message": message,
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
    rendered: list[str] = []
    for token in command:
        rendered.append(token.format(**values))
    return rendered


def build_template_values(record: dict[str, Any], config: dict[str, Any], prompt_path: Path) -> dict[str, Any]:
    return {
        "prompt": record["prompt"],
        "negative_prompt": record["negative_prompt"],
        "seed": record["seed"],
        "output_path": record["output_path"],
        "width": config["width"],
        "height": config["height"],
        "model_path": config["model_path"],
        "model_family": config["model_family"],
        "dtype": config["dtype"],
        "device": config["device"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "temperature": config["temperature"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "prompt_payload_path": prompt_path.as_posix(),
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
        "model_family": config["model_family"],
        "dtype": config["dtype"],
        "device": config["device"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "temperature": config["temperature"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "case_id": record["case_id"],
        "scene_id": record["scene_id"],
        "candidate_id": record["candidate_id"],
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_mock_adapter(record: dict[str, Any], config: dict[str, Any]) -> None:
    output_path = Path(record["output_path"])
    ensure_parent(output_path)
    draw_placeholder(output_path, record, config["width"], config["height"])


def run_command_adapter(record: dict[str, Any], config: dict[str, Any]) -> None:
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


def run_python_adapter(record: dict[str, Any], config: dict[str, Any]) -> None:
    from qwen_image_infer import generate_from_payload

    generate_from_payload(build_payload(record, config))


def run_single_record(record: dict[str, Any], config: dict[str, Any]) -> None:
    adapter_type = config["adapter_type"]
    if adapter_type == "mock":
        run_mock_adapter(record, config)
        return
    if adapter_type == "python":
        run_python_adapter(record, config)
        return
    if adapter_type == "command":
        run_command_adapter(record, config)
        return
    raise RuntimeError(f"Unsupported adapter type: {adapter_type}")


def append_job_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line.rstrip() + "\n")


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config)
    shard = read_json(args.shard)
    status_path = args.status_path or Path(shard.get("status_path") or config["status_path"])
    status_data = load_status(status_path, shard, config)
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
        "model_family": config["model_family"],
        "adapter_type": config["adapter_type"],
        "selected_record_count": len(selected_records),
        "success_count": 0,
        "failed_count": 0,
        "records": [],
    }

    append_job_log(job_log, f"Starting {shard['job_name']} with {len(selected_records)} record(s)")

    for record in selected_records:
        message = ""
        success = False
        for attempt in range(1, config["retries"] + 2):
            upsert_status_record(status_data, record, "running", attempt, "started")
            save_status(status_path, status_data)
            try:
                run_single_record(record, config)
                upsert_status_record(status_data, record, "success", attempt, "generated")
                save_status(status_path, status_data)
                append_job_log(
                    job_log,
                    f"SUCCESS {record_key(record)} seed={record['seed']} output={record['output_path']}",
                )
                summary["success_count"] += 1
                summary["records"].append(
                    {
                        "record_key": record_key(record),
                        "state": "success",
                        "attempt": attempt,
                        "output_path": record["output_path"],
                    }
                )
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                upsert_status_record(status_data, record, "failed", attempt, message)
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
                }
            )

    write_json(summary_path, summary)
    print(
        f"Completed {shard['job_name']}: "
        f"{summary['success_count']} success, {summary['failed_count']} failed"
    )
    print(f"Job log: {job_log}")
    print(f"Summary: {summary_path}")
    if config["adapter_type"] == "command":
        print(
            "Adapter command template: "
            f"{shlex.join(list(config['adapter_command']))}"
        )
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
