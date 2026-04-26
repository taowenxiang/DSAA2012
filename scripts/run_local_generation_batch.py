"""Run the full Member B batch in one local process.

This loader is the practical path for a single 4-GPU Qwen-Image instance:
the pipeline is loaded once, then reused across all selected candidate
records.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from generate_images import read_json, write_json
from run_hpc_generation import (
    append_job_log,
    index_status_records,
    load_runtime_config,
    load_status,
    record_key,
    run_single_record,
    save_status,
    should_run_record,
    upsert_status_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "member_b_generation_config.local_4gpu.json",
        help="Member B generation config JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs") / "intermediate" / "generation_manifest.json",
        help="Generation manifest produced by scripts/generate_images.py.",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Optional override for the shared generation status JSON.",
    )
    parser.add_argument(
        "--job-name",
        default="member_b_local_all",
        help="Logical job name used for logs and summary files.",
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


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config)
    manifest = read_json(args.manifest)

    status_path = args.status_path or Path(config["status_path"])
    status_data = load_status(status_path, {"job_name": args.job_name}, config)
    status_index = index_status_records(status_data)

    logs_dir = Path(config["logs_dir"])
    job_log = logs_dir / f"{args.job_name}.log"
    summary_path = logs_dir / f"{args.job_name}.summary.json"

    selected_records = [
        record
        for record in manifest["candidates"]
        if should_run_record(record, status_index, args.only_failed)
    ]
    if args.max_records is not None:
        selected_records = selected_records[: args.max_records]

    summary = {
        "job_name": args.job_name,
        "mode": "local_batch",
        "model_family": config["model_family"],
        "adapter_type": config["adapter_type"],
        "selected_record_count": len(selected_records),
        "success_count": 0,
        "failed_count": 0,
        "started_at_epoch": int(time.time()),
        "records": [],
    }

    append_job_log(job_log, f"Starting {args.job_name} with {len(selected_records)} record(s)")

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

    summary["completed_at_epoch"] = int(time.time())
    write_json(summary_path, summary)

    print(
        f"Completed {args.job_name}: "
        f"{summary['success_count']} success, {summary['failed_count']} failed"
    )
    print(f"Job log: {job_log}")
    print(f"Summary: {summary_path}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
