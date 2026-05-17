"""Prepare per-story cast LoRA datasets from generated cast seed images."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qwen_story_utils import normalize_id, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "cast_seed_prompts",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data") / "qwen_story" / "cast_seed_images",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data") / "qwen_story" / "cast_lora_datasets",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clean and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    prepared_cases = 0
    for prompt_path in sorted(args.prompt_dir.glob("*.jsonl")):
        rows = read_jsonl(prompt_path)
        if not rows:
            continue
        case_id = str(rows[0]["case_id"])
        case_key = normalize_id(case_id)
        case_image_dir = args.image_dir / case_key
        if not case_image_dir.exists():
            print(f"Skip {case_id}: missing image dir {case_image_dir}")
            continue

        dataset_dir = args.out_root / case_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict] = []
        index = 1
        for row in rows:
            stem = row["id"]
            matches = sorted(case_image_dir.glob(f"{stem}_cand*.png"))
            if not matches:
                matches = sorted(case_image_dir.glob(f"{stem}.png"))
            for image_path in matches:
                out_stem = f"{index:06d}"
                target_image = dataset_dir / f"{out_stem}{image_path.suffix.lower()}"
                shutil.copy2(image_path, target_image)
                caption = str(row.get("caption") or row["prompt"]).strip()
                (dataset_dir / f"{out_stem}.txt").write_text(caption + "\n", encoding="utf-8")
                manifest_rows.append(
                    {
                        "file_name": target_image.name,
                        "text": caption,
                        "case_id": case_id,
                        "source_prompt_id": row["id"],
                        "source_image": image_path.as_posix(),
                    }
                )
                index += 1

        if not manifest_rows:
            print(f"Skip {case_id}: no generated images matched prompts")
            continue
        write_jsonl(dataset_dir / "metadata.jsonl", manifest_rows)
        prepared_cases += 1
        print(f"Prepared cast LoRA dataset for {case_id}: {len(manifest_rows)} image(s)")

    print(f"Prepared {prepared_cases} cast dataset(s) under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
