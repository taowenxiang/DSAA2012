"""Select one final candidate image per scene from the generation manifest.

The script is designed to be robust in lightweight environments:

- if Pillow is available, it uses simple image-quality and continuity features
- otherwise, it falls back to byte-level file features so the pipeline still
  closes end-to-end
- CLIP-style text-image scoring can be added later without changing the output
  schema
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs") / "intermediate" / "generation_manifest.json"
DEFAULT_OUTPUT = Path("outputs") / "intermediate" / "selection_results.json"


@dataclass(frozen=True)
class CandidateRecord:
    case_id: str
    scene_id: int
    candidate_id: int
    seed: int
    prompt: str
    negative_prompt: str
    output_path: Path
    status: str


@dataclass
class CandidateFeatures:
    quality_score: float
    continuity_vector: list[float]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Generation manifest produced by scripts/generate_images.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the selection result JSON.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pillow", "bytes"],
        default="auto",
        help="Feature backend for scoring candidates.",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.55,
        help="Weight for single-image quality features.",
    )
    parser.add_argument(
        "--continuity-weight",
        type=float,
        default=0.45,
        help="Weight for continuity with the previous selected panel.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def entropy_from_bytes(raw: bytes) -> float:
    if not raw:
        return 0.0
    counts = [0] * 256
    for byte in raw:
        counts[byte] += 1
    total = len(raw)
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def chunk_means(raw: bytes, count: int = 32) -> list[float]:
    if not raw:
        return [0.0] * count
    chunk_size = max(1, len(raw) // count)
    means: list[float] = []
    for index in range(count):
        start = index * chunk_size
        end = len(raw) if index == count - 1 else min(len(raw), (index + 1) * chunk_size)
        chunk = raw[start:end]
        if not chunk:
            means.append(0.0)
            continue
        means.append(sum(chunk) / (255.0 * len(chunk)))
    return means


def choose_backend(name: str) -> str:
    if name in {"bytes", "pillow"}:
        return name
    try:
        import PIL  # noqa: F401

        return "pillow"
    except ImportError:
        return "bytes"


def extract_byte_features(path: Path) -> CandidateFeatures:
    raw = path.read_bytes()
    file_size = len(raw)
    entropy = entropy_from_bytes(raw)
    vector = chunk_means(raw)

    normalized_entropy = clamp01(entropy / 8.0)
    normalized_size = clamp01(math.log1p(file_size) / 12.0)
    quality = clamp01((normalized_entropy * 0.65) + (normalized_size * 0.35))

    return CandidateFeatures(
        quality_score=quality,
        continuity_vector=vector,
        metadata={
            "file_size_bytes": file_size,
            "byte_entropy": round(entropy, 6),
        },
    )


def extract_pillow_features(path: Path) -> CandidateFeatures:
    from PIL import Image, ImageStat

    image = Image.open(path).convert("RGB")
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    mean = float(stat.mean[0])
    stddev = float(stat.stddev[0])
    histogram = grayscale.histogram()
    total = float(sum(histogram)) or 1.0
    vector = [count / total for count in histogram]

    brightness_score = 1.0 - min(abs(mean - 127.5) / 127.5, 1.0)
    contrast_score = clamp01(stddev / 64.0)
    file_size_score = clamp01(math.log1p(path.stat().st_size) / 12.0)
    quality = clamp01(
        (brightness_score * 0.35) + (contrast_score * 0.4) + (file_size_score * 0.25)
    )

    return CandidateFeatures(
        quality_score=quality,
        continuity_vector=vector,
        metadata={
            "width": image.width,
            "height": image.height,
            "mean_brightness": round(mean, 4),
            "stddev_brightness": round(stddev, 4),
            "file_size_bytes": path.stat().st_size,
        },
    )


def extract_features(path: Path, backend: str) -> CandidateFeatures:
    if backend == "pillow":
        return extract_pillow_features(path)
    return extract_byte_features(path)


def parse_manifest(path: Path) -> list[CandidateRecord]:
    manifest = read_json(path)
    records: list[CandidateRecord] = []
    for raw in manifest.get("candidates", []):
        records.append(
            CandidateRecord(
                case_id=str(raw["case_id"]),
                scene_id=int(raw["scene_id"]),
                candidate_id=int(raw["candidate_id"]),
                seed=int(raw["seed"]),
                prompt=str(raw["prompt"]),
                negative_prompt=str(raw.get("negative_prompt", "")),
                output_path=Path(str(raw["output_path"])),
                status=str(raw.get("status", "unknown")),
            )
        )
    return records


def group_records(records: list[CandidateRecord]) -> dict[str, dict[int, list[CandidateRecord]]]:
    grouped: dict[str, dict[int, list[CandidateRecord]]] = {}
    for record in records:
        grouped.setdefault(record.case_id, {}).setdefault(record.scene_id, []).append(record)
    for scenes in grouped.values():
        for scene_records in scenes.values():
            scene_records.sort(key=lambda item: item.candidate_id)
    return grouped


def score_candidate(
    features: CandidateFeatures,
    previous_features: CandidateFeatures | None,
    quality_weight: float,
    continuity_weight: float,
) -> dict[str, float]:
    continuity = 0.0
    if previous_features is not None:
        continuity = cosine_similarity(
            features.continuity_vector, previous_features.continuity_vector
        )
        continuity = clamp01((continuity + 1.0) / 2.0)

    total = (features.quality_score * quality_weight) + (continuity * continuity_weight)
    return {
        "quality": round(features.quality_score, 6),
        "continuity": round(continuity, 6),
        "total": round(total, 6),
    }


def select_case(
    case_id: str,
    scenes: dict[int, list[CandidateRecord]],
    backend: str,
    quality_weight: float,
    continuity_weight: float,
) -> dict[str, Any]:
    selected_panels: list[dict[str, Any]] = []
    previous_features: CandidateFeatures | None = None

    for scene_id in sorted(scenes):
        candidates = scenes[scene_id]
        evaluated: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None

        for record in candidates:
            if not record.output_path.exists():
                raise SystemExit(f"Missing candidate image: {record.output_path}")

            features = extract_features(record.output_path, backend)
            scores = score_candidate(
                features=features,
                previous_features=previous_features,
                quality_weight=quality_weight,
                continuity_weight=continuity_weight,
            )

            entry = {
                "candidate_id": record.candidate_id,
                "seed": record.seed,
                "output_path": record.output_path.as_posix(),
                "scores": scores,
                "feature_backend": backend,
                "feature_metadata": features.metadata,
                "prompt": record.prompt,
                "_features": features,
            }
            evaluated.append(entry)

            if best_entry is None:
                best_entry = entry
                continue

            current_total = entry["scores"]["total"]
            best_total = best_entry["scores"]["total"]
            if current_total > best_total:
                best_entry = entry
            elif current_total == best_total and entry["candidate_id"] < best_entry["candidate_id"]:
                best_entry = entry

        assert best_entry is not None
        previous_features = best_entry["_features"]

        alternatives: list[dict[str, Any]] = []
        for entry in evaluated:
            alternatives.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "seed": entry["seed"],
                    "output_path": entry["output_path"],
                    "scores": entry["scores"],
                    "feature_metadata": entry["feature_metadata"],
                }
            )

        selected_panels.append(
            {
                "scene_id": scene_id,
                "selected_candidate_id": best_entry["candidate_id"],
                "selected_seed": best_entry["seed"],
                "selected_path": best_entry["output_path"],
                "scores": best_entry["scores"],
                "alternatives": alternatives,
            }
        )

    return {
        "case_id": case_id,
        "selected_panels": selected_panels,
        "selected_sequence": [panel["selected_path"] for panel in selected_panels],
    }


def main() -> int:
    args = parse_args()
    backend = choose_backend(args.backend)

    quality_weight = max(0.0, args.quality_weight)
    continuity_weight = max(0.0, args.continuity_weight)
    weight_sum = quality_weight + continuity_weight
    if weight_sum == 0.0:
        raise SystemExit("At least one weight must be positive")
    quality_weight /= weight_sum
    continuity_weight /= weight_sum

    records = parse_manifest(args.manifest)
    if not records:
        raise SystemExit(f"No candidate records found in {args.manifest}")

    grouped = group_records(records)
    cases: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        cases.append(
            select_case(
                case_id=case_id,
                scenes=grouped[case_id],
                backend=backend,
                quality_weight=quality_weight,
                continuity_weight=continuity_weight,
            )
        )

    result = {
        "selection_version": "member-c-v1",
        "manifest_path": args.manifest.as_posix(),
        "backend": backend,
        "weights": {
            "quality": round(quality_weight, 6),
            "continuity": round(continuity_weight, 6),
        },
        "case_count": len(cases),
        "panel_count": sum(len(case["selected_panels"]) for case in cases),
        "cases": cases,
    }
    write_json(args.output, result)

    print(
        f"Selected {result['panel_count']} final panel image(s) "
        f"across {result['case_count']} case(s) using backend={backend}."
    )
    print(f"Selection results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
