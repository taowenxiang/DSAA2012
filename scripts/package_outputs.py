"""Package selected story images into the final output directory."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any


DEFAULT_SELECTION = Path("outputs") / "intermediate" / "selection_results.json"
DEFAULT_OUTPUT_DIR = Path("outputs") / "final"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "submission_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Selection results produced by scripts/rerank_candidates.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where final packaged images will be written.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path for the packaged submission manifest JSON.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before packaging.",
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


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")

        chunk_length = struct.unpack("!I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or chunk_length != 13:
            raise ValueError(f"PNG missing IHDR chunk: {path}")

        width = struct.unpack("!I", handle.read(4))[0]
        height = struct.unpack("!I", handle.read(4))[0]
        return width, height


def package_selection(
    selection: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    packaged_cases: list[dict[str, Any]] = []
    image_count = 0

    for case in selection.get("cases", []):
        case_id = str(case["case_id"])
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        panels: list[dict[str, Any]] = []
        for panel in case.get("selected_panels", []):
            scene_id = int(panel["scene_id"])
            source_path = Path(str(panel["selected_path"]))
            if not source_path.exists():
                raise SystemExit(f"Selected image does not exist: {source_path}")

            target_path = case_dir / f"scene_{scene_id}.png"
            shutil.copy2(source_path, target_path)
            width, height = read_png_size(target_path)

            panels.append(
                {
                    "scene_id": scene_id,
                    "source_path": source_path.as_posix(),
                    "packaged_path": target_path.as_posix(),
                    "selected_candidate_id": panel["selected_candidate_id"],
                    "selected_seed": panel["selected_seed"],
                    "width": width,
                    "height": height,
                    "scores": panel["scores"],
                }
            )
            image_count += 1

        packaged_cases.append(
            {
                "case_id": case_id,
                "panel_count": len(panels),
                "panels": panels,
            }
        )

    return {
        "package_version": "member-c-v1",
        "source_selection": selection.get("selection_version", "unknown"),
        "case_count": len(packaged_cases),
        "image_count": image_count,
        "cases": packaged_cases,
    }


def main() -> int:
    args = parse_args()
    selection = read_json(args.selection)

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = package_selection(selection, args.output_dir)
    write_json(args.manifest_output, manifest)

    print(
        f"Packaged {manifest['image_count']} final image(s) "
        f"across {manifest['case_count']} case(s)."
    )
    print(f"Final output directory: {args.output_dir}")
    print(f"Submission manifest: {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
