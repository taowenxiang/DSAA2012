"""Prepare square SD3 Storyboard LoRA training data from local and external sources."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from sd3_story_utils import (
    TRIGGER_STORYBOARD,
    clean_text,
    iter_image_files,
    write_jsonl,
)


LOCAL_LICENSE_NOTE = "Generated task-local images from this course project; use for course research only."
COMIX_LICENSE_NOTE = "CoMix pages: public-domain comic scans with CC BY-SA 4.0 metadata; cite dataset card."
FLINTSTONES_LICENSE_NOTE = "FlintstonesSV++: use for academic-course experiments and cite dataset card."
FLINTSTONES_GROUP_KEYS = (
    "story_id",
    "video_id",
    "episode",
    "episode_id",
    "global_id",
    "movie_id",
    "scene_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "sd3_story" / "train_storyboard",
    )
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["local", "comix", "flintstones"],
        default=["local"],
        help="Data sources to prepare. External sources require datasets and network/cache access.",
    )
    parser.add_argument(
        "--local-runs",
        nargs="+",
        default=[
            "outputs/runs/run_0003_storybook/final",
            "outputs/runs/run_0002_watercolor/final",
        ],
    )
    parser.add_argument("--max-local", type=int, default=0)
    parser.add_argument("--max-comix", type=int, default=80)
    parser.add_argument("--max-flintstones", type=int, default=80)
    parser.add_argument("--comix-dataset", default="emanuelevivoli/comix-v0_1-pages")
    parser.add_argument("--comix-split", default="train")
    parser.add_argument("--flintstones-dataset", default="Janak12/FlintstonesSV_Plus_Plus")
    parser.add_argument("--flintstones-split", default="train")
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing prepared training samples.",
    )
    return parser.parse_args()


def fit_with_padding(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def compose_panels(paths: list[Path], size: int, layout: str = "horizontal") -> Image.Image:
    panel_count = len(paths)
    if panel_count < 2:
        raise ValueError("Need at least two panels")
    if layout == "horizontal":
        panel_w = size // panel_count
        panel_h = size
        canvas = Image.new("RGB", (size, size), "white")
        for index, path in enumerate(paths):
            panel = Image.open(path).convert("RGB")
            panel = fit_crop(panel, panel_w, panel_h)
            canvas.paste(panel, (index * panel_w, 0))
        draw = ImageDraw.Draw(canvas)
        for index in range(1, panel_count):
            x = index * panel_w
            draw.line((x, 0, x, size), fill=(0, 0, 0), width=5)
        return canvas

    panel_w = size
    panel_h = size // panel_count
    canvas = Image.new("RGB", (size, size), "white")
    for index, path in enumerate(paths):
        panel = Image.open(path).convert("RGB")
        panel = fit_crop(panel, panel_w, panel_h)
        canvas.paste(panel, (0, index * panel_h))
    draw = ImageDraw.Draw(canvas)
    for index in range(1, panel_count):
        y = index * panel_h
        draw.line((0, y, size, y), fill=(0, 0, 0), width=5)
    return canvas


def fit_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    target_ratio = width / height
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_item(
    image: Image.Image,
    out_dir: Path,
    index: int,
    text: str,
    source: str,
    num_panels: int,
    layout: str,
    license_note: str,
) -> dict[str, Any]:
    file_name = f"{index:06d}.png"
    out_path = out_dir / file_name
    image.save(out_path)
    txt_path = out_dir / f"{index:06d}.txt"
    txt_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return {
        "file_name": file_name,
        "text": text,
        "source": source,
        "num_panels": num_panels,
        "layout": layout,
        "license_note": license_note,
    }


def collect_local_cases(run_dir: Path) -> list[tuple[str, list[Path]]]:
    cases: list[tuple[str, list[Path]]] = []
    if not run_dir.exists():
        return cases
    for case_dir in sorted([item for item in run_dir.iterdir() if item.is_dir()]):
        panels = sorted(case_dir.glob("scene_*.png"))
        if len(panels) >= 2:
            cases.append((case_dir.name, panels))
    return cases


def prepare_local(args: argparse.Namespace, out_dir: Path, start_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = start_index
    for run in args.local_runs:
        run_dir = Path(run)
        for case_id, panels in collect_local_cases(run_dir):
            if args.max_local and len(rows) >= args.max_local:
                return rows
            for panel_count in [2, 3]:
                if len(panels) < panel_count:
                    continue
                image = compose_panels(panels[:panel_count], args.resolution, layout="horizontal")
                text = clean_text(
                    f"a {TRIGGER_STORYBOARD} {panel_count}-panel storyboard illustration, "
                    "clear panel boundaries, left-to-right story flow, consistent character, "
                    "coherent color palette, cinematic storybook style"
                )
                rows.append(
                    save_item(
                        image=image,
                        out_dir=out_dir,
                        index=index,
                        text=text,
                        source=f"local:{run_dir.as_posix()}:{case_id}",
                        num_panels=panel_count,
                        layout="horizontal",
                        license_note=LOCAL_LICENSE_NOTE,
                    )
                )
                index += 1
    return rows


def load_dataset_streaming(name: str, split: str, cache_dir: Path | None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    return load_dataset(name, **kwargs)


def image_from_row(row: dict[str, Any]) -> Image.Image | None:
    for key, value in row.items():
        if hasattr(value, "convert"):
            return value.convert("RGB")
        if isinstance(value, dict):
            path = value.get("path")
            if path:
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    continue
    return None


def caption_from_row(row: dict[str, Any]) -> str:
    for key in ["caption", "text", "description", "sentence"]:
        if key in row and row[key]:
            return str(row[key])
    return ""


def flintstones_group_key(row: dict[str, Any]) -> str | None:
    for key in FLINTSTONES_GROUP_KEYS:
        if key in row and row[key] is not None:
            return f"{key}:{row[key]}"
    return None


def save_flintstones_group(
    group: list[tuple[Image.Image, str]],
    out_dir: Path,
    index: int,
    resolution: int,
    source_suffix: str,
) -> dict[str, Any]:
    temp_paths: list[Path] = []
    temp_dir = out_dir / "_tmp_flintstones"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for temp_index, (img, _) in enumerate(group[:3]):
        temp_path = temp_dir / f"{index}_{temp_index}.png"
        img.save(temp_path)
        temp_paths.append(temp_path)

    try:
        image3 = compose_panels(temp_paths, resolution, layout="horizontal")
    finally:
        for path in temp_paths:
            try:
                path.unlink()
            except OSError:
                pass

    joined_caption = "; ".join(clean_text(item[1]) for item in group[:3] if item[1])
    text = clean_text(
        f"a {TRIGGER_STORYBOARD} three-panel cartoon storyboard illustration, "
        f"clear panel boundaries, consistent cartoon characters, {joined_caption}"
    )
    return save_item(
        image=image3,
        out_dir=out_dir,
        index=index,
        text=text,
        source=f"flintstones:{source_suffix}",
        num_panels=3,
        layout="horizontal",
        license_note=FLINTSTONES_LICENSE_NOTE,
    )


def prepare_comix(args: argparse.Namespace, out_dir: Path, start_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = start_index
    ds = load_dataset_streaming(
        args.comix_dataset,
        split=args.comix_split,
        cache_dir=args.dataset_cache_dir,
    )
    for row in ds:
        image = image_from_row(row)
        if image is None:
            continue
        if min(image.size) < 256:
            continue
        prepared = fit_with_padding(image, args.resolution)
        raw_title = row.get("title") or row.get("book_title") or row.get("comic") or ""
        text = clean_text(
            f"a {TRIGGER_STORYBOARD} comic storyboard page, clear panel layout, "
            f"visual story sequence, coherent composition, {raw_title}"
        )
        rows.append(
            save_item(
                image=prepared,
                out_dir=out_dir,
                index=index,
                text=text,
                source="comix",
                num_panels=0,
                layout="page",
                license_note=COMIX_LICENSE_NOTE,
            )
        )
        index += 1
        if len(rows) >= args.max_comix:
            break
    return rows


def prepare_flintstones(args: argparse.Namespace, out_dir: Path, start_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = start_index
    ds = load_dataset_streaming(
        args.flintstones_dataset,
        split=args.flintstones_split,
        cache_dir=args.dataset_cache_dir,
    )
    grouped: dict[str, list[tuple[Image.Image, str]]] = {}
    fallback_buffer: list[tuple[Image.Image, str]] = []
    warned_no_group_key = False
    for row in ds:
        image = image_from_row(row)
        if image is None:
            continue
        item = (image, caption_from_row(row))
        group_key = flintstones_group_key(row)
        if group_key is None:
            if not warned_no_group_key:
                print(
                    "Warning: FlintstonesSV++ rows do not expose a recognized story grouping key; "
                    "falling back to consecutive triples."
                )
                warned_no_group_key = True
            fallback_buffer.append(item)
            if len(fallback_buffer) < 3:
                continue
            group = fallback_buffer[:3]
            fallback_buffer = fallback_buffer[3:]
            source_suffix = "consecutive"
        else:
            grouped.setdefault(group_key, []).append(item)
            if len(grouped[group_key]) < 3:
                continue
            group = grouped[group_key][:3]
            grouped[group_key] = grouped[group_key][3:]
            source_suffix = group_key

        rows.append(
            save_flintstones_group(
                group=group,
                out_dir=out_dir,
                index=index,
                resolution=args.resolution,
                source_suffix=source_suffix,
            )
        )
        index += 1
        if len(rows) >= args.max_flintstones:
            break
    temp_dir = out_dir / "_tmp_flintstones"
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    return rows


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if "local" in args.sources:
        local_rows = prepare_local(args, out_dir, len(rows) + 1)
        rows.extend(local_rows)
        print(f"Prepared {len(local_rows)} local storyboard sample(s).")
    if "comix" in args.sources:
        comix_rows = prepare_comix(args, out_dir, len(rows) + 1)
        rows.extend(comix_rows)
        print(f"Prepared {len(comix_rows)} CoMix sample(s).")
    if "flintstones" in args.sources:
        flint_rows = prepare_flintstones(args, out_dir, len(rows) + 1)
        rows.extend(flint_rows)
        print(f"Prepared {len(flint_rows)} FlintstonesSV++ sample(s).")

    if not rows:
        raise SystemExit("No training samples were prepared.")
    manifest_count = write_jsonl(out_dir / "metadata.jsonl", rows)
    print(f"Wrote {manifest_count} training sample manifest row(s): {out_dir / 'metadata.jsonl'}")
    print(f"Image count in {out_dir}: {len(iter_image_files(out_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
