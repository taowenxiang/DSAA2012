"""Rank Qwen story page candidates with page-level consistency signals."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from qwen_story_utils import read_jsonl
from sd3_story_utils import infer_prompt_id


CANDIDATE_RE = re.compile(r"_cand(?P<candidate>\d+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="vertical")
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--disable-clipscore", action="store_true")
    parser.add_argument("--require-dreamsim", action="store_true")
    parser.add_argument("--require-clipscore", action="store_true")
    parser.add_argument("--dreamsim-cache-dir", type=Path, default=None)
    parser.add_argument("--torch-hub-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def split_panels(image: Image.Image, num_panels: int, layout: str) -> list[Image.Image]:
    w, h = image.size
    panels: list[Image.Image] = []
    for index in range(num_panels):
        if layout == "vertical":
            top = int(index * h / num_panels)
            bottom = int((index + 1) * h / num_panels)
            panels.append(image.crop((0, top, w, bottom)))
        else:
            left = int(index * w / num_panels)
            right = int((index + 1) * w / num_panels)
            panels.append(image.crop((left, 0, right, h)))
    return panels


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def rgb_histogram_vector(image: Image.Image, bins: int = 16) -> list[float]:
    rgb = image.convert("RGB")
    histogram = rgb.histogram()
    merged: list[float] = []
    channel_size = 256
    bucket_size = channel_size // bins
    for channel in range(3):
        start = channel * channel_size
        values = histogram[start : start + channel_size]
        for bucket in range(bins):
            bucket_start = bucket * bucket_size
            bucket_end = channel_size if bucket == bins - 1 else (bucket + 1) * bucket_size
            merged.append(float(sum(values[bucket_start:bucket_end])))
    total = sum(merged) or 1.0
    return [value / total for value in merged]


def mean_rgb(image: Image.Image) -> tuple[float, float, float]:
    stat = ImageStat.Stat(image.convert("RGB"))
    return (float(stat.mean[0]), float(stat.mean[1]), float(stat.mean[2]))


def color_similarity(left: Image.Image, right: Image.Image) -> float:
    left_rgb = mean_rgb(left)
    right_rgb = mean_rgb(right)
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_rgb, right_rgb)))
    max_distance = math.sqrt(3 * (255.0**2))
    return clamp01(1.0 - distance / max_distance)


def gutter_clarity_score(image: Image.Image, num_panels: int, layout: str) -> float:
    grayscale = image.convert("L")
    w, h = grayscale.size
    strip_scores: list[float] = []
    strip_half = 6
    for index in range(1, num_panels):
        if layout == "vertical":
            y = int(index * h / num_panels)
            top = max(0, y - strip_half)
            bottom = min(h, y + strip_half)
            strip = grayscale.crop((0, top, w, bottom))
        else:
            x = int(index * w / num_panels)
            left = max(0, x - strip_half)
            right = min(w, x + strip_half)
            strip = grayscale.crop((left, 0, right, h))
        stat = ImageStat.Stat(strip)
        stddev = float(stat.stddev[0])
        strip_scores.append(clamp01(1.0 - stddev / 64.0))
    if not strip_scores:
        return 0.0
    return sum(strip_scores) / len(strip_scores)


def score_layout(image: Image.Image, num_panels: int, layout: str) -> float:
    w, h = image.size
    orientation_ok = 1.0
    if layout == "vertical" and h < w:
        orientation_ok = 0.0
    if layout == "horizontal" and w < h:
        orientation_ok = 0.0
    gutter_score = gutter_clarity_score(image, num_panels, layout)
    return clamp01((orientation_ok * 0.35) + (gutter_score * 0.65))


def score_color_coherence(panels: list[Image.Image]) -> float:
    if len(panels) < 2:
        return 0.0
    scores: list[float] = []
    for left_index in range(len(panels)):
        for right_index in range(left_index + 1, len(panels)):
            scores.append(color_similarity(panels[left_index], panels[right_index]))
    return sum(scores) / len(scores)


def score_histogram_coherence(panels: list[Image.Image]) -> float:
    if len(panels) < 2:
        return 0.0
    vectors = [rgb_histogram_vector(panel) for panel in panels]
    scores: list[float] = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            similarity = cosine_similarity(vectors[left_index], vectors[right_index])
            scores.append(clamp01((similarity + 1.0) / 2.0))
    return sum(scores) / len(scores)


def score_quality(image: Image.Image, panels: list[Image.Image]) -> float:
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    mean = float(stat.mean[0])
    stddev = float(stat.stddev[0])
    brightness_score = 1.0 - min(abs(mean - 127.5) / 127.5, 1.0)
    contrast_score = clamp01(stddev / 64.0)
    panel_balance = 0.0
    if panels:
        panel_means = [float(ImageStat.Stat(panel.convert("L")).mean[0]) for panel in panels]
        spread = max(panel_means) - min(panel_means)
        panel_balance = clamp01(1.0 - spread / 128.0)
    return clamp01((brightness_score * 0.25) + (contrast_score * 0.5) + (panel_balance * 0.25))


def maybe_clipscore(
    prompt: str,
    image: Image.Image,
    clip_model: str,
    enabled: bool,
    required: bool = False,
) -> float | None:
    if not enabled:
        if required:
            raise RuntimeError("CLIPScore is required but clipscore scoring is disabled")
        return None
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel

        def _unwrap_feature_tensor(value: Any) -> Any:
            if hasattr(value, "image_embeds") and value.image_embeds is not None:
                return value.image_embeds
            if hasattr(value, "text_embeds") and value.text_embeds is not None:
                return value.text_embeds
            if hasattr(value, "pooler_output") and value.pooler_output is not None:
                return value.pooler_output
            if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
                hidden = value.last_hidden_state
                if getattr(hidden, "ndim", None) == 3:
                    return hidden[:, 0, :]
                return hidden
            return value

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(clip_model)
        model = CLIPModel.from_pretrained(clip_model).to(device)
        model.eval()

        with torch.no_grad():
            image_inputs = processor(images=image, return_tensors="pt")
            text_inputs = processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features = _unwrap_feature_tensor(image_features)
            text_features = _unwrap_feature_tensor(text_features)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            similarity = torch.matmul(text_features, image_features.T).squeeze()
        return float(similarity.detach().cpu().item() * 100.0)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise RuntimeError(f"CLIPScore required but unavailable: {exc}") from exc
        return None


def score_perceptual_coherence(
    panels: list[Image.Image],
    *,
    require_dreamsim: bool = False,
    dreamsim_cache_dir: Path | None = None,
    torch_hub_dir: Path | None = None,
) -> tuple[float, str]:
    if len(panels) < 2:
        return 0.0, "none"
    try:
        import torch
        from dreamsim import dreamsim
        if torch_hub_dir is not None:
            torch_hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(torch_hub_dir.as_posix())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"pretrained": True, "device": device}
        if dreamsim_cache_dir is not None:
            dreamsim_cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = dreamsim_cache_dir.as_posix()
        model, preprocess = dreamsim(**kwargs)
        similarities: list[float] = []
        tensors = [preprocess(panel).to(device) for panel in panels]
        for left_index in range(len(tensors)):
            for right_index in range(left_index + 1, len(tensors)):
                with torch.no_grad():
                    distance = float(model(tensors[left_index], tensors[right_index]).detach().cpu().item())
                similarities.append(1.0 / (1.0 + max(distance, 0.0)))
        return sum(similarities) / len(similarities), "dreamsim"
    except Exception as exc:  # noqa: BLE001
        if require_dreamsim:
            raise RuntimeError(f"DreamSim required but unavailable: {exc}") from exc
        return score_histogram_coherence(panels), "histogram"


def load_prompt_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(path):
        rows[str(item["id"])] = item
    return rows


def aggregate_total_score(
    *,
    layout_score: float,
    perceptual_score: float,
    color_score: float,
    quality_score: float,
    clipscore: float | None,
) -> float:
    weighted = (
        (layout_score * 0.35)
        + (perceptual_score * 0.3)
        + (color_score * 0.2)
        + (quality_score * 0.15)
    )
    if clipscore is not None:
        # CLIPScore raw ranges vary, so gently normalize and blend it as a weak prior.
        weighted = (weighted * 0.9) + (clamp01(clipscore / 100.0) * 0.1)
    return clamp01(weighted)


def build_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)

    prompts: list[dict[str, Any]] = []
    for prompt_id in sorted(grouped):
        ranked = sorted(grouped[prompt_id], key=lambda item: (-item["combined_score"], item["candidate"]))
        prompts.append(
            {
                "prompt_id": prompt_id,
                "top_candidates": ranked[:top_k],
            }
        )
    return {
        "summary_version": "qwen-story-page-rank-v1",
        "prompt_count": len(prompts),
        "top_k": top_k,
        "prompts": prompts,
    }


def main() -> int:
    args = parse_args()
    if args.require_clipscore and args.disable_clipscore:
        raise SystemExit("--require-clipscore cannot be combined with --disable-clipscore")
    prompt_map = load_prompt_map(args.prompts)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.image_dir.glob("*.png")):
        prompt_id = infer_prompt_id(path.name)
        prompt_row = prompt_map.get(prompt_id)
        if prompt_row is None:
            continue

        image = Image.open(path).convert("RGB")
        num_panels = int(prompt_row.get("num_panels", args.num_panels))
        layout = str(prompt_row.get("layout", args.layout))
        panels = split_panels(image, num_panels=num_panels, layout=layout)

        layout_score = score_layout(image, num_panels=num_panels, layout=layout)
        perceptual_score, perceptual_backend = score_perceptual_coherence(
            panels,
            require_dreamsim=bool(args.require_dreamsim),
            dreamsim_cache_dir=args.dreamsim_cache_dir,
            torch_hub_dir=args.torch_hub_dir,
        )
        color_score = score_color_coherence(panels)
        quality_score = score_quality(image, panels)
        clipscore = maybe_clipscore(
            prompt=str(prompt_row["prompt"]),
            image=image,
            clip_model=args.clip_model,
            enabled=not args.disable_clipscore,
            required=bool(args.require_clipscore),
        )
        combined_score = aggregate_total_score(
            layout_score=layout_score,
            perceptual_score=perceptual_score,
            color_score=color_score,
            quality_score=quality_score,
            clipscore=clipscore,
        )

        candidate_match = CANDIDATE_RE.search(path.stem)
        rows.append(
            {
                "image_path": path.as_posix(),
                "prompt_id": prompt_id,
                "case_id": prompt_row.get("case_id", prompt_id),
                "num_panels": num_panels,
                "layout": layout,
                "candidate": int(candidate_match.group("candidate")) if candidate_match else -1,
                "seed": int(candidate_match.group("seed")) if candidate_match else -1,
                "layout_score": round(layout_score, 6),
                "perceptual_score": round(perceptual_score, 6),
                "perceptual_backend": perceptual_backend,
                "color_score": round(color_score, 6),
                "quality_score": round(quality_score, 6),
                "clipscore": None if clipscore is None else round(clipscore, 6),
                "combined_score": round(combined_score, 6),
            }
        )

    if not rows:
        raise SystemExit(f"No candidate rows found in {args.image_dir}")

    rows.sort(key=lambda item: (str(item["prompt_id"]), -float(item["combined_score"]), int(item["candidate"])))

    rank_index: dict[str, int] = {}
    for row in rows:
        prompt_id = str(row["prompt_id"])
        rank_index[prompt_id] = rank_index.get(prompt_id, 0) + 1
        row["rank_within_prompt"] = rank_index[prompt_id]

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    except Exception:  # noqa: BLE001
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.summary_out is not None:
        summary = build_summary(rows, top_k=args.top_k)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Saved ranking results to {args.out}")
    if args.summary_out is not None:
        print(f"Saved ranking summary to {args.summary_out}")
    print("Top-ranked candidates:")
    for row in [item for item in rows if int(item["rank_within_prompt"]) == 1][:10]:
        print(
            f"{row['prompt_id']} cand={row['candidate']} combined={row['combined_score']:.4f} "
            f"layout={row['layout_score']:.4f} perceptual={row['perceptual_score']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
