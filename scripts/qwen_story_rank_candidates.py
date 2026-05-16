"""Rank one-shot Qwen story page candidates by CLIPScore and DreamSim."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from sd3_story_utils import infer_prompt_id
from qwen_story_utils import read_jsonl


CANDIDATE_RE = re.compile(r"_cand(?P<candidate>\d+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="vertical")
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
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


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def main() -> int:
    args = parse_args()

    import torch
    from dreamsim import dreamsim
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms import PILToTensor

    prompt_map = {item["id"]: item["prompt"] for item in read_jsonl(args.prompts)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_metric = CLIPScore(model_name_or_path=args.clip_model).to(device)
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)

    rows: list[dict] = []
    for path in sorted(args.image_dir.glob("*.png")):
        prompt_id = infer_prompt_id(path.name)
        prompt = prompt_map.get(prompt_id)
        if prompt is None:
            continue

        image = Image.open(path).convert("RGB")
        clip_tensor = PILToTensor()(image).unsqueeze(0).to(device)
        clipscore = clip_metric(clip_tensor, [prompt]).detach().cpu().item()

        panels = split_panels(image, args.num_panels, args.layout)
        if len(panels) < 2:
            continue
        pair_distances: list[float] = []
        panel_tensors = [dreamsim_preprocess(panel).to(device) for panel in panels]
        for left_index in range(len(panel_tensors)):
            for right_index in range(left_index + 1, len(panel_tensors)):
                with torch.no_grad():
                    distance = dreamsim_model(panel_tensors[left_index], panel_tensors[right_index])
                pair_distances.append(distance.detach().cpu().item())
        dreamsim_mean = sum(pair_distances) / len(pair_distances)

        candidate_match = CANDIDATE_RE.search(path.stem)
        rows.append(
            {
                "image_path": path.as_posix(),
                "prompt_id": prompt_id,
                "candidate": int(candidate_match.group("candidate")) if candidate_match else -1,
                "clipscore": clipscore,
                "dreamsim_distance": dreamsim_mean,
            }
        )

    if not rows:
        raise SystemExit(f"No candidate rows found in {args.image_dir}")

    df = pd.DataFrame(rows)
    clip_z = zscore(df["clipscore"].tolist())
    dreamsim_z = zscore(df["dreamsim_distance"].tolist())
    df["clipscore_z"] = clip_z
    df["dreamsim_z"] = dreamsim_z
    df["combined_score"] = df["clipscore_z"] - df["dreamsim_z"]
    df["rank_within_prompt"] = (
        df.groupby("prompt_id")["combined_score"].rank(ascending=False, method="first").astype(int)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["prompt_id", "rank_within_prompt", "candidate"]).to_csv(args.out, index=False)
    print(f"Saved ranking CSV to {args.out}")
    print(df.sort_values("combined_score", ascending=False).head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
