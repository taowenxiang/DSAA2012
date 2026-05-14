"""Evaluate SD3 outputs with CLIPScore prompt alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from sd3_story_utils import infer_prompt_id, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms import PILToTensor

    prompt_map = {item["id"]: item["prompt"] for item in read_jsonl(args.prompts)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPScore(model_name_or_path=args.model).to(device)
    rows: list[dict] = []
    for path in sorted(args.image_dir.glob("*.png")):
        prompt_id = infer_prompt_id(path.name)
        prompt = prompt_map.get(prompt_id)
        if prompt is None:
            continue
        image = Image.open(path).convert("RGB")
        tensor = PILToTensor()(image).unsqueeze(0).to(device)
        score = metric(tensor, [prompt]).detach().cpu().item()
        rows.append({"image_path": path.as_posix(), "prompt_id": prompt_id, "clipscore": score})
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if not df.empty:
        print(df.groupby("prompt_id")["clipscore"].mean())
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
