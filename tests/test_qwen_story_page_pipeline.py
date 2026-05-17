from __future__ import annotations

import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from qwen_story_infer import build_payload
from qwen_story_rank_candidates import (
    maybe_clipscore,
    score_color_coherence,
    score_layout,
    score_perceptual_coherence,
    split_panels,
)
from run_qwen_story_page_pipeline import build_scene_setting_paths
from qwen_story_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    build_page_prompt_row,
)
from PIL import Image


class BuildPagePromptRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.story = {
            "case_id": "01",
            "characters": ["Mina", "Mochi"],
            "panels": [
                {
                    "scene_id": 1,
                    "action": "Mina opens the bakery door",
                    "setting_hint": "outside a small bakery",
                },
                {
                    "scene_id": 2,
                    "action": "Mina shares bread with Mochi",
                    "setting_hint": "inside the warm bakery",
                },
                {
                    "scene_id": 3,
                    "action": "They walk home together at sunset",
                    "setting_hint": "on a quiet street",
                },
            ],
        }

    def test_two_scene_prompt_uses_vertical_page_language(self) -> None:
        row = build_page_prompt_row(self.story, num_panels=2, layout="vertical")
        prompt = row["prompt"]

        self.assertIn("two-panel flat multi-panel storyboard layout", prompt)
        self.assertIn("stacked vertical panels", prompt)
        self.assertIn("clear gutters", prompt)
        self.assertIn("top-to-bottom narrative flow", prompt)
        self.assertIn("single flat canvas", prompt)
        self.assertNotIn("storybook page", prompt)
        self.assertIn("top panel: Mina opens the bakery door", prompt)
        self.assertIn("bottom panel: Mina shares bread with Mochi", prompt)
        self.assertEqual(row["width"], 1024)
        self.assertEqual(row["height"], 1024)

    def test_three_scene_prompt_mentions_middle_panel_and_shared_cast(self) -> None:
        row = build_page_prompt_row(self.story, num_panels=3, layout="vertical")
        prompt = row["prompt"]

        self.assertIn("three-panel flat multi-panel storyboard layout", prompt)
        self.assertIn("same recurring cast across all panels", prompt)
        self.assertIn("single flat canvas", prompt)
        self.assertNotIn("storybook page", prompt)
        self.assertIn("top panel: Mina opens the bakery door", prompt)
        self.assertIn("middle panel: Mina shares bread with Mochi", prompt)
        self.assertIn("bottom panel: They walk home together at sunset", prompt)
        self.assertEqual(row["width"], 1024)
        self.assertEqual(row["height"], 1536)

    def test_default_negative_prompt_discourages_layout_failures(self) -> None:
        self.assertIn("broken panel layout", DEFAULT_NEGATIVE_PROMPT)
        self.assertIn("extra panels", DEFAULT_NEGATIVE_PROMPT)
        self.assertIn("unclear panel borders", DEFAULT_NEGATIVE_PROMPT)
        self.assertIn("open book", DEFAULT_NEGATIVE_PROMPT)
        self.assertIn("book spine", DEFAULT_NEGATIVE_PROMPT)
        self.assertIn("center crease", DEFAULT_NEGATIVE_PROMPT)


class BuildPayloadTests(unittest.TestCase):
    def test_page_prompt_row_builds_page_native_payload(self) -> None:
        row = {
            "id": "01_3scene",
            "case_id": "01",
            "num_panels": 3,
            "layout": "vertical",
            "width": 1024,
            "height": 1536,
            "prompt": "three-panel storybook page, stacked vertical panels",
            "negative_prompt": "extra panels",
            "global_lora_path": "artifacts/loras/page",
            "global_lora_weight_name": "pytorch_lora_weights.safetensors",
            "global_lora_scale": 0.4,
            "extra_loras": [{"role": "cast", "path": "artifacts/loras/cast", "weight_name": "cast.safetensors", "scale": 0.8}],
        }

        class Args:
            model = "artifacts/models/Qwen-Image-2512"
            width = 0
            height = 0
            dtype = "bfloat16"
            device = "cuda"
            steps = 28
            guidance = 4.5
            global_lora_path = ""
            global_lora_weight_name = ""
            global_lora_scale = 0.55
            cpu_offload = True

        payload = build_payload(row, Args(), candidate=2, seed=3030, output_path="out.png")
        self.assertEqual(payload["width"], 1024)
        self.assertEqual(payload["height"], 1536)
        self.assertEqual(payload["scene_id"], 3)
        self.assertTrue(payload["cpu_offload"])
        self.assertEqual(payload["style_lora_path"], "artifacts/loras/page")
        self.assertEqual(len(payload["extra_loras"]), 1)


class SplitPanelsTests(unittest.TestCase):
    def test_split_panels_vertical_two_panel(self) -> None:
        image = Image.new("RGB", (10, 20), "white")
        panels = split_panels(image, num_panels=2, layout="vertical")
        self.assertEqual([panel.size for panel in panels], [(10, 10), (10, 10)])

    def test_split_panels_vertical_three_panel(self) -> None:
        image = Image.new("RGB", (12, 30), "white")
        panels = split_panels(image, num_panels=3, layout="vertical")
        self.assertEqual([panel.size for panel in panels], [(12, 10), (12, 10), (12, 10)])

    def test_color_coherence_prefers_matching_panel_colors(self) -> None:
        matching = [
            Image.new("RGB", (8, 8), (120, 100, 80)),
            Image.new("RGB", (8, 8), (122, 102, 82)),
            Image.new("RGB", (8, 8), (121, 101, 81)),
        ]
        drifting = [
            Image.new("RGB", (8, 8), (255, 0, 0)),
            Image.new("RGB", (8, 8), (0, 255, 0)),
            Image.new("RGB", (8, 8), (0, 0, 255)),
        ]
        self.assertGreater(score_color_coherence(matching), score_color_coherence(drifting))

    def test_layout_score_prefers_clear_vertical_gutters(self) -> None:
        clean = Image.new("RGB", (24, 72), "white")
        noisy = Image.new("RGB", (24, 72), (128, 128, 128))
        self.assertGreaterEqual(score_layout(clean, num_panels=3, layout="vertical"), score_layout(noisy, num_panels=3, layout="vertical"))

    def test_perceptual_coherence_falls_back_when_dreamsim_load_fails(self) -> None:
        panels = [
            Image.new("RGB", (8, 8), (120, 100, 80)),
            Image.new("RGB", (8, 8), (122, 102, 82)),
        ]
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_dreamsim = types.SimpleNamespace(
            dreamsim=mock.Mock(side_effect=RuntimeError("network download failed"))
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "dreamsim": fake_dreamsim}):
            score, backend = score_perceptual_coherence(panels)
        self.assertEqual(backend, "histogram")
        self.assertGreaterEqual(score, 0.0)

    def test_perceptual_coherence_raises_when_dreamsim_is_required(self) -> None:
        panels = [
            Image.new("RGB", (8, 8), (120, 100, 80)),
            Image.new("RGB", (8, 8), (122, 102, 82)),
        ]
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_dreamsim = types.SimpleNamespace(
            dreamsim=mock.Mock(side_effect=RuntimeError("network download failed"))
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "dreamsim": fake_dreamsim}):
            with self.assertRaises(RuntimeError):
                score_perceptual_coherence(panels, require_dreamsim=True)

    def test_clipscore_returns_none_when_runtime_fails(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        self.assertIsNone(maybe_clipscore("prompt", image, "openai/clip-vit-base-patch32", enabled=False))

    def test_clipscore_raises_when_required_but_disabled(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        with self.assertRaises(RuntimeError):
            maybe_clipscore(
                "prompt",
                image,
                "openai/clip-vit-base-patch32",
                enabled=False,
                required=True,
            )


class SceneSettingPathTests(unittest.TestCase):
    def test_build_scene_setting_paths_uses_scene_specific_directories(self) -> None:
        paths = build_scene_setting_paths(Path("outputs/page_pipeline"), 3)
        self.assertEqual(paths.prompt_file.as_posix(), "outputs/page_pipeline/prompts/page_prompts_3scene.jsonl")
        self.assertEqual(paths.candidate_dir.as_posix(), "outputs/page_pipeline/candidates/3scene")
        self.assertEqual(paths.export_dir.as_posix(), "outputs/page_pipeline/final/3scene_top1")


if __name__ == "__main__":
    unittest.main()
