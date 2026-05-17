from __future__ import annotations

import unittest

from PIL import Image

from scripts.qwen_story_prepare_hybrid_lora_data import (
    build_page_caption,
    build_panel_caption,
    parse_page_prompt_text,
    parse_raw_story_blocks,
    split_vertical_panels,
)


class ParseRawStoryBlocksTests(unittest.TestCase):
    def test_parses_two_three_scene_blocks(self) -> None:
        raw_text = """
[SCENE-1] A girl walks into a library.
[SEP]
[SCENE-2] She reaches for a book.
[SEP]
[SCENE-3] She reads by the window.

[SCENE-1] A boy runs through a market.
[SEP]
[SCENE-2] He buys fruit.
[SEP]
[SCENE-3] He walks home smiling.
"""
        self.assertEqual(
            parse_raw_story_blocks(raw_text),
            [
                [
                    "A girl walks into a library.",
                    "She reaches for a book.",
                    "She reads by the window.",
                ],
                [
                    "A boy runs through a market.",
                    "He buys fruit.",
                    "He walks home smiling.",
                ],
            ],
        )

    def test_parses_single_page_prompt_text(self) -> None:
        prompt_text = """
three-panel storybook page, stacked vertical panels, clear gutters
[SCENE-1] She sits by the window in a sunny coffee shop.
[SEP]
[SCENE-2] He walks in holding two cups of latte.
[SEP]
[SCENE-3] They smile and chat over coffee.
"""
        self.assertEqual(
            parse_page_prompt_text(prompt_text),
            [
                "She sits by the window in a sunny coffee shop.",
                "He walks in holding two cups of latte.",
                "They smile and chat over coffee.",
            ],
        )


class SplitVerticalPanelsTests(unittest.TestCase):
    def test_splits_vertical_page_into_three_panels(self) -> None:
        image = Image.new("RGB", (9, 12), "white")
        panels = split_vertical_panels(image, num_panels=3)
        self.assertEqual(len(panels), 3)
        self.assertEqual([panel.size for panel in panels], [(9, 4), (9, 4), (9, 4)])


class CaptionTests(unittest.TestCase):
    def test_page_caption_mentions_three_panel_story(self) -> None:
        scenes = [
            "A woman in a kimono stands under cherry blossoms.",
            "A samurai draws his katana.",
            "She offers him tea as petals fall.",
        ]
        caption = build_page_caption(scenes)
        self.assertIn("three-panel storybook page", caption)
        self.assertIn("same recurring cast across all panels", caption)
        self.assertIn("top panel: A woman in a kimono stands under cherry blossoms.", caption)

    def test_panel_caption_mentions_continuous_story_and_panel_index(self) -> None:
        caption = build_panel_caption(
            "A detective examines a footprint.",
            panel_index=2,
            num_panels=3,
        )
        self.assertIn("panel 2 of a continuous three-panel story", caption)
        self.assertIn("same recurring cast from the source page", caption)
        self.assertIn("A detective examines a footprint.", caption)


if __name__ == "__main__":
    unittest.main()
