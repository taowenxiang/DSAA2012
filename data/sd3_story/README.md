# SD3 Story Data

This directory stores lightweight, reproducible inputs for the SD3 storyboard
workflow.

- `validation_prompts/`: tracked JSONL prompts generated from `data/task_a`.
- `train_storyboard/`: generated LoRA training images and captions, ignored by Git.
- `train_character/`: generated Character LoRA seed images and captions, ignored by Git.
- `human_eval/`: generated human evaluation CSV sheets, ignored by Git.

Regenerate prompts:

```bash
python scripts/sd3_build_story_prompts.py --input data/task_a --out-dir data/sd3_story/validation_prompts
```

Regenerate local storyboard training data:

```bash
python scripts/sd3_prepare_storyboard_data.py --sources local --out-dir data/sd3_story/train_storyboard --resolution 768
```
