# Member A: Story Parsing and Prompt Construction

This module implements the Task A text pipeline:

1. Read `TaskA/*.txt` story files.
2. Parse `[SCENE-n]` blocks and `<Character>` tags.
3. Resolve simple leading pronouns such as `She`, `He`, `It`, and `They`.
4. Save deterministic parsed JSON files.
5. Build one prompt JSON file per story case for the local image generator.

## Data

The Story data is:

```text
data/task_a
```

`data/task_b` contains Virtual Try-On images and is not used by this task.

## Commands

Run from the project root:

```powershell
python scripts/parse_story.py
python scripts/build_prompts.py
python scripts/validate_member_a.py
```

Outputs are written to:

```text
outputs/intermediate/parsed/*.parsed.json
outputs/intermediate/prompts/*.prompts.json
```

## Interface for Member B

Member B can load each `*.prompts.json` file and read:

- `case_id`
- `characters`
- `panel_prompts`
- `panel_prompts[].scene_id`
- `panel_prompts[].prompt`
- `panel_prompts[].negative_prompt`
- `panel_prompts[].seed_offset`

The `seed_offset` is deterministic and can be combined with a base seed in the
generation module.
