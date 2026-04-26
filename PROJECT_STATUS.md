# DSAA2012 Project Status

## Current Canonical Structure

The root-level directories below are the main working copy:

- `scripts/`: pipeline scripts
- `configs/`: prompt and generation configs
- `data/`: input data
- `docs/`: project notes and specification
- `outputs/`: generated intermediate files, candidate images, logs

## What Is Already Done

### Member A

- `scripts/parse_story.py` parses all Task A story files into deterministic JSON.
- `scripts/build_prompts.py` converts parsed stories into per-panel prompt packages.
- `scripts/validate_member_a.py` validates the parsed and prompt outputs.
- Current validation status: `16` story cases passed.

### Member B

- `scripts/generate_images.py` builds a deterministic generation manifest from prompt JSON files.
- `scripts/run_hpc_generation.py` supports shard-based batch generation with retry and status tracking.
- `scripts/run_local_generation_batch.py` supports one-process local batch generation.
- `scripts/qwen_image_infer.py` provides the Qwen-Image inference entrypoint with pipeline caching.
- Current generation plan: `16` cases x `3` panels x `2` candidates = `96` candidate images.
- Current execution status: the local 4-GPU summary reports `96` selected, `96` succeeded, `0` failed.

### Existing Outputs

- `outputs/intermediate/parsed/`: parsed story JSON for `16` cases
- `outputs/intermediate/prompts/`: prompt JSON for `16` cases
- `outputs/intermediate/generation_manifest.json`: full candidate manifest
- `outputs/candidates/`: generated candidate image files
- `outputs/logs/`: batch logs and summaries

## What Is Not Done Yet

### Member C / Downstream Selection

- A robust `rerank_candidates.py` still needs to be aligned with the new manifest/candidate layout.
- Candidate selection should read from `outputs/intermediate/generation_manifest.json` and/or `outputs/candidates/`.
- The current root `scripts/rerank_candidates.py` is an early prototype and does not yet match the integrated pipeline cleanly.

### Final Packaging

- `package_outputs.py` has not been implemented yet.
- Final selected images still need to be renamed/copied into the official submission format.

### Reproducibility and Environment

- A final `requirements.txt` or environment file is still missing.
- The README still contains placeholder team information and an idealized structure.
- If you want the project to be portable across machines, the model path in `configs/member_b_generation_config*.json` will need to be adjusted from the current local/HPC path.

## Recommended Next Steps

1. Keep the root-level `configs/`, `data/`, `docs/`, `outputs/`, and `scripts/` folders as the only working structure.
2. Update or rewrite `scripts/rerank_candidates.py` so it consumes the integrated manifest and candidate image paths.
3. Implement `scripts/package_outputs.py` for final submission formatting.
4. Add a reproducible environment file and clean README.
5. Adjust the generation config model path if the project needs to run on another machine.
