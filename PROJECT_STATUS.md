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

- `outputs/runs/run_000x_<style_id>/intermediate/parsed/`: parsed story JSON for `16` cases
- `outputs/runs/run_000x_<style_id>/intermediate/prompts/`: style-specific prompt JSON
- `outputs/runs/run_000x_<style_id>/intermediate/generation_manifest.json`: full run manifest
- `outputs/runs/run_000x_<style_id>/intermediate/selection_results.json`: final per-scene selection results
- `outputs/runs/run_000x_<style_id>/final/`: packaged final images ready for submission review
- `outputs/runs/run_000x_<style_id>/metadata/run_metadata.json`: this run's experiment parameters and config snapshot
- `outputs/legacy/storybook_seed/`: archived legacy candidate seed data for default storybook runs

### Member C

- `scripts/rerank_candidates.py` now reads the integrated manifest and candidate image paths.
- `scripts/package_outputs.py` packages the selected final images.
- `scripts/run_story_pipeline.py` now creates numbered run folders and records experiment metadata.
- A style preset system now supports `storybook`, `watercolor`, `anime`, and `paper_cutout`.
- Current selection status: `run_0001_storybook` and `run_0002_watercolor` both produced `16` cases and `48` packaged final panel images.

## What Is Still Left

### Submission Alignment

- The exact official submission naming/layout still needs to be double-checked against the course spec.
- If the teacher requires a flatter directory structure or different filenames, `scripts/package_outputs.py` should be adjusted.

### Reproducibility and Environment

- `requirements.txt` is now present, but version pinning may still be needed on the final machine.
- The README still contains placeholder team information and an idealized structure.
- If you want the project to be portable across machines, the model path and optional IP-Adapter paths in `configs/member_b_generation_config*.json` will need to be adjusted from the current local/HPC path.

### Report and Polish

- Fill in the final team information in `README.md`.
- Add report screenshots, qualitative examples, and method discussion.
- Optionally upgrade reranking later with CLIP if the final runtime environment includes the required dependencies.
- If you want a true reference-image demo, provide real `reference_image_path` values and valid IP-Adapter weights.

## Recommended Next Steps

1. Keep the root-level `configs/`, `data/`, `docs/`, `outputs/`, and `scripts/` folders as the only working structure.
2. Confirm the teacher's final submission directory and filename rules.
3. Adjust `scripts/package_outputs.py` if the official format differs from the current `outputs/runs/run_000x_<style_id>/final/{case_id}/scene_{scene_id}.png` layout.
4. Fill in team metadata and polish the report.
5. Adjust the generation config model path if the project needs to run on another machine.
