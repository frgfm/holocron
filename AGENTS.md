# Holocron contributor guide for coding agents

Holocron is a compact computer-vision research library. Prefer a small change on an existing model, trainer, reference, or utility seam over a new framework.

## Work safely

- Use Python 3.11+ and the built-in typing syntax already used by the project.
- Start with `uv run --python 3.12 --extra test pytest -q <focused tests>`.
- Run `make quality` after focused tests. Run the broader suite only after the focused path is green.
- Keep CPU, CUDA, dataset training, export parity, and benchmark evidence separate. Never report an unrun gate as passed.
- Do not edit generated `llms.txt` files. Update source documentation under `docs/docs/`.

## Public research workflow

- Discover models through `holocron.models.list_models`, `get_model_info`, `list_checkpoints`, and `get_model`; do not index module `__dict__` values.
- Use typed `Checkpoint` metadata and the central loader. Never bypass recorded hashes for published weights.
- Save resumable work with trainer checkpoint schema v2. A model-only or legacy checkpoint is a warm restart, not exact resume.
- Treat `manifest.json` as the completion marker for a run bundle. A directory without it is incomplete.
- Preserve maturity labels: `validated`, `preview`, and `experimental`. Validation requires a published checkpoint, exact provenance, a reproducible recipe, standard task metrics, and a completed run manifest.

## Model changes

A model-family PR should include the factory and exports, focused forward/backward tests, checkpoint metadata when weights exist, standard task metrics, a reference recipe, export parity, and documentation. Without the full evidence set, keep the model preview or experimental.

Reuse `forward_features`, `forward_head`, `get_classifier`, and `reset_classifier` for classifier research access. Avoid family-specific alternatives.

Detection fixes need mathematical oracle tests. A tiny overfit proves learning behavior only; it is not benchmark evidence. Segmentation and detection claims use standard task metrics, not shape tests or custom error alone.

## Scope boundaries

- Prefer the Python API and machine-readable artifacts. Do not add an MCP server, hosted service, queue, or generic agent framework.
- Avoid dependencies for logic already covered by PyTorch, torchvision, or the standard library. The COCO evaluator is the exception: use the optional `pycocotools` integration rather than reimplementing it.
- Keep unrelated work out of the same commit. Parallel agents use separate worktrees and return commits for integration.
