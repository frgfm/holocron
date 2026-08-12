---
name: holocron-research-workbench
description: Use Holocron as a trustworthy computer-vision research workbench. Use this skill whenever a user or coding agent needs to discover or compare Holocron models, inspect model maturity or checkpoints, start or resume a reproducible training run, interpret a run bundle, extract classifier features, evaluate detection or segmentation, export a model, benchmark latency, or add a model with evidence-backed support claims.
---

# Holocron research workbench

Use Holocron's public Python contracts as the source of truth. Keep model selection, empirical evidence, and owner policy separate.

## 1. Discover before importing a factory

```python
from holocron.models import get_model, get_model_info, list_checkpoints, list_models

names = list_models(task="classification", pretrained=True)
info = get_model_info(names[0])
checkpoints = list_checkpoints(info.name)
model = get_model(info.name, checkpoint=checkpoints[0] if checkpoints else None)
```

Filter by the user's task and constraints. Report each candidate's maturity and weight availability. Do not silently treat `preview` or `experimental` as validated, and do not invent a universal “best model” score.

## 2. Inspect checkpoint evidence

Use the typed checkpoint metadata for architecture, preprocessing, categories, recipe, evaluation, parameter count, size, and SHA-256. If a required field is absent, say `not reported`; do not fill it from a similarly named model or paper result.

Published paper numbers and Holocron run results are different evidence. Keep dataset, split, resolution, hardware, and metric protocol attached to every comparison.

## 3. Train into a completed run bundle

Pass `run_dir` to `Trainer.fit_n_epochs(...)`. The returned `RunResult` is the in-process result; `manifest.json` is the durable completion marker. Read `metrics.jsonl` and artifact hashes from the bundle rather than scraping console output.

An interrupted bundle without `manifest.json` is incomplete. Preserve it for diagnosis, but do not compare it as a finished experiment.

## 4. Resume exactly when possible

Load checkpoint schema v2 and pass its dictionary to `Trainer.load`. Schema v2 restores model, optimizer, scheduler, scaler, trainer position, recorded configuration, and available RNG state. Keep the original scheduler horizon.

Legacy checkpoints restore model and basic counters only. Call that a warm restart. Custom DataLoader generator state is not serialized; disclose that limit when it affects determinism.

## 5. Use task-appropriate evidence

- Classification: top-1/top-5 on the recorded split and preprocessing.
- Detection: AP50 and AP50:95 from the optional COCO evaluator. A fixed-batch overfit only proves the optimization path moves.
- Segmentation: mean IoU plus the recorded class mapping and ignore-index policy.

CPU tests do not prove CUDA behavior. Export success does not prove eager/export parity unless outputs were compared. Latency numbers need device, dtype, input shape, warmup, synchronization, and distribution—not a single wall-clock mean.

## 6. Reuse researcher interfaces

For classifiers, prefer `forward_features`, `forward_head`, `get_classifier`, and `reset_classifier`. Keep `forward(x)` equivalent to `forward_head(forward_features(x))`.

Use the repository export and latency scripts for machine-readable reports. Do not build a second exporter or benchmark harness inside the caller's project.

## 7. Diagnose failures in order

1. Confirm model name, task, maturity, and checkpoint architecture.
2. Confirm preprocessing, categories, dataset split, and target encoding.
3. Check checkpoint schema and whether `manifest.json` exists.
4. Reproduce on the smallest fixed batch.
5. Separate CPU correctness from CUDA, export, and benchmark gates.
6. Fix the shared root cause and leave one focused regression test.

## Report back

Return:

1. selected model/checkpoint and maturity;
2. exact command or Python call used;
3. completed metrics and artifact paths;
4. gates actually run;
5. unverified or experimental boundaries;
6. next empirical gate.
