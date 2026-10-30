# Replay a deployment precision sweep

Compare exported classifiers on real devices with identical images, preprocessing
and batch size one. Prepare once, copy the bundle, then measure each device. The
target needs only Python and its inference runtimes; Holocron and PyTorch are not
required. ONNX Runtime and NCNN use one CPU thread; Core ML manages its own threads.

## Prepare a frozen bundle

Export using the existing `scripts/export_to_onnx.py`,
`scripts/export_to_coreml.py` and upstream NCNN/quantization tools. This workflow
freezes their outputs; it does not choose or rebuild conversion recipes.

Create a configuration. Paths resolve from this file:

```json
{
  "schema_version": 1,
  "manifest": "validation.json",
  "accuracy_manifest": "subset.json",
  "reference": "reference-subset.npy",
  "cases": "cases.json",
  "checkpoint": "checkpoint.pth",
  "calibration": "calibration.json",
  "provenance": "conversion-protocol.json"
}
```

Only `schema_version`, `manifest`, `reference` and `cases` are required.
`accuracy_manifest` selects an ordered subset for accuracy; timings always use the
full manifest. Reference logits must follow the accuracy manifest order, with
shape `(samples, classes)`, finite values and no pickle. The checkpoint and
conversion-protocol files are preserved as provenance; their relationship to the
exported weights is supplied by the caller.

The image manifest uses paths relative to itself, exact raw-byte SHA-256 hashes,
and ordered class names. For example (replace the sample hash):

```json
{
  "input_shape": [1, 3, 224, 224],
  "resize": 232,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "categories": ["cat", "dog"],
  "samples": [{"path": "val/cat/1.jpg", "label": 0, "sha256": "..."}]
}
```

Preprocessing is RGB decode, bilinear short-edge resize, rounded center crop,
FP32 multiplication by `1/255`, then normalization. Each bundle freezes its chosen
resolution; the tool does not require 224. An accuracy subset has the same metadata
and a selection of the same sample records. Optional `calibration.json` contains
`{"samples": [...]}` with the same path/label/hash fields. Calibration bytes shared
with validation are rejected.

Define named cases, including the requested precision:

```json
{
  "onnx-fp32": {
    "backend": "onnx", "artifact": "model.onnx",
    "requested_precision": "fp32", "threads": 1
  },
  "ncnn-fp16-arithmetic": {
    "backend": "ncnn", "artifact": "model-fp16.bin", "param": "model.param",
    "requested_precision": "fp16-arithmetic", "threads": 1,
    "use_fp16_packed": true, "use_fp16_storage": true,
    "use_fp16_arithmetic": true, "require_arm_fp16": true
  },
  "coreml-fp16-all": {
    "backend": "coreml", "artifact": "model.mlpackage",
    "requested_precision": "fp16", "compute_units": "ALL"
  }
}
```

Artifact paths resolve from `cases.json`. NCNN options default to false;
`use_bf16_storage` and `use_bf16_packed` are also supported. Core ML expects tensor
input `images` and output `logits`; compute units can be `CPU_ONLY`, `CPU_AND_GPU`,
`CPU_AND_NE` or `ALL`. ONNX/NCNN discover their input/output names.

```sh
python scripts/prepare_deployment.py config.json bundle/
```

Run preparation in the export environment with NumPy and the selected runtimes
installed. The new directory contains copies of all inputs, reference logits,
runner source, provenance, per-runtime pinned requirements and `bundle.json` with
file hashes. Symlinks are dereferenced. Existing destinations are refused; a
failed preparation does not publish a partial bundle. Keep these large bundles
outside Git.

## Run locally or over SSH

Create a runtime-only environment on each target. For example:

```sh
uv venv --python 3.11 runtime
uv pip install --python runtime/bin/python -r bundle/requirements-onnx.txt -r bundle/requirements-ncnn.txt
# On macOS, add -r bundle/requirements-coreml.txt for Core ML cases.
runtime/bin/python bundle/scripts/eval_deployment.py bundle/ results/ --resume
```

Use `--cases NAME ...` or `--backends onnx ncnn` to select candidates. Core ML
prediction on other operating systems and recognized missing operator/CPU support
are recorded as `unsupported`. Broken artifacts, missing packages, version
mismatches and inference errors are `failed`, produce logs and return a nonzero
exit code. Failures do not prevent other candidates from running.

For several machines, create `targets.tsv` with three tab-separated fields:
SSH alias (or `local`), Python executable, and an isolated remote working directory.
Use unique aliases and whitespace-free paths. Remote paths must be absolute.

```text
local	/absolute/runtime/bin/python	unused
dev@devbox	/home/dev/runtime/bin/python	/home/dev/sweep
rimward-edge-01	/home/rimward/runtime/bin/python	/home/rimward/sweep
rimward-edge-02	/home/rimward/runtime/bin/python	/home/rimward/sweep
```

```sh
sh scripts/run_deployment_sweep.sh bundle/ targets.tsv results/
```

The launcher requires SSH, rsync and shasum. It copies changed bytes into a bundle
hash directory, runs hosts in parallel, and collects their reports and worker logs.
It does not install packages or modify services. A host lock prevents concurrent
sweeps under the same OS user. Results live under each host and an identity derived
from the bundle, hardware/OS, installed packages, Python and thread environment.

## Interpret results

`report.json` lists expected cases and marks whether the sweep finished. It
preserves per-case settings, accuracy, precision probes, raw timings,
versions and errors; `summary.csv` compares successful cases and shows unsupported
or failed cases. File hashes and frozen runner source are verified before execution.
Accuracy uses atomic 100-image chunks. `--resume` reuses matching completed accuracy
or chunks; probes and timings always run again.

Each supported case has separate accuracy and two fresh timing processes. Each
timing process performs ten warmups and 100 paired inference-only and in-memory
JPEG-to-logits measurements. Disk reads and model loading are outside the timed
region; `load_seconds` includes runtime imports and model loading. Peak RSS is the
entire timing process lifetime, including Python, runtime
imports, loading and preprocessing; it excludes GPU/Neural Engine memory.

Precision probes retain ONNX operator input types and NCNN weighted-layer output
storage bits. These reveal some fallbacks, such as an FP16 graph executing FP32
convolutions. Storage bits do not prove arithmetic precision. Core ML compute-unit
allowances do not prove placement or per-operation precision. Do not infer compute
precision from an artifact filename.

On devices with the Linux thermal sensor, accuracy pauses at 68°C until 60°C;
timing pauses at 65°C between 20-sample segments. Initial timing cooldown starts at
64°C. A 600-second cooldown deadline fails the case. Waits are excluded from latency
and recorded separately. There is no rewarm after segment cooldown: these are
bounded bursts, not sustained throughput measurements. Existing device workloads
remain active. Frozen inputs and procedures are reproducible; cross-device logits
and exact timings need not be identical. Pin target OS/Python and inspect the
recorded transitive dependency versions when comparing runs.
