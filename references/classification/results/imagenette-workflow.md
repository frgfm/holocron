# Imagenette training and ONNX replay

This comparison checks whether a Holocron model and an ordinary torchvision
module can use the same trainer, export their selected weights to ONNX, and
replay without Holocron or PyTorch. A pass requires both replays to match the
PyTorch top-1 predictions on all 200 validation images without metadata edits.

## Reproduce

Use a revision that includes the completed-trial exporter from
[#617](https://github.com/frgfm/holocron/pull/617). Keep all generated files
outside the repository. Set `DATA` to a new directory outside Git.

```sh
uv sync --locked --extra test --extra quality --extra docs
mkdir -p "$DATA"
curl -fL --retry 2 -o "$DATA/imagenette2-160.tgz" \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
printf '%s  %s\n' \
  64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5 \
  "$DATA/imagenette2-160.tgz" | shasum -a 256 -c -
tar -xzf "$DATA/imagenette2-160.tgz" -C "$DATA"
uv run --no-sync python -m references.classification.prepare_workflow_comparison \
  "$DATA/imagenette2-160" "$DATA/frozen"
OMP_NUM_THREADS=2 uv run --no-sync python -m references.classification.compare_workflow \
  "$DATA/frozen" "$DATA/run" > "$DATA/workflow.log" 2>&1
```

The split script sorts paths and samples with seed 0. It takes 100 images per
class from the archive's training split, then draws 40 per class from its
validation split and divides them equally between validation and test.
It saves membership and file hashes and rejects duplicate contents across
splits. The comparison checks that both trial records match those hashes.

Both models start from random weights and train for three CPU epochs with seed
0, two threads, batch size 20, AdamW with learning rate 0.001 and weight decay
0.01, and cosine decay. The reference transforms use a 128 px training crop and
a 144 px resize followed by a 128 px evaluation crop. Validation loss selects
the weights. Ties keep the earlier epoch. The runner writes both checkpoint
hashes before decoding the test split.

The log records training, export, environment setup and replay commands.
`split-manifest.json`, `selections.json`, the trial records, each bundle's
`bundle.json`, and the raw replay reports preserve the inputs and provenance.
`results.json` records the decision and any failure. The held-out score uses
PyTorch. The ONNX accuracy check uses validation images because the exporter
bundles the trial's validation split.

Replay uses one CPU thread and batch size one. The table reports the median of
the two fresh timing processes' medians and the larger process lifetime peak
RSS. Each process performs 10 warmups and 100 paired measurements. The
image-to-logits timing includes decode and preprocessing. Both latency
measurements exclude model load and disk I/O. Peak RSS includes runtime
startup and model load.

## Measured results

**Continue with the handoff.** Both bundles replayed without metadata edits
and matched 200/200 PyTorch validation predictions. Training used main
`7559b80e` with draft exporter `70a9be57`. Export and replay were repeated
from the same frozen checkpoints on draft `4f3d2cd0`. The ONNX file hashes
were identical across the two revisions. Refresh from merged main once #617
lands.

| Model | Selected epoch | PyTorch test correct | ONNX validation correct | ONNX inference | Image to logits | Replay peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Holocron ResNet-18 | 3 | 85/200 | 81/200 | 14.45 ms | 15.08 ms | 145.1 MiB |
| torchvision MobileNetV3 Small | 1 | 20/200 | 20/200 | 2.34 ms | 2.85 ms | 86.7 MiB |

Measured on an Apple M3 Pro with macOS 15.8.3 and ONNX Runtime 1.31.0.
[Compact results and hashes](imagenette-workflow.json) include training time
and training RSS. This fixed recipe, single seed, and small dataset measure
integration only. Accuracy and runtime results apply to these selected weights
and this machine.

For PyTorch scoring, use batches of four for MobileNet and one for ResNet.
Five measured passes over the same 200 validation images followed one warmup
in each process. MobileNet's median fell from 4.72–10.23 s across two
batch-one processes to 2.41 s. ResNet stays at batch one because batch 20
slowed it down. Every scored top-1 prediction agreed across batch sizes.
Peak RSS for the process that scored both models was 485 MiB with the mixed
batch sizes, versus 474–480 MiB when both used batch one. The ONNX replay
protocol still uses batch one for both models.
