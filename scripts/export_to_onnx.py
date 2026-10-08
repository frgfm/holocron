# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model ONNX export
"""

import argparse
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnxruntime
import torch

from holocron import models

_RTOL, _ATOL = 1e-3, 3e-5


def _outputs(prediction):
    if isinstance(prediction, torch.Tensor):
        return {"logits": prediction}
    return {f"{key}_{idx}": value for idx, detection in enumerate(prediction) for key, value in detection.items()}


def _detection_order(actual, expected, labels, expected_labels):
    order, owners = [-1] * len(actual), [-1] * len(expected)
    for idx in range(len(actual)):
        # Revisit ambiguous matches rather than consuming a later record's only match.
        pending, parents = [idx], {}
        for candidate in pending:
            distance = ((expected - actual[candidate]).abs() / (_ATOL + _RTOL * expected.abs())).amax(dim=1)
            matches = ((distance <= 1) & (expected_labels == labels[candidate])).nonzero()
            for match in sorted(matches.flatten().tolist(), key=lambda match: owners[match] >= 0):
                if match in parents:
                    continue
                parents[match] = candidate
                if owners[match] < 0:
                    current = match
                    while current >= 0:
                        record = parents[current]
                        previous = order[record]
                        order[record] = current
                        owners[current] = record
                        current = previous
                    break
                pending.append(owners[match])
            else:
                continue
            break
        else:
            raise AssertionError("No matching box/score/label record")
    return torch.tensor(order, dtype=torch.long)


def _match_detections(outputs, reference):
    # Match complete records: tied scores and coordinate round-off can change their order.
    for name, boxes in list(outputs.items()):
        if not name.startswith("boxes_"):
            continue
        if boxes.shape != reference[name].shape:
            raise AssertionError(f"Detection counts differ for {name!r}")
        fields = [key + name[5:] for key in ("boxes", "scores", "labels")]
        expected = torch.cat((reference[name], reference[fields[1]][:, None]), dim=1)
        actual = torch.cat((boxes, outputs[fields[1]][:, None]), dim=1)
        indices = _detection_order(actual, expected, outputs[fields[2]], reference[fields[2]])
        for field in fields:
            reference[field] = reference[field][indices]


@torch.inference_mode()
def export_model(model, images, path):
    """Export a CPU FP32 evaluation model at a fixed input shape and verify runtime parity.

    Raises:
        ValueError: if either runtime produces non-finite outputs
    """
    model.eval()
    samples = (images, torch.zeros_like(images), torch.rand_like(images))
    references = [_outputs(model(sample)) for sample in samples]
    if hasattr(model, "to_deploy"):
        model = model.to_deploy()
    path = Path(path)
    with TemporaryDirectory(dir=path.parent) as directory:
        candidate = Path(directory) / path.name
        torch.onnx.export(
            model,
            images,
            candidate,
            export_params=True,
            opset_version=20,
            dynamo=False,
            input_names=["images"],
            output_names=list(references[0]),
        )
        onnx.checker.check_model(str(candidate))
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = torch.get_num_threads()
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(str(candidate), sess_options=options, providers=["CPUExecutionProvider"])
        for idx, (sample, reference) in enumerate(zip(samples, references, strict=True)):
            outputs = dict(
                zip(reference, map(torch.from_numpy, session.run(None, {"images": sample.numpy()})), strict=True)
            )
            _match_detections(outputs, reference)
            for name, expected in reference.items():
                actual = outputs[name]
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    raise ValueError(f"Non-finite values in output {name!r} on verification input {idx}")
                floating = expected.is_floating_point()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=_RTOL if floating else 0,
                    atol=_ATOL if floating else 0,
                    msg=lambda message, name=name, idx=idx: f"Output {name!r}, verification input {idx}: {message}",
                )
        del session
        candidate.replace(path)


def main(args):
    task = models.get_model_info(args.arch).task
    factory = getattr(getattr(models, task), args.arch)
    kwargs = {} if args.in_channels == 3 else {"in_channels": args.in_channels}
    if "pretrained_backbone" in inspect.signature(factory).parameters:
        kwargs["pretrained_backbone"] = False
    if args.num_classes is not None:
        kwargs["num_classes"] = args.num_classes
    model = models.get_model(args.arch, pretrained=args.pretrained and args.checkpoint is None, **kwargs).eval()
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict.get("model", state_dict), strict=True)
    images = torch.rand((args.batch_size, args.in_channels, args.height, args.width))
    export_model(model, images, args.path)
    print(f"Exported {args.path}; ONNX Runtime CPU outputs match PyTorch on three inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model ONNX export", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", choices=models.list_models(), help="Architecture to use")
    parser.add_argument("--height", type=int, default=224, help="The height of the input image")
    parser.add_argument("--width", type=int, default=224, help="The width of the input image")
    parser.add_argument("--in-channels", type=int, default=3, help="The number of channels of the input image")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size used for the model")
    parser.add_argument("--num-classes", type=int, default=None, help="Override the model's number of output classes")
    parser.add_argument("--path", type=str, default="./model.onnx", help="The path of the output file")
    parser.add_argument("--checkpoint", type=str, default=None, help="The checkpoint to restore")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
