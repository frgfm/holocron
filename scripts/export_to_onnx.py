# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model ONNX export
"""

import argparse
import inspect

import numpy as np
import onnx
import onnxruntime
import torch

from holocron import models


def _outputs(prediction):
    if isinstance(prediction, torch.Tensor):
        return {"logits": prediction}
    return {f"{key}_{idx}": value for idx, detection in enumerate(prediction) for key, value in detection.items()}


def _sort_detections(outputs):
    # Equal scores can reorder detections across runtimes. Compare complete box/score/label records.
    for name, boxes in list(outputs.items()):
        if name.startswith("boxes_"):
            # Round sorting keys only: compare the original values at the tolerance below.
            order = np.lexsort((
                outputs["scores" + name[5:]].numpy().round(5),
                outputs["labels" + name[5:]].numpy(),
                *boxes.numpy().round(5).T[::-1],
            ))
            for key in ("boxes", "scores", "labels"):
                field = key + name[5:]
                outputs[field] = outputs[field][order]
    return outputs


@torch.inference_mode()
def export_model(model, images, path):
    """Export a CPU FP32 evaluation model at a fixed input shape and verify runtime parity.

    Raises:
        ValueError: if either runtime produces non-finite outputs
    """
    model.eval()
    samples = (images, torch.zeros_like(images), torch.rand_like(images))
    references = [_outputs(model(sample)) for sample in samples]
    if hasattr(model, "reparametrize"):
        model.reparametrize()
    elif hasattr(model, "to_deploy"):
        model = model.to_deploy()
    elif hasattr(model, "fuse"):
        model.fuse()
    torch.onnx.export(
        model,
        images,
        path,
        export_params=True,
        opset_version=20,
        dynamo=False,
        input_names=["images"],
        output_names=list(references[0]),
    )
    onnx.checker.check_model(str(path))
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for idx, (sample, reference) in enumerate(zip(samples, references, strict=True)):
        outputs = dict(
            zip(reference, map(torch.from_numpy, session.run(None, {"images": sample.numpy()})), strict=True)
        )
        _sort_detections(outputs)
        _sort_detections(reference)
        for name, expected in reference.items():
            actual = outputs[name]
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise ValueError(f"Non-finite values in output {name!r} on verification input {idx}")
            floating = expected.is_floating_point()
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-3 if floating else 0,
                atol=1e-5 if floating else 0,
                msg=lambda message, name=name, idx=idx: f"Output {name!r}, verification input {idx}: {message}",
            )


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
