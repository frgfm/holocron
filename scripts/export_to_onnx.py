# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model ONNX export
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from holocron import models


def load_model(arch: str, *, pretrained: bool = False, checkpoint: str | Path | None = None) -> torch.nn.Module:
    """Create an evaluation model and optionally restore its state.

    Returns:
        The restored model in evaluation mode.

    Raises:
        TypeError: If the architecture or checkpoint has an invalid type.
    """
    try:
        model = models.get_model(arch, pretrained=pretrained and checkpoint is None)
    except ValueError as exc:
        raise TypeError(f"unknown architecture: {arch}") from exc

    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = saved["model"] if isinstance(saved, Mapping) and isinstance(saved.get("model"), Mapping) else saved
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint must be a state dict or contain a 'model' state dict")
        model.load_state_dict(state_dict, strict=True)

    if hasattr(model, "reparametrize"):
        model.reparametrize()
    return model.eval()


def export_model(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    path: str | Path,
    *,
    verify: bool = False,
    report: bool = False,
    artifacts_dir: str | Path | None = None,
) -> torch.onnx.ONNXProgram:
    """Export a model with a dynamic batch dimension.

    Returns:
        The exported ONNX program.

    Raises:
        RuntimeError: If the dynamo exporter does not return a program.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(artifacts_dir) if artifacts_dir is not None else output_path.parent
    if report:
        report_dir.mkdir(parents=True, exist_ok=True)

    program = torch.onnx.export(
        model,
        (img_tensor,),
        output_path,
        export_params=True,
        opset_version=20,
        dynamo=True,
        input_names=("input",),
        output_names=("output",),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
        verify=verify,
        report=report,
        artifacts_dir=report_dir,
        verbose=False,
    )
    if program is None:  # pragma: no cover - dynamo=True always returns a program
        raise RuntimeError("ONNX export did not return a program")
    return program


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    model = load_model(args.arch, pretrained=args.pretrained, checkpoint=args.checkpoint)
    img_tensor = torch.rand((args.batch_size, args.in_channels, args.height, args.width))
    export_model(
        model,
        img_tensor,
        args.path,
        verify=args.verify,
        report=args.report,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model ONNX export", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--height", type=int, default=224, help="The height of the input image")
    parser.add_argument("--width", type=int, default=224, help="The width of the input image")
    parser.add_argument("--in-channels", type=int, default=3, help="The number of channels of the input image")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size used for the model")
    parser.add_argument("--path", type=Path, default=Path("model.onnx"), help="The output ONNX file")
    parser.add_argument("--checkpoint", type=Path, help="A raw state dict or trainer checkpoint to restore")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model-zoo weights")
    parser.add_argument("--verify", action="store_true", help="Verify the export with ONNX Runtime")
    parser.add_argument("--report", action="store_true", help="Write the torch.export conversion report")
    parser.add_argument("--artifacts-dir", type=Path, help="Directory for the conversion report")
    main(parser.parse_args())
