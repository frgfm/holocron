import argparse
import inspect

import onnx
import onnxruntime
import pytest
import torch
from torch import nn

from holocron import models
from holocron.models.detection.yolo import _post_process  # noqa: PLC2701
from scripts.export_to_onnx import export_model, main


@pytest.mark.parametrize("arch", models.list_models())
def test_model_onnx_inference(arch, tmp_path):
    torch.manual_seed(42)
    task = models.get_model_info(arch).task
    factory = getattr(getattr(models, task), arch)
    kwargs = {"num_classes": 3}
    if "pretrained_backbone" in inspect.signature(factory).parameters:
        kwargs["pretrained_backbone"] = False
    side = 448 if arch == "yolov1" else 64
    model = models.get_model(arch, **kwargs).eval()
    # Detector factories can initialize constant heads; exercise input-dependent predictions.
    if task == "detection":
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and not torch.count_nonzero(module.weight):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    # Give class logits a margin instead of rounding to the same confidence.
                    nn.init.normal_(module.bias, std=0.1)
        model.box_score_thresh = 0
    path = tmp_path / "model.onnx"
    images = torch.rand(1, 3, side, side)
    if task == "detection":
        with torch.inference_mode():
            assert model(images)[0]["boxes"].shape[0] > 0
    export_model(model, images, path)
    graph = onnx.load(path, load_external_data=False)
    assert graph.opset_import[0].version == 20
    assert [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim] == [1, 3, side, side]
    path.unlink()


@pytest.mark.parametrize("nms", [False, True])
def test_yolo26_onnx_batch(nms, tmp_path):
    torch.manual_seed(42)
    model = models.get_model("yolo26n", num_classes=3, box_score_thresh=0, nms=nms)
    export_model(model, torch.rand(2, 3, 64, 96), tmp_path / "model.onnx")


def test_detection_onnx_empty_and_nonempty(tmp_path):
    class PostProcess(nn.Module):
        def forward(self, images):  # noqa: PLR6301
            boxes = images.new_tensor([[[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]])
            boxes = boxes.expand(images.shape[0], -1, -1)
            scores = images.new_tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]]).expand(images.shape[0], -1, -1)
            return _post_process(boxes, images[:, 0, 0, :3], scores)

    images = torch.ones(2, 3, 4, 4)
    images[1] = 0
    model = PostProcess()
    assert [len(prediction["boxes"]) for prediction in model(images)] == [2, 0]
    export_model(model, images, tmp_path / "model.onnx")


def test_onnx_verification_catches_traced_input_branch(tmp_path):
    class InputBranch(nn.Module):
        def forward(self, images):  # noqa: PLR6301
            return images + (1 if images.sum() > 0 else 0)

    with pytest.raises(AssertionError, match="not close"):
        export_model(InputBranch(), torch.ones(1, 3, 4, 4), tmp_path / "model.onnx")


@pytest.mark.parametrize("trainer_checkpoint", [False, True])
def test_onnx_checkpoint_cli(trainer_checkpoint, tmp_path):
    model = models.get_model("resnet18", num_classes=3, in_channels=1)
    checkpoint = tmp_path / "model.pth"
    state = model.state_dict()
    torch.save({"model": state, "epoch": 1} if trainer_checkpoint else state, checkpoint)
    main(
        argparse.Namespace(
            arch="resnet18",
            checkpoint=str(checkpoint),
            pretrained=False,
            num_classes=3,
            in_channels=1,
            batch_size=1,
            height=64,
            width=96,
            path=str(tmp_path / "model.onnx"),
        )
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = onnxruntime.InferenceSession(
        str(tmp_path / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    images = torch.rand(1, 1, 64, 96)
    with torch.inference_mode():
        expected = model.eval()(images)
    torch.testing.assert_close(
        torch.from_numpy(session.run(None, {"images": images.numpy()})[0]), expected, rtol=1e-3, atol=1e-5
    )
