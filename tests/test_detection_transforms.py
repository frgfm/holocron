import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from references.detection.transforms import VOCTargetTransform, convert_to_relative


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("tensor_image", [False, True])
def test_paired_resize_flip_preserves_boxes_and_labels(relative, tensor_image):
    annotation = {
        "annotation": {
            "object": [
                {"name": "cat", "bndbox": {"xmin": 2, "ymin": 1, "xmax": 8, "ymax": 5}},
                {"name": "dog", "bndbox": {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 10}},
            ]
        }
    }
    transforms = [VOCTargetTransform(["cat", "dog"]), T.Resize((8, 12)), T.RandomHorizontalFlip(1)]
    if relative:
        transforms.append(convert_to_relative)
    transforms.extend([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.ToPureTensor(),
    ])
    image = Image.new("RGB", (20, 10))
    if tensor_image:
        image = T.ToImage()(image)
    image, target = T.Compose(transforms)(image, annotation)
    expected = torch.tensor([[7.2, 0.8, 10.8, 4], [0, 0, 12, 8]])
    if relative:
        expected /= torch.tensor([12, 8, 12, 8])
    torch.testing.assert_close(target["boxes"], expected)
    assert image.shape == (3, 8, 12)
    assert type(target["boxes"]) is torch.Tensor
    assert target["labels"].tolist() == [0, 1]
    assert target["labels"].dtype == torch.int64


def test_empty_detection_targets_keep_box_shape():
    pipeline = T.Compose([
        VOCTargetTransform(["cat"]),
        T.Resize((8, 12)),
        T.RandomHorizontalFlip(1),
        convert_to_relative,
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ])
    _, target = pipeline(Image.new("RGB", (20, 10)), {"annotation": {"object": []}})
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
