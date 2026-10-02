import math

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from holocron.models import segmentation
from holocron.trainer import SegmentationTrainer
from holocron.trainer.utils import split_normalization_params
from references.segmentation import train
from references.segmentation import transforms as seg_transforms


class FixedLogits(nn.Module):
    """Predict fixed class scores to verify exact confusion-matrix metrics."""

    def __init__(self, predictions, num_classes):
        super().__init__()
        self.logits = nn.Parameter(10 * nn.functional.one_hot(predictions, num_classes).permute(0, 3, 1, 2).float())

    def forward(self, x):
        """Expand fixed scores to the current batch size.

        Returns:
            Class logits for every image.
        """
        return self.logits.expand(x.shape[0], -1, -1, -1)


def make_trainer(model, targets, criterion=None, batch_size=1):
    loader = DataLoader(
        TensorDataset(torch.zeros(len(targets), 3, *targets.shape[-2:]), targets), batch_size=batch_size
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255) if criterion is None else criterion
    return SegmentationTrainer(
        model,
        loader,
        loader,
        criterion,
        torch.optim.SGD(model.parameters(), lr=0.01),
        num_classes=model.logits.shape[1],
    )


def test_metrics_exclude_absent_classes_but_include_false_positives():
    model = FixedLogits(torch.tensor([[[0, 0], [1, 2]]]), 4)
    learner = make_trainer(model, torch.tensor([[[0, 1], [1, 255]]]))
    metrics = learner.evaluate()
    assert metrics["acc_global"] == pytest.approx(2 / 3)
    assert metrics["mean_iou"] == pytest.approx(0.5)
    # A prediction of an otherwise absent class must contribute a zero IoU.
    learner.val_loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), torch.tensor([[[0, 1], [1, 0]]])))
    metrics = learner.evaluate()
    assert metrics["mean_iou"] == pytest.approx((1 / 3 + 1 / 2) / 3)


def test_metrics_honor_in_range_ignore_index():
    model = FixedLogits(torch.tensor([[[1, 1], [1, 0]]]), 3)
    learner = make_trainer(model, torch.tensor([[[0, 0], [1, 1]]]), nn.CrossEntropyLoss(ignore_index=0))
    assert learner.evaluate()["acc_global"] == pytest.approx(0.5)
    assert learner.evaluate()["mean_iou"] == pytest.approx(0.25)
    assert learner.evaluate(ignore_index=255)["acc_global"] == pytest.approx(0.25)


def test_all_ignored_batches_have_zero_loss_and_gradients():
    learner = make_trainer(FixedLogits(torch.zeros(1, 2, 2, dtype=torch.long), 3), torch.full((1, 2, 2), 255))
    images, targets = next(iter(learner.train_loader))
    loss = learner._get_loss(images, targets)
    assert loss.item() == 0
    loss.backward()
    assert (learner.model.logits.grad == 0).all()
    assert learner.evaluate() == {"val_loss": 0.0, "acc_global": 0.0, "mean_iou": 0.0}


def test_validation_loss_is_weighted_by_batch_size():
    targets = torch.tensor([[[0]], [[0]], [[1]]])
    model = FixedLogits(torch.zeros(1, 1, 1, dtype=torch.long), 3)
    learner = make_trainer(model, targets, batch_size=2)
    expected = nn.functional.cross_entropy(model(torch.zeros(3, 3, 1, 1)), targets).item()
    assert learner.evaluate()["val_loss"] == pytest.approx(expected)


def test_void_batches_do_not_dilute_validation_loss():
    model = FixedLogits(torch.zeros(1, 1, 1, dtype=torch.long), 3)
    learner = make_trainer(model, torch.tensor([[[1]], [[255]]]))
    expected = nn.functional.cross_entropy(model(torch.zeros(1, 3, 1, 1)), torch.ones(1, 1, 1, dtype=torch.long)).item()
    assert learner.evaluate()["val_loss"] == pytest.approx(expected)


def test_empty_validation_fails_clearly():
    learner = make_trainer(
        FixedLogits(torch.zeros(1, 2, 2, dtype=torch.long), 3), torch.empty(0, 2, 2, dtype=torch.long)
    )
    with pytest.raises(ValueError, match="at least one batch"):
        learner.evaluate()


def test_unet3p_normalization_parameter_groups_are_disjoint_and_complete():
    model = segmentation.UNet3p([4, 8, 16])
    norm_parameters, other_parameters = split_normalization_params(model)
    norm_ids = {id(parameter) for parameter in norm_parameters}
    other_ids = {id(parameter) for parameter in other_parameters}
    assert len(norm_ids) == len(norm_parameters)
    assert len(other_ids) == len(other_parameters)
    assert norm_ids.isdisjoint(other_ids)
    assert norm_ids | other_ids == {id(parameter) for parameter in model.parameters()}


def test_torchvision_auxiliary_output_loss_and_gradients():
    class AuxiliaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Conv2d(3, 3, 1)
            self.aux = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return {"out": self.main(x), "aux": self.aux(x)}

    model = AuxiliaryModel()
    learner = SegmentationTrainer(
        model, None, None, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.01)
    )
    images = torch.rand(2, 3, 4, 5)
    targets = torch.randint(3, (2, 4, 5))
    loss, output = learner._get_loss(images, targets, return_logits=True)
    assert torch.allclose(
        loss,
        nn.functional.cross_entropy(model.main(images), targets)
        + 0.5 * nn.functional.cross_entropy(model.aux(images), targets),
    )
    assert output.shape == (2, 3, 4, 5)
    loss.backward()
    assert model.main.weight.grad is not None
    assert model.aux.weight.grad is not None


def test_paired_transforms_keep_mask_labels_and_ignore_padding():
    target = np.zeros((7, 9), dtype=np.uint8)
    target[2:5, 3:7] = 20
    image = np.repeat(target[..., None], 3, axis=-1)
    pipeline = seg_transforms.Compose([
        seg_transforms.RandomCrop(11),
        seg_transforms.RandomHorizontalFlip(1),
        seg_transforms.ToTensor(),
    ])
    images, targets = pipeline(Image.fromarray(image), Image.fromarray(target))
    assert images.shape == (3, 11, 11)
    assert targets.shape == (11, 11)
    assert targets.dtype == torch.long
    assert set(targets.unique().tolist()) == {0, 20, 255}
    valid = targets != 255
    assert torch.equal((images[0] * 255).round().long()[valid], targets[valid])
    resized = seg_transforms.Resize((17, 19))(Image.fromarray(image), Image.fromarray(target))[1]
    assert set(np.unique(np.array(resized))) == {0, 20}


@pytest.mark.parametrize("arch", ["unet", "unetpp", "unet3p"])
@pytest.mark.parametrize("loss", ["crossentropy", "focal", "mc"])
def test_reference_training_and_checkpoint_reload(arch, loss, monkeypatch, tmp_path):
    class TinyVOC(Dataset):
        def __init__(self, _root, image_set, download, transforms):
            self.transforms = transforms
            self.image_set = image_set
            assert download

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            target = np.zeros((39, 43), dtype=np.uint8)
            target[4:25, 8:29] = 20
            target[:2] = 255
            image = np.full((39, 43, 3), 64 + idx, dtype=np.uint8)
            return self.transforms(Image.fromarray(image), Image.fromarray(target))

    monkeypatch.setattr(train, "VOCSegmentation", TinyVOC)
    model_types = {"unet": segmentation.UNet, "unetpp": segmentation.UNetpp, "unet3p": segmentation.UNet3p}

    def build_model(pretrained, num_classes):
        assert not pretrained
        layout = [4, 8, 16] if arch == "unet3p" else [4, 8]
        return model_types[arch](layout, num_classes=num_classes)

    monkeypatch.setitem(train.segmentation.__dict__, arch, build_model)
    checkpoint = tmp_path / "model.pth"
    args = train.get_parser().parse_args(
        f"{tmp_path} --arch {arch} --loss {loss} --img-size 35 -b 2 -j 0 --epochs 1 "
        f"--grad-acc 2 --norm-wd 0 --output-file {checkpoint}".split()
    )
    train.main(args)
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == 1
    assert state["step"] == 2
    assert math.isfinite(state["min_loss"])
    assert state["model"]["classifier.weight"].shape[0] == (63 if loss == "mc" else 21)
    args.resume = str(checkpoint)
    args.test_only = True
    train.main(args)


def test_plotting_small_batch_preserves_training_tensors(monkeypatch):
    monkeypatch.setattr(train.plt, "show", lambda: None)
    images = torch.rand(1, 3, 8, 8)
    targets = torch.full((1, 8, 8), 255)
    originals = images.clone(), targets.clone()
    train.plot_samples(images, targets, ignore_index=255)
    train.plot_predictions(images, torch.rand(1, 3, 8, 8), targets, ignore_index=255)
    assert torch.equal(images, originals[0])
    assert torch.equal(targets, originals[1])
    train.plt.close("all")


def test_local_voc_directory_trains_without_download(monkeypatch, tmp_path):
    voc = tmp_path / "VOCdevkit" / "VOC2012"
    for directory in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
        (voc / directory).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (39, 43), (120, 100, 80)).save(voc / "JPEGImages" / "sample.jpg")
    Image.new("L", (39, 43), 1).save(voc / "SegmentationClass" / "sample.png")
    for split in ("train", "val"):
        (voc / "ImageSets" / "Segmentation" / f"{split}.txt").write_text("sample\n", encoding="utf-8")

    def build_model(pretrained, num_classes):
        assert not pretrained
        return segmentation.UNet([4, 8], num_classes=num_classes)

    monkeypatch.setitem(train.segmentation.__dict__, "unet", build_model)
    checkpoint = tmp_path / "local-voc.pth"
    args = train.get_parser().parse_args(
        f"{tmp_path} --epochs 1 --img-size 35 -b 2 -j 0 --output-file {checkpoint}".split()
    )
    train.main(args)
    assert checkpoint.is_file()
