import pytest
import torch

from holocron.models import recognition
from holocron.utils import CTCCodec, prefix_beam_decode


def test_public_backbone_transfer_and_length_aware_ctc_backward():
    torch.manual_seed(8)
    classifier = recognition.CharacterClassifier(num_classes=3)
    reader = recognition.CTCRecognizer(num_classes=3)
    reader.backbone.load_state_dict(classifier.backbone.state_dict())
    for key, value in classifier.backbone.state_dict().items():
        assert torch.equal(value, reader.backbone.state_dict()[key])
    images = torch.rand(3, 1, 32, 40) * 2 - 1
    lengths = torch.tensor([10, 6, 8])
    assert classifier(images, lengths).shape == (3, 3)
    assert reader.backbone(images).shape == (3, 96, 4, 10)
    outputs = reader(images, lengths)
    assert outputs.shape == (10, 3, 4)
    assert torch.allclose(outputs.exp().sum(-1), torch.ones(10, 3))
    codec = CTCCodec("AB ")
    targets = torch.cat([codec.encode(text) for text in ("AA", "AB", "B A")])
    loss = torch.nn.functional.ctc_loss(outputs, targets, lengths, torch.tensor([2, 2, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert reader.backbone.features[0].weight.grad.abs().sum() > 0
    assert reader.context.weight_ih_l0.grad.abs().sum() > 0


@pytest.mark.parametrize("model", [recognition.CharacterClassifier, recognition.CTCRecognizer])
def test_public_recognition_model_state_dict_roundtrip_and_class_count(model):
    first, restored = model(num_classes=4).eval(), model(num_classes=4).eval()
    restored.load_state_dict(first.state_dict())
    images, lengths = torch.zeros(2, 1, 32, 16), torch.tensor([4, 3])
    with torch.no_grad():
        assert torch.equal(first(images, lengths), restored(images, lengths))
    with pytest.raises(ValueError, match="positive"):
        model(num_classes=0)


@pytest.mark.parametrize("shape", [(2, 3, 32, 16), (2, 1, 48, 16), (2, 1, 32, 15), (0, 1, 32, 16)])
def test_public_recognition_rejects_incompatible_images(shape):
    with pytest.raises(ValueError):
        recognition.CharacterBackbone()(torch.zeros(shape))


@pytest.mark.parametrize(
    "lengths", [torch.tensor([0, 4]), torch.tensor([4, 5]), torch.tensor([4]), torch.tensor([4.0, 3.0])]
)
def test_public_recognition_rejects_incompatible_lengths(lengths):
    with pytest.raises(ValueError, match="lengths"):
        recognition.CTCRecognizer(num_classes=3)(torch.zeros(2, 1, 32, 16), lengths)


def test_public_unicode_codec_and_decoder_use_the_same_alphabet():
    codec = CTCCodec("éA .")
    assert codec.encode("éé A.").tolist() == [1, 1, 3, 2, 4]
    assert codec.decode([1, 1, 0, 1, 3, 2, 4, 0]) == "éé A."
    probabilities = torch.full((8, 5), -20.0)
    probabilities[torch.arange(8), torch.tensor([1, 1, 0, 1, 3, 2, 4, 0])] = 0
    assert prefix_beam_decode(probabilities.numpy(), codec) == "éé A."
    assert codec.encode("").shape == (0,)
    assert not prefix_beam_decode(probabilities[:0].numpy(), codec)
    with pytest.raises(ValueError, match="alphabet"):
        codec.encode("?")
    with pytest.raises(ValueError, match="indices"):
        codec.decode([5])
    with pytest.raises(ValueError, match="shape"):
        prefix_beam_decode(probabilities[:, :4].numpy(), codec)
