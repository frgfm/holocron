import re
from pathlib import Path

from PIL import Image


def test_homepage_quickstart(monkeypatch, tmp_path):
    homepage = (Path(__file__).parents[1] / "docs" / "docs" / "index.md").read_text()
    match = re.search(
        r"<!-- quickstart-example-start -->\s*```python\n(?P<code>.*?)\n```\s*<!-- quickstart-example-end -->",
        homepage,
        re.DOTALL,
    )
    assert match is not None

    image_path = tmp_path / "image.png"
    Image.new("RGB", (320, 240)).save(image_path)

    monkeypatch.setattr("holocron.models.utils.load_pretrained_params", lambda *_args, **_kwargs: None)
    namespace = {"path_to_an_image": image_path}
    exec(compile(match["code"], "docs/docs/index.md", "exec"), namespace)  # noqa: S102

    checkpoint = namespace["checkpoint"]
    probabilities = namespace["probabilities"]
    resize, _, _, normalize = namespace["transform"].transforms
    assert namespace["preprocessing"] is checkpoint.pre_processing
    assert tuple(resize.size) == checkpoint.pre_processing.input_shape[1:]
    assert resize.interpolation == checkpoint.pre_processing.interpolation
    assert tuple(normalize.mean) == checkpoint.pre_processing.mean
    assert tuple(normalize.std) == checkpoint.pre_processing.std
    assert namespace["input_tensor"].shape == (1, *checkpoint.pre_processing.input_shape)
    assert probabilities.shape == (len(checkpoint.meta.categories),)
    assert namespace["label"] in checkpoint.meta.categories
    assert 0 <= namespace["confidence"] <= 1
