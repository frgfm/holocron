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


def test_checkpoint_chart_matches_table():
    repo_root = Path(__file__).parents[1]
    models_page = (repo_root / "docs" / "docs" / "reference" / "models" / "models.md").read_text()
    documented = {
        match["checkpoint"]: (float(match["acc1"]), float(match["params"]))
        for match in re.finditer(
            r"^\| \[`(?P<checkpoint>[^`]+\.IMAGENETTE)`\]\[[^\]]+\] \| "
            r"(?P<acc1>\d+\.\d+)% \| [^|]+ \| (?P<params>\d+(?:\.\d+)?)M \|",
            models_page,
            re.MULTILINE,
        )
    }

    chart = (repo_root / "docs" / "docs" / "img" / "checkpoint-accuracy-vs-parameters.svg").read_text()
    points = [
        match.groupdict()
        for match in re.finditer(
            r'<g class="checkpoint[^"]*" data-checkpoint="(?P<checkpoint>[^"]+)" '
            r'data-acc1="(?P<acc1>[\d.]+)" data-params="(?P<params>[\d.]+)"(?P<flags>[^>]*)>',
            chart,
        )
    ]
    plotted = {
        point["checkpoint"]: (float(point["acc1"]), float(point["params"]))
        for point in points
    }

    assert len(points) == len(plotted) == len(documented) == 27
    assert plotted == documented
    assert [point["checkpoint"] for point in points if 'data-default="true"' in point["flags"]] == [
        "ResNet18_Checkpoint.IMAGENETTE"
    ]

    expected_pareto = {
        checkpoint
        for checkpoint, (acc1, params) in documented.items()
        if not any(
            other_checkpoint != checkpoint
            and other_params <= params
            and other_acc1 >= acc1
            and (other_params < params or other_acc1 > acc1)
            for other_checkpoint, (other_acc1, other_params) in documented.items()
        )
    }
    plotted_pareto = {
        point["checkpoint"] for point in points if 'data-pareto="true"' in point["flags"]
    }
    assert plotted_pareto == expected_pareto
