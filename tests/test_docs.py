import math
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
            r'data-acc1="(?P<acc1>[\d.]+)" data-params="(?P<params>[\d.]+)"(?P<flags>[^>]*)>'
            r"(?P<body>.*?)</g>",
            chart,
            re.DOTALL,
        )
    ]
    plotted = {point["checkpoint"]: (float(point["acc1"]), float(point["params"])) for point in points}

    assert len(points) == len(plotted) == len(documented) == 27
    assert plotted == documented
    assert [point["checkpoint"] for point in points if 'data-default="true"' in point["flags"]] == [
        "ResNet18_Checkpoint.IMAGENETTE"
    ]

    positions = {}
    for point in points:
        circle = re.search(r'<circle class="point" cx="([\d.]+)" cy="([\d.]+)"', point["body"])
        if circle is not None:
            x, y = map(float, circle.groups())
        else:
            diamond = re.search(r'<path class="default-marker" d="([^"]+)"', point["body"])
            assert diamond is not None
            coordinates = [float(value) for value in re.findall(r"[\d.]+", diamond[1])]
            x = sum(coordinates[::2]) / (len(coordinates) / 2)
            y = sum(coordinates[1::2]) / (len(coordinates) / 2)

        positions[point["checkpoint"]] = (x, y)
        acc1, params = plotted[point["checkpoint"]]
        expected_x = 90 + (math.log10(params) - math.log10(3)) / (math.log10(200) - math.log10(3)) * 750
        expected_y = 570 - (acc1 - 87) / (96 - 87) * 470
        assert math.isclose(x, expected_x, abs_tol=0.11)
        assert math.isclose(y, expected_y, abs_tol=0.11)

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
    plotted_pareto = {point["checkpoint"] for point in points if 'data-pareto="true"' in point["flags"]}
    assert plotted_pareto == expected_pareto

    frontier = re.search(r'<polyline class="frontier-line" points="([^"]+)"', chart)
    assert frontier is not None
    frontier_points = [tuple(map(float, point.split(","))) for point in frontier[1].split()]
    expected_frontier = [
        positions[checkpoint] for checkpoint in sorted(expected_pareto, key=lambda name: documented[name][1])
    ]
    assert frontier_points == expected_frontier
