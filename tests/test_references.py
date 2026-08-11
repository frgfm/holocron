import importlib.util
import re
import shlex
import sys
from pathlib import Path

import pytest

REFERENCE_ROOT = Path(__file__).parents[1] / "references"
DATA_PATHS = {
    "classification": "imagenette2-320/",
    "detection": "VOC2012",
    "segmentation": "VOC2012",
}


def _get_parser(task: str):
    script = REFERENCE_ROOT / task / "train.py"
    spec = importlib.util.spec_from_file_location(f"holocron_reference_{task}", script)
    assert spec is not None
    assert spec.loader is not None

    sys.path.insert(0, str(script.parent))
    try:
        sys.modules.pop("transforms", None)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("transforms", None)

    return module.get_parser()


@pytest.mark.parametrize(
    ("task", "expected_arch", "expected_size"),
    [
        ("classification", "darknet19", None),
        ("detection", "yolov2", 416),
        ("segmentation", "unet3p", 256),
    ],
)
def test_reference_parser_defaults(task: str, expected_arch: str, expected_size: int | None):
    args = _get_parser(task).parse_args([DATA_PATHS[task]])

    assert args.arch == expected_arch
    assert args.seed == 0
    if expected_size is not None:
        assert args.img_size == expected_size


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_reference_parser_reproducibility_options(task: str):
    args = _get_parser(task).parse_args([DATA_PATHS[task], "--seed", "7", "--img-size", "128"])

    assert args.seed == 7
    assert args.img_size == 128


@pytest.mark.parametrize("task", DATA_PATHS)
def test_documented_reference_commands_parse(task: str):
    readme = (REFERENCE_ROOT / task / "README.md").read_text()
    data_path = DATA_PATHS[task]
    commands = re.findall(rf"(?:python train\.py )?({re.escape(data_path)} --arch [^|`\n]+)", readme)

    assert commands
    parser = _get_parser(task)
    for command in commands:
        parser.parse_args(shlex.split(command))
