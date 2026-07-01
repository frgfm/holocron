# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Single source of truth for the pretrained model-zoo table.

The table is derived statically (no import of the package, no torch) from the ``Checkpoint`` metadata
declared in ``holocron/models/classification`` — both the ``*_Checkpoint`` enums (new API) and the
legacy ``default_cfgs`` dicts. It is written between the AUTOGEN markers in every target file.

Usage::

    python .github/generate_model_zoo.py           # rewrite the tables in place
    python .github/generate_model_zoo.py --check    # fail (exit 1) if a target is out of date
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "holocron" / "models"
START = "<!-- AUTOGEN:MODEL_ZOO START - edit via .github/generate_model_zoo.py -->"
END = "<!-- AUTOGEN:MODEL_ZOO END -->"
TARGETS = [ROOT / "README.md", ROOT / "docs" / "docs" / "index.md"]

# Dataset enum member -> human label
DATASETS = {"IMAGENETTE": "Imagenette (10)", "IMAGENET1K": "ImageNet-1k (1000)", "CIFAR10": "CIFAR-10 (10)"}
# `_checkpoint()` hardcodes this preprocessing input shape for every checkpoint it builds.
CHECKPOINT_INPUT = "224×224"  # noqa: RUF001 - the multiplication sign is the intended display character


def _lit(node: ast.AST | None):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _from_checkpoint_call(call: ast.Call) -> dict | None:
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    if not _lit(kwargs.get("arch")):
        return None
    dataset = "IMAGENETTE"
    if isinstance(kwargs.get("dataset"), ast.Attribute):
        dataset = kwargs["dataset"].attr  # e.g. Dataset.IMAGENET1K -> "IMAGENET1K"
    acc1 = _lit(kwargs.get("acc1"))
    num_params = _lit(kwargs.get("num_params"))
    return {
        "input": CHECKPOINT_INPUT,
        "dataset": DATASETS.get(dataset, dataset),
        "acc1": f"{acc1 * 100:.1f}" if acc1 is not None else "—",
        "params": f"{num_params / 1e6:.1f}" if num_params is not None else "—",
        "legacy": False,
    }


def _from_legacy_cfg(cfg_dict: ast.Dict) -> dict | None:
    url = None
    shape = None
    preset = None  # detected from `**IMAGENETTE.__dict__` / `**IMAGENET.__dict__` unpacking
    for key, value in zip(cfg_dict.keys, cfg_dict.values, strict=True):
        if key is None:  # dict unpacking, e.g. **IMAGENETTE.__dict__
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                preset = value.value.id
        elif _lit(key) == "url":
            url = _lit(value)
        elif _lit(key) == "input_shape":
            shape = _lit(value)
        elif _lit(key) == "classes" and preset is None:
            classes = _lit(value)
            preset = "IMAGENETTE" if classes and len(classes) == 10 else None
    if not url:
        return None
    return {
        "input": f"{shape[-2]}×{shape[-1]}" if shape else "—",  # noqa: RUF001 - intended display character
        "dataset": {"IMAGENETTE": "Imagenette (10)", "IMAGENET": "ImageNet-1k (1000)"}.get(preset, "—"),
        "acc1": "—",
        "params": "—",
        "legacy": True,
    }


def _checkpoint_entry(classdef: ast.ClassDef) -> tuple[str, dict] | None:
    """Return ``(arch, row)`` for the enum's ``DEFAULT`` checkpoint, or ``None``.

    Resolves the ``DEFAULT = <MEMBER>`` alias so the row reflects what ``pretrained=True`` actually
    loads, rather than assuming the default is the first-declared member.
    """
    members: dict[str, ast.Call] = {}
    default_name: str | None = None
    for stmt in classdef.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        name, value = stmt.targets[0].id, stmt.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "_checkpoint":
            members[name] = value
        elif name == "DEFAULT" and isinstance(value, ast.Name):
            default_name = value.id
    call = members.get(default_name) if default_name else None
    if call is None:  # no DEFAULT alias resolved -> fall back to the first-declared checkpoint
        call = next(iter(members.values()), None)
    if call is None:
        return None
    arch = _lit(next((kw.value for kw in call.keywords if kw.arg == "arch"), None))
    entry = _from_checkpoint_call(call)
    return (arch, entry) if (arch and entry) else None


def _legacy_entries(dict_node: ast.Dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(arch, row)`` for each weighted arch declared in a ``default_cfgs`` dict."""
    for key, value in zip(dict_node.keys, dict_node.values, strict=True):
        arch = _lit(key)
        if isinstance(arch, str) and isinstance(value, ast.Dict):
            entry = _from_legacy_cfg(value)
            if entry:
                yield arch, entry


def _is_default_cfgs(node: ast.AST) -> ast.Dict | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == "default_cfgs" for t in targets):
            return node.value
    return None


def collect_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    # The zoo table lives under the "Image classification" heading; segmentation/detection (a single
    # legacy checkpoint and none, respectively) are described in prose next to it.
    for path in sorted((MODELS_DIR / "classification").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ClassDef) and node.name.endswith("_Checkpoint"):
                entry = _checkpoint_entry(node)
                if entry:
                    rows[entry[0]] = entry[1]  # new-API checkpoints take precedence
            elif (cfgs := _is_default_cfgs(node)) is not None:
                for arch, row in _legacy_entries(cfgs):
                    rows.setdefault(arch, row)
    return rows


def render_table(rows: dict[str, dict]) -> str:
    lines = [
        "| Model | Input | Training dataset | Top-1 acc (%) | Params (M) |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{a}` | {r['input']} | {r['dataset']} | {r['acc1']} | {r['params']} |" for a, r in sorted(rows.items())
    ]
    table = "\n".join(lines)
    if any(r["legacy"] for r in rows.values()):
        table += "\n\n_Rows showing `—` are legacy checkpoints whose accuracy/params are not recorded in metadata._"
    return table


def apply(content: str, table: str, path: Path) -> str:
    if START not in content or END not in content:
        raise SystemExit(f"Missing AUTOGEN markers in {path.relative_to(ROOT)}")
    head, rest = content.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n\n{table}\n\n{END}{tail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any target is out of date")
    args = parser.parse_args()

    table = render_table(collect_rows())
    stale = []
    for path in TARGETS:
        current = path.read_text(encoding="utf-8")
        updated = apply(current, table, path)
        if updated != current:
            stale.append(path.relative_to(ROOT))
            if not args.check:
                path.write_text(updated, encoding="utf-8")

    if args.check and stale:
        names = ", ".join(str(p) for p in stale)
        print(f"Model zoo table is out of date in: {names}\nRun `make model-zoo` to regenerate.", file=sys.stderr)
        return 1
    print(f"Model zoo table {'is up to date' if args.check else 'written'} ({len(collect_rows())} models).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
