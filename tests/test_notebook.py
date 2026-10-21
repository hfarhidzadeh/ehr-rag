"""Tests for the walkthrough notebook builder (``notebooks/build_walkthrough.py``).

The fast tests check the generated notebook's structure without executing it. The slow test executes the
whole notebook (needs the built indexes; about 1-2 minutes).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import nbformat
import pytest

from ehr_rag.config import PROJECT_ROOT

BUILDER = PROJECT_ROOT / "notebooks" / "build_walkthrough.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_walkthrough", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> nbformat.NotebookNode:
    path = _builder().build(tmp_path_factory.mktemp("nb") / "walkthrough.ipynb")
    return nbformat.read(path, as_version=4)


def _sources(nb, kind: str) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == kind]


def test_notebook_is_valid_and_unexecuted(built):
    nbformat.validate(built)
    assert built.metadata["kernelspec"]["name"] == "python3"
    code = [c for c in built.cells if c.cell_type == "code"]
    assert code and all(not c.outputs and c.execution_count is None for c in code)


def test_all_sections_present_in_order(built):
    headings = [m.group(1) for src in _sources(built, "markdown") for m in re.finditer(r"^## (\d+)\.", src, re.M)]
    assert headings == [str(i) for i in range(11)]


def test_every_code_cell_compiles(built):
    for src in _sources(built, "code"):
        compile(src, "<cell>", "exec")


def test_claude_call_is_guarded_by_credentials(built):
    code = "\n".join(_sources(built, "code"))
    # The only cell that can reach the network is behind has_claude_credentials().
    calls = [src for src in _sources(built, "code") if "pipe.generator = claude" in src]
    assert len(calls) == 1 and calls[0].lstrip().startswith("if has_claude_credentials():")
    assert 'load_settings(llm="extractive")' in code
    assert "HF_HUB_OFFLINE" in code


def test_controls_cover_both_directions(built):
    code = "\n".join(_sources(built, "code"))
    kinds = re.findall(r'control\("[^"]+",\s*[^,]+,\s*"(positive|negative)"', code, re.S)
    assert {"positive", "negative"} <= set(kinds)
    assert len(kinds) >= 15


def test_no_real_data_sources(built):
    code = "\n".join(_sources(built, "code")).lower()
    for marker in ("http://", "https://", "mimic", "fhir", "read_csv("):
        assert marker not in code


@pytest.mark.slow
def test_notebook_executes_without_errors(tmp_path):
    from nbclient import NotebookClient

    nb = nbformat.read(_builder().build(tmp_path / "walkthrough.ipynb"), as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}).execute()
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 4
    text = "".join(o.get("text", "") for o in outputs if o.output_type == "stream")
    assert "[FAIL]" not in text and "[PASS]" in text
