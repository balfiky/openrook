"""Parse-check every bundled UI module.

A module-scope SyntaxError (e.g. a duplicate function declaration) kills
the entire ES module silently in the browser — v0.30.9 shipped with the
whole settings workspace dead because of one. ``node --check`` parses
each file as an ES module without executing it, so this catches that
class of bug in plain CI with no browser and no LLM.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "interface" / "static"
JS_FILES = sorted(STATIC_DIR.rglob("*.js"))

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available for JS parse checks"
)


def test_static_js_files_found():
    assert len(JS_FILES) >= 8


@pytest.mark.parametrize("js_file", JS_FILES, ids=lambda p: str(p.relative_to(STATIC_DIR)))
def test_module_parses(js_file: pathlib.Path, tmp_path: pathlib.Path):
    # node --check treats .mjs as an ES module; the bundled UI uses bare
    # .js extensions, so check via a temp .mjs copy.
    target = tmp_path / (js_file.stem + ".mjs")
    target.write_bytes(js_file.read_bytes())
    proc = subprocess.run(
        ["node", "--check", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"{js_file.name}: {proc.stderr[-1500:]}"
