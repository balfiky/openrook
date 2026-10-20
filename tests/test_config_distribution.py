"""Public source checkouts ship references, never an operator's live config."""

from fnmatch import fnmatch
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tracked_paths():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("Git tracking checks require a source checkout with Git installed")
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split("\0")) - {""}


def test_live_runtime_config_is_not_tracked(tracked_paths):
    assert "runtime_config.yaml" not in tracked_paths, (
        "runtime_config.yaml contains local settings and credentials; "
        "remove it from Git with git rm --cached runtime_config.yaml"
    )


def test_tracked_runtime_examples_do_not_select_a_model(tracked_paths):
    examples = sorted(
        path for path in tracked_paths
        if fnmatch(Path(path).name, "runtime_config*.example.yaml")
    )
    assert examples, "The source checkout must include runtime config examples"
    for name in examples:
        config = yaml.safe_load((ROOT / name).read_text(encoding="utf-8")) or {}
        model = config.get("llm_model")
        assert model is None or isinstance(model, str) and not model.strip(), (
            f"{name} must leave llm_model empty so users choose from their backend"
        )
