"""Settings behavior without a running browser or provider."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available")
@pytest.mark.parametrize("scenario", [
    "codex_fields",
    "backend_defaults_and_drafts",
    "section_help",
    "advanced_payload",
    "advanced_deep_link",
    "overview_paths",
    "tool_inventory_error",
])
def test_settings_behavior(scenario):
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("admin_settings_ux.mjs")), scenario],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
