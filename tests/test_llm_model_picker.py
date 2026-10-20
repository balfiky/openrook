"""Model choice behavior without a browser or live provider."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available")
@pytest.mark.parametrize("scenario", [
    "multiple_choose_manual",
    "sole_visible",
    "preserve_configured",
    "empty_ollama",
    "error_and_retry",
    "network_failure",
    "plaintext_failure",
    "stale_url_response",
    "manual_while_loading",
    "codex_default",
    "backend_reset",
])
def test_model_picker_choices(scenario):
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("llm_model_picker.mjs")), scenario],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
