"""Browser response decoding handles plain-text and proxy errors."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available")
def test_readable_http_errors():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("http_responses.mjs"))],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
