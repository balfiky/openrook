"""Keep browser identity and credentials across the product rename."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available")
def test_browser_storage_migrates_without_losing_identity_or_credentials():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("browser_storage.mjs"))],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
