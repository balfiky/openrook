"""Exercise browser chat state and feedback without a live model or browser."""

from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available")


@pytest.mark.parametrize("scenario", [
    "session_saved",
    "session_saved_one",
    "session_no_new_memory",
    "session_failure",
    "session_network_failure",
    "explanation_card",
    "plain_strategy_labels",
    "history_restore",
    "history_restore_failure",
    "history_auth_restore",
    "history_settings_race",
    "chat_pending",
    "end_pending",
    "rest_pending",
    "slow_reply",
    "fast_reply",
    "backend_failure",
    "network_failure",
    "plaintext_failure",
    "connection_result",
    "connection_summary",
    "theme_tokens",
    "wizard_model_choice",
    "wizard_preserves_independence_setting",
    "wizard_presets_no_model",
])
def test_chat_feedback_and_actions(scenario):
    harness = Path(__file__).with_name("chat_ux.mjs")
    result = subprocess.run(
        ["node", str(harness), scenario],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
