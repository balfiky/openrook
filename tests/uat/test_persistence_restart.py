from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.uat.conftest import assert_no_browser_errors, expect_json

pytestmark = pytest.mark.uat


def test_session_files_survive_web_server_restart(uat_server, page):
    first = expect_json(
        uat_server.post(
            "/v1/chat",
            {
                "message": "remember that my UAT preference is concise status updates",
                "user_id": "uat_persist",
                "chat_id": "default",
                "include_debug": True,
            },
        )
    )
    assert first["session_key"] == "web:uat_persist:default"

    session_dir = uat_server.data_dir / "web_uat_persist" / "sessions"
    state_file = session_dir / "default.json"
    history_file = session_dir / "default.history.json"
    assert state_file.exists()
    assert history_file.exists()

    uat_server.restart()
    ready = expect_json(uat_server.get("/v1/ready"))
    assert ready["status"] == "ready"

    page.add_init_script(
        "localStorage.setItem('openrook_user_id', 'uat_persist');"
        "sessionStorage.setItem('openrook_chat_id', 'default');"
    )
    page.goto(uat_server.base_url + "/")
    expect(page.locator(".message-row.user .msg-body")).to_have_text(
        "remember that my UAT preference is concise status updates", timeout=15_000,
    )
    expect(page.locator(".message-row.assistant .msg-body")).to_have_count(1)

    second = expect_json(
        uat_server.post(
            "/v1/chat",
            {
                "message": "continue the setup check",
                "user_id": "uat_persist",
                "chat_id": "default",
                "include_debug": True,
            },
        )
    )
    assert second["session_key"] == "web:uat_persist:default"
    assert state_file.exists()
    assert history_file.exists()

    page.reload()
    expect(page.locator(".message-row.user .msg-body")).to_have_count(2, timeout=15_000)
    expect(page.locator(".message-row.user .msg-body").last).to_have_text(
        "continue the setup check",
    )
    expect(page.locator(".message-row.assistant .msg-body")).to_have_count(2)
    assert_no_browser_errors(page)
