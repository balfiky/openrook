from __future__ import annotations

import pytest
from playwright.sync_api import expect

from tests.uat.conftest import assert_no_browser_errors, expect_json

pytestmark = pytest.mark.uat


def test_clean_web_chat_exposes_debug_panels(uat_server, page):
    page.goto(uat_server.base_url + "/")
    expect(page.locator("#msgInput")).to_be_visible()
    expect(page.locator("#wizardOverlay")).not_to_have_class("wizard-overlay open")

    page.fill("#msgInput", "hello, can you help me remember concise updates?")
    page.click("#sendBtn")

    assistant_messages = page.locator(".message-row.assistant .msg-body")
    expect(assistant_messages).to_have_count(1, timeout=45_000)
    expect(page.locator("#whyBtn")).to_be_enabled()

    page.click("#whyBtn")
    reply = page.locator(".message-row.assistant").first
    explanation = reply.locator("aside.response-explanation")
    expect(explanation).to_be_visible()
    expect(explanation).to_contain_text("Why this response?")
    expect(assistant_messages).to_have_count(1)
    explanation.get_by_role("button", name="Dismiss explanation").click()
    expect(explanation).to_be_hidden()
    reply.get_by_role("button", name="Why this response?", exact=True).click()
    expect(explanation).to_be_visible()
    expect(reply.locator(".response-explanation")).to_have_count(1)

    reply_text = assistant_messages.first.inner_text()
    page.reload()
    expect(assistant_messages).to_have_count(1, timeout=15_000)
    expect(assistant_messages.first).to_have_text(reply_text)
    expect(page.locator(".response-explanation")).to_have_count(0)
    expect(page.locator("#whyBtn")).to_be_disabled()

    page.goto(uat_server.base_url + "/settings#observability")
    expect(page.locator("#page-observability")).to_be_visible(timeout=15_000)
    expect(page.locator("#personaAdminPage")).to_contain_text("web:", timeout=15_000)
    expect(page.locator("#personaAdminPage")).to_contain_text("Mental And Emotional State")
    expect(page.locator("#personaAdminPage")).to_contain_text("Skills And Tools")

    assert_no_browser_errors(page)


def test_v1_chat_include_debug_has_structural_payload(uat_server):
    response = uat_server.post(
        "/v1/chat",
        {
            "message": "I am worried about my deadline and need practical help.",
            "user_id": "uat_api",
            "chat_id": "default",
            "include_debug": True,
        },
    )
    payload = expect_json(response)
    debug = payload["debug"]

    assert debug["strategy_trace"]["selected"] == debug["response_strategy"]
    assert "explanation" in debug
    assert set(debug["explanation"]) == {
        "interpretation",
        "strategy",
        "state",
        "memory",
        "life_history",
        "tools",
        "limits",
    }
    relationship_view = debug["relationship_view"]
    assert relationship_view["modulators"]["arousal"]["delta"] is None
    assert "life_influence" in relationship_view
    assert "memory_used" in relationship_view
