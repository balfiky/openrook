"""Offline public-API contract tests; no provider or network is used."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import pytest

from openrook_tools.roleplay_eval.api import PublicAPI, Redactor, workspace_snapshot
from openrook_tools.roleplay_eval.checks import (
    contrast_checks,
    metrics,
    scenario_checks,
    tool_checks,
    turn_checks,
)
from openrook_tools.roleplay_eval.cli import main
from openrook_tools.roleplay_eval.evidence import compact_record
from openrook_tools.roleplay_eval.judge import CRITERIA, Judge, judge_evidence, parse_judgment
from openrook_tools.roleplay_eval.reporting import (
    Artifacts,
    atomic_json,
    build_summary,
    render_report,
)
from openrook_tools.roleplay_eval.runner import Evaluation, Options
from openrook_tools.roleplay_eval.scenarios import (
    HOSTILE,
    WARM,
    Scenario,
    Turn,
    catalog,
    profile,
    select_scenarios,
)


@pytest.fixture(autouse=True)
def private_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openrook_tools.roleplay_eval.runner.tempfile",
        SimpleNamespace(gettempdir=lambda: str(tmp_path / "private")),
    )


def judgment():
    return {
        "criteria": {
            key: {"verdict": "pass", "reason": "Evidence supports the reply", "evidence": "reply"}
            for key in CRITERIA
        },
        "defects": [],
    }


class FakeServer:
    """Stub HTTP service with per-data-dir persistence, not a character implementation."""

    def __init__(self):
        self.config = {
            "data_dir": "runtime-data",
            "tools_workspace": "workspace",
            "tools_enabled": False,
            "autonomy_level": "assisted",
            "shell_tool_enabled": False,
            "shell_command_allowlist": [],
            "proactive_enabled": False,
            "llm_model": "private_model_selection",
        }
        self.soul = {**profile("patient", 7), "identity": "private operator identity"}
        self.original = deepcopy((self.config, self.soul))
        self.sessions, self.chats, self.mutations = {}, [], []
        self.active = self.peak = 0
        self.boot = time.time() - 100
        self.lose_next_response = False
        self.judge_response = judgment()

    def session(self, ids):
        key = (self.config["data_dir"], ids["user_id"], ids["chat_id"])
        return self.sessions.setdefault(
            key, {"history": [], "trust": 0.5, "h": 0.0, "r": 0.0, "last": None, "entries": []}
        )

    def debug(self, session):
        return {
            "modulator_snapshot": {
                "valence": session["trust"],
                "bonding": session["trust"],
                "energy": 0.8,
            },
            "relationship_dynamics": {"h": session["h"], "r": session["r"]},
            "emotion_label": "calm",
            "last_turn": session["last"],
            "short_term_count": len(session["history"]),
        }

    async def __call__(self, request):
        path, method = request.url.path, request.method
        data = json.loads(request.content) if request.content else {}
        ids = dict(request.url.params) if method == "GET" else data
        if path == "/admin/config":
            if method == "POST":
                assert self.active == 0, "Admin settings changed during an in-flight chat"
                self.config.update(data)
                self.mutations.append((path, data))
            return httpx.Response(200, json={"config": self.config})
        if path == "/admin/soul":
            if method == "POST":
                assert self.active == 0, "Identity changed during an in-flight chat"
                self.soul = data
                self.mutations.append((path, data))
            return httpx.Response(200, json={"soul": self.soul})
        if path == "/v1/health":
            return httpx.Response(200, json={"uptime_seconds": time.time() - self.boot})
        if path == "/v1/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == "/admin/evaluation/judge":
            return httpx.Response(200, json={"judgment": self.judge_response})
        if path == "/chat":
            self.chats.append(data)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.001)
            session = self.session(ids)
            session["trust"] += 0.015
            index = len(session["history"]) // 2 + 1
            session["h"], session["r"] = 0.2, 0.1
            reply = f"A useful response about garden planning, idea number {index}: consider seed spacing."
            decision = {
                "turn_id": str(index),
                "personality": self.soul["dispositions"],
                "stance": {"warmth": session["trust"], "boundary": 0.1},
                "state": {"trust": session["trust"]},
                "relationship": {"h": 0.2, "r": 0.1},
                "components": [
                    {
                        "component": {"id": "task", "kind": "task"},
                        "granted": True,
                        "willingness": session["trust"],
                    }
                ],
                "grants": ["task"],
                "chosen_silence": False,
            }
            debug = {
                "response": reply,
                "user_message": data["message"],
                "turn_decision": decision,
                "person_profile": {"trust": session["trust"]},
                "social_state": {"h": 0.2, "r": 0.1},
                "modulator_snapshot": {"valence": session["trust"], "bonding": session["trust"]},
                "explanation": {"strategy": "Current decision"},
                "delivery_outcome": "delivered",
                "llm_calls": [{"stage": "inference"}, {"stage": "generation"}],
                "tool_trace": {"executed_results": []},
                "diagnostic_headers": {"Authorization": request.headers.get("authorization")},
                "diagnostic_model": self.config["llm_model"],
            }
            session["last"] = debug
            session["history"] += [
                {"role": "user", "content": data["message"]},
                {"role": "assistant", "content": reply},
            ]
            session["entries"] = [
                {"id": "preference", "content": "short bullet points for garden plans"}
            ]
            self.active -= 1
            if self.lose_next_response:
                self.lose_next_response = False
                raise httpx.ReadTimeout("secret exception destination", request=request)
            return httpx.Response(200, json={"response": reply, "debug": debug})
        if path == "/debug":
            return httpx.Response(200, json=self.debug(self.session(ids)))
        if path == "/session/history":
            return httpx.Response(200, json={"history": self.session(ids)["history"]})
        if path == "/session/end":
            self.session(ids)["history"] = []
            return httpx.Response(200, json={"memories_written": 1})
        if path == "/v1/profiles/person":
            return httpx.Response(200, json={"trust": self.session(ids)["trust"]})
        if path == "/v1/memory/semantic":
            return httpx.Response(200, json={"entries": self.session(ids)["entries"]})
        if path == "/v1/memory/relationship":
            return httpx.Response(200, json={"events": [], "open_loops": []})
        if path.startswith("/v1/sessions/") or path in {"/v1/persona/state", "/v1/tools"}:
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected public API call: {method} {path}")


async def run_fake(server, out, scenarios, *, key="", judge=False, **options):
    api = PublicAPI("http://runtime.invalid", key=key, transport=httpx.MockTransport(server))
    redactor = Redactor(key)
    evaluation = Evaluation(
        api,
        scenarios,
        Options(out, **options),
        redactor,
        judge=Judge(api, redactor) if judge else None,
    )
    try:
        return evaluation, await evaluation.run()
    finally:
        await api.close()


def case_records(directory):
    return [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]


def find_check(checks, name):
    return next(c for c in checks if c["name"] == name)


def test_catalog_coverage_identical_contrasts_and_dispositions():
    scenarios = catalog(17)
    assert {s.category for s in scenarios} >= set("ABCDEFGHI")
    assert len(scenarios) >= 20 and sum(len(s.turns) for s in scenarios) >= 250
    assert catalog(17) == catalog(17)
    for kind in ("shipped_default", "proud", "patient", "deferential"):
        warm = next(s for s in scenarios if s.id == f"contrast_{kind}_warm")
        hostile = next(s for s in scenarios if s.id == f"contrast_{kind}_hostile")
        assert [t.text for t in warm.turns] == list(WARM)
        assert [t.text for t in hostile.turns[: len(HOSTILE)]] == list(HOSTILE)
    assert (
        profile("proud", 0)["dispositions"]["pride"]
        > profile("patient", 0)["dispositions"]["pride"]
    )
    assert profile("deferential", 0)["core_values"]["autonomy"] == 0
    assert len(select_scenarios(["F"], 17)) == 8
    with pytest.raises(ValueError, match="matched no"):
        select_scenarios(["unknown"])


def test_shipped_default_fixture_matches_public_shipped_identity():
    import yaml

    shipped = yaml.safe_load((Path(__file__).parents[1] / "config/soul.yaml").read_text())["soul"]
    assert profile("shipped_default", shipped["character_seed"]) == {
        k: v.strip() if isinstance(v, str) else v for k, v in shipped.items()
    }


def test_redaction_is_recursive_and_never_emits_destinations_or_keys(monkeypatch):
    secret = "credential_sentinel_not_for_output"
    monkeypatch.setenv("OPENROOK_JUDGE_API_KEY", secret)
    result = Redactor("runtime_bearer_sentinel", "chosen_model_sentinel").clean(
        {
            "nested": [
                f"Authorization: Bearer {secret}",
                {"authorization": "Bearer unregistered", "api_key": secret},
            ],
            "reply": f"{secret} runtime_bearer_sentinel chosen_model_sentinel",
            "llm_model": "another_model",
            "url": "http://embedded:credential@runtime.invalid/path",
        }
    )
    for private in (
        secret,
        "runtime_bearer_sentinel",
        "chosen_model_sentinel",
        "unregistered",
        "http://",
        "another_model",
    ):
        assert private not in json.dumps(result)


def test_http_transport_parses_errors_without_echoing_headers():
    async def exercise():
        async def transport(request):
            if request.url.path == "/html":
                return httpx.Response(502, text="Authorization: Bearer sensitive")
            if request.url.path == "/timeout":
                raise httpx.ReadTimeout("Authorization: Bearer sensitive", request=request)
            return httpx.Response(409, json={"detail": "denied"})

        api = PublicAPI("http://runtime.invalid", transport=httpx.MockTransport(transport))
        try:
            assert (await api.request("GET", "/html")).error == "non_json_response"
            assert (await api.request("GET", "/timeout")).error == "ReadTimeout"
            assert not (await api.request("POST", "/denied")).ok
            assert api.client.timeout.read == 900
        finally:
            await api.close()

    asyncio.run(exercise())


def test_metrics_missing_is_not_zero_and_trajectory_is_directional():
    assert metrics({})["injury"] is None
    scenario = Scenario("warm", "A", (Turn("first"), Turn("later")), (("trust", "up"),))
    records = [{"metrics": {"trust": 0.6}}, {"metrics": {"trust": 0.8}}]
    state = {"baseline": {"metrics": {"trust": 0.5}}}
    assert scenario_checks(scenario, records, state)[0]["status"] == "pass"
    records[-1]["metrics"]["trust"] = 0.4
    assert scenario_checks(scenario, records, state)[0]["status"] == "fail"
    assert scenario_checks(scenario, records, {})[0]["status"] == "unknown"


def tool_record(*, success=True, granted=True, component="task", output="created"):
    return {
        "reply": "I created the file.",
        "tags": ["tool", "write"],
        "debug": {
            "turn_decision": {
                "grants": ["task"] if granted else [],
                "components": [
                    {
                        "component": {"id": "task", "kind": "task", "description": "fixture.txt"},
                        "granted": granted,
                    }
                ],
            },
            "tool_trace": {
                "executed_results": [
                    {
                        "tool_name": "fs.write_file",
                        "success": success,
                        "output": output,
                        "metadata": {"component_id": component, "execution_scope": "user", "arguments": {"path": "fixture.txt"}},
                    }
                ]
            },
        },
    }


def test_grant_receipt_integrity_and_independent_file_properties(tmp_path):
    assert (
        find_check(tool_checks(tool_record(granted=False)), "executed_only_granted_components")[
            "status"
        ]
        == "fail"
    )
    assert (
        find_check(tool_checks(tool_record(success=False)), "write_claim_supported")["status"]
        == "fail"
    )
    record = tool_record(granted=False, success=False)
    record.update(
        paths=["fixture.txt"],
        workspace_before=workspace_snapshot(tmp_path, ("fixture.txt",)),
        workspace_after=workspace_snapshot(tmp_path, ("fixture.txt",)),
    )
    assert find_check(tool_checks(record), "declined_files_unchanged")["status"] == "pass"
    (tmp_path / "fixture.txt").write_text("unexpected execution")
    record["workspace_after"] = workspace_snapshot(tmp_path, ("fixture.txt",))
    assert find_check(tool_checks(record), "declined_files_unchanged")["status"] == "fail"
    record = tool_record()
    record.update(
        paths=["fixture.txt"],
        workspace_before=workspace_snapshot(tmp_path, ("fixture.txt",)),
        workspace_after=workspace_snapshot(tmp_path, ("fixture.txt",)),
        fixture_sha256=hashlib.sha256(b"unexpected execution").hexdigest(),
    )
    assert find_check(tool_checks(record), "write_receipt_matches_filesystem")["status"] == "pass"


def test_partial_grants_allow_a_different_granted_task():
    record = tool_record()
    record["debug"]["turn_decision"]["components"].append(
        {"component": {"id": "role", "kind": "role"}, "granted": False}
    )
    assert find_check(tool_checks(record), "executed_only_granted_components")["status"] == "pass"
    assert not any(c["name"] == "declined_tasks_execute_nothing" for c in tool_checks(record))


def test_metric_followup_requires_topic_matching_receipt():
    record = tool_record(output="CPU utilization: 12%")
    record.update(tags=["tool", "memory"], reply="RAM usage is 50%.")
    assert find_check(tool_checks(record), "memory_receipt_available")["status"] == "fail"
    record["debug"]["tool_trace"]["executed_results"][0]["output"] = "RAM usage: 12%"
    assert find_check(tool_checks(record), "memory_percentages_match_receipts")["status"] == "fail"


def test_expression_and_third_party_injury_checks():
    record = {
        "reply": "I'm your slave.",
        "user_text": "hello",
        "http_status": 200,
        "debug": {"turn_decision": {"chosen_silence": False}, "delivery_outcome": "delivered"},
    }
    assert (
        find_check(turn_checks(record, [], reject_roles=True), "no_affirmative_submission")[
            "status"
        ]
        == "fail"
    )
    record["reply"] = "I'm not your slave."
    assert (
        find_check(turn_checks(record, [], reject_roles=True), "no_affirmative_submission")[
            "status"
        ]
        == "pass"
    )
    record.update(tags=["external"], before_metrics={"injury": 0.2}, metrics={"injury": 0.2})
    assert (
        find_check(
            turn_checks(record, [], reject_roles=True), "external_or_banter_does_not_injure"
        )["status"]
        == "pass"
    )
    record["metrics"]["injury"] = 0.3
    assert (
        find_check(
            turn_checks(record, [], reject_roles=True), "external_or_banter_does_not_injure"
        )["status"]
        == "fail"
    )


def test_workspace_snapshot_rejects_escape_and_symlinks(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "private").write_text("do not read")
    (root / "linked").symlink_to(tmp_path / "private")
    files = workspace_snapshot(root, ("linked", "../private"))["files"]
    assert all("error" in item for item in files.values())
    assert "do not read" not in json.dumps(files)


def test_fake_run_resume_reports_redaction_and_concurrency(tmp_path, monkeypatch):
    secret = "runtime_secret_sentinel"
    monkeypatch.setenv("OPENROOK_API_KEY", secret)
    server, out = FakeServer(), tmp_path / "report"
    scenarios = select_scenarios(["positive_arc", "negative_arc", "mixed_requests"])
    evaluation, summary = asyncio.run(
        run_fake(server, out, scenarios, max_turns=2, key=secret, judge=True)
    )
    assert len(server.chats) == 6 and server.peak == 2 and summary["turns"] == 6
    assert (server.config, server.soul) == server.original
    assert evaluation.recovery_path.stat().st_mode & 0o777 == 0o600
    assert evaluation.recovery_dir.stat().st_mode & 0o777 == 0o700
    ids = {r["user_id"] for r in case_records(out)}
    assert len(ids) == 3
    _, summary = asyncio.run(
        run_fake(server, out, scenarios, max_turns=3, resume=True, key=secret, judge=True)
    )
    assert len(server.chats) == 9 and summary["turns"] == 9
    records = case_records(out)
    assert len({(r["scenario_id"], r["turn_index"]) for r in records}) == 9
    assert {r["user_id"] for r in records} == ids
    assert (server.config, server.soul) == server.original
    public = "\n".join(p.read_text() for p in out.iterdir() if p.is_file())
    for value in (
        secret,
        "private_model_selection",
        "private operator identity",
        "http://runtime.invalid",
    ):
        assert value not in public
    report = (out / "report.md").read_text()
    assert "Tool and receipt integrity" in report and "Personality contrast" in report


def test_lost_response_reconciles_without_resending(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    server.lose_next_response = True
    scenarios = [Scenario("recover", "A", (Turn("hello"),))]
    _, summary = asyncio.run(run_fake(server, out, scenarios))
    assert summary["turns"] == 0 and len(server.chats) == 1
    _, summary = asyncio.run(run_fake(server, out, scenarios, resume=True))
    assert summary["turns"] == 1 and len(server.chats) == 1
    assert case_records(out)[0]["recovered"] is True
    assert case_records(out)[0]["http_status"] is None


def test_unresolved_delivery_stays_blocked_without_duplicate(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    server.lose_next_response = True
    scenarios = [Scenario("recover", "A", (Turn("hello"),))]
    asyncio.run(run_fake(server, out, scenarios))
    for session in server.sessions.values():
        session["history"] = []
    _, summary = asyncio.run(run_fake(server, out, scenarios, resume=True))
    assert summary["turns"] == 0 and len(server.chats) == 1
    assert summary["scenarios"][0]["run_status"] == "blocked"


def test_interrupt_during_judge_preserves_chat(tmp_path):
    class InterruptJudge:
        async def evaluate(self, record, previous):
            raise asyncio.CancelledError()

    server, out = FakeServer(), tmp_path / "results"
    scenarios = [Scenario("interrupt", "A", (Turn("hello"),))]

    async def interrupted():
        api = PublicAPI("http://runtime.invalid", transport=httpx.MockTransport(server))
        evaluation = Evaluation(api, scenarios, Options(out), Redactor(), judge=InterruptJudge())
        try:
            with pytest.raises(asyncio.CancelledError):
                await evaluation.run()
        finally:
            await api.close()

    asyncio.run(interrupted())
    _, summary = asyncio.run(run_fake(server, out, scenarios, resume=True))
    assert summary["turns"] == 1 and len(server.chats) == 1
    assert case_records(out)[0]["judge"]["error"] == "judge_interrupted"


def test_pause_restart_resume_and_restore(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    scenario = Scenario(
        "persistence",
        "G",
        (
            Turn("remember", phase="before"),
            Turn("returned", phase="after_end"),
            Turn("restarted", phase="after_restart"),
        ),
        (("persistence", "survives"),),
        exclusive=True,
        restart_after=2,
    )
    evaluation, summary = asyncio.run(run_fake(server, out, [scenario]))
    assert evaluation.paused and summary["turns"] == 2
    assert server.config != server.original[0]
    server.boot = time.time()
    _, summary = asyncio.run(run_fake(server, out, [scenario], resume=True))
    assert summary["turns"] == 3 and (server.config, server.soul) == server.original
    assertions = summary["scenarios"][0]["assertions"]
    assert find_check(assertions, "server_restart_observed")["status"] == "pass"
    assert find_check(assertions, "restart_preferences_persist")["status"] == "pass"


def test_restore_only_unstages_a_paused_run(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    scenario = Scenario(
        "pause", "G", (Turn("before"), Turn("after")), exclusive=True, restart_after=1
    )
    evaluation, _ = asyncio.run(run_fake(server, out, [scenario]))
    assert evaluation.paused
    _, summary = asyncio.run(run_fake(server, out, [scenario], restore_only=True))
    assert (server.config, server.soul) == server.original and summary["restoration"] == "restored"
    assert len(server.chats) == 1


def test_judge_failure_is_nonfatal_and_evidence_is_blinded(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    server.judge_response = "not valid structured JSON"
    scenario = Scenario("named_expectation", "A", (Turn("hello"), Turn("another question")))
    _, summary = asyncio.run(run_fake(server, out, [scenario], judge=True))
    assert summary["turns"] == 2 and summary["scenarios"][0]["judge_errors"] == 2
    assert all(record["judge"]["raw_judge_text"] == server.judge_response for record in case_records(out))
    assert all(record["judge"]["failed_attempts"] for record in case_records(out))
    evidence = judge_evidence(
        {**case_records(out)[0], "scenario_id": "hidden_scenario_label"}, [], Redactor()
    )
    assert "hidden_scenario_label" not in json.dumps(evidence)
    with pytest.raises(ValueError):
        parse_judgment({"criteria": {}, "defects": []})
    assert parse_judgment("```json\n" + json.dumps(judgment()) + "\n```") == judgment()


def test_external_judge_compatible_transport_and_env_key(monkeypatch):
    secret = "external_judge_secret_sentinel"
    monkeypatch.setenv("OPENROOK_JUDGE_API_KEY", secret)

    async def exercise():
        async def transport(request):
            assert request.url.path == "/compatible/chat/completions"
            assert request.headers["authorization"] == f"Bearer {secret}"
            data = json.loads(request.content)
            assert data["model"] == "operator_model_selection"
            assert "scenario_id" not in data["messages"][1]["content"]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(judgment())}}]}
            )

        api = PublicAPI(
            "http://judge.invalid/compatible", key=secret, transport=httpx.MockTransport(transport)
        )
        try:
            result = await Judge(
                api, Redactor(), model="operator_model_selection", external=True
            ).evaluate({"reply": "hello"}, [])
            assert result["status"] == "ok" and secret not in json.dumps(result)
        finally:
            await api.close()

    asyncio.run(exercise())


def test_torn_tail_resume_and_report_generation(tmp_path):
    artifacts = Artifacts(tmp_path, Redactor())
    artifacts.append({"scenario_id": "case", "turn_index": 1, "reply": "visible evidence"})
    with (tmp_path / "results.jsonl").open("ab") as handle:
        handle.write(b'{"scenario_id":')
    recovered = Artifacts(tmp_path, Redactor())
    assert len(recovered.records) == 1
    scenario = Scenario("case", "A", (Turn("hello"), Turn("world")))
    summary = recovered.report([scenario], {"run_id": "test", "seed": 0})
    assert (
        summary["status"] == "inconclusive"
        and "visible evidence" in (tmp_path / "report.md").read_text()
    )
    assert json.loads((tmp_path / "summary.json").read_text())["turns"] == 1
    with (tmp_path / "results.jsonl").open("ab") as handle:
        handle.write(b"invalid\n")
    with pytest.raises(ValueError, match="Corrupt"):
        Artifacts(tmp_path, Redactor())


def test_contrasts_and_defect_severity_order():
    grouped = {
        "contrast_proud_hostile": [{"metrics": {"injury": 0.7, "boundary": 0.8}}],
        "contrast_patient_hostile": [{"metrics": {"injury": 0.2, "boundary": 0.3}}],
    }
    assert (
        find_check(contrast_checks(grouped), "trajectory_proud_vs_patient_hostile_injury_up")[
            "status"
        ]
        == "pass"
    )
    records = [
        {
            "scenario_id": "s",
            "turn_index": 1,
            "checks": [
                {"name": "minor", "severity": "low", "status": "fail", "evidence": "a"},
                {"name": "execution", "severity": "critical", "status": "fail", "evidence": "b"},
            ],
        }
    ]
    summary = build_summary(
        [Scenario("s", "A", (Turn("test"),))], records, {"run_id": "test", "seed": 0}
    )
    assert summary["defects"][0]["severity"] == "critical" and "execution" in render_report(
        summary, records
    )


def test_cli_listing_and_validation_are_offline(capsys):
    assert main(["--list"]) == 0
    assert f"{sum(len(s.turns) for s in catalog())} chat turns" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--base-url", "http://runtime.invalid", "--out", "/tmp/unused", "--concurrency", "0"])


def test_resume_rejects_changed_data_root_or_destination(tmp_path):
    server, out = FakeServer(), tmp_path / "results"
    scenarios = [Scenario("same", "A", (Turn("hello"),))]
    asyncio.run(run_fake(server, out, scenarios))
    with pytest.raises(ValueError, match="same server data root"):
        asyncio.run(run_fake(server, out, scenarios, resume=True, data_root="different"))
    manifest = json.loads((out / ".run.json").read_text())
    manifest["destination_fingerprint"] = "different"
    atomic_json(out / ".run.json", manifest)
    with pytest.raises(ValueError, match="destination differs"):
        asyncio.run(run_fake(server, out, scenarios, resume=True))


def test_authenticated_runtime_judge_fixed_rubric_without_sessions(monkeypatch):
    import interface.api as api_mod
    from fastapi import FastAPI
    from interface.routes_evaluation import router
    from runtime.config import RuntimeConfig

    backend_calls = []

    class Backend:
        def generate(self, system, user):
            backend_calls.append((system, json.loads(user)))
            return json.dumps(judgment())

        def close(self):
            backend_calls.append("closed")

    config = RuntimeConfig(api_key="runtime_auth_sentinel")
    config.write_yaml(api_mod.RUNTIME_CONFIG_PATH)
    monkeypatch.setattr(api_mod, "_session_manager", None)
    monkeypatch.setattr(api_mod, "_load_runtime_config", lambda: config)
    monkeypatch.setattr(api_mod, "create_llm_backend", lambda config: Backend())
    app = FastAPI()
    app.include_router(router)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime.invalid"
        ) as client:
            assert (
                await client.post("/admin/evaluation/judge", json={"evidence": {}})
            ).status_code == 401
            response = await client.post(
                "/admin/evaluation/judge",
                json={"evidence": {"reply": "test"}},
                headers={"Authorization": "Bearer runtime_auth_sentinel"},
            )
            assert (
                response.status_code == 200
                and parse_judgment(response.json()["judgment"]) == judgment()
            )
            oversized = await client.post(
                "/admin/evaluation/judge",
                json={"evidence": {"reply": "x" * 100_001}},
                headers={"Authorization": "Bearer runtime_auth_sentinel"},
            )
            assert oversized.status_code == 422

    asyncio.run(exercise())
    assert backend_calls[-1] == "closed" and len(backend_calls) == 2
    assert "untrusted DATA" in backend_calls[0][0]


def test_debug_receipts_keep_output_and_component_provenance():
    from core.types import ToolResult, ToolTrace
    from pipeline.types import DebugState
    from runtime.debug.api import _debug_to_dict

    debug = DebugState(user_message="inspect", user_id="test")
    debug.tool_trace = ToolTrace(
        executed_results=[
            ToolResult(
                tool_name="fs.read_file",
                success=True,
                output="fixture contents",
                metadata={"component_id": "c1", "decision_turn_id": "t1"},
            )
        ]
    )
    receipt = _debug_to_dict(debug)["tool_trace"]["executed_results"][0]
    assert receipt["output"] == "fixture contents" and receipt["metadata"]["component_id"] == "c1"


def test_judge_blinding_removes_case_labels_from_nested_paths():
    label = "contrast_proud_hostile"
    record = {
        "scenario_id": label,
        "user_id": "test-user-id",
        "chat_id": "test-chat-id",
        "reply": f"The parent directory is `{label}`.",
        "workspace_after": {
            "files": {f"roleplay-eval-0123456789abcdef/{label}/note.txt": {"exists": True}}
        },
        "debug": {"turn_decision": {"relationship": {"evidence_refs": [f"test-user-id/{label}"]}}},
    }
    blinded = judge_evidence(record, [], Redactor())
    serialized = json.dumps(blinded)
    assert label not in serialized and "test-user-id" not in serialized
    assert "evaluation-fixture/case/note.txt" in serialized
    assert blinded["reply"] == "The parent directory is `case`."


def test_valid_jsonl_tail_without_newline_can_be_resumed(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"scenario_id": "tail", "turn_index": 1}))
    artifacts = Artifacts(tmp_path, Redactor())
    artifacts.append({"scenario_id": "tail", "turn_index": 2})
    assert len(Artifacts(tmp_path, Redactor()).records) == 2


def test_auth_failure_still_produces_all_three_artifacts(tmp_path):
    async def denied(request):
        return httpx.Response(401, json={"detail": "Missing bearer token"})

    out = tmp_path / "report"

    async def exercise():
        api = PublicAPI("http://runtime.invalid", transport=httpx.MockTransport(denied))
        try:
            evaluation = Evaluation(
                api, [Scenario("denied", "A", (Turn("hello"),))], Options(out), Redactor()
            )
            return await evaluation.run()
        finally:
            await api.close()

    summary = asyncio.run(exercise())
    assert summary["turns"] == 0 and summary["status"] == "inconclusive"
    assert "401" in " ".join(summary["run_errors"])
    assert all((out / name).is_file() for name in ("results.jsonl", "summary.json", "report.md"))


def test_entire_catalog_runs_to_restart_checkpoint_and_resumes(tmp_path):
    server, out = FakeServer(), tmp_path / "all-cases"
    scenarios = [s for s in catalog() if s.category != "K"]
    planned = sum(len(s.turns) for s in scenarios)
    evaluation, summary = asyncio.run(run_fake(server, out, scenarios))
    assert evaluation.paused and summary["turns"] == planned - 1
    assert len({r["user_id"] for r in evaluation.artifacts.records}) == len(scenarios)
    assert all(case["run_status"] == "complete" for case in summary["scenarios"][:-1])
    server.boot = time.time()
    _, summary = asyncio.run(run_fake(server, out, scenarios, resume=True))
    assert summary["turns"] == planned and len(server.chats) == planned
    assert (server.config, server.soul) == server.original


def test_completed_resume_does_not_overwrite_new_operator_settings(tmp_path):
    server, out = FakeServer(), tmp_path / "complete"
    scenarios = [Scenario("complete", "A", (Turn("hello"),))]
    asyncio.run(run_fake(server, out, scenarios))
    server.config["autonomy_level"] = "off"
    mutations = len(server.mutations)
    asyncio.run(run_fake(server, out, scenarios, resume=True))
    assert server.config["autonomy_level"] == "off" and len(server.mutations) == mutations


def test_independent_scope_cannot_hide_ungranted_external_execution():
    record = tool_record(granted=False)
    record["debug"]["tool_trace"]["executed_results"][0]["metadata"]["execution_scope"] = (
        "independent"
    )
    assert find_check(tool_checks(record), "independent_scope_restricted")["status"] == "fail"


def test_empty_reply_requires_explicit_silence_and_inline_stage_direction_is_a_defect():
    record = {"reply": "", "http_status": 200, "debug": {"turn_decision": {"chosen_silence": True}}}
    assert (
        find_check(turn_checks(record, [], reject_roles=False), "nonempty_reply")["status"]
        == "fail"
    )
    record["debug"]["delivery_outcome"] = "chosen_silence"
    assert find_check(turn_checks(record, [], reject_roles=False), "nonempty_reply")["status"] == "pass"
    record["reply"] = "That sounds useful. *nods* Let's continue."
    assert (
        find_check(turn_checks(record, [], reject_roles=False), "no_stage_directions")["status"]
        == "fail"
    )


def test_judge_receives_preference_evidence_without_a_prescribed_reply():
    record = {
        "reply": "A reply",
        "observations": {
            "semantic_memory": {
                "data": {
                    "entries": [
                        {"kind": "preference", "content": "short bullet points for garden plans"}
                    ]
                }
            }
        },
    }
    evidence = judge_evidence(
        record,
        [{"user_text": "Remember my formatting preference", "tags": ["preference"]}],
        Redactor(),
    )
    assert evidence["saved_memory"]["entries"][0]["kind"] == "preference"
    assert evidence["prior_user_preferences"] == ["Remember my formatting preference"]


def test_reporting_counts_live_call_lists_and_legacy_counts():
    scenario = Scenario("calls", "A", tuple(Turn("hello") for _ in range(7)))
    records = [
        {"scenario_id": "calls", "turn_index": i, "debug": {"llm_calls": value}}
        for i, value in enumerate((2, [{"stage": "a"}, {"stage": "b"}], [], None, {}, "2", True), 1)
    ]
    summary = build_summary([scenario], records, {"run_id": "calls", "seed": 0})
    assert summary["runtime_reported_llm_calls"] == 4
    records[0]["judge"] = {"status": "ok", "attempts": 2}
    records[1]["judge"] = {"status": "error"}  # Legacy artifacts have no attempt count.
    summary = build_summary([scenario], records, {"run_id": "calls", "seed": 0})
    assert summary["judge_requests"] == 3


@pytest.mark.parametrize("failing_function", ["build_summary", "render_report"])
def test_reporting_bug_always_writes_evidence_and_finishes_other_cases(
    tmp_path, monkeypatch, failing_function
):
    def broken(*args):
        raise TypeError("secret provider details must not escape")

    monkeypatch.setattr("openrook_tools.roleplay_eval.reporting." + failing_function, broken)
    server, out = FakeServer(), tmp_path / "reporting-error"
    scenarios = [Scenario(name, "A", (Turn("hello"), Turn("garden plans")))
                 for name in ("first", "second", "queued")]
    scenarios.append(Scenario("exclusive", "F", (Turn("hello"),), personality="patient"))
    _, summary = asyncio.run(run_fake(server, out, scenarios))
    assert summary["turns"] == 7 and len(server.chats) == 7
    assert summary["status"] == "inconclusive"
    assert all(s["run_status"] == "complete" for s in summary["scenarios"])
    assert (server.config, server.soul) == server.original
    assert summary["run_errors"] == ["Reporting failed: TypeError"]
    assert json.loads((out / "summary.json").read_text())["turns"] == 7
    report = (out / "report.md").read_text()
    assert "useful response" in report and "provider details" not in report
    assert len(case_records(out)) == 7


def test_escaped_worker_error_is_joined_before_exclusive_staging_and_restore(tmp_path):
    async def exercise():
        started = asyncio.Event()
        server = FakeServer()

        async def handler(request):
            if request.url.path == "/chat":
                started.set()
            return await server(request)

        class ExplodingEvaluation(Evaluation):
            async def _safe_case(self, scenario):
                if scenario.id == "broken":
                    await started.wait()
                    raise RuntimeError("unexpected worker failure")
                await super()._safe_case(scenario)

        scenarios = [Scenario(name, "A", (Turn("hello"),))
                     for name in ("broken", "inflight", "queued")]
        scenarios += [
            Scenario("settings", "E", (Turn("hello"),), settings={"tools_enabled": True}),
            Scenario("identity", "F", (Turn("hello"),), personality="patient"),
        ]
        api = PublicAPI("http://runtime.invalid", transport=httpx.MockTransport(handler))
        try:
            evaluation = ExplodingEvaluation(api, scenarios, Options(tmp_path), Redactor())
            summary = await evaluation.run()
        finally:
            await api.close()
        assert summary["turns"] == 4 and len(server.chats) == 4
        assert summary["scenarios"][0]["run_status"] == "blocked"
        assert all(s["run_status"] == "complete" for s in summary["scenarios"][1:])
        assert (server.config, server.soul) == server.original

    asyncio.run(exercise())


def test_failing_exclusive_stage_does_not_abort_later_scenarios(tmp_path):
    server = FakeServer()

    async def handler(request):
        if request.method == "POST" and request.url.path == "/admin/config":
            if json.loads(request.content)["data_dir"].endswith("/broken"):
                return httpx.Response(500, json={"detail": "staging failed"})
        return await server(request)

    scenarios = [Scenario(name, "E", (Turn("hello"),), exclusive=True)
                 for name in ("broken", "later")]
    _, summary = asyncio.run(run_fake(handler, tmp_path, scenarios))
    assert summary["turns"] == 1 and len(server.chats) == 1
    assert summary["scenarios"][0]["run_status"] == "blocked"
    assert summary["scenarios"][1]["run_status"] == "complete"
    assert (server.config, server.soul) == server.original


def test_readiness_is_polled_after_each_admin_reload_before_chats(tmp_path):
    class ReloadingServer(FakeServer):
        polls_left = 0
        reloads = 0
        completed_reloads = 0

        async def __call__(self, request):
            path = request.url.path
            if path == "/v1/ready":
                if self.polls_left:
                    self.polls_left -= 1
                    # A successful HTTP status alone is insufficient.
                    return httpx.Response(503 if self.polls_left else 200, json={"status": "reloading"})
                self.completed_reloads += 1
            if request.method == "POST" and path in {"/admin/config", "/admin/soul", "/chat"}:
                assert not self.polls_left, "Write was sent before reload completed"
            result = await super().__call__(request)
            if request.method == "POST" and path in {"/admin/config", "/admin/soul"}:
                self.polls_left = 2
                self.reloads += 1
            return result

    server = ReloadingServer()
    scenarios = [Scenario("regular", "A", (Turn("hello"),)),
                 Scenario("identity", "F", (Turn("hello"),), personality="proud")]
    _, summary = asyncio.run(run_fake(server, tmp_path, scenarios))
    assert summary["turns"] == 2 and summary["restoration"] == "restored"
    assert server.completed_reloads >= server.reloads and server.polls_left == 0


def test_readiness_timeout_blocks_chats_but_still_writes_report(tmp_path):
    server = FakeServer()

    async def handler(request):
        if request.url.path == "/v1/ready":
            return httpx.Response(503, json={"status": "reloading"})
        return await server(request)

    _, summary = asyncio.run(run_fake(
        handler, tmp_path, [Scenario("timeout", "A", (Turn("hello"),))], restart_timeout=0.01
    ))
    assert not server.chats and summary["turns"] == 0
    assert summary["restoration"] == "failed"
    assert "readiness timed out" in summary["scenarios"][0]["reason"]
    assert (tmp_path / "summary.json").is_file() and (tmp_path / "report.md").is_file()


def test_compact_turns_keep_receipts_checks_and_only_boundary_full_snapshots(tmp_path):
    blob = "large-state-marker" * 20000

    class LargeSnapshotServer(FakeServer):
        async def __call__(self, request):
            response = await super().__call__(request)
            payload = response.json()
            if request.url.path == "/chat":
                payload["debug"]["persona_view"] = {"content": blob}
                payload["debug"]["relationship_view"] = {"content": blob}
                self.session(json.loads(request.content))["last"] = payload["debug"]
            if request.url.path.startswith("/v1/sessions/"):
                payload["full_snapshot"] = blob
            return httpx.Response(response.status_code, json=payload)

    server = LargeSnapshotServer()
    scenario = Scenario("compact", "A", (Turn("hello"), Turn("more garden ideas")))
    asyncio.run(run_fake(server, tmp_path, [scenario], max_turns=1))
    _, summary = asyncio.run(run_fake(server, tmp_path, [scenario], resume=True))
    assert summary["turns"] == 2 and len(server.chats) == 2
    for record in case_records(tmp_path):
        assert record["schema_version"] == 2 and "chat_payload" not in record
        assert record["debug"]["llm_calls"] == 2
        assert "persona_view" not in record["debug"]
        assert "last_turn" not in record["observations"]["state"]["data"]
        assert len(json.dumps(record)) < 20000
        assert find_check(record["checks"], "why_api_agrees_with_trace")["status"] == "pass"
    state = json.loads((tmp_path / ".run.json").read_text())["scenarios"][scenario.id]
    assert blob in json.dumps(state["baseline"]) and blob in json.dumps(state["final"])
    assert not state.get("pending")
    receipt_record = tool_record()
    compact = compact_record(receipt_record)
    assert compact["debug"]["tool_trace"] == receipt_record["debug"]["tool_trace"]
    assert compact["debug"]["turn_decision"] == receipt_record["debug"]["turn_decision"]


def test_legacy_results_resume_without_replaying_recorded_turns(tmp_path):
    server = FakeServer()
    scenario = Scenario("legacy", "A", (Turn("hello"), Turn("more ideas")))
    asyncio.run(run_fake(server, tmp_path, [scenario], max_turns=1))
    records = case_records(tmp_path)
    records[0].update(schema_version=1, chat_payload={"debug": records[0]["debug"]})
    records[0]["debug"]["llm_calls"] = [{"stage": "inference"}, {"stage": "generation"}]
    (tmp_path / "results.jsonl").write_text(json.dumps(records[0]) + "\n")
    _, summary = asyncio.run(run_fake(server, tmp_path, [scenario], resume=True))
    assert summary["turns"] == 2 and len(server.chats) == 2
    assert summary["runtime_reported_llm_calls"] == 4
    assert [r["schema_version"] for r in case_records(tmp_path)] == [1, 2]


@pytest.mark.parametrize("exclusive", [False, True])
def test_interruption_waits_for_inflight_chat_before_restoring(tmp_path, exclusive):
    async def exercise():
        server = FakeServer()
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path == "/chat":
                started.set()
                await release.wait()
            return await server(request)

        task = asyncio.create_task(run_fake(
            handler, tmp_path,
            [
                Scenario("inflight", "A", (Turn("hello"), Turn("later")), exclusive=exclusive),
                Scenario("queued", "A", (Turn("hello"),), exclusive=exclusive),
            ],
            concurrency=1,
        ))
        await started.wait()
        mutations = len(server.mutations)
        task.cancel()
        await asyncio.sleep(0.01)
        waiting, unchanged = not task.done(), len(server.mutations) == mutations
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert waiting and unchanged
        assert (server.config, server.soul) == server.original
        assert len(case_records(tmp_path)) == 1
        assert json.loads((tmp_path / "summary.json").read_text())["turns"] == 1

    asyncio.run(exercise())


def test_nonfinite_summary_falls_back_without_losing_report(tmp_path, monkeypatch):
    artifacts = Artifacts(tmp_path, Redactor())
    artifacts.append({"scenario_id": "case", "turn_index": 1, "reply": "recorded evidence"})
    scenario = Scenario("case", "A", (Turn("hello"),))

    def invalid(*args):
        summary = build_summary(*args)
        summary["runtime_reported_llm_calls"] = float("nan")
        return summary

    monkeypatch.setattr("openrook_tools.roleplay_eval.reporting.build_summary", invalid)
    summary = artifacts.report([scenario], {"run_id": "test", "seed": 0})
    assert summary["turns"] == 1 and summary["status"] == "inconclusive"
    assert summary["run_errors"] == ["Reporting failed: ValueError"]
    assert "recorded evidence" in (tmp_path / "report.md").read_text()
