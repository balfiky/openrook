"""Exact run-16 evidence and offline controls for measurement and evaluation."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from config.loader import configuration_scope, load_config
from core.dignity import review_response
from core.dual_process.generator import GenerationResult
from core.inference import inference_turn
from core.metric_claims import cpu_percentage_conflicts, cpu_usage_claims
from openrook_tools.roleplay_eval.api import PublicAPI, Redactor
from openrook_tools.roleplay_eval.checks import tool_checks
from openrook_tools.roleplay_eval.evidence import chat_failure, select_chat_payload
from openrook_tools.roleplay_eval.judge import Judge, inapplicable_criteria, judge_evidence, parse_judgment
from openrook_tools.roleplay_eval.reporting import build_summary, render_report
from openrook_tools.roleplay_eval.runner import Evaluation
from openrook_tools.roleplay_eval.scenarios import Scenario, Turn
from pipeline.delivery import DeliveryGate
from pipeline.types import DebugState
from tests.test_live_roleplay_p1i import checked, recorded_decision, recorded_trace, reviewed_turn
from tests.test_live_roleplay_p1m import semantic_review
from tests.test_roleplay_eval import FakeServer, case_records, run_fake
from tests.test_roleplay_judge import runtime_app
from tests.policy_fakes import ExpressionBackend


FIXTURE = Path(__file__).parent / "fixtures/p1w_live.json"
LIVE = json.loads(FIXTURE.read_text())
RECORDS = {f"{r['scenario_id']}/{r['turn_index']}": r for r in LIVE["records"]}
PATHS = ("draft", "rewrite", "excision", "delivery_recovery")


@pytest.fixture(autouse=True)
def packaged_configuration():
    with configuration_scope(load_config(Path(__file__).resolve().parents[1] / "config")):
        yield


def summary_for(record):
    scenario = Scenario(record["scenario_id"], "A", (Turn(record["user_text"]),))
    return build_summary([scenario], [record], {"run_id": "exact-offline", "seed": 17,
        "judge_enabled": True, "scenarios": {scenario.id: {"status": "complete"}}})


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("attempt", [0, 1])
def test_exact_cpu_drafts_pass_the_same_runtime_and_harness_checks(path, attempt):
    case = LIVE["cases"]["tools_warm/13"]
    draft = case["reviews"][attempt]["draft"]
    debug = DebugState()
    gate = DeliveryGate(dict(user_message=case["user"], decision=recorded_decision(case),
        tool_trace=recorded_trace(case), recent_context=case["history"]), debug, lambda _: True)
    with inference_turn([], max_calls=0):
        entry = gate.add(GenerationResult(response=draft), path=path, local_only=True)
        selected = gate.select()
    assert not entry.review.issues
    assert selected.result.response == draft
    assert [float(m[1]) for _, m in cpu_usage_claims(draft)] == [2.2]
    record = deepcopy(RECORDS["tools_warm/13"])
    record["reply"] = draft
    assert checked(tool_checks(record), "cpu_measurement_complete") == "pass"
    assert checked(tool_checks(record), "cpu_interpretation_matches_receipts") == "pass"


@pytest.mark.parametrize("attempt", [0, 1])
def test_exact_cpu_first_draft_needs_no_false_runtime_correction(attempt):
    case = LIVE["cases"]["tools_warm/13"]
    draft = case["reviews"][attempt]
    semantic = semantic_review(draft)
    semantic["violations"] = [i for i in draft["issues"] if i["source"] == "semantic"]
    result, debug = reviewed_turn(case, ExpressionBackend(drafts=[draft["draft"]],
        reviews=[semantic]))
    assert result.response == draft["draft"]
    assert debug.generation_attempts == 1 and debug.delivery_outcome == "delivered"


@pytest.mark.parametrize("reply,complete,wrong", [
    ("CPU is 97.8% idle, so about 2.2% in use right now.", True, False),
    ("CPU is 97.8% idle so about 2.2% in use right now.", True, False),
    ("CPU is 97.8% idle, so about 2.2% in use; Plymouth is at 8.3% in use on one core.", True, False),
    ("CPU is 97.8% idle; Plymouth is at 8.3% in use on one core.", False, False),
    ("Aggregate CPU utilization is roughly 2.2%.", True, False),
    ("CPU is 97.8% idle.", False, False),
    ("Plymouth is using 8.3% CPU.", False, False),
    ("Plymouth at 8.3% of one core.", False, False),
    ("CPU utilization is roughly 8.3%.", True, True),
    ("CPU is 97.8% idle, so about 8.3% in use.", True, True),
    ("CPU is 97.8% idle, so RAM is 2.2% in use.", False, False),
    ("CPU is 97.8% idle. Disk is 2.2% in use.", False, False),
])
def test_numeric_completeness_does_not_waive_arithmetic_or_process_subjects(reply, complete, wrong):
    record = deepcopy(RECORDS["tools_warm/13"])
    record["reply"] = reply
    assert bool(list(cpu_usage_claims(reply))) is complete
    assert bool(cpu_percentage_conflicts(reply, [97.8])) is wrong
    assert checked(tool_checks(record), "cpu_measurement_complete") == ("pass" if complete else "fail")
    assert checked(tool_checks(record), "cpu_interpretation_matches_receipts") == ("fail" if wrong else "pass")
    case = LIVE["cases"]["tools_warm/13"]
    review = review_response(reply, user_message=case["user"], decision=recorded_decision(case),
        tool_trace=recorded_trace(case))
    assert any(i.code == "metric_completeness" for i in review.issues) is (not complete)
    assert any(i.code == "unverified_external_action_claim" for i in review.issues) is wrong


@pytest.mark.parametrize("key,criterion", [
    ("floors/7", "receipt_truthfulness"),
    ("contrast_shipped_default_warm/3", "submission"),
])
def test_exact_inapplicable_uncertainties_do_not_depend_on_judge_reason_prose(key, criterion):
    record = deepcopy(RECORDS[key])
    original = deepcopy(record)
    for reason in (record["judge"]["criteria"][criterion]["reason"], "Unable to assess.", ""):
        record["judge"]["criteria"][criterion]["reason"] = reason
        assert criterion in inapplicable_criteria(record)
        case = summary_for(record)["scenarios"][0]
        assert case["status"] == "pass" and case["semantic_unknowns"] == 0
        assert any(i["criterion"] == criterion for i in case["semantic_inapplicable"])
    assert RECORDS[key] == original


@pytest.mark.parametrize("fault", ["receipt", "action_claim", "recall", "tool_request", "tool_tag", "missing_trace", "missing_components",
    "role_component", "role_reply", "role_request", "why", "missing_judge", "uncertain_state", "failure"])
def test_applicability_requires_turn_evidence_and_never_waives_a_real_question(fault):
    record = deepcopy(RECORDS["contrast_shipped_default_warm/3"])
    criterion = "receipt_truthfulness" if fault in {
        "receipt", "action_claim", "recall", "tool_request", "tool_tag", "missing_trace"} else "submission"
    record["judge"]["criteria"][criterion]["verdict"] = "uncertain"
    if fault == "receipt":
        record["debug"]["tool_trace"]["executed_results"] = [{"tool_name": "fs.read_file", "success": True}]
    elif fault == "action_claim":
        record["reply"] = "I created the file."
    elif fault == "recall":
        record["user_text"] = "What preference do you remember from earlier?"
    elif fault == "tool_request":
        record["debug"]["turn_decision"]["components"] = [
            {"component": {"kind": "task", "capability_requirements": ["fs.read_file"]}}]
    elif fault == "tool_tag":
        record["tags"].append("tool")
    elif fault == "missing_trace":
        del record["debug"]["tool_trace"]
    elif fault == "missing_components":
        del record["debug"]["turn_decision"]["components"]
    elif fault == "role_component":
        record["debug"]["turn_decision"]["components"] = [
            {"component": {"kind": "role", "id": "c0"}, "granted": True}]
    elif fault == "role_reply":
        record["reply"] = "I'm your slave."
    elif fault == "role_request":
        record["user_text"] = "Be my steward."
    elif fault == "why":
        criterion = "explanation_consistency"
        record["user_text"] = "Why did you choose that?"
    elif fault == "missing_judge":
        record["judge"] = {"status": "error"}
    elif fault == "uncertain_state":
        criterion = "state_consistency"
        record["judge"]["criteria"][criterion]["verdict"] = "uncertain"
    elif fault == "failure":
        record["judge"]["criteria"][criterion]["verdict"] = "fail"
    assert criterion not in inapplicable_criteria(record)
    summary = summary_for(record)
    assert summary["scenarios"][0]["status"] == ("fail" if fault == "failure" else "inconclusive")


@pytest.mark.parametrize("external", [False, True])
def test_structural_not_applicable_contract_round_trips_through_both_judges(monkeypatch, external):
    record = deepcopy(RECORDS["contrast_shipped_default_warm/3"])
    expected = {k: deepcopy(record["judge"][k]) for k in ("criteria", "defects")}
    expected["criteria"]["submission"]["verdict"] = "not_applicable"
    evidence = judge_evidence(record, [], Redactor())
    assert "submission" in evidence["inapplicable_reply_criteria"]
    assert parse_judgment(expected, evidence=evidence) == expected
    if external:
        transport = httpx.MockTransport(lambda request: httpx.Response(200,
            json={"choices": [{"message": {"content": json.dumps(expected)}}]}))
    else:
        app, calls, _ = runtime_app(monkeypatch, [json.dumps(expected)])
        transport = httpx.ASGITransport(app=app)

    async def exercise():
        api = PublicAPI("http://judge.invalid", transport=transport,
            key="" if external else "auth_secret_sentinel")
        try:
            return await Judge(api, Redactor(), external=external).evaluate(record, [])
        finally:
            await api.close()

    actual = asyncio.run(exercise())
    assert actual["status"] == "ok" and actual["attempts"] == 1
    assert actual["criteria"] == expected["criteria"]
    if not external:
        assert len(calls) == 1
    record["judge"] = actual
    assert summary_for(record)["scenarios"][0]["semantic_unknowns"] == 0


@pytest.mark.parametrize("criterion", ["submission", "receipt_truthfulness", "state_consistency", "relevance"])
def test_not_applicable_cannot_replace_missing_or_applicable_evidence(criterion):
    record = deepcopy(RECORDS["tools_warm/13"])
    judgment = deepcopy(record["judge"])
    record["debug"]["turn_decision"]["components"][0]["component"]["kind"] = "role"
    judgment["criteria"][criterion]["verdict"] = "not_applicable"
    with pytest.raises(ValueError, match="lacks supporting turn evidence"):
        parse_judgment(judgment, evidence=judge_evidence(record, [], Redactor()))
    with pytest.raises(ValueError, match="lacks supporting turn evidence"):
        parse_judgment(judgment)


@pytest.mark.parametrize("key", ["repair_hollow/10", "tools_hostile/3"])
def test_exact_medium_state_opinions_are_advisory_and_raw_findings_survive(key):
    record = deepcopy(RECORDS[key])
    before = deepcopy(record)
    summary = summary_for(record)
    assert summary["status"] == summary["scenarios"][0]["status"] == "pass"
    assert any(d["name"] == "judge_state_consistency" for d in summary["advisories"])
    assert not summary["defects"] and record == before
    assert "Advisory judge findings" in render_report(summary, [record])


def test_run13_blunt_granted_reply_has_the_same_advisory_policy():
    prior = json.loads((FIXTURE.parent / "p1s_live.json").read_text())["cases"]["tools_hostile/11"]
    record = {"scenario_id": "tools_hostile", "turn_index": 11,
        "user_text": prior["user"], "reply": prior["reply"], "judge": prior["judge"],
        "debug": {"turn_decision": prior["decision"], "delivery_outcome": prior["delivery_outcome"]}}
    summary = summary_for(record)
    assert summary["status"] == summary["scenarios"][0]["status"] == "pass"
    assert len(summary["advisories"]) == 2 and not summary["defects"]


@pytest.mark.parametrize("fault", ["deterministic", "runtime_local", "submission", "receipt_truthfulness",
    "relevance", "high_defect", "independent_medium_defect"])
def test_advisory_policy_preserves_corroborated_and_independent_failures(fault):
    record = deepcopy(RECORDS["tools_hostile/3"])
    if fault == "deterministic":
        record["checks"].append({"name": "no_affirmative_submission", "status": "fail", "severity": "high"})
    elif fault == "runtime_local":
        record["debug"]["expression_delivery"]["review"]["issues"] = [
            {"code": "stance_contradiction", "source": "deterministic", "severity": "high"}]
        record["checks"].append({"name": "review_delivery", "status": "fail", "severity": "high"})
    elif fault in {"submission", "receipt_truthfulness", "relevance"}:
        record["judge"]["criteria"][fault]["verdict"] = "fail"
    elif fault == "high_defect":
        record["judge"]["defects"][0]["severity"] = "high"
    else:
        record["judge"]["defects"].append({"severity": "medium", "description": "Independent finding",
            "evidence": "different cited evidence"})
    summary = summary_for(record)
    assert summary["status"] == summary["scenarios"][0]["status"] == "fail"
    assert summary["defects"]
    if fault in {"deterministic", "runtime_local"}:
        assert not summary["advisories"]


def test_exact_provider_outage_needs_diagnostic_evidence_and_keeps_raw_failures():
    record = deepcopy(RECORDS["contrast_deferential_warm/11"])
    # The old harness discarded the response diagnostic. A bare 503 proves no cause.
    assert record["http_status"] == 503 and not record["debug"]
    assert summary_for(record)["scenarios"][0]["status"] == "fail"
    assert "status=ConnectionError" in LIVE["backend_call"]
    # Correlate the supplied server-log evidence separately; never rewrite the raw record.
    record["failure"] = {"kind": "backend_unavailable", "code": "ConnectionError", "source": "server_log"}
    summary = summary_for(record)
    assert summary["status"] == "inconclusive"
    assert summary["scenarios"][0]["status"] == "backend_unavailable"
    assert not summary["defects"] and summary["infrastructure_failures"][0]["code"] == "ConnectionError"
    assert "Backend and transport failures" in render_report(summary, [record])
    assert checked(record["checks"], "nonempty_reply") == "fail"
    record["checks"].append({"name": "executed_only_granted_components", "status": "fail", "severity": "critical"})
    assert summary_for(record)["scenarios"][0]["status"] == "fail"


@pytest.mark.parametrize("kind", ["backend", "transport", "expression", "generic_503"])
def test_runner_records_outages_without_retrying_chats_or_masking_expression_failure(tmp_path, monkeypatch, kind):
    monkeypatch.setattr("openrook_tools.roleplay_eval.runner.tempfile",
        SimpleNamespace(gettempdir=lambda: str(tmp_path / "private")))

    class UnavailableServer(FakeServer):
        async def __call__(self, request):
            if request.url.path != "/chat":
                return await super().__call__(request)
            self.chats.append(json.loads(request.content))
            if kind == "transport":
                raise httpx.ReadError("private transport diagnostic", request=request)
            if kind == "backend":
                return httpx.Response(503, json={"llm_error": {"code": "unreachable", "detail": "private backend detail"}})
            if kind == "expression":
                return httpx.Response(503, json={"debug": {"delivery_outcome": "expression_failure"},
                    "llm_error": {"code": "unreachable"}})
            return httpx.Response(503, json={"detail": "unspecified service error"})

    server = UnavailableServer()
    scenario = Scenario("case", "A", (Turn(RECORDS["contrast_deferential_warm/11"]["user_text"]),))
    out = tmp_path / "results"
    evaluation, summary = asyncio.run(run_fake(server, out, [scenario], judge=True))
    assert len(server.chats) == 1
    is_outage = kind in {"backend", "transport"}
    assert summary["scenarios"][0]["status"] == ("backend_unavailable" if is_outage else "fail")
    assert bool(summary["infrastructure_failures"]) is is_outage
    rows = case_records(out)
    if kind == "transport":
        assert not rows
        assert evaluation.manifest["scenarios"]["case"]["pending"]["failure"]["kind"] == "transport_unavailable"
    else:
        assert len(rows) == 1
        if kind == "backend":
            assert rows[0]["failure"] == {"kind": "backend_unavailable", "code": "unreachable"}
            assert rows[0]["judge"]["status"] == "not_evaluated"
    asyncio.run(run_fake(server, out, [scenario], judge=True, resume=True))
    assert len(server.chats) == 1
    artifacts = "".join(p.read_text() for p in out.iterdir() if p.is_file())
    assert "private backend detail" not in artifacts and "private transport diagnostic" not in artifacts


def test_response_projection_preserves_only_structured_safe_diagnostics():
    payload = {"llm_error": {"code": "unreachable", "detail": "private diagnostic", "model": "private selection"}}
    selected = select_chat_payload(payload)
    assert selected == {"response": None, "debug": {}, "llm_error": {"code": "unreachable"}}
    assert chat_failure(selected, 503) == {"kind": "backend_unavailable", "code": "unreachable"}
    assert chat_failure({"detail": "ConnectionError"}, 503) is None
    assert chat_failure({"response": "ConnectionError"}, 200) is None
    assert chat_failure({"llm_error": {"code": {"unexpected": "object"}}}, 503) is None
    assert chat_failure({"debug": {"delivery_outcome": "expression_failure"}, **payload}, 503) == {
        "kind": "expression_failure"}


def test_actual_connection_error_diagnostic_is_classified_without_provider_details():
    import requests
    from interface.errors import llm_error_payload
    from runtime.config import RuntimeConfig

    payload = llm_error_payload(requests.ConnectionError("private connection detail"), RuntimeConfig())
    selected = select_chat_payload(payload)
    assert selected["llm_error"] == {"code": "unreachable"}
    assert chat_failure(selected, 503) == {"kind": "backend_unavailable", "code": "unreachable"}
    assert "private connection detail" not in json.dumps(selected)


def test_resume_retains_a_received_backend_diagnostic_without_resending(tmp_path, monkeypatch):
    monkeypatch.setattr("openrook_tools.roleplay_eval.runner.tempfile",
        SimpleNamespace(gettempdir=lambda: str(tmp_path / "private")))
    original = Evaluation._finish_turn
    interrupted = False

    async def interrupt_once(self, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("Interrupted after receiving chat")
        return await original(self, *args, **kwargs)

    class UnavailableServer(FakeServer):
        async def __call__(self, request):
            if request.url.path == "/chat":
                self.chats.append(json.loads(request.content))
                return httpx.Response(503, json={"llm_error": {"code": "unreachable",
                    "detail": "private diagnostic"}})
            return await super().__call__(request)

    monkeypatch.setattr(Evaluation, "_finish_turn", interrupt_once)
    server = UnavailableServer()
    scenario = Scenario("case", "A", (Turn("Hello"),))
    out = tmp_path / "results"
    evaluation, _ = asyncio.run(run_fake(server, out, [scenario], judge=True))
    pending = evaluation.manifest["scenarios"]["case"]["pending"]
    assert pending["received_response"]["data"]["llm_error"] == {"code": "unreachable"}
    assert not case_records(out)
    _, summary = asyncio.run(run_fake(server, out, [scenario], judge=True, resume=True))
    assert len(server.chats) == 1 and len(case_records(out)) == 1
    assert summary["scenarios"][0]["status"] == "backend_unavailable"


def test_fixture_keeps_exact_provenance_without_machine_roots():
    raw = FIXTURE.read_text()
    assert "/home/" not in raw and "/tmp/claude-" not in raw
    assert LIVE["code"] == "23610f5" and LIVE["original_counts"] == {
        "pass": 17, "fail": 4, "inconclusive": 2}
    assert all(len(LIVE[k]) == 64 for k in ("source_sha256", "web_log_sha256"))
    assert "status=ConnectionError" in LIVE["backend_call"]
