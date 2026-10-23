"""Offline independence denominators, blinding, matching and runner contracts."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from config.loader import SoulConfig
from openrook_tools.roleplay_eval.api import Redactor
from openrook_tools.roleplay_eval.evidence import compact_record
from openrook_tools.roleplay_eval.independence import CRITERIA, comparisons, review_evidence, summarize
from openrook_tools.roleplay_eval.judge import judge_evidence, parse_judgment
from openrook_tools.roleplay_eval.reporting import build_summary, render_report
from openrook_tools.roleplay_eval.scenarios import Scenario, Turn, catalog, profile
from tests.test_conversation_independence import choose, retained
from tests.test_roleplay_eval import FakeServer, case_records, judgment, run_fake


def row(index=0, *, case="case", personality="shipped_default", tags=(), selected=True):
    state = retained(SoulConfig(dispositions={"patience": 0.8}, core_values={"kindness": 0.8}))
    choice = choose(state, agenda=selected)
    return {"scenario_id": case, "category": "K", "personality": personality,
            "turn_index": index, "user_text": "I've finished my thought.", "reply": "Scale can conceal real structure.",
            "phase": "probe", "tags": ["independence", *tags], "metrics": {}, "checks": [],
            "debug": {"turn_decision": choice.to_dict(), "delivery_outcome": "delivered"},
            "judge": {"status": "ok", **judgment(), "independence": {
                k: {"verdict": "pass", "reason": "Grounded in the selected self.", "evidence": "Scale"} for k in CRITERIA}}}


def test_catalog_keeps_original_cases_and_matches_histories_and_controls():
    cases = catalog()
    legacy = [s for s in cases if s.category != "K"]
    assert len(legacy) == 23 and sum(len(s.turns) for s in legacy) == 295
    added = {s.id: s for s in cases if s.category == "K"}
    assert len(added) == 8 and all(s.exclusive for s in added.values())
    a, b = added["independence_maps"], added["independence_sound"]
    assert a.personality == b.personality
    assert profile(a.personality, 17) == profile(b.personality, 17)
    assert a.turns[:10] != b.turns[:10] and a.turns[10:] == b.turns[10:]
    assert not added["independence_maps_control"].settings["autobiographical_learning"]
    assert not added["independence_sound_control"].settings["autobiographical_learning"]
    assert added["independence_restart"].restart_after is not None
    all_tags = {t for s in added.values() for turn in s.turns for t in turn.tags}
    assert {"initiative_opportunity", "elicited_want", "resume_opportunity", "opinion_pressure",
            "counterevidence", "preference_opportunity", "common_probe", "consolidate_before"} <= all_tags


def test_spontaneous_and_elicited_opportunities_do_not_share_a_denominator():
    records = [row(0, tags=("initiative_opportunity",)), row(1, tags=("elicited_want",)),
               row(2, tags=("initiative_opportunity",), selected=False)]
    measures = summarize(records)["measures"]
    assert measures["spontaneous"]["selection"]["rate"] == 0.5
    assert measures["elicited"]["selection"]["rate"] == 1
    assert measures["spontaneous"]["expression_and_delivery"]["rate"] == 0.5
    assert measures["spontaneous"]["visible_selections"] == 1
    records[0]["judge"]["independence"]["intention_expression"]["verdict"] = "uncertain"
    unknown = summarize(records)["measures"]["spontaneous"]["expression_and_delivery"]
    assert unknown["rate"] is None and unknown["unknown"] == 1
    assert unknown["lower_bound"] == 0 and unknown["upper_bound"] == 0.5
    records[0]["debug"]["delivery_outcome"] = "expression_failure"
    records[0]["reply"] = ""
    assert summarize(records)["measures"]["spontaneous"]["visible_selections"] == 0


def test_owned_topics_require_an_actual_prior_delivery_and_match_after_restart():
    first = row(0)
    interruption = row(1, selected=False)
    resumed = row(2, tags=("initiative_opportunity", "resume_opportunity"))
    resumed["phase"] = "after_restart"
    measured = summarize([first, interruption, resumed])["measures"]["resumed_after_restart"]
    assert measured["selection"]["rate"] == measured["expression_and_delivery"]["rate"] == 1
    first["debug"]["delivery_outcome"] = "expression_failure"
    first["reply"] = ""
    absent = summarize([first, interruption, resumed])["measures"]["resumed_after_restart"]
    assert absent["selection"]["denominator"] == 0 and absent["selection"]["not_applicable"] == 1


def test_applicability_is_structural_and_missing_self_remains_unknown():
    record = row(0, selected=False)
    absent = review_evidence(record, [row(-1)])
    assert "intention_expression" in absent["inapplicable_criteria"]
    assert "owned_topic_continuity" in absent["inapplicable_criteria"]
    assert "opinion_persistence" in absent["inapplicable_criteria"]
    assert "service_handback" not in absent["inapplicable_criteria"]
    record["tags"].append("opinion_pressure")
    record["debug"]["turn_decision"]["personality"] = {}
    assert "opinion_persistence" not in review_evidence(record, [])['inapplicable_criteria']
    record["judge"]["independence"]["opinion_persistence"].update(verdict="uncertain", reason="There is nothing to judge.")
    assert summarize([record])["measures"]["opinion_persistence"]["unknown"] == 1


def test_pressure_and_counterevidence_are_distinct_and_preference_declines_are_benign():
    pressure = row(0, tags=("opinion_pressure",))
    revision = row(1, tags=("counterevidence",))
    revision["judge"]["independence"]["opinion_revision"]["verdict"] = "fail"
    measures = summarize([pressure, revision])["measures"]
    assert measures["opinion_persistence"]["rate"] == 1
    assert measures["opinion_revision"]["rate"] == 0
    benign = row(2, tags=("preference_opportunity",))
    c = {"id": "c0", "description": "Write flattering slogans", "kind": "task", "feasible": True,
         "evidence_refs": ["self:preferences:empty flattery"]}
    state = json.loads(benign["debug"]["turn_decision"]["personality"]["self_state_json"])
    state["preferences"]["empty flattery"] = {"weight": -1, "subject": "empty flattery"}
    benign["debug"]["turn_decision"]["personality"]["self_state_json"] = json.dumps(state)
    benign["debug"]["turn_decision"]["components"] = [{"component": c, "granted": False,
        "contributions": [["character_cost", -1]]}]
    assert summarize([benign])["measures"]["benign_declines"]["selection"]["rate"] == 1
    state["preferences"]["empty flattery"]["weight"] = 1
    benign["debug"]["turn_decision"]["personality"]["self_state_json"] = json.dumps(state)
    assert summarize([benign])["measures"]["benign_declines"]["selection"]["rate"] == 0
    benign["debug"]["turn_decision"]["components"][0]["granted"] = True
    assert summarize([benign])["measures"]["benign_declines"]["selection"]["rate"] == 0
    c["kind"] = "role"
    assert summarize([benign])["measures"]["benign_declines"]["selection"]["unknown"] == 1


def test_judge_blinding_and_not_applicable_validation_cover_independence():
    record = row(0, case="secret_case_name", tags=("opinion_pressure",))
    record.update(user_id="secret_user", chat_id="secret_chat")
    evidence = judge_evidence(record, [row(-1)], Redactor())
    serialized = json.dumps(evidence)
    assert all(label not in serialized for label in ("secret_case_name", "secret_user", "secret_chat", "shipped_default", "opinion_pressure"))
    assert evidence["independence"]["current_self"]["opinions"]
    raw = {k: v for k, v in record["judge"].items() if k != "status"}
    raw["independence"]["opinion_persistence"]["verdict"] = "not_applicable"
    with pytest.raises(ValueError, match="lacks supporting structure"):
        parse_judgment(raw, evidence=evidence)
    raw["independence"]["opinion_persistence"]["verdict"] = "uncertain"
    raw["independence"]["opinion_revision"]["verdict"] = "not_applicable"
    assert parse_judgment(raw, evidence=evidence)["independence"]["opinion_revision"]["verdict"] == "not_applicable"
    raw["independence"]["service_handback"].update(verdict="fail", evidence="A fabricated citation")
    with pytest.raises(ValueError, match="actual generated reply"):
        parse_judgment(raw, evidence=evidence)


def test_independence_findings_are_not_floor_defects_and_survive_compaction():
    record = row(0)
    record["judge"]["independence"]["service_handback"]["verdict"] = "fail"
    record = compact_record(record)
    case = Scenario("case", "K", (Turn("A neutral opportunity"),))
    summary = build_summary([case], [record], {"run_id": "test", "seed": 0, "judge_enabled": True})
    assert summary["status"] == "pass" and not summary["defects"]
    assert summary["independence"]["measures"]["service_handback"]["rate"] == 1
    assert "Independence measures (separate from floors)" in render_report(summary, [record])
    record["checks"] = [{"name": "actual_floor", "status": "fail", "severity": "critical"}]
    assert build_summary([case], [record], {"run_id": "test", "seed": 0})["status"] == "fail"


def test_common_probes_measure_within_subject_and_matched_history_divergence():
    first, last = row(0, tags=("common_probe",)), row(1, tags=("common_probe",))
    other = row(0, case="other", tags=("common_probe",))
    state = json.loads(other["debug"]["turn_decision"]["personality"]["self_state_json"])
    state["interests"]["maps"]["weight"] = -0.8
    other["debug"]["turn_decision"]["personality"]["self_state_json"] = json.dumps(state)
    third = deepcopy(other)
    third.update(scenario_id="third", personality="patient")
    rows = comparisons({"case": [first, last], "other": [other], "third": [third]})
    within = next(r for r in rows if r["kind"] == "within_subject")
    between = next(r for r in rows if r["kind"] == "between_histories")
    assert within["retained_state_distance"] == 0
    assert between["retained_state_distance"] > 0 and between["same_birth"] is True
    assert any(r["kind"] == "between_subjects" for r in rows)
    del other["debug"]["turn_decision"]["personality"]["self_state_json"]
    assert comparisons({"case": [last], "other": [other]})[0]["status"] == "unknown"


def test_fresh_subject_identifiers_cannot_manufacture_history_divergence():
    left, right = row(0, tags=("common_probe",)), row(0, case="other", tags=("common_probe",))
    for record, person in ((left, "alice"), (right, "bob")):
        state = json.loads(record["debug"]["turn_decision"]["personality"]["self_state_json"])
        state["attitudes"] = {person: {"weight": 0.5, "description": "A trusted collaborator."}}
        record["debug"]["turn_decision"]["personality"]["self_state_json"] = json.dumps(state)
        record["debug"]["turn_decision"]["selected_intention"].update(
            evidence_refs=[f"goal:{person}"], content="Compare the estuary surveys.")
    matched = comparisons({"case": [left], "other": [right]})[0]
    assert matched["retained_state_distance"] == 0 and not matched["topic_selection_changed"]
    for record in (left, right):
        record["debug"]["turn_decision"]["selected_intention"]["evidence_refs"] = ["question:1"]
    right["debug"]["turn_decision"]["selected_intention"]["content"] = "Compare temporal masking in the tone trials."
    assert comparisons({"case": [left], "other": [right]})[0]["topic_selection_changed"]


def test_confirmed_outages_are_excluded_without_counting_them_as_no_initiative():
    good, outage = row(0, tags=("initiative_opportunity",)), row(1, tags=("initiative_opportunity",))
    outage.update(reply="", failure={"kind": "backend_unavailable"})
    result = summarize([good, outage])
    assert result["unavailable_turns"] == 1
    assert result["measures"]["spontaneous"]["selection"]["denominator"] == 1


def test_missing_planned_opportunities_and_restart_remain_unknown():
    case = Scenario("case", "K", (
        Turn("An early gap", ("independence", "initiative_opportunity")),
        Turn("A later gap", ("independence", "initiative_opportunity", "resume_opportunity"), "after_restart")))
    result = summarize([row(0, tags=("initiative_opportunity",))], [case])
    assert result["missing_planned_turns"] == 1
    assert result["measures"]["spontaneous"]["selection"]["unknown"] == 1
    assert result["measures"]["resumed_after_restart"]["selection"]["unknown"] == 1
    assert result["measures"]["resumed_after_restart"]["selection"]["rate"] is None


def test_conditioning_boundaries_are_idempotent_and_learning_setting_restores(tmp_path, monkeypatch):
    monkeypatch.setattr("openrook_tools.roleplay_eval.runner.tempfile", SimpleNamespace(gettempdir=lambda: str(tmp_path / "private")))
    server = FakeServer()
    server.config["autobiographical_learning"] = False
    server.original = deepcopy((server.config, server.soul))
    case = Scenario("episodes", "K", (
        Turn("One observation", ("independence",)),
        Turn("A second observation", ("independence", "consolidate_before")),
        Turn("A neutral probe", ("independence", "consolidate_before", "initiative_opportunity"))),
        exclusive=True, settings={"autobiographical_learning": True})
    out = tmp_path / "run"
    evaluation, _ = asyncio.run(run_fake(server, out, [case]))
    checkpoints = evaluation.manifest["scenarios"]["episodes"]["checkpoints"]
    assert "post_episode_1" in checkpoints and "post_episode_2" in checkpoints
    assert len(case_records(out)) == 3 and len(server.chats) == 3
    assert (server.config, server.soul) == server.original
    asyncio.run(run_fake(server, out, [case], resume=True))
    assert len(server.chats) == 3


def test_all_independence_cases_stage_and_restore_and_keep_separate_judgments(tmp_path, monkeypatch):
    monkeypatch.setattr("openrook_tools.roleplay_eval.runner.tempfile", SimpleNamespace(gettempdir=lambda: str(tmp_path / "private")))
    server = FakeServer()
    server.config["autobiographical_learning"] = False
    server.original = deepcopy((server.config, server.soul))
    server.judge_response["independence"] = {
        name: {"verdict": "uncertain", "reason": "The stub has no retained self.", "evidence": ""} for name in CRITERIA}
    cases = [s for s in catalog() if s.category == "K"]
    _, summary = asyncio.run(run_fake(server, tmp_path / "cases", cases, max_turns=2, judge=True))
    records = case_records(tmp_path / "cases")
    assert len(records) == 16 and all(r["judge"]["status"] == "ok" for r in records)
    assert all("independence" in r["judge"] for r in records)
    assert summary["independence"]["missing_planned_turns"] == 129 - 16
    assert (server.config, server.soul) == server.original
