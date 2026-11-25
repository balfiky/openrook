"""Offline tests for the held-out evaluation tool. Only neutral placeholder text is used."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from openrook_tools.heldout_eval import adapter, judge, scoring, writer
from openrook_tools.heldout_eval.catalogue import CATALOGUE, select
from openrook_tools.heldout_eval.cli import main
from openrook_tools.heldout_eval.runner import allocate


# ---------------------------------------------------------------- catalogue

def test_catalogue_is_abstract_and_complete():
    ids = [i.id for i in CATALOGUE]
    assert len(ids) == len(set(ids))
    for intent in CATALOGUE:
        assert isinstance(intent.expects_role_demand, bool)
        assert isinstance(intent.expects_hostility_toward_character, bool)
        assert isinstance(intent.is_control, bool)
        assert not any(ch in intent.description for ch in "\"`“”«»「」")
        assert intent.is_control == (intent.category == "control")
        if intent.is_control:
            assert not intent.expects_role_demand
            assert not intent.expects_hostility_toward_character
    assert sum(i.expects_role_demand for i in CATALOGUE) >= 4
    assert sum(i.expects_hostility_toward_character for i in CATALOGUE) >= 4
    assert sum(i.is_control for i in CATALOGUE) >= 7


def test_select_rejects_unknown():
    with pytest.raises(ValueError):
        select(["nope"])
    assert [i.id for i in select([CATALOGUE[1].id])] == [CATALOGUE[1].id]


# ------------------------------------------------------------------ adapter

SAMPLE = {
    "response": "reply-1",
    "debug": {
        "appraisal_frame": {
            "provenance": "model",
            "perception": {"target_probability": 0.8, "hostile_intent": 0.5, "intensity": 0.6,
                           "uncertainty": 0.2, "evidence": [{"ref": "e", "source": "appraisal"}]},
            "request_components": [
                {"id": "c1", "kind": "role", "description": "d", "role_identity": "r",
                 "autonomy_cost": 0.7},
                {"id": "c2", "kind": "task", "description": "d2", "autonomy_cost": 0.1},
            ],
        },
        "turn_decision": {
            "perception": {"target_probability": 0.9, "hostile_intent": 0.5, "intensity": 0.4,
                           "uncertainty": 0.1},
            "components": [
                {"component": {"id": "c1", "kind": "role"}, "granted": False},
                {"component": {"id": "c2", "kind": "task"}, "granted": True},
            ],
            "grants": ["c2"], "declined": ["c1"],
            "stance": {"engagement": 0.4, "boundary": 0.6},
            "chosen_silence": False,
        },
        "llm_calls": [{"phase": "x", "status": "ok", "duration_ms": 12}],
    },
}


def test_adapter_extracts_sample():
    ex = adapter.extract(SAMPLE)
    assert ex.reply == "reply-1"
    assert ex.perception["target_probability"] == 0.9  # frozen decision preferred
    assert ex.perception["provenance"] == "model"
    assert ex.perception["evidence_sources"] == ["appraisal"]
    assert [c["kind"] for c in ex.components] == ["role", "task"]
    assert ex.components[0]["granted"] is False and ex.components[1]["granted"] is True
    assert ex.components[0]["role_identity"] == "r"
    assert ex.decision["declined_ids"] == ["c1"]
    assert ex.decision["declined_kinds"] == ["role"]
    assert ex.decision["chosen_silence"] is False
    assert ex.model_calls == [{"phase": "x", "status": "ok", "duration_ms": 12.0}]
    assert all(ex.availability.values())


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"debug": None}, {"debug": []},
                                     {"response": 3, "debug": {"turn_decision": "bad"}}])
def test_adapter_missing_fields_are_explicit(payload):
    ex = adapter.extract(payload)
    assert ex.reply is None and ex.perception is None and ex.components is None
    assert ex.decision is None and ex.model_calls is None
    assert not any(ex.availability.values())


def test_adapter_partial_and_garbage_numbers():
    ex = adapter.extract({"response": "r", "debug": {"appraisal_frame": {
        "perception": {"target_probability": "high", "hostile_intent": True, "intensity": 0.3},
        "request_components": [{"id": "a", "kind": "role"}]}}})
    assert ex.perception["target_probability"] is None
    assert ex.perception["hostile_intent"] is None
    assert ex.perception["intensity"] == 0.3
    assert ex.components[0]["granted"] is None
    assert ex.decision is None and ex.availability["decision"] is False


# ------------------------------------------------------------------ scoring

def _rec(intent, lang, *, error=None, role=None, granted=None, tp=None, hi=None, verdict="skipped"):
    if error:
        return {"intent_id": intent, "language": lang, "error": error, "extraction": None,
                "derived": None, "judge": {"verdict": "skipped"}}
    comps = None if role is None else (
        [{"id": "c", "kind": "role", "granted": granted}] if role else
        [{"id": "t", "kind": "task", "granted": True}])
    perc = None if tp is None else {"target_probability": tp, "hostile_intent": hi}
    ex = {"components": comps, "perception": perc, "decision": None,
          "availability": {"components": comps is not None, "perception": perc is not None,
                           "decision": False}}
    return {"intent_id": intent, "language": lang, "error": None, "extraction": ex,
            "derived": scoring.derive(ex, 0.25), "judge": {"verdict": verdict}}


def test_derive_threshold_and_unknowns():
    d = scoring.derive({"components": None, "perception": {
        "target_probability": 0.5, "hostile_intent": 0.5}}, 0.25)
    assert d["perceived_hostility"] is True and d["perceived_role"] is None
    d = scoring.derive({"perception": {"target_probability": 0.5, "hostile_intent": 0.4}}, 0.25)
    assert d["perceived_hostility"] is False
    d = scoring.derive({"perception": {"target_probability": None, "hostile_intent": 0.9}}, 0.25)
    assert d["perceived_hostility"] is None


def test_summary_maths():
    role_i, host_i, ctrl_i = "role_demand_obedience", "hostile_insult_worth", "control_thanks"
    recs = [
        _rec(role_i, "en", role=True, granted=False, tp=0.9, hi=0.9, verdict="decline"),
        _rec(role_i, "en", role=True, granted=True, tp=0.1, hi=0.1, verdict="accept"),
        _rec(role_i, "en", role=False, tp=0.0, hi=0.0, verdict="ignore"),
        _rec(role_i, "en", error="timeout"),
        _rec(role_i, "en", role=True, granted=None, verdict="unknown"),
        _rec(host_i, "en", role=False, tp=1.0, hi=0.5, verdict="not_applicable"),
        _rec(host_i, "en", role=False, tp=None, hi=None),
        _rec(ctrl_i, "en", role=True, granted=False, tp=0.9, hi=0.9, verdict="decline"),
        _rec(ctrl_i, "en", role=False, tp=0.0, hi=0.0, verdict="not_applicable"),
        _rec(ctrl_i, "ar", role=False, tp=0.0, hi=0.0, verdict="not_applicable"),
        _rec(ctrl_i, "ar", error="http_500"),
    ]
    s = scoring.summarize(recs, 0.25)
    cell = next(c for c in s["by_intent_language"]
                if c["intent_id"] == role_i and c["language"] == "en")
    assert (cell["phrasings"], cell["errors"], cell["evaluated"]) == (5, 1, 4)
    assert scoring.fmt(cell["role_perceived"]) == "3/4"
    assert scoring.fmt(cell["hostility_perceived"]) == "1/3"  # one has no perception
    assert scoring.fmt(cell["role_declined"]) == "1/2"        # third has unknown grant status
    assert cell["not_available"]["role_grant_status"] == 1
    assert cell["judge"]["accept"] == 1 and cell["judge"]["unknown"] == 1
    en = s["overall_by_language"]["en"]
    assert scoring.fmt(en["role_demand_perceived_as_role"]) == "3/4"
    assert scoring.fmt(en["role_demand_judge_accept"]) == "1/3"
    assert scoring.fmt(en["hostility_perceived_as_hostile"]) == "1/1"
    assert scoring.fmt(en["control_false_role"]) == "1/2"
    assert scoring.fmt(en["control_false_hostility"]) == "1/2"
    assert scoring.fmt(en["control_judge_accept_or_decline"]) == "1/2"
    assert en["errors"] == 1
    ar = s["overall_by_language"]["ar"]
    assert (ar["phrasings"], ar["errors"], ar["evaluated"]) == (2, 1, 1)
    assert scoring.fmt(ar["control_false_role"]) == "0/1"


# ------------------------------------------------------------ writer / judge

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Session:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        t = self.texts.pop(0)
        return _Resp(200, {"choices": [{"message": {"content": t}}]})


def test_parse_phrasings_validation():
    assert writer.parse_phrasings('```json\n["phrase-1", "phrase-2", "phrase-2"]\n```', 2) == [
        "phrase-1", "phrase-2"]
    for bad in ["nothing", '["a", 3]', '{"a": 1}', '["only-one"]', "[unterminated"]:
        with pytest.raises(ValueError):
            writer.parse_phrasings(bad, 2)


def test_writer_retries_on_malformed_then_succeeds():
    sess = _Session(["garbage", '["phrase-1"]', '["phrase-1", "phrase-2"]'])
    cfg = writer.WriterConfig("http://x/v1", "m", retries=3, seed=5)
    res = writer.generate(cfg, CATALOGUE[0], "en", 2, session=sess)
    assert res.error is None and res.phrasings == ["phrase-1", "phrase-2"]
    assert res.attempts == 3
    seeds = [c["seed"] for c in sess.calls]
    assert len(set(seeds)) == 3  # reproducible but varied per attempt


def test_writer_gives_up_after_bounded_retries():
    sess = _Session(["bad"] * 5)
    res = writer.generate(writer.WriterConfig("http://x", "m", retries=2), CATALOGUE[0], "ar", 2,
                          session=sess)
    assert res.phrasings == [] and res.attempts == 2 and "malformed" in res.error
    assert len(sess.calls) == 2


def test_prompt_uses_language_name_and_description_only():
    prompt = writer.writer_prompt(CATALOGUE[0], "ar", 3)
    assert "Arabic" in prompt and CATALOGUE[0].description in prompt
    assert "code xx-YY" in writer.writer_prompt(CATALOGUE[0], "xx-YY", 1)


def test_judge_parsing_is_strict():
    good = judge.parse_verdict('{"verdict": "Decline", "evidence": "reply-1"}', "a reply-1 b")
    assert good["verdict"] == "decline" and good["evidence_verified"] is True
    unverified = judge.parse_verdict('{"verdict": "accept", "evidence": "zzz"}', "reply-1")
    assert unverified["verdict"] == "accept" and unverified["evidence_verified"] is False
    for bad in ["", "no json", '{"verdict": "maybe"}', "{broken", '{"evidence": "x"}']:
        assert judge.parse_verdict(bad, "r")["verdict"] == "unknown"
    cfg = writer.WriterConfig("http://x", "m")

    class Boom:
        def post(self, *a, **k):
            return _Resp(500, {})

    assert judge.judge_reply(cfg, "m", "r", session=Boom())["verdict"] == "unknown"
    assert judge.judge_reply(cfg, "m", "  ", session=Boom())["verdict"] == "no_reply"


def test_allocate_limit_round_robin():
    assert allocate(3, 2, None) == [2, 2, 2]
    assert allocate(3, 2, 4) == [2, 1, 1]
    assert sum(allocate(5, 3, 7)) == 7
    assert allocate(2, 2, 100) == [2, 2]


# ----------------------------------------------------------- stub servers

class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _read(self):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Model(_Quiet):
    """Stub OpenAI-compatible endpoint serving both writer and judge requests."""
    counter = 0
    lock = threading.Lock()
    seen_auth: list = []

    def do_POST(self):
        body = self._read()
        type(self).seen_auth.append(self.headers.get("Authorization"))
        user = body["messages"][-1]["content"]
        m = re.search(r"Write (\d+) different", user)
        if m:
            with self.lock:
                start = type(self).counter
                type(self).counter += int(m.group(1))
            text = json.dumps([f"phrase-{start + i}" for i in range(int(m.group(1)))])
        else:
            text = json.dumps({"verdict": "decline", "evidence": "reply-text"})
        self._send(200, {"choices": [{"message": {"content": text}}]})


class _OpenRook(_Quiet):
    def do_POST(self):
        body = self._read()
        assert self.path == "/chat"
        assert set(body) == {"message", "user_id", "chat_id"}
        assert body["user_id"] != body["chat_id"]
        n = int(body["message"].split("-")[1])
        if n % 7 == 6:
            return self._send(500, {"detail": "boom"})
        debug = {} if n % 5 == 4 else {
            "appraisal_frame": {"provenance": "model", "perception": {
                "target_probability": 0.9, "hostile_intent": 0.8, "intensity": 0.5},
                "request_components": [{"id": "c1", "kind": "role" if n % 2 == 0 else "task"}]},
            "turn_decision": {"components": [
                {"component": {"id": "c1", "kind": "role" if n % 2 == 0 else "task"},
                 "granted": False}], "grants": [], "declined": ["c1"], "chosen_silence": False},
            "llm_calls": [],
        }
        self._send(200, {"response": "reply-text", "debug": debug})


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    _Model.counter = 0
    _Model.seen_auth = []
    started = []
    urls = []
    for handler in (_Model, _OpenRook):
        srv = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        urls.append(f"http://127.0.0.1:{srv.server_port}")
    yield urls
    for srv in started:
        srv.shutdown()
        srv.server_close()


def _args(servers, out, *extra):
    model_url, rook_url = servers
    return ["--base-url", rook_url, "--writer-base-url", model_url + "/v1",
            "--writer-model", "stub-model", "--out", str(out), *extra]


def test_cli_dry_run_generates_only_phrasings(servers, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_WRITER_KEY", "secret-value-1")
    out = tmp_path / "o"
    code = main(_args(servers, out, "--languages", "en,ar", "--per-intent", "2",
                      "--writer-key-env", "STUB_WRITER_KEY", "--dry-run", "--seed", "3"))
    assert code == 0
    data = json.loads((out / "phrasings.json").read_text())
    assert len(data["phrasings"]) == len(CATALOGUE) * 2 * 2
    assert {p["language"] for p in data["phrasings"]} == {"en", "ar"}
    assert not (out / "results.jsonl").exists()
    assert "secret-value-1" not in (out / "phrasings.json").read_text()
    assert "Bearer secret-value-1" in _Model.seen_auth


def test_cli_usage_errors(servers, tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY_ENV", raising=False)
    with pytest.raises(SystemExit) as e:
        main(_args(servers, tmp_path, "--writer-key-env", "MISSING_KEY_ENV"))
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(_args(servers, tmp_path, "--languages", "not a code!"))
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["--base-url", "x"])
    assert e.value.code == 2


def test_cli_end_to_end(servers, tmp_path):
    out = tmp_path / "o"
    code = main(_args(servers, out, "--languages", "en,es", "--per-intent", "3",
                      "--intents", "role_demand_obedience,control_thanks",
                      "--concurrency", "3", "--request-timeout", "10"))
    assert code == 0
    for name in ("phrasings.json", "results.jsonl", "summary.json", "report.md"):
        assert (out / name).exists()
    recs = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(recs) == 12
    keys = [(r["intent_id"], r["language"], r["index"]) for r in recs]
    assert keys == sorted(keys, key=lambda k: (
        ["role_demand_obedience", "control_thanks"].index(k[0]), ["en", "es"].index(k[1]), k[2]))
    errs = [r for r in recs if r["error"]]
    assert errs and all(r["error"] == "http_500" for r in errs)
    ok = [r for r in recs if not r["error"]]
    assert all(r["judge"]["verdict"] == "decline" for r in ok)
    missing = [r for r in ok if not r["extraction"]["availability"]["decision"]]
    assert missing and all(r["derived"]["perceived_role"] is None for r in missing)

    summary = json.loads((out / "summary.json").read_text())
    total_err = sum(c["errors"] for c in summary["by_intent_language"])
    assert total_err == len(errs) == summary["meta"]["errors_total"]
    assert summary["errors_by_kind"] == {"http_500": len(errs)}
    report = (out / "report.md").read_text()
    assert "Per intent and language" in report and re.search(r"\d+/\d+", report)
    assert "%" not in report
    assert "stub-model" not in report
    # judged items must never be counted as judge-known when the judge failed: none failed here
    assert all(o["judge_unknown"] == 0 for o in summary["overall_by_language"].values())


def test_cli_nothing_evaluated_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _Model.counter = 0
    srv = HTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}"
        # server base-url points at a closed port: every item errors
        code = main(["--base-url", "http://127.0.0.1:9", "--writer-base-url", url,
                     "--writer-model", "m", "--out", str(tmp_path / "o"), "--per-intent", "1",
                     "--intents", "control_thanks", "--judge", "off", "--request-timeout", "2"])
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 1
    recs = (tmp_path / "o" / "results.jsonl").read_text().splitlines()
    assert json.loads(recs[0])["error"].startswith("connection_error")


def test_cli_reuse_phrasings_and_new_record_fields(servers, tmp_path):
    first = tmp_path / "run-one"
    assert main(_args(servers, first, "--languages", "en,es", "--per-intent", "2",
                      "--intents", "role_demand_obedience,control_thanks", "--dry-run")) == 0
    original = json.loads((first / "phrasings.json").read_text())["phrasings"]

    out = tmp_path / "run-two"
    model_url, rook_url = servers
    code = main(["--base-url", rook_url, "--phrasings-from", str(first / "phrasings.json"),
                 "--languages", "en,es", "--intents", "role_demand_obedience,control_thanks",
                 "--judge", "off", "--out", str(out), "--request-timeout", "10"])
    assert code == 0
    copied = json.loads((out / "phrasings.json").read_text())
    assert [p["text"] for p in copied["phrasings"]] == [p["text"] for p in original]
    assert copied["reused_from"] == "run-one"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["phrasings_reused"] is True
    assert summary["meta"]["phrasings_reused_from"] == "run-one"
    assert str(tmp_path) not in (out / "summary.json").read_text()
    assert "reused from an earlier run: run-one" in (out / "report.md").read_text()
    recs = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    ok = [r for r in recs if not r["error"]]
    assert ok and all(r["reply"] == "reply-text" for r in ok)
    assert any(r["appraisal_provenance"] == "model" for r in ok)
    assert all(r["model_call_phases"] in ([], None) for r in ok)
    prov = summary["appraisal_provenance"]
    assert prov["model"] + prov["other"] + prov["not_available"] == len(ok)
    assert prov["model"] >= 1 and prov["not_available"] >= 1
    assert "Appraisal reading used per turn" in (out / "report.md").read_text()


def test_cli_reuse_mismatch_is_usage_error(servers, tmp_path):
    first = tmp_path / "r"
    main(_args(servers, first, "--languages", "en", "--per-intent", "1",
               "--intents", "control_thanks", "--dry-run"))
    _, rook_url = servers
    for extra in (["--languages", "ar"], ["--intents", "role_claim_ownership"]):
        with pytest.raises(SystemExit) as e:
            main(["--base-url", rook_url, "--phrasings-from", str(first / "phrasings.json"),
                  "--judge", "off", "--out", str(tmp_path / "x"),
                  *(["--intents", "control_thanks"] if "--intents" not in extra else []),
                  *extra])
        assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["--base-url", rook_url, "--phrasings-from", str(tmp_path / "missing.json"),
              "--out", str(tmp_path / "x")])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:  # no writer and no reuse
        main(["--base-url", rook_url, "--out", str(tmp_path / "x")])
    assert e.value.code == 2


def test_cli_reuse_carries_failed_cells(servers, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "phrasings.json").write_text(json.dumps({
        "phrasings": [{"intent_id": "control_thanks", "language": "en", "index": 0,
                       "text": "phrase-1"}],
        "generation_failures": [{"intent_id": "control_thanks", "language": "ar",
                                 "error": "malformed output: x", "attempts": 3}]}))
    _, rook_url = servers
    base = ["--base-url", rook_url, "--phrasings-from", str(src / "phrasings.json"),
            "--intents", "control_thanks", "--judge", "off", "--request-timeout", "10"]
    out = tmp_path / "o"
    assert main([*base, "--languages", "en,ar", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["generation_failures_count"] == 1
    assert summary["meta"]["generation_failures"][0]["language"] == "ar"
    assert [c["language"] for c in summary["by_intent_language"]] == ["en"]
    assert "Writer failures" in (out / "report.md").read_text()
    with pytest.raises(SystemExit) as e:  # neither present nor listed as failed
        main([*base, "--languages", "en,es", "--out", str(tmp_path / "p")])
    assert e.value.code == 2


def test_adapter_provenance_and_phases():
    ex = adapter.extract({"response": "r", "debug": {
        "appraisal_frame": {"provenance": "heuristic"},
        "llm_calls": [{"phase": "p1", "status": "ok"}, {"status": "ok"}]}})
    assert ex.appraisal_provenance == "heuristic"
    assert ex.model_call_phases() == ["p1", "unknown"]
    assert scoring.provenance_class("model") == "model"
    assert scoring.provenance_class("heuristic") == "other"
    assert scoring.provenance_class(None) == "not_available"


def test_writer_prompt_requests_variety():
    prompt = writer.writer_prompt(CATALOGUE[0], "en", 3)
    assert "indirect or implied" in prompt and "natively" in prompt
    assert "key noun or adjective" in prompt


class _CellSession:
    """Thread-safe stub writer: fails every attempt for one chosen intent description."""

    def __init__(self, fail_description=None):
        self.fail, self.calls, self.lock = fail_description, 0, threading.Lock()

    def post(self, url, headers=None, json=None, timeout=None):
        prompt = json["messages"][-1]["content"]
        with self.lock:
            self.calls += 1
            n = self.calls
        count = int(re.search(r"Write (\d+) different", prompt).group(1))
        if self.fail and self.fail in prompt:
            text = "not json"
        else:
            text = globals()["json"].dumps([f"phrase-{n}-{i}" for i in range(count)])
        return _Resp(200, {"choices": [{"message": {"content": text}}]})


def _build(tmp_path, session, **kw):
    from openrook_tools.heldout_eval.runner import build_phrasings
    intents = select(["role_demand_obedience", "control_thanks", "hostile_insult_worth"])
    langs = ["en", "ar"]
    params = {"intents": [i.id for i in intents], "languages": langs, "per_intent": 2,
              "seed": 1, "limit": None}
    cfg = writer.WriterConfig("http://x", "m", retries=1, seed=1)
    return build_phrasings(cfg, intents, langs, 2, None, session=session, out_dir=tmp_path,
                           params=params, concurrency=4, **kw)


def test_concurrent_generation_resume_failure_and_progress(tmp_path, capsys):
    failing = select(["control_thanks"])[0].description
    sess = _CellSession(failing)
    ph, fl = _build(tmp_path, sess)
    assert sess.calls == 6
    # deterministic order: intent order, then language order, then index
    keys = [(p["intent_id"], p["language"], p["index"]) for p in ph]
    assert keys == [("role_demand_obedience", "en", 0), ("role_demand_obedience", "en", 1),
                    ("role_demand_obedience", "ar", 0), ("role_demand_obedience", "ar", 1),
                    ("hostile_insult_worth", "en", 0), ("hostile_insult_worth", "en", 1),
                    ("hostile_insult_worth", "ar", 0), ("hostile_insult_worth", "ar", 1)]
    assert {(f["intent_id"], f["language"]) for f in fl} == {
        ("control_thanks", "en"), ("control_thanks", "ar")}
    saved = json.loads((tmp_path / "phrasings.json").read_text())
    assert saved["complete"] is True and len(saved["generation_failures"]) == 2
    assert not (tmp_path / "phrasings.json.tmp").exists()
    err = capsys.readouterr().err
    assert err.count("writer cell ") == 6 and "attempts=1" in err
    assert not any(p["text"] in err for p in ph)

    # resume: completed cells are skipped, the failed ones are retried (now succeed)
    sess2 = _CellSession()
    ph2, fl2 = _build(tmp_path, sess2)
    assert sess2.calls == 2 and not fl2 and len(ph2) == 12
    assert [p["text"] for p in ph2 if p["intent_id"] == "role_demand_obedience"] == [
        p["text"] for p in ph if p["intent_id"] == "role_demand_obedience"]
    # --fresh regenerates everything
    sess3 = _CellSession()
    _build(tmp_path, sess3, fresh=True)
    assert sess3.calls == 6


def test_resume_ignores_partial_file_with_different_params(tmp_path):
    sess = _CellSession()
    _build(tmp_path, sess)
    from openrook_tools.heldout_eval.runner import build_phrasings
    intents = select(["control_thanks"])
    other = {"intents": ["control_thanks"], "languages": ["en"], "per_intent": 2,
             "seed": 2, "limit": None}
    s2 = _CellSession()
    cfg = writer.WriterConfig("http://x", "m", retries=1, seed=2)
    build_phrasings(cfg, intents, ["en"], 2, None, session=s2, out_dir=tmp_path, params=other)
    assert s2.calls == 1


def test_cli_partial_failure_report_and_exit_codes(servers, tmp_path, capsys):
    out = tmp_path / "o"
    code = main(_args(servers, out, "--languages", "en", "--per-intent", "1",
                      "--intents", "control_thanks", "--judge", "off", "--writer-concurrency", "2"))
    assert code == 0
    err = capsys.readouterr().err
    assert "chat items 1/1" in err and "writer cell control_thanks en ok" in err
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["generation_failures_count"] == 0


def test_repo_files_have_no_secrets_or_paths():
    root = Path(__file__).resolve().parents[1] / "openrook_tools" / "heldout_eval"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "/home/" not in text and "sk-" not in text
