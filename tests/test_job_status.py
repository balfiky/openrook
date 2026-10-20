from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.job_status import inspect_job, main, process_details


def record_job(tmp_path, **updates):
    record = {
        "status": "running",
        "pid": 12345,
        "process_identity": "boot:100",
        "started_at": 100,
        "heartbeat_at": 130,
        **updates,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(record))
    return path


def test_dead_process_is_interrupted_without_rewriting_evidence(tmp_path, monkeypatch):
    path = record_job(tmp_path)
    original = path.read_bytes()

    def missing(pid, signal):
        assert pid == 12345 and signal == 0
        raise ProcessLookupError

    monkeypatch.setattr("evals.job_status.os.kill", missing)
    monkeypatch.setattr("evals.job_status.process_details", lambda pid: None)
    result = inspect_job(tmp_path, now=240)
    assert result["status"] == "interrupted"
    assert result["recorded_status"] == "running"
    assert result["process_alive"] is False
    assert result["heartbeat_stale"] is True
    assert result["heartbeat_age_seconds"] == 110
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("identity", "state", "alive"),
    [
        ("boot:100", "S", True),
        ("other-boot:100", "S", False),
        ("boot:101", "S", False),
        ("boot:100", "Z", False),
        ("boot:100", "X", False),
    ],
)
def test_identity_reuse_and_zombie_detection(tmp_path, monkeypatch, identity, state, alive):
    record_job(tmp_path)
    monkeypatch.setattr("evals.job_status.os.kill", lambda pid, signal: None)
    monkeypatch.setattr(
        "evals.job_status.process_details", lambda pid: {"identity": identity, "state": state}
    )
    result = inspect_job(tmp_path, now=140)
    assert result["process_alive"] is alive
    assert result["status"] == ("running" if alive else "interrupted")


def test_stale_heartbeat_alone_does_not_prove_process_death(tmp_path, monkeypatch):
    record_job(tmp_path)
    monkeypatch.setattr("evals.job_status.os.kill", lambda pid, signal: None)
    monkeypatch.setattr("evals.job_status.process_details", lambda pid: None)
    result = inspect_job(tmp_path, now=900)
    assert result["status"] == "running"
    assert result["heartbeat_stale"] is True


def test_permission_denial_does_not_claim_process_death(tmp_path, monkeypatch):
    record_job(tmp_path)

    def denied(pid, signal):
        raise PermissionError

    monkeypatch.setattr("evals.job_status.os.kill", denied)
    monkeypatch.setattr("evals.job_status.process_details", lambda pid: None)
    result = inspect_job(tmp_path, now=140)
    assert result["status"] == "running"
    assert result["process_alive"] is None


def test_non_posix_inspection_never_sends_a_termination_signal(tmp_path, monkeypatch):
    record_job(tmp_path)

    def unexpected(pid, signal):
        raise AssertionError("Non-POSIX os.kill is not a safe liveness probe")

    monkeypatch.setattr("evals.job_status.os", SimpleNamespace(name="nt", kill=unexpected))
    monkeypatch.setattr("evals.job_status.process_details", lambda pid: None)
    result = inspect_job(tmp_path, now=140)
    assert result["status"] == "running"
    assert result["process_alive"] is None


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_finished_job_retains_recorded_verdict(tmp_path, monkeypatch, status):
    record_job(tmp_path, status=status)

    def unexpected(pid, signal):
        raise AssertionError("A finished job needs no process probe")

    monkeypatch.setattr("evals.job_status.os.kill", unexpected)
    assert inspect_job(tmp_path, now=900)["status"] == status


@pytest.mark.parametrize("pid", [None, True, 0, -1, "12345"])
def test_invalid_pid_never_signals_an_unintended_process(tmp_path, pid):
    record_job(tmp_path, pid=pid)
    with pytest.raises(ValueError, match="process id"):
        inspect_job(tmp_path)


def test_historical_job_without_identity_remains_inspectable(tmp_path, monkeypatch):
    record_job(tmp_path, process_identity=None)
    monkeypatch.setattr("evals.job_status.os.kill", lambda pid, signal: None)
    monkeypatch.setattr(
        "evals.job_status.process_details", lambda pid: {"identity": "boot:200", "state": "S"}
    )
    assert inspect_job(tmp_path, now=140)["status"] == "running"


def test_current_process_has_stable_linux_identity():
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux process identity is unavailable")
    first = process_details(os.getpid())
    second = process_details(os.getpid())
    assert first is not None
    assert first["identity"] == second["identity"]


def test_cli_reports_failure_without_mutation(tmp_path, monkeypatch, capsys):
    record_job(tmp_path, status="failed")
    monkeypatch.setattr("sys.argv", ["job_status", str(tmp_path)])
    assert main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
