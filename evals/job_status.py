"""Inspect a local study process without changing its recorded evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def process_details(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return {"identity": f"{boot}:{int(fields[19])}", "state": fields[0]}
    except (OSError, IndexError, ValueError):
        return None


def inspect_job(output: str | Path, *, now: float | None = None) -> dict:
    recorded = json.loads((Path(output) / "job.json").read_text())
    observed_at = time.time() if now is None else now
    heartbeat = recorded.get("heartbeat_at", recorded["started_at"])
    status = {
        **recorded,
        "recorded_status": recorded["status"],
        "observed_at": observed_at,
        "heartbeat_age_seconds": max(0, observed_at - heartbeat),
        "process_alive": None,
    }
    if recorded["status"] != "running":
        return status
    pid = recorded.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("Running job has no valid positive process id")
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            status["process_alive"] = True
        except ProcessLookupError:
            status["process_alive"] = False
        except PermissionError:
            pass
    details = process_details(pid)
    if details is not None:
        if details["state"] in {"Z", "X"}:
            status["process_alive"] = False
        elif recorded.get("process_identity") is not None:
            status["process_alive"] = details["identity"] == recorded["process_identity"]
    if status["process_alive"] is False:
        status["status"] = "interrupted"
        status["interruption_reason"] = "The recorded execution process is no longer alive"
    status["heartbeat_stale"] = status["heartbeat_age_seconds"] > 90
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Local study job directory")
    arguments = parser.parse_args()
    status = inspect_job(arguments.output)
    print(json.dumps(status, sort_keys=True))
    return int(status["status"] not in {"running", "completed"})


if __name__ == "__main__":
    raise SystemExit(main())
