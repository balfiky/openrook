"""Scoped clocks and identifiers for recorded transition replay."""

from __future__ import annotations

import time as system_time
import uuid as system_uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime as SystemDateTime, timezone
from typing import Any


_recording: ContextVar[dict[str, Any] | None] = ContextVar("openrook_transition_recording", default=None)


class Clock:
    def time(self) -> float:
        frame = _recording.get()
        if frame is not None and frame["replay"]:
            index = frame["time_index"]
            if index >= len(frame["times"]):
                raise RuntimeError("Replay exhausted captured clock readings")
            frame["time_index"] += 1
            return frame["times"][index]
        value = system_time.time()
        if frame is not None:
            frame["times"].append(value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(system_time, name)


class Identifiers:
    def uuid4(self) -> system_uuid.UUID:
        frame = _recording.get()
        if frame is not None and frame["replay"]:
            index = frame["id_index"]
            if index >= len(frame["ids"]):
                raise RuntimeError("Replay exhausted captured identifiers")
            frame["id_index"] += 1
            return system_uuid.UUID(frame["ids"][index])
        value = system_uuid.uuid4()
        if frame is not None:
            frame["ids"].append(str(value))
        return value


clock = Clock()
identifiers = Identifiers()


class datetime(SystemDateTime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(clock.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.fromtimestamp(clock.time(), timezone.utc).replace(tzinfo=None)


@contextmanager
def capture(replay: dict[str, Any] | None = None):
    existing = _recording.get()
    if existing is not None:
        yield None
        return
    frame = {
        "times": list(replay["times"]) if replay else [],
        "ids": list(replay["ids"]) if replay else [],
        "replay": replay is not None,
        "time_index": 0,
        "id_index": 0,
    }
    token = _recording.set(frame)
    try:
        yield frame
        if replay is not None and (
            frame["time_index"] != len(frame["times"]) or frame["id_index"] != len(frame["ids"])
        ):
            raise RuntimeError("Replay did not consume the complete recorded transition")
    finally:
        _recording.reset(token)
