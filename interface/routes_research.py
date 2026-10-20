"""Passive research observations; these routes never initialize a subject."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from interface import api as api_mod
from runtime.research_store import ResearchStore


router = APIRouter(prefix="/v1/research", dependencies=[Depends(api_mod._require_bearer)])


def _store() -> ResearchStore:
    manager = api_mod._session_manager
    config = manager.config if manager is not None else api_mod._load_runtime_config()
    store = ResearchStore(config, initialize=False)
    if not store.path.is_file():
        raise HTTPException(status_code=404, detail="No recorded research run")
    return store


@router.get("/manifest")
async def manifest() -> dict:
    return _store().get("run", "manifest") or {}


@router.get("/state")
async def state() -> dict:
    store = _store()
    return {
        "agent": store.get("agent", "state"),
        "goals": store.list("goal"),
        "goal_executions": store.list("goal_execution"),
        "completion_assessments": store.list("completion_assessment"),
        "predictions": store.list("prediction"),
        "procedures": store.list("procedure"),
        "learning_estimates": store.list("learning_state"),
        "attention": store.list("attention"),
        "dispositions": store.list("disposition"),
        "learning": store.list("learning"),
        "actions": store.list("action"),
        "deliveries": store.list("delivery"),
    }


@router.get("/events")
async def events(
    after: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=1000)
) -> dict:
    return {"events": _store().observations(after=after, limit=limit)}
