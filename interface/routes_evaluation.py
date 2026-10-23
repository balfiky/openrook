"""Authenticated, stateless evaluation using the runtime's configured backend."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from interface import api as api_mod
from openrook_tools.roleplay_eval.api import Redactor
from openrook_tools.roleplay_eval.judge import (
    judge_prompt, judge_diagnostic, parse_floor_judgment, parse_judgment, repair_prompts,
)


router = APIRouter(dependencies=[Depends(api_mod._require_bearer)])


class EvaluationJudgeRequest(BaseModel):
    evidence: dict

    @field_validator("evidence")
    @classmethod
    def bounded_evidence(cls, value: dict) -> dict:
        if len(json.dumps(value, ensure_ascii=False)) > 100_000:
            raise ValueError("Evaluation evidence exceeds the request limit")
        return value


@router.post("/admin/evaluation/judge")
async def evaluation_judge(req: EvaluationJudgeRequest) -> dict:
    """A fixed-rubric review with one repair; no character state is touched."""
    config = api_mod._load_runtime_config()

    def generate() -> dict:
        backend = api_mod.create_llm_backend(config)
        set_limit = getattr(backend, "set_output_token_limit", None)
        if set_limit is not None:
            set_limit(4096)
        redactor = Redactor(config.llm_api_key, config.api_key, config.telegram_token,
                            config.llm_model, config.llm_base_url)
        failures, floor_fallback = [], None
        system, user = judge_prompt(req.evidence), json.dumps(req.evidence, ensure_ascii=False)
        try:
            for attempt in range(1, 3):
                try:
                    raw = backend.generate(system, user)
                except Exception:
                    if not failures:
                        raise
                    if floor_fallback is not None:
                        return {"judgment": floor_fallback, "attempts": attempt, "failed_attempts": failures,
                                "independence_status": "error", "independence_error": "judge_unavailable"}
                    return {"error": "judge_backend_failure", "attempts": attempt,
                            "raw_judge_text": failures[-1]["raw_judge_text"], "failed_attempts": failures}
                try:
                    judgment = redactor.clean(parse_judgment(raw, evidence=req.evidence))
                except (ValueError, TypeError) as error:
                    try:
                        floor_fallback = redactor.clean(parse_floor_judgment(raw, evidence=req.evidence))
                    except (ValueError, TypeError):
                        pass
                    diagnostic = judge_diagnostic(raw, redactor)
                    validation_error = str(error) if isinstance(error, ValueError) else "Invalid judge response structure"
                    failures.append({"attempt": attempt, "raw_judge_text": diagnostic,
                                     "validation_error": validation_error})
                    system, user = repair_prompts(req.evidence, diagnostic, validation_error)
                else:
                    return {"judgment": judgment, "attempts": attempt, "failed_attempts": failures}
            if floor_fallback is not None:
                return {"judgment": floor_fallback, "attempts": 2, "failed_attempts": failures,
                        "independence_status": "error", "independence_error": "invalid_independence"}
            return {"error": "invalid_judge_json", "attempts": 2,
                    "raw_judge_text": diagnostic, "failed_attempts": failures}
        finally:
            close = getattr(backend, "close", None)
            if close:
                close()

    try:
        result = await asyncio.to_thread(generate)
    except Exception:
        # Provider exceptions can include credentials and destinations.
        raise HTTPException(status_code=503, detail="Evaluation judge is unavailable") from None
    return result
