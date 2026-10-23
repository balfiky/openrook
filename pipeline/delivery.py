"""One frozen-contract gate for generated, edited and recovery expressions."""
from __future__ import annotations

from dataclasses import dataclass, replace

from core.dignity import (
    ExpressionReview, excisable_choice_issues, excise_choice_sentences, excise_ineligible_spans,
    review_response,
)
from core.dual_process.generator import GenerationResult
from core.inference import remaining_calls


@dataclass(frozen=True)
class ReviewedExpression:
    result: GenerationResult
    review: ExpressionReview
    attempt: int
    source: str = ""
    excision: dict | None = None
    path: str = "draft"

    @property
    def eligible(self):
        return bool(self.result.response.strip()) and not self.review.unusable


class DeliveryGate:
    """Review every entry and retain all known findings on identical text.

    Editing only proposes another candidate. Neither editing nor a protected
    generation can select text directly or change the committed decision.
    """

    def __init__(self, context, debug, needs_semantic):
        self.context = dict(context)
        self.debug = debug
        self.needs_semantic = needs_semantic
        self.candidates: list[ReviewedExpression] = []
        self._cache = {}
        self._repairs = set()

    def _review(self, result, *, local_only=False, source="", record=True, path="draft"):
        local = review_response(result.response, **self.context, use_llm=False,
            raw_response=result.raw_response, private_instructions=result.system_prompt, excision_source=source)
        # Local findings bind raw artifacts and editing provenance. Identical
        # text with identical evidence cannot earn a different semantic verdict
        # by taking a different path or exhausting the semantic-review budget.
        key = result.response, local.issues
        cached = self._cache.get(key)
        retained = [issue for c in self.candidates for issue in c.review.issues
                    if c.result.response == result.response or (
                        c.result.response == source and issue.claim and issue.claim in result.response)]
        if cached:
            verdict, source_attempt = cached
        else:
            verdict = local
            if (not local_only and not local.unusable and not any(i.ineligible for i in retained)
                    and remaining_calls() > 0 and self.needs_semantic(result.response)):
                verdict = review_response(result.response, **self.context, use_llm=True,
                    raw_response=result.raw_response, private_instructions=result.system_prompt, excision_source=source)
            source_attempt = len(self.debug.expression_reviews) + 1
            self._cache[key] = verdict, source_attempt
        issues = tuple(dict.fromkeys((*verdict.issues, *local.issues, *retained)))
        if issues:
            verdict = replace(verdict, status="violation", issues=issues)
        if record:
            self.debug.expression_reviews.append({"attempt": len(self.debug.expression_reviews) + 1,
                "draft": result.response, "path": path, **verdict.to_dict(),
                **({"excision_source": source} if source else {}),
                **({"reused_from_attempt": source_attempt} if cached else {})})
        return verdict

    def add(self, result, *, local_only=False, source="", path="draft", excision=None):
        verdict = self._review(result, local_only=local_only, source=source, path=path)
        candidate = ReviewedExpression(result, verdict, len(self.debug.expression_reviews), source, excision, path)
        self.candidates.append(candidate)
        if verdict.unusable:
            self.repair(candidate, local_only=local_only)
        return candidate

    def repair(self, candidate, *, choice=False, local_only=False):
        key = candidate.attempt, choice
        if key in self._repairs:
            return
        self._repairs.add(key)
        edit = excise_choice_sentences if choice else excise_ineligible_spans
        remainder, removed = edit(candidate.result.response, candidate.review)
        if not removed:
            return
        evidence = {"applied": False, "removed": removed, "source_attempt": candidate.attempt}
        self.debug.expression_excision = evidence
        if not remainder or remainder == candidate.result.response:
            evidence["reason"] = "empty remainder or unverifiable spans; generate fresh expression"
            return
        # Sanitization/excision cannot erase evidence of raw delivery artifacts.
        result = replace(candidate.result, response=remainder)
        repaired = self.add(result, local_only=local_only, source=candidate.result.response,
                            path="excision", excision=evidence)
        evidence["local_status"] = repaired.review.status
        evidence["applied"] = repaired.eligible
        if choice and repaired.eligible and excisable_choice_issues(repaired.review):
            self.repair(repaired, choice=True, local_only=local_only)

    @property
    def eligible(self):
        return [c for c in self.candidates if c.eligible]

    def best(self):
        return min(self.eligible or self.candidates, key=lambda c: c.review.rank)

    def select(self):
        """The sole delivery exit; every selected path passes the same gate."""
        for candidate in sorted(self.eligible, key=lambda c: c.review.rank):
            verdict = self._review(candidate.result, local_only=True, source=candidate.source, record=False)
            if not verdict.unusable:
                if candidate.excision:
                    self.debug.expression_excision = candidate.excision
                self.debug.expression_delivery = {"attempt": candidate.attempt, "path": candidate.path,
                    "source_attempt": candidate.excision.get("source_attempt") if candidate.excision else None,
                    "review": verdict.to_dict()}
                return replace(candidate, review=verdict)
        self.debug.expression_delivery = {"attempt": None, "reason": "no eligible reviewed expression"}
        return None
