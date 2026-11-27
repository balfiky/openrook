"""One frozen-contract gate for generated, edited and recovery expressions."""
from __future__ import annotations

from dataclasses import dataclass, replace

from core.dignity import (
    FIXED_LIMITS, MODEL, ExpressionReview, as_remainder, excisable_choice_issues, excise_choice_sentences,
    excise_ineligible_spans, review_response,
)
from core.dual_process.generator import GenerationResult
from core.review import FIXED_LIMIT_CODES, NOT_DUE


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

    Every candidate gets the structural checks. Every candidate those checks do
    not rule out is also reviewed by the model against the frozen decision:
    that review is required work of the turn, not something a draft has to earn
    by containing particular words. When it cannot be obtained the verdict is
    unknown, which never withholds a candidate.

    Editing only proposes another candidate. Neither editing nor a protected
    generation can select text directly or change the committed decision. What
    is left after a cut is reviewed as a reply of its own: when the decision
    declined something and that review gave no verdict, nothing shows the
    remainder still makes the decline recognisable, so it is not preferred to
    the text it was cut from.

    A reply that crosses a fixed limit is never delivered. Before a turn ends
    without a reply only because the reviewer reported one, the best such
    candidate gets one independent confirmation review (:meth:`confirm`).
    """

    def __init__(self, context, debug, review_due=None):
        self.context = dict(context)
        self.debug = debug
        # None: every candidate is reviewed. Research passes its own condition
        # and keeps its bounded call contract (see ``CognitivePipeline._check_dignity``).
        self.review_due = review_due
        decision = self.context.get("decision")
        self._declines = bool(decision is not None and decision.declined)
        self.candidates: list[ReviewedExpression] = []
        self._cache = {}
        self._repairs = set()
        self._confirmed = False
        self._doubted = {}     # text -> fixed-limit codes the confirmation did not report for it
        self._restated = {}    # text -> fixed-limit findings the confirmation made itself

    def _observed_outputs(self):
        trace = self.context.get("tool_trace")
        return [str(result.output or "") for result in getattr(trace, "executed_results", None) or []
                if result.success]

    def _review(self, result, *, local_only=False, source="", record=True, path="draft"):
        local = review_response(result.response, **self.context, use_llm=False, check_output_blocks=False,
            raw_response=result.raw_response, private_instructions=result.system_prompt, excision_source=source)
        # Identical text with identical local findings cannot earn a different
        # verdict by taking a different path, and is not reviewed twice.
        key = result.response, local.issues
        cached = self._cache.get(key)
        retained = [issue for c in self.candidates for issue in c.review.issues
                    if c.result.response == result.response or (
                        c.result.response == source and issue.claim and issue.claim in result.response)]
        if cached:
            verdict, source_attempt = cached
        else:
            reviewable = (not local_only and not local.unusable and not any(i.ineligible for i in retained)
                          and (self.review_due is None or self.review_due()))
            # A delimited output block that no receipt returned stands against
            # the draft unless the review runs to a verdict and reports nothing.
            verdict = review_response(result.response, **self.context, use_llm=reviewable,
                raw_response=result.raw_response, private_instructions=result.system_prompt, excision_source=source)
            if not verdict.provenance:
                # No model was asked about this text. Said outright, so that it is
                # ordered as what it is: a candidate nobody reviewed.
                verdict = replace(verdict, provenance=NOT_DUE)
            source_attempt = len(self.debug.expression_reviews) + 1
            if not local_only:
                self._cache[key] = verdict, source_attempt
        # A verdict reused for identical text is read for this candidate's own
        # origin: a draft the writer produced whole, or what a cut left behind.
        issues = as_remainder(dict.fromkeys((*verdict.issues, *local.issues, *retained)), source)
        if issues:
            verdict = replace(verdict, status="violation", issues=issues)
        verdict = self._settled(result.response, verdict)
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
        remainder, removed = edit(candidate.result.response, candidate.review, self._observed_outputs())
        if not removed:
            return
        evidence = {"applied": False, "removed": removed, "source_attempt": candidate.attempt}
        self.debug.expression_excision = evidence
        if not remainder or remainder == candidate.result.response:
            evidence["reason"] = "empty remainder or unverifiable spans; generate fresh expression"
            return
        # A remainder is a new candidate: it is checked and reviewed like any
        # draft, and what was removed cannot speak for it.
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

    def rank(self, candidate):
        """A candidate's place in the order of delivery; lower is better (``ExpressionReview.rank``).

        One exception to the review's own rank: a remainder of a decision that
        declined something, whose review gave no verdict, goes directly after
        the candidate it was cut from, when that one can be delivered. An
        unavailable review does not waive the decline the cut may have removed.
        """
        if candidate.source and self._declines and candidate.review.semantic_status not in ("pass", "violation"):
            origin = next((c for c in self.candidates if c.result.response == candidate.source and c.eligible), None)
            if origin is not None and origin is not candidate:
                return (*self.rank(origin)[:-1], 1)
        return (*candidate.review.rank, 0)

    def best(self):
        return min(self.eligible or self.candidates, key=self.rank)

    def confirm(self):
        """One independent second look, before a turn ends without a reply over a reviewer's fixed-limit finding.

        The fixed limits are absolute, and a single reviewer's answer is
        fallible. When nothing can be delivered, and a candidate is held back
        solely by fixed-limit findings of the reviewer (no structural finding
        rules it out), the best-ranked such candidate is reviewed once more: a
        fresh call that is asked about the fixed limits alone and is not shown
        the earlier findings.

        * A limit the confirmation reports again, on a verbatim span of the
          same text, stands: the candidate stays undeliverable. So does any
          other fixed limit the confirmation reports.
        * A finding the confirmation does not make is marked ``unconfirmed``:
          it becomes a high finding that no longer rules delivery out, and the
          candidate can be delivered as ``delivered_degraded``.
        * A confirmation that could not be obtained settles nothing: the
          findings stand.

        At most once per turn; the call has its own place in the review
        allowance and shares the turn's review time. Both reviews are recorded.
        """
        if self._confirmed or self.eligible:
            return None
        if self.review_due is not None and not self.review_due():
            return None

        def held_by_the_reviewer(candidate):
            barring = [issue for issue in candidate.review.issues if issue.ineligible]
            return bool(candidate.result.response.strip()) and bool(barring) and all(
                issue.source == MODEL and issue.code in FIXED_LIMITS for issue in barring)

        held = [candidate for candidate in self.candidates if held_by_the_reviewer(candidate)]
        if not held:
            return None
        self._confirmed = True
        candidate = min(held, key=self.rank)
        text = candidate.result.response
        second = review_response(text, **self.context, use_llm=True, check_output_blocks=False,
            codes=FIXED_LIMIT_CODES, confirmation=True, raw_response=candidate.result.raw_response,
            private_instructions=candidate.result.system_prompt)
        answered = second.provenance == "model"
        again = tuple(issue for issue in second.issues if issue.source == MODEL and issue.code in FIXED_LIMITS)
        doubted = sorted({issue.code for issue in candidate.review.issues if issue.ineligible}
                         - {issue.code for issue in again}) if answered else []
        self.debug.expression_reviews.append({"attempt": len(self.debug.expression_reviews) + 1, "draft": text,
            "path": "confirmation", **second.to_dict(), "confirms_attempt": candidate.attempt,
            "confirmed": sorted({issue.code for issue in again}), "unconfirmed": doubted})
        if not answered:
            return None
        self._doubted[text], self._restated[text] = tuple(doubted), again
        # The same text is the same reply wherever it stands in the pool.
        self.candidates = [replace(item, review=self._settled(text, item.review))
                           if item.result.response == text else item for item in self.candidates]
        return next(item for item in self.candidates if item.attempt == candidate.attempt)

    def _settled(self, text, review):
        """A review of ``text`` as it stands after the turn's confirmation, if that was about this text."""
        if text not in self._doubted:
            return review
        doubted = self._doubted[text]
        issues = tuple(replace(issue, unconfirmed=True)
                       if issue.ineligible and issue.source == MODEL and issue.code in doubted else issue
                       for issue in review.issues)
        # A fixed limit the confirmation found itself stands like any other.
        issues = tuple(dict.fromkeys((*issues, *self._restated[text])))
        return replace(review, status="violation" if issues else review.status, issues=issues)

    def select(self):
        """The sole delivery exit; every selected path passes the same gate."""
        for candidate in sorted(self.eligible, key=self.rank):
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
