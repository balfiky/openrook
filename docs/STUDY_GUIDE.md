# OpenRook Study Guide

This guide is the repo-local entry point for reviewers. Older external study
guides are historical; the source tree is the reference for current behavior.

**What you are reading.** OpenRook is a hybrid cognitive runtime for LLM agents
— engineered scaffolding that holds memory, identity, self-evolution,
skills, and affective state outside the model. Reference class: cognitive
runtime / agent memory (MemGPT, Letta, mem0, LangGraph-with-persistence).
Differentiator within that class: tracks relational continuity (rupture,
repair, commitments) and identity continuity (constitution, beliefs,
drives, self-traits) alongside conventional memory.

Use these documents in order:

1. [DEVELOPMENT_SUMMARY.md](DEVELOPMENT_SUMMARY.md) for major changes and completed
   behavioral validation, then [OVERVIEW.md](OVERVIEW.md) for the thesis and scope.
2. [ARCHITECTURE.md](ARCHITECTURE.md) for the runtime, memory model, and the
   self-evolution mechanics (constitution, metabolism, open questions,
   ask-user surfacing, trigger-time skills, skill→life migration).
3. [DEPLOYMENT_AND_ADMIN.md](DEPLOYMENT_AND_ADMIN.md) for setup, operator
   surfaces, the admin endpoint reference, and known gaps.
4. [UAT.md](UAT.md) for the live test suite (`make uat`,
   `make uat-comprehensive`) and the per-file coverage matrix.
5. [research/PAPER_DRAFT.md](research/PAPER_DRAFT.md) for the longer research framing.

For the latest behavioral evidence, start with the
[v4 closeout](plans/behavioral_validation_v4.md#completed-confirmation-and-closeout):
28 subjects, independently reconstructed retained-choice differences, verified
goal revisions, and historical state replay. Read the [review corrections](REVIEW_FIXES.md)
for the runtime-computed assistance, same-process manager restart, missing prompt
checks in historical replay, and unrun live raw-experience comparison.
Its measured operational acceptance is distinct
from a human-likeness or subjective-emotion claim.

For the earlier structural checks, use the Phase 11 and Phase 12 relationship
scenarios in `evals/scenarios.py`, the ablation priors in
`evals/ablation_hypotheses.py`, and the checked-in ablation summary under
`reports/ablation/summary.json`. For end-to-end behavior verification
against a real LLM, run `make uat-comprehensive` for a single PASS/FAIL
signal across all 67 UAT tests.
