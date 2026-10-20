"""Reproducible OpenRook documentation diagrams.

Each builder returns a Canvas using the shared diagram toolkit. Regenerate the
SVG sources and PNG previews with ``python3 nur_tools/render_diagram_pngs.py``.
The hand-authored feature overview is handled by the renderer separately.
"""

from __future__ import annotations

from nur_tools.diagram_toolkit import Canvas, THEME

INK_SOFT = THEME["ink_soft"]


def _legend(c: Canvas, y: int, items: list[tuple[str, str]]) -> None:
    """Keep a compact legend inside a dedicated, full-width footer frame."""
    c.group(40, y, c.width - 80, 68, "Legend")
    width = (c.width - 112) / len(items)
    for index, item in enumerate(items):
        c.legend(56 + index * width, y + 48, [item])


# ---------------------------------------------------------------------------
# Conversation hosts, shared cognition, and the optional research owner
# ---------------------------------------------------------------------------


def build_runtime_architecture() -> Canvas:
    c = Canvas(
        1600, 640,
        "OpenRook — Runtime Architecture",
        "Conversation hosts share a session core. Research mode adds one persistent subject owner.",
    )
    c.group(40, 116, 1520, 164, "Channels → hosts → shared cognition → model", "process")
    c.group(40, 324, 1520, 184, "Tools and durable state", "storage")

    # Draw connections first; each store has its own attachment and route.
    for left, right in ((328, 376), (676, 724), (1084, 1132)):
        c.arrow((left, 204), (right, 204))
    c.connector([(840, 256), (840, 284), (280, 284), (280, 364)], dashed=True)
    c.connector([(872, 256), (872, 296), (600, 296), (600, 364)])
    c.arrow((904, 256), (904, 364))
    c.connector([(936, 256), (936, 312), (1376, 312), (1376, 364)], dashed=True)

    c.tile(56, 152, 272, 104, "Client channels", [
        "Terminal + optional Telegram", "Browser chat / admin UI", "HTTP API clients",
    ], color="client", title_size=17)
    c.tile(376, 152, 300, 104, "Runtime hosts", [
        "openrook: console + Telegram", "openrook-web: FastAPI + WebSocket", "HTTP authentication at the host",
    ], title_size=17)
    c.tile(724, 152, 360, 104, "Session + cognition core", [
        "SessionManager → UserSession", "per-user locking / backpressure", "CognitivePipeline processes each turn",
    ], color="attention", title_size=17)
    c.tile(1132, 152, 412, 104, "Runtime LLM backend", [
        "auto · provider · openai_compatible", "codex · minimax", "Ollama uses openai_compatible",
    ], color="external", title_size=17)

    c.tile(56, 364, 320, 120, "Tool executor", [
        "Internal skill tools remain available", "External tools: tools_enabled", "Shell: separate explicit opt-in", "Native + regex selection; same policy",
    ], color="danger")
    c.cylinder(424, 364, 352, 120, "Per-user storage", [
        "nur.db: memory, profiles, relationships", "sessions/<chat_id>.json", "sessions/<chat_id>.history.json",
    ])
    c.cylinder(824, 364, 320, 120, "Shared identity storage", [
        "shared/self_model.db", "shared/life_history.db", "retained across user deletion",
    ])
    c.cylinder(1192, 364, 352, 120, "Research subject owner", [
        "single owner lock + background loop", "shared/agent_runtime.db", "manifest, goals, receipts, checkpoints",
    ])
    c.note(56, 536, "Configuration: runtime_config.yaml · Identity: $OPENROOK_CONFIG_DIR/soul.yaml", size=12, fill=INK_SOFT)
    _legend(c, 552, [("client", "channels"), ("process", "runtime"), ("attention", "cognition"), ("external", "real LLM"), ("storage", "durable state"), ("danger", "tool policy")])
    return c


# ---------------------------------------------------------------------------
# Per-turn path and separate session-end consolidation
# ---------------------------------------------------------------------------


def build_single_turn_flow() -> Canvas:
    c = Canvas(
        1600, 648,
        "Single-Turn Cognitive Flow",
        "Conversation-mode path. LLM cost depends on appraisal, deliberation, tools, checking, and retries; there is no fixed aggregate ceiling.",
    )
    columns = [(48, 280), (356, 280), (664, 252), (944, 292), (1264, 288)]
    for index in range(4):
        c.arrow((columns[index][0] + columns[index][1], 272), (columns[index + 1][0], 272))
    c.connector([(1408, 408), (1408, 444), (1040, 444), (1040, 472)], dashed=True)

    c.tile(48, 136, 280, 272, "1. State + context", [
        "Decay + anticipation: no LLM", "Contagion / person baseline shift", "", "Social appraisal", "model-backed when enabled", "rule fallback when unavailable", "", "Event classification + PSI update", "short-term and spike memory writes", "", "Retrieve memories, profiles,", "relationships and open loops", "Detect contradictions",
    ], title_size=17, body_size=12)
    c.tile(356, 136, 280, 272, "2. Deliberation + tools", [
        "Inner dialogue: conditional", "calls vary with rounds + parse retries", "high arousal can skip the slow path", "", "Tool selection + execution", "native or regex selection", "additional calls / rounds vary", "internal skill tools remain available", "external tools require opt-in", "", "Defense shaping: no LLM", "injects a prompt instruction",
    ], color="attention", title_size=17, body_size=12)
    c.tile(664, 188, 252, 168, "3. Generate", [
        "Build prompt from current", "state, memories, profiles,", "relationship context and", "deliberation candidate", "", "One generation call", "on the ordinary turn path",
    ], color="external", title_size=17, body_size=12)
    c.tile(944, 136, 292, 272, "4. Self-check", [
        "Rule checks: no LLM", "tone / confidence / energy fit", "", "Select model checker when:", "event intensity > 0.85,", "contradictions, or", "inner-dialogue deadlock", "optional additional call", "", "Regenerate when check fails", "additional generation call", "grounding checks may reject output",
    ], title_size=17, body_size=12)
    c.tile(1264, 136, 288, 272, "5. Per-turn updates", [
        "Memory and relationship writes", "trust / bonding / maturity deltas", "", "Drain energy", "per-message and spike drain", "", "Assemble DebugState", "modulators, dialogue, defense,", "unresolved items and memories", "", "Return response + debug state", "to the calling channel",
    ], color="neutral", title_size=17, body_size=12)
    c.tile(48, 472, 1504, 80, "Separate path: end_session() consolidation", [
        "Short-term → long-term digestion happens at session end; model digestion or optional autobiographical ingestion can add inference calls.",
    ], color="storage", title_size=16, body_size=13)
    _legend(c, 564, [("process", "state / checks"), ("attention", "conditional work"), ("external", "model generation"), ("neutral", "per-turn updates"), ("storage", "session-end work")])
    return c


# ---------------------------------------------------------------------------
# User deletion and retained shared state
# ---------------------------------------------------------------------------


def build_memory_persistence() -> Canvas:
    c = Canvas(
        1600, 680,
        "Memory and Persistence",
        "Ordinary-mode user deletion removes that user's directory. Shared identity and research records require a separate stopped-owner procedure.",
    )
    c.group(40, 120, 936, 192, "Retained: data/shared/", "storage")
    c.group(40, 340, 936, 204, "Deleted: data/<platform>_<user_id>/", "danger")
    c.group(1000, 120, 560, 424, "Deletion behavior", "danger")

    c.cylinder(60, 164, 280, 120, "self_model.db", [
        "assistant observations", "extracted traits", "defense events",
    ])
    c.cylinder(360, 164, 280, 120, "life_history.db", [
        "formative experiences", "belief / drive revisions", "shared identity history",
    ])
    c.cylinder(660, 164, 296, 120, "agent_runtime.db", [
        "research manifest + state", "goals, actions and outcomes", "checkpoints + learning",
    ])
    c.cylinder(60, 384, 416, 136, "nur.db", [
        "memories + semantic memories", "relationship events + open loops", "person / topic observations + traits",
    ])
    c.tile(504, 384, 452, 136, "Session files + transcripts", [
        "sessions/<chat_id>.json: engine snapshot", "sessions/<chat_id>.history.json: hot transcript", "engine_state.json: legacy snapshot", "", "All removed with the per-user directory",
    ], color="blank", title_size=16)

    c.tile(1024, 164, 512, 112, "Ordinary-mode deletion", [
        "DELETE /v1/users/{platform}/{user_id}", "or admin typed DELETE confirmation", "evicts sessions and removes the per-user directory", "best-effort row counts may be null",
    ], color="danger", title_size=16)
    c.tile(1024, 296, 512, 112, "Retained beyond a user reset", [
        "All three shared databases and replay_initial/", "study outputs, replay captures and workspace files", "Shared evidence may still contain user-derived material", "Global configuration is retained too",
    ], color="storage", title_size=16)
    c.tile(1024, 428, 512, 92, "Research mode", [
        "HTTP DELETE and mutating /admin calls return 409", "Stop the owning process before backup or deletion", "See PRIVACY.md for complete subject reset",
    ], color="attention", title_size=16)
    c.tile(40, 576, 1520, 80, "Global configuration — outside the per-user deletion boundary", [
        "runtime_config.yaml · config/soul.yaml or $OPENROOK_CONFIG_DIR/soul.yaml · read at startup; ordinary-mode admin writes are separate",
    ], color="neutral", title_size=16, body_size=13)
    return c


# ---------------------------------------------------------------------------
# Authentication and shared tool executor policy
# ---------------------------------------------------------------------------


def build_auth_tool_safety() -> Canvas:
    c = Canvas(
        1600, 720,
        "Auth and Tool-Safety Boundaries",
        "HTTP authentication and tool availability are separate controls. Disabling external tools still leaves internal skill-registry tools available.",
    )
    c.group(40, 120, 600, 476, "HTTP request gate", "process")
    c.group(672, 120, 888, 476, "Tool selection → common executor and policy", "danger")

    c.arrow((340, 252), (340, 292))
    c.arrow((1108, 252), (1108, 292))
    c.connector([(1108, 392), (1108, 420), (892, 420), (892, 452)])
    c.connector([(1140, 392), (1140, 420), (1320, 420), (1320, 452)])

    c.tile(64, 164, 552, 88, "Protected data + mutation endpoints", [
        "/chat, /debug, /config, /admin/* APIs, /v1/*, /ws", "configured api_key: require valid Bearer token",
    ], title_size=16)
    c.tile(64, 292, 552, 100, "Authentication result", [
        "No configured key: authentication is open", "Configured key: invalid / missing token → 401", "Valid token → endpoint's own authorization and checks",
    ], title_size=16)
    c.tile(64, 452, 552, 100, "Public shells + health", [
        "GET /, /admin HTML and /settings HTML", "/v1/health and /v1/ready", "The underlying administrative APIs enforce authentication",
    ], color="client", title_size=16)

    c.tile(696, 164, 840, 88, "Selection routes", [
        "Native tool calls when wired; regex intent fallback otherwise", "Both routes enter the same executor and policy boundary",
    ], color="neutral", title_size=16)
    c.tile(696, 292, 840, 100, "ToolExecutor + registered capabilities", [
        "tools_enabled=false: external tools off; internal skill tools remain (including writes)", "tools_enabled=true: external tools are registered; native runner may be wired", "Every selected tool must be registered and pass its execution policy",
    ], color="attention", title_size=16)
    c.tile(696, 452, 392, 120, "Filesystem / external tools", [
        "External capabilities require tools_enabled", "Filesystem paths stay in tools_workspace", "Out-of-workspace paths are refused", "Network tools have their own policies",
    ], color="danger", title_size=15)
    c.tile(1120, 452, 416, 120, "shell.run_command", [
        "Also requires shell_tool_enabled", "Explicit opt-in; trusted callers only", "Configured allowlist and shell policy apply", "Shell remains off by default",
    ], color="danger", title_size=15)
    c.tile(40, 620, 1520, 76, "Tracked starter defaults", [
        "api_key='' · cors_origins=[] · tools_enabled=false · tools_workspace='' · shell_tool_enabled=false. Review deployment settings before exposing a host.",
    ], color="neutral", title_size=15, body_size=12)
    return c


# ---------------------------------------------------------------------------
# Structural evaluations and report provenance
# ---------------------------------------------------------------------------


def build_evaluation_harness() -> Canvas:
    c = Canvas(
        1600, 648,
        "Evaluation and Ablation Harness",
        "The checked-in structural reports use the real CognitivePipeline with an evaluation-only mock LLM backend (2026-05-16, git 3d3f869c).",
    )
    c.group(40, 120, 432, 348, "Inputs", "client")
    c.group(512, 120, 500, 348, "Execute + stamp", "process")
    c.group(1052, 120, 508, 348, "Artifacts", "storage")

    c.arrow((448, 216), (536, 216))
    c.arrow((988, 216), (1076, 216))
    c.arrow((762, 268), (762, 308))
    c.arrow((988, 376), (1076, 376))
    c.arrow((1306, 268), (1306, 308), dashed=True)

    c.tile(64, 164, 384, 104, "python3 -m evals.ablation", [
        "--backend mock --tag phase11", "--report-dir reports/ablation", "Live backend choices exist; these records used mock",
    ], color="client", title_size=17, body_size=11)
    c.tile(64, 308, 384, 136, "Scenario + feature variants", [
        "Baseline and individual feature removals", "no_life_history_context is included", "baseline_prompt_only: five recorded toggles off", "Same pipeline; no separate agent comparator", "Assertions grade structural effects",
    ], color="attention", title_size=16)
    c.tile(536, 164, 452, 104, "Fresh pipeline per scenario", [
        "InstrumentedBackend records calls and latency", "Run scripted turns; check strategy / state / memory", "Mock responses test structure, not live model quality",
    ], title_size=17)
    c.tile(536, 308, 452, 136, "Provenance + outcome labels", [
        "git SHA / branch / dirty flag; backend and model", "SHA-256 fingerprints from FINGERPRINT_FILES", "expected_failure / unexpected_failure", "no_effect / newly_passing", "File list maintained in evals/provenance.py",
    ], color="attention", title_size=16)
    c.tile(1076, 164, 460, 104, "Per-variant JSON", [
        "Turns, assertions, latency and LLM calls", "Provenance accompanies the run",
    ], color="storage", title_size=17)
    c.tile(1076, 308, 460, 136, "summary.json", [
        "Merged variant results + predictions", "Phase 11 baseline: 6/6", "No relationship memory: 4/6", "Historical baseline_prompt_only control: 4/6", "Both losses match the two expected failures",
    ], color="storage", title_size=17)
    c.tile(40, 500, 1520, 124, "Honest framing", [
        "Structural ablations do not grade wording quality or compare OpenRook against an independently built prompt-only agent.",
        "baseline_prompt_only is the recorded control with five historical feature toggles off; it does not establish that broader comparison.",
        "All four checked-in structural summaries used mock. Later real-model research studies are separate evidence (see RESEARCH_RUNTIME.md).",
        "Token / cost / retry fields stay null when the client did not extract them.",
    ], color="neutral", title_size=17, body_size=13)
    return c


# ---------------------------------------------------------------------------
# Bounded historical evidence, not the current research frontier
# ---------------------------------------------------------------------------


def build_component_claim_map() -> Canvas:
    c = Canvas(
        1600, 744,
        "Historical Phase 11 — Component Claim Map",
        "Structural mock-backend evidence from 2026-05-16 (git 3d3f869c). Claims are limited to this suite and the cited component behavior.",
    )
    tiles = [
        ("Relationship memory", "client", "LOAD-BEARING IN THIS SUITE", ["Removal breaks the two open-loop scenarios.", "Four of six pass; no unexpected failures."]),
        ("Semantic memory", "neutral", "NEGATIVE CONTROL", ["This suite does not exercise semantic retrieval.", "Removal has no effect on all six scenarios."]),
        ("Inner dialogue", "attention", "NOT FALSIFIABLE HERE", ["Dialogue does not move the strategy selector.", "Removal has no effect on all six scenarios."]),
        ("Defense mechanisms", "attention", "NOT FALSIFIABLE HERE", ["Prompt shaping affects wording, not graded here.", "Removal has no effect on all six scenarios."]),
        ("Self / other profiling", "storage", "ARCHITECTURAL CLAIM", ["ProfileStore covers person, topic and assistant self.", "This design alone is not a benchmark result."]),
        ("Emotional modulators", "process", "UNIT + INTEGRATION", ["Numeric decay, spikes, energy and context shifts.", "Tests establish mechanics, not experienced emotion."]),
    ]
    for index, (title, color, badge, lines) in enumerate(tiles):
        row, col = divmod(index, 3)
        x, y = 48 + col * 512, 132 + row * 216
        c.tile(x, y, 480, 192, title, color="blank", title_size=18)
        c.pill(x + 20, y + 48, 440, 32, badge, color=color)
        for line_index, line in enumerate(lines):
            c.note(x + 20, y + 116 + line_index * 22, line, size=12, fill=INK_SOFT)
    c.tile(48, 572, 1504, 76, "Later evidence is separate", [
        "For v4 closed-loop goal execution and grounded-appraisal studies, see RESEARCH_RUNTIME.md and docs/OVERVIEW.md; this historical map does not summarize them.",
    ], color="neutral", title_size=16, body_size=12)
    _legend(c, 664, [("client", "suite-specific effect"), ("attention", "not tested here"), ("neutral", "negative control"), ("storage", "design claim"), ("process", "mechanics tested")])
    return c


# ---------------------------------------------------------------------------
# Illustrative conversation, retained state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persistent research subject: bounded closed-loop execution
# ---------------------------------------------------------------------------


def build_research_runtime_autonomy() -> Canvas:
    c = Canvas(
        1600, 684,
        "Research Runtime — Persistent Subject Loop",
        "Opt-in closed-loop research mode. One owner coordinates model decisions, permitted actions, receipts, and durable state.",
    )
    c.group(40, 116, 1520, 384, "Single owner lock: shared/agent_runtime.owner.lock · serialized subject mutations", "attention")

    # A clockwise loop keeps every connection in the gaps between stages.
    top_x = [64, 456, 848, 1240]
    widths = [344, 344, 344, 296]
    for index in range(3):
        c.arrow((top_x[index] + widths[index], 216), (top_x[index + 1], 216))
    c.arrow((1388, 268), (1388, 324))
    for right, left in ((1240, 1192), (848, 800), (456, 408)):
        c.arrow((right, 384), (left, 384))
    c.connector([(64, 384), (48, 384), (48, 216), (64, 216)], dashed=True)

    c.tile(64, 164, 344, 104, "1. Wake + restore context", [
        "timer / registered wake condition", "resume only when budgets allow", "load observations, goals and state",
    ], title_size=16)
    c.tile(456, 164, 344, 104, "2. Deliberate", [
        "model selects activity or rest / wait", "use known records + destinations", "reflect / contact are also available",
    ], color="external", title_size=16)
    c.tile(848, 164, 344, 104, "3. Select / revise a goal", [
        "persist criteria, dependencies, revision", "retain continuation and wake condition", "changed criteria require new evidence",
    ], color="attention", title_size=16)
    c.tile(1240, 164, 296, 104, "4. Act through tools", [
        "registered executor + policy", "bounded execution steps", "may choose no action",
    ], color="danger", title_size=16)
    c.tile(1240, 324, 296, 120, "5. Record receipts", [
        "action result / failure / unknown", "goal and revision attribution", "public feedback is evidence", "invocation failure is not success",
    ], color="storage", title_size=16)
    c.tile(848, 324, 344, 120, "6. Assess completion claims", [
        "applicable verifier checks receipts", "supported / contradicted / unverified", "subjective criteria stay subjective", "assessment happens when completion is claimed",
    ], title_size=15, body_size=11)
    c.tile(456, 324, 344, 120, "7. Learn + appraise", [
        "configured public-outcome learning", "grounded appraisal / attention", "updates future decision context", "no claim of experienced emotion",
    ], color="process", title_size=16)
    c.tile(64, 324, 344, 120, "8. Checkpoint + rest", [
        "persist subject + session state", "wait for a later wake or input", "budget exhaustion → resource wait", "study limit → resource stop",
    ], color="storage", title_size=16)
    c.note(64, 476, "Logical feedback flow: learning hooks run on public outcomes; verification is conditional on a completion claim, not a mandatory call after each action.", size=12, fill=INK_SOFT)

    c.cylinder(40, 532, 960, 120, "shared/agent_runtime.db + replay records", [
        "manifest · goals / revisions · actions / receipts · learning state · checkpoints",
        "Activation replay frames capture transitions and request/state evidence; replay checks have documented limits.",
        "Shared Life History and self-model stores retain identity context alongside the research records.",
    ])
    c.tile(1032, 532, 528, 120, "Observer: read-only", [
        "Reads durable records without advancing the subject", "Does not own the loop or mutate subject state", "Budgets and stop limits apply across activations", "See RESEARCH_RUNTIME.md for modes and replay limits",
    ], color="client", title_size=17)
    return c


# ---------------------------------------------------------------------------
# Social preview: generated artwork with current branding
# ---------------------------------------------------------------------------


BUILDERS = {
    "runtime-architecture": build_runtime_architecture,
    "single-turn-cognitive-flow": build_single_turn_flow,
    "memory-persistence-model": build_memory_persistence,
    "auth-tool-safety-boundary": build_auth_tool_safety,
    "evaluation-ablation-harness": build_evaluation_harness,
    "component-claim-map": build_component_claim_map,
    "research-runtime-autonomy": build_research_runtime_autonomy,
}
