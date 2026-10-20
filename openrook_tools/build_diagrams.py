"""Eight reproducible documentation diagrams, in shared light and dark themes.

Regenerate SVG sources and PNG previews with
``python3 openrook_tools/render_diagram_pngs.py``. Long explanations live in the
adjacent documentation captions; these figures expose the central relationship.
"""

from __future__ import annotations

from math import cos, hypot, sin

from openrook_tools.brand import LIGHT, MONO, Theme
from openrook_tools.diagram_toolkit import Canvas, loop_geometry


def build_research_runtime_autonomy(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(1200, 1200, "A persistent subject, one continuing loop",
               "One owner advances wake, deliberation, goals, permitted actions, receipts, conditional completion assessment, public-outcome learning and checkpoints around saved subject state; observers never advance the subject.",
               theme, "research-runtime-autonomy", "loop")
    c.region(64, 136, 1072, 936, "ONE SUBJECT OWNER  ·  serialized by the owner lock")
    cx, cy, radius = 600, 608, 376
    boxes, angles = loop_geometry(cx, cy, radius, 8, 248, 88)
    # Every ring segment is part of the same circle. Marker tip is at refX=8.
    for k, (_, exit_angle) in enumerate(angles):
        entry_angle = angles[(k + 1) % 8][0]
        a = (cx + radius * cos(exit_angle), cy + radius * sin(exit_angle))
        b = (cx + radius * cos(entry_angle), cy + radius * sin(entry_angle))
        c.path(f"M{a[0]:.3f},{a[1]:.3f} A{radius},{radius} 0 0 1 {b[0]:.3f},{b[1]:.3f}",
               extra=' data-connector="ring"')
    # Loop-specific radii: the snapped station center and hub center define
    # each ray. Stop six pixels before the hub, at eight distinct ports.
    for x, y, w, h in boxes:
        dx, dy = x + w / 2 - cx, y + h / 2 - cy
        distance = hypot(dx, dy)
        ux, uy = dx / distance, dy / distance
        station_edge = min(w / 2 / abs(ux) if ux else float("inf"),
                           h / 2 / abs(uy) if uy else float("inf"))
        hub_edge = min(172 / abs(ux) if ux else float("inf"),
                       84 / abs(uy) if uy else float("inf"))
        start = (cx + (distance - station_edge) * ux,
                 cy + (distance - station_edge) * uy)
        end = (cx + (hub_edge + 6) * ux, cy + (hub_edge + 6) * uy)
        c.path(f"M{start[0]:.3f},{start[1]:.3f} L{end[0]:.3f},{end[1]:.3f}",
               dashed=True, extra=' data-connector="spoke" opacity="0.65"')
    c.label(616, 402, "CONTEXT", anchor="start")
    c.label(616, 802, "RECEIPTS", anchor="start")
    c.label(388, 594, "LEARNING")
    c.label(484, 428, "CHECKPOINT")
    names = [
        ("Wake", "timer · condition · budget", "step"),
        ("Deliberate", "activity, rest or wait", "step"),
        ("Choose goal", "select / revise", "step"),
        ("Act", "registered tools + policy", "step"),
        ("Record receipt", "result / failure / unknown", "store"),
        ("Verify claim", "only on a completion claim", "optional"),
        ("Learn & appraise", "public outcomes", "step"),
        ("Checkpoint & rest", "save state · wait for wake", "store"),
    ]
    for index, (box, (title, sub, role)) in enumerate(zip(boxes, names)):
        c.node(*box, title, sub, role=role, focal=index == 5, number=f"0{index + 1}", size=20)
    # Hub is state, with a quiet plum tint; verification is the focal station.
    c.rect(428, 524, 344, 168, fill=theme.page)
    c.rect(428, 524, 344, 168, fill=theme.plum, stroke=theme.plum, extra=' fill-opacity="0.08" data-role="hub"')
    c.text(600, 564, "Persistent subject state", size=23, weight=600, anchor="middle")
    c.text(600, 595, "shared/agent_runtime.db", size=16, anchor="middle", font=MONO, fill=theme.muted)
    c.line(452, 612, 748, 612)
    c.text(600, 640, "manifest · goals · receipts", size=17, anchor="middle")
    c.text(600, 666, "learning · checkpoints", size=17, anchor="middle")
    c.text(852, 956, "May choose no action.", size=16, fill=theme.muted)
    c.text(760, 212, "Budget exhausted → resource wait", size=16, fill=theme.muted)
    c.text(760, 238, "Study limit → resource stop", size=16, fill=theme.muted)
    # Observer is outside ownership and has no route through the operating ring.
    c.add(f'<g transform="translate(64 1092)" fill="none" stroke="{theme.muted}" '
          'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
          '<path d="M0,12 Q12,-2 24,12 Q12,26 0,12 Z"/>'
          '<circle cx="12" cy="12" r="4"/></g>')
    c.text(104, 1110, "Observer (read-only) · reads the hub, never advances the loop",
           size=18, fill=theme.muted)
    c.text(48, 1150, "A model saying “done” is not completion.", size=22, weight=600, fill=theme.amber_ink)
    c.text(48, 1180, "Logical feedback flow; learning hooks run on public outcomes. No claim of experienced emotion.", size=16, fill=theme.muted)
    return c


def build_runtime_architecture(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(1400, 752, "One cognition core, several ways in",
               "Clients reach openrook or openrook-web, which share session management and cognition, a runtime LLM backend, policy-gated tools and durable stores; research mode adds one subject owner.",
               theme, "runtime-architecture", "architecture")
    c.text(48, 124, "Conversation hosts share the same processing path.", size=18, fill=theme.muted)
    c.region(1012, 436, 340, 216, "RESEARCH MODE ADDS", dashed=True)
    for a, b in ((248, 296), (560, 608), (944, 992)):
        c.connector([(a, 252), (b, 252)])
    c.connector([(656, 308), (656, 360), (180, 360), (180, 496)], dashed=True)
    c.connector([(716, 308), (716, 392), (500, 392), (500, 496)])
    c.connector([(800, 308), (800, 496)])
    c.connector([(888, 308), (888, 416), (1296, 416), (1296, 508)], dashed=True)
    c.node(48, 208, 200, 88, "Clients", "chat · UI · HTTP", role="input")
    c.node(296, 196, 264, 112, "Hosts", "openrook / openrook-web")
    c.node(608, 196, 336, 112, "Session + cognition", "SessionManager · pipeline", focal=True)
    c.node(992, 196, 360, 112, "LLM backend", "configured runtime adapter", role="external")
    c.text(992, 340, "auto · provider · codex · minimax", size=15, font=MONO, fill=theme.muted)
    c.text(992, 367, "openai_compatible (also Ollama)", size=15, font=MONO, fill=theme.muted)
    c.node(48, 496, 264, 96, "Tool executor", "registered tools · policy gate")
    c.node(356, 496, 288, 96, "Per-user state", "memory · profiles · relations", role="store")
    c.node(692, 496, 296, 96, "Shared identity", "self model · life history", role="store")
    c.node(1036, 508, 292, 96, "One subject owner", "background loop · saved state", role="store")
    c.text(356, 628, "openrook.db + sessions/", size=16, font=MONO, fill=theme.muted)
    c.text(692, 624, "self_model.db", size=15, font=MONO, fill=theme.muted)
    c.text(692, 648, "life_history.db", size=15, font=MONO, fill=theme.muted)
    c.text(1056, 634, "shared/agent_runtime.db", size=15, font=MONO, fill=theme.muted)
    c.footer("Tools and backend choices are explicit runtime configuration. Research mode serializes one persistent subject.")
    return c


def build_single_turn_flow(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(1400, 848, "A turn is more than one model call",
               "Conversation-mode stages separate deterministic runtime work from possible appraisal, inner-dialogue, tool-selection, generation, self-check and regeneration model calls; end_session consolidation is a separate path with additional possible inference.",
               theme, "single-turn-cognitive-flow", "swimlane")
    c.text(48, 124, "Conversation mode · stage order runs left to right", size=18, fill=theme.muted)
    columns = [160, 400, 640, 880, 1120]
    for x in columns[:-1]:
        c.connector([(x + 216, 268), (x + 240, 268)])
    c.line(48, 364, 1352, 364)
    c.line(48, 512, 1352, 512)
    c.text(48, 244, "RUNTIME", size=13, weight=600, fill=theme.muted)
    c.text(48, 268, "no model", size=14, fill=theme.muted)
    c.text(48, 292, "for these", size=14, fill=theme.muted)
    c.text(48, 316, "operations", size=14, fill=theme.muted)
    stages = [
        ("State + context", "decay · retrieve"),
        ("Deliberation", "defense · tool policy"),
        ("Generate", "build prompt"),
        ("Self-check", "rule fit · grounding"),
        ("Per-turn updates", "save · drain energy"),
    ]
    for i, (x, (name, sub)) in enumerate(zip(columns, stages)):
        c.node(x, 216, 216 if i < 4 else 232, 104, name, sub,
               focal=i == 3, number=f"0{i + 1}", size=20)
    c.text(48, 412, "MODEL", size=13, weight=600, fill=theme.muted)
    c.text(48, 436, "call sites", size=14, fill=theme.muted)
    calls = [(160, 412, "Appraisal", False), (400, 412, "Inner dialogue", False),
             (400, 452, "Tool selection", False), (640, 412, "Generation", True),
             (880, 412, "LLM self-check", False), (880, 452, "Regenerate", False)]
    for x, y, name, ordinary in calls:
        c.add(f'<circle cx="{x + 8}" cy="{y - 6}" r="5" fill="{theme.muted if ordinary else theme.page}" stroke="{theme.muted}" stroke-width="1.5" data-call="{name}"/>')
        c.text(x + 24, y, name, size=18)
    c.text(640, 444, "one if reached", size=15, fill=theme.muted)
    c.text(1120, 412, "No model call", size=18, fill=theme.muted)
    c.text(160, 550, "LLM self-check is selected by intensity > 0.85, contradictions, or dialogue deadlock.", size=18,
           fill=theme.amber_ink)
    c.text(160, 580, "Regeneration is conditional on a failed check or the configured low-coherence path.", size=16, fill=theme.muted)
    c.text(48, 650, "SEPARATE", size=13, weight=600, fill=theme.muted)
    c.text(48, 674, "at session", size=14, fill=theme.muted)
    c.text(48, 698, "end", size=14, fill=theme.muted)
    c.connector([(432, 680), (504, 680)], dashed=True)
    c.node(160, 636, 272, 88, "end_session()", "explicit consolidation path", role="input", mono=True, size=20)
    c.node(504, 636, 592, 88, "Consolidate memory", "digestion / autobiographical ingestion may call a model", role="store")
    c.footer("○  may call model     ●  ordinary generation when reached",
             second="No fixed aggregate ceiling: enabled branches, dialogue/tool rounds, retries and regeneration change cost.")
    return c


def build_evaluation_harness(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(
        1200, 544, "How structural evidence becomes a report",
        "The ablation command runs fresh CognitivePipelines for scenarios, stamps provenance and outcome labels, and writes per-variant JSON and a combined summary; the historical checked-in reports use an evaluation-only mock backend.",
        theme, "evaluation-ablation-harness", "data flow",
    )
    c.text(48, 128, "CHECKED-IN REPORTS  ·  EVALUATION-ONLY MOCK BACKEND",
           size=14, weight=600, fill=theme.muted)
    c.text(48, 156, "Phase 11 · 2026-05-16 · 3d3f869c",
           size=16, fill=theme.muted)
    c.text(48, 208, "python3 -m evals.ablation", size=16, font=MONO)

    boxes = [(48, 236, 268, 108), (340, 236, 196, 108),
             (560, 236, 208, 108), (792, 236, 176, 108),
             (992, 236, 160, 108)]
    for left, right in zip(boxes, boxes[1:]):
        c.connector([(left[0] + left[2], 290), (right[0], 290)])
    steps = [
        ("Run ablations", "scenarios × removals", "input", False, False),
        ("Fresh pipeline", "one per scenario", "step", False, False),
        ("Stamp evidence", "provenance + labels", "step", True, False),
        ("Variant JSON", "one per variant", "store", False, False),
        ("summary.json", "combined deltas", "store", False, True),
    ]
    for index, (box, (title, sub, role, focal, mono)) in enumerate(zip(boxes, steps)):
        c.node(*box, title, sub, role=role, focal=focal, number=f"0{index + 1}",
               mono=mono, size=15 if mono else 20)
    c.text(48, 394, "The real cognitive pipeline, with mock model responses.",
           size=21, weight=600)
    c.footer(
        "Historical structural checks; wording quality and independent-agent comparisons are outside this suite.",
        second="Later real-model research studies are separate evidence.",
    )
    return c


def build_component_claim_map(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(
        1200, 808, "What Phase 11 actually supports",
        "Historical Phase 11 mock-backend evidence classifies relationship memory as load-bearing in this suite, semantic memory as a negative control, inner dialogue and defense as not falsifiable here, self and other profiling as an architectural claim, and emotional modulators as tested mechanics.",
        theme, "component-claim-map", "matrix",
    )
    c.text(48, 128, "HISTORICAL PHASE 11  ·  MOCK BACKEND",
           size=14, weight=600, fill=theme.muted)
    c.text(48, 156, "2026-05-16 · 3d3f869c", size=16, fill=theme.muted)
    c.text(72, 204, "COMPONENT", size=13, weight=600, fill=theme.muted)
    c.text(584, 204, "STATUS IN THIS SUITE", size=13, weight=600, fill=theme.muted)
    c.line(48, 220, 1152, 220)
    rows = [
        ("Relationship memory", "Load-bearing in this suite",
         "Removal: 4/6 pass · two expected failures", "solid"),
        ("Semantic memory", "Negative control",
         "Removal: 6/6 pass · retrieval not exercised", "empty"),
        ("Inner dialogue", "Not falsifiable here",
         "Removal: 6/6 pass · strategy unchanged", "slash"),
        ("Defense mechanisms", "Not falsifiable here",
         "Removal: 6/6 pass · wording not graded", "slash"),
        ("Self / other profiling", "Architectural claim",
         "Shared design, not a benchmark result", "square"),
        ("Emotional modulators", "Unit + integration mechanics",
         "Numeric state, not experienced emotion", "double"),
    ]
    for index, (name, status, sub, shape) in enumerate(rows):
        y = 232 + index * 80
        focal = index == 0
        color = theme.amber_ink if focal else theme.ink
        c.add(f'<g data-node="{name}" data-role="matrix-row">')
        if focal:
            c.rect(560, y, 592, 72, fill=theme.page)
            c.rect(560, y, 592, 72, fill=theme.amber, stroke=theme.amber_ink,
                   extra=' fill-opacity="0.10" data-focal="true"')
        c.text(72, y + 44, name, size=22, weight=600)
        gx, gy = 592, y + 36
        if shape in ("solid", "empty", "double"):
            c.add(f'<circle cx="{gx}" cy="{gy}" r="10" '
                  f'fill="{color if shape == "solid" else "none"}" '
                  f'stroke="{color}" stroke-width="1.8" data-role="status-mark"/>')
            if shape == "double":
                c.add(f'<circle cx="{gx}" cy="{gy}" r="5" fill="none" '
                      f'stroke="{color}" stroke-width="1.5"/>')
        elif shape == "slash":
            c.add(f'<path d="M{gx - 6},{gy + 10} L{gx + 6},{gy - 10}" '
                  f'fill="none" stroke="{color}" stroke-width="2" data-role="status-mark"/>')
        else:
            c.rect(gx - 9, gy - 9, 18, 18, stroke=color, radius=0, sw=1.8,
                   extra=' data-role="status-mark"')
        c.text(620, y + 32, status, size=21, weight=600, fill=color)
        c.text(620, y + 57, sub, size=16, fill=theme.muted)
        c.add("</g>")
        if index < len(rows) - 1:
            c.line(48, y + 76, 1152, y + 76)
    c.line(48, 732, 1152, 732)
    c.text(48, 766, "Later evidence:", size=16, fill=theme.muted)
    c.text(184, 766, "RESEARCH_RUNTIME.md · OVERVIEW.md", size=16,
           font=MONO, fill=theme.muted)
    return c


def build_memory_persistence(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(
        1280, 824, "A user reset has a precise boundary",
        "Ordinary-mode deletion removes the per-user directory, while shared identity and research databases and global configuration are retained; research-mode DELETE and mutating admin requests return HTTP 409.",
        theme, "memory-persistence-model", "nested",
    )
    c.region(48, 136, 1184, 472, "data/", dashed=False)
    c.region(80, 196, 544, 380, "shared/  ·  RETAINED", dashed=False)
    c.rect(656, 196, 544, 380, fill=theme.amber, radius=8,
           extra=' fill-opacity="0.05"')
    c.region(656, 196, 544, 380, "<platform>_<user_id>/  ·  DELETED",
             focal=True, dashed=True)
    c.node(104, 260, 496, 80, "self_model.db",
           "assistant observations + traits", role="store", mono=True)
    c.node(104, 360, 496, 80, "life_history.db",
           "formative experience + identity", role="store", mono=True)
    c.node(104, 460, 496, 80, "agent_runtime.db",
           "research state + evidence", role="store", mono=True)
    c.node(680, 260, 496, 108, "openrook.db",
           "memory · relationships · profiles", role="store", mono=True)
    c.node(680, 408, 496, 132, "Session files", role="step")
    c.text(700, 506, "sessions/*.json  ·  *.history.json",
           size=15, font=MONO, fill=theme.muted)
    c.text(72, 652, "Global configuration", size=21, weight=600)
    c.text(72, 680, "runtime_config.yaml  ·  soul.yaml",
           size=16, font=MONO, fill=theme.muted)
    c.text(920, 668, "outside user deletion", size=16, fill=theme.muted)
    c.footer(
        "Research mode: HTTP DELETE and mutating admin requests return 409.",
        second="Stop the subject owner before a complete research reset. Shared evidence may retain user-derived material.",
    )
    return c


def build_auth_tool_safety(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(
        1280, 848, "Two independent gates",
        "Configured HTTP bearer authentication and tool registration are independent: internal skill tools remain available, external tools require tools_enabled, shell also requires shell_tool_enabled, and native or regex selection uses the same executor policy.",
        theme, "auth-tool-safety-boundary", "flowchart",
    )
    c.region(48, 136, 560, 600, "HTTP ACCESS", dashed=False)
    c.region(640, 136, 592, 600, "TOOL AVAILABILITY", dashed=False)
    c.connector([(236, 276), (236, 332)])
    c.connector([(236, 424), (236, 492)])
    c.label(250, 464, "OPEN / VALID", anchor="start")
    c.connector([(392, 380), (496, 380), (496, 492)])
    c.label(492, 366, "INVALID")
    c.connector([(936, 276), (936, 332)])
    c.connector([(936, 560), (936, 624)])
    c.label(950, 600, "REGISTERED", anchor="start")
    c.node(80, 204, 480, 72, "Protected requests",
           "chat · admin APIs · protected routes", role="input")
    c.node(80, 332, 312, 92, "Bearer authentication",
           "api_key set → valid token", focal=True, size=20)
    c.node(80, 492, 312, 80, "Endpoint checks",
           "route-specific controls")
    c.node(432, 492, 128, 80, "401",
           "refused", role="optional")
    c.node(80, 624, 480, 88, "Public shells + health",
           "/ · /admin · /settings · health / ready", role="input")
    c.node(672, 204, 528, 72, "Native or regex selection",
           "same executor and policy", role="input")
    c.add('<g data-node="Availability rules" data-role="boundary" data-focal="true">')
    c.rect(672, 332, 528, 228, fill=theme.page)
    c.rect(672, 332, 528, 228, fill=theme.amber,
           stroke=theme.amber_ink, extra=' fill-opacity="0.10"')
    c.text(692, 372, "Availability rules", size=21, weight=600,
           fill=theme.amber_ink)
    c.line(692, 392, 1180, 392)
    for y, name, condition in (
        (420, "Internal skills", "always available"),
        (472, "External tools", "tools_enabled"),
        (524, "Shell (external)", "+ shell_tool_enabled"),
    ):
        c.text(692, y, name, size=17, weight=600)
        if name == "Internal skills":
            c.text(876, y, condition, size=16, fill=theme.muted)
        else:
            c.text(876, y, condition, size=15, font=MONO, fill=theme.muted)
    c.add("</g>")
    c.node(672, 624, 528, 88, "ToolExecutor",
           "registration · workspace · execution policy")
    c.footer(
        "No api_key: HTTP access is open. A configured key requires a valid Bearer token.",
        second="Internal skill tools remain available, including writes, when external tools are off.",
    )
    return c


def build_feature_overview(theme: Theme = LIGHT) -> Canvas:
    c = Canvas(
        1200, 816, "State connects the conversation to action",
        "Web UI, Telegram, Console and REST API connect through persistent memory, relationships, modeled mood, identity and evolution to reviewed and enabled skills and opt-in external tools: web search, workspace-scoped files and separately enabled shell.",
        theme, "feature-overview", "layer stack",
    )
    c.text(48, 124, "Many ways in. One continuing cognitive state.", size=18, fill=theme.muted)
    c.connector([(80, 260), (80, 420)])
    c.connector([(80, 472), (80, 640)])
    for y, label in ((240, "IN"), (452, "STATE"), (672, "ACT")):
        c.text(80, y, label, size=15, weight=600, fill=theme.muted, anchor="middle")

    # Three layers; the members are tiles within a layer, not separate stages.
    for y, height, name, focal in (
        (160, 160, "Channels", False),
        (352, 188, "Persistent state", True),
        (572, 160, "Capabilities", False),
    ):
        c.add(f'<g data-node="{name}" data-role="layer" data-focal="{str(focal).lower()}">')
        c.rect(136, y, 1016, height, fill=theme.page if focal else theme.panel)
        if focal:
            c.rect(136, y, 1016, height, fill=theme.amber, extra=' fill-opacity="0.10"')
        c.rect(136, y, 1016, height, stroke=theme.amber_ink if focal else theme.border)
        c.text(160, y + 36, name, size=24, weight=600,
               fill=theme.amber_ink if focal else theme.ink)
        c.add("</g>")
    c.text(1128, 196, "Shared cognition across interfaces", size=16,
           fill=theme.muted, anchor="end")
    c.text(1128, 388, "Persists beyond a single prompt", size=16,
           fill=theme.amber_ink, anchor="end")
    c.text(1128, 608, "External tools require tools_enabled", size=15,
           fill=theme.muted, anchor="end")

    # Local 24px line glyphs keep the overview independent of icon fonts.
    icons = {
        "Web UI": '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M2,8 H22 M6,5.5 H7"/>',
        "Telegram": '<path d="M2,11 L22,3 L17,22 L11,16 L7,20 L7,13 L22,3 M7,13 L17,22"/>',
        "Console": '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M6,9 L10,12 L6,15 M13,16 H18"/>',
        "REST API": '<path d="M8,5 L2,12 L8,19 M16,5 L22,12 L16,19 M14,3 L10,21"/>',
        "Memory": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3,5 V19 C3,23 21,23 21,19 V5 M3,12 C3,16 21,16 21,12"/>',
        "Relationships": '<circle cx="7" cy="7" r="3"/><circle cx="18" cy="10" r="3"/><path d="M1,21 V18 C1,12 13,12 13,18 V21 M15,16 C20,14 23,17 23,21"/>',
        "Modeled mood": '<path d="M2,6 H22 M2,12 H22 M2,18 H22"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="10" cy="18" r="2"/>',
        "Identity": '<rect x="2" y="3" width="20" height="18" rx="2"/><circle cx="9" cy="9" r="2.5"/><path d="M5,17 C5,12 13,12 13,17 M16,8 H19 M16,13 H19"/>',
        "Evolution": '<path d="M4,3 V17 Q4,21 8,21 H20 M4,14 L11,8 L16,12 L22,4 M16,4 H22 V10"/>',
        "Skills": '<path d="M6,3 H16 L21,8 V21 H6 Z M16,3 V8 H21 M10,12 H17 M10,16 H17 M2,7 H3 M2,12 H3 M2,17 H3"/>',
        "Web search": '<circle cx="10" cy="10" r="7"/><path d="M15,15 L22,22 M3,10 H17 M10,3 C6,7 6,13 10,17 C14,13 14,7 10,3"/>',
        "Files": '<path d="M2,7 V21 H22 V6 H12 L9,3 H2 Z M2,10 H22"/>',
        "Shell": '<path d="M4,5 L11,12 L4,19 M13,19 H21"/>',
    }

    def tile(x, y, width, name, sub, *, state=False):
        height = 104 if state else 80
        c.add(f'<g data-member="{name}">')
        c.rect(x, y, width, height, fill=theme.page, stroke=theme.border)
        c.add(f'<g transform="translate({x + 16} {y + 16})" fill="none" '
              f'stroke="{theme.muted}" stroke-width="1.6" stroke-linecap="round" '
              f'stroke-linejoin="round" aria-hidden="true">{icons[name]}</g>')
        tx, ty = (x + 16, y + 62) if state else (x + 52, y + 32)
        c.text(tx, ty, name, size=19 if state else 21, weight=600)
        c.text(tx, ty + 25, sub, size=16, fill=theme.muted)
        c.add("</g>")

    for index, (name, sub) in enumerate((
        ("Web UI", "inspect + chat"), ("Telegram", "messaging bot"),
        ("Console", "terminal chat"), ("REST API", "HTTP access"),
    )):
        tile(160 + index * 244, 216, 232, name, sub)
    for index, (name, sub) in enumerate((
        ("Memory", "recall across turns"), ("Relationships", "per-user history"),
        ("Modeled mood", "numeric state"), ("Identity", "shared self model"),
        ("Evolution", "learned changes"),
    )):
        tile(160 + index * 196, 412, 184, name, sub, state=True)
    for index, (name, sub) in enumerate((
        ("Skills", "review + enable"), ("Web search", "tools opt-in"),
        ("Files", "workspace-scoped"), ("Shell", "separate shell flag"),
    )):
        tile(160 + index * 244, 628, 232, name, sub)
    c.footer("State persists across turns. Modeled mood is numeric state, not experienced emotion.")
    return c


BUILDERS = {
    "research-runtime-autonomy": build_research_runtime_autonomy,
    "runtime-architecture": build_runtime_architecture,
    "single-turn-cognitive-flow": build_single_turn_flow,
    "memory-persistence-model": build_memory_persistence,
    "auth-tool-safety-boundary": build_auth_tool_safety,
    "evaluation-ablation-harness": build_evaluation_harness,
    "component-claim-map": build_component_claim_map,
    "feature-overview": build_feature_overview,
}
