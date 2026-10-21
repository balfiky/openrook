"""Tool-aware deliberation bridge.

Coordinates the full cognitive tool loop:
  1. Detect if the user message warrants tool use (heuristic)
  2. Derive action variables from modulator state
  3. Build a ToolIntent
  4. Run the action arbiter → ToolDecision
  5. Execute the tool if approved
  6. Appraise the result → ToolObservation
  7. Apply emotional deltas to the engine
  8. Optionally loop (bounded)
  9. Return ToolTrace + context summary for the generator

All tool intent detection in this phase is deterministic heuristic.
Full LLM-driven structured proposals are a future enhancement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from core.action_variables import derive_action_variables
from core.character_vector import CharacterVector
from core.life_influence import (
    LifeInfluence,
    action_variable_deltas,
    apply_character_vector_to_action_variables,
    apply_life_influence_to_action_variables,
)
from core.task_planning import (
    detect_multi_step_intent,
    execute_plan,
    is_continue_request,
    is_status_request,
    summarize_plan_status,
)
from core.tool_appraisal import appraise_tool_result
from core.types import (
    ActionVariables,
    AgencyDecision,
    ModulatorState,
    PersonProfile,
    TaskPlan,
    TaskTrace,
    ToolCategory,
    ToolDecision,
    ToolIntent,
    ToolObservation,
    ToolResult,
    ToolTrace,
)
from openrook_tools.executor import ToolExecutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_MAX_EXECUTIONS = 4
HARD_CAP_EXECUTIONS = 6

# Per-autonomy budgets. high_risk gets generous headroom for chains like
# clone -> install -> build -> run -> verify; lower autonomy stays tight.
AUTONOMY_EXECUTION_BUDGETS = {
    "off":        (0, 0),
    "assisted":   (3, 5),
    "autonomous": (6, 10),
    "high_risk":  (16, 32),
}


def autonomy_execution_budget(autonomy_level: str) -> tuple[int, int]:
    """Return (default, hard_cap) tool-execution budget for an autonomy level."""
    return AUTONOMY_EXECUTION_BUDGETS.get(
        _normalize_autonomy_level(autonomy_level),
        (DEFAULT_MAX_EXECUTIONS, HARD_CAP_EXECUTIONS),
    )

# Arbiter decision thresholds (Section 11 of design spec)
REFUSE_RISK_TOLERANCE = 0.4       # destructive + risk below this → refuse
CLARIFY_AUTONOMY_BIAS = 0.30     # autonomy below this → clarify
CLARIFY_WRITE_THRESHOLD = 0.65   # write/destructive + clarification above this → clarify
DEFER_URGENCY_THRESHOLD = 0.15   # urgency below this → defer


# ---------------------------------------------------------------------------
# Tool loop result
# ---------------------------------------------------------------------------

@dataclass
class ToolLoopResult:
    """What the tool loop returns to the pipeline."""
    trace: ToolTrace
    action_variables: ActionVariables
    tool_context_summary: str  # summarized for the generator — not raw output
    life_influence_effects: dict[str, float] = field(default_factory=dict)
    capability_gaps: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic tool intent detection (fallback router)
# ---------------------------------------------------------------------------
#
# Tool intent routing is two-tier. When an OpenAI-compatible backend is
# configured, NativeToolCallRunner (openrook_tools/native_orchestrator.py) routes
# via real model tool-calls and this table is bypassed. The rules below are
# the fallback for backends without a tool-call API (Codex CLI), runtimes
# with no runner attached, and offline tests.
#
# Rules are matched strictly in table order: the first rule whose tool is
# available, whose pattern matches, and whose extractor yields arguments
# wins. Table order therefore encodes precedence — more specific phrasings
# must stay above broader ones. To add a rule: insert an IntentRule at the
# right precedence position and add a canonical example utterance to
# RULE_EXAMPLES in tests/test_intent_rules.py (the coverage test fails on
# any rule without a reachable example).


@dataclass(frozen=True)
class IntentRule:
    """One declarative heuristic-routing rule.

    ``name`` is a stable identifier referenced by tests and failure reports.
    ``extractor`` is a key understood by :func:`_extract_args`; a rule only
    fires when extraction yields arguments (a matching pattern whose
    extraction fails falls through to later rules). The tool's risk category
    is not duplicated here — the executor registry stays the single source
    of truth and the arbiter looks it up at decision time.
    """

    name: str
    tool: str
    extractor: str
    patterns: tuple[re.Pattern[str], ...]


def _rule(
    name: str,
    tool: str,
    extractor: str,
    *patterns: str,
    flags: int = re.I,
) -> IntentRule:
    return IntentRule(
        name=name,
        tool=tool,
        extractor=extractor,
        patterns=tuple(re.compile(p, flags) for p in patterns),
    )


INTENT_RULES: tuple[IntentRule, ...] = (
    # Skill registry — generic self-capability import/enabling.
    _rule(
        "skills_list", "skills.list", "empty",
        r"\b(?:list|show)\s+(?:the\s+)?"
        r"(?:imported\s+|enabled\s+|runtime\s+)?skills?\b",
    ),
    _rule(
        "skills_enable", "skills.enable", "skill_id",
        r"\b(?:enable|activate)\s+(?:the\s+)?"
        r"(?:skill|capability|module|integration)\s+"
        r"([A-Za-z0-9][A-Za-z0-9._-]{0,79})\b",
    ),
    _rule(
        "skills_disable", "skills.disable", "skill_id",
        r"\b(?:disable|deactivate)\s+(?:the\s+)?"
        r"(?:skill|capability|module|integration)\s+"
        r"([A-Za-z0-9][A-Za-z0-9._-]{0,79})\b",
    ),
    _rule(
        "skills_delete", "skills.delete", "skill_id",
        r"\b(?:delete|remove)\s+(?:the\s+)?"
        r"(?:skill|capability|module|integration)\s+"
        r"([A-Za-z0-9][A-Za-z0-9._-]{0,79})\b",
    ),
    _rule(
        "skills_audit", "skills.audit", "optional_skill_id",
        r"\b(?:audit|inspect|check|show)\b.{0,80}"
        r"\b(?:audit\s+log|audit|problem|errors?|warnings?|status|"
        r"skill|capability|module|integration)\b",
        flags=re.I | re.S,
    ),
    _rule(
        "skills_audit_problem", "skills.audit", "optional_skill_id",
        r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?problem\b.{0,100}"
        r"\b(?:skill|audit|import|registry)\b",
        flags=re.I | re.S,
    ),
    _rule(
        "skills_create", "skills.create_from_request", "skill_request",
        r"\b(?:create|make|add|build|import|install|register)\b"
        r".{0,120}\b(?:skill|capability|module|integration)\b",
        flags=re.I | re.S,
    ),
    _rule(
        "skills_make_permanent", "skills.create_from_request", "skill_request",
        r"\b(?:make|save|turn)\b.{0,60}\b(?:this|it)\b.{0,60}"
        r"\b(?:permanent|persistent|persisted)\b.{0,60}"
        r"\b(?:skill|capability|module|integration)?\b",
        flags=re.I | re.S,
    ),
    # Filesystem — specific path first
    _rule(
        "fs_read_file", "fs.read_file", "path",
        r"\bread(?:ing)?\s+(?:the\s+)?file\s+(\S+)",
    ),
    _rule(
        "fs_show_file", "fs.read_file", "path",
        r"\b(?:show|cat|display)\s+(?:the\s+)?(?:contents?\s+of\s+)?(\S+\.\w+)",
    ),
    _rule(
        "fs_list_dir_path", "fs.list_dir", "path",
        r"\blist\s+(?:the\s+)?(?:files?\s+in\s+|dir(?:ectory)?\s+)(\S+)",
    ),
    _rule("fs_ls", "fs.list_dir", "path", r"\b(?:ls)\s+(\S+)"),
    _rule(
        "fs_whats_in_path", "fs.list_dir", "path",
        r"\bwhat(?:'s| is)\s+in\s+((?:/|~)\S*)",
    ),
    # Filesystem — no explicit path (default to / or .)
    _rule(
        "fs_list_files_default", "fs.list_dir", "default_root",
        r"\blist\s+(?:the\s+)?(?:system\s+)?files?\b",
    ),
    _rule(
        "fs_show_files_default", "fs.list_dir", "default_root",
        r"\b(?:show|display)\s+(?:me\s+)?(?:the\s+)?(?:system\s+)?files?\b",
    ),
    _rule(
        "fs_what_files_here", "fs.list_dir", "default_cwd",
        r"\bwhat\s+files?\b.*\b(?:here|available|exist)\b",
    ),
    # Search/glob
    _rule(
        "fs_search_quoted_in_path", "fs.search_text", "pattern_path",
        r"\bsearch\s+(?:for\s+)?[\"']([^\"']+)[\"']\s+in\s+(\S+)",
    ),
    _rule(
        "fs_grep", "fs.search_text", "pattern_path",
        r"\bgrep\s+[\"']?(\S+)[\"']?\s+(\S+)",
    ),
    _rule(
        "fs_glob_in_path", "fs.glob_paths", "pattern_path",
        r"\bfind\s+files?\s+(?:matching\s+)?[\"']?([^\"'\s]+)[\"']?\s+in\s+(\S+)",
    ),
    # Write/delete
    _rule(
        "fs_write_quoted_to_path", "fs.write_file", "content_path",
        r"\b(?:write|save)\s+[\"']([^\"']+)[\"']\s+to\s+(?:file\s+)?(\S+)",
    ),
    _rule(
        "fs_create_file_with_content", "fs.write_file", "path_content",
        r"\bcreate\s+(?:a\s+)?file\s+(\S+)\s+(?:with|containing)\s+[\"']([^\"']+)[\"']",
    ),
    _rule(
        "fs_delete_path", "fs.delete_path", "path",
        r"\bdelete\s+(?:the\s+)?(?:file|dir(?:ectory)?)\s+(\S+)",
    ),
    _rule("fs_rm", "fs.delete_path", "path", r"\brm\s+(\S+)"),
    # System inspection
    _rule(
        "system_memory_usage_query", "system.memory_usage", "empty",
        r"\b(?:what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?|give(?:\s+me)?"
        r"|read|fetch|report|grab)\b.{0,80}\b(?:memory|ram)\b.{0,80}"
        r"\b(?:usage|utili[sz]ation|used|free|available|total)\b",
    ),
    _rule(
        "system_memory_how_much", "system.memory_usage", "empty",
        r"\b(?:how\s+much|what(?:'s|\s+is))\b.{0,80}\b(?:memory|ram)\b"
        r".{0,80}\b(?:left|available|used|free|total)\b",
    ),
    _rule(
        "system_free_command", "system.memory_usage", "empty",
        r"^\s*free(?:\s+-[a-z]+)?\s*$",
    ),
    _rule(
        "system_memory_status", "system.memory_usage", "empty",
        r"\b(?:give(?:\s+me)?|read|show|get|tell(?:\s+me)?|fetch|report|check)\b"
        r".{0,80}\b(?:current|live|active)?\s*\b(?:system\s+)?(?:memory|ram)\b"
        r"\s*(?:usage|utili[sz]ation|status|info(?:rmation)?|stats?)?\b",
    ),
    # Shell
    _rule(
        "shell_hostname_question", "shell.run_command", "cmd_hostname",
        r"\b(?:what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?)\b.{0,80}"
        r"\b(?:your|the)?\s*(?:host\s*name|hostname|machine\s+name"
        r"|node\s+name|server\s+name)\b",
    ),
    _rule(
        "shell_machine_name_running", "shell.run_command", "cmd_hostname",
        r"\bname\s+of\s+the\s+machine\b.{0,80}\b(?:running|run)\b",
    ),
    _rule(
        "shell_hostname_bare", "shell.run_command", "cmd_hostname",
        r"^\s*hostname\s*$",
    ),
    _rule(
        "shell_run_known_command", "shell.run_command", "cmd_capture",
        r"\b(?:run|execute|issue|check|show)\s+(?:the\s+)?(?:command\s+)?"
        r"(hostname|uname(?:\s+-a)?)\b",
    ),
    _rule(
        "shell_uname_bare", "shell.run_command", "cmd_capture",
        r"^\s*(uname(?:\s+-a)?)\s*$",
    ),
    _rule(
        "shell_cat_etc_hostname", "shell.run_command", "cmd_cat_path",
        r"\bcat\s+(/etc/hostname)\b",
    ),
    _rule(
        "shell_disk_space_question", "shell.run_command", "cmd_disk_root",
        r"\b(?:how\s+much|what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?)\b"
        r".{0,80}\b(?:disk|drive|filesystem|storage)\b.{0,80}"
        r"\b(?:space|usage|free|left|available)\b",
    ),
    _rule(
        "shell_df_bare", "shell.run_command", "cmd_capture",
        r"^\s*(df(?:\s+-h)?(?:\s+/)?)\s*$",
    ),
    _rule(
        "shell_run_quoted", "shell.run_command", "cmd",
        r"\brun\s+(?:the\s+)?(?:command\s+)?[`\"']([^`\"']+)[`\"']",
    ),
    _rule(
        "shell_execute_quoted", "shell.run_command", "cmd",
        r"\bexecute\s+[`\"']([^`\"']+)[`\"']",
    ),
    _rule(
        "shell_run_explicit", "shell.run_command", "cmd_explicit",
        r"\b(?:run|execute|issue)\s+(?:the\s+)?(?:command\s+)?"
        r"([^\n`;&|]+)\s*$",
    ),
    _rule(
        "shell_do_explicit", "shell.run_command", "cmd_explicit",
        r"^\s*do\s+([^\n`;&|]+)\s*$",
    ),
    # Web — specific patterns first
    _rule(
        "web_fetch_url", "web.fetch", "url",
        r"\bfetch\s+(?:the\s+)?(?:url\s+|page\s+(?:at\s+)?)?(https?://\S+)",
    ),
    _rule(
        "web_extract_text_url", "web.extract_text", "url",
        r"\bextract\s+(?:the\s+)?text\s+(?:from\s+)?(https?://\S+)",
    ),
    _rule(
        "web_get_text_url", "web.extract_text", "url",
        r"\bget\s+(?:the\s+)?(?:readable\s+)?text\s+(?:from\s+|of\s+)?(https?://\S+)",
    ),
    # Web — search with various phrasings
    _rule(
        "web_search_the_web", "web.search", "query",
        r"\bsearch\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?[\"']?(.+?)[\"']?\s*$",
    ),
    _rule(
        "web_search_online_suffix", "web.search", "query",
        r"\bsearch\s+(?:for\s+)?[\"']?(.+?)[\"']?\s+(?:on(?:line|\s+the\s+(?:web|internet)))",
    ),
    _rule(
        "web_look_up", "web.search", "query",
        r"\blook\s*up\s+[\"']?(.+?)[\"']?\s*(?:online|on\s+the\s+(?:web|internet))?\s*$",
    ),
    _rule(
        "web_google", "web.search", "query",
        r"\b(?:google|search\s+for)\s+[\"']?(.+?)[\"']?\s*$",
    ),
    _rule(
        "web_browse_for", "web.search", "query",
        r"\bbrowse\s+(?:the\s+)?(?:web|internet)\s+(?:for\s+)?[\"']?(.+?)[\"']?\s*$",
    ),
    _rule(
        "web_find_about_online", "web.search", "query",
        r"\bfind\s+(?:me\s+)?(?:info(?:rmation)?|(?:some(?:thing)?)\s+)?\s*(?:about|on)\s+"
        r"[\"']?(.+?)[\"']?\s+(?:online|on\s+the\s+(?:web|internet))",
    ),
    _rule(
        "web_released_since", "web.search", "query_full",
        r"\b(?:find|give|show|list|tell)\b.{0,140}"
        r"\b(?:released|published|announced|launched|updated)\s+"
        r"(?:after|since|in|during|this)\b.*$",
    ),
    _rule(
        "web_top_best_released", "web.search", "query_full",
        r"\b(?:top|best)\b.{0,140}"
        r"\b(?:released|published|announced|launched|updated|after|since|20[2-9]\d)\b.*$",
    ),
    _rule(
        "web_latest_news", "web.search", "query_full",
        r"\b(?:latest|newest|recent)\b.{0,100}"
        r"\b(?:news|updates?|release|version|price|prices|score|scores|results?)\b.*$",
    ),
    _rule(
        "web_news_about", "web.search", "query_full",
        r"\b(?:news|updates?)\s+(?:about|on|for)\s+[\"']?(.+?)[\"']?\s*$",
    ),
    _rule(
        "web_what_happened_recently", "web.search", "query_full",
        r"\bwhat\s+happened\s+(?:with|to|about|on|in)\s+[\"']?.+?[\"']?\s+"
        r"(?:today|recently|this\s+(?:week|month|year))\b.*$",
    ),
    _rule(
        "web_current_facts", "web.search", "query_full",
        r"\b(?:current|live|today'?s)\s+"
        r"(?:price|weather|score|scores|results?|status|exchange\s+rate|stock\s+price)\s+"
        r"(?:of|for|in)?\s*[\"']?.+?[\"']?\s*$",
    ),
    _rule(
        "web_who_is_current", "web.search", "query_full",
        r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:current|new|latest)\s+.+$",
    ),
    _rule(
        "web_what_is_latest", "web.search", "query_full",
        r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:current|latest|newest)\s+"
        r"(?:version|release|price|weather|score|status|exchange\s+rate)\b.*$",
    ),
    _rule(
        "web_check_latest", "web.search", "query_full",
        r"\b(?:check|look\s*up|find)\b.{0,40}"
        r"\b(?:latest|current|recent|today'?s)\b.{2,120}$",
    ),
    # Browser
    _rule(
        "browser_open_url", "browser.open_url", "url",
        r"\bopen\s+(?:the\s+)?(?:url\s+|page\s+(?:at\s+)?)?(https?://\S+)\s+in\s+(?:the\s+)?browser",
    ),
    _rule(
        "browser_browse_to_url", "browser.open_url", "url",
        r"\bbrowse\s+(?:to\s+)?(https?://\S+)",
    ),
    _rule(
        "browser_get_page_text", "browser.get_page_text", "url",
        r"\bget\s+(?:the\s+)?page\s+text\s+(?:from\s+|of\s+)?(https?://\S+)",
    ),
    _rule(
        "browser_screenshot_url", "browser.screenshot", "url",
        r"\bscreenshot\s+(?:of\s+)?(https?://\S+)",
    ),
    _rule(
        "browser_take_screenshot", "browser.screenshot", "url",
        r"\btake\s+(?:a\s+)?screenshot(?:\s+of\s+(https?://\S+))?",
    ),
    # Calendar
    _rule(
        "calendar_list_events", "calendar.list_events", "date",
        r"\b(?:list|show|what(?:'s| are)?)\s+(?:my\s+)?(?:calendar\s+)?events?\s+"
        r"(?:for\s+|on\s+)(\S+)",
    ),
    _rule(
        "calendar_check_calendar", "calendar.list_events", "date",
        r"\b(?:check|view)\s+(?:my\s+)?calendar\s+(?:for\s+|on\s+)(\S+)",
    ),
    _rule(
        "calendar_create_event", "calendar.create_event", "title_date",
        r"\bcreate\s+(?:a\s+)?(?:calendar\s+)?event\s+[\"']([^\"']+)[\"']\s+"
        r"(?:on|at|from)\s+(\S+)",
    ),
)


def match_intent_rule(
    user_message: str,
    available_tools: set[str],
) -> tuple[IntentRule, dict[str, Any]] | None:
    """Return the first rule (with extracted arguments) that fires.

    Exposed separately from :func:`detect_tool_intent` so tests can assert
    which named rule captured an utterance.
    """
    for rule in INTENT_RULES:
        if rule.tool not in available_tools:
            continue
        for pattern in rule.patterns:
            m = pattern.search(user_message)
            if m is None:
                continue
            args = _extract_args(m, rule.extractor)
            if args is not None:
                return rule, args
    return None


def detect_tool_intent(
    user_message: str,
    available_tools: set[str],
) -> ToolIntent | None:
    """Detect if a user message warrants tool use via heuristics.

    Returns a ToolIntent if an obvious action request is found,
    None for conversational messages.
    """
    matched = match_intent_rule(user_message, available_tools)
    if matched is None:
        return None
    rule, args = matched
    return ToolIntent(
        tool_name=rule.tool,
        arguments=args,
        reason=f"User requested: {rule.tool}",
        expected_outcome=f"Execute {rule.tool}",
    )


def detect_capability_gap(user_message: str, available_tools: set[str]) -> dict[str, Any] | None:
    """Detect a likely capability gap when no current tool can satisfy the request."""
    text = (user_message or "").lower()
    if "pdf" in text and not any("pdf" in name.lower() for name in available_tools):
        return {
            "id": "gap:pdf_reader",
            "gap_type": "pdf_reader",
            "recent_recurrence": 1,
            "frustration_intensity": 0.4,
        }
    if re.search(r"\b(transcribe|speech[- ]to[- ]text|audio)\b", text) and not any(
        "transcribe" in name.lower() or "speech" in name.lower() or "audio" in name.lower()
        for name in available_tools
    ):
        return {
            "id": "gap:speech_to_text",
            "gap_type": "speech_to_text",
            "recent_recurrence": 1,
            "frustration_intensity": 0.35,
        }

    gap_patterns = (
        (
            "image_generation",
            r"\b(?:generate|create|draw|edit)\b.{0,60}"
            r"\b(?:image|picture|photo|avatar|logo)\b",
        ),
        (
            "video_processing",
            r"\b(?:video|clip|movie)\b.{0,80}"
            r"\b(?:summari[sz]e|transcribe|edit|analy[sz]e)\b",
        ),
        (
            "spreadsheet",
            r"(?:\b(?:xlsx|spreadsheet|excel|csv)\b.{0,80}"
            r"\b(?:analy[sz]e|chart|pivot|clean|merge)\b|"
            r"\b(?:analy[sz]e|chart|pivot|clean|merge)\b.{0,80}"
            r"\b(?:xlsx|spreadsheet|excel|csv)\b)",
        ),
        (
            "database",
            r"\b(?:query|inspect|migrate|connect)\b.{0,80}"
            r"\b(?:database|postgres|mysql|sqlite|sql)\b",
        ),
        (
            "code_execution",
            r"\b(?:run|execute|debug|test)\b.{0,80}"
            r"\b(?:code|script|notebook|program)\b",
        ),
    )
    available_blob = " ".join(name.lower() for name in available_tools)
    for gap_type, pattern in gap_patterns:
        if re.search(pattern, text) and not all(
            part in available_blob for part in gap_type.split("_")
        ):
            return {
                "id": f"gap:{gap_type}",
                "gap_type": gap_type,
                "recent_recurrence": 1,
                "frustration_intensity": 0.3,
            }
    return None


def _extract_args(
    match: re.Match, extractor: str,
) -> dict[str, Any] | None:
    """Extract structured arguments from a regex match."""
    groups = match.groups()

    if extractor == "default_root":
        return {"path": "/"}
    elif extractor == "empty":
        return {}
    elif extractor == "skill_request":
        request = re.sub(r"\s+", " ", match.string or "").strip()
        return {
            "request": request,
            "name_hint": _extract_skill_name_hint(request),
            "source_url": _extract_source_url(request),
            "enable": True,
        } if request else None
    elif extractor == "optional_skill_id":
        return {"skill_id": _extract_skill_id_hint(match.string or "")}
    elif extractor == "default_cwd":
        return {"path": "."}
    elif extractor == "cmd_hostname":
        return {"cmd": "hostname"}
    elif extractor == "cmd_disk_root":
        return {"cmd": "df -h /"}
    elif extractor == "query_full":
        return {"query": _clean_search_query(match.group(0))}

    if not groups:
        return None

    if extractor == "path":
        return {"path": groups[0]}
    elif extractor == "cmd":
        return {"cmd": groups[0]}
    elif extractor == "cmd_capture":
        return {"cmd": groups[0]}
    elif extractor == "cmd_explicit":
        cmd = _clean_shell_command(groups[0])
        return {"cmd": cmd} if cmd else None
    elif extractor == "cmd_cat_path":
        return {"cmd": f"cat {groups[0]}"}
    elif extractor == "query":
        return {"query": _clean_search_query(groups[0])}
    elif extractor == "url":
        return {"url": groups[0]}
    elif extractor == "pattern_path" and len(groups) >= 2:
        return {"pattern": groups[0], "path": groups[1]}
    elif extractor == "content_path" and len(groups) >= 2:
        return {"content": groups[0], "path": groups[1]}
    elif extractor == "path_content" and len(groups) >= 2:
        return {"path": groups[0], "content": groups[1]}
    elif extractor == "date":
        return {"date": groups[0]}
    elif extractor == "title_date" and len(groups) >= 2:
        return {"title": groups[0], "start": groups[1], "end": groups[1]}
    elif extractor == "skill_id":
        return {"skill_id": groups[0]}
    return None


def _clean_search_query(text: str) -> str:
    """Normalize a user phrase into a stable web-search query."""
    query = re.sub(r"\s+", " ", text).strip()
    query = query.strip("`\"' ")
    query = query.rstrip("?.! ")
    return query


def _extract_source_url(text: str) -> str:
    match = re.search(r"https?://\S+", text or "", flags=re.I)
    return match.group(0).rstrip(").,;") if match else ""


def _extract_skill_name_hint(text: str) -> str:
    match = re.search(
        r"\b(?:called|named)\s+([A-Za-z0-9][A-Za-z0-9 _.-]{1,80})",
        text or "",
        flags=re.I,
    )
    return match.group(1).strip() if match else ""


def _extract_skill_id_hint(text: str) -> str:
    match = re.search(
        r"\b(?:skill|capability|module|integration)\s+"
        r"([A-Za-z0-9][A-Za-z0-9._-]{0,79})\b",
        text or "",
        flags=re.I,
    )
    return match.group(1).strip() if match else ""


_NON_COMMAND_STARTS = {
    "a",
    "again",
    "an",
    "anything",
    "here",
    "it",
    "me",
    "my",
    "necessary",
    "needed",
    "now",
    "please",
    "required",
    "right",
    "something",
    "stuff",
    "that",
    "the",
    "there",
    "thing",
    "this",
    "us",
    "we",
    "you",
    "your",
}


def _clean_shell_command(text: str) -> str | None:
    """Normalize an explicitly requested shell command without inventing args."""
    cmd = re.sub(r"\s+", " ", text).strip(" `\"'")
    cmd = cmd.rstrip(".!")
    if not cmd:
        return None
    first = cmd.split()[0].lower()
    if first in _NON_COMMAND_STARTS or first.startswith("-"):
        return None
    if not re.match(r"^[A-Za-z0-9_./~+-][A-Za-z0-9_./~+@%=-]*$", first):
        return None
    return cmd


# ---------------------------------------------------------------------------
# Follow-up context rewriting (fallback router only)
# ---------------------------------------------------------------------------
#
# Terse follow-ups ("now check again", "run it", "give me the raw output")
# carry no routable intent on their own, so before the INTENT_RULES table
# runs, the pipeline prepends the most recent user message that looked
# tool-actionable. This is a context-rewriting concern, not routing: none
# of these phrasings route directly. The native model-tool-call path does
# not use this — it gets a bounded conversation window instead.


@dataclass(frozen=True)
class FollowupRewriteRule:
    """Rewrites a terse follow-up by attaching prior actionable context.

    ``trigger`` decides the current message is a terse tool follow-up;
    ``history_hint`` decides a prior user message looked tool-actionable
    enough to prepend.
    """

    name: str
    trigger: re.Pattern[str]
    history_hint: re.Pattern[str]


FOLLOWUP_REWRITE_RULES: tuple[FollowupRewriteRule, ...] = (
    FollowupRewriteRule(
        name="terse_tool_followup",
        trigger=re.compile(
            r"("
            r"\b(?:issue|run|execute|use)\s+(?:the\s+)?"
            r"(?:needed|required|necessary|right)\s+command\b"
            r"|^\s*(?:go|do\s+it|do\s+that|please\s+do|run\s+it|check\s+again|try\s+again|"
            r"create\s+it|make\s+it|i\s+want\s+you\s+to\s+create\s+it|for\s+yourself|"
            r"now\s+check\s+again|give\s+me\s+(?:the\s+)?(?:raw\s+)?output\b.*|"
            r"show\s+me\s+(?:the\s+)?output\b.*)\s*[.!?]*\s*$"
            r")",
            re.IGNORECASE,
        ),
        history_hint=re.compile(
            r"\b(?:hostname|host\s*name|machine\s+name|uname|/etc/hostname|"
            r"disk|drive|filesystem|storage|space|df\s+-h|"
            r"skill|capability|capabilities|module|integration|registry|"
            r"search|web|internet|fetch|read\s+file|list\s+files|"
            r"cat\s+/|grep|calendar|events?|"
            r"(?:run|execute|issue)\s+(?:the\s+)?(?:command\s+)?[A-Za-z0-9_./~+-]|"
            r"^\s*do\s+[A-Za-z0-9_./~+-])\b",
            re.IGNORECASE,
        ),
    ),
)


def attach_followup_context(
    user_message: str,
    conversation_history: list[dict[str, str]],
) -> str:
    """Attach the prior actionable user request to a terse tool follow-up.

    Returns the message unchanged when it is not a terse follow-up or no
    prior user message looks tool-actionable.
    """
    for rule in FOLLOWUP_REWRITE_RULES:
        if not rule.trigger.search(user_message):
            continue
        for message in reversed(conversation_history):
            if message.get("role") == "user" and message.get("content"):
                content = message["content"]
                if rule.history_hint.search(content):
                    return f"{content}\n{user_message}"
    return user_message


# ---------------------------------------------------------------------------
# Action arbiter — cognitive decision layer
# ---------------------------------------------------------------------------

def make_tool_decision(
    intent: ToolIntent,
    action_vars: ActionVariables,
    category: ToolCategory,
    trust: float,
    agency_decision: AgencyDecision | None = None,
    autonomy_level: str = "autonomous",
) -> ToolDecision:
    """Decide whether to execute, clarify, defer, or refuse.

    Deterministic and testable. Driven by action variables,
    tool category, and trust level.
    """
    normalized_autonomy = _normalize_autonomy_level(autonomy_level)
    agency_action = agency_decision.action if agency_decision else "comply"

    if normalized_autonomy == "off":
        return ToolDecision(
            decision="refuse",
            intent=intent,
            rationale="Autonomy mode is off; tools are not allowed",
        )

    if agency_action in {"refuse", "demand_repair"}:
        return ToolDecision(
            decision="refuse",
            intent=intent,
            rationale=(
                f"Agency stance is {agency_action}: "
                f"{agency_decision.tool_instruction if agency_decision else ''}"
            ).strip(),
        )

    if agency_action == "disengage" and category != ToolCategory.READ_ONLY:
        return ToolDecision(
            decision="refuse",
            intent=intent,
            rationale=(
                "Agency stance is disengage; low energy allows simple read-only "
                "inspection but blocks side-effecting work"
            ),
        )

    if normalized_autonomy == "assisted" and category in (
        ToolCategory.WRITE,
        ToolCategory.DESTRUCTIVE,
        ToolCategory.EXTERNAL_ACTION,
    ):
        return ToolDecision(
            decision="clarify",
            intent=intent,
            rationale=f"Assisted autonomy requires confirmation for {category.value} action",
        )

    if agency_action == "slow_down" and category in (
        ToolCategory.WRITE,
        ToolCategory.DESTRUCTIVE,
        ToolCategory.EXTERNAL_ACTION,
    ):
        return ToolDecision(
            decision="clarify",
            intent=intent,
            rationale="Agency stance is slow_down; risky action needs confirmation",
        )

    if agency_action == "resist" and category in (
        ToolCategory.WRITE,
        ToolCategory.DESTRUCTIVE,
    ):
        return ToolDecision(
            decision="clarify",
            intent=intent,
            rationale="Agency stance is resist; write/destructive action needs a clearer request",
        )

    if normalized_autonomy == "high_risk":
        return ToolDecision(
            decision="execute",
            intent=intent,
            rationale="High-risk autonomy approved within configured tool scope",
        )

    # Refuse: destructive tool + low risk tolerance
    if category == ToolCategory.DESTRUCTIVE and action_vars.risk_tolerance < REFUSE_RISK_TOLERANCE:
        return ToolDecision(
            decision="refuse",
            intent=intent,
            rationale=f"Risk tolerance too low ({action_vars.risk_tolerance:.2f}) for destructive action",
        )

    # Clarify: low autonomy bias → ask first
    if action_vars.autonomy_bias < CLARIFY_AUTONOMY_BIAS:
        return ToolDecision(
            decision="clarify",
            intent=intent,
            rationale=f"Autonomy bias too low ({action_vars.autonomy_bias:.2f}), need confirmation",
        )

    # Clarify: write/destructive + high clarification threshold
    if category in (ToolCategory.WRITE, ToolCategory.DESTRUCTIVE):
        if action_vars.clarification_threshold > CLARIFY_WRITE_THRESHOLD:
            return ToolDecision(
                decision="clarify",
                intent=intent,
                rationale=f"Clarification threshold high ({action_vars.clarification_threshold:.2f}) for {category.value} action",
            )

    # Defer: extremely low urgency (too tired / no drive)
    if action_vars.action_urgency < DEFER_URGENCY_THRESHOLD and category != ToolCategory.READ_ONLY:
        return ToolDecision(
            decision="defer",
            intent=intent,
            rationale=f"Action urgency too low ({action_vars.action_urgency:.2f}), deferring",
        )

    # Execute
    return ToolDecision(
        decision="execute",
        intent=intent,
        rationale="Action approved",
    )


def _normalize_autonomy_level(value: str) -> str:
    return value if value in {"off", "assisted", "autonomous", "high_risk"} else "autonomous"


# ---------------------------------------------------------------------------
# Summarize tool results for the generator (bounded output only)
# ---------------------------------------------------------------------------

def _summarize_for_generator(observations: list[ToolObservation], results: list[ToolResult]) -> str:
    """Build a concise summary of tool execution for the generator prompt.

    The generator sees bounded shell output so it can report observed command
    results instead of guessing. Other tool outputs stay summarized.
    """
    if not observations:
        return ""

    parts: list[str] = []
    for obs, res in zip(observations, results):
        if res.success:
            details: list[str] = [obs.summary]
            if res.side_effect_summary and res.side_effect_summary != "none":
                details.append(res.side_effect_summary)
            metadata = _format_result_metadata(res.metadata)
            if metadata:
                details.append(metadata)
            output = _format_result_output(res)
            if output:
                details.append(output)
            parts.append(f"[{res.tool_name}] " + " | ".join(details))
        else:
            details = [f"Failed: {res.error}"]
            if res.side_effect_summary and res.side_effect_summary != "none":
                details.append(res.side_effect_summary)
            metadata = _format_result_metadata(res.metadata)
            if metadata:
                details.append(metadata)
            output = _format_result_output(res)
            if output:
                details.append(output)
            parts.append(f"[{res.tool_name}] " + " | ".join(details))

    return "\n".join(parts)


def _format_result_output(result: ToolResult, limit: int = 1200) -> str:
    if result.tool_name not in {
        "shell.run_command",
        "system.hostname",
        "system.uname",
        "system.disk_usage",
        "system.memory_usage",
        "system.installed_packages",
        "web.search",
        "web.fetch",
        "web.extract_text",
        "skills.audit",
        "skills.create_from_request",
        "skills.import_markdown",
    } or not result.output:
        return ""
    text = result.output.strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return f"output={text!r}"


# ---------------------------------------------------------------------------
# Main tool loop
# ---------------------------------------------------------------------------

def _with_character_state(
    action_vars: ActionVariables,
    life_influence: LifeInfluence | None,
    character_vector: CharacterVector | None,
    category: ToolCategory,
) -> ActionVariables:
    if character_vector is not None:
        return apply_character_vector_to_action_variables(
            action_vars,
            character_vector,
            read_only_action=category == ToolCategory.READ_ONLY,
        )
    if life_influence is None or life_influence.is_neutral:
        return action_vars
    return apply_life_influence_to_action_variables(
        action_vars,
        life_influence,
        read_only_action=category == ToolCategory.READ_ONLY,
    )


def _action_effects(before: ActionVariables, after: ActionVariables) -> dict[str, float]:
    return {
        key: value for key, value in action_variable_deltas(before, after).items()
        if value != 0.0
    }


def _build_step_gate(
    action_vars: ActionVariables,
    trust: float,
    agency_decision: AgencyDecision | None,
    autonomy_level: str,
) -> Callable[[Any, ToolCategory], bool]:
    """Build a per-step authorization gate for plan execution.

    Each plan step is re-run through :func:`make_tool_decision` against its
    *own* tool category, so a benign leading step cannot launder a
    WRITE/DESTRUCTIVE/EXTERNAL_ACTION step past the autonomy arbiter. Only an
    ``execute`` decision allows the step; anything else (clarify/defer/refuse)
    blocks it and halts the plan.
    """

    def _gate(step: Any, category: ToolCategory) -> bool:
        step_intent = ToolIntent(
            tool_name=step.tool_name,
            arguments=step.arguments,
            reason=f"Plan step: {getattr(step, 'description', step.tool_name)[:80]}",
            expected_outcome="",
            urgency=action_vars.action_urgency,
            risk_tolerance=action_vars.risk_tolerance,
            autonomy_bias=action_vars.autonomy_bias,
            clarification_threshold=action_vars.clarification_threshold,
            persistence_drive=action_vars.persistence_drive,
        )
        decision = make_tool_decision(
            step_intent,
            action_vars,
            category,
            trust,
            agency_decision=agency_decision,
            autonomy_level=autonomy_level,
        )
        return decision.decision == "execute"

    return _gate


def run_tool_loop(
    user_message: str,
    state: ModulatorState,
    person: PersonProfile | None,
    defense_active: bool,
    executor: ToolExecutor,
    engine: Any,  # EmotionalEngine — avoid circular import
    max_executions: int | None = None,
    hard_cap: int | None = None,
    active_plan: TaskPlan | None = None,
    agency_decision: AgencyDecision | None = None,
    autonomy_level: str = "autonomous",
    life_influence: LifeInfluence | None = None,
    character_vector: CharacterVector | None = None,
    requested_intent: ToolIntent | None = None,
) -> ToolLoopResult:
    """Run the cognitive tool loop.

    Called from the pipeline after inner dialogue, before defense mechanisms.
    Returns trace, action variables, and a generator-ready summary.

    When ``max_executions`` / ``hard_cap`` are not given, the budget is
    derived from ``autonomy_level`` via :func:`autonomy_execution_budget`.

    If active_plan is provided and the user says "continue"/"next step",
    resumes that plan instead of detecting new intent.
    """
    if max_executions is None or hard_cap is None:
        budget_default, budget_cap = autonomy_execution_budget(autonomy_level)
        if max_executions is None:
            max_executions = budget_default
        if hard_cap is None:
            hard_cap = budget_cap
    trust = person.trust if person else 0.5

    # 1. Derive action variables
    action_vars = derive_action_variables(state, trust=trust, defense_active=defense_active)

    available = set(executor.registry.names())

    # 1a. Check for plan status request
    if active_plan and not active_plan.is_terminal and is_status_request(user_message):
        status_summary = summarize_plan_status(active_plan)
        return ToolLoopResult(
            trace=ToolTrace(),
            action_variables=action_vars,
            tool_context_summary=status_summary,
        )

    # 1b. Check for plan continuation
    if active_plan and not active_plan.is_terminal and is_continue_request(user_message):
        task_trace = execute_plan(
            active_plan, executor, action_vars, engine=engine,
            step_gate=_build_step_gate(
                action_vars, trust, agency_decision, autonomy_level,
            ),
        )
        # Collect results and observations from plan steps
        executed_results, observations = _collect_plan_results(active_plan)
        summary = _summarize_plan_for_generator(active_plan, task_trace)
        return ToolLoopResult(
            trace=ToolTrace(
                executed_results=executed_results,
                observations=observations,
                loop_count=task_trace.steps_executed,
                task_trace=task_trace,
            ),
            action_variables=action_vars,
            tool_context_summary=summary,
        )

    # 2. Detect multi-step intent first
    plan = detect_multi_step_intent(user_message, available)
    if plan is not None:
        # Arbiter check: use first step's tool for category check. An unknown
        # tool fails closed (EXTERNAL_ACTION), matching the per-step gate, so
        # a missing-from-registry tool is never treated as READ_ONLY for a
        # gating decision.
        first_step = plan.steps[0]
        cap = executor.registry.get(first_step.tool_name)
        first_category = cap.category if cap else ToolCategory.EXTERNAL_ACTION
        plan_action_vars = _with_character_state(
            action_vars,
            life_influence,
            character_vector,
            first_category,
        )
        life_effects = _action_effects(action_vars, plan_action_vars)
        # Build a synthetic intent for arbiter
        synthetic_intent = ToolIntent(
            tool_name=first_step.tool_name,
            arguments=first_step.arguments,
            reason=f"Multi-step plan: {plan.goal[:80]}",
            expected_outcome=f"Execute {len(plan.steps)}-step plan",
            urgency=plan_action_vars.action_urgency,
            risk_tolerance=plan_action_vars.risk_tolerance,
            autonomy_bias=plan_action_vars.autonomy_bias,
            clarification_threshold=plan_action_vars.clarification_threshold,
            persistence_drive=plan_action_vars.persistence_drive,
        )
        decision = make_tool_decision(
            synthetic_intent,
            plan_action_vars,
            first_category,
            trust,
            agency_decision=agency_decision,
            autonomy_level=autonomy_level,
        )

        if decision.decision != "execute":
            return ToolLoopResult(
                trace=ToolTrace(
                    proposed_intents=[synthetic_intent],
                    final_decision=decision,
                    loop_count=0,
                    task_trace=TaskTrace(plan=plan, plan_outcome=""),
                ),
                action_variables=plan_action_vars,
                tool_context_summary="",
                life_influence_effects=life_effects,
            )

        # Execute the plan — each step gated against its own category so the
        # first step's category cannot authorize riskier later steps.
        task_trace = execute_plan(
            plan, executor, plan_action_vars, engine=engine,
            step_gate=_build_step_gate(
                plan_action_vars, trust, agency_decision, autonomy_level,
            ),
        )
        executed_results, observations = _collect_plan_results(plan)
        summary = _summarize_plan_for_generator(plan, task_trace)

        return ToolLoopResult(
            trace=ToolTrace(
                proposed_intents=[synthetic_intent],
                final_decision=decision,
                executed_results=executed_results,
                observations=observations,
                loop_count=task_trace.steps_executed,
                task_trace=task_trace,
            ),
            action_variables=plan_action_vars,
            tool_context_summary=summary,
            life_influence_effects=life_effects,
        )

    # 3. Single-step: detect tool intent
    intent = requested_intent or detect_tool_intent(user_message, available)

    # No tool needed → empty trace
    if intent is None:
        gap = detect_capability_gap(user_message, available)
        return ToolLoopResult(
            trace=ToolTrace(),
            action_variables=action_vars,
            tool_context_summary="",
            capability_gaps=[gap] if gap else [],
        )

    # Populate intent with derived action variables
    intent.urgency = action_vars.action_urgency
    intent.risk_tolerance = action_vars.risk_tolerance
    intent.autonomy_bias = action_vars.autonomy_bias
    intent.clarification_threshold = action_vars.clarification_threshold
    intent.persistence_drive = action_vars.persistence_drive

    # 4. Action arbiter
    capability = executor.registry.get(intent.tool_name)
    category = capability.category if capability else ToolCategory.READ_ONLY
    action_vars = _with_character_state(
        action_vars,
        life_influence,
        character_vector,
        category,
    )
    life_effects = _action_effects(
        derive_action_variables(state, trust=trust, defense_active=defense_active),
        action_vars,
    )
    intent.urgency = action_vars.action_urgency
    intent.risk_tolerance = action_vars.risk_tolerance
    intent.autonomy_bias = action_vars.autonomy_bias
    intent.clarification_threshold = action_vars.clarification_threshold
    intent.persistence_drive = action_vars.persistence_drive

    decision = make_tool_decision(
        intent,
        action_vars,
        category,
        trust,
        agency_decision=agency_decision,
        autonomy_level=autonomy_level,
    )

    proposed_intents: list[ToolIntent] = [intent]
    executed_results: list[ToolResult] = []
    observations: list[ToolObservation] = []
    loop_count = 0

    if decision.decision != "execute":
        # Not executing — return trace with decision only
        return ToolLoopResult(
            trace=ToolTrace(
                proposed_intents=proposed_intents,
                final_decision=decision,
                loop_count=0,
            ),
            action_variables=action_vars,
            tool_context_summary="",
            life_influence_effects=life_effects,
        )

    # 5. Execute + appraise loop (bounded)
    current_intent = intent
    while loop_count < min(max_executions, hard_cap):
        loop_count += 1

        # Execute
        result = executor.execute(current_intent.tool_name, current_intent.arguments)
        executed_results.append(result)

        # Appraise
        cap = executor.registry.get(current_intent.tool_name)
        cat = cap.category if cap else ToolCategory.READ_ONLY
        observation = appraise_tool_result(result, cat)
        observations.append(observation)

        # Apply emotional deltas to engine
        _apply_emotional_deltas(engine, observation)

        # Check if we should continue
        if not observation.continue_tool_loop:
            break

    # 6. Build summary for generator
    summary = _summarize_for_generator(observations, executed_results)

    return ToolLoopResult(
        trace=ToolTrace(
            proposed_intents=proposed_intents,
            final_decision=decision,
            executed_results=executed_results,
            observations=observations,
            loop_count=loop_count,
        ),
        action_variables=action_vars,
        tool_context_summary=summary,
        life_influence_effects=life_effects,
    )


def _collect_plan_results(
    plan: TaskPlan,
) -> tuple[list[ToolResult], list[ToolObservation]]:
    """Collect executed results and observations from plan steps."""
    results: list[ToolResult] = []
    observations: list[ToolObservation] = []
    for step in plan.steps:
        if step.result is not None:
            results.append(step.result)
        if step.observation is not None:
            observations.append(step.observation)
    return results, observations


def _summarize_plan_for_generator(
    plan: TaskPlan, task_trace: TaskTrace,
) -> str:
    """Build a concise summary of multi-step plan execution for the generator."""
    parts: list[str] = []
    parts.append(f"[Plan: {plan.goal[:100]}]")
    parts.append(f"Outcome: {task_trace.plan_outcome} "
                 f"({task_trace.steps_succeeded}/{task_trace.steps_executed} succeeded)")

    for step in plan.steps:
        if step.result is not None:
            if step.result.success:
                details = []
                if step.observation is not None:
                    details.append(step.observation.summary)
                if (
                    step.result.side_effect_summary
                    and step.result.side_effect_summary != "none"
                ):
                    details.append(step.result.side_effect_summary)
                metadata = _format_result_metadata(step.result.metadata)
                if metadata:
                    details.append(metadata)
                output = _format_result_output(step.result)
                if output:
                    details.append(output)
                summary = " | ".join(details) if details else "success"
                parts.append(f"  [{step.tool_name}] OK: {summary}")
            else:
                parts.append(f"  [{step.tool_name}] FAIL: {step.result.error}")

    return "\n".join(parts)


def _format_result_metadata(metadata: dict[str, Any]) -> str:
    """Return only compact, low-risk metadata hints for generation."""
    if not metadata:
        return ""
    hints: list[str] = []
    for key in (
        "count",
        "match_count",
        "result_count",
        "bytes_written",
        "exit_code",
        "skill_id",
        "status",
        "enabled",
        "error_count",
        "warning_count",
        "truncated",
    ):
        if key in metadata:
            hints.append(f"{key}={metadata[key]}")
    return ", ".join(hints)


def _apply_emotional_deltas(engine: Any, observation: ToolObservation) -> None:
    """Apply an observation's emotional deltas to the engine state."""
    state = engine.state
    for modulator, delta in observation.emotional_delta.items():
        if hasattr(state, modulator):
            current = getattr(state, modulator)
            new_val = max(0.0, min(1.0, current + delta))
            setattr(state, modulator, new_val)
