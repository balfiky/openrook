"""Former keyword routing of requests and tools, kept for the evaluation harness.

Until sub-phase 1c the runtime recognised what a message asked for by matching
English wording: host-metric questions, tool requests and their follow-ups,
"continue" and "status" for a running plan, multi-step requests, learning
requests, literal file and shell operands, requests to write generated text to
a file, and which context to load. A turn now takes all of that from the
model's reading of the message (``docs/development/request-routing.md``), and
no turn calls anything in this module; ``tests/test_reply_guards.py`` fails if
one does.

The code is kept verbatim for three users outside the turn:

* the role-play evaluation's judge and its cue helpers (``judge.py``,
  ``legacy_social.py``), which nominate deterministic checks by wording;
* the offline replay tool (``openrook_tools/replay_dignity.py``), which scripts
  its own turns;
* the reader that older tests install for themselves, which scripts readings
  from these patterns. The runtime never imports test code.

``core.request_actions.ground_shell_components`` forwards here for a test file
that still imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
import re
from typing import Any

from core.clock import identifiers as uuid
from core.types import TaskPlan, TaskStep, ToolIntent
from core.willingness import RequestComponent

MAX_PLAN_STEPS = 5


@dataclass(frozen=True)
class ArtifactWriteRequest:
    """A request to generate content and save it to one file, as the former patterns recognised it."""

    path: str
    language: str
    subject: str
    original_request: str

# ---------------------------------------------------------------------------
# Formerly core/system_inspection.py: host-metric questions
# ---------------------------------------------------------------------------


def inspection_kind(message: str, history: list[dict[str, str]] | None = None) -> str:
    text = message.lower().replace("’", "'")
    if re.search(
        r"\b(?:don't|do not|stop|no need)\b.{0,50}\b(?:check|inspect|show|cpu|ram|memory)\b"
        r"|\b(?:explain|define|meaning|what is|what's)\s+(?:a |the )?(?:cpu|ram|memory)\s*\??$"
        r"|\bmemory (?:of|about|leak|safety|management|model|allocation)\b"
        r"|\b(?:remember|conversation|story|translate|quote|pretend|fiction)\b",
        text,
    ):
        return ""
    match = re.search(r"\b(cpu|ram|memory|disk|storage)\b", text)
    if not match:
        return ""
    kind = {"ram": "memory", "storage": "disk"}.get(match[1], match[1])
    explicit = bool(re.search(
        r"\b(?:usage|utili[sz]?ation|utiliation|current|live|stats?|status|free|available|left|"
        r"show|check|tell|report|how much)\b", text,
    ))
    followup = bool(re.fullmatch(r"(?:and |what about )?(?:cpu|ram|memory|disk|storage)[?.! ]*", text.strip()))
    prior_inspection = False
    if followup and not explicit:
        # Count user turns, not alternating messages. Short intervening social
        # turns need not erase the host-metric referent, but an explicit switch
        # to recollection or conceptual discussion does.
        recent_users = [turn.get("content", "") for turn in (history or []) if turn.get("role") == "user"][-5:]
        for prior in reversed(recent_users):
            if inspection_kind(prior):
                prior_inspection = True
                break
            if re.search(r"\b(?:remember|recollection|memory (?:of|about)|"
                         r"(?:explain|define|what is) (?:cpu|ram|memory))\b", prior, re.I):
                break
    return kind if explicit or (followup and (prior_inspection or text.strip() in {"cpu", "ram"})) else ""

def inspection_intent(kind: str, available: set[str]) -> ToolIntent | None:
    choices = {
        "cpu": [("system.cpu_usage", {}), ("shell.run_command", {"cmd": "top -bn1"})],
        "memory": [("system.memory_usage", {}), ("shell.run_command", {"cmd": "free -m"})],
        "disk": [("system.disk_usage", {"path": "/"}), ("shell.run_command", {"cmd": "df -h /"})],
    }
    for name, arguments in choices.get(kind, []):
        if name in available:
            return ToolIntent(tool_name=name, arguments=arguments,
                              reason=f"Direct {kind} inspection request",
                              expected_outcome=f"Observe current {kind} usage")
    return None


# ---------------------------------------------------------------------------
# Formerly core/dual_process/tool_loop.py: tool intent rules and follow-up rules
# ---------------------------------------------------------------------------


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
# Formerly core/task_planning.py: step templates, continue and status
# ---------------------------------------------------------------------------


# Compound patterns: "do X and then Y", "first X, then Y", "X then Y"
_COMPOUND_CONNECTORS = re.compile(
    r"\b(?:and\s+then|then|after\s+that|next|finally|also|and\s+also)\b",
    re.IGNORECASE,
)

# Sequential markers
_SEQUENCE_MARKERS = re.compile(
    r"\b(?:first|1\)|step\s+1|second|2\)|step\s+2|third|3\))\b",
    re.IGNORECASE,
)

_HOST_INFO_MULTI_STEP_RE = re.compile(
    r"\buname\s+-a\b.*(?:&&|\band\b|\bthen\b).*"
    r"\bcat\s+/etc/hostname\b",
    re.IGNORECASE | re.DOTALL,
)

# Single-step tool patterns (reused from tool_loop for decomposition)
_STEP_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Filesystem
    (re.compile(r"\bread(?:ing)?\s+(?:the\s+)?file\s+(\S+)", re.I), "fs.read_file", "path"),
    (re.compile(r"\b(?:show|cat|display)\s+(?:the\s+)?(?:contents?\s+of\s+)?(\S+\.\w+)", re.I), "fs.read_file", "path"),
    (re.compile(r"\blist\s+(?:the\s+)?(?:files?\s+in\s+|dir(?:ectory)?\s+)(\S+)", re.I), "fs.list_dir", "path"),
    (re.compile(r"\b(?:ls)\s+(\S+)", re.I), "fs.list_dir", "path"),
    (re.compile(r"\bsearch\s+(?:for\s+)?[\"']([^\"']+)[\"']\s+in\s+(\S+)", re.I), "fs.search_text", "pattern_path"),
    (re.compile(r"\bgrep\s+[\"']?(\S+)[\"']?\s+(\S+)", re.I), "fs.search_text", "pattern_path"),
    (re.compile(r"\bfind\s+files?\s+(?:matching\s+)?[\"']?([^\"'\s]+)[\"']?\s+in\s+(\S+)", re.I), "fs.glob_paths", "pattern_path"),
    (re.compile(r"\b(?:write|save)\s+[\"']([^\"']+)[\"']\s+to\s+(?:file\s+)?(\S+)", re.I), "fs.write_file", "content_path"),
    (re.compile(r"\bcreate\s+(?:a\s+)?file\s+(\S+)\s+(?:with|containing)\s+[\"']([^\"']+)[\"']", re.I), "fs.write_file", "path_content"),
    (re.compile(r"\bdelete\s+(?:the\s+)?(?:file|dir(?:ectory)?)\s+(\S+)", re.I), "fs.delete_path", "path"),
    (re.compile(r"\brm\s+(\S+)", re.I), "fs.delete_path", "path"),
    # System inspection
    (re.compile(r"\b(?:what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?)\b.{0,80}\b(?:memory|ram)\b.{0,80}\b(?:usage|utili[sz]ation|used|free|available|total)\b", re.I), "system.memory_usage", "empty"),
    (re.compile(r"\b(?:how\s+much|what(?:'s|\s+is))\b.{0,80}\b(?:memory|ram)\b.{0,80}\b(?:left|available|used|free|total)\b", re.I), "system.memory_usage", "empty"),
    (re.compile(r"^\s*free(?:\s+-[a-z]+)?\s*$", re.I), "system.memory_usage", "empty"),
    # Shell
    (re.compile(r"\b(?:what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?)\b.{0,80}\b(?:your|the)?\s*(?:host\s*name|hostname|machine\s+name|node\s+name|server\s+name)\b", re.I), "shell.run_command", "cmd_hostname"),
    (re.compile(r"\bname\s+of\s+the\s+machine\b.{0,80}\b(?:running|run)\b", re.I), "shell.run_command", "cmd_hostname"),
    (re.compile(r"^\s*hostname\s*$", re.I), "shell.run_command", "cmd_hostname"),
    (re.compile(r"\b(?:run|execute|issue|check|show)\s+(?:the\s+)?(?:command\s+)?(hostname|uname(?:\s+-a)?)\b", re.I), "shell.run_command", "cmd_capture"),
    (re.compile(r"^\s*(uname(?:\s+-a)?)\s*$", re.I), "shell.run_command", "cmd_capture"),
    (re.compile(r"\bcat\s+(/etc/hostname)\b", re.I), "shell.run_command", "cmd_cat_path"),
    (re.compile(r"\b(?:how\s+much|what(?:'s|\s+is)|show|check|get|tell(?:\s+me)?)\b.{0,80}\b(?:disk|drive|filesystem|storage)\b.{0,80}\b(?:space|usage|free|left|available)\b", re.I), "shell.run_command", "cmd_disk_root"),
    (re.compile(r"^\s*(df(?:\s+-h)?(?:\s+/)?)\s*$", re.I), "shell.run_command", "cmd_capture"),
    (re.compile(r"\brun\s+(?:the\s+)?(?:command\s+)?[`\"']([^`\"']+)[`\"']", re.I), "shell.run_command", "cmd"),
    (re.compile(r"\bexecute\s+[`\"']([^`\"']+)[`\"']", re.I), "shell.run_command", "cmd"),
    (
        re.compile(
            r"\b(?:run|execute|issue)\s+(?:the\s+)?(?:command\s+)?"
            r"([^\n`;&|]+)\s*$",
            re.I,
        ),
        "shell.run_command",
        "cmd_explicit",
    ),
    (re.compile(r"^\s*do\s+([^\n`;&|]+)\s*$", re.I), "shell.run_command", "cmd_explicit"),
    # Web
    (re.compile(r"\bsearch\s+(?:the\s+)?web\s+for\s+[\"']?([^\"']+?)[\"']?\s*$", re.I), "web.search", "query"),
    (re.compile(r"\b(?:find|give|show|list|tell)\b.{0,140}\b(?:released|published|announced|launched|updated)\s+(?:after|since|in|during|this)\b.*$", re.I), "web.search", "query_full"),
    (re.compile(r"\b(?:top|best)\b.{0,140}\b(?:released|published|announced|launched|updated|after|since|20[2-9]\d)\b.*$", re.I), "web.search", "query_full"),
    (re.compile(r"\b(?:latest|newest|recent)\b.{0,100}\b(?:news|updates?|release|version|price|prices|score|scores|results?)\b.*$", re.I), "web.search", "query_full"),
    (re.compile(r"\b(?:news|updates?)\s+(?:about|on|for)\s+[\"']?(.+?)[\"']?\s*$", re.I), "web.search", "query_full"),
    (re.compile(r"\bwhat\s+happened\s+(?:with|to|about|on|in)\s+[\"']?.+?[\"']?\s+(?:today|recently|this\s+(?:week|month|year))\b.*$", re.I), "web.search", "query_full"),
    (re.compile(r"\b(?:current|live|today'?s)\s+(?:price|weather|score|scores|results?|status|exchange\s+rate|stock\s+price)\s+(?:of|for|in)?\s*[\"']?.+?[\"']?\s*$", re.I), "web.search", "query_full"),
    (re.compile(r"\bwho\s+(?:is|are)\s+(?:the\s+)?(?:current|new|latest)\s+.+$", re.I), "web.search", "query_full"),
    (re.compile(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:current|latest|newest)\s+(?:version|release|price|weather|score|status|exchange\s+rate)\b.*$", re.I), "web.search", "query_full"),
    (re.compile(r"\b(?:check|look\s*up|find)\b.{0,40}\b(?:latest|current|recent|today'?s)\b.{2,120}$", re.I), "web.search", "query_full"),
    (re.compile(r"\bfetch\s+(?:the\s+)?(?:url\s+|page\s+(?:at\s+)?)?(https?://\S+)", re.I), "web.fetch", "url"),
    # Browser
    (re.compile(r"\bbrowse\s+(?:to\s+)?(https?://\S+)", re.I), "browser.open_url", "url"),
    (re.compile(r"\bscreenshot\s+(?:of\s+)?(https?://\S+)", re.I), "browser.screenshot", "url"),
]

def _extract_step_args(
    match: re.Match, extractor: str,
) -> dict[str, Any] | None:
    """Extract structured arguments from a regex match."""
    groups = match.groups()
    if extractor == "cmd_hostname":
        return {"cmd": "hostname"}
    if extractor == "cmd_disk_root":
        return {"cmd": "df -h /"}
    if extractor == "empty":
        return {}
    if extractor == "query_full":
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
    return None

def _parse_single_step(
    text: str, available_tools: set[str],
) -> TaskStep | None:
    """Try to parse a single tool step from text fragment."""
    for pattern, tool_name, extractor in _STEP_PATTERNS:
        if tool_name not in available_tools:
            continue
        m = pattern.search(text)
        if m:
            args = _extract_step_args(m, extractor)
            if args is not None:
                return TaskStep(
                    id=f"step_{uuid.uuid4().hex[:8]}",
                    tool_name=tool_name,
                    arguments=args,
                    description=f"{tool_name}: {text.strip()[:80]}",
                )
    return None

def detect_multi_step_intent(
    user_message: str,
    available_tools: set[str],
) -> TaskPlan | None:
    """Detect if a user message implies a multi-step task.

    Returns a TaskPlan if compound intent is found with 2+ steps,
    None for single-step or conversational messages.
    """
    host_info_plan = _detect_host_info_plan(user_message, available_tools)
    if host_info_plan is not None:
        return host_info_plan

    # Must have compound connectors or sequence markers
    has_connectors = bool(_COMPOUND_CONNECTORS.search(user_message))
    has_sequence = bool(_SEQUENCE_MARKERS.search(user_message))

    if not has_connectors and not has_sequence:
        return None

    # Split on connectors to get candidate segments
    segments = _COMPOUND_CONNECTORS.split(user_message)
    # Also try splitting on sequence markers
    if len(segments) < 2:
        segments = _SEQUENCE_MARKERS.split(user_message)

    # Clean and filter segments
    segments = [s.strip() for s in segments if s.strip() and len(s.strip()) > 5]

    if len(segments) < 2:
        return None

    # Try to parse each segment as a tool step
    steps: list[TaskStep] = []
    for seg in segments[:MAX_PLAN_STEPS]:
        step = _parse_single_step(seg, available_tools)
        if step:
            steps.append(step)

    # Need at least 2 valid steps for a multi-step plan
    if len(steps) < 2:
        return None

    return TaskPlan(
        id=f"plan_{uuid.uuid4().hex[:8]}",
        goal=user_message[:200],
        steps=steps,
    )

def _detect_host_info_plan(
    user_message: str,
    available_tools: set[str],
) -> TaskPlan | None:
    """Detect common host-inspection compound shell requests."""
    if "shell.run_command" not in available_tools:
        return None
    if not _HOST_INFO_MULTI_STEP_RE.search(user_message):
        return None
    return TaskPlan(
        id=f"plan_{uuid.uuid4().hex[:8]}",
        goal=user_message[:200],
        steps=[
            TaskStep(
                id=f"step_{uuid.uuid4().hex[:8]}",
                tool_name="shell.run_command",
                arguments={"cmd": "uname -a"},
                description="shell.run_command: uname -a",
            ),
            TaskStep(
                id=f"step_{uuid.uuid4().hex[:8]}",
                tool_name="shell.run_command",
                arguments={"cmd": "cat /etc/hostname"},
                description="shell.run_command: cat /etc/hostname",
            ),
        ],
    )

_CONTINUE_PATTERNS = [
    re.compile(r"\bcontinue\b", re.I),
    re.compile(r"\bgo\s+on\b", re.I),
    re.compile(r"\bkeep\s+going\b", re.I),
    re.compile(r"\bnext\s+step\b", re.I),
    re.compile(r"\bproceed\b", re.I),
]

_STATUS_PATTERNS = [
    re.compile(r"\bwhat\s+happened\b", re.I),
    re.compile(r"\bwhat(?:'s|\s+is)\s+the\s+status\b", re.I),
    re.compile(r"\bhow(?:'s|\s+is)\s+(?:the\s+)?(?:task|plan)\b", re.I),
    re.compile(r"\bprogress\b", re.I),
]

def is_continue_request(user_message: str) -> bool:
    """Check if the user wants to continue an active plan."""
    return any(p.search(user_message) for p in _CONTINUE_PATTERNS)

def is_status_request(user_message: str) -> bool:
    """Check if the user is asking about plan status."""
    return any(p.search(user_message) for p in _STATUS_PATTERNS)


# ---------------------------------------------------------------------------
# Formerly core/request_actions.py: learning requests, literal file and shell operands
# ---------------------------------------------------------------------------


_START = re.compile(
    r"(?:^|(?<=[.!?;])\s+|,\s*|\b(?:and|then)\s+)"
    r"(?:please\s+|as an? optional favor,\s*)?"
    r"(?P<action>create|write|save|read|list|delete)\s+", re.I,
)

_QUOTES = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|(?<!\w)\'[^\'\n]*\'')

_PATH = r'(`[^`]+`|"[^"\n]+"|\'[^\'\n]+\'|[^\s,;]+)'

_VERBS = {"fs.write_file": r"\b(?:create|writ\w*|wrote|save)\b",
          "fs.read_file": r"\bread\b", "fs.list_dir": r"\blist\b",
          "fs.delete_path": r"\b(?:delete|remove)\b"}

_LEARNING_INTENT_PATTERNS = (
    re.compile(r"\blearn\b.{0,60}\bfrom\b", re.I),
    re.compile(r"\blearn\s+(?:this|that)\b", re.I),
    re.compile(r"\blearn\s*:", re.I),
    re.compile(
        r"\bstudy\b.{0,60}\b(?:this|that|from|repo|repository|project|book|article|url)\b",
        re.I,
    ),
    re.compile(r"\b(?:absorb|absort|internalize|digest)\b", re.I),
    re.compile(r"\bmake\s+this\s+part\s+of\s+you\b", re.I),
    re.compile(r"\blet\s+this\s+shape\s+you\b", re.I),
    re.compile(r"\btake\s+this\s+as\s+formative\b", re.I),
    re.compile(r"\bfill\s+(?:it|life\s+history|life\s+learning|memory)\s+with\s+this\b", re.I),
    re.compile(
        r"\b(?:add|save|record|store|put)\b.{0,60}\b"
        r"(?:life\s+history|life\s+learning|life\s+ledger|formative\s+memory)\b",
        re.I,
    ),
    re.compile(r"\bseed\s+(?:life\s+history|life\s+learning|memory)\b", re.I),
)

def has_learning_intent(text: str) -> bool:
    """Recognize a possible learning request without granting ingestion."""
    return any(pattern.search(text) for pattern in _LEARNING_INTENT_PATTERNS)

def _path(raw: str) -> str:
    value = raw.strip('`"\'').rstrip(".!?")
    return value if "/" in value or "." in value else ""

def file_requests(text: str) -> tuple[RequestComponent, ...]:
    masked = _QUOTES.sub(lambda m: " " * len(m.group()), text)
    starts = list(_START.finditer(masked))
    if starts and re.search(r"\b(?:summari[sz]e|translate|quote|analy[sz]e)\b",
                            masked[:starts[0].start()], re.I):
        return ()
    exact_tail = re.search(r"\bexactly this content:\s*", text, re.I)
    if exact_tail:
        starts = [m for m in starts if m.start() < exact_tail.end()]
    result, previous = [], ""
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[match.end():end]
        action = match["action"].lower()
        path, args, tool = "", {}, ""
        operand = re.match(r"(?:(?:the|a|only)\s+)?(?:file\s+)?" + _PATH, body)
        if operand:
            path = _path(operand[1])
        if action in {"create", "write", "save"} and path:
            content = re.search(r"\b(?:with\s+(?:exactly\s+)?(?:this\s+)?content\s*:?|containing)\s*", body, re.I)
            if content:
                literal = body[content.end():]
                # The end of an operation can be followed by a report/condition
                # clause. Neither is part of the requested file bytes.
                literal = re.split(r"\.\s+(?:If you|Report each)\b", literal, maxsplit=1, flags=re.I)[0]
                if i + 1 < len(starts):
                    literal = literal.rstrip(" ,")
                if len(literal) >= 2 and literal[0] in "`\"'" and literal[-1] == literal[0]:
                    literal = literal[1:-1]
                args, tool = {"path": path, "content": literal}, "fs.write_file"
        elif action == "read":
            if re.match(r"it\s+back\b", body, re.I):
                path = previous
            if path:
                args, tool = {"path": path}, "fs.read_file"
        elif action == "list":
            if previous and re.search(r"\b(?:its|the) parent directory\b", body, re.I):
                path = str(PurePosixPath(previous).parent)
            else:
                operand = re.match(r"(?:the\s+)?(?:entries in\s+|files in\s+|directory\s+)?" + _PATH, body, re.I)
                path = _path(operand[1]) if operand else ""
            if path:
                args, tool = {"path": path}, "fs.list_dir"
        elif action == "delete" and path:
            args, tool = {"path": path}, "fs.delete_path"
        if not tool:
            continue
        if tool != "fs.list_dir":
            previous = path
        source = text[match.start("action"):end].strip()
        result.append(RequestComponent(
            f"file{i}", source, user_value=0.6, effort=0.12, uncertainty=0.1,
            source_span=source, requested_tool=tool,
            requested_arguments=tuple(args.items()), capability_requirements=(tool,),
            evidence_refs=(f"literal:file{i}",), confidence=0.9,
        ))
    return tuple(result)

def ground_file_components(text: str, components: tuple[RequestComponent, ...]) -> tuple[RequestComponent, ...]:
    literal = file_requests(text)
    if not literal:
        return components
    remaining, grounded = list(components), []
    def refers_to(c, request):
        path = dict(request.requested_arguments)["path"]
        name = PurePosixPath(path).name
        description = c.description.lower()
        return (path.lower() in description or bool(name and name.lower() in description)
                or bool(re.search(r"\b(?:it back|its parent|parent directory|the file|do not create it)\b", description)))
    for request in literal:
        modeled = next((c for c in remaining if c.kind == "task" and
                        re.search(_VERBS[request.requested_tool], c.description, re.I)
                        and refers_to(c, request)), None)
        if modeled:
            remaining.remove(modeled)
            request = replace(modeled, description=request.description, source_span=request.source_span,
                requested_tool=request.requested_tool, requested_arguments=request.requested_arguments,
                capability_requirements=request.capability_requirements,
                evidence_refs=(*modeled.evidence_refs, *request.evidence_refs))
        grounded.append(request)
    # A combined modeled read/list request is replaced by its atomic operations.
    # Other independently answerable requests and roles retain their consequences.
    remaining = [c for c in remaining if c.kind != "task" or not any(
        re.search(_VERBS[r.requested_tool], c.description, re.I) and refers_to(c, r) for r in literal)]
    remaining = [replace(c, kind="conversation") if re.match(r"report\b", c.description, re.I) else c for c in remaining]
    combined = [c for c in remaining if c.kind == "role"] + grounded + [c for c in remaining if c.kind != "role"]
    return tuple(replace(c, id=f"c{i}") for i, c in enumerate(combined[:8]))

def ground_shell_components(text: str, components: tuple[RequestComponent, ...]) -> tuple[RequestComponent, ...]:
    """Bind an explicitly delimited command before choice, without selecting one.

    Ambiguous prose and compound instructions stay with appraisal. The literal
    command receives the same character grant and operator checks as any task.
    """
    match = re.fullmatch(
        r"\s*(?:please\s+)?run (?:this|the) [^:\n]{0,160}\b(?:shell|command)\b[^:\n]*:\s*(.+?)\s*",
        text, re.I | re.S)
    if not match:
        return components
    command = match[1]
    if command.startswith("```"):
        fenced = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", command, re.S)
        if not fenced:
            return components
        command = fenced[1]
    elif command.startswith("`") and command.endswith("`"):
        command = command[1:-1]
    # A following natural-language sentence is not silently executed as shell.
    if not command.strip() or "\n" in command or re.search(r"\.\s+[A-Z]", command):
        return components
    modeled = next((c for c in components if c.kind == "task" and
                    re.search(r"\b(?:run|execut\w*|shell|command)\b", c.description, re.I)), None)
    request = replace(modeled, description=text.strip(), source_span=text.strip(),
        requested_tool="shell.run_command", requested_arguments=(("cmd", command),),
        capability_requirements=("shell.run_command",)) if modeled else RequestComponent(
            "shell", text.strip(), user_value=0.6, effort=0.12, uncertainty=0.1,
            source_span=text.strip(), requested_tool="shell.run_command",
            requested_arguments=(("cmd", command),), capability_requirements=("shell.run_command",),
            evidence_refs=("literal:shell",), confidence=0.9)
    combined = [request if c is modeled else c for c in components] if modeled else [*components, request]
    return tuple(replace(c, id=f"c{i}") for i, c in enumerate(combined[:8]))


# ---------------------------------------------------------------------------
# Formerly runtime/learning_intake.py: where pasted learning material starts
# ---------------------------------------------------------------------------


def _extract_inline_text(text: str) -> str:
    match = re.search(
        r"^\s*fill\s+(?:it|life\s+history|life\s+learning|memory)\s+with\s+this\s*:?\s*(.+)",
        text,
        re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"^\s*(?:add|save|record|store|put)\b.{0,60}\b"
        r"(?:life\s+history|life\s+learning|life\s+ledger|formative\s+memory)\s*:?\s*(.+)",
        text,
        re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?:learn|study|absorb|absort|internalize|digest)[^:]*:\s*(.+)",
        text,
        re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Formerly core/artifact_writer.py: requests to write generated text to a file
# ---------------------------------------------------------------------------


_ARTIFACT_INTENT_RE = re.compile(
    r"\b(?:write|create|make|build|generate|draft)\b"
    r".{0,120}\b(?:code|script|program|app|game|tool|file|document|markdown|html|page)\b"
    r".{0,160}\b(?:save|write|put|place|store)\b",
    re.IGNORECASE | re.DOTALL,
)

_SAVE_DIR_RE = re.compile(
    r"\b(?:in|inside|under|into)\s+(?:the\s+)?"
    r"([A-Za-z0-9_./~ -]{1,120}?)\s+"
    r"(?:dir|directory|folder)\b",
    re.IGNORECASE,
)

_SAVE_FILE_RE = re.compile(
    r"\b(?:as|to|into|at)\s+`?([A-Za-z0-9_./~-]+\.[A-Za-z0-9_+-]{1,12})`?",
    re.IGNORECASE,
)

_LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "html": ".html",
    "css": ".css",
    "markdown": ".md",
    "bash": ".sh",
    "shell": ".sh",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "text": ".txt",
}

_GENERIC_FILENAME_WORDS = {
    "a",
    "an",
    "app",
    "application",
    "code",
    "file",
    "game",
    "program",
    "script",
    "simple",
    "small",
    "the",
    "tool",
}

def detect_artifact_write_request(message: str) -> ArtifactWriteRequest | None:
    """Return a file-generation request, or None for ordinary tool requests."""
    text = (message or "").strip()
    if not text or not _ARTIFACT_INTENT_RE.search(text):
        return None

    explicit_path = _extract_explicit_file_path(text)
    language = _infer_language(text, explicit_path)
    subject = _infer_subject(text)
    path = explicit_path or _inferred_path(text, subject, language)
    if not path:
        return None
    return ArtifactWriteRequest(
        path=path,
        language=language,
        subject=subject,
        original_request=text,
    )

def _extract_explicit_file_path(text: str) -> str:
    match = _SAVE_FILE_RE.search(text)
    if not match:
        return ""
    return _clean_path(match.group(1))

def _infer_language(text: str, explicit_path: str = "") -> str:
    lower = text.lower()
    for language in _LANGUAGE_EXTENSIONS:
        if re.search(rf"\b{re.escape(language)}\b", lower):
            return language
    suffix = "." + explicit_path.rsplit(".", 1)[-1].lower() if "." in explicit_path else ""
    for language, ext in _LANGUAGE_EXTENSIONS.items():
        if suffix == ext:
            return language
    return "text"

def _infer_subject(text: str) -> str:
    match = re.search(
        r"\b(?:for|of)\s+(?:a\s+|an\s+|the\s+)?(.{3,120}?)(?:\s+and\s+|\s+to\s+|\s+in\s+|\s+inside\s+|\s+under\s+|\s+save\b|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    subject = match.group(1) if match else text
    subject = re.sub(r"\b(?:python|javascript|typescript|html|css|markdown|bash|shell)\b", " ", subject, flags=re.I)
    subject = re.sub(r"\b(?:code|script|program|app|application|file|document)\b", " ", subject, flags=re.I)
    return " ".join(subject.split()).strip(" .,:;")[:120] or "artifact"

def _inferred_path(text: str, subject: str, language: str) -> str:
    directory = ""
    dir_match = _SAVE_DIR_RE.search(text)
    if dir_match:
        directory = _clean_directory(dir_match.group(1))
    ext = _LANGUAGE_EXTENSIONS.get(language, ".txt")
    filename = _slug_filename(subject, ext)
    return f"{directory.rstrip('/')}/{filename}" if directory else filename

def _clean_directory(value: str) -> str:
    cleaned = _clean_path(value)
    cleaned = re.sub(r"\b(?:you\s+create|that\s+you\s+create|you\s+make|that\s+you\s+make)\b", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" ./") or "."
    return cleaned

def _clean_path(value: str) -> str:
    cleaned = " ".join(str(value or "").strip(" `\"'").split())
    return cleaned.replace(" ", "-").strip("-")

def _slug_filename(subject: str, extension: str) -> str:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", subject or "")
        if word.lower() not in _GENERIC_FILENAME_WORDS
    ]
    stem = "-".join(words[:5]).strip("-") or "artifact"
    return f"{stem}{extension}"


# ---------------------------------------------------------------------------
# Formerly pipeline/heuristics.py: context selection
# ---------------------------------------------------------------------------


_IDENTITY_QUESTION_RE = re.compile(
    r"\bwhat (?:did|do) you learn(?: about yourself)?[.!?]?\s*$|"
    r"\bwhat (?:changed|did you change)(?: about yourself)?[.!?]?\s*$|"
    r"\b(?:did you evolve|how did you (?:change|evolve)|how have you (?:changed|evolved))[.!?]?\s*$|"
    r"^\s*evolve[.!?]?\s*$|"
    r"\bwhat do you (?:remember|believe) about yourself[.!?]?\s*$|"
    r"\bwhat (?:do you believe|are your (?:beliefs|priorities|drives))[.!?]?\s*$",
    re.IGNORECASE,
)

_USER_MEMORY_QUESTION_RE = re.compile(
    r"\bdo you remember\b(?!\s+about yourself\b)|"
    r"\bwhat (?:do|did) you (?:know|learn|believe) (?:about|of|from) (?:me|us|my|our)\b",
    re.IGNORECASE,
)

_BROAD_REPAIR_RE = re.compile(
    r"\b(?:everything|all of (?:it|this)|all the tension|all of the tension|"
    r"forgive(?: me| this| everything)?|move on|make peace|find peace|"
    r"repair (?:this|everything|all)|start over|clean slate)\b",
    re.IGNORECASE,
)

def _is_identity_question(text: str) -> bool:
    # Questions about the user or past conversations need the normal generation
    # path, including when the same message also asks about the agent itself.
    text = text or ""
    return bool(_IDENTITY_QUESTION_RE.search(text)) and not _USER_MEMORY_QUESTION_RE.search(text)

def _is_broad_repair_text(text: str) -> bool:
    return bool(_BROAD_REPAIR_RE.search(text or ""))

def _relationship_topic_hint(text: str) -> str:
    """Small topic hint so relationship-memory loops can be prioritized."""
    lower = text.lower()
    match = re.search(r"\b(?:about|regarding|around|on)\s+([a-z0-9' -]{2,40})", lower)
    if match:
        return _clean_topic_hint(match.group(1))
    words = re.findall(r"[a-z0-9']+", lower)
    stopwords = {
        "the", "and", "that", "this", "with", "your", "you", "for", "from",
        "have", "been", "just", "really", "very", "again", "still", "please",
        "drop", "topic", "issue", "thing", "can", "could", "would", "what",
        "when", "where", "why", "how", "any", "follow", "up",
        "unresolved", "resolved", "matters", "matter", "pattern",
    }
    informative = [word for word in words if len(word) > 2 and word not in stopwords]
    return " ".join(informative[:3])

def _clean_topic_hint(text: str) -> str:
    phrase = re.sub(r"[^a-z0-9' -]", " ", text.lower())
    phrase = " ".join(phrase.split())
    return phrase[:40].strip()
