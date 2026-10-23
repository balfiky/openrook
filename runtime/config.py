"""Runtime configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields as dc_fields


_SECRET_FIELDS = {
    "telegram_token",
    "llm_api_key",
    "api_key",
}
_LEGACY_FIELD_ALIASES = {
    "proactive_max_per_session": "proactive_density_reference",
    "proactive_cooldown": "proactive_recovery_seconds",
}

# Valid values for mcp_tool_categories entries. "cognitive" is reserved for
# OpenRook-internal tools and may not be assigned to MCP tools by operators.
VALID_MCP_TOOL_CATEGORIES = ("read_only", "write", "destructive", "external_action")


def normalize_mcp_tool_categories(value: object) -> dict[str, str]:
    """Validate and normalize an mcp_tool_categories mapping.

    Keys are MCP tool names (exact, or fnmatch glob patterns such as
    ``mcp.github.*``); values must be one of VALID_MCP_TOOL_CATEGORIES.
    Raises ValueError with a clear message on any invalid entry so a bad
    config fails at load time instead of silently miscategorizing tools.
    """
    if not isinstance(value, dict):
        raise ValueError(
            "mcp_tool_categories must be a mapping of tool name (or fnmatch "
            "pattern) to category string"
        )
    normalized: dict[str, str] = {}
    for name, category in value.items():
        key = str(name).strip()
        cat = str(category).strip().lower()
        if not key:
            raise ValueError("mcp_tool_categories: tool name keys must be non-empty")
        if cat not in VALID_MCP_TOOL_CATEGORIES:
            raise ValueError(
                f"mcp_tool_categories[{key!r}]: invalid category {category!r}; "
                f"must be one of: {', '.join(VALID_MCP_TOOL_CATEGORIES)}"
            )
        normalized[key] = cat
    return normalized


@dataclass
class RuntimeConfig:
    """Configuration for the OpenRook Runtime."""

    # Storage
    data_dir: str = "data"

    # Backpressure
    max_queue_per_user: int = 3
    max_active_sessions: int = 10

    # Session lifecycle
    session_timeout_seconds: float = 1800.0  # 30 minutes

    # Channels
    console_enabled: bool = True

    # Telegram
    telegram_token: str = ""
    telegram_allowlist: set[str] = field(default_factory=set)
    telegram_poll_timeout: int = 30
    dedupe_ttl: float = 60.0

    # LLM backend — OpenRook always calls a real model. There is no mock path.
    # Valid: "auto", "provider", "openai_compatible", "codex", "minimax".
    llm_backend: str = "auto"
    llm_base_url: str = ""         # API endpoint for OpenAI-compatible backends
    llm_model: str = ""            # model name for OpenAI-compatible backends
    llm_api_key: str = ""          # generic API key (optional for local endpoints)
    llm_request_timeout_seconds: float = 600.0  # allow slow local CPU generation
    # None adapts to the primary deadline and successful generation latency.
    # An explicit auxiliary deadline is still capped by the primary deadline.
    llm_auxiliary_timeout_seconds: float | None = None
    llm_auxiliary_cooldown_seconds: float = 30.0

    # Debug API
    debug_host: str = "127.0.0.1"
    debug_port: int = 8077

    # Proactive behavior (Phase 8)
    proactive_enabled: bool = False
    proactive_idle_threshold: float = 300.0    # seconds idle before proactive check
    proactive_density_reference: int = 3       # action density reference for recovery pressure
    proactive_recovery_seconds: float = 300.0  # recovery curve after proactive actions
    proactive_check_interval: float = 60.0     # how often the proactive loop runs

    # Character independence runtime
    character_independence: bool = False
    research_mode: bool = False
    research_birth_policy: str = "uncommitted"
    research_action_framing: str = "agent_owned"
    research_action_feedback: str = "receipts"
    research_execution_mode: str = "batch"
    research_context_mode: str = "full"
    research_memory_mode: str = "legacy"
    research_progress_estimates: bool = False
    research_appraisal_mode: str = "legacy"
    research_dispositions: bool = False
    research_json_schema: bool = False
    research_inference_retries: int = 0
    research_max_steps_per_activation: int = 3
    research_initial_session: str = ""
    research_calls_per_day: int = 1000
    research_episode_idle_seconds: float = 300.0
    research_episode_max_turns: int = 50
    research_reflection_interval: float = 3600.0
    # Social appraisal: when True the backend LLM reads the interpersonal
    # meaning of each user message (no regex); the deterministic appraisal is
    # the fallback when no LLM is available or its output won't parse.
    appraisal_use_llm: bool = True
    # Ordinary conversations can develop the self at consolidation. Grounded
    # evidence accumulation and bounded rates protect continuity. Explicit False
    # is the no-learning control; research episode policy is unchanged.
    autobiographical_learning: bool = True
    coherence_min_score: float = 0.6
    coherence_max_regenerations: int = 2
    pending_intake_ttl_turns: int = 3

    # Public integration API (standalone web server)
    api_key: str = ""                          # bearer token; empty = auth disabled
    cors_origins: list[str] = field(default_factory=list)  # CORS allowlist; empty = same-origin only

    # Agentic tool runtime
    # Tools are off by default: user-text heuristics can invoke destructive
    # operations (fs.delete_path, shell.run_command, etc.), and the default
    # HTTP surface does not authenticate every request. Opt in explicitly.
    tools_enabled: bool = False
    # How independently the assistant may act once tools are enabled:
    # off | assisted | autonomous | high_risk
    autonomy_level: str = "assisted"
    # Workspace root for filesystem tools when enabled. Empty string means
    # "<data_dir>/workspace". Filesystem tool calls that resolve outside this
    # root are refused.
    tools_workspace: str = ""
    # Shell tool is a separate opt-in even when tools_enabled=True: subprocess
    # execution is a larger blast radius than bounded file I/O.
    shell_tool_enabled: bool = False
    # Optional shell command allowlist. Empty (default) = unrestricted shell
    # once shell_tool_enabled=True. Non-empty = only commands whose first
    # token exactly matches an entry may run, and shell chaining/redirection
    # metacharacters (;, |, &, backticks, $(, <, >, newline) are rejected so
    # a second command cannot be smuggled past the executable check.
    shell_command_allowlist: list[str] = field(default_factory=list)
    # Operator trust controls for MCP tool categorization. MCP servers supply
    # their own tool names/descriptions, and OpenRook's keyword inference trusts
    # those strings — a misleadingly named tool can land in READ_ONLY and skip
    # the confirmation gate. ``mcp_tool_categories`` maps a tool name (exact
    # match, or fnmatch glob pattern like "mcp.github.*") to a category
    # ("read_only" | "write" | "destructive" | "external_action") and is
    # applied BEFORE keyword inference. ``mcp_strict_categories: true``
    # disables keyword inference entirely: any MCP tool without an explicit
    # override is categorized external_action (confirmation-gated under
    # assisted autonomy).
    mcp_tool_categories: dict[str, str] = field(default_factory=dict)
    mcp_strict_categories: bool = False

    # Learning schedule (Sprint 5). The LearningBudget caps how often OpenRook can
    # surface an open question to the user inside chat. Metabolism min elapsed
    # days is the rate limit on the wall-clock reflection tick.
    learning_budget_kind: str = "local"        # "local" | "cloud" (cloud not yet implemented)
    learning_max_questions_per_day: int = 3
    learning_max_seconds_per_day: float = 1800.0
    metabolism_min_elapsed_days: float = 1.0

    # Self-action layer (v0.30). The pipeline runs a self-intent proposer
    # between the user-driven tool loop and the defense stage so OpenRook can
    # invoke any registered tool for its own reasons. There is no catalog
    # restriction and no per-turn cap. ``owner_chat_id`` is the Telegram
    # chat used by ``self.alert_owner`` when OpenRook chooses to reach out
    # out-of-band.
    owner_chat_id: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.llm_request_timeout_seconds, bool)
            or not isinstance(self.llm_request_timeout_seconds, (int, float))
            or not math.isfinite(self.llm_request_timeout_seconds)
            or self.llm_request_timeout_seconds < 1
        ):
            raise ValueError("llm_request_timeout_seconds must be a finite number of at least 1 second")
        for name, minimum in (("llm_auxiliary_timeout_seconds", 1), ("llm_auxiliary_cooldown_seconds", 0)):
            value = getattr(self, name)
            if value is None and name == "llm_auxiliary_timeout_seconds":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be a finite number of at least {minimum} seconds")
        if not isinstance(self.research_json_schema, bool):
            raise ValueError("Research JSON schema mode must be boolean")
        if type(self.research_inference_retries) is not int or not 0 <= self.research_inference_retries <= 2:
            raise ValueError("Research inference retries must be an integer from zero to two")
        if self.research_birth_policy not in {"uncommitted", "configured"}:
            raise ValueError("Research birth policy must be uncommitted or configured")
        if self.research_action_framing not in {"agent_owned", "conversational"}:
            raise ValueError("Research action framing must be agent_owned or conversational")
        if self.research_action_feedback not in {"receipts", "legacy"}:
            raise ValueError("Research action feedback must be receipts or legacy")
        if self.research_execution_mode not in {"batch", "closed_loop"}:
            raise ValueError("Research execution mode must be batch or closed_loop")
        if self.research_context_mode not in {"full", "task_only"}:
            raise ValueError("Research context mode must be full or task_only")
        if self.research_memory_mode not in {"legacy", "none", "episodes", "procedures", "raw"}:
            raise ValueError("Invalid research memory mode")
        if self.research_appraisal_mode not in {"legacy", "expression", "attention", "neutral"}:
            raise ValueError("Invalid research appraisal mode")
        if not isinstance(self.research_progress_estimates, bool) or not isinstance(self.research_dispositions, bool):
            raise ValueError("Research learning switches must be boolean")
        if self.research_memory_mode == "raw" and (
            self.research_context_mode != "task_only"
            or self.research_progress_estimates
            or self.research_appraisal_mode != "legacy"
            or self.research_dispositions
        ):
            raise ValueError(
                "Raw experience requires task-only context without progress estimates, grounded appraisal, or dispositions"
            )
        if (
            isinstance(self.research_max_steps_per_activation, bool)
            or not isinstance(self.research_max_steps_per_activation, int)
            or not 1 <= self.research_max_steps_per_activation <= 16
        ):
            raise ValueError("Research execution requires 1 to 16 steps per activation")
        if self.research_execution_mode == "closed_loop" and self.research_action_feedback != "receipts":
            raise ValueError("Closed-loop research execution requires receipt feedback")
        if self.research_mode:
            self.character_independence = True
            self.proactive_enabled = True
            if self.autonomy_level == "assisted":
                self.autonomy_level = "autonomous"
        if self.research_calls_per_day < 1 or self.research_episode_max_turns < 1:
            raise ValueError("Research resource and episode limits must be positive")
        if self.proactive_check_interval <= 0 or self.research_reflection_interval <= 0:
            raise ValueError("Runtime intervals must be positive")

    def learning_budget(self):
        """Build a LearningBudget instance from the configured settings."""
        from runtime.learning_budget import from_config as _budget_from_config
        return _budget_from_config({
            "budget": self.learning_budget_kind,
            "local": {
                "max_questions_per_day": self.learning_max_questions_per_day,
                "max_seconds_per_day": self.learning_max_seconds_per_day,
            },
        })

    @classmethod
    def from_yaml(cls, path: str) -> RuntimeConfig:
        """Load config from a YAML file.  Missing keys use defaults."""
        import yaml

        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Convert list → set for allowlist
        al = data.pop("telegram_allowlist", None)
        if al is not None:
            data["telegram_allowlist"] = {str(x) for x in al}
        # Normalize cors_origins to list[str]
        co = data.get("cors_origins")
        if co is not None:
            data["cors_origins"] = [str(x) for x in co]
        # Normalize shell_command_allowlist to list[str]
        sca = data.get("shell_command_allowlist")
        if sca is not None:
            data["shell_command_allowlist"] = [str(x) for x in sca]
        # Validate + normalize mcp_tool_categories (fails loud on bad values)
        mtc = data.get("mcp_tool_categories")
        if mtc is not None:
            data["mcp_tool_categories"] = normalize_mcp_tool_categories(mtc)
        for legacy_key, current_key in _LEGACY_FIELD_ALIASES.items():
            if current_key not in data and legacy_key in data:
                data[current_key] = data[legacy_key]
        # Drop unknown keys so __init__ doesn't blow up
        known = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_yaml_dict(self) -> dict[str, object]:
        """Serialize to a YAML-friendly dict in dataclass field order."""
        data: dict[str, object] = {}
        for field_def in dc_fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, set):
                value = sorted(value)
            data[field_def.name] = value
        return data

    def write_yaml(self, path: str) -> None:
        """Persist this config to YAML."""
        import yaml

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # The file can hold secrets (api_key, llm_api_key, telegram_token);
        # create it owner-read/write only instead of the default umask, and
        # tighten pre-existing files on every save.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(
                self.to_yaml_dict(),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def to_public_dict(self) -> dict[str, object]:
        """Serialize for UI/API use without exposing secret values."""
        data = self.to_yaml_dict()
        for field_name in _SECRET_FIELDS:
            data[field_name] = ""
        return data

    def secret_status(self) -> dict[str, bool]:
        """Return whether each secret-like field is configured."""
        return {
            field_name: bool(getattr(self, field_name))
            for field_name in sorted(_SECRET_FIELDS)
        }

    @property
    def effective_llm_api_key(self) -> str:
        return self.llm_api_key or os.environ.get("LLM_API_KEY", "")

    @property
    def shared_db_path(self) -> str:
        return os.path.join(self.data_dir, "shared", "self_model.db")

    @property
    def resolved_tools_workspace(self) -> str:
        """Absolute path to the filesystem workspace for sandboxed tools."""
        root = self.tools_workspace.strip() or os.path.join(self.data_dir, "workspace")
        return os.path.realpath(root)

    def user_data_dir(self, rel_key: str) -> str:
        """Per-user data directory.  rel_key is platform:user_id.

        Sanitizes path-dangerous characters (``/``, ``\\``, ``..``) in
        addition to the key separator so a hostile ``user_id`` cannot escape
        ``data_dir``. This is the filesystem-boundary backstop behind the
        per-endpoint token validation, covering every transport (HTTP,
        Telegram, restore) regardless of which caller built the key.
        """
        safe = _safe_path_token(rel_key)
        return os.path.join(self.data_dir, safe)

    def user_db_path(self, rel_key: str) -> str:
        from runtime.storage import resolve_user_db_path

        return resolve_user_db_path(self.user_data_dir(rel_key))

    def user_state_path(self, rel_key: str) -> str:
        """Legacy per-user engine-state path kept for backward-compatible restore."""
        return os.path.join(self.user_data_dir(rel_key), "engine_state.json")

    def session_state_path(self, session_key: str) -> str:
        """Per-session engine-state path.

        Session state is keyed by ``platform:user_id:chat_id`` so different chat
        contexts for the same user restore independently while still sharing the
        same per-user relational storage.
        """
        platform, user_id, chat_id = session_key.split(":", 2)
        rel_key = f"{platform}:{user_id}"
        safe_chat = _safe_path_token(chat_id)
        return os.path.join(self.user_data_dir(rel_key), "sessions", f"{safe_chat}.json")

    def session_history_path(self, session_key: str) -> str:
        """Per-session hot transcript path restored until the session is ended."""
        state_path = self.session_state_path(session_key)
        root, _ext = os.path.splitext(state_path)
        return f"{root}.history.json"


def _safe_path_token(value: str) -> str:
    """Filesystem-safe token for path segments derived from runtime keys."""
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )
