# Security Policy

## Supported Versions

OpenRook is currently a research prototype. Only the `main` branch is
supported. There are no backported security fixes.

## Reporting a Vulnerability

Please **do not** file public GitHub issues for security or privacy
vulnerabilities. Instead, email the maintainer directly:

- **Contact:** balfiky@yahoo.com
- **Subject prefix:** `[openrook-security]`

Please include:

- A description of the issue and the impact
- Steps to reproduce, ideally with a minimal script or config
- Your suggested fix if you have one

You should receive an acknowledgement within a few business days. We'll work
with you on a disclosure timeline. Public disclosure should wait until a fix
is available or we've agreed on a timeline.

## Known Sensitive Surfaces

These areas process or persist potentially sensitive user content and warrant
extra scrutiny:

- `runtime/sessions/persistence.py` — writes per-session engine state to
  `data/<platform>_<user_id>/sessions/<chat_id>.json` and active hot
  transcripts to `data/<platform>_<user_id>/sessions/<chat_id>.history.json`.
  `data/` is gitignored; never commit it.
- `core/memory/long_term.py`, `core/memory/relationship.py`,
  `core/memory/semantic.py` — SQLite tables in `data/<platform>_<user_id>/nur.db`
  storing distilled user memory (`memories`), relationship events
  (`relationship_events`), open loops (`open_loops`), and semantic preferences
  (`semantic_memories`). The shared self-model lives in
  `data/shared/self_model.db`. See [PRIVACY.md](PRIVACY.md) for full layout.
- `runtime/life_history.py` — shared identity-level experience and evolution
  storage in `data/shared/life_history.db`. Pasted text and local-file
  excerpts can be persisted as evidence for belief/drive changes.
- `runtime_config.yaml` — may contain API keys. The secret fields
  (enumerated in `runtime/config.py:_SECRET_FIELDS`) are `telegram_token`,
  `llm_api_key`, and `api_key`. Keep real values out of
  committed files; use `runtime_config.example.yaml` as the tracked starter
  and inject real values locally or via environment variables.
- `interface/api.py` and `interface/v1.py` — CORS and bearer-auth handling.
  The default `cors_origins: []` is same-origin only; avoid widening it
  without careful thought.
- `llm_backend: codex` — delegates response generation to the local Codex CLI.
  OpenRook invokes it with `codex exec --sandbox read-only --ephemeral`; keep
  `OPENROOK_CODEX_WORKDIR` unset unless the backend should have read-only context
  from a specific local directory.
- `runtime/channels/telegram.py` — allowlist logic controls who can chat.
  Misconfiguration exposes the system to arbitrary Telegram users.
- `/admin` — operator console for configuration, diagnostics, redacted export,
  backup, guarded session reset, guarded user deletion, skill management,
  and Life History intake/observability. Set `api_key` before exposing the
  server beyond localhost.

## Agentic Tool Runtime

OpenRook ships with a builtin agentic tool layer (`nur_tools/builtin/`) that can
match user text to actions like `fs.read_file`, `fs.write_file`,
`fs.delete_path`, `shell.run_command`, and web/browser/calendar calls via
regex heuristics in `core/dual_process/tool_loop.py`. **External tools are
off by default** (`tools_enabled: false`). With `tools_enabled: false`,
`runtime/tools.py` still wires OpenRook's internal **skill-registry tools**
(`skills.create`, `skills.audit`, `skills.enable`, etc.) into the executor
so the assistant can manage durable skills through the observable tool
trace; filesystem, browser, web, calendar, and shell capabilities remain
excluded until `tools_enabled: true`, and `shell.run_command` additionally
requires `shell_tool_enabled: true`.

Additionally, the native tool-controller system prompt
(`nur_tools/native_orchestrator.py`) only emits the "diagnose, install,
retry" directive — which lets the model unilaterally run package-manager
commands on missing-dependency errors — when
`autonomy_level: high_risk`. Under `assisted` (default) and `autonomous`
the prompt explicitly instructs the model to report dependency failures
honestly rather than silently install packages.

When you turn external tools on:

- Filesystem tools are sandboxed to
  `tools_workspace` — paths that resolve outside that directory are
  refused. The default workspace is `<data_dir>/workspace`.
- Life History local-file intake also reads only from `tools_workspace`;
  operators should treat that workspace as readable by the assistant runtime.
- `shell.run_command` is a **separate** opt-in via `shell_tool_enabled: true`
  because subprocess execution has a larger blast radius than bounded
  file I/O. The shell tool's layered model is: two opt-in flags
  (`tools_enabled` + `shell_tool_enabled`) → the autonomy confirmation gate
  (assisted autonomy confirms every destructive action; only `high_risk`
  skips confirmation entirely) → a hard 600s
  timeout cap on every command → an optional `shell_command_allowlist` plus
  a workspace-default working directory. When `cwd` is omitted, commands run
  in `tools_workspace` instead of the server process directory; an explicit
  `cwd` is still honored (and a missing one is a clean tool error, not an
  exception).
- `shell_command_allowlist` (default empty) narrows the blast radius: when
  non-empty, only commands whose first token exactly matches an entry may
  run, and chaining/redirection metacharacters (`;`, `|`, `&`, backticks,
  `$(`, `${`, `<`, `>`, newline) are rejected so a second command — or
  env-var content smuggled via `${...}` expansion — cannot slip past the
  executable check. Allowlist mode deliberately trades
  shell features (pipes, redirects, substitution, expansion) for containment. Be
  plain about the alternative: with an **empty** allowlist and
  `autonomy_level: high_risk`, the shell tool is full remote code
  execution as the server user — by design.
- MCP tools are categorized (read-only / write / destructive /
  external-action) by keyword inference over the **server-supplied** tool
  name and description (`nur_tools/mcp/adapter.py`). That inference is a
  trust decision on strings the MCP server controls: a misleadingly named
  `get_*` tool with side effects would be treated as read-only and skip
  the confirmation gate. Two operator controls counter this:
  `mcp_tool_categories` pins categories per tool (exact name or fnmatch
  pattern → `read_only`/`write`/`destructive`/`external_action`; pins are
  applied before inference, and invalid values fail at config load), and
  `mcp_strict_categories: true` disables inference entirely so every
  unpinned MCP tool is `external_action` and confirmation-gated under
  assisted autonomy. Use strict mode for any MCP server you do not fully
  trust.
- Outbound web access (`web.fetch`, `web.extract_text`, browser tools,
  and LLM backend base URLs) is guarded against SSRF by
  `runtime/security.py::validate_http_url`: literal private/loopback/
  link-local/reserved IPs are rejected, non-literal hostnames are DNS-
  resolved (A + AAAA) and every resolved address must pass the same
  policy, unresolvable hosts are rejected, and HTTP redirects are
  followed manually with each hop re-validated (capped at 5 hops).
  Local LLM backends (`localhost`/loopback, e.g. ollama) are allowed via
  `allow_loopback`; trusted deployments can opt into private targets with
  `OPENROOK_ALLOW_PRIVATE_WEB_FETCH` / `OPENROOK_ALLOW_PRIVATE_LLM_URLS`. Known
  residual risk: validation resolves DNS at check time while the request
  resolves again at connect time, leaving a TOCTOU/DNS-rebinding window;
  back the guard with egress controls where that matters.
- The HTTP surface (`/chat`, `/v1/chat`) routes user messages through the
  same regex intent parser, so authenticating the endpoint is not a
  substitute for sandboxing — an authenticated attacker (or the user's
  own mis-addressed command) can still reach the tool layer. Set
  `api_key` and keep `shell_tool_enabled: false` unless you trust every
  caller.

Bearer auth on `openrook-web` is **opt-in**, matching the convention of
other local-LLM tools (Ollama, LM Studio, Jupyter). With no `api_key`
set in `runtime_config.yaml`, the HTTP surface is open. On a non-loopback
bind (`--host 0.0.0.0` or any non-`127.0.0.1`/`localhost`/`::1` value)
the launcher prints a one-screen warning to stderr so the operator
knows the surface is reachable from the network.

Set `api_key: <a-strong-token>` in `runtime_config.yaml` and restart to
require bearer auth on `/chat`, `/debug`, `/config`, `/session/end`,
`/rest`, `/ws`, every `/admin/*` JSON endpoint, and every `/v1/*`
endpoint except `/v1/health` and `/v1/ready`. This is recommended any
time more than one person can reach the host (multi-user tailnet,
shared LAN, public IP).

On first run the launcher also auto-creates `runtime_config.yaml` with
safe defaults (no LLM configured yet, tools off, shell off, no Telegram) if
the file does not already exist, so a clean `pip install` → `openrook-web`
flow works without a separate `openrook-setup` step. The browser setup
wizard at `/` walks the operator through model, identity, and tool
settings.

When `api_key` is set, every mutating and data-bearing endpoint on the
standalone web server requires `Authorization: Bearer <api_key>`:
`/chat`, `/debug`, `/config` (GET and POST), `/session/end`, `/rest`,
`/ws`, every `/admin/*` JSON endpoint (status, config, test, diagnostics,
export, backup, sessions/reset, users/delete), and every `/v1/*`
endpoint except `/v1/health` and `/v1/ready`. Static assets (`GET /`,
`GET /admin` HTML shell) remain open so the bundled UI can bootstrap and
prompt for the bearer token client-side.

## Responsible Use

OpenRook persists emotional and relational state about its users. Do not deploy it
against people who have not consented to that persistence. See
[PRIVACY.md](PRIVACY.md) for retention and deletion guidance and
[docs/DEPLOYMENT_AND_ADMIN.md](docs/DEPLOYMENT_AND_ADMIN.md) for deployment
hardening.
