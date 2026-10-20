# Deployment And Admin Guide

This guide is for running OpenRook as an open-source product instance, not just as a
local research demo.

Use it together with:

- [SECURITY.md](../SECURITY.md) for threat model and sensitive surfaces
- [PRIVACY.md](../PRIVACY.md) for retention, inspection, and deletion
- [OPENROOK_MIGRATION.md](OPENROOK_MIGRATION.md) before upgrading a Nūr installation
- [DEVELOPMENT_SUMMARY.md](DEVELOPMENT_SUMMARY.md) for major runtime changes and recorded validation

## Recommended Local Install

```bash
git clone https://github.com/balfiky/openrook.git
cd openrook
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
openrook-web
```

For a normal local setup, run:

```bash
openrook-setup
```

That command creates `runtime_config.yaml`, creates data/workspace folders,
prompts for model backend, identity, Telegram, and tool settings, and exits. It
does not start a browser.

Start the browser UI when you want it:

```bash
openrook-web --config runtime_config.yaml
```

If you prefer the browser wizard instead of terminal prompts, run it explicitly:

```bash
openrook-setup --web --config runtime_config.yaml
```

Open:

- Chat UI: `http://localhost:8000`
- Settings workspace: `http://localhost:8000/settings`
- API docs: `http://localhost:8000/docs`

A first-run setup wizard opens automatically on first launch. It walks through
LLM backend selection, runtime channels, seed identity, and optional first
Life History material in six steps (Welcome → Connect LLM → Runtime & Channels
→ Identity → First Experience → Done).
Life History material can be pasted or uploaded from the browser as text or
Markdown, so a normal user does not need to place files into the data directory.
You can skip it and configure the same runtime fields later from `/settings`.

## Git Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "git+https://github.com/balfiky/openrook.git"
export OPENROOK_CONFIG_DIR=~/.config/openrook   # see below — do this before the first run
mkdir -p "$OPENROOK_CONFIG_DIR"
openrook-setup
```

Use the source checkout, its built wheel, or the Git URL above. These instructions
do not assume a PyPI release. The repository is `balfiky/openrook` (formerly
`balfiky/nur`, which redirects); the distribution and primary commands use `openrook`.

For local-only use, the default `openrook-web` command binds to `127.0.0.1`. For
LAN/public testing, point it at a public host:

```bash
openrook-web --host 0.0.0.0 --port 8000 --config runtime_config.yaml
```

On first run, `openrook-web` auto-creates `runtime_config.yaml` with safe defaults
(no LLM configured yet, tools off, shell off, no Telegram) if the file does not
already exist, then opens the browser setup wizard at `/`.

Authentication is **opt-in**. With no `api_key` set in
`runtime_config.yaml`, the HTTP surface is open — same convention as
Ollama, LM Studio, and Jupyter defaults. `openrook-web` prints a one-screen
warning to stderr when binding to a non-loopback host without a token,
so the operator knows the surface is reachable from the network.

To require bearer auth (recommended when more than one person can reach
the host), set `api_key: <a-strong-token>` in `runtime_config.yaml` and
restart. Clients then authenticate with `Authorization: Bearer <api_key>`
against `/chat`, `/admin/*`, `/v1/*`, `/ws`, and other protected
endpoints. The browser settings workspace has an **API Token** button
that stores the token in the browser session.

`openrook-web` now also accepts `--config /path/to/runtime_config.yaml` when you need
to run from a systemd working directory that differs from the config location.

## Uninstalling A Local Workspace

For a normal local install, preview removal first:

```bash
openrook-uninstall --dry-run
```

Then remove the OpenRook-created runtime config and data directory:

```bash
openrook-uninstall
```

The command requires typing `uninstall` before it deletes anything. It removes
`runtime_config.yaml` and the configured `data_dir`. It does **not** delete a
custom `tools_workspace` outside `data_dir` unless you pass
`--remove-external-workspace`, and it does **not** delete
`$OPENROOK_CONFIG_DIR/soul.yaml` unless you pass `--remove-identity`.

After the workspace is removed, uninstall the Python package if desired:

```bash
python3 -m pip uninstall openrook
```

## OPENROOK_CONFIG_DIR — keeping your identity across upgrades

When installed from a packaged distribution, `config/soul.yaml` lives inside the
package directory and may be overwritten on package upgrade. Set
`OPENROOK_CONFIG_DIR` to a directory you control before the first run:

```bash
export OPENROOK_CONFIG_DIR=~/.config/openrook
mkdir -p "$OPENROOK_CONFIG_DIR"
```

With the env var set:
- The settings workspace writes `soul.yaml` to `$OPENROOK_CONFIG_DIR/soul.yaml` instead
  of the package-internal path.
- On `pip install --upgrade`, your customized identity is untouched.
- The settings workspace shows a note when the env var is **not** set so operators
  don't silently lose their identity on the next upgrade.

The env var governs identity/configuration and prompt overrides, not the runtime
configuration file. `runtime_config.yaml` is normally written
to the current working directory, or to the explicit path passed through
`openrook-setup --config` / `openrook-web --config`.

For systemd or Docker deployments, pass it as an environment variable:

```bash
# systemd
Environment=OPENROOK_CONFIG_DIR=/etc/openrook

# Docker
docker run -e OPENROOK_CONFIG_DIR=/config -v /host/config:/config ...
```

## Settings Workspace

The `/settings` route serves the built-in operator workspace. It uses the same
runtime config primitives as `runtime_config.yaml`; it is not a separate config
system. Legacy browser routes `/admin`, `/persona`, and `/dashboard` redirect
into this one settings page so there is one place to manage and inspect OpenRook.

The chat UI exposes one Settings button. Clicking it opens the full settings
workspace. The left panel switches between Overview, Settings, Skills,
Observability, and Life History pages. Settings includes runtime, character
independence, model, tool access, identity, access, channel, and maintenance
controls together; monitoring is grouped under Observability and Life History.

The Observability section reads active sessions from the shared runtime manager,
so Web, Telegram, console, and future channel sessions appear in one place.
Refreshing observability only inspects state: it does not create sessions, call
the pipeline, or mutate emotional state, memory, Life History, or skills.

The first-run **Setup Wizard** walks through LLM backend, agent identity,
optional first formative material, runtime/channel settings, and completion in
six steps. It is the recommended path for first-time configuration; `/settings`
contains the same ongoing configuration controls after setup.

In ordinary mode, the settings workspace lets you:

- Configure LLM provider, model, base URL, and API keys
- Configure Telegram token, allowlist, polling, and dedupe settings
- Configure character independence (currently informational only — see
  Known Gaps below), coherence checks, pending-intake behavior, bearer auth,
  CORS, tools, shell-tool opt-in, and workspace paths
- Set the operator-authored **Constitution** — a short stable orientation
  string rendered above evolving beliefs in every prompt
- Inspect and manage **Open Questions** — epistemic gaps surfaced by
  reflection (contradiction, low-confidence, drive-gap) with Abandon and
  Resolve actions
- Upload, import, audit, and enable external Agent Skills
- Feed pasted or uploaded formative text/Markdown files into **Life History**
- Observe experience count, evolution events, current beliefs, and drive shifts
- Inspect unified emotional, perception, relationship, Life, memory, skills,
  and tool state across active channels from **Observability**
- Test LLM, Telegram, and storage settings before relying on them
- Inspect diagnostics: Python version, uptime, active sessions, storage paths
- Apply saved config to the live web session manager and Telegram poller
- Restart the web server process with typed confirmation when process-bound
  settings change
- Export a redacted runtime config
- Create a local zip backup under `<data_dir>/backups`
- Reset one active session with typed confirmation
- Delete one user's persisted data with typed confirmation

In research mode, observation remains read-only. Mutating `/admin/*` and config
requests, rest controls, and all DELETE requests return `409`; this includes
backup creation and administrative resets. See [research backup](#research-backup-with-the-owner-stopped)
and [reset and delete](#reset-and-delete) for stopped-owner procedures.

Secret values are never returned by admin API responses. Blank secret fields
mean "keep the current value"; explicit clear checkboxes remove stored YAML
secrets.

## Skills Admin

The **Skills** page imports uploaded `SKILL.md`/Markdown files, zipped skill
folders, server-side skill folders, or pasted `SKILL.md` content. Browser upload
is the normal path; server paths are an advanced escape hatch. Imported skills
are disabled until review. The audit reports metadata, tool hints, risk flags,
bundled scripts, resource counts, and unsupported platform-specific hints.

Enabled skills are injected into generation as bounded private guidance. They
do not execute bundled scripts or bypass tool policy. If a skill needs web,
filesystem, shell, or browser actions, those actions still require the normal
runtime tool settings, workspace restrictions, auth posture, and shell opt-in.

## First Production Hardening Checklist

For localhost-only personal use, `api_key` may stay empty. Before binding
OpenRook beyond your own machine, consider whether bearer auth is needed:

1. If more than one person can reach the host (multi-user tailnet,
   shared LAN, public IP), set `api_key: <a-strong-token>` in
   `runtime_config.yaml` and restart. `openrook-web` prints a warning on
   non-loopback binds without `api_key` so you don't miss this.
   For single-user tailnets or trusted home networks, leaving
   `api_key` empty matches the Ollama/LM Studio convention and is
   acceptable.
2. Keep `cors_origins: []` unless a separate browser origin needs access.
3. Keep `tools_enabled: false` until you explicitly need agentic tools.
4. Keep `shell_tool_enabled: false` unless every authenticated user is trusted.
5. Set `telegram_allowlist` before enabling a Telegram bot.
6. Confirm `data_dir` points to a durable, backed-up location.
7. Run `/admin/test/llm`, `/admin/test/telegram`, and `/admin/test/storage`.
8. Review [PRIVACY.md](../PRIVACY.md) before letting other people interact with
   the instance.

The safe default posture is localhost-only, a configured local LLM backend
(Ollama or Codex CLI), tools off, shell off, and no public CORS origins.

## Runtime Config

`runtime_config.yaml` is the source of truth for deployment settings. The
settings workspace reads and writes this file.

Saving config through `/settings` writes `runtime_config.yaml`, evicts active
web sessions so new pipelines use the saved settings, and restarts the built-in
Telegram poller. If you edit `runtime_config.yaml` outside the browser, use
Maintenance -> **Apply Saved Config**. Process-bound settings such as
`debug_host`, `debug_port`, and `cors_origins` need Maintenance -> **Restart Web
Server** or a manual service restart.

Important fields:

| Field | Production guidance |
|-------|---------------------|
| `api_key` | Optional bearer token for hardened deployments. Empty means no browser/API token flow. When set, it protects administrative data/mutation APIs (`/admin/*`), `/chat`, `/config`, `/ws`, and `/v1/*` except health/ready. The `/settings` HTML shell is public; its data and actions use authenticated APIs. |
| `llm_backend` | Use `provider` for hosted gateways, `openai_compatible` for local/remote compatible servers (Ollama, LM Studio, vLLM), or `codex` for the local Codex CLI. There is no mock backend in production — OpenRook always calls a real model. |
| `llm_base_url` / `llm_model` | Required for hosted and OpenAI-compatible backends. |
| `llm_api_key` | Generic provider/gateway key. Prefer local-only YAML or environment injection. |
| `telegram_token` | Enables the Telegram channel when set. |
| `telegram_allowlist` | Numeric Telegram user IDs allowed to talk to the bot. |
| `cors_origins` | Browser origin allowlist. Empty means same-origin only. |
| `data_dir` | Persistent data root; default is `data`. |
| `tools_enabled` | Enables external tools; off by default. Internal skill-registry tools remain available, including registry mutations, so disabling this is not a text-only or no-write mode. |
| `tools_workspace` | Filesystem sandbox root for tools; blank means `<data_dir>/workspace`. |
| `shell_tool_enabled` | Separate opt-in for local shell execution; keep off by default. Supports normal shell syntax such as pipes once enabled. Commands run in `tools_workspace` unless the call supplies an explicit `cwd`. |
| `shell_command_allowlist` | Optional shell containment (YAML-only for now). Empty = unrestricted. Non-empty = only commands whose first word matches an entry run, and chaining/redirection metacharacters are rejected. Example: `["git", "ls", "rg"]`. |
| `mcp_tool_categories` | Per-tool risk-category pins for MCP tools (YAML-only for now). Maps a tool name — exact, or fnmatch pattern like `mcp.github.*`, matched against the namespaced and bare name — to `read_only` / `write` / `destructive` / `external_action`. Pins are applied before keyword inference; invalid values fail at config load. |
| `mcp_strict_categories` | When `true`, keyword inference for MCP tool categories is disabled: every MCP tool without an `mcp_tool_categories` pin is treated as `external_action` (confirmation-gated under assisted autonomy). Recommended for untrusted MCP servers. Default `false`. |

The Codex backend does not use `llm_api_key`. It calls the local `codex` CLI
with a read-only ephemeral session. The `/settings` model dropdown is populated
from the installed Codex CLI model catalog. Leave `OPENROOK_CODEX_WORKDIR` unset for
normal chat use; set it only when Codex should have read-only context from a
specific local directory.

## Telegram Commands

Telegram command handling is local to the runtime channel and does not trigger
the typing loop or normal chat generation.

| Command | Behavior |
|---|---|
| `/help` or `/commands` | Show the supported command list. |
| `/start` | Send a short greeting plus the command list. |
| `/status` | Show the active session's emotional modulators, if one exists. |
| `/mental` or `/mood` | Create/inspect the current session and report mental-state diagnostics. |
| `/new` | Start a fresh hot conversation while keeping relationship memory. Clears the active hot transcript. |
| `/reset` | Digest, save, and close the active session. Clears the active hot transcript. |
| `/debug` | Show a compact last-turn debug summary for the active session. |
| `/state` | Show compact current emotion, strategy, modulators, and open-loop count without processing a turn. |
| `/why` | Show the deterministic explanation for the last response. |
| `/memory` | Show current-turn relationship, long-term, and semantic memory counts. |
| `/loops` | List active relationship loops from the last debug state. |
| `/repair` | Show recent rupture, repair, recurring-tension, and commitment context. |
| `/persona [section]` | Unified persona observability view. Optional sections: `emotions`, `perception`, `life`, `memory`, `skills`, `tools`, `all`. |

Telegram keeps the active hot transcript under
`<data_dir>/telegram_<user_id>/sessions/<chat_id>.history.json` and restores it
after idle eviction or runtime restart. The transcript is separate from
relationship and semantic memory; `/new`, `/reset`, and explicit session-end
operations clear it.

## Life History Admin

The **Life** page is the first observability surface for OpenRook's identity-level
experience ledger. It supports:

- Pasted text intake for short formative material
- Browser upload intake for text/Markdown/reStructuredText files
- Advanced local text/Markdown file intake from inside `tools_workspace`
- Explicit conversation learning intake when a user says "learn from" /
  "study this" / "digest" with a URL or sufficiently long pasted material
- An evolution snapshot for first/latest experience, dominant drive,
  strongest drive drift, and change-type mix
- An evolution timeline of belief, drive, self-trait, and worldview changes
- An experience ledger showing what was ingested and why it was salient
- Current belief and drive summaries

After intake, normal runtime sessions retrieve a compact generation context
from this ledger: current beliefs, drive shifts, and recent evolution events.
Raw excerpts stay in the admin/SQLite ledger; they are not injected into every
chat prompt.

Conversation learning is deliberately explicit. A request such as "learn from
this project https://github.com/..." fetches readable source text, records the
URL as `source_ref`, writes evolution events, and appends a short learning
receipt to the reply. A normal web search or casual discussion does not write
to Life History.

Data is stored in `data/shared/life_history.db`. This is shared assistant
identity data, not per-user chat memory. User deletion does not remove it.
For an ordinary-mode shared-identity reset, stop the service, take any needed
backup, and remove `data/shared/self_model.db`, `data/shared/life_history.db`,
and their SQLite sidecars. Per-user memories and session state remain.
This is not a research-subject reset: `data/shared/agent_runtime.db` retains a
resumable manifest and checkpoints. For a complete reset, follow
[the stopped-owner procedure](../PRIVACY.md#full-runtime-and-research-subject-reset).

## Running The Web Server

Development/local:

```bash
openrook-web
```

Fallback:

```bash
python3 -m interface.api
```

By default this starts Uvicorn on `127.0.0.1:8000`. If you run behind a reverse
proxy, terminate TLS at the proxy and keep OpenRook bound to localhost unless you
have a specific reason not to.

Example reverse-proxy shape:

```text
Internet -> HTTPS reverse proxy -> http://127.0.0.1:8000
```

OpenRook does not currently ship a systemd unit or Dockerfile. For production-like
usage today, run it under your existing process supervisor and persist both
`runtime_config.yaml` and `data_dir`.

## Backup And Export

In ordinary mode, from `/settings`:

- **Export Config** downloads a JSON export with secret values redacted.
- **Create Backup** creates a zip archive under `<data_dir>/backups`.

The backup includes:

- `runtime_config.redacted.json`
- data files under `data_dir`

The backup intentionally skips the backup directory itself to avoid recursive
archives.

Backups may still contain user memory data. Treat backup files as sensitive
even though runtime config secrets are redacted.

### Research backup with the owner stopped

Research mode rejects **Create Backup** with `409`. Stop the research owner and
any other process writing to the same storage, then copy the entire configured
`data_dir` to a separate backup location outside that directory. Include all
shared stores, per-user/session records, and any SQLite sidecar files as one
snapshot; copying only `agent_runtime.db` does not capture the whole subject.
Copy associated study outputs and the configuration needed to restore the run
separately. Keep the owner stopped until the snapshot is complete.

## Reset And Delete

In ordinary mode, destructive admin actions require exact typed confirmations:

- Restart web server: confirmation must be `RESTART`
- Reset session: confirmation must be `RESET <session_key>`
- Delete user: confirmation must be `DELETE <platform>:<user_id>`

User deletion removes only the selected user's per-user data directory and live
sessions. Shared self-model, Life History, and research stores remain. See
[PRIVACY.md](../PRIVACY.md#deletion) for the full deletion contract.

Research mode rejects these administrative mutations and all DELETE requests
with `409`. Stop its owner before any manual deletion. For one user's local
records, follow [manual user deletion](../PRIVACY.md#manual-user-deletion-with-the-service-stopped);
for a fresh subject, follow the
[full runtime reset](../PRIVACY.md#full-runtime-and-research-subject-reset).
There is no broad "delete all data" control.

## Updating

**Source install:**

```bash
git pull
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/test_interface.py tests/test_interface_v1.py -q
```

**Git install:**

```bash
python3 -m pip install --upgrade "git+https://github.com/balfiky/openrook.git"
```

Then restart `openrook-web`.

If `OPENROOK_CONFIG_DIR` is set, your identity file is untouched. If it is not set,
a package upgrade may overwrite the bundled `config/soul.yaml` with the
defaults shipped in the new package.

If your deployment has real user data, create a backup before upgrading.

## Self-Evolution Surfaces

These surfaces let an operator inspect and shape the identity-level
evolution layer described in
[ARCHITECTURE.md § Self-Evolution Model](ARCHITECTURE.md#self-evolution-model).

| Endpoint | Purpose |
|---|---|
| `GET /admin/identity/constitution` | Read the operator-authored stable orientation string |
| `PUT /admin/identity/constitution` | Replace the constitution (≤2000 chars, whitespace-trimmed) |
| `GET /admin/life/open-questions` | List queue with `?status=open\|pursuing\|resolved\|abandoned\|all` and counts |
| `POST /admin/life/open-questions/{id}/abandon` | Operator override; transitions open/pursuing → abandoned |
| `POST /admin/life/open-questions/{id}/resolve` | Mark a question resolved; takes optional resolution_experience_id |
| `POST /admin/life/metabolism/tick` | Manually trigger the wall-clock metabolism tick (idempotent under the ≥1-day rate limit) |

The Settings → **Life History** page surfaces all of these as UI: Constitution
panel with Save button, Open Questions list with Abandon, and the metabolism
tick fires automatically on session start.

## Tests And Release Readiness

The repository includes a Makefile with three test entry points:

```bash
make test                 # non-UAT unit/integration suite, no live model required
make uat                  # 67 UAT tests, ~14 min, requires OPENROOK_UAT_LIVE=1 + a real backend
make uat-comprehensive    # one PASS/FAIL aggregator over the whole UAT suite
```

Use `make uat-comprehensive` before a public release for a single green/red
signal. See [UAT.md](UAT.md) for env vars, backend selection, headed mode,
and per-test artifact paths.

## Known Gaps

- **`character_independence` alone is not an administrative lock.** Use
  `research_mode` for the implemented persistent subject and read-only observation
  routes. Research birth manifests fix initial conditions across restart;
  starting a new condition requires a separate run. Do not treat the legacy
  wizard toggle as equivalent to the research preset.
- **No skill-from-URL download.** `skills.create_from_request` accepts a
  `source_url` argument but only stores it as a metadata reference — no
  fetch is performed. Operators acquire skills via paste, file path, or
  zip upload only.
- **Autobiographical learning is selective.** With `autobiographical_learning`
  enabled, salient distilled sessions enter Life History at session end,
  including first-use database creation. Research mode enables it automatically;
  ordinary chat leaves it opt-in. It is not a raw-transcript ingestion switch.

## Operational Notes

- Health probe: `GET /v1/health`
- Readiness probe: `GET /v1/ready`
- Admin diagnostics: `GET /admin/diagnostics`
- Stable integration API: `/v1/*`
- Built-in OpenAPI docs: `/docs`

When `api_key` is set, admin and integration calls need:

```bash
Authorization: Bearer <api_key>
```

The browser UI stores the token locally in the browser after you enter or rotate
it through the settings workspace.
