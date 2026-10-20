# OpenRook migration guide

OpenRook (formerly Nūr).

The rename preserves existing records, configured identities, and research
subjects. Existing personal names and birth manifests remain authoritative.

## Names and compatibility

| Surface | Preferred name | Compatibility behavior |
|---|---|---|
| Python distribution | `openrook` | Replaces `project-nur`; avoid installing both into one environment |
| Runtime commands | `openrook`, `openrook-web`, `openrook-setup` | `nur`, `nur-web`, `nur-setup` remain aliases |
| Other commands | `openrook-uninstall`, `openrook-genesis-reset`, `openrook-emotional-qa`, `openrook-uat`, `openrook-validate` | Corresponding `nur-*` commands remain aliases |
| Environment settings | `OPENROOK_*` | Corresponding `NUR_*` values are fallback aliases |
| Configuration class | `config.OpenRookConfig` | `config.NurConfig` is the same class |
| Runtime class | `runtime.app.OpenRookApp` | `runtime.app.NurApp` is the same class |
| Python client | `interface.client.OpenRookClient`, `OpenRookAPIError` | `NurClient`, `NurAPIError` remain aliases |
| Tool modules | `openrook_tools.*` | `nur_tools.*` imports remain available through a deprecated compatibility shim |
| Health identity | `/v1/health`: `service: "openrook"` | Update monitoring that compares the old `"nur"` value literally |

Every runtime `OPENROOK_*` setting takes precedence over its `NUR_*` alias,
even if the new variable is explicitly empty. This lets an operator clear a
legacy setting without it unexpectedly reactivating. Do not set both prefixes
unless that precedence is intentional.

Using a `nur` / `nur-*` command or reading a `NUR_*` fallback prints a one-line
deprecation notice to stderr once per process. Importing `nur_tools` emits
`DeprecationWarning`; change imports and module commands to `openrook_tools`,
for example `python3 -m openrook_tools.validate --mode quick`.

Examples include `OPENROOK_CONFIG_DIR`, `OPENROOK_RUNTIME_CONFIG`,
`OPENROOK_DISABLE_DOCS`, `OPENROOK_CODEX_BIN`, `OPENROOK_CODEX_WORKDIR`,
`OPENROOK_CODEX_TIMEOUT_SECONDS`, `OPENROOK_CODEX_MODELS_CACHE`,
`OPENROOK_ALLOW_PRIVATE_LLM_URLS`, `OPENROOK_ALLOW_PRIVATE_WEB_FETCH`,
`OPENROOK_EVAL_HOST`, and `OPENROOK_UAT_*`. Unrelated provider variables such
as `LLM_API_KEY` are unchanged. The project's Makefile and GitHub UAT workflow
use only `OPENROOK_*`; rename any repository variables used by those jobs.

## Upgrade safely

1. Stop every running server and research owner normally so sessions and
   checkpoints flush. Keep old code stopped while the upgraded runtime runs.
2. Back up `runtime_config.yaml`, the configured data directory, and any identity
   or prompt overrides. If you edited the old package's `config/` files directly,
   copy those edits to a user-owned configuration directory before uninstalling.
3. Prefer a new virtual environment, then install this checkout with
   `python3 -m pip install .` or `python3 -m pip install -e ".[dev]"`.
4. In the existing workspace, keep the same runtime-config path, data directory,
   and identity override directory. Use `OPENROOK_CONFIG_DIR` to point at that
   existing directory, or leave a working `NUR_CONFIG_DIR` setting in place.
5. Start `openrook-web --config runtime_config.yaml` or
   `openrook --config runtime_config.yaml`. Do not start both as separate research
   owners of the same data directory.

If reusing the same Python environment instead, remove the old distribution
**before** installing OpenRook, because both distributions own shared module paths:

```bash
python3 -m pip uninstall project-nur
python3 -m pip install -e ".[dev]"
openrook --version
openrook-web --version
openrook-validate --mode quick
```

`pip uninstall project-nur` removes the old installed code, not the workspace.
**Do not run `nur-uninstall` or `openrook-uninstall` as a migration step:** those
commands remove runtime configuration and data. Do not reset genesis or rerun
setup to accomplish the rename.

## Per-user databases

New installations store per-user records in
`data/<platform>_<user_id>/openrook.db`. When the runtime resolves a user's
database path, it handles existing storage before opening that user's stores:

- If only `nur.db` exists, SQLite's backup API copies a consistent database,
  including committed `nur.db-wal` content, into a temporary file beside it.
  After flushing that file, the runtime publishes `openrook.db` atomically
  without replacing an existing destination. Session creation and migration
  locks serialize concurrent resolution; publication also handles races
  between processes.
- The original `nur.db` and its SQLite sidecars remain as a recovery copy.
  No source records are deleted, and schemas remain compatible.
- If both names already exist, the runtime uses `openrook.db`, leaves
  `nur.db` untouched, and logs a warning. It does not merge the two databases.
- A failed migration raises an error and leaves the source intact rather than
  starting a new empty database.

The old copy is not synchronized after migration. Do not restart an old server
against the same directory: its subsequent writes would remain in `nur.db`.
Keep the complete pre-upgrade backup if you need to roll back the whole
installation.

Ordinary-mode `DELETE /v1/users/{platform}/{user_id}` removes the entire user's
directory, including both database names and their sidecars. Shared identity
stores and separately stored backups remain outside that deletion boundary.
Research-mode administration remains read-only; stop the research owner before
an offline backup or reset.

## Browser state, backups, and skills

On page load, browser storage moves `nur_user_id`, `nur_chat_id`,
`nur_api_token`, and `nur_wizard_dismissed` to their corresponding
`openrook_*` keys, then removes the old keys. An existing current key wins;
the migration preserves user identity, selected chat, authentication, and the
wizard dismissal without overwriting newer settings.

New backups use the `openrook-backup-*` prefix and redacted configuration
format `openrook.runtime_config.redacted.v1`. Existing `nur-backup-*` archives
remain accessible. Restoration is an offline operation: stop the runtime,
extract an archive's `data/` contents into a separate empty directory, point
`data_dir` at that restored directory, and restore the runtime configuration
and secrets from your separate backup.
There is no automated configuration-import endpoint. The redacted configuration
envelopes, including historical `nur.runtime_config.redacted.v1` exports, remain
archival metadata; either envelope can accompany the extracted data.
Data-directory archives preserve whichever database names exist, and restored
legacy-only data migrates when first opened. Research initial-state snapshots
include databases under either name using SQLite backups; replay and evaluation
data copies use the same per-user path resolution.

New skill metadata uses `openrook_tool`; loading still accepts `nur_tool`.
Existing mappings therefore remain usable while new metadata uses the
current name.

## Identity and historical evidence

Shared `life_history.db`, `self_model.db`, and `agent_runtime.db` names,
session snapshots, and manifests retain their existing roles. The default
name for a newly configured identity is OpenRook; customized `soul.yaml`
values and existing research birth manifests are preserved.

Historical releases, research protocols, report files, source fingerprints,
and recorded model outputs remain original evidence, including their old
names. Request replay of historical captures requires their historical code
revision: structured-output schema and other request-level names have changed.

The GitHub repository is `https://github.com/balfiky/openrook`; old
`balfiky/nur` URLs redirect. Existing clones keep working, or update them with
`git remote set-url origin https://github.com/balfiky/openrook.git`. Local clones
can use any directory name. No package is published by these instructions.

For the substantive runtime changes delivered before this rename, see
[Development summary](DEVELOPMENT_SUMMARY.md).
