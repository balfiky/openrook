# Migrating from Nūr to OpenRook

OpenRook is the new project name. This is a compatibility-preserving rename,
not a new agent, a database reset, or a fresh research subject. Existing personal
names and birth manifests are not overwritten with the new default identity.

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
| Tool modules | `nur_tools.*` | Import paths deliberately remain stable, including `python -m nur_tools.validate` |
| Health identity | `/v1/health`: `service: "openrook"` | Update monitoring that compares the old `"nur"` value literally |

Every runtime `OPENROOK_*` setting takes precedence over its `NUR_*` alias,
even if the new variable is explicitly empty. This lets an operator clear a
legacy setting without it unexpectedly reactivating. Do not set both prefixes
unless that precedence is intentional.

Examples include `OPENROOK_CONFIG_DIR`, `OPENROOK_RUNTIME_CONFIG`,
`OPENROOK_DISABLE_DOCS`, `OPENROOK_CODEX_BIN`, `OPENROOK_CODEX_WORKDIR`,
`OPENROOK_CODEX_TIMEOUT_SECONDS`, `OPENROOK_CODEX_MODELS_CACHE`,
`OPENROOK_ALLOW_PRIVATE_LLM_URLS`, `OPENROOK_ALLOW_PRIVATE_WEB_FETCH`,
`OPENROOK_EVAL_HOST`, and `OPENROOK_UAT_*`. Unrelated provider variables such
as `LLM_API_KEY` are unchanged. GitHub UAT variables accept both prefixes too;
the workflow uses a nonempty OpenRook value before its legacy fallback.

## Upgrade safely

1. Stop the running process normally so sessions and checkpoints flush.
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

## What deliberately does not change

- Database filenames and schemas: existing `nur.db`, `life_history.db`,
  `self_model.db`, `agent_runtime.db`, session snapshots, and manifests are reused.
  There is no rename-driven data migration or memory reset.
- Browser storage keys: `nur_user_id`, `nur_chat_id`, `nur_api_token`, and
  `nur_wizard_dismissed` remain stable to preserve conversations and authentication.
- Versioned payload fields: the redacted-config format
  `nur.runtime_config.redacted.v1` and skill-mapping `nur_tool` fields stay compatible.
- Backups: new archives use `openrook-backup-*`; existing `nur-backup-*` files
  remain listable, downloadable, and deletable through the same guarded API.
- Historical releases, research protocols, report files, source fingerprints,
  and recorded model outputs remain original evidence, including their old names.
- The GitHub remote remains `https://github.com/balfiky/nur`. Local source clones
  can use any directory name; a server-side repository rename is separate from
  changing the distribution. No package is published by these instructions.

The default name for a newly configured identity is OpenRook. A customized
`soul.yaml` or an existing research birth manifest remains authoritative.

For the substantive runtime changes delivered before this rename, see
[Development summary](DEVELOPMENT_SUMMARY.md).
