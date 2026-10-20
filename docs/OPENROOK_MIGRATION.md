# OpenRook upgrade note

1. Stop running servers and research owners so sessions and checkpoints flush.
2. Back up the configured data directory, `runtime_config.yaml`, and any identity
   or prompt overrides. Keep customized configuration outside installed packages.
3. Install into a new virtual environment with `python3 -m pip install .` or
   `python3 -m pip install -e ".[dev]"`.
4. Keep the existing configuration path, data directory, and identity overrides.
   Use `OPENROOK_CONFIG_DIR` for a separate identity directory and
   `OPENROOK_RUNTIME_CONFIG` for a separate runtime configuration file.
5. Start `openrook-web --config runtime_config.yaml` or
   `openrook --config runtime_config.yaml`. Run only one research owner for each
   data directory.

Use `openrook` / `openrook-*` commands, `OPENROOK_*` settings, and
`openrook_tools` imports. The public classes are `config.OpenRookConfig`,
`runtime.app.OpenRookApp`, `interface.client.OpenRookClient`, and
`interface.client.OpenRookAPIError`. Health checks report `service: "openrook"`.
Deprecated commands, imports, settings, and metadata aliases are no longer read.

Existing per-user databases migrate automatically to
`data/<platform>_<user_id>/openrook.db` using SQLite backup, including committed
write-ahead data. The original database remains a recovery copy. If both database
names exist, the runtime uses `openrook.db` and warns; it does not merge records.
A failed migration leaves the source intact and fails instead of opening an
empty replacement. Keep earlier runtimes stopped after upgrading because the
recovery copy is not synchronized. Do not run `openrook-uninstall`, reset
genesis, or repeat setup as an upgrade step.

Browser settings use `openrook_*` storage keys. Backups use `openrook-backup-*`
and `openrook.runtime_config.redacted.v1`; skills use `openrook_tool` metadata.
For offline data recovery, stop the runtime, extract backed-up `data/` into an
empty directory, point `data_dir` at it, and restore configuration and secrets
from the separate backup. Restored per-user databases migrate when first opened.

Customized identities, shared databases, session records, and existing birth
manifests retain their data. Historical request replay requires the matching
code revision because schema and request names have changed. See
[Development summary](DEVELOPMENT_SUMMARY.md) for runtime changes.
