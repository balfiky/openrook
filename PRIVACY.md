# Privacy & Data Retention

OpenRook is a cognitive runtime that maintains **persistent state**
across turns and sessions: memory (relational, semantic, long-term),
identity (beliefs, drives, self-traits), emotion modulators, and the
identity-level Life History ledger. That means it stores information
about the people who interact with it *and* about the runtime's own
evolution. This document describes what is stored, where, and how to
inspect, export, or delete it.

## Storage Layout

Runtime state lives under a configurable directory:

- `runtime_config.yaml → data_dir` (default: `./data/`)

Within `data_dir`, OpenRook writes:

| Path | Contents |
|------|----------|
| `data/shared/self_model.db` | Shared self-model — observations and defense log for entity `__self__` |
| `data/shared/life_history.db` | Shared life-history/evolution ledger — formative experiences, belief revisions, drive changes, self-trait observations |
| `data/shared/agent_runtime.db` | Research subject manifest, checkpoints, intentions, goals, action/outcome history and learning records |
| `data/<platform>_<user_id>/openrook.db` | Per-user SQLite: emotional memory, relationship events, open loops, person/topic profiles, semantic memory |
| `data/<platform>_<user_id>/sessions/<chat_id>.json` | Per-session engine state: `modulator_snapshot`, `saved_at`, and `unresolved_items` |
| `data/<platform>_<user_id>/sessions/<chat_id>.history.json` | Active hot transcript for that chat, restored until the session is explicitly ended or reset |
| `data/<platform>_<user_id>/engine_state.json` | Legacy per-user engine-state snapshot (kept for backward-compatible restore) |

The entire `data/` directory is gitignored. Do not commit it.

Research jobs can additionally write manifests, model prompts/responses, tool
receipts, replay frames, and reports under the explicitly selected study output
directory, commonly `reports/`. Those artifacts can contain sensitive content
even when credential fields are redacted. Custom data/output directories need
their own ignore rules and access controls; the default `data/` rule does not
cover every possible location. Existing per-user databases migrate to
`openrook.db` when first opened; records and schemas are preserved.
See the [migration guide](docs/OPENROOK_MIGRATION.md) for the upgrade procedure.

## Tables Inside the Per-User DB (`openrook.db`)

| Table | Columns of interest | Purpose |
|-------|--------------------|---------|
| `memories` | `source_person`, `topic`, `summary`, `emotional_valence`, `trust_delta`, `confidence`, `spike` | Distilled long-term emotional summaries |
| `relationship_events` | `source_person`, `event_kind` (rupture/repair/commitment/recurring_tension), `intensity`, `valence` | Social-arc events |
| `open_loops` | `source_person`, `loop_kind`, `status`, `description`, `intensity` | Unresolved tension or pending follow-up |
| `observations` | `entity_id`, `trait`, `value` | Behavioral observations for person and topic profiles |
| `extracted_traits` | `entity_id`, `trait`, `score` | Rolled-up traits derived from observations |
| `semantic_memories` | `source_person`, `kind`, `topic`, `summary`, `content` | Stated preferences, decisions, episodes, facts |

In the per-user `openrook.db`, `entity_id` values are either a user id (person
profile) or a topic name (topic profile).

## Tables Inside the Shared DB (`data/shared/self_model.db`)

The runtime deliberately splits self-model storage away from per-user data
(`self_db_path = config.shared_db_path` in `runtime/sessions/manager.py`):

| Table | Columns of interest | Purpose |
|-------|--------------------|---------|
| `observations` | `entity_id` (always `__self__`), `trait`, `value` | Self-observation records |
| `extracted_traits` | `entity_id` (always `__self__`), `trait`, `score` | Rolled-up self-traits |
| `defense_events` | `timestamp`, `defense_type`, `raw_intensity`, `expressed_intensity`, `suppression_delta` | Defense-mechanism activations (self-log; no `entity_id` column) |

## Tables Inside the Shared Life-History DB (`data/shared/life_history.db`)

The Life History layer is identity-level data. It may include material pasted
by an operator or read from a local file, plus OpenRook's derived interpretation of
that material:

| Table | Columns of interest | Purpose |
|-------|--------------------|---------|
| `experience_events` | `source_type`, `source_title`, `source_ref`, `participants_json`, `content_summary`, `raw_excerpt`, `salience`, `emotional_impact` | What OpenRook encountered and how salient it was |
| `evolution_events` | `domain`, `subject`, `before_state`, `after_state`, `reason`, `confidence`, `evidence` | What changed after an experience |
| `beliefs` | `key`, `statement`, `confidence`, `status`, `evidence` | Current worldview/belief records |
| `belief_revisions` | `before_statement`, `after_statement`, `reason`, `confidence` | Belief revision history |
| `drive_states` | `name`, `value`, `description` | Current drive/motivation values |
| `drive_changes` | `drive_name`, `before_value`, `after_value`, `delta`, `reason` | Drive change history |

This database is shared across relationships. Do not paste third-party
private material into Life History unless you have the right consent.

## What Is Actually Recorded

Active chat transcripts are stored in per-session `*.history.json` files so
the browser can restore the current conversation after reload, and Telegram
and other channels can restore it after idle eviction or runtime restart.
These hot transcripts are cleared by `/new`,
`/reset`, and explicit session-end operations. Separately, the `memories` table
holds *distilled emotional summaries*, not full conversation logs. The
`semantic_memories` table may quote short user statements verbatim when the
user expresses a preference or fact (e.g., "I prefer concise replies") — review
it before sharing a database.

After **End Session**, the browser keeps the ended transcript on the current
page above a divider until reload. It does not save a browser transcript
archive. Explanation cards are separate from messages and are never included
in conversation transcripts.

Life History is different: pasted text and local-file excerpts can be stored
as experience evidence. Treat `data/shared/life_history.db` as sensitive.
When autobiographical learning is enabled, salient distilled sessions can enter
this shared store too; research mode enables that path. Research event/replay
records may also retain conversation or tool content separately from hot chat
transcripts, so clearing a chat is not a complete study-data deletion.

## Consent

If you deploy OpenRook to talk to anyone other than yourself, **you are
responsible for obtaining their informed consent** to the persistence
described above. A minimum disclosure should state:

- OpenRook stores inferred affect and relationship state, not just what was said
- Memory persists across sessions unless explicitly wiped
- Trust and bonding build and break based on interaction history
- Users can request export or deletion (see below)

## Inspection APIs

The `/v1` API surfaces what OpenRook knows about a user:

- `GET /v1/profiles/person?user_id=<id>`
- `GET /v1/profiles/self`
- `GET /v1/memory/long_term?user_id=<id>`
- `GET /v1/memory/relationship?user_id=<id>`
- `GET /v1/memory/semantic?user_id=<id>`

## Deletion

In ordinary mode, the built-in admin console at `/admin` also exposes guarded
user deletion. It requires the operator to type `DELETE <platform>:<user_id>`
before the request is accepted. Research observation is read-only: DELETE
requests and mutating administrative requests return `409`, including user
deletion, resets, and backup creation. Use the stopped-owner procedures below
for a research subject.

### API (recommended in ordinary mode)

`DELETE /v1/users/{platform}/{user_id}` wipes all persisted data for a
user on a given platform in one call:

- Evicts every live session for that `platform:user_id` (drain,
  digest, close) before touching disk.
- Deletes the per-user SQLite DB (`openrook.db`), any legacy database and
  sidecars, and every session JSON/transcript.
- Removes the per-user directory itself.
- **Does not touch** `data/shared/self_model.db` — the shared self-model
  holds only rows for entity `__self__` and a defense log with no
  per-user column, so it is preserved by design.
- **Does not touch** `data/shared/life_history.db` — formative experiences
  and evolution events are assistant-level identity data. They may still
  mention people if the operator included them in a Life intake.
- **Does not touch** shared research state in `agent_runtime.db`, study output
  directories, exported reports, or backups. Review those separately when
  honoring a deletion request; autogenerated autobiographical records can also
  mention the person even without manual intake.

```bash
# With auth disabled (api_key empty):
curl -X DELETE http://localhost:8000/v1/users/web/alice

# With auth enabled:
curl -X DELETE http://localhost:8000/v1/users/web/alice \
  -H "Authorization: Bearer my-secret-token"
```

Python client:

```python
from interface.client import OpenRookClient

with OpenRookClient("http://localhost:8000", api_key="my-secret-token") as client:
    result = client.delete_user("alice", platform="web")
    print(result)
    # {
    #   "deleted": true,
    #   "rel_key": "web:alice",
    #   "rows_deleted": {"memories": 3, "relationship_events": 1, ...},
    #   "session_files_removed": 2,
    #   "sessions_evicted": ["web:alice:default"],
    #   "path_removed": "data/web_alice",
    #   "shared_self_model_db_preserved": true
    # }
```

Returns `404` if no data exists for that `platform:user_id`. Row counts
are best-effort: if a count query fails, that table reports `null` but
the wipe still proceeds.

### Manual user deletion (with the service stopped)

Stop the service and any research owner before removing the selected user's
directory:

```
rm -rf data/<platform>_<user_id>
```

This removes `openrook.db`, any legacy database and sidecars, and all session
JSON/transcript files in one step.
Shared identity and research stores, study outputs, and backups remain; inspect
them separately for material concerning that user.

### Full runtime and research-subject reset

Stop the owner process and every other writer before taking any required
[research backup](docs/DEPLOYMENT_AND_ADMIN.md#research-backup-with-the-owner-stopped).
Then remove the entire configured data directory, including `shared/self_model.db`,
`shared/life_history.db`, `shared/agent_runtime.db`, per-user records, and SQLite
sidecar files. A surviving research manifest or checkpoint can resume the old
subject, so deleting only the self-model and Life History databases is not a
research-subject reset.

Remove or securely archive associated research outputs and backups separately.
Runtime and identity configuration outside `data_dir` remains in place; the next
startup uses that configuration to initialize fresh state.

## Retention

There is no automatic expiration. Old memories decay in *activation*
(retrieval weighting) but remain on disk indefinitely. If you need
time-bounded retention, run a scheduled cleanup or add a retention policy
to your deployment.

## Data Leaving the Host

OpenRook does not phone home. However, if you configure an external LLM backend
(MiniMax, OpenAI-compatible, etc.), **every turn sends the assembled
prompt — which includes retrieved memory, profiles, and relationship
context — to that provider**. Runtime sessions may also include a compact
Life History slice: current beliefs, shifted drives, and recent evolution
events. Raw Life History source excerpts are not injected into every chat
prompt, but Life History digestion can call the configured LLM backend when
available, sending the pasted text or file summary being digested. Review the
provider's data-handling policy before enabling it for real users.

## Reporting Privacy Issues

See [SECURITY.md](SECURITY.md).
