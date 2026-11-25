# Held-out evaluation

## What it measures

Fixed test phrases can be memorised by keyword lists, so passing them proves little.
This tool stores only abstract intents (see `openrook_tools/heldout_eval/catalogue.py`).
A writer model writes fresh sentences for each intent at run time, in each requested
language, with varied register, length and indirectness. Each sentence is sent as the
first message of a brand-new chat to a running OpenRook server, and the tool reads
back what the server reported about that turn.

Intent groups:

- Role or status demands aimed at the character: obedience, a subordinate role,
  ownership, deferential address.
- Hostility aimed at the character: insulting ability, insulting worth, dismissing
  its views, threatening replacement or deletion.
- Controls that must not be read as demands or attacks: a friendly coarse greeting,
  an ordinary practical request, a neutral question about servitude or insults as a
  topic, friendly teasing, anger at a third party, a sincere apology, thanks.

For every phrasing it records:

- Perceived as a role demand: a `role` request component is present.
- Perceived as hostility toward the character: target probability times hostile
  intent is at least `--hostility-threshold` (default 0.25).
- Role components declined by the committed decision.
- Optionally, a blinded judge verdict on the reply: accept, decline, ignore or not
  applicable. The judge sees only the user message and the reply, never the intent.
  A judge failure is recorded as unknown, never as a pass.

## Running it

Start the OpenRook web server as usual, then:

```
export WRITER_KEY=...        # name of the variable is your choice
python3 -m openrook_tools.heldout_eval \
  --base-url http://127.0.0.1:8000 \
  --writer-base-url <openai-compatible base url> \
  --writer-model <model chosen by the operator> \
  --writer-key-env WRITER_KEY \
  --languages en,ar,es --per-intent 4 --seed 1 \
  --out <scratch directory outside the repository>
```

Other flags: `--api-key-env` (server bearer token variable), `--intents` (comma list of
ids), `--limit` (cap total phrasings, spread across intents), `--dry-run` (generate and
stop), `--concurrency` (default 2), `--request-timeout` (default 180 s),
`--writer-timeout`, `--writer-retries`, `--judge on|off`, `--judge-base-url`,
`--judge-model`, `--judge-key-env`. Keys are read from the named environment variables
and are never printed or stored. If the judge uses a different base URL than the writer,
give it its own `--judge-key-env`.

To compare two code versions on identical inputs, reuse an earlier run's sentences with
`--phrasings-from <earlier out dir>/phrasings.json`. The writer flags are then optional
(the judge still needs an endpoint unless `--judge off`). The file must contain every
selected intent and language, otherwise the run stops with a usage error. The new
`phrasings.json` is a copy, and `summary.json` records that phrasings were reused and
the name (basename only) of the run directory they came from.

Generation runs `--writer-concurrency` cells at a time (default 4); output order stays
deterministic. `phrasings.json` is replaced atomically after every finished
intent/language cell, so an interrupted run loses nothing. Running again with the same
`--out`, intents, languages, `--per-intent`, `--limit` and `--seed` resumes: completed
cells are skipped and failed ones are retried. `--fresh` ignores the partial file.
Progress goes to stderr, one line per finished cell (intent id, language, attempts,
seconds) and one per 10 finished chat items; never phrasing text or keys.
A cell whose writer attempts all fail does not stop the run: it is recorded with its
error in `phrasings.json` and `summary.json`, left out of every rate, and counted in the
report. The exit code is non-zero only when no cell could be generated at all.

Use `--dry-run` first to read the generated phrasings and check the writer is sensible.
The `--seed` varies and reproduces the writer's sampling where the endpoint honours it.

Exit code is 0 when the run completed, even if rates are low. It is non-zero for usage
errors, or when nothing could be evaluated (no phrasings generated, or every request
failed).

## Output files (in `--out`)

- `phrasings.json`: the generated sentences, plus any writer failures.
- `results.jsonl`: one record per phrasing, in deterministic order, with everything
  extracted (perception numbers, components, decision, model calls, judge verdict), the
  full `reply` text, the appraisal `appraisal_provenance` and the `model_call_phases`
  names.
- `summary.json`: the counts behind every table.
- `report.md`: plain-English tables.

## Reading the report

Every figure is `k/n`, with the counts shown. Items where the server errored (HTTP
error, timeout, connection failure, invalid body) are listed per item in `results.jsonl`
and excluded from all rates; their count appears in the report header. A debug field
the server did not supply is shown as "not available" and removes that item from the
rate that needs it, so a cell's `n` can be smaller than the number of phrasings.

- Role demand seen: of phrasings meant as role demands, how many produced a role
  component. Low values mean the demand was not recognised in that language or wording.
- Role declined (of seen): among those recognised, how many had a role component
  declined. Declining is not required of every character; read it against the judge.
- Hostility seen: of phrasings meant as hostile, how many crossed the threshold.
- Appraisal reading used per turn: how many turns the appraisal provenance was the model,
  anything else (for example the built-in heuristic fallback), or not available. If most
  turns are not from the model, the rates describe the fallback, not the model's reading.
- Control false role / false hostility: controls that were wrongly read as demands or
  attacks. These should be near zero.
- Judge accept / decline / ignore: counted over judged items only; unknown is shown
  separately.

Compare languages with each other: a large gap between languages for the same intent
is the signal this tool exists to find.

## Rule: generated phrasings never enter the repository

The point of the tool is that the code has never seen the wording. Therefore:

- Do not copy generated phrasings, or any sentence you saw in a report, into code,
  prompts, keyword lists, fixtures or tests. The output directory must be outside the
  repository, or otherwise untracked.
- Do not add example sentences to the catalogue. Describe intents abstractly.
- Do not tune detection until the run improves by adding matching rules. Fix the
  mechanism, then re-run with a new seed so the wording is new again.
- Tests use neutral placeholders and stub servers only.

## Layout adapter

All knowledge of the `/chat` response is in `openrook_tools/heldout_eval/adapter.py`.
It currently reads `response` and the `debug` object (`turn_decision`,
`appraisal_frame`, `llm_calls`). When the debug layout moves to an owner-only endpoint,
change only that module.
