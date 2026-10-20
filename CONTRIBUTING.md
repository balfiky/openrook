# Contributing to OpenRook

OpenRook is a research-oriented hybrid cognitive architecture. Contributions
that sharpen the architecture, improve inspectability, or strengthen
empirical validation are especially welcome.

## Ground Rules

- **Respect the single-mind boundary.** The cognitive pipeline owns
  emotional state, memory, self-model, and tool policy. Runtime/channel
  code must not shadow or duplicate those concerns.
- **Math first, LLM only where necessary.** Modulator updates, decay,
  profile math, strategy selection, and retrieval ranking must remain
  deterministic and unit-testable.
- **Every change needs a test.** Either a focused unit test, an eval
  scenario, or a regression lock in `tests/test_regressions.py`.
- **Keep prompts and constants config-driven.** New thresholds go in
  `config/*.yaml`; new prompts go in `config/prompts/`.

## Development Setup

```bash
git clone https://github.com/balfiky/openrook.git
cd openrook
python3 -m pip install -e ".[dev]"
python3 -m pytest -q        # full suite; see CI for the current collected count
```

Optional:

```bash
python3 -m evals --backend mock --tag phase11    # behavioral eval pack
```

## What Goes Where

Code:

| Change kind | Primary path |
|---|---|
| Cognitive logic | `core/` (emotional engine, memory, profiles, dual process, appraisal, defense) |
| Turn orchestration | `pipeline/` (orchestrator in `pipeline/core.py`) |
| Session / channel / runtime | `runtime/` |
| HTTP surface | `interface/api.py`, `interface/v1.py` |
| Web UI | `interface/static/index.html` |
| Evals and scenarios | `evals/`, `tests/calibration/scenarios.py` |
| Tests | `tests/` (unit by module; integration in `test_interface*`, `test_runtime*`, `test_regressions.py`) |

Docs:

| Change kind | Update |
|---|---|
| Runtime hosts, backends, or shared cognition | `nur_tools/build_diagrams.py::build_runtime_architecture` + `docs/ARCHITECTURE.md` Runtime Topology section |
| Persistent research lifecycle | `nur_tools/build_diagrams.py::build_research_runtime_autonomy` + `docs/RESEARCH_RUNTIME.md` |
| Pipeline stage order | `nur_tools/build_diagrams.py::build_single_turn_flow` + `docs/ARCHITECTURE.md` Single-Turn section |
| Storage / table changes | `nur_tools/build_diagrams.py::build_memory_persistence` + `docs/ARCHITECTURE.md` + `PRIVACY.md` |
| HTTP route or auth changes | `nur_tools/build_diagrams.py::build_auth_tool_safety` + `docs/ARCHITECTURE.md` + `SECURITY.md` |
| Admin / wizard behavior | `docs/DEPLOYMENT_AND_ADMIN.md` |
| New eval variant | `nur_tools/build_diagrams.py::build_evaluation_harness` + `evals/ablation_hypotheses.py` **before** running the eval + `docs/ARCHITECTURE.md` Evaluation section + `docs/OVERVIEW.md §4` results table |
| Component evidence status changes | `nur_tools/build_diagrams.py::build_component_claim_map` + `docs/OVERVIEW.md §5` |
| Channels, persistent state, or headline capabilities | `nur_tools/build_diagrams.py::build_feature_overview` + `README.md` Architecture At A Glance section |
| Shared visual theme or fonts | `nur_tools/brand.py` + diagram and showcase parity tests; inspect both themes |
| Modulator config, event impacts, or emotion labels | `nur_tools/build_showcase.py` (engine-replay hero, fingerprints, cover) — re-render; `tests/test_showcase_visuals.py` must pass |
| New headline study result | `nur_tools/build_showcase.py` data constants + `tests/test_showcase_visuals.py` |
| Release notes | `CHANGELOG.md` under a new version header |

## Regenerating Diagrams

Architecture diagrams are generated from Python by
`nur_tools/build_diagrams.py`, rendered to SVG by the toolkit in
`nur_tools/diagram_toolkit.py`, and rasterized to PNG by CairoSVG, included in
the `dev` extra. Each of the eight builders produces `-light.svg`, `-dark.svg`,
`-light.png`, and `-dark.png` assets tracked in `docs/diagrams/`. The feature
overview uses the same builder and rendering path.

| Diagram | Builder | Type | Focal element |
|---|---|---|---|
| Research runtime autonomy | `build_research_runtime_autonomy` | Loop | Conditional claim verification and persistent state |
| Runtime architecture | `build_runtime_architecture` | Architecture | Session and cognition core |
| Single-turn cognitive flow | `build_single_turn_flow` | Swimlane | Conditional self-check, with model calls alongside runtime stages |
| Memory persistence model | `build_memory_persistence` | Nested containment | Per-user deletion boundary |
| Auth and tool safety | `build_auth_tool_safety` | Flowchart | Independent authentication and tool gates |
| Evaluation ablation harness | `build_evaluation_harness` | Data flow | Provenance and outcome labels |
| Component claim map | `build_component_claim_map` | Evidence matrix | Suite-specific relationship-memory effect |
| Feature overview | `build_feature_overview` | Layer stack | Persistent state |

Install **DejaVu Sans**, **DejaVu Serif**, and **DejaVu Sans Mono** before rendering
(for example, the `fonts-dejavu-core` package on Debian/Ubuntu). Use the shared
`DIAGRAM_SERIF` stack (the brand `SERIF` stack with DejaVu Serif first for
CairoSVG) for titles, sans for node names, and mono only for technical
tokens such as paths and flags. Fonts are system dependencies,
not installed by pip: substitutions can change text widths or produce missing
glyph boxes even when the SVG is valid. Use `fc-match 'DejaVu Sans'` to check
resolution on systems with fontconfig.

After editing the generator, run:

```bash
python3 nur_tools/render_diagram_pngs.py
```

Inspect each light rendering on white and each dark rendering on `#0d1117`,
at approximately 900px wide. Check glyphs, text fit, connector routes, and
contrast. Commit the regenerated SVGs and PNGs together; deterministic builder
tests compare every diagram and theme against the tracked SVG.

Embed diagrams using a theme-aware picture with the light SVG as the fallback:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/runtime-architecture-dark.svg">
  <img src="diagrams/runtime-architecture-light.svg" alt="Runtime hosts, cognition, model, tools, and stores.">
</picture>
```

Adjust the relative paths for the document. Keep long explanations and evidence
qualifications in the surrounding prose.

Design language for new diagrams:

- Shared `Theme`, `LIGHT`, `DARK`, and font tokens live in `nur_tools/brand.py`;
  both diagram and showcase builders use them. Primitives live in
  `nur_tools/diagram_toolkit.py`.
- Keep transparent backgrounds and the established ink, muted, rule, amber,
  plum, and modulator tokens. Use accent on at most two focal elements.
- Delete repetition first: one idea per node, a short name and one short
  sublabel, roughly 4/10 visual density, and at most nine nodes before splitting.
- Distinguish focal elements, steps, stores, external systems, inputs,
  optional or asynchronous work, and boundaries through shape and line treatment.
- Use flat shapes, no shadows, and corner radii at most 8px.
- Route connectors orthogonally or as true arcs, with unique attachment points.
  Avoid crossed or overlapping paths, diagonal slants, and lines behind
  unrelated boxes. Put connector labels on masking rectangles.
- Keep legends outside the drawing area, or omit them when the roles are clear.

## Adding An Eval Scenario

1. Author the scenario in `evals/scenarios.py` or
   `tests/calibration/scenarios.py` depending on scope.
2. Before running, register the outcome hypothesis in
   `evals/ablation_hypotheses.py`: one row per (scenario, variant)
   with `expected_failure`, `unexpected_failure`, `no_effect`, or
   `newly_passing`. This is the pre-registered prediction the eval
   report labels each outcome against.
3. Run `python3 -m evals --backend mock --tag <your-tag>` for a
   fast offline pass, then `--backend provider` (or similar) for a
   live-LLM run when relevant.
4. If a new component becomes load-bearing or shifts category, update
   `docs/OVERVIEW.md §5` and
   `nur_tools/build_diagrams.py::build_component_claim_map`.

## Writing A Regression Test

Regressions for confirmed-fixed bugs live in
`tests/test_regressions.py`. Each test should name the original bug in a
comment and assert the fix. Keep them short; one class per bug family.

## Pull Request Checklist

- [ ] Tests pass: `python3 -m pytest`
- [ ] Eval suite passes when relevant: `python3 -m evals`
- [ ] Diagrams regenerated if any architecture-sensitive change touched
      the build generator
- [ ] Docs updated per the matrix above
- [ ] `CHANGELOG.md` updated under an appropriate version header
- [ ] No secrets, API keys, or user session data committed
- [ ] For behavioral changes: a scenario that locks the new behavior
- [ ] For new cognitive components: an ablation-friendly toggle so the
      system can run with/without the component for validation

## What We Are Especially Looking For

- **Empirical validation**: ablation studies, baseline comparisons,
  blinded human ratings, benchmarks against real LLM backends (not
  mocks).
- **Scientific grounding**: tightening the mapping between
  implementation and the cited theories (PSI, ACT-R, CLARION), or
  honest reframing where the current implementation diverges.
- **Privacy and safety**: retention controls, data-export/delete APIs,
  consent surfaces, anthropomorphism guardrails.

## Reporting Issues

Open a GitHub issue. For anything security- or privacy-sensitive, see
[SECURITY.md](SECURITY.md) before filing publicly.

## Code Of Conduct

Be respectful, be direct, disagree on substance. No personal attacks.
