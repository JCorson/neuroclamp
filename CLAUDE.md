# Agent Instructions

When working with Python, invoke the relevant /astral:<skill> for uv, ty, and ruff to ensure best practices are followed.

### Code Intelligence

Prefer LSP over Grep/Glob/Read for code navigation:
- `goToDefinition` / `goToImplementation` to jump to source
- `findReferences` to see all usages across the codebase
- `workspaceSymbol` to find where something is defined
- `documentSymbol` to list all symbols in a file
- `hover` for type info without reading the file
- `incomingCalls` / `outgoingCalls` for call hierarchy

Before renaming or changing a function signature, use
`findReferences` to find all call sites first.

Use Grep/Glob only for text/pattern searches (comments,
strings, config values) where LSP doesn't help.

After writing or editing code, check LSP diagnostics before
moving on. Fix any type errors or missing imports immediately.
Only fall back to Grep/Glob when the LSP cannot answer the question (e.g. searching for a raw string pattern, or locating non-Python files).

## Project overview

Patch clamp experiment simulator with a core library (`neuroclamp/`) and a Reflex web UI (`neuroclamp_ui/`). The core library models ion channels, gating variables, voltage/current clamp protocols, and the HH differential equations. The UI provides interactive controls and live trace plots.

## Spelling

Use **US English** in all code, comments, docstrings, and documentation. LLM-assisted writing on neuroscience topics tends to drift toward British forms (depolarise, hyperpolarise, behaviour, modelling, centre, fibre, colour, analyse, initialise, etc.) because of the training-corpus mix. Always emit the US form: `depolarize`, `hyperpolarize`, `behavior`, `modeling`, `center`, `fiber`, `color`, `analyze`, `initialize`, etc.

## Biological accuracy

**Prioritize biological accuracy** when designing or modifying neuron presets. If a preset produces behavior that doesn't match known physiology (wrong resting potential, wrong firing pattern, missing channels), fix the underlying biology rather than working around it. When a full biological fix is out of scope (e.g. a channel type not yet implemented), document the known limitation explicitly in the preset comment and in an issue — do not silently accept a biologically wrong default.

## Comments and docstrings

Comments and docstrings explain the **current** design — what the code does and why it's biologically or numerically correct — in the present tense. They are not a changelog.

Do **not** include:

- Issue or PR numbers cited as "this fixes…" or "this changed in…" (`#347`, `#348`).
- Temporal framing about prior states: "before this PR", "post-#348", "prior to the alignment", "was previously", "originally proposed", "legacy", "restores".
- Descriptions of removed code or the bug that was fixed ("was silently stretched 2.5×", "the earlier rate was a 100 kHz protocol re-interpreted as 40 kHz").
- "Widened from X to Y" or "extended from X to Y" framings — just state the current band/duration and why.

Investigation history belongs in the PR body and the issue thread, not in source.

Issue links **are** appropriate when documenting an open limitation that has not been fixed yet (see `## Biological accuracy`).

## Architecture

- `neuroclamp/` — pure Python library, **no Reflex dependency**. All simulation logic lives here. Notable subpackages: `neuroclamp/analysis/` (post-hoc metrics: AP metrics, F-I, G-V, I-V, passive properties, burst metrics, etc.) and `neuroclamp/protocols/` (current/voltage protocol builders).
- `neuroclamp_ui/` — Reflex application. State is split across `neuroclamp_ui/state/` (neuron, protocol, simulation, analysis, visibility, log); components are Python functions returning `rx.Component`.

Keep these layers cleanly separated: do not import Reflex types into `neuroclamp/`, and do not import `neuroclamp_ui` modules from `neuroclamp/`.

Any file the library or the UI reads at runtime must live inside its package (`neuroclamp/` or `neuroclamp_ui/`) and be located relative to `__file__`. Paths that reach up to the repository root work in a checkout but break the installed wheel.

## UI framework

The UI uses **Reflex** (v0.8+), a Python framework that compiles to Next.js.

Install UI dependencies and run the dev server:

```bash
uv sync --frozen --extra=ui
uv run reflex run
```

The app serves on `http://localhost:3000`.

The UI dependencies are the `ui` optional extra, not a dependency group,
because the package is published to PyPI as `neuroclamp` (built with hatchling).
Installed users start the app with `neuroclamp ui` (`neuroclamp_ui/cli.py`), which
runs Reflex in a per-user working directory prepared by
`neuroclamp_ui/launcher.py`. The Reflex config lives in
`neuroclamp_ui/_rxconfig.py`; the root `rxconfig.py` re-exports it for
development and is not packaged. `tools/check_dist.py` verifies the built
artifacts.

The root `Dockerfile` builds a production image from an allowlist
`.dockerignore`: any new top-level file or directory the app reads at runtime
must be added to that allowlist, or the image will not include it. The
`Docker` CI workflow builds the image and runs `tools/smoke_test.py` (a
Playwright check that a simulation runs and Export CSV works) against it.

## Package manager

This project uses **uv**. Use `/uv run --frozen` to run commands and `/uv add` to add dependencies. Never use `pip` directly.

## Docstring requirements

Every function and method — public, private, and dunder — must have a Google-style docstring. Ruff enforces public items automatically; private methods (`_foo`) are not checked by ruff but are still required.

- Omit type annotations from `Args:` — they are already in the signature

## Documentation

There are two documentation tracks, plus the docstrings that feed the API reference. Evaluate and update whatever a change touches:

- **Library documentation** — the mkdocs site, for developers using `neuroclamp` in Python. Prose pages live in `docs/` (`index.md`, `presets.md`, `protocols-and-analysis.md`) and the API reference is auto-generated under `docs/api/`. Update the relevant prose page when library behavior or supported options change. Build it to catch broken references: `uv run --frozen --group=docs mkdocs build --strict`. Avoid bare `$` and keep to Markdown that renders under both mkdocs and GitHub.
- **Application documentation** — the in-app `/help` route, a user guide for the web app. Pages live in `neuroclamp_ui/help_content/*.md` and are rendered by `rx.markdown`; add a new topic to `_TOPICS` in `neuroclamp_ui/docs_loader.py` when you add a page. Update these when the UI's controls, panels, or workflow change. Screenshots live in `neuroclamp_ui/help_content/screenshots/` (the app copies them into the generated `assets/screenshots/` at startup, which Reflex serves at `/screenshots/`) and are regenerated with `uv run --frozen --group browser python tools/capture_screenshots.py` (against a running `uv run reflex run --env prod` — production mode avoids the dev hot-reloader restarting the backend mid-capture); refresh them when the UI's appearance changes.
- **Changelog** — `CHANGELOG.md` (Keep a Changelog format) is for users of the library and the app. Add an entry under `## [Unreleased]` for any change to what the library or the app does: a new feature, a change in behavior, a change to the public `neuroclamp` API (always flag breaking changes), or a fix that changes simulation results. Describe what changed for the user, not how the code changed. Leave out refactors, tests, CI, tooling, and dependency bumps; the PR history covers those.
- **Docstrings** — Google-style docstrings on public symbols feed the mkdocstrings API reference. When you add or change a public symbol in `neuroclamp/`, update its docstring and add a `:::` entry under `docs/api/` if it is a new top-level export.

## Testing conventions

Tests are split into five buckets under `tests/`:

- `tests/unit/` — fast, pure-function tests; no simulation runs. Run alone during development: `uv run --frozen -m pytest tests/unit`
- `tests/integration/` — end-to-end protocol → simulation pipeline tests. Simulation-calling tests extracted from mixed files are named `*_simulation.py`.
- `tests/ui/` — Reflex and plotting layer tests; skipped when `reflex` is not installed.
- `tests/e2e/` — headless full-pipeline tests that drive Reflex state handlers directly (no dev server, no browser). **Not in default `testpaths`** — run explicitly with `uv run --frozen -m pytest tests/e2e`.
- `tests/parity/` — golden-trace parity tests for behavior-preserving changes. **Not in default `testpaths`** — run explicitly with `uv run --frozen -m pytest tests/parity` (see below).

The shared `hh_model` fixture lives in `tests/conftest.py` and is discoverable by every test subdirectory. `tests/e2e/conftest.py` adds e2e-specific fixtures.

Pytest is configured with `-n auto --dist=loadscope` (parallel via xdist, tests in the same module pinned to the same worker).

`tests/parity/` compares every neuron preset × protocol preset against a stored baseline in `tests/parity/golden/`. The baseline records what `main` produced, not what is correct, so intentional tuning of presets or kinetics is expected to make it stale and does not need it updated. Use it for changes that should not alter results (integrator refactors, rate functions expressed as data, alternative backends): refresh the baseline on `main` with `uv run --frozen python -m tests.parity._golden`, then run `uv run --frozen -m pytest tests/parity` on the work branch. During such work a failure is a regression; do not loosen the tolerances to make it pass.

Additional conventions:
- Use plain functions, not test classes
- Use `pytest.approx` for float comparisons
- Every test function needs a Google-style docstring

## Type checking

ty enforces types in `neuroclamp/` (core library) and UI code (`neuroclamp_ui/`). Reflex, Plotly, and `neuroclamp_ui` imports are replaced with `Any` for UI files and selected test files — see the `[[tool.ty.overrides]]` section in `pyproject.toml`. Do not spend time fixing ty errors in UI code.

## Releasing

The package is published to PyPI as `neuroclamp` by `.github/workflows/release.yml`,
using Trusted Publishing (no API tokens). The version is static in
`pyproject.toml`.

1. Bump the version: `uv version --bump patch` (or `minor`/`major`).
2. In `CHANGELOG.md`, move the `## [Unreleased]` entries into the version's
   section and date it as `## [X.Y.Z] - YYYY-MM-DD` (create the section, or
   date an existing pending one such as `## [0.1.0] - Unreleased`).
3. Merge to `main`, then tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   The workflow fails if the tag does not match the project version.
4. The workflow builds, checks the artifacts (`tools/check_dist.py`), smoke-tests
   the wheel with and without the `ui` extra, publishes to TestPyPI, and then to
   PyPI (the `pypi` environment may require approval).

To dry-run against TestPyPI without a tag, run the workflow manually
(`workflow_dispatch`). To check artifacts locally: `uv build` then
`uv run --no-project python tools/check_dist.py dist`.

## GitHub CLI conventions

`gh issue view <number>` triggers a GraphQL deprecation error because it fetches `projectCards`. Always request explicit fields instead:

```bash
gh issue view <number> --json number,title,body,state,labels,assignees,comments
```

## Working on GitHub issues

When addressing a GitHub issue, create a dedicated branch off `main` before writing any code:

```bash
git checkout main
git pull
git checkout -b <branch-name>
```

Name the branch descriptively (e.g. `issue-42-voltage-clamp-fix`). Do all work for that issue on that branch. Do not commit issue-related changes directly to `main`.

## Committing during multi-step plans

When executing a plan that has discrete numbered steps, commit after each step
rather than staging everything at the end. Each commit should cover exactly one
logical change (one fix, one refactor, one feature addition) so that the git
log reflects the plan's structure.

If a step touches several files but they all belong to the same logical change,
group them in a single commit. If a file contains changes from different plan
steps, stage only the relevant hunks for each commit.

Run the per-commit checks listed below before every commit — not just at the end.

## Checks

Checks are tiered: a fast set before every commit, and the full suite before
a PR is opened. All must pass with no errors.

**Before every commit:** `/ruff check .`, `/ruff format --check .`, `/ty check`,
and `/uv run --frozen -m pytest tests/unit`.
If the commit touches `neuroclamp_ui/`, also run `tests/ui` and `tests/e2e`.
If the commit touches `docs/` or public docstrings, also run
`uv run --frozen --group=docs mkdocs build --strict`.

**Before opening a PR:** the default suite (`/uv run --frozen -m pytest`,
which covers `tests/unit`, `tests/integration` and `tests/ui`) plus
`/uv run --frozen -m pytest tests/e2e`, which is outside the default
`testpaths`. `tests/integration` runs in CI only on a cron job, so this
local run is its only pre-merge check. If the branch touches `docs/` or
public docstrings, also run the strict docs build. If the branch refactors
simulation code without meaning to change results, also run
`uv run --frozen -m pytest tests/parity` against a baseline refreshed on `main`.

**Before finishing a task that does not end in a PR:** run the pre-PR checks.
