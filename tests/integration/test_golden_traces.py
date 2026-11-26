"""Golden-trace regression tests: the library's numerical output must not drift.

Each test re-simulates one stored run (neuron preset × protocol preset ×
sweep) and compares it with the reference in ``tests/integration/golden/``
using a timing-aware tolerance band. See :mod:`tests.integration._golden` for
coverage, storage, and tolerance details.

A failure means simulation results changed. If that is intended (for example,
a deliberate fix to a preset's kinetics), regenerate the references and add a
``CHANGELOG.md`` entry describing the change::

    uv run --frozen python -m tests.integration._golden
"""

from __future__ import annotations

import json

import pytest

from tests.integration._golden import (
    GOLDEN_DIR,
    GoldenRun,
    compare,
    golden_file,
    golden_runs,
    simulate_run,
)

_REGENERATE = "uv run --frozen python -m tests.integration._golden"
RUNS = golden_runs()


def test_manifest_matches_current_presets() -> None:
    """The stored runs match the current neuron and protocol presets.

    A new or renamed preset, or a protocol whose sweep count changed, needs
    its golden traces regenerated before the per-run tests are meaningful.
    """
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    stored = [GoldenRun(**run) for run in manifest["runs"]]
    assert stored == RUNS, f"Golden runs are out of date; run: {_REGENERATE}"


@pytest.mark.parametrize("run", RUNS, ids=[run.test_id for run in RUNS])
def test_matches_golden_trace(run: GoldenRun) -> None:
    """A run's voltage, total current, and Ca²⁺ match the stored reference.

    Args:
        run: The neuron preset, protocol preset, and sweep to simulate.
    """
    if not golden_file(run.neuron).exists():
        pytest.fail(f"No golden traces for {run.neuron!r}; run: {_REGENERATE}")
    failures = compare(run, simulate_run(run))
    assert not failures, (
        f"{run.test_id} differs from its golden trace:\n  "
        + "\n  ".join(failures)
        + f"\nIf this change is intended, run: {_REGENERATE}"
    )
