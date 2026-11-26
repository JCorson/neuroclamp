"""Parity tests: behavior-preserving changes must reproduce the golden baseline.

Each test re-simulates one stored run (neuron preset × protocol preset ×
sweep) and compares it with the baseline in ``tests/parity/golden/`` using a
timing-aware tolerance band. See :mod:`tests.parity._golden` for coverage,
storage, tolerance, and workflow details.

These tests are opt-in (outside the default ``testpaths``) because intentional
tuning of presets is expected to change the output. Run them for refactors
that should not change results::

    uv run --frozen -m pytest tests/parity

A failure means the output differs from the baseline. If the baseline is
simply out of date with ``main``, refresh it there first::

    uv run --frozen python -m tests.parity._golden
"""

from __future__ import annotations

import json

import pytest

from tests.parity._golden import (
    GOLDEN_DIR,
    GoldenRun,
    compare,
    golden_file,
    golden_runs,
    simulate_run,
)

_REGENERATE = "uv run --frozen python -m tests.parity._golden"
RUNS = golden_runs()


def test_manifest_matches_current_presets() -> None:
    """The stored runs match the current neuron and protocol presets.

    A new or renamed preset, or a protocol whose sweep count changed, means
    the baseline predates the current presets and must be refreshed before
    the per-run tests are meaningful.
    """
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    stored = [GoldenRun(**run) for run in manifest["runs"]]
    assert stored == RUNS, f"Baseline is out of date; refresh it: {_REGENERATE}"


@pytest.mark.parametrize("run", RUNS, ids=[run.test_id for run in RUNS])
def test_matches_golden_trace(run: GoldenRun) -> None:
    """A run's voltage, total current, and Ca²⁺ match the stored baseline.

    Args:
        run: The neuron preset, protocol preset, and sweep to simulate.
    """
    if not golden_file(run.neuron).exists():
        pytest.fail(f"No baseline for {run.neuron!r}; generate it: {_REGENERATE}")
    failures = compare(run, simulate_run(run))
    assert not failures, (
        f"{run.test_id} differs from the baseline:\n  "
        + "\n  ".join(failures)
        + "\nIf this branch should preserve behavior, this is a regression. If"
        + f" the baseline predates main, refresh it on main: {_REGENERATE}"
    )
