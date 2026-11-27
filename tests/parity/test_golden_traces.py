"""Parity tests: behavior-preserving changes must reproduce the golden baseline.

Each test re-simulates one stored run (neuron preset × protocol preset ×
sweep) and compares it with the baseline in ``tests/parity/golden/`` using a
timing-aware tolerance band. See :mod:`tests.parity._golden` for coverage,
storage, tolerance, and workflow details.

Every run is also checked on the Rust backend against the same baseline.

These tests are opt-in (outside the default ``testpaths``) because intentional
tuning of presets is expected to change the output. Run them for refactors
that should not change results::

    uv run --frozen -m pytest tests/parity

A failure means the output differs from the baseline. If the baseline is
simply out of date with ``main``, regenerate it from ``main``'s code (see the
workflow in :mod:`tests.parity._golden`).
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
    manifest_settings,
    simulate_run,
)

_REGENERATE = (
    "regenerate it from main's code (workflow in tests/parity/_golden.py):"
    " uv run --frozen python -m tests.parity._golden --out <this checkout's"
    " tests/parity/golden>"
)
RUNS = golden_runs()


def _manifest() -> dict:
    """Read the stored manifest, failing with the regenerate hint if absent.

    Returns:
        The parsed ``manifest.json``.
    """
    path = GOLDEN_DIR / "manifest.json"
    if not path.exists():
        pytest.fail(f"No baseline manifest at {path}; {_REGENERATE}")
    return json.loads(path.read_text())


def test_manifest_matches_current_presets() -> None:
    """The stored runs match the current neuron and protocol presets.

    A new or renamed preset, or a protocol whose sweep count changed, means
    the baseline predates the current presets and must be refreshed before
    the per-run tests are meaningful.
    """
    stored = [GoldenRun(**run) for run in _manifest()["runs"]]
    assert stored == RUNS, f"Baseline is out of date; {_REGENERATE}"


def test_manifest_matches_tolerance_settings() -> None:
    """The stored bands were built with the current stride and tolerances.

    The comparison reads the bands from the baseline, so changing a tolerance
    constant has no effect until the baseline is regenerated, and editing the
    stored bands by hand would loosen the check without touching the code.
    Either shows up here as a mismatch.
    """
    manifest = _manifest()
    recorded = {key: manifest.get(key) for key in manifest_settings()}
    assert recorded == manifest_settings(), (
        f"Baseline tolerances differ from the code; {_REGENERATE}"
    )


@pytest.mark.parametrize("run", RUNS, ids=[run.test_id for run in RUNS])
def test_matches_golden_trace(run: GoldenRun) -> None:
    """A run's voltage, total current, and Ca²⁺ match the stored baseline.

    Args:
        run: The neuron preset, protocol preset, and sweep to simulate.
    """
    if not golden_file(run.neuron).exists():
        pytest.fail(f"No baseline for {run.neuron!r}; {_REGENERATE}")
    failures = compare(run, simulate_run(run))
    assert not failures, (
        f"{run.test_id} differs from the baseline:\n  "
        + "\n  ".join(failures)
        + "\nIf this branch should preserve behavior, this is a regression. If"
        + f" the baseline predates main, {_REGENERATE}"
    )


@pytest.mark.parametrize("run", RUNS, ids=[run.test_id for run in RUNS])
def test_rust_backend_matches_golden_trace(run: GoldenRun) -> None:
    """A run on the Rust backend matches the stored baseline.

    The baseline comes from the Python backend, so this checks the two
    backends against each other with the same tolerance band.

    Args:
        run: The neuron preset, protocol preset, and sweep to simulate.
    """
    if not golden_file(run.neuron).exists():
        pytest.fail(f"No baseline for {run.neuron!r}; generate it: {_REGENERATE}")
    failures = compare(run, simulate_run(run, backend="rust"))
    assert not failures, (
        f"{run.test_id} on the Rust backend differs from the baseline:\n  "
        + "\n  ".join(failures)
    )
