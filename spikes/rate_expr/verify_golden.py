"""Check alternative implementations against the library's golden traces.

Uses the same references and tolerance bands as
``tests/parity/test_golden_traces.py``:

- ``tree``: expression-tree rates on the Python backend, every run (a stand-in
  for the rates-as-data migration).
- ``rust``: the compiled ``neuroclamp_core`` backend, current-clamp runs only
  (voltage clamp is not ported). Skipped if the extension is not built.

Reports pass/fail counts and the largest fraction of the band used.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from importlib.util import find_spec

from backend import simulate_current_clamp
from translate import to_tree_neuron

import neuroclamp as nc
from neuroclamp.constants import CURRENT_CLAMP
from tests.parity._golden import (
    GoldenRun,
    _protocol,
    compare,
    golden_runs,
    max_excess,
)

HAVE_RUST = find_spec("neuroclamp_core") is not None


def _check(run: GoldenRun) -> dict[str, tuple[bool, float]]:
    """Run one golden case through each available alternative backend.

    Args:
        run: The golden run to check.

    Returns:
        ``{backend: (passed, max_fraction_of_band)}`` for each backend run.
    """
    neuron = nc.NEURON_PRESETS[run.neuron]()
    stimulus = _protocol(run.neuron, run.protocol)[run.sweep]
    is_cc = run.clamp_mode == CURRENT_CLAMP
    simulate = nc.simulate_current_clamp if is_cc else nc.simulate_voltage_clamp

    results = {"tree": simulate(to_tree_neuron(neuron), stimulus)}
    if HAVE_RUST and is_cc:
        results["rust"] = simulate_current_clamp(neuron, stimulus, backend="rust")
    return {
        name: (not compare(run, result), max_excess(run, result))
        for name, result in results.items()
    }


def main() -> None:
    """Print a pass/fail summary per backend."""
    runs = golden_runs()
    with ProcessPoolExecutor() as pool:
        outcomes = list(pool.map(_check, runs))
    if not HAVE_RUST:
        print("neuroclamp_core is not built; skipping the Rust backend.")
    for backend in ("tree", "rust"):
        rows = [
            (run, out[backend]) for run, out in zip(runs, outcomes) if backend in out
        ]
        if not rows:
            continue
        passed = sum(ok for _, (ok, _) in rows)
        worst_run, (_, worst) = max(rows, key=lambda row: row[1][1])
        print(
            f"{backend:5s} {passed}/{len(rows)} runs within tolerance; "
            f"worst uses {worst:.1%} of the band ({worst_run.test_id})"
        )
        for run, (ok, _) in rows:
            if not ok:
                print(f"  FAIL {run.test_id}")


if __name__ == "__main__":
    main()
