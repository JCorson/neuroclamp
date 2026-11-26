"""Time whole protocol families: Python, the app's process pool, and Rust.

For each preset × current-clamp protocol family (sweeps built with the
neuron's protocol adjustments, as in the app), reports the best-of-3 wall
time for:

- ``py serial``: ``neuroclamp.simulate_current_clamp`` per sweep.
- ``py pool``: ``neuroclamp.simulate_batch`` (what the app runs today),
  including process-pool startup.
- ``rust serial``: the compiled backend, one sweep at a time.
- ``rust batch``: the compiled backend, all sweeps in parallel threads.

Also checks that the batched Rust results equal the sequential ones exactly
and stay close to Python. Needs the extension built.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from backend import simulate_current_clamp, simulate_current_clamp_batch

import neuroclamp as nc

CASES = [
    ("Squid Giant Axon (Classic HH)", "F-I Curve"),
    ("Purkinje Neuron", "F-I Curve"),
    ("Hippocampal CA1 Pyramidal Neuron", "F-I Curve"),
    ("Purkinje Neuron", "Hyperpolarization Steps"),
    ("Hippocampal CA1 Pyramidal Neuron", "Hyperpolarization Steps"),
]


def _best_of(fn: Callable[[], Any], repeats: int = 3) -> float:
    """Return the fastest wall time of several calls, in seconds.

    Args:
        fn: Zero-argument function to time.
        repeats: Number of calls.

    Returns:
        Minimum elapsed time in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    """Print the timing table and the agreement checks."""
    print(f"{os.cpu_count()} CPU cores; times are best of 3 wall-clock runs\n")
    header = (
        f"{'preset / protocol':48s} {'sweeps':>6s} {'py serial':>10s} "
        f"{'py pool':>9s} {'rust serial':>11s} {'rust batch':>10s} {'vs pool':>8s}"
    )
    print(header)
    for neuron_key, protocol in CASES:
        neuron = nc.NEURON_PRESETS[neuron_key]()
        sweeps = nc.build_protocol_from_preset(protocol, neuron_key)

        rust_batch = simulate_current_clamp_batch(neuron, sweeps, backend="rust")
        rust_serial = [
            simulate_current_clamp(neuron, s, backend="rust") for s in sweeps
        ]
        python = [nc.simulate_current_clamp(neuron, s) for s in sweeps]
        assert all(
            np.array_equal(a, b) for a, b in zip(rust_batch, rust_serial, strict=True)
        ), "batched Rust differs from sequential Rust"
        max_dv = max(
            float(np.max(np.abs(a["voltage"] - b["voltage"])))
            for a, b in zip(rust_batch, python, strict=True)
        )

        t_py = _best_of(lambda: [nc.simulate_current_clamp(neuron, s) for s in sweeps])
        t_pool = _best_of(
            lambda: simulate_current_clamp_batch(neuron, sweeps, backend="python")
        )
        t_rs = _best_of(
            lambda: [simulate_current_clamp(neuron, s, backend="rust") for s in sweeps]
        )
        t_rb = _best_of(
            lambda: simulate_current_clamp_batch(neuron, sweeps, backend="rust")
        )
        label = f"{neuron_key.split(' (')[0]} / {protocol}"
        print(
            f"{label:48s} {len(sweeps):6d} {t_py:9.2f}s {t_pool:8.2f}s "
            f"{t_rs * 1e3:9.1f}ms {t_rb * 1e3:8.1f}ms {t_pool / t_rb:7.0f}x"
            f"   (max |dV| vs Python {max_dv:.1e} mV)"
        )


if __name__ == "__main__":
    main()
