"""Compare the Rust current-clamp backend against the Python library.

Checks every field of the structured result (voltage, each channel current,
Itotal, each gate, ca_i) for every preset, then times both backends.
"""

from __future__ import annotations

import time

import numpy as np
from backend import simulate_current_clamp
from verify_sims import I_STEP

import patch_sim as ps

SQUID = "Squid Giant Axon (Classic HH)"


def _compare(neuron: ps.Neuron, current: np.ndarray) -> tuple[float, str, float]:
    """Run both backends and find the largest per-field deviation.

    Args:
        neuron: Neuron to simulate.
        current: Injected current per time step in µA/cm².

    Returns:
        ``(max_abs_diff, worst_field, max_abs_voltage_diff)``.

    Raises:
        AssertionError: If the two results have different fields.
    """
    py = simulate_current_clamp(neuron, current, backend="python")
    rs = simulate_current_clamp(neuron, current, backend="rust")
    assert py.dtype.names == rs.dtype.names, (py.dtype.names, rs.dtype.names)
    worst, worst_field = 0.0, ""
    for name in py.dtype.names:
        diff = float(np.max(np.abs(py[name] - rs[name])))
        if diff > worst:
            worst, worst_field = diff, name
    dv = float(np.max(np.abs(py["voltage"] - rs["voltage"])))
    return worst, worst_field, dv


def _best_of(neuron: ps.Neuron, current: np.ndarray, backend: str) -> float:
    """Return the fastest of three runs in seconds.

    Args:
        neuron: Neuron to simulate.
        current: Injected current per time step in µA/cm².
        backend: ``"python"`` or ``"rust"``.

    Returns:
        Minimum wall time in seconds.
    """
    times = []
    for _ in range(3):
        start = time.perf_counter()
        simulate_current_clamp(neuron, current, backend=backend)  # type: ignore[arg-type]
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    """Print per-preset agreement and a quick timing comparison."""
    print("Agreement, 150 ms current step (max |Python - Rust| over all fields):")
    for key, factory in ps.NEURON_PRESETS.items():
        worst, field, dv = _compare(factory(), I_STEP)
        label = " <- squid (target preset)" if key == SQUID else ""
        print(
            f"  {key:36s} max|dV| {dv:.1e} mV, worst field {field!r} {worst:.1e}{label}"
        )

    print("\nTiming, 150 ms current step (best of 3, includes lowering and wrapping):")
    for key in [SQUID, "Purkinje Neuron"]:
        neuron = ps.NEURON_PRESETS[key]()
        simulate_current_clamp(neuron, I_STEP, backend="rust")  # warm the model cache
        t_py = _best_of(neuron, I_STEP, "python")
        t_rs = _best_of(neuron, I_STEP, "rust")
        print(f"  {key:36s} python {t_py * 1e3:7.1f} ms  rust {t_rs * 1e3:6.2f} ms")


if __name__ == "__main__":
    main()
