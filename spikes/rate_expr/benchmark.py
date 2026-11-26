"""Time today's rates vs. expression trees vs. codegen on the Python backend.

The voltage-clamp optimizer only tabulates rates that are instances of
``VoltageOnlyRate``. This script patches that check to also accept any rate
whose ``uses_calcium`` is ``False``, which is the change the design on #410
proposes. Without it, tree gates skip tabulation and voltage clamp slows down.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from codegen import Compiled
from translate import to_tree_neuron
from verify_sims import I_STEP, V_STEP

import patch_sim as ps
import patch_sim.clamp_simulations as cs
from patch_sim.rates import VoltageOnlyRate

PRESETS = [
    "Squid Giant Axon (Classic HH)",
    "Purkinje Neuron",
    "Hippocampal CA1 Pyramidal Neuron",
]


class _TabulableMeta(type):
    """Metaclass whose isinstance check accepts any calcium-free rate."""

    def __instancecheck__(cls, obj: object) -> bool:
        """Accept ``VoltageOnlyRate`` instances and calcium-free tree rates.

        Args:
            obj: The object being checked.

        Returns:
            Whether the rate can be tabulated over a voltage protocol.
        """
        return isinstance(obj, VoltageOnlyRate) or (
            getattr(obj, "uses_calcium", True) is False
        )


class _Tabulable(metaclass=_TabulableMeta):
    """Stand-in for ``VoltageOnlyRate`` in the tabulation check."""


def _best_of(fn: Callable[..., Any], *args: Any, repeats: int = 3) -> float:
    """Return the fastest wall time over several runs.

    Args:
        fn: Function to time.
        *args: Positional arguments for ``fn``.
        repeats: Number of runs.

    Returns:
        Minimum elapsed time in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    """Print current- and voltage-clamp timings for three backends."""
    cs.VoltageOnlyRate = _Tabulable  # type: ignore[misc]
    print(
        f"{'preset':34s} {'CC today/tree/codegen (s)':>27s}  VC today/tree/codegen (s)"
    )
    for key in PRESETS:
        neuron = ps.NEURON_PRESETS[key]()
        variants = [neuron, to_tree_neuron(neuron), to_tree_neuron(neuron, Compiled)]
        cc = [_best_of(ps.simulate_current_clamp, n, I_STEP) for n in variants]
        vc = [_best_of(ps.simulate_voltage_clamp, n, V_STEP) for n in variants]
        a = ps.simulate_current_clamp(variants[0], I_STEP)["voltage"]
        b = ps.simulate_current_clamp(variants[2], I_STEP)["voltage"]
        print(
            f"{key:34s} {' / '.join(f'{t:.3f}' for t in cc):>27s}  "
            f"{' / '.join(f'{t:.3f}' for t in vc)}   "
            f"codegen max|dV| {np.max(np.abs(a - b)):.1e} mV"
        )


if __name__ == "__main__":
    main()
