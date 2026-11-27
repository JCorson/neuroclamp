"""Current-clamp simulation on the compiled Rust core.

:func:`simulate_current_clamp_rust` is what
:func:`~neuroclamp.simulate_current_clamp` and
:func:`~neuroclamp.simulate_current_clamp_from_state` run with
``backend="rust"``.  It lowers the neuron to a model spec
(:func:`neuroclamp._spec.lower`), integrates it in ``neuroclamp._core``, and
returns the same structured array the Python backend does.

The Rust integrator follows the Python integrator's order of floating-point
operations, so the two agree to within rounding.  The remaining differences
come from integer powers above 2 (gate powers and Hill coefficients), which
Rust computes by repeated multiplication instead of C ``pow``; they are of
order 1e-12 relative.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np

from ._spec import lower
from .constants import SIM_SAMPLING_FREQ

if TYPE_CHECKING:
    from .clamp_simulations import SimulationResult
    from .neuron import Neuron


def simulate_current_clamp_rust(
    neuron: Neuron,
    current_external: np.ndarray,
    initial_V: float,
    initial_gating_state: dict[str, float],
    initial_ca_i: float,
) -> SimulationResult:
    """Run a current-clamp simulation on the Rust core.

    Args:
        neuron: The neuron to simulate.  Every gate must be built from the
            rate expressions in :mod:`neuroclamp.rates`.
        current_external: Injected current per time step in µA/cm²; already
            checked to be non-empty and finite.
        initial_V: Initial membrane voltage in mV.
        initial_gating_state: Initial gate values keyed by gate name.
        initial_ca_i: Initial intracellular Ca²⁺ in mM; held constant when
            the neuron has no Ca²⁺ dynamics.

    Returns:
        Structured array with the same fields as the Python backend's.

    Raises:
        neuroclamp._spec.UnloweredRateError: If a gate's rate is not built
            from the standard expression forms.
        ValueError: If ``initial_gating_state`` does not have exactly the
            neuron's gate names.
    """
    from . import _core

    gates = neuron.all_gating_variables
    expected = sorted(g.name for g in gates)
    if sorted(initial_gating_state) != expected:
        raise ValueError(
            "gating_state keys do not match the neuron's gating variables: "
            f"got {sorted(initial_gating_state)}, expected {expected}"
        )
    model = _core.Model(json.dumps(lower(neuron)))
    # The same expression as the Python backend's time step, so both use the
    # identical dt and time axis.
    dt_ms = 1.0 / SIM_SAMPLING_FREQ * 1000.0
    columns = model.simulate_current_clamp(
        np.ascontiguousarray(current_external, dtype=np.float64),
        float(initial_V),
        [float(initial_gating_state[g.name]) for g in gates],
        float(initial_ca_i),
        dt_ms,
    )
    return _to_structured(neuron, columns, dt_ms)


def _to_structured(
    neuron: Neuron, columns: np.ndarray, dt_ms: float
) -> SimulationResult:
    """Wrap the Rust core's column block in the library's structured array.

    Args:
        neuron: The simulated neuron, which fixes the column order.
        columns: ``(n_steps, n_columns)`` output of the Rust core.
        dt_ms: Time step in ms.

    Returns:
        The structured result, with a ``"time"`` field prepended.
    """
    names = ["voltage"]
    names += [ch.current_name for ch in neuron.channels]
    names.append("Itotal")
    names += [g.name for g in neuron.all_gating_variables]
    if neuron.calcium_dynamics is not None:
        names.append("ca_i")
    if columns.shape[1] != len(names):
        raise RuntimeError(
            f"Rust core returned {columns.shape[1]} columns, expected {len(names)}"
        )
    n_steps = columns.shape[0]
    dtype = np.dtype([("time", np.float64)] + [(name, np.float64) for name in names])
    result = np.empty(n_steps, dtype=dtype)
    result["time"] = np.linspace(0.0, (n_steps - 1) * dt_ms, n_steps)
    for index, name in enumerate(names):
        result[name] = columns[:, index]
    return cast("SimulationResult", result)
