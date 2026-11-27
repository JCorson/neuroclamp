"""Current- and voltage-clamp simulation on the compiled Rust core.

:func:`simulate_current_clamp_rust` and :func:`simulate_voltage_clamp_rust`
are what the library's simulation entry points run with ``backend="rust"``.
Each lowers the neuron to a model spec (:func:`neuroclamp._spec.lower`),
integrates it in ``neuroclamp._core``, and returns the same structured array
the Python backend does, with the same fields in the same order.

The Rust integrators follow the Python integrators' order of floating-point
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
    from . import _core
    from .clamp_simulations import SimulationResult
    from .neuron import Neuron

# The same expression as the Python backend's time step, so both use the
# identical dt and time axis.
_DT_MS = 1.0 / SIM_SAMPLING_FREQ * 1000.0


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
    gates0 = _initial_gates(neuron, initial_gating_state)
    columns = _compile(neuron).simulate_current_clamp(
        np.ascontiguousarray(current_external, dtype=np.float64),
        float(initial_V),
        gates0,
        float(initial_ca_i),
        _DT_MS,
    )
    channels = [ch.current_name for ch in neuron.channels]
    order = ["voltage", *channels, "Itotal"]
    return _to_structured(neuron, columns, order)


def simulate_voltage_clamp_rust(
    neuron: Neuron,
    voltage_protocol: np.ndarray,
    initial_gating_state: dict[str, float],
    initial_ca_i: float,
) -> SimulationResult:
    """Run a voltage-clamp simulation on the Rust core.

    Args:
        neuron: The neuron to simulate.  Every gate must be built from the
            rate expressions in :mod:`neuroclamp.rates`.
        voltage_protocol: Prescribed voltage per time step in mV; already
            checked to be non-empty and finite.
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
    gates0 = _initial_gates(neuron, initial_gating_state)
    columns = _compile(neuron).simulate_voltage_clamp(
        np.ascontiguousarray(voltage_protocol, dtype=np.float64),
        gates0,
        float(initial_ca_i),
        _DT_MS,
    )
    channels = [ch.current_name for ch in neuron.channels]
    order = ["voltage", "Itotal", *channels]
    return _to_structured(neuron, columns, order)


def _compile(neuron: Neuron) -> _core.Model:
    """Lower a neuron and build the Rust model from its spec.

    Args:
        neuron: The neuron to compile.

    Returns:
        The compiled model.
    """
    from . import _core

    return _core.Model(json.dumps(lower(neuron)))


def _initial_gates(neuron: Neuron, gating_state: dict[str, float]) -> list[float]:
    """Order a gating-state dict as the spec's gate list.

    Args:
        neuron: The neuron, whose gates fix the order.
        gating_state: Gate values keyed by gate name.

    Returns:
        The gate values in ``neuron.all_gating_variables`` order.

    Raises:
        ValueError: If the keys are not exactly the neuron's gate names.
    """
    gates = neuron.all_gating_variables
    expected = sorted(g.name for g in gates)
    if sorted(gating_state) != expected:
        raise ValueError(
            "gating_state keys do not match the neuron's gating variables: "
            f"got {sorted(gating_state)}, expected {expected}"
        )
    return [float(gating_state[g.name]) for g in gates]


def _to_structured(
    neuron: Neuron, columns: np.ndarray, leading: list[str]
) -> SimulationResult:
    """Wrap the Rust core's column block in the library's structured array.

    The Rust core always returns the columns ``voltage``, one current per
    channel, ``Itotal``, one value per gate, and ``ca_i`` with Ca²⁺ dynamics.
    The result lists ``time`` first, then ``leading`` (the clamp mode's order
    of voltage, total current and channel currents), then the gates and
    ``ca_i``, matching the Python backend's fields.

    Args:
        neuron: The simulated neuron, which fixes the column names.
        columns: ``(n_steps, n_columns)`` output of the Rust core.
        leading: Field order for the voltage, ``Itotal`` and channel columns.

    Returns:
        The structured result.

    Raises:
        RuntimeError: If the Rust core returned an unexpected number of columns.
    """
    trailing = [g.name for g in neuron.all_gating_variables]
    if neuron.calcium_dynamics is not None:
        trailing.append("ca_i")
    rust_order = ["voltage"]
    rust_order += [ch.current_name for ch in neuron.channels]
    rust_order += ["Itotal", *trailing]
    if columns.shape[1] != len(rust_order):
        raise RuntimeError(
            f"Rust core returned {columns.shape[1]} columns, expected {len(rust_order)}"
        )
    index = {name: i for i, name in enumerate(rust_order)}
    names = [*leading, *trailing]
    n_steps = columns.shape[0]
    dtype = np.dtype([("time", np.float64)] + [(name, np.float64) for name in names])
    result = np.empty(n_steps, dtype=dtype)
    result["time"] = np.linspace(0.0, (n_steps - 1) * _DT_MS, n_steps)
    for name in names:
        result[name] = columns[:, index[name]]
    return cast("SimulationResult", result)
