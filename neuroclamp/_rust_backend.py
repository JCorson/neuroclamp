"""Current- and voltage-clamp simulation on the compiled Rust core.

:func:`simulate_current_clamp_rust` and :func:`simulate_voltage_clamp_rust`
are what the library's simulation entry points run with ``backend="rust"``,
and their ``*_batch_rust`` counterparts are what
:func:`~neuroclamp.simulate_batch` runs, with the sweeps in parallel on the
core's thread pool.  Each lowers the neuron to a model spec
(:func:`neuroclamp._spec.lower`), integrates it in ``neuroclamp._core``, and
returns the same structured arrays the Python backend does, with the same
fields in the same order.

The Rust integrators follow the Python integrators' order of floating-point
operations, so the two agree to within rounding.  The remaining differences
come from integer powers above 2 (gate powers and Hill coefficients), which
Rust computes by repeated multiplication instead of C ``pow``; they are of
order 1e-12 relative.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
        ValueError: If gate names repeat across channels, or if
            ``initial_gating_state`` does not have exactly the neuron's gate
            names.
    """
    model, (gates0,) = _prepare(neuron, [initial_gating_state])
    columns = model.simulate_current_clamp(
        _as_float_array(current_external),
        float(initial_V),
        gates0,
        float(initial_ca_i),
        _DT_MS,
    )
    return _to_structured(neuron, columns, _current_clamp_order(neuron))


def simulate_current_clamp_batch_rust(
    neuron: Neuron,
    protocols: Sequence[np.ndarray],
    initial_V: float,
    initial_gating_state: dict[str, float],
    initial_ca_i: float,
) -> list[SimulationResult]:
    """Run current-clamp sweeps in parallel on the Rust core.

    Every sweep starts from the same state, as every fresh current-clamp run
    does.  The sweeps run on a thread pool inside one call, so the neuron is
    lowered and compiled once.

    Args:
        neuron: The neuron to simulate.  Every gate must be built from the
            rate expressions in :mod:`neuroclamp.rates`.
        protocols: Injected current per time step in µA/cm², one array per
            sweep; already checked to be non-empty and finite.  Sweeps may
            differ in length.
        initial_V: Initial membrane voltage in mV.
        initial_gating_state: Initial gate values keyed by gate name.
        initial_ca_i: Initial intracellular Ca²⁺ in mM.

    Returns:
        One structured array per sweep, in protocol order, each identical to
        :func:`simulate_current_clamp_rust` on that protocol.

    Raises:
        neuroclamp._spec.UnloweredRateError: If a gate's rate is not built
            from the standard expression forms.
        ValueError: If gate names repeat across channels, or if
            ``initial_gating_state`` does not have exactly the neuron's gate
            names.
    """
    model, (gates0,) = _prepare(neuron, [initial_gating_state])
    blocks = model.simulate_current_clamp_batch(
        [_as_float_array(p) for p in protocols],
        float(initial_V),
        gates0,
        float(initial_ca_i),
        _DT_MS,
    )
    order = _current_clamp_order(neuron)
    return [_to_structured(neuron, columns, order) for columns in blocks]


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
        ValueError: If gate names repeat across channels, or if
            ``initial_gating_state`` does not have exactly the neuron's gate
            names.
    """
    model, (gates0,) = _prepare(neuron, [initial_gating_state])
    columns = model.simulate_voltage_clamp(
        _as_float_array(voltage_protocol),
        gates0,
        float(initial_ca_i),
        _DT_MS,
    )
    return _to_structured(neuron, columns, _voltage_clamp_order(neuron))


def simulate_voltage_clamp_batch_rust(
    neuron: Neuron,
    protocols: Sequence[np.ndarray],
    initial_gating_states: Sequence[dict[str, float]],
    initial_ca_i: float,
) -> list[SimulationResult]:
    """Run voltage-clamp sweeps in parallel on the Rust core.

    Each sweep has its own initial gate values, because a fresh voltage-clamp
    run starts at steady state at its protocol's first voltage.  The sweeps
    run on a thread pool inside one call, so the neuron is lowered and
    compiled once.

    Args:
        neuron: The neuron to simulate.  Every gate must be built from the
            rate expressions in :mod:`neuroclamp.rates`.
        protocols: Prescribed voltage per time step in mV, one array per
            sweep; already checked to be non-empty and finite.  Sweeps may
            differ in length.
        initial_gating_states: Initial gate values keyed by gate name, one
            dict per sweep.
        initial_ca_i: Initial intracellular Ca²⁺ in mM, shared by every sweep.

    Returns:
        One structured array per sweep, in protocol order, each identical to
        :func:`simulate_voltage_clamp_rust` on that protocol.

    Raises:
        neuroclamp._spec.UnloweredRateError: If a gate's rate is not built
            from the standard expression forms.
        ValueError: If gate names repeat across channels, or if a gating
            state does not have exactly the neuron's gate names.
    """
    model, gates0 = _prepare(neuron, initial_gating_states)
    blocks = model.simulate_voltage_clamp_batch(
        [_as_float_array(p) for p in protocols],
        gates0,
        [float(initial_ca_i)] * len(protocols),
        _DT_MS,
    )
    order = _voltage_clamp_order(neuron)
    return [_to_structured(neuron, columns, order) for columns in blocks]


def _prepare(
    neuron: Neuron, gating_states: Sequence[dict[str, float]]
) -> tuple[_core.Model, list[list[float]]]:
    """Compile a neuron and order initial gate values for the Rust core.

    The neuron is lowered first: lowering rejects duplicate gate names, which
    would otherwise surface as a confusing gating-state key mismatch.

    Args:
        neuron: The neuron to compile.
        gating_states: Initial gate values keyed by gate name, one dict per
            simulation.

    Returns:
        The compiled model and, for each dict, its gate values in
        ``neuron.all_gating_variables`` order.

    Raises:
        ValueError: If gate names repeat across channels, or the keys of a
            gating state are not exactly the neuron's gate names.
    """
    from . import _core

    spec = lower(neuron)
    gates = neuron.all_gating_variables
    expected = sorted(g.name for g in gates)
    ordered: list[list[float]] = []
    for gating_state in gating_states:
        if sorted(gating_state) != expected:
            raise ValueError(
                "gating_state keys do not match the neuron's gating variables: "
                f"got {sorted(gating_state)}, expected {expected}"
            )
        ordered.append([float(gating_state[g.name]) for g in gates])
    return _core.Model(json.dumps(spec)), ordered


def _as_float_array(values: np.ndarray) -> np.ndarray:
    """Return a protocol as the contiguous ``float64`` array the core reads.

    Args:
        values: A stimulus or voltage protocol.

    Returns:
        The same values as a C-contiguous ``float64`` array.
    """
    return np.ascontiguousarray(values, dtype=np.float64)


def _current_clamp_order(neuron: Neuron) -> list[str]:
    """Return the current-clamp result's leading field order.

    Args:
        neuron: The simulated neuron.

    Returns:
        ``voltage``, the channel currents, then ``Itotal``.
    """
    return ["voltage", *(ch.current_name for ch in neuron.channels), "Itotal"]


def _voltage_clamp_order(neuron: Neuron) -> list[str]:
    """Return the voltage-clamp result's leading field order.

    Args:
        neuron: The simulated neuron.

    Returns:
        ``voltage``, ``Itotal``, then the channel currents.
    """
    return ["voltage", "Itotal", *(ch.current_name for ch in neuron.channels)]


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
