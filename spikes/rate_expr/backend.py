"""Current-clamp entry point with an optional compiled backend.

``simulate_current_clamp(neuron, current, backend="rust")`` lowers the neuron,
runs the Rust integrator in ``spikes/patch_sim_core``, and returns the same
structured array as :func:`patch_sim.simulate_current_clamp`. The default
``backend="python"`` simply calls the library.

The extension is imported lazily, so this module works without it built.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from lower import lower
from translate import to_tree_neuron

import patch_sim as ps
from patch_sim.clamp_simulations import _initialize_gating_variables, _setup_simulation
from patch_sim.constants import SIM_SAMPLING_FREQ
from patch_sim.neuron import Neuron

Backend = Literal["python", "rust"]

_BUILD_HINT = (
    "The Rust backend is not built. From the repo root run:\n"
    "  VIRTUAL_ENV=.venv uvx maturin develop --release --uv "
    "-m spikes/patch_sim_core/Cargo.toml"
)


@lru_cache(maxsize=32)
def _compiled_model(spec_json: str) -> Any:
    """Build (and cache) a compiled model from its JSON spec.

    Args:
        spec_json: The JSON-encoded output of :func:`lower.lower`.

    Returns:
        A ``patch_sim_core.Model`` instance.

    Raises:
        ImportError: If the extension has not been built.
    """
    try:
        import patch_sim_core
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    return patch_sim_core.Model(spec_json)


def _has_tree_rates(neuron: Neuron) -> bool:
    """Report whether every gate already uses lowerable expression-tree rates.

    Args:
        neuron: The neuron to inspect.

    Returns:
        ``True`` if all rates provide ``to_spec``.
    """
    return all(
        hasattr(g.alpha, "to_spec") and hasattr(g.beta, "to_spec")
        for g in neuron.all_gating_variables
    )


def _result_dtype(neuron: Neuron) -> np.dtype:
    """Build the structured dtype used by ``patch_sim.simulate_current_clamp``.

    Args:
        neuron: The simulated neuron.

    Returns:
        Field order: time, voltage, channel currents, Itotal, gates, ca_i.
    """
    fields: list[tuple[str, type]] = [("time", np.float64), ("voltage", np.float64)]
    fields += [(ch.current_name, np.float64) for ch in neuron.channels]
    fields.append(("Itotal", np.float64))
    fields += [(gv.name, np.float64) for gv in neuron.all_gating_variables]
    if neuron.calcium_dynamics is not None:
        fields.append(("ca_i", np.float64))
    return np.dtype(fields)


def simulate_current_clamp(
    neuron: Neuron,
    current_external: np.ndarray,
    *,
    backend: Backend = "python",
) -> np.ndarray:
    """Simulate current clamp with the selected backend.

    Initial conditions are computed in Python exactly as the library does
    (gates at steady state at ``v_rest``, Ca²⁺ at ``ca_init`` or ``ca_rest``),
    so both backends start from identical state.

    Args:
        neuron: The neuron to simulate. Gates using today's callable rates are
            translated to expression trees automatically for the Rust backend.
        current_external: Injected current per time step in µA/cm².
        backend: ``"python"`` (the library) or ``"rust"`` (compiled spike).

    Returns:
        A structured array with the same fields as
        :func:`patch_sim.simulate_current_clamp`.

    Raises:
        ValueError: If ``backend`` is unknown or the input is invalid.
    """
    if backend == "python":
        return ps.simulate_current_clamp(neuron, current_external)
    if backend != "rust":
        raise ValueError(f"unknown backend {backend!r}; use 'python' or 'rust'")

    current = np.ascontiguousarray(current_external, dtype=np.float64)
    if current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("current_external must be non-empty and finite.")

    cd = neuron.calcium_dynamics
    ca0 = (cd.ca_init if cd.ca_init is not None else cd.ca_rest) if cd else 0.0
    init = _initialize_gating_variables(neuron, neuron.v_rest, ca0)
    gates0 = [init[gv.name] for gv in neuron.all_gating_variables]

    tree_neuron = neuron if _has_tree_rates(neuron) else to_tree_neuron(neuron)
    model = _compiled_model(json.dumps(lower(tree_neuron)))
    dt, time = _setup_simulation(current.size, SIM_SAMPLING_FREQ)
    columns = model.simulate_current_clamp(current, neuron.v_rest, gates0, ca0, dt)

    result = np.empty(current.size, dtype=_result_dtype(neuron))
    result["time"] = time
    for k, name in enumerate(result.dtype.names[1:]):
        result[name] = columns[:, k]
    return result
