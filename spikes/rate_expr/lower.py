"""Lower a neuron with expression-tree gates to a flat, JSON-able model spec.

Run directly to print the spec size for every preset and the squid spec.
"""

from __future__ import annotations

import json
from typing import Any

from translate import to_tree_neuron

import patch_sim as ps
from patch_sim.neuron import Neuron
from patch_sim.utils import SAFE_EXP_CLIP_MAX


def lower(neuron: Neuron) -> dict[str, Any]:
    """Flatten a neuron into the spec a compiled backend consumes.

    Everything constant over a simulation (Q10 factor, non-Ca²⁺ reversal
    potentials) is resolved here so the backend needs no electrochemistry
    beyond the live Ca²⁺ Nernst potential.

    Args:
        neuron: A neuron whose gates all use expression-tree rates.

    Returns:
        A JSON-serializable dict describing the model.

    Raises:
        TypeError: If any gate uses an opaque callable rate.
    """
    gates = list(neuron.all_gating_variables)
    index = {id(g): i for i, g in enumerate(gates)}
    for g in gates:
        for rate in (g.alpha, g.beta):
            if not hasattr(rate, "to_spec"):
                raise TypeError(
                    f"gate {g.name!r}: {type(rate).__name__} is an opaque "
                    "callable; use the Python backend"
                )
    cd = neuron.calcium_dynamics
    return {
        "version": 1,
        "c_m": neuron.C_m,
        "phi": neuron.q10_factor,
        "exp_clip": SAFE_EXP_CLIP_MAX,
        "gates": [
            {
                "name": g.name,
                "alpha": g.alpha.to_spec(),  # type: ignore[attr-defined]
                "beta": g.beta.to_spec(),  # type: ignore[attr-defined]
                "uses_calcium": g.alpha.uses_calcium  # type: ignore[attr-defined]
                or g.beta.uses_calcium,  # type: ignore[attr-defined]
            }
            for g in gates
        ],
        "channels": [
            {
                "name": ch.name,
                "g_max": ch.g_max,
                "gates": [[index[id(g)], g.power] for g in ch.gating_variables],
                "carries_calcium": ch.carries_calcium,
                "e_rev": None
                if ch.carries_calcium
                else neuron.reversal_potentials[ch.name],
            }
            for ch in neuron.channels
        ],
        "calcium": None
        if cd is None
        else {
            "alpha_ca": cd.alpha_ca,
            "tau_ca": cd.tau_ca,
            "ca_rest": cd.ca_rest,
            "ca_out": neuron.Ca_out,
            "temperature_k": neuron.T,
            "nernst_floor_mm": 1e-9,
        },
    }


def main() -> None:
    """Print spec sizes for every preset and the full squid spec."""
    for key, factory in ps.NEURON_PRESETS.items():
        spec = json.dumps(lower(to_tree_neuron(factory())))
        print(f"{key:36s} {len(spec):6d} bytes JSON")
    squid = ps.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    print(json.dumps(lower(to_tree_neuron(squid)), indent=1))


if __name__ == "__main__":
    main()
