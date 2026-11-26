"""Write benchmark fixtures for ``spikes/neuroclamp_core/examples/bench.rs``.

Each fixture holds one preset's lowered spec (as a JSON string) and the initial
state the library would use (gates at steady state at ``v_rest``, Ca²⁺ at
``ca_init`` or ``ca_rest``), so the Rust benchmark runs without Python.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lower import lower
from translate import to_tree_neuron

import neuroclamp as nc
from neuroclamp.clamp_simulations import _initialize_gating_variables
from neuroclamp.constants import SIM_SAMPLING_FREQ

OUT_DIR = Path(__file__).parents[1] / "neuroclamp_core" / "bench_fixtures"


def _slug(name: str) -> str:
    """Turn a preset name into a filesystem-safe identifier.

    Args:
        name: A neuron preset name.

    Returns:
        Lowercase name with non-alphanumeric runs replaced by ``_``.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    """Write one fixture per neuron preset."""
    OUT_DIR.mkdir(exist_ok=True)
    for key, factory in nc.NEURON_PRESETS.items():
        neuron = factory()
        cd = neuron.calcium_dynamics
        ca0 = (cd.ca_init if cd.ca_init is not None else cd.ca_rest) if cd else 0.0
        init = _initialize_gating_variables(neuron, neuron.v_rest, ca0)
        fixture = {
            "name": key,
            "spec_json": json.dumps(lower(to_tree_neuron(neuron))),
            "v0": neuron.v_rest,
            "gates0": [init[gv.name] for gv in neuron.all_gating_variables],
            "ca0": ca0,
            "dt": 1000.0 / SIM_SAMPLING_FREQ,
        }
        path = OUT_DIR / f"{_slug(key)}.json"
        path.write_text(json.dumps(fixture, indent=1) + "\n")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
