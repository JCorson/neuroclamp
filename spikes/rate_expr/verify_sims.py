"""Run every preset with tree gates and compare against the existing rates."""

from __future__ import annotations

import numpy as np
from translate import to_tree_neuron

import patch_sim as ps

# 150 ms current step (10 ms baseline, 100 ms at 8 µA/cm², 40 ms recovery).
I_STEP = np.concatenate([np.zeros(400), np.full(4000, 8.0), np.zeros(1600)])
# 70 ms voltage step from −80 mV to −10 mV.
V_STEP = np.concatenate(
    [np.full(400, -80.0), np.full(1600, -10.0), np.full(800, -80.0)]
)


def main() -> None:
    """Print max voltage (current clamp) and current (voltage clamp) deviations."""
    for key, factory in ps.NEURON_PRESETS.items():
        neuron = factory()
        tree_neuron = to_tree_neuron(neuron)
        cc_a = ps.simulate_current_clamp(neuron, I_STEP)
        cc_b = ps.simulate_current_clamp(tree_neuron, I_STEP)
        vc_a = ps.simulate_voltage_clamp(neuron, V_STEP)
        vc_b = ps.simulate_voltage_clamp(tree_neuron, V_STEP)
        dv = np.max(np.abs(cc_a["voltage"] - cc_b["voltage"]))
        scale = max(float(np.max(np.abs(vc_a["Itotal"]))), 1e-12)
        di = np.max(np.abs(vc_a["Itotal"] - vc_b["Itotal"])) / scale
        print(f"{key:36s} CC max|dV| = {dv:.1e} mV   VC max rel dI = {di:.1e}")


if __name__ == "__main__":
    main()
