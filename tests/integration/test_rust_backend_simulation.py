"""Simulations on the Rust backend agree with the Python backend.

The Rust core follows the Python integrator's order of floating-point
operations; the only differences come from integer powers above 2, which it
computes by repeated multiplication, so results agree to about 1e-9 mV even
through action potentials.
"""

from __future__ import annotations

import numpy as np
import pytest

import neuroclamp as nc
from neuroclamp import Neuron
from neuroclamp.calcium import CalciumDynamics
from neuroclamp.channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from neuroclamp.rates import ExpLinear, Sigmoid, VoltageOnlyFn

PRESETS = list(nc.NEURON_PRESETS.items())
PRESET_IDS = [key for key, _ in PRESETS]

# 40 ms at 40 kHz with a 25 ms depolarizing step, enough to fire every preset.
_STEPS = 1600
_STIMULUS = np.zeros(_STEPS)
_STIMULUS[200:1200] = 10.0

# Hold at -80 mV, step to 0 mV for 20 ms, then return: activates and
# inactivates every channel type in the presets.
_COMMAND = np.full(_STEPS, -80.0)
_COMMAND[200:1000] = 0.0


def _assert_results_match(python: np.ndarray, rust: np.ndarray) -> None:
    """Assert two results have the same fields and agree to within rounding.

    Args:
        python: Result from the Python backend.
        rust: Result from the Rust backend.
    """
    assert rust.dtype == python.dtype
    np.testing.assert_array_equal(rust["time"], python["time"])
    for name in python.dtype.names or ():
        np.testing.assert_allclose(
            rust[name], python[name], rtol=1e-9, atol=1e-9, err_msg=name
        )


def _calcium_channel_neuron(*, dynamics: bool) -> Neuron:
    """Build a small neuron with a Ca²⁺ channel, with or without Ca²⁺ dynamics.

    Args:
        dynamics: Whether to attach :class:`CalciumDynamics`.

    Returns:
        The neuron.
    """
    gate = GatingVariable(
        name="d",
        power=2,
        alpha=Sigmoid(1.0, -20.0, 5.0),
        beta=Sigmoid(1.0, -20.0, -5.0),
    )
    return Neuron(
        channels=(
            IonChannel(
                name="CaL",
                g_max=0.5,
                gating_variables=(gate,),
                reversal_spec=NernstSpec(IonSpecies.CALCIUM),
                carries_calcium=True,
            ),
            IonChannel(
                name="L",
                g_max=0.1,
                gating_variables=(),
                reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
            ),
        ),
        calcium_dynamics=CalciumDynamics() if dynamics else None,
    )


@pytest.mark.parametrize(("key", "factory"), PRESETS, ids=PRESET_IDS)
def test_rust_backend_matches_python_for_every_preset(key, factory) -> None:
    """Every preset gives the same current-clamp result on both backends.

    Args:
        key: Preset name.
        factory: Preset constructor.
    """
    neuron = factory()
    python = nc.simulate_current_clamp(neuron, _STIMULUS)
    rust = nc.simulate_current_clamp(neuron, _STIMULUS, backend="rust")
    _assert_results_match(python, rust)


def test_rust_backend_from_state_matches_python() -> None:
    """Continuing from a mid-run state agrees across backends."""
    neuron = nc.NEURON_PRESETS["Purkinje Neuron"]()
    first = nc.simulate_current_clamp(neuron, _STIMULUS[:800])
    last = first[-1]
    gates = {g.name: float(last[g.name]) for g in neuron.all_gating_variables}
    v0, ca0 = float(last["voltage"]), float(last["ca_i"])
    rest = _STIMULUS[800:]
    python = nc.simulate_current_clamp_from_state(neuron, rest, v0, gates, ca0)
    rust = nc.simulate_current_clamp_from_state(
        neuron, rest, v0, gates, ca0, backend="rust"
    )
    _assert_results_match(python, rust)


@pytest.mark.parametrize("dynamics", [True, False], ids=["dynamics", "constant"])
def test_rust_backend_handles_calcium_with_and_without_dynamics(dynamics) -> None:
    """Ca²⁺ channels agree whether Ca²⁺ evolves or is held at ``Ca_in``.

    Args:
        dynamics: Whether the neuron has Ca²⁺ dynamics.
    """
    neuron = _calcium_channel_neuron(dynamics=dynamics)
    python = nc.simulate_current_clamp(neuron, _STIMULUS)
    rust = nc.simulate_current_clamp(neuron, _STIMULUS, backend="rust")
    _assert_results_match(python, rust)
    assert ("ca_i" in rust.dtype.names) == dynamics


def test_rust_backend_rejects_uncompilable_rates() -> None:
    """A gate with a plain-callable rate cannot run on the Rust backend."""
    gate = GatingVariable(
        name="w",
        power=1,
        alpha=VoltageOnlyFn(lambda v, ca: 0.1),
        beta=ExpLinear(0.01, -55.0, 10.0),
    )
    neuron = Neuron(
        channels=(
            IonChannel(
                name="Kx",
                g_max=1.0,
                gating_variables=(gate,),
                reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
            ),
        )
    )
    with pytest.raises(TypeError, match="channel 'Kx', gate 'w'"):
        nc.simulate_current_clamp(neuron, _STIMULUS, backend="rust")


def test_rust_backend_rejects_mismatched_gating_state() -> None:
    """A from-state run with the wrong gate names fails on the Rust backend too."""
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    with pytest.raises(ValueError, match="gating_state keys"):
        nc.simulate_current_clamp_from_state(
            neuron, _STIMULUS, -65.0, {"m": 0.05}, backend="rust"
        )


@pytest.mark.parametrize("backend", ["numba", "Rust", ""])
def test_unknown_backend_is_rejected(backend) -> None:
    """Only ``"python"`` and ``"rust"`` are accepted backend names.

    Args:
        backend: An invalid backend name.
    """
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    with pytest.raises(ValueError, match="backend must be"):
        nc.simulate_current_clamp(neuron, _STIMULUS, backend=backend)


@pytest.mark.parametrize(("key", "factory"), PRESETS, ids=PRESET_IDS)
def test_rust_voltage_clamp_matches_python_for_every_preset(key, factory) -> None:
    """Every preset gives the same voltage-clamp result on both backends.

    Args:
        key: Preset name.
        factory: Preset constructor.
    """
    neuron = factory()
    python = nc.simulate_voltage_clamp(neuron, _COMMAND)
    rust = nc.simulate_voltage_clamp(neuron, _COMMAND, backend="rust")
    _assert_results_match(python, rust)


def test_rust_voltage_clamp_from_state_matches_python() -> None:
    """Continuing a voltage clamp from a mid-run state agrees across backends."""
    neuron = nc.NEURON_PRESETS["Hippocampal CA1 Pyramidal Neuron"]()
    first = nc.simulate_voltage_clamp(neuron, _COMMAND[:600])
    last = first[-1]
    gates = {g.name: float(last[g.name]) for g in neuron.all_gating_variables}
    ca0 = float(last["ca_i"])
    rest = _COMMAND[600:]
    python = nc.simulate_voltage_clamp_from_state(neuron, rest, gates, ca0)
    rust = nc.simulate_voltage_clamp_from_state(
        neuron, rest, gates, ca0, backend="rust"
    )
    _assert_results_match(python, rust)


@pytest.mark.parametrize("dynamics", [True, False], ids=["dynamics", "constant"])
def test_rust_voltage_clamp_handles_calcium(dynamics) -> None:
    """Voltage-clamp Ca²⁺ currents agree whether Ca²⁺ evolves or is constant.

    Args:
        dynamics: Whether the neuron has Ca²⁺ dynamics.
    """
    neuron = _calcium_channel_neuron(dynamics=dynamics)
    python = nc.simulate_voltage_clamp(neuron, _COMMAND)
    rust = nc.simulate_voltage_clamp(neuron, _COMMAND, backend="rust")
    _assert_results_match(python, rust)


def test_rust_voltage_clamp_rejects_uncompilable_rates() -> None:
    """A plain-callable rate cannot run voltage clamp on the Rust backend."""
    gate = GatingVariable(
        name="w",
        power=1,
        alpha=VoltageOnlyFn(lambda v, ca: 0.1),
        beta=ExpLinear(0.01, -55.0, 10.0),
    )
    neuron = Neuron(
        channels=(
            IonChannel(
                name="Kx",
                g_max=1.0,
                gating_variables=(gate,),
                reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
            ),
        )
    )
    with pytest.raises(TypeError, match="channel 'Kx', gate 'w'"):
        nc.simulate_voltage_clamp(neuron, _COMMAND, backend="rust")


@pytest.mark.parametrize("backend", ["numba", "Rust", ""])
def test_unknown_backend_is_rejected_in_voltage_clamp(backend) -> None:
    """Voltage clamp rejects unknown backend names too.

    Args:
        backend: An invalid backend name.
    """
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    with pytest.raises(ValueError, match="backend must be"):
        nc.simulate_voltage_clamp(neuron, _COMMAND, backend=backend)
