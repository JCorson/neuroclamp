"""``simulate_batch(..., backend="rust")`` runs protocol families on the Rust core.

The Rust batch runs every sweep of a family in one call to the compiled core,
on its thread pool.  Each sweep must equal a single Rust run on that protocol
exactly (the batch only parallelizes), and agree with the Python backend
within the parity suite's timing-aware tolerance: pacemaking presets turn
last-bit rounding differences into nanosecond spike-time shifts over long
runs, so a flat tolerance would flag the steep spike edges.
"""

from __future__ import annotations

import numpy as np
import pytest

import neuroclamp as nc
from neuroclamp import Neuron
from neuroclamp.channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from neuroclamp.clamp_simulations import SimulationResult
from neuroclamp.constants import CURRENT_CLAMP
from neuroclamp.rates import ExpLinear, VoltageOnlyFn
from tests.parity._golden import tolerance_band

# Every multi-sweep protocol preset, in both clamp modes.
FAMILIES = [
    key
    for key in nc.PROTOCOL_PRESETS
    if nc.build_protocol_from_preset(key, "Purkinje Neuron").shape[0] > 1
]
NEURONS = ["Purkinje Neuron", "Hippocampal CA1 Pyramidal Neuron"]


def _simulate_fn(protocol: str):
    """Return the single-run function for a protocol preset's clamp mode.

    Args:
        protocol: Key in :data:`neuroclamp.PROTOCOL_PRESETS`.

    Returns:
        :func:`neuroclamp.simulate_current_clamp` or
        :func:`neuroclamp.simulate_voltage_clamp`.
    """
    if nc.PROTOCOL_PRESETS[protocol]["clamp_mode"] == CURRENT_CLAMP:
        return nc.simulate_current_clamp
    return nc.simulate_voltage_clamp


@pytest.mark.parametrize("neuron_key", NEURONS)
@pytest.mark.parametrize("protocol", FAMILIES)
def test_rust_batch_equals_single_rust_runs(neuron_key, protocol) -> None:
    """Each sweep of a Rust batch equals a single Rust run on that protocol.

    Args:
        neuron_key: Neuron preset name.
        protocol: Multi-sweep protocol preset name.
    """
    neuron = nc.NEURON_PRESETS[neuron_key]()
    sweeps = list(nc.build_protocol_from_preset(protocol, neuron_key))
    simulate_fn = _simulate_fn(protocol)
    batch = list(nc.simulate_batch(neuron, sweeps, simulate_fn, backend="rust"))
    assert len(batch) == len(sweeps)
    for result, sweep in zip(batch, sweeps, strict=True):
        single = simulate_fn(neuron, sweep, backend="rust")
        assert result.dtype == single.dtype
        np.testing.assert_array_equal(result, single)


@pytest.mark.parametrize("protocol", FAMILIES)
def test_rust_batch_matches_python_batch(protocol) -> None:
    """A Rust batch agrees with the Python backend sweep by sweep.

    Voltage, total current and Ca²⁺ are compared with the parity suite's
    tolerance band.

    Args:
        protocol: Multi-sweep protocol preset name.
    """
    neuron = nc.NEURON_PRESETS["Purkinje Neuron"]()
    sweeps = list(nc.build_protocol_from_preset(protocol, "Purkinje Neuron"))
    simulate_fn = _simulate_fn(protocol)
    rust = list(nc.simulate_batch(neuron, sweeps, simulate_fn, backend="rust"))
    for index, (result, sweep) in enumerate(zip(rust, sweeps, strict=True)):
        python = simulate_fn(neuron, sweep)
        assert result.dtype == python.dtype
        for name in ("voltage", "Itotal", "ca_i"):
            excess = np.abs(result[name] - python[name]) - tolerance_band(
                name, python[name]
            )
            assert excess.max() <= 0.0, (protocol, index, name, excess.max())


@pytest.mark.parametrize(
    "simulate_fn", [nc.simulate_current_clamp, nc.simulate_voltage_clamp]
)
def test_rust_batch_accepts_sweeps_of_different_lengths(simulate_fn) -> None:
    """Sweeps in one batch may differ in length.

    Args:
        simulate_fn: Single-run function for the clamp mode under test.
    """
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    base = -65.0 if simulate_fn is nc.simulate_voltage_clamp else 0.0
    sweeps = [np.full(n, base) for n in (400, 1000, 250)]
    batch = list(nc.simulate_batch(neuron, sweeps, simulate_fn, backend="rust"))
    assert [len(result) for result in batch] == [400, 1000, 250]
    for result, sweep in zip(batch, sweeps, strict=True):
        np.testing.assert_array_equal(
            result, simulate_fn(neuron, sweep, backend="rust")
        )


def test_rust_voltage_clamp_batch_starts_each_sweep_at_its_own_holding() -> None:
    """Voltage-clamp sweeps start at steady state at their own first voltage."""
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    sweeps = [np.full(200, holding) for holding in (-100.0, -65.0, -30.0)]
    batch = list(nc.simulate_batch(neuron, sweeps, backend="rust"))
    starts = [float(result["h"][0]) for result in batch]
    # h inactivates with depolarization, so its steady state falls.
    assert starts[0] > starts[1] > starts[2]


def test_rust_batch_of_no_protocols_yields_nothing() -> None:
    """An empty family yields no results on the Rust backend either."""
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    assert list(nc.simulate_batch(neuron, [], backend="rust")) == []


def test_rust_batch_rejects_other_simulation_functions() -> None:
    """Only the two clamp runners have a Rust counterpart."""
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()

    def custom(neuron: Neuron, protocol: np.ndarray) -> SimulationResult:
        """Stand in for a caller-supplied simulation function.

        Args:
            neuron: The neuron.
            protocol: The protocol.

        Returns:
            The Python current-clamp result.
        """
        return nc.simulate_current_clamp(neuron, protocol)

    with pytest.raises(ValueError, match="not custom"):
        list(nc.simulate_batch(neuron, [np.zeros(10)], custom, backend="rust"))


@pytest.mark.parametrize(
    ("simulate_fn", "bad"),
    [
        (nc.simulate_current_clamp, np.array([])),
        (nc.simulate_current_clamp, np.array([0.0, np.nan])),
        (nc.simulate_voltage_clamp, np.array([])),
        (nc.simulate_voltage_clamp, np.array([-65.0, np.inf])),
    ],
    ids=["cc-empty", "cc-nan", "vc-empty", "vc-inf"],
)
def test_rust_batch_validates_every_protocol(simulate_fn, bad) -> None:
    """An empty or non-finite sweep anywhere in the family is rejected.

    Args:
        simulate_fn: Single-run function for the clamp mode under test.
        bad: The invalid sweep.
    """
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    good = np.full(20, -65.0)
    with pytest.raises(ValueError, match="must not"):
        list(nc.simulate_batch(neuron, [good, bad], simulate_fn, backend="rust"))


def test_rust_batch_rejects_uncompilable_rates() -> None:
    """A neuron with a plain-callable rate cannot run a Rust batch."""
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
        list(nc.simulate_batch(neuron, [np.full(20, -65.0)], backend="rust"))


@pytest.mark.parametrize("backend", ["numba", "Rust", ""])
def test_batch_rejects_unknown_backend(backend) -> None:
    """``simulate_batch`` accepts only the known backend names.

    Args:
        backend: An invalid backend name.
    """
    neuron = nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]()
    with pytest.raises(ValueError, match="backend must be"):
        list(nc.simulate_batch(neuron, [np.zeros(10)], backend=backend))
