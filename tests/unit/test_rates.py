"""Unit tests for the gating-variable rate hierarchy in ``neuroclamp.rates``."""

from __future__ import annotations

import numpy as np
import pytest

from neuroclamp import Neuron
from neuroclamp.channels import GatingVariable, IonChannel, IonSpecies, NernstSpec
from neuroclamp.clamp_simulations import _precompute_rate_tables
from neuroclamp.electrochemistry import boltzmann_cosh_rates
from neuroclamp.rates import CalciumDependentFn, Rate, VoltageOnlyFn


class _UnmarkedRate(Rate):
    """A rate that subclasses ``Rate`` directly, without declaring Ca²⁺ use."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Return a constant rate.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            0.1 per ms.
        """
        return 0.1


def _const(V: float, ca_i: float) -> float:
    """Return a constant rate for use in adapters.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        0.2 per ms.
    """
    return 0.2


def test_voltage_only_rates_do_not_use_calcium() -> None:
    """``VoltageOnlyFn`` and Boltzmann-cosh rates report no Ca²⁺ dependence."""
    alpha, beta = boltzmann_cosh_rates(
        half=-40.0, slope=8.0, tau_scale=5.0, tau_floor=0.1
    )
    assert VoltageOnlyFn(_const).uses_calcium is False
    assert alpha.uses_calcium is False
    assert beta.uses_calcium is False


def test_calcium_dependent_rate_uses_calcium() -> None:
    """``CalciumDependentFn`` reports a Ca²⁺ dependence."""
    assert CalciumDependentFn(_const).uses_calcium is True


def test_unmarked_rate_defaults_to_using_calcium() -> None:
    """A ``Rate`` that does not declare its Ca²⁺ use is treated as Ca²⁺-dependent.

    This is the conservative default: such a rate is never tabulated in
    voltage clamp, so it cannot be cached with a wrong ca_i.
    """
    assert _UnmarkedRate().uses_calcium is True


def test_voltage_clamp_tabulates_only_calcium_free_gates() -> None:
    """Rate tables are built exactly for gates whose rates do not read ca_i."""
    channel = IonChannel(
        name="X",
        g_max=1.0,
        gating_variables=(
            GatingVariable("v_only", 1, VoltageOnlyFn(_const), VoltageOnlyFn(_const)),
            GatingVariable(
                "ca_dep", 1, VoltageOnlyFn(_const), CalciumDependentFn(_const)
            ),
            GatingVariable("unmarked", 1, _UnmarkedRate(), VoltageOnlyFn(_const)),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(channels=(channel,))
    protocol = np.full(5, -60.0)

    alpha_tables, beta_tables = _precompute_rate_tables(neuron, protocol)

    assert alpha_tables["v_only"] is not None
    assert beta_tables["v_only"] is not None
    # Tables have the Q10 factor folded in.
    assert alpha_tables["v_only"] == pytest.approx(np.full(5, 0.2 * neuron.q10_factor))
    for name in ("ca_dep", "unmarked"):
        assert alpha_tables[name] is None
        assert beta_tables[name] is None
