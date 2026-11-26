"""Unit tests for the gating-variable rate hierarchy in ``neuroclamp.rates``."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from neuroclamp import Neuron
from neuroclamp.channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
    NernstSpec,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
)
from neuroclamp.channels.auxiliary import _alpha_qSK, _beta_qSK
from neuroclamp.clamp_simulations import _precompute_rate_tables
from neuroclamp.electrochemistry import boltzmann_cosh_rates
from neuroclamp.rates import (
    Add,
    CalciumDependentFn,
    Const,
    Div,
    Exp,
    ExpLinear,
    Expr,
    Hill,
    Max,
    Mul,
    Rate,
    Sech,
    Sigmoid,
    Sub,
    VoltageOnlyFn,
    inf_tau,
)


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


# ---------------------------------------------------------------------------
# Rate expressions
# ---------------------------------------------------------------------------

_VOLTAGES = [-120.0, -80.0, -65.0, -55.0, -40.0, -20.0, 0.0, 30.0]


def test_leaf_forms_match_their_formulas() -> None:
    """Each leaf form evaluates its documented formula."""
    V = -50.0
    assert Const(0.3)(V, 0.0) == pytest.approx(0.3)
    assert Exp(2.0, -40.0, -10.0)(V, 0.0) == pytest.approx(2.0 * math.exp(1.0))
    assert Sigmoid(4.0, -40.0, 5.0)(V, 0.0) == pytest.approx(4.0 / (1 + math.exp(2)))
    assert ExpLinear(0.1, -40.0, 10.0)(V, 0.0) == pytest.approx(
        0.1 * -10.0 / (1 - math.exp(1.0))
    )
    assert Sech(3.0, -40.0, 20.0)(V, 0.0) == pytest.approx(3.0 / math.cosh(-0.5))
    assert Hill(0.001, 2)(V, 0.001) == pytest.approx(0.5)


def test_exp_linear_returns_its_limit_at_the_singularity() -> None:
    """ExpLinear returns ``k · s`` at ``V = vh`` and is continuous there."""
    rate = ExpLinear(0.01, -55.0, 10.0)
    assert rate(-55.0, 0.0) == pytest.approx(0.1)
    assert rate(-55.0 + 1e-4, 0.0) == pytest.approx(0.1, rel=1e-5)


def test_exponential_arguments_are_clipped() -> None:
    """Extreme voltages give finite values, as with ``safe_exp``."""
    assert math.isfinite(Exp(1.0, 0.0, 0.01)(1e4, 0.0))
    assert Sigmoid(1.0, 0.0, 0.01)(-1e4, 0.0) == pytest.approx(0.0)


def test_operators_build_nodes_and_wrap_numbers() -> None:
    """Arithmetic builds combinator nodes; numbers become ``Const``."""
    x = Sigmoid(1.0, -40.0, 8.0)
    assert 1.0 - x == Sub(Const(1.0), x)
    assert 2 * x == Mul(Const(2.0), x)
    assert x / 4 == Div(x, Const(4.0))
    assert x + 1 == Add(x, Const(1.0))
    assert (1.0 / (x + 1))(-40.0, 0.0) == pytest.approx(1 / 1.5)


def test_uses_calcium_is_derived_from_the_tree() -> None:
    """An expression reads ca_i exactly when it contains a ``Hill`` node."""
    voltage_only = Max(Sech(10.0, -20.0, 20.0), Const(1.0))
    calcium = Hill(0.001, 1) * Sigmoid(1.0, -20.0, 10.0)
    assert voltage_only.uses_calcium is False
    assert calcium.uses_calcium is True
    alpha, beta = inf_tau(calcium, voltage_only)
    assert alpha.uses_calcium is True
    assert beta.uses_calcium is True


def test_inf_tau_gives_the_requested_steady_state_and_time_constant() -> None:
    """``alpha / (alpha + beta) = inf`` and ``1 / (alpha + beta) = tau``."""
    inf = Sigmoid(1.0, -45.0, -6.0)
    tau = Sech(50.0, -45.0, 12.0) + 2.0
    alpha, beta = inf_tau(inf, tau)
    for V in _VOLTAGES:
        a, b = alpha(V, 0.0), beta(V, 0.0)
        assert a / (a + b) == pytest.approx(inf(V, 0.0))
        assert 1 / (a + b) == pytest.approx(tau(V, 0.0))


def test_expressions_compare_by_value_and_pickle() -> None:
    """Expressions are value objects that survive a pickle round trip."""
    alpha, _ = inf_tau(
        Sigmoid(1.0, -40.0, 8.0), Max(Sech(5.0, -40.0, 16.0), Const(0.1))
    )
    restored = pickle.loads(pickle.dumps(alpha))
    assert restored == alpha
    assert hash(restored) == hash(alpha)
    assert restored(-30.0, 0.0) == alpha(-30.0, 0.0)


def test_voltage_clamp_tabulates_voltage_only_expressions() -> None:
    """Expression gates are tabulated unless they contain a ``Hill`` node."""
    v_alpha, v_beta = inf_tau(Sigmoid(1.0, -40.0, 8.0), 0.5)
    ca_alpha, ca_beta = inf_tau(Hill(0.0003, 4), 10.0)
    channel = IonChannel(
        name="X",
        g_max=1.0,
        gating_variables=(
            GatingVariable("v_gate", 1, v_alpha, v_beta),
            GatingVariable("ca_gate", 1, ca_alpha, ca_beta),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    neuron = Neuron(channels=(channel,))
    alpha_tables, _ = _precompute_rate_tables(neuron, np.full(3, -40.0))
    assert alpha_tables["v_gate"] == pytest.approx(
        np.full(3, 0.5 / 0.5 * neuron.q10_factor)
    )
    assert alpha_tables["ca_gate"] is None


@pytest.mark.parametrize(
    ("existing", "expression"),
    [
        (alpha_n, ExpLinear(0.01, -55.0, 10.0)),
        (beta_n, Exp(0.125, -65.0, -80.0)),
        (alpha_m, ExpLinear(0.1, -40.0, 10.0)),
        (beta_m, Exp(4.0, -65.0, -18.0)),
        (alpha_h, Exp(0.07, -65.0, -20.0)),
        (beta_h, Sigmoid(1.0, -35.0, 10.0)),
    ],
    ids=["alpha_n", "beta_n", "alpha_m", "beta_m", "alpha_h", "beta_h"],
)
def test_expressions_reproduce_hh52_rates(existing: Rate, expression: Expr) -> None:
    """The six Hodgkin-Huxley (1952) rates are single expression leaves.

    Args:
        existing: The library's HH52 rate function.
        expression: Its expression equivalent.
    """
    for V in [*_VOLTAGES, -55.0 + 1e-7, -40.0 - 1e-7]:
        assert expression(V, 0.0) == pytest.approx(existing(V, 0.0), rel=1e-12)


@pytest.mark.parametrize("inverted", [False, True])
def test_inf_tau_reproduces_boltzmann_cosh_rates(inverted: bool) -> None:
    """``boltzmann_cosh_rates`` is a Sigmoid steady state and floored Sech tau.

    Args:
        inverted: Whether the gate opens at hyperpolarized potentials.
    """
    half, slope, tau_scale, tau_floor = -50.0, 8.0, 200.0, 20.0
    existing = boltzmann_cosh_rates(
        half=half,
        slope=slope,
        tau_scale=tau_scale,
        tau_floor=tau_floor,
        inverted=inverted,
    )
    expression = inf_tau(
        Sigmoid(1.0, half, -slope if inverted else slope),
        Max(Sech(tau_scale, half, 2 * slope), Const(tau_floor)),
    )
    for V in _VOLTAGES:
        for old, new in zip(existing, expression, strict=True):
            assert new(V, 0.0) == pytest.approx(old(V, 0.0), rel=1e-12)


def test_inf_tau_reproduces_the_sk_calcium_gate() -> None:
    """The SK channel's Ca²⁺-gated rates are a Hill steady state over 10 ms."""
    expression = inf_tau(Hill(0.0003, 4), 10.0)
    for ca in [0.0, 5e-5, 1e-4, 3e-4, 1e-3, 1e-2]:
        assert expression[0](-60.0, ca) == pytest.approx(_alpha_qSK(-60.0, ca))
        assert expression[1](-60.0, ca) == pytest.approx(_beta_qSK(-60.0, ca))
