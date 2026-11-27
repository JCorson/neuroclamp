"""Unit tests for the gating-variable rate hierarchy in ``neuroclamp.rates``."""

from __future__ import annotations

import copy
import math
import pickle
from collections.abc import Callable
from typing import cast

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


def test_exp_linear_with_negated_k_and_s_gives_the_mirrored_form() -> None:
    """``ExpLinear(-k, vh, -s)`` equals ``k·x / (exp(x/s) − 1)`` and is positive."""
    rate = ExpLinear(-0.28, 40.0, -5.0)
    for V in (-80.0, 0.0, 40.0 + 1e-3, 60.0):
        x = V - 40.0
        assert rate(V, 0.0) == pytest.approx(0.28 * x / (math.exp(x / 5.0) - 1.0))
        assert rate(V, 0.0) > 0.0
    assert rate(40.0, 0.0) == pytest.approx(0.28 * 5.0)


def _plateau(V: float, ca_i: float) -> float:
    """Return a constant voltage-only test rate.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        0.5 per ms.
    """
    return 0.5


def _ca_linear(V: float, ca_i: float) -> float:
    """Return a test rate proportional to ca_i.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        ``2 · ca_i`` per ms.
    """
    return 2.0 * ca_i


def test_expressions_combine_with_adapter_rates() -> None:
    """Adapters can be operands of expressions on either side of an operator."""
    sig = Sigmoid(1.0, -40.0, 8.0)
    voltage_only = VoltageOnlyFn(_plateau)
    ca_dependent = CalciumDependentFn(_ca_linear)

    product = sig * voltage_only
    assert product == Mul(sig, voltage_only)
    assert product(-40.0, 0.0) == pytest.approx(0.25)
    assert not product.uses_calcium

    reflected = voltage_only + sig
    assert reflected == Add(voltage_only, sig)
    assert reflected(-40.0, 0.0) == pytest.approx(1.0)

    with_ca = sig - ca_dependent
    assert with_ca(-40.0, 0.1) == pytest.approx(0.3)
    assert with_ca.uses_calcium

    alpha, beta = inf_tau(sig, voltage_only)
    assert alpha(-40.0, 0.0) == pytest.approx(1.0)
    assert beta(-40.0, 0.0) == pytest.approx(1.0)
    assert pickle.loads(pickle.dumps(product)) == product


def test_operators_reject_unsupported_operands_with_type_error() -> None:
    """Non-numeric, non-rate operands raise TypeError via NotImplemented."""
    sig = Sigmoid(1.0, -40.0, 8.0)
    assert sig.__add__("a") is NotImplemented
    with pytest.raises(TypeError):
        sig + "a"  # type: ignore[operator]
    with pytest.raises(TypeError):
        "a" * sig  # type: ignore[operator]
    with pytest.raises(TypeError):
        inf_tau(sig, "a")  # type: ignore[arg-type]


def test_inf_tau_accepts_a_numeric_steady_state() -> None:
    """A numeric ``inf`` is wrapped in ``Const`` like a numeric ``tau``."""
    alpha, beta = inf_tau(0.25, 5.0)
    assert alpha.a == Const(0.25)
    assert alpha(-60.0, 0.0) == pytest.approx(0.05)
    assert beta(-60.0, 0.0) == pytest.approx(0.15)
    assert not alpha.uses_calcium
    assert not beta.uses_calcium


def test_exp_linear_shares_the_channel_singularity_threshold() -> None:
    """``ExpLinear`` uses the same singularity tolerance as the HH52 rates."""
    from neuroclamp.channels import SINGULARITY_THRESHOLD
    from neuroclamp.channels._traub_miles import (
        SINGULARITY_THRESHOLD as TRAUB_THRESHOLD,
    )

    assert ExpLinear(0.1, -40.0, 10.0).tol == SINGULARITY_THRESHOLD
    assert TRAUB_THRESHOLD is SINGULARITY_THRESHOLD


@pytest.mark.parametrize("form", [Exp, Sigmoid, ExpLinear, Sech])
def test_leaf_forms_reject_a_zero_voltage_scale(
    form: Callable[[float, float, float], Expr],
) -> None:
    """A zero ``s`` fails when the expression is built, not mid-simulation.

    Args:
        form: The leaf class under test.
    """
    with pytest.raises(ValueError, match="nonzero"):
        form(1.0, -40.0, 0.0)


@pytest.mark.parametrize(("kd", "n"), [(0.0, 4), (-1e-3, 4), (5e-4, 0)])
def test_hill_rejects_non_positive_parameters(kd: float, n: int) -> None:
    """``Hill`` requires a positive half-activation constant and coefficient.

    Args:
        kd: Half-activation concentration in mM.
        n: Hill coefficient.
    """
    with pytest.raises(ValueError, match="positive"):
        Hill(kd, n)


# ---------------------------------------------------------------------------
# Compiled evaluation
# ---------------------------------------------------------------------------


def _scaled_v(V: float, ca_i: float) -> float:
    """A voltage-only rate written as a plain function.

    Args:
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

    Returns:
        A rate that grows linearly with voltage.
    """
    return 0.01 * (V + 100.0)


def _ca_rate(V: float, ca_i: float) -> float:
    """A Ca²⁺-dependent rate written as a plain function.

    Args:
        V: Membrane voltage in mV (ignored).
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        A rate proportional to ca_i.
    """
    return 50.0 * ca_i


_COMPILED_CASES: list[Expr] = [
    Const(0.3),
    Exp(0.125, -65.0, -80.0),
    Sigmoid(4.0, -40.0, 5.0),
    ExpLinear(0.1, -40.0, 10.0),
    ExpLinear(-0.28, -16.2, -5.0, tol=1e-7),
    Sech(10.0, -20.0, 20.0),
    Hill(0.0003, 4) * Sigmoid(1.0, -20.0, 10.0),
    Max(Sech(200.0, -50.0, 16.0), Const(20.0)),
    *inf_tau(
        Sigmoid(1.0, -45.5, -6.4),
        1.0 / (Exp(0.128, -38.0, -18.0) + Sigmoid(4.0, -15.0, 5.0)),
    ),
    *inf_tau(
        Hill(0.001, 1) * Sigmoid(1.0, -20.0, 10.0),
        Max(Sech(10.0, -20.0, 20.0), Const(1.0)),
    ),
    20.0 + Sigmoid(180.0, -50.0, 5.0) - 2 * Exp(1.0, -60.0, 30.0),
    VoltageOnlyFn(_scaled_v) * Sigmoid(1.0, -40.0, 8.0) + CalciumDependentFn(_ca_rate),
]


@pytest.mark.parametrize("expr", _COMPILED_CASES, ids=lambda e: type(e).__name__)
def test_compiled_evaluation_matches_the_reference_exactly(expr: Expr) -> None:
    """Calling an expression gives exactly what ``evaluate`` gives.

    The compiled function performs the same operations in the same order, so
    results are compared for exact equality, including at the exp-linear
    singular points.

    Args:
        expr: An expression to evaluate both ways.
    """
    voltages = [-120.0, -65.0, -55.0, -40.0, -16.2, -16.2 + 5e-8, 0.0, 40.0, 1e4]
    for V in voltages:
        for ca in (0.0, 5e-5, 3e-4, 1e-2):
            assert expr(V, ca) == expr.evaluate(V, ca), (V, ca)


def test_compiled_expressions_still_pickle_copy_and_compare() -> None:
    """Compiling on first call leaves pickling, copying and equality intact."""
    alpha, beta = inf_tau(
        Sigmoid(1.0, -40.0, 8.0), Max(Sech(5.0, -40.0, 16.0), Const(0.1))
    )
    before = alpha(-30.0, 0.0)  # compiles and caches

    restored = pickle.loads(pickle.dumps(alpha))
    assert restored == alpha
    assert hash(restored) == hash(alpha)
    assert restored(-30.0, 0.0) == before
    assert copy.deepcopy(beta)(-30.0, 0.0) == beta(-30.0, 0.0)
    assert repr(alpha) == repr(restored)


class _Doubled(Expr):
    """An ``Expr`` subclass defined outside ``neuroclamp.rates``.

    It implements only ``evaluate``, so the compiler calls it as a function.
    """

    def __init__(self, inner: Expr) -> None:
        """Wrap an expression.

        Args:
            inner: The expression to double.
        """
        self.inner = inner

    def evaluate(self, V: float, ca_i: float) -> float:
        """Return twice the wrapped expression.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            Twice the inner value.
        """
        return 2.0 * self.inner(V, ca_i)


def test_expr_subclasses_only_need_evaluate() -> None:
    """A custom ``Expr`` that implements only ``evaluate`` compiles and combines."""
    custom = _Doubled(Sigmoid(1.0, -40.0, 8.0))
    assert custom(-40.0, 0.0) == pytest.approx(1.0)
    assert (custom + 1.0)(-40.0, 0.0) == pytest.approx(2.0)


def test_parameter_values_are_never_executed_as_code() -> None:
    """Node parameters reach the compiled function as values, not source text.

    A constant holding code-like text comes back unchanged instead of being
    executed, and non-finite constants work as ordinary floats.
    """
    payload = "__import__('os').getcwd()"
    assert Const(cast("float", payload))(-65.0, 0.0) == payload
    assert math.isnan(Const(math.nan)(-65.0, 0.0))
    assert Const(math.inf)(-65.0, 0.0) == math.inf


class _ConstantExp(Exp):
    """An ``Exp`` subclass whose ``evaluate`` ignores the parent's formula."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Return a fixed value.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            Always 99.0.
        """
        return 99.0


class _SwappedSub(Sub):
    """A ``Sub`` subclass that computes ``b − a``."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``b − a``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The reversed difference.
        """
        return self.b(V, ca_i) - self.a(V, ca_i)


def test_subclass_evaluate_overrides_are_honored_when_called() -> None:
    """A built-in form subclassed with a new ``evaluate`` runs that ``evaluate``.

    The override applies whether the subclass is the root of the expression
    or an operand nested inside it.
    """
    leaf = _ConstantExp(1.0, 0.0, 1.0)
    assert leaf(0.0, 0.0) == 99.0
    assert (leaf + 0.0)(0.0, 0.0) == 99.0
    assert (2.0 * leaf)(0.0, 0.0) == 198.0

    swapped = _SwappedSub(Const(1.0), Const(5.0))
    assert swapped(-65.0, 0.0) == 4.0
    assert (swapped * 2.0)(-65.0, 0.0) == 8.0


@pytest.mark.parametrize("n_terms", [250, 2000])
def test_deep_expressions_evaluate_like_the_reference(n_terms: int) -> None:
    """A deeply nested expression is callable and matches ``evaluate``.

    Summing many terms one at a time builds a chain of ``Add`` nodes far
    deeper than Python can parse as a single nested expression.

    Args:
        n_terms: Number of summed terms, which sets the tree depth.
    """
    expr = Const(0.0)
    for i in range(n_terms):
        expr = expr + Sigmoid(1.0, -40.0 + i, 5.0)
    if n_terms <= 250:
        assert expr(-30.0, 0.0) == expr.evaluate(-30.0, 0.0)
    else:
        # The reference walk recurses once per node, too deep to call directly.
        expected = sum(
            Sigmoid(1.0, -40.0 + i, 5.0).evaluate(-30.0, 0.0) for i in range(n_terms)
        )
        assert expr(-30.0, 0.0) == pytest.approx(expected)
