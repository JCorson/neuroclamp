"""Tests for lowering a neuron to the flat model spec (``neuroclamp._spec``).

The spec is the contract with compiled backends, so these tests include a
small reference evaluator written only from the format documented in
:mod:`neuroclamp._spec`.  Each lowered rate must reproduce the rate object's
own :meth:`~neuroclamp.rates.Expr.evaluate` exactly: the evaluator performs
the same floating-point operations in the same order, so a mismatch in even
the last bit means the spec dropped or reordered something.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pytest

import neuroclamp as nc
from neuroclamp import Neuron
from neuroclamp._spec import SPEC_VERSION, UnloweredRateError, lower, lower_expr
from neuroclamp.calcium import CalciumDynamics
from neuroclamp.channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from neuroclamp.channels.base import _CA_NERNST_FLOOR
from neuroclamp.clamp_simulations import _clip_state
from neuroclamp.electrochemistry import nernst_potential
from neuroclamp.rates import (
    Const,
    Exp,
    ExpLinear,
    Hill,
    InfTauAlpha,
    InfTauBeta,
    Max,
    Rate,
    Sigmoid,
    VoltageOnlyFn,
    inf_tau,
)
from neuroclamp.utils import SAFE_COSH_CLIP, SAFE_EXP_CLIP_MAX, SAFE_EXP_CLIP_MIN

PRESETS = list(nc.NEURON_PRESETS.items())
PRESET_IDS = [key for key, _ in PRESETS]

# Ca²⁺ values covering zero, rest, and well above every Hill half-point.
_CA_VALUES = (0.0, 5e-5, 3e-4, 2e-3, 0.05)


def _clip_exp(x: float, clip: float) -> float:
    """Return ``exp(x)`` with ``x`` clipped to ``[-clip, clip]`` (NaN passes).

    Args:
        x: The argument.
        clip: The spec's ``exp_clip``.

    Returns:
        The clipped exponential.
    """
    return math.exp(min(max(x, -clip), clip))


def _clip_cosh(x: float, clip: float) -> float:
    """Return ``cosh(x)`` with ``x`` clipped to ``[-clip, clip]``.

    Args:
        x: The argument.
        clip: The spec's ``exp_clip``.

    Returns:
        The clipped hyperbolic cosine.
    """
    return math.cosh(min(max(x, -clip), clip))


def _eval_node(node: dict[str, Any], v: float, ca: float, clip: float) -> float:
    """Evaluate a spec expression node from the documented format alone.

    Args:
        node: An ``{"op": ...}`` node.
        v: Membrane voltage in mV.
        ca: Intracellular Ca²⁺ in mM.
        clip: The spec's ``exp_clip``.

    Returns:
        The node's value.
    """
    op = node["op"]
    if op == "Const":
        return node["value"]
    if op in ("Add", "Sub", "Mul", "Div", "Max"):
        a = _eval_node(node["a"], v, ca, clip)
        b = _eval_node(node["b"], v, ca, clip)
        if op == "Add":
            return a + b
        if op == "Sub":
            return a - b
        if op == "Mul":
            return a * b
        if op == "Div":
            return a / b
        return max(a, b)
    if op == "Hill":
        ca_n = ca ** node["n"]
        return ca_n / (ca_n + node["kd"] ** node["n"])
    k, vh, s = node["k"], node["vh"], node["s"]
    if op == "Exp":
        return k * _clip_exp((v - vh) / s, clip)
    if op == "Sigmoid":
        return k / (1.0 + _clip_exp(-(v - vh) / s, clip))
    if op == "Sech":
        return k / _clip_cosh((v - vh) / s, clip)
    if op == "ExpLinear":
        x = v - vh
        if abs(x) < node["tol"]:
            return k * s
        return k * x / (1.0 - _clip_exp(-x / s, clip))
    raise AssertionError(f"unknown op {op!r}")


def _eval_rates(
    kinetics: dict[str, Any], v: float, ca: float, clip: float
) -> tuple[float, float]:
    """Evaluate a gate's ``(alpha, beta)`` from its spec kinetics.

    Args:
        kinetics: The gate's ``kinetics`` entry.
        v: Membrane voltage in mV.
        ca: Intracellular Ca²⁺ in mM.
        clip: The spec's ``exp_clip``.

    Returns:
        ``(alpha, beta)`` in 1/ms, before the Q10 factor.
    """
    if kinetics["form"] == "inf_tau":
        inf = _eval_node(kinetics["inf"], v, ca, clip)
        tau = _eval_node(kinetics["tau"], v, ca, clip)
        return inf / tau, (1.0 - inf) / tau
    assert kinetics["form"] == "alpha_beta"
    return (
        _eval_node(kinetics["alpha"], v, ca, clip),
        _eval_node(kinetics["beta"], v, ca, clip),
    )


def _singular_points(node: dict[str, Any]) -> list[float]:
    """Collect the ``vh`` of every ``ExpLinear`` node in a spec subtree.

    Args:
        node: Any spec value (dict, list, or scalar).

    Returns:
        The singular-point voltages, so tests can probe the limit branch.
    """
    if isinstance(node, list):
        return [vh for item in node for vh in _singular_points(item)]
    if not isinstance(node, dict):
        return []
    found = [node["vh"]] if node.get("op") == "ExpLinear" else []
    return found + [vh for value in node.values() for vh in _singular_points(value)]


def _voltages(spec: dict[str, Any]) -> list[float]:
    """Return probe voltages: a grid plus points at and near each singularity.

    Args:
        spec: A lowered model spec.

    Returns:
        Voltages in mV.
    """
    grid = [-150.0 + 2.5 * i for i in range(121)]
    near = [
        vh + offset
        for vh in _singular_points(spec["gates"])
        for offset in (0.0, 5e-7, -5e-7, 2e-6, -2e-6)
    ]
    return grid + near


def _channel(
    name: str, gates: tuple[GatingVariable, ...], *, calcium: bool = False
) -> IonChannel:
    """Build a small test channel.

    Args:
        name: Channel name.
        gates: Its gating variables.
        calcium: Whether it carries Ca²⁺.

    Returns:
        The channel.
    """
    species = IonSpecies.CALCIUM if calcium else IonSpecies.POTASSIUM
    return IonChannel(
        name=name,
        g_max=1.5,
        gating_variables=gates,
        reversal_spec=NernstSpec(species),
        carries_calcium=calcium,
    )


def _gate(name: str, alpha: Rate, beta: Rate, power: float = 1) -> GatingVariable:
    """Build a test gating variable.

    Args:
        name: Gate name.
        alpha: Forward rate.
        beta: Backward rate.
        power: Gate power.

    Returns:
        The gate.
    """
    return GatingVariable(name=name, power=power, alpha=alpha, beta=beta)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("key", "factory"), PRESETS, ids=PRESET_IDS)
def test_every_preset_lowers_and_round_trips_through_json(key, factory) -> None:
    """Every preset lowers to plain data that JSON preserves exactly.

    Args:
        key: Preset name.
        factory: Preset constructor.
    """
    spec = lower(factory())
    assert spec["version"] == SPEC_VERSION
    assert json.loads(json.dumps(spec)) == spec


@pytest.mark.parametrize(("key", "factory"), PRESETS, ids=PRESET_IDS)
def test_lowered_rates_reproduce_every_preset_gate_exactly(key, factory) -> None:
    """The spec's rates equal each gate's own ``evaluate`` to the last bit.

    Probes a voltage grid, each exp-linear singular point and both sides of
    its tolerance, and Ca²⁺ values from zero to saturating.

    Args:
        key: Preset name.
        factory: Preset constructor.
    """
    neuron = factory()
    spec = json.loads(json.dumps(lower(neuron)))
    clip = spec["exp_clip"]
    gates = neuron.all_gating_variables
    assert [g["name"] for g in spec["gates"]] == [g.name for g in gates]
    for gate, gate_spec in zip(gates, spec["gates"], strict=True):
        assert gate_spec["uses_calcium"] == (
            gate.alpha.uses_calcium or gate.beta.uses_calcium
        )
        ca_values = _CA_VALUES if gate_spec["uses_calcium"] else (0.0,)
        for v in _voltages(spec):
            for ca in ca_values:
                expected = (gate.alpha.evaluate(v, ca), gate.beta.evaluate(v, ca))
                got = _eval_rates(gate_spec["kinetics"], v, ca, clip)
                assert got == expected, (key, gate.name, v, ca)


@pytest.mark.parametrize(("key", "factory"), PRESETS, ids=PRESET_IDS)
def test_lowered_channels_match_the_neuron(key, factory) -> None:
    """Channel conductances, gate references, and reversal potentials match.

    Args:
        key: Preset name.
        factory: Preset constructor.
    """
    neuron = factory()
    spec = lower(neuron)
    gate_names = [g["name"] for g in spec["gates"]]
    assert spec["c_m"] == neuron.C_m
    assert spec["phi"] == neuron.q10_factor
    assert [c["name"] for c in spec["channels"]] == [ch.name for ch in neuron.channels]
    for ch, ch_spec in zip(neuron.channels, spec["channels"], strict=True):
        assert ch_spec["g_max"] == ch.g_max
        assert ch_spec["carries_calcium"] == ch.carries_calcium
        assert [(gate_names[i], p) for i, p in ch_spec["gates"]] == [
            (g.name, g.power) for g in ch.gating_variables
        ]
        if ch.carries_calcium:
            assert ch_spec["e_rev"] is None
        else:
            assert ch_spec["e_rev"] == neuron.reversal_potentials[ch.name]


def test_inf_tau_pairs_lower_to_the_inf_tau_form() -> None:
    """A gate built with ``inf_tau`` carries ``inf`` and ``tau`` once."""
    alpha, beta = inf_tau(Sigmoid(1.0, -40.0, 8.0), 0.2)
    neuron = Neuron(channels=(_channel("K", (_gate("n", alpha, beta),)),))
    kinetics = lower(neuron)["gates"][0]["kinetics"]
    assert kinetics == {
        "form": "inf_tau",
        "inf": {"op": "Sigmoid", "k": 1.0, "vh": -40.0, "s": 8.0},
        "tau": {"op": "Const", "value": 0.2},
    }


def test_mismatched_inf_tau_halves_lower_to_alpha_beta() -> None:
    """Rates from two different ``inf_tau`` pairs are lowered separately."""
    alpha, _ = inf_tau(Sigmoid(1.0, -40.0, 8.0), 0.2)
    _, beta = inf_tau(Sigmoid(1.0, -40.0, 8.0), 0.5)
    neuron = Neuron(channels=(_channel("K", (_gate("n", alpha, beta),)),))
    kinetics = lower(neuron)["gates"][0]["kinetics"]
    assert kinetics["form"] == "alpha_beta"
    for v in (-80.0, -40.0, 0.0):
        assert _eval_rates(kinetics, v, 0.0, SAFE_EXP_CLIP_MAX) == (
            alpha.evaluate(v, 0.0),
            beta.evaluate(v, 0.0),
        )


def test_standalone_inf_tau_nodes_expand_to_division() -> None:
    """``InfTauAlpha``/``InfTauBeta`` outside a pair become ``Div`` trees."""
    inf, tau = Sigmoid(1.0, -40.0, 8.0), Exp(2.0, -60.0, 20.0)
    combined = InfTauAlpha(inf, tau) + 2.0 * InfTauBeta(inf, tau)
    node = lower_expr(combined)
    assert node["a"]["op"] == "Div"
    assert node["b"]["b"] == {
        "op": "Div",
        "a": {"op": "Sub", "a": {"op": "Const", "value": 1.0}, "b": lower_expr(inf)},
        "b": lower_expr(tau),
    }
    for v in (-100.0, -40.0, 30.0):
        assert _eval_node(node, v, 0.0, SAFE_EXP_CLIP_MAX) == combined.evaluate(v, 0.0)


def test_hill_coefficient_is_an_integer() -> None:
    """Hill coefficients are emitted as ints and non-integers are rejected."""
    assert lower_expr(Hill(3e-4, 4)) == {"op": "Hill", "kd": 3e-4, "n": 4}
    assert isinstance(lower_expr(Hill(3e-4, 4.0))["n"], int)  # ty: ignore[invalid-argument-type]
    with pytest.raises(UnloweredRateError, match="whole number"):
        lower_expr(Hill(3e-4, 2.5))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("n", [math.inf, math.nan])
def test_non_finite_hill_coefficient_is_rejected(n) -> None:
    """An infinite or NaN Hill coefficient raises ``UnloweredRateError``.

    Args:
        n: The non-finite coefficient.
    """
    if math.isnan(n):
        # ``Hill`` rejects NaN itself, so bypass its validation.
        hill = object.__new__(Hill)
        object.__setattr__(hill, "kd", 3e-4)
        object.__setattr__(hill, "n", n)
    else:
        hill = Hill(3e-4, n)
    with pytest.raises(UnloweredRateError, match="not finite"):
        lower_expr(hill)


def test_non_integer_gate_power_is_rejected_with_its_location() -> None:
    """A fractional power would be truncated by the integer spec, so it raises."""
    neuron = Neuron(
        channels=(_channel("Kx", (_gate("h", Const(0.1), Const(0.2), 1.5),)),)
    )
    with pytest.raises(UnloweredRateError, match="channel 'Kx', gate 'h'.*power"):
        lower(neuron)


def test_whole_float_gate_power_lowers_as_an_int() -> None:
    """A power of ``3.0`` is emitted as the integer ``3``."""
    neuron = Neuron(
        channels=(_channel("Kx", (_gate("n", Const(0.1), Const(0.2), 3.0),)),)
    )
    power = lower(neuron)["channels"][0]["gates"][0][1]
    assert power == 3
    assert isinstance(power, int)


@pytest.mark.parametrize("power", [math.inf, math.nan])
def test_non_finite_gate_power_is_rejected(power) -> None:
    """An infinite or NaN gate power raises ``UnloweredRateError``.

    Args:
        power: The non-finite power.
    """
    neuron = Neuron(
        channels=(_channel("Kx", (_gate("h", Const(0.1), Const(0.2), power),)),)
    )
    with pytest.raises(UnloweredRateError, match="whole number"):
        lower(neuron)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_expression_parameter_is_rejected(value) -> None:
    """Non-finite parameters cannot be written as strict JSON, so they raise.

    Args:
        value: The non-finite parameter.
    """
    with pytest.raises(UnloweredRateError, match="not finite"):
        lower_expr(Max(Exp(1.0, -60.0, 10.0), Const(value)))


def test_non_finite_neuron_constant_is_rejected() -> None:
    """A non-finite conductance raises ``ValueError`` naming the channel."""
    channel = _channel("Kx", (_gate("n", Const(0.1), Const(0.2)),))
    object.__setattr__(channel, "g_max", math.inf)
    neuron = Neuron(channels=(channel,))
    with pytest.raises(ValueError, match="'Kx' g_max"):
        lower(neuron)


def test_callable_adapter_is_rejected_with_its_location() -> None:
    """A plain-callable rate cannot be lowered; the error names channel and gate."""
    opaque = VoltageOnlyFn(lambda v, ca: 0.1)
    neuron = Neuron(
        channels=(_channel("Kx", (_gate("w", opaque, ExpLinear(0.01, -55.0, 10.0)),)),)
    )
    with pytest.raises(UnloweredRateError, match="channel 'Kx', gate 'w'"):
        lower(neuron)


def test_callable_adapter_inside_an_expression_is_rejected() -> None:
    """An adapter nested inside an otherwise standard expression is rejected."""
    rate = Exp(1.0, -60.0, 10.0) + VoltageOnlyFn(lambda v, ca: 0.1)
    with pytest.raises(UnloweredRateError, match="VoltageOnlyFn"):
        lower_expr(rate)


def test_subclass_of_a_builtin_form_is_rejected() -> None:
    """A subclass may change ``evaluate``, so it is not lowered as its base."""

    @dataclass(frozen=True)
    class Doubled(Exp):
        """An ``Exp`` that doubles its value."""

        def evaluate(self, V: float, ca_i: float) -> float:
            """Return twice the exponential.

            Args:
                V: Membrane voltage in mV.
                ca_i: Intracellular Ca²⁺ in mM.

            Returns:
                The doubled value.
            """
            return 2.0 * super().evaluate(V, ca_i)

    with pytest.raises(UnloweredRateError, match="Doubled"):
        lower_expr(Doubled(1.0, -60.0, 10.0))


def test_duplicate_gate_names_across_channels_are_rejected() -> None:
    """Gates sharing a name would share state in Python; the spec refuses them."""
    gate = _gate("n", Const(0.1), Const(0.2))
    neuron = Neuron(channels=(_channel("K1", (gate,)), _channel("K2", (gate,))))
    with pytest.raises(ValueError, match="unique across channels"):
        lower(neuron)


def test_neuron_without_calcium_has_no_calcium_blocks() -> None:
    """Without Ca²⁺ channels or dynamics, both calcium entries are ``None``."""
    spec = lower(nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]())
    assert spec["calcium_nernst"] is None
    assert spec["calcium_dynamics"] is None


def test_calcium_channel_without_dynamics_keeps_the_nernst_constants() -> None:
    """Held-constant Ca²⁺ still needs the live E_Ca constants."""
    gate = _gate("d", Sigmoid(1.0, -20.0, 5.0), Sigmoid(1.0, -20.0, -5.0))
    neuron = Neuron(channels=(_channel("CaL", (gate,), calcium=True),))
    spec = lower(neuron)
    assert spec["calcium_dynamics"] is None
    assert spec["calcium_nernst"] is not None
    assert spec["channels"][0]["e_rev"] is None


def test_calcium_blocks_reproduce_e_ca_and_dynamics() -> None:
    """The documented E_Ca formula and Ca²⁺ ODE match the library's."""
    cd = CalciumDynamics(alpha_ca=0.002, tau_ca=80.0, ca_rest=5e-5)
    gate = _gate("d", Sigmoid(1.0, -20.0, 5.0), Sigmoid(1.0, -20.0, -5.0))
    neuron = Neuron(
        channels=(_channel("CaL", (gate,), calcium=True),), calcium_dynamics=cd
    )
    spec = lower(neuron)
    nernst, dynamics = spec["calcium_nernst"], spec["calcium_dynamics"]
    assert dynamics == {"alpha_ca": 0.002, "tau_ca": 80.0, "ca_rest": 5e-5}
    for ca in (0.0, 1e-12, 5e-5, 1e-3):
        expected = nernst_potential(
            2, neuron.T, neuron.Ca_out, max(ca, _CA_NERNST_FLOOR)
        )
        got = (
            (nernst["gas_constant"] * nernst["temperature_k"] / (2 * nernst["faraday"]))
            * 1000
            * math.log(nernst["ca_out"] / max(ca, nernst["floor_mm"]))
        )
        # ``nernst_potential`` uses ``np.log``, whose SIMD paths may round
        # differently from libm's ``math.log`` in the last bit.
        assert got == pytest.approx(expected, rel=1e-15, abs=0.0)
        i_ca = -3.0
        assert -dynamics["alpha_ca"] * i_ca - (ca - dynamics["ca_rest"]) / dynamics[
            "tau_ca"
        ] == cd.derivative(i_ca, ca)


def test_numerical_bounds_match_the_integrator() -> None:
    """The spec's clip bound covers exp and cosh, and V bounds match RK4's."""
    spec = lower(nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]())
    assert spec["exp_clip"] == SAFE_EXP_CLIP_MAX == -SAFE_EXP_CLIP_MIN == SAFE_COSH_CLIP
    assert spec["v_clamp_mv"] == [-150.0, 150.0]


def test_state_bounds_match_the_integrator() -> None:
    """``gate_bounds`` and ``ca_floor_mm`` reproduce the integrator's clipping."""
    spec = lower(nc.NEURON_PRESETS["Squid Giant Axon (Classic HH)"]())
    lo, hi = spec["gate_bounds"]
    state = {"a": -0.25, "b": 0.5, "c": 1.25}
    assert _clip_state(state) == {k: max(lo, min(hi, x)) for k, x in state.items()}
    assert spec["ca_floor_mm"] == 0.0
