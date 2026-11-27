"""Lower a neuron to a flat, JSON-serializable model spec.

A compiled simulation backend cannot call Python rate functions, so it needs
the model as plain data.  :func:`lower` produces that description: every
gate's kinetics as a tree of standard forms, every channel's conductance,
gate powers, and reversal potential, and the constants of the intracellular
Ca²⁺ dynamics.  Everything that is constant over a simulation (the Q10
factor, the reversal potentials of non-Ca²⁺ channels) is resolved here, so a
backend needs no electrochemistry beyond the live Ca²⁺ Nernst potential.

The spec describes the model only.  The initial state (voltage, gate values,
Ca²⁺) and the stimulus are passed to the backend per simulation.

Format (version :data:`SPEC_VERSION`)::

    {
      "version": 3,
      "c_m": 1.0,                      # µF/cm²
      "phi": 1.0,                      # Q10 factor multiplying every rate
      "exp_clip": 100.0,               # |x| bound for exp and cosh arguments
      "v_clamp_mv": [-150.0, 150.0],   # voltage bounds at every RK4 sub-step
      "gate_bounds": [0.0, 1.0],       # gate bounds after every full RK4 step
      "ca_floor_mm": 0.0,              # Ca²⁺ floor after every full RK4 step
      "gates": [
        {"name": "m", "uses_calcium": false,
         "kinetics": {"form": "alpha_beta", "alpha": EXPR, "beta": EXPR}},
        {"name": "q", "uses_calcium": true,
         "kinetics": {"form": "inf_tau", "inf": EXPR, "tau": EXPR}},
      ],
      "channels": [
        {"name": "Na", "g_max": 120.0, "gates": [[0, 3], [1, 1]],
         "carries_calcium": false, "e_rev": 50.0},
        {"name": "CaL", "g_max": 1.0, "gates": [[2, 2]],
         "carries_calcium": true, "e_rev": null},
      ],
      "calcium_nernst": {"ca_out": 2.0, "temperature_k": 310.15,
                         "floor_mm": 1e-9, "gas_constant": 8.314,
                         "faraday": 96485.0},
      "calcium_dynamics": {"alpha_ca": 0.002, "tau_ca": 80.0,
                           "ca_rest": 5e-5},
    }

Each ``EXPR`` is a node ``{"op": NAME, ...}`` whose remaining keys are the
fields of the :mod:`neuroclamp.rates` class of the same name:

- Leaves: ``Const`` (``value``), ``Exp``, ``Sigmoid``, ``Sech``
  (``k``, ``vh``, ``s``), ``ExpLinear`` (``k``, ``vh``, ``s``, ``tol``),
  ``Hill`` (``kd``, integer ``n``).
- Combinators: ``Add``, ``Sub``, ``Mul``, ``Div``, ``Max`` (``a``, ``b``).

A gate whose rates are the pair built by :func:`~neuroclamp.rates.inf_tau`
lowers to the ``inf_tau`` form, so a backend evaluates ``inf`` and ``tau``
once for both rates: ``alpha = inf / tau`` and ``beta = (1 - inf) / tau``.
Each gate obeys ``dx/dt = phi * (alpha * (1 - x) - beta * x)``.  The two
clamp modes of the Python integrator round this differently, and a backend
that must match Python to the last bit follows the same order:

- Current clamp: ``phi * (alpha * (1 - x) - beta * x)``.
- Voltage clamp: ``(phi * alpha) * (1 - x) - (phi * beta) * x``, with
  ``phi * alpha`` and ``phi * beta`` each rounded first.

The state advances by classic RK4.  In current clamp, the voltage passed to
each intermediate stage is clamped to ``v_clamp_mv``.  After each full step,
every gate is clipped to ``gate_bounds`` and Ca²⁺ is floored at
``ca_floor_mm``; the intermediate stages are not clipped.

A channel's current is ``g_max * prod(x_i ** p_i) * (V - E)``.  ``gates``
lists ``[index into "gates", power]`` pairs.  ``e_rev`` is fixed for
non-Ca²⁺ channels and ``null`` for Ca²⁺ channels, whose ``E_Ca`` is the
Nernst potential from the live intracellular Ca²⁺, in mV and evaluated in
this order: ``(gas_constant * temperature_k / (2 * faraday)) * 1000 *
ln(ca_out / max(ca_i, floor_mm))``.  ``calcium_nernst`` is present exactly when
some channel carries Ca²⁺.  ``calcium_dynamics`` is ``null`` when Ca²⁺ is
held constant, and otherwise gives
``d[Ca]/dt = -alpha_ca * I_Ca - (ca_i - ca_rest) / tau_ca``.  With
``calcium_dynamics`` null, Ca²⁺ stays at the initial value passed with the
state; the Python integrator starts it at the neuron's ``Ca_in``.

Every number in the spec is finite, so the spec is strict JSON.

Only the built-in expression forms can be lowered.  A gate that uses a plain
callable adapter (:class:`~neuroclamp.rates.VoltageOnlyFn`,
:class:`~neuroclamp.rates.CalciumDependentFn`), a custom
:class:`~neuroclamp.rates.Rate`, or a subclass of a built-in form makes
:func:`lower` raise :class:`UnloweredRateError`; such neurons run only on the
Python backend.  A whole-number gate power or Hill coefficient is required
for the same reason: the spec stores both as integers.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from .channels.base import _CA_NERNST_FLOOR, GatingVariable
from .clamp_simulations import _V_CLAMP_MAX, _V_CLAMP_MIN
from .electrochemistry import F, R
from .neuron import Neuron
from .rates import (
    Add,
    Const,
    Div,
    Exp,
    ExpLinear,
    Hill,
    InfTauAlpha,
    InfTauBeta,
    Max,
    Mul,
    Rate,
    Sech,
    Sigmoid,
    Sub,
    _Binary,
)
from .utils import SAFE_EXP_CLIP_MAX

#: Version of the spec format :func:`lower` produces.
SPEC_VERSION = 3

# Classes lowered to an ``op`` node of the same name.  Membership is by exact
# type: a subclass may override ``evaluate``, so lowering it as its base form
# could silently change the model.
_LEAF_OPS: frozenset[type] = frozenset({Const, Exp, Sigmoid, ExpLinear, Sech, Hill})
_BINARY_OPS: frozenset[type] = frozenset({Add, Sub, Mul, Div, Max})


class UnloweredRateError(TypeError):
    """A rate cannot be expressed in the model spec.

    Raised by :func:`lower` for a rate that is not built from the standard
    expression forms, such as a plain callable wrapped in
    :class:`~neuroclamp.rates.VoltageOnlyFn`.
    """


def lower_expr(rate: Rate) -> dict[str, Any]:
    """Lower one rate expression to its spec node.

    Standalone :class:`~neuroclamp.rates.InfTauAlpha` and
    :class:`~neuroclamp.rates.InfTauBeta` nodes (outside a gate's
    ``inf_tau`` pair) expand to the ``Div`` and ``Sub`` nodes that perform
    the same operations.

    Args:
        rate: The rate to lower.

    Returns:
        A JSON-serializable ``{"op": ..., ...}`` dict.

    Raises:
        UnloweredRateError: If the rate, or any node inside it, is not one of
            the built-in expression forms, has a non-finite parameter, or is a
            ``Hill`` whose coefficient is not a whole number.
    """
    cls = type(rate)
    if cls in _LEAF_OPS:
        node: dict[str, Any] = {"op": cls.__name__}
        for f in fields(rate):  # ty: ignore[invalid-argument-type]
            value = float(getattr(rate, f.name))
            if not math.isfinite(value):
                raise UnloweredRateError(
                    f"{cls.__name__}.{f.name} = {value!r} is not finite and"
                    " cannot be lowered to strict JSON"
                )
            node[f.name] = value
        if isinstance(rate, Hill):
            node["n"] = _integer_hill_coefficient(rate.n)
        return node
    if cls in _BINARY_OPS:
        assert isinstance(rate, _Binary)
        return {"op": cls.__name__, "a": lower_expr(rate.a), "b": lower_expr(rate.b)}
    if cls is InfTauAlpha:
        assert isinstance(rate, InfTauAlpha)
        return {"op": "Div", "a": lower_expr(rate.a), "b": lower_expr(rate.b)}
    if cls is InfTauBeta:
        assert isinstance(rate, InfTauBeta)
        inf, tau = lower_expr(rate.a), lower_expr(rate.b)
        one_minus_inf = {"op": "Sub", "a": {"op": "Const", "value": 1.0}, "b": inf}
        return {"op": "Div", "a": one_minus_inf, "b": tau}
    raise UnloweredRateError(
        f"{cls.__name__} is not a built-in rate expression, so it cannot be"
        " lowered; build the rate from the forms in neuroclamp.rates"
    )


def _whole_number(value: float, what: str) -> int:
    """Return a value the spec stores as an integer, checking that it is one.

    Args:
        value: The number, such as a Hill coefficient or a gate power.
        what: Its description for the error message.

    Returns:
        ``value`` as an ``int``.

    Raises:
        UnloweredRateError: If ``value`` is a bool, infinite, NaN, or not a
            whole number.
    """
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or float(value) != int(value)
    ):
        raise UnloweredRateError(
            f"{what} {value!r} is not a whole number and cannot be lowered"
        )
    return int(value)


def _integer_hill_coefficient(n: float) -> int:
    """Return a Hill coefficient as an ``int``, as the spec requires.

    Args:
        n: The coefficient.

    Returns:
        ``n`` as an ``int``.

    Raises:
        UnloweredRateError: If ``n`` is not a whole number.
    """
    return _whole_number(n, "Hill coefficient")


def _finite(value: float, what: str) -> float:
    """Return a neuron scalar as a ``float``, checking that it is finite.

    Args:
        value: The scalar.
        what: Its description for the error message.

    Returns:
        ``value`` as a ``float``.

    Raises:
        ValueError: If ``value`` is infinite or NaN, which strict JSON cannot
            represent.
    """
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} = {result!r} is not finite and cannot be lowered")
    return result


def _lower_kinetics(gate: GatingVariable) -> dict[str, Any]:
    """Lower a gate's rates, keeping the steady-state/time-constant form.

    Args:
        gate: The gate.

    Returns:
        ``{"form": "inf_tau", "inf": ..., "tau": ...}`` when the gate's rates
        are an :func:`~neuroclamp.rates.inf_tau` pair on the same ``inf``
        and ``tau``, else ``{"form": "alpha_beta", "alpha": ..., "beta": ...}``.
    """
    alpha, beta = gate.alpha, gate.beta
    if (
        type(alpha) is InfTauAlpha
        and type(beta) is InfTauBeta
        and alpha.a == beta.a
        and alpha.b == beta.b
    ):
        inf, tau = lower_expr(alpha.a), lower_expr(alpha.b)
        return {"form": "inf_tau", "inf": inf, "tau": tau}
    return {"form": "alpha_beta", "alpha": lower_expr(alpha), "beta": lower_expr(beta)}


def lower(neuron: Neuron) -> dict[str, Any]:
    """Flatten a neuron into the model spec a compiled backend consumes.

    See the module docstring for the format.  The result contains only
    dicts, lists, strings, numbers, booleans, and ``None``, so it survives a
    round trip through :func:`json.dumps` and :func:`json.loads` unchanged.

    Args:
        neuron: The neuron to lower.

    Returns:
        The model spec.

    Raises:
        UnloweredRateError: If any gate's rate is not built from the standard
            expression forms, or a gate's power is not a whole number.  The
            message names the channel and gate.
        ValueError: If two channels have gates with the same name (the Python
            integrator keys gate state by name, so such gates would share one
            state variable, which the spec cannot express), or if a
            conductance, reversal potential, or other neuron constant is not
            finite.
    """
    names = [gate.name for gate in neuron.all_gating_variables]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"gate names must be unique across channels, got {duplicates}")
    gate_specs: list[dict[str, Any]] = []
    channel_gates: list[list[list[int]]] = []
    for channel in neuron.channels:
        channel_gates.append([])
        for gate in channel.gating_variables:
            try:
                power = _whole_number(gate.power, "gate power")
                kinetics = _lower_kinetics(gate)
            except UnloweredRateError as err:
                raise UnloweredRateError(
                    f"channel {channel.name!r}, gate {gate.name!r}: {err}"
                ) from err
            channel_gates[-1].append([len(gate_specs), power])
            gate_specs.append(
                {
                    "name": gate.name,
                    "uses_calcium": gate.alpha.uses_calcium or gate.beta.uses_calcium,
                    "kinetics": kinetics,
                }
            )
    any_calcium_channel = any(ch.carries_calcium for ch in neuron.channels)
    cd = neuron.calcium_dynamics
    return {
        "version": SPEC_VERSION,
        "c_m": _finite(neuron.C_m, "C_m"),
        "phi": _finite(neuron.q10_factor, "q10_factor"),
        "exp_clip": float(SAFE_EXP_CLIP_MAX),
        "v_clamp_mv": [float(_V_CLAMP_MIN), float(_V_CLAMP_MAX)],
        "gate_bounds": [0.0, 1.0],
        "ca_floor_mm": 0.0,
        "gates": gate_specs,
        "channels": [
            {
                "name": ch.name,
                "g_max": _finite(ch.g_max, f"channel {ch.name!r} g_max"),
                "gates": gates,
                "carries_calcium": ch.carries_calcium,
                "e_rev": None
                if ch.carries_calcium
                else _finite(
                    neuron.reversal_potentials[ch.name], f"channel {ch.name!r} e_rev"
                ),
            }
            for ch, gates in zip(neuron.channels, channel_gates, strict=True)
        ],
        "calcium_nernst": {
            "ca_out": _finite(neuron.Ca_out, "Ca_out"),
            "temperature_k": _finite(neuron.T, "T"),
            "floor_mm": _CA_NERNST_FLOOR,
            "gas_constant": R,
            "faraday": float(F),
        }
        if any_calcium_channel
        else None,
        "calcium_dynamics": None
        if cd is None
        else {
            "alpha_ca": _finite(cd.alpha_ca, "alpha_ca"),
            "tau_ca": _finite(cd.tau_ca, "tau_ca"),
            "ca_rest": _finite(cd.ca_rest, "ca_rest"),
        },
    }
