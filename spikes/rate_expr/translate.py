"""Map every existing rate object in ``patch_sim.channels`` to an expression tree."""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from collections.abc import Callable

from exprs import (
    Const,
    Exp,
    ExpLinear,
    Expr,
    Hill,
    Max,
    Sech,
    Sigmoid,
    boltzmann_cosh,
    inf_tau,
    traub_miles,
)

import patch_sim.channels as chpkg
from patch_sim.channels import (
    GatingVariable,
    hh52,
    pospischil,
    purkinje,
    snc,
    stn,
    thalamic,
    trn,
)
from patch_sim.channels import auxiliary as aux
from patch_sim.channels import mainen_sejnowski as ms
from patch_sim.electrochemistry import BoltzmannCoshRate
from patch_sim.neuron import Neuron
from patch_sim.rates import Rate

_REGISTRY: dict[int, Expr] = {}


def _reg(original: Rate, tree: Expr) -> None:
    """Register the tree that replaces an existing rate object.

    Args:
        original: The existing module-level rate instance.
        tree: Its expression-tree equivalent.
    """
    _REGISTRY[id(original)] = tree


def _reg_pair(alpha: Rate, beta: Rate, pair: tuple[Expr, Expr]) -> None:
    """Register an (alpha, beta) pair at once.

    Args:
        alpha: Existing alpha rate.
        beta: Existing beta rate.
        pair: The (alpha, beta) trees.
    """
    _reg(alpha, pair[0])
    _reg(beta, pair[1])


# --- HH52
_reg(hh52.alpha_n, ExpLinear(0.01, -55, 10))
_reg(hh52.beta_n, Exp(0.125, -65, -80))
_reg(hh52.alpha_m, ExpLinear(0.1, -40, 10))
_reg(hh52.beta_m, Exp(4.0, -65, -18))
_reg(hh52.alpha_h, Exp(0.07, -65, -20))
_reg(hh52.beta_h, Sigmoid(1.0, -35, 10))

# --- Traub-Miles families
for _mod, _prefix, _vt in [
    (pospischil, "pospischil", pospischil.POSPISCHIL_VT),
    (thalamic, "thalamic_relay", thalamic.THALAMIC_RELAY_VT),
    (trn, "trn", trn.TRN_VT),
    (purkinje, "purkinje", purkinje.PURKINJE_VT),
    (snc, "dopaminergic", snc.DOPAMINERGIC_VT),
]:
    for _name, _tree in traub_miles(_vt).items():
        _reg(getattr(_mod, f"{_prefix}_{_name}"), _tree)

# --- auxiliary
_reg(aux._alpha_r, Exp(1.0, 14.59 / -0.086, -1 / 0.086))
_reg(aux._beta_r, Exp(1.0, 1.87 / 0.0701, 1 / 0.0701))
_reg(aux._alpha_a, ExpLinear(0.02, -51.9, 10, tol=1e-7))
_reg(aux._beta_a, ExpLinear(-0.0175, -24.9, -10, tol=1e-7))
_reg(aux._alpha_b, Exp(0.0016, -73, -18))
_reg(aux._beta_b, Sigmoid(0.05, -13, 10))
_reg_pair(
    aux._alpha_hr,
    aux._beta_hr,
    inf_tau(Sigmoid(1.0, -55, -8), Sigmoid(150.0, -40, 10) + 1.0),
)
_reg_pair(
    aux._alpha_q,
    aux._beta_q,
    inf_tau(
        Hill(0.001, 1) * Sigmoid(1.0, -20, 10),
        Max(Sech(10.0, -20, 20), Const(1.0)),
    ),
)
_reg_pair(aux._alpha_qSK, aux._beta_qSK, inf_tau(Hill(0.0003, 4), 10.0))

# --- TRN T-type inactivation
_reg_pair(
    trn._alpha_ft_trn,
    trn._beta_ft_trn,
    inf_tau(Sigmoid(1.0, -80, -9), 20.0 + Sigmoid(180.0, -50, 5)),
)

# --- Mainen-Sejnowski
_pa = ms._MAINEN_SEJNOWSKI_ALPHA_PREFACTOR
_pb = ms._MAINEN_SEJNOWSKI_BETA_PREFACTOR
_reg(ms.mainen_sejnowski_alpha_n, ExpLinear(_pa, 25.0, 9.0))
_reg(ms.mainen_sejnowski_beta_n, ExpLinear(-_pb, 25.0, -9.0))
_reg(ms.mainen_sejnowski_alpha_m, ExpLinear(ms._MAINEN_SEJNOWSKI_NA_RA, -35.0, 9.0))
_reg(ms.mainen_sejnowski_beta_m, ExpLinear(-ms._MAINEN_SEJNOWSKI_NA_RB, -35.0, -9.0))
_h_rate = ExpLinear(ms._MAINEN_SEJNOWSKI_NA_RD, -50.0, 5.0) + ExpLinear(
    -ms._MAINEN_SEJNOWSKI_NA_RG, -75.0, -5.0
)
_reg_pair(
    ms.mainen_sejnowski_alpha_h,
    ms.mainen_sejnowski_beta_h,
    inf_tau(Sigmoid(1.0, -65.0, -6.2), 1.0 / _h_rate),
)

# --- STN
_reg_pair(stn._stn_alpha_m, stn._stn_beta_m, inf_tau(Sigmoid(1.0, -40, 8), 0.2))
_reg_pair(
    stn._stn_alpha_h,
    stn._stn_beta_h,
    inf_tau(
        Sigmoid(1.0, -45.5, -6.4),
        1.0 / (Exp(0.128, -38, -18) + Sigmoid(4.0, -15, 5)),
    ),
)


def translate(rate: Rate) -> Expr | None:
    """Return the expression tree for an existing rate object.

    Args:
        rate: Any rate attached to a gating variable in ``patch_sim``.

    Returns:
        The equivalent tree, or ``None`` if the rate is not covered.
    """
    if isinstance(rate, BoltzmannCoshRate):
        alpha, beta = boltzmann_cosh(
            rate.half,
            rate.slope,
            rate.tau_scale,
            rate.tau_floor,
            inverted=rate.inverted,
            tau_cosh_scale=rate.cosh_scale,
            tau_rate=rate.tau_rate,
        )
        return alpha if rate.is_alpha else beta
    if isinstance(rate, trn._ShiftedTrnAlphaH):
        return traub_miles(trn.TRN_VT + rate.h_v_half_shift)["alpha_h"]
    if isinstance(rate, trn._ShiftedTrnBetaH):
        return traub_miles(trn.TRN_VT + rate.h_v_half_shift)["beta_h"]
    return _REGISTRY.get(id(rate))


def all_module_rates() -> list[tuple[str, Rate]]:
    """Find every Rate instance bound at module level in ``patch_sim.channels``.

    Returns:
        ``(qualified_name, rate)`` pairs, deduplicated by object identity.
    """
    found: dict[int, tuple[str, Rate]] = {}
    for info in pkgutil.iter_modules(chpkg.__path__):
        mod = importlib.import_module(f"patch_sim.channels.{info.name}")
        for key, value in vars(mod).items():
            if isinstance(value, Rate):
                found.setdefault(id(value), (f"{info.name}.{key}", value))
    return list(found.values())


def _identity(rate: Rate) -> Rate:
    """Return the rate unchanged.

    Args:
        rate: A rate object.

    Returns:
        The same object.
    """
    return rate


def to_tree_neuron(neuron: Neuron, wrap: Callable[[Expr], Rate] = _identity) -> Neuron:
    """Rebuild a neuron with every gate's rates replaced by expression trees.

    Args:
        neuron: A neuron built from the existing channel factories.
        wrap: Optional transform applied to each tree (e.g. codegen).

    Returns:
        A new neuron with identical channels except for the rate objects.

    Raises:
        KeyError: If any gate uses a rate with no registered tree.
    """
    channels = []
    for ch in neuron.channels:
        gates = []
        for gv in ch.gating_variables:
            alpha, beta = translate(gv.alpha), translate(gv.beta)
            if alpha is None or beta is None:
                raise KeyError(f"{ch.name}.{gv.name}: rate not translated")
            gates.append(GatingVariable(gv.name, gv.power, wrap(alpha), wrap(beta)))
        channels.append(dataclasses.replace(ch, gating_variables=tuple(gates)))
    return dataclasses.replace(neuron, channels=tuple(channels))
