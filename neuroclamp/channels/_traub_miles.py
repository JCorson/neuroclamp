"""Shared Traub-Miles (1991) rate expressions.

The cell-type families (Pospischil, thalamic relay, TRN, Purkinje, SNc
dopaminergic) use the same six Traub-Miles forms and differ only in the
voltage threshold ``vt``.  :func:`traub_miles_rates` builds the six rates for
a given ``vt`` as rate expressions (see :mod:`neuroclamp.rates`).
"""

from typing import NamedTuple

from ..rates import Exp, ExpLinear, Sigmoid


class TraubMilesRates(NamedTuple):
    """The six Traub-Miles rates for one voltage threshold, in 1/ms.

    Attributes:
        alpha_m: Na⁺ activation forward rate,
            ``0.32 (V − vt − 13) / (1 − exp(−(V − vt − 13) / 4))``; limit 1.28
            at ``V = vt + 13``.
        beta_m: Na⁺ activation backward rate,
            ``0.28 (V − vt − 40) / (exp((V − vt − 40) / 5) − 1)``; limit 1.4 at
            ``V = vt + 40``.
        alpha_h: Na⁺ inactivation forward rate,
            ``0.128 exp(−(V − vt − 17) / 18)``.
        beta_h: Na⁺ inactivation backward rate,
            ``4 / (1 + exp(−(V − vt − 40) / 5))``.
        alpha_n: K⁺ activation forward rate,
            ``0.032 (V − vt − 15) / (1 − exp(−(V − vt − 15) / 5))``; limit 0.16
            at ``V = vt + 15``.
        beta_n: K⁺ activation backward rate, ``0.5 exp(−(V − vt − 10) / 40)``.
    """

    alpha_m: ExpLinear
    beta_m: ExpLinear
    alpha_h: Exp
    beta_h: Sigmoid
    alpha_n: ExpLinear
    beta_n: Exp


def traub_miles_rates(vt: float) -> TraubMilesRates:
    """Build the six Traub-Miles rates for a cell-type voltage threshold.

    Args:
        vt: Cell-type voltage threshold in mV.

    Returns:
        The rates as expressions; see :class:`TraubMilesRates` for each form.
    """
    return TraubMilesRates(
        alpha_m=ExpLinear(0.32, vt + 13.0, 4.0),
        # 0.28 x / (exp(x / 5) − 1) is the mirrored exp-linear form, so both
        # k and s change sign.
        beta_m=ExpLinear(-0.28, vt + 40.0, -5.0),
        alpha_h=Exp(0.128, vt + 17.0, -18.0),
        beta_h=Sigmoid(4.0, vt + 40.0, 5.0),
        alpha_n=ExpLinear(0.032, vt + 15.0, 5.0),
        beta_n=Exp(0.5, vt + 10.0, -40.0),
    )
