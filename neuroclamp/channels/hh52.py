"""Hodgkin-Huxley (1952) squid axon channel factories and rate functions.

This module provides the six classic HH rates as module-level rate
expressions (see :mod:`neuroclamp.rates`) and four factory functions that
bundle them into IonChannel objects.  Each rate is a callable
``(V, ca_i) -> float`` in 1/ms and does not depend on ``ca_i``.

The passive leak is split into two non-specific conductances:
- :func:`make_na_leak_channel`: Na⁺ leak, reversal via Nernst equation for Na⁺.
- :func:`make_k_leak_channel`: K⁺ leak, reversal via Nernst equation for K⁺.

This mirrors the biophysical reality of background channels (TREK/TRAAK K⁺
channels + persistent Na⁺ leak) and removes the unphysiological dependence on
intracellular [Cl⁻] that arose when using a single chloride-Nernst leak.
"""

from ..rates import SINGULARITY_THRESHOLD, Exp, ExpLinear, Sigmoid
from .base import GatingVariable, IonChannel, IonSpecies, NernstSpec

__all__ = [
    "SINGULARITY_THRESHOLD",
    "alpha_n",
    "beta_n",
    "alpha_m",
    "beta_m",
    "alpha_h",
    "beta_h",
    "make_na_channel",
    "make_k_channel",
    "make_na_leak_channel",
    "make_k_leak_channel",
]


#: Forward rate for K⁺ activation gate n:
#: ``0.01 (V + 55) / (1 − exp(−(V + 55) / 10))``, limit 0.1 at V = −55 mV.
alpha_n = ExpLinear(0.01, -55.0, 10.0)

#: Backward rate for K⁺ activation gate n: ``0.125 exp(−(V + 65) / 80)``.
beta_n = Exp(0.125, -65.0, -80.0)

#: Forward rate for Na⁺ activation gate m:
#: ``0.1 (V + 40) / (1 − exp(−(V + 40) / 10))``, limit 1.0 at V = −40 mV.
alpha_m = ExpLinear(0.1, -40.0, 10.0)

#: Backward rate for Na⁺ activation gate m: ``4 exp(−(V + 65) / 18)``.
beta_m = Exp(4.0, -65.0, -18.0)

#: Forward rate for Na⁺ inactivation gate h: ``0.07 exp(−(V + 65) / 20)``.
alpha_h = Exp(0.07, -65.0, -20.0)

#: Backward rate for Na⁺ inactivation gate h: ``1 / (1 + exp(−(V + 35) / 10))``.
beta_h = Sigmoid(1.0, -35.0, 10.0)


def make_na_channel(g_max: float) -> IonChannel:
    """Create the fast sodium channel (Na⁺).

    Uses the classic Hodgkin-Huxley kinetics: activation gate *m* (power 3)
    and inactivation gate *h* (power 1).  The reversal potential is computed
    dynamically via the Nernst equation for Na⁺.

    Gating variable names are ``"m"`` and ``"h"``, matching the simulation
    result fields used throughout the simulator.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~neuroclamp.channels.IonChannel` representing the fast Na⁺
        channel.
    """
    return IonChannel(
        name="Na",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="m", power=3, alpha=alpha_m, beta=beta_m),
            GatingVariable(name="h", power=1, alpha=alpha_h, beta=beta_h),
        ),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_k_channel(g_max: float) -> IonChannel:
    """Create the delayed rectifier potassium channel (K⁺).

    Uses the classic HH activation gate *n* (power 4).  The reversal potential
    is computed dynamically via the Nernst equation for K⁺.

    The gating variable name is ``"n"``, matching the simulation result field
    used throughout the simulator.

    Args:
        g_max: Maximum conductance in mS/cm².

    Returns:
        An :class:`~neuroclamp.channels.IonChannel` representing the delayed
        rectifier K⁺ channel.
    """
    return IonChannel(
        name="K",
        g_max=g_max,
        gating_variables=(
            GatingVariable(name="n", power=4, alpha=alpha_n, beta=beta_n),
        ),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )


def make_na_leak_channel(g_max: float) -> IonChannel:
    """Create the sodium leak channel (Na⁺ background conductance).

    No gating variables — conductance is always *g_max*.  The reversal
    potential is computed dynamically via the Nernst equation for Na⁺,
    representing persistent sodium leak channels (e.g. NALCN).

    Args:
        g_max: Maximum (and constant) conductance in mS/cm².

    Returns:
        An :class:`~neuroclamp.channels.IonChannel` representing the Na⁺ leak
        channel with current field ``INaL``.
    """
    return IonChannel(
        name="NaL",
        g_max=g_max,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.SODIUM),
    )


def make_k_leak_channel(g_max: float) -> IonChannel:
    """Create the potassium leak channel (K⁺ background conductance).

    No gating variables — conductance is always *g_max*.  The reversal
    potential is computed dynamically via the Nernst equation for K⁺,
    representing two-pore-domain K⁺ background channels (e.g. TREK, TRAAK).

    Args:
        g_max: Maximum (and constant) conductance in mS/cm².

    Returns:
        An :class:`~neuroclamp.channels.IonChannel` representing the K⁺ leak
        channel with current field ``IKL``.
    """
    return IonChannel(
        name="KL",
        g_max=g_max,
        gating_variables=(),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
