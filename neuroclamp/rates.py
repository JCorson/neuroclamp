"""Gating-variable rate-function class hierarchy.

Every alpha/beta rate attached to a :class:`~neuroclamp.channels.GatingVariable`
is an instance of :class:`Rate`.  :attr:`Rate.uses_calcium` reports whether a
rate's output depends on ca_i.  The voltage-clamp optimizer tabulates rates
that do not read ca_i over the prescribed voltage protocol; Ca²⁺-dependent
rates are evaluated scalar-wise in the RK4 loop.  The abstract subclasses
:class:`VoltageOnlyRate` and :class:`CalciumDependentRate` fix
``uses_calcium`` to ``False`` and ``True`` respectively.

Concrete adapters :class:`VoltageOnlyFn` and :class:`CalciumDependentFn` wrap
plain ``(V, ca_i) -> float`` callables so existing module-level rate functions
can be promoted into the type hierarchy without changing their bodies.  Both
adapters are frozen dataclasses, matching the picklability contract relied on
by :func:`~neuroclamp.clamp_simulations.simulate_batch`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


class Rate(ABC):
    """Abstract base class for gating-variable rate functions."""

    @property
    def uses_calcium(self) -> bool:
        """Whether the rate's output depends on ca_i.

        Rates that do not read ca_i are tabulated over the prescribed voltage
        protocol in voltage clamp.  The base class conservatively reports
        ``True``, so a rate is only tabulated when it says it is safe to.

        Returns:
            ``True`` if the rate may depend on ca_i.
        """
        return True

    @abstractmethod
    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the rate.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The rate in 1/ms.
        """


class VoltageOnlyRate(Rate):
    """Base class for rates whose output is independent of ca_i.

    Subclasses are safe to tabulate over a prescribed voltage protocol.
    """

    @property
    def uses_calcium(self) -> bool:
        """Report that the rate does not read ca_i.

        Returns:
            Always ``False``.
        """
        return False


class CalciumDependentRate(Rate):
    """Base class for rates whose output depends on ca_i.

    Subclasses must be evaluated scalar-wise at the current (V, ca_i).
    """

    @property
    def uses_calcium(self) -> bool:
        """Report that the rate reads ca_i.

        Returns:
            Always ``True``.
        """
        return True


@dataclass(frozen=True)
class VoltageOnlyFn(VoltageOnlyRate):
    """Adapter wrapping a plain callable as a :class:`VoltageOnlyRate`.

    .. warning::
        Wrapping a function that *does* depend on ``ca_i`` in
        ``VoltageOnlyFn`` causes silent numerical errors: the voltage-clamp
        precompute pass evaluates the rate with ``ca_i=0.0`` and caches the
        result.  Use :class:`CalciumDependentFn` for any rate that reads
        ``ca_i``.

    Attributes:
        fn: Plain ``(V, ca_i) -> float`` callable.  Must be independent of
            ``ca_i`` — wrap Ca²⁺-dependent functions with
            :class:`CalciumDependentFn` instead.
    """

    fn: Callable[[float, float], float]

    def __call__(self, V: float, ca_i: float) -> float:
        """Delegate to the wrapped callable.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The rate in 1/ms.
        """
        return self.fn(V, ca_i)


@dataclass(frozen=True)
class CalciumDependentFn(CalciumDependentRate):
    """Adapter wrapping a plain callable as a :class:`CalciumDependentRate`.

    Attributes:
        fn: Plain ``(V, ca_i) -> float`` callable whose output depends on
            ``ca_i``.
    """

    fn: Callable[[float, float], float]

    def __call__(self, V: float, ca_i: float) -> float:
        """Delegate to the wrapped callable.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The rate in 1/ms.
        """
        return self.fn(V, ca_i)
