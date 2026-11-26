"""Gating-variable rate-function class hierarchy.

Every alpha/beta rate attached to a :class:`~neuroclamp.channels.GatingVariable`
is an instance of :class:`Rate`.  :attr:`Rate.uses_calcium` reports whether a
rate's output depends on ca_i.  The voltage-clamp optimizer tabulates rates
that do not read ca_i over the prescribed voltage protocol; Ca²⁺-dependent
rates are evaluated scalar-wise in the RK4 loop.  The abstract subclasses
:class:`VoltageOnlyRate` and :class:`CalciumDependentRate` fix
``uses_calcium`` to ``False`` and ``True`` respectively.

Rates can be written two ways:

- **Expressions** (:class:`Expr` and its subclasses) describe a rate as a
  small tree of standard forms, for example ``inf_tau(Sigmoid(1.0, -40.0,
  8.0), 0.2)``.  They are plain data, so they can be compared, pickled, and
  inspected, and they report :attr:`~Rate.uses_calcium` automatically.
- **Adapters** :class:`VoltageOnlyFn` and :class:`CalciumDependentFn` wrap
  plain ``(V, ca_i) -> float`` callables, for kinetics no expression form
  covers.  The caller declares the Ca²⁺ dependence by choosing the adapter.

All rate types are frozen dataclasses, matching the picklability contract
relied on by :func:`~neuroclamp.clamp_simulations.simulate_batch`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .utils import safe_cosh, safe_exp


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


# ---------------------------------------------------------------------------
# Rate expressions
# ---------------------------------------------------------------------------

#: Distance in mV from an :class:`ExpLinear` singularity within which its
#: analytic limit is returned instead of the 0/0 formula.
EXP_LINEAR_TOL: float = 1e-6


class Expr(Rate):
    """Base class for rate expressions: rate functions described as data.

    An expression is a small tree of the leaf forms below (:class:`Exp`,
    :class:`Sigmoid`, :class:`ExpLinear`, :class:`Sech`, :class:`Hill`,
    :class:`Const`) joined by arithmetic (:class:`Add`, :class:`Sub`,
    :class:`Mul`, :class:`Div`, :class:`Max`).  Every node is a frozen
    dataclass and a callable :class:`Rate`, so an expression can be used
    anywhere a rate function can.  ``+ - * /`` build combinator nodes and wrap
    plain numbers in :class:`Const`, so kinetics read like their equations::

        tau_h = 1.0 / (Exp(0.128, -38.0, -18.0) + Sigmoid(4.0, -15.0, 5.0))

    :attr:`uses_calcium` is derived from the tree: an expression reads ca_i
    exactly when it contains a :class:`Hill` node.  Because the structure is
    plain data, expressions compare by value, pickle for
    :func:`~neuroclamp.clamp_simulations.simulate_batch`, and can be inspected
    or translated by other tools.
    """

    def children(self) -> tuple["Expr", ...]:
        """Return this node's direct sub-expressions.

        Returns:
            The child nodes; empty for leaf forms.
        """
        return ()

    @property
    def uses_calcium(self) -> bool:
        """Whether any node in this expression reads ca_i.

        Returns:
            ``True`` if a child (or this node) depends on ca_i.
        """
        return any(child.uses_calcium for child in self.children())

    def __add__(self, other: "Expr | float") -> "Expr":
        """Build ``self + other``.

        Args:
            other: Right operand; numbers become :class:`Const`.

        Returns:
            An :class:`Add` node.
        """
        return Add(self, _as_expr(other))

    def __radd__(self, other: "Expr | float") -> "Expr":
        """Build ``other + self``.

        Args:
            other: Left operand; numbers become :class:`Const`.

        Returns:
            An :class:`Add` node.
        """
        return Add(_as_expr(other), self)

    def __sub__(self, other: "Expr | float") -> "Expr":
        """Build ``self - other``.

        Args:
            other: Right operand; numbers become :class:`Const`.

        Returns:
            A :class:`Sub` node.
        """
        return Sub(self, _as_expr(other))

    def __rsub__(self, other: "Expr | float") -> "Expr":
        """Build ``other - self``.

        Args:
            other: Left operand; numbers become :class:`Const`.

        Returns:
            A :class:`Sub` node.
        """
        return Sub(_as_expr(other), self)

    def __mul__(self, other: "Expr | float") -> "Expr":
        """Build ``self * other``.

        Args:
            other: Right operand; numbers become :class:`Const`.

        Returns:
            A :class:`Mul` node.
        """
        return Mul(self, _as_expr(other))

    def __rmul__(self, other: "Expr | float") -> "Expr":
        """Build ``other * self``.

        Args:
            other: Left operand; numbers become :class:`Const`.

        Returns:
            A :class:`Mul` node.
        """
        return Mul(_as_expr(other), self)

    def __truediv__(self, other: "Expr | float") -> "Expr":
        """Build ``self / other``.

        Args:
            other: Right operand; numbers become :class:`Const`.

        Returns:
            A :class:`Div` node.
        """
        return Div(self, _as_expr(other))

    def __rtruediv__(self, other: "Expr | float") -> "Expr":
        """Build ``other / self``.

        Args:
            other: Left operand; numbers become :class:`Const`.

        Returns:
            A :class:`Div` node.
        """
        return Div(_as_expr(other), self)


def _as_expr(value: "Expr | float") -> Expr:
    """Wrap a plain number in :class:`Const`; pass expressions through.

    Args:
        value: An expression or a number.

    Returns:
        An expression node.
    """
    return value if isinstance(value, Expr) else Const(float(value))


@dataclass(frozen=True)
class Const(Expr):
    """A constant value.

    Attributes:
        value: The constant, in the units of the surrounding expression.
    """

    value: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Return the constant.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            :attr:`value`.
        """
        return self.value


@dataclass(frozen=True)
class Exp(Expr):
    """Exponential in voltage: ``k · exp((V − vh) / s)``.

    Attributes:
        k: Amplitude (the value at ``V = vh``).
        vh: Reference voltage in mV.
        s: Voltage scale in mV; negative for a rate that falls with
            depolarization.
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the exponential (argument clipped by ``safe_exp``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k * safe_exp((V - self.vh) / self.s)


@dataclass(frozen=True)
class Sigmoid(Expr):
    """Boltzmann sigmoid: ``k / (1 + exp(−(V − vh) / s))``.

    With ``k = 1`` this is a steady-state activation curve; a negative ``s``
    gives an inactivation (decreasing) curve.

    Attributes:
        k: Maximum value.
        vh: Half-point voltage in mV.
        s: Slope factor in mV.
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the sigmoid (argument clipped by ``safe_exp``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / (1.0 + safe_exp(-(V - self.vh) / self.s))


@dataclass(frozen=True)
class ExpLinear(Expr):
    """Exp-linear (linoid) form: ``k · (V − vh) / (1 − exp(−(V − vh) / s))``.

    The classic Hodgkin-Huxley activation rate.  It has a removable
    singularity at ``V = vh``, where its analytic limit ``k · s`` is returned
    when ``|V − vh| < tol``.  For ``s < 0`` the curve is mirrored, which
    expresses rates written as ``k · x / (exp(x / s) − 1)``.

    Attributes:
        k: Slope of the linear asymptote in 1/(ms·mV).
        vh: Singular-point voltage in mV.
        s: Voltage scale in mV.
        tol: Distance from ``vh`` within which the limit is returned.
    """

    k: float
    vh: float
    s: float
    tol: float = EXP_LINEAR_TOL

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the exp-linear form.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        x = V - self.vh
        if abs(x) < self.tol:
            return self.k * self.s
        return self.k * x / (1.0 - safe_exp(-x / self.s))


@dataclass(frozen=True)
class Sech(Expr):
    """Hyperbolic secant: ``k / cosh((V − vh) / s)``.

    The bell-shaped voltage dependence used for gating time constants.

    Attributes:
        k: Peak value (at ``V = vh``).
        vh: Center voltage in mV.
        s: Voltage scale in mV.
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the secant (argument clipped by ``safe_cosh``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / safe_cosh((V - self.vh) / self.s)


@dataclass(frozen=True)
class Hill(Expr):
    """Hill function of intracellular Ca²⁺: ``caⁿ / (caⁿ + kdⁿ)``.

    The only node that reads ca_i, so any expression containing it reports
    :attr:`uses_calcium` as ``True``.

    Attributes:
        kd: Half-activation concentration in mM.
        n: Hill coefficient.
    """

    kd: float
    n: int

    @property
    def uses_calcium(self) -> bool:
        """Report that this node reads ca_i.

        Returns:
            Always ``True``.
        """
        return True

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the Hill function.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            Fractional activation in [0, 1].
        """
        ca_n = ca_i**self.n
        return ca_n / (ca_n + self.kd**self.n)


@dataclass(frozen=True)
class _Binary(Expr):
    """Abstract base for nodes that combine two sub-expressions.

    Attributes:
        a: Left operand.
        b: Right operand.
    """

    a: Expr
    b: Expr

    def children(self) -> tuple[Expr, ...]:
        """Return both operands.

        Returns:
            ``(a, b)``.
        """
        return (self.a, self.b)


@dataclass(frozen=True)
class Add(_Binary):
    """Sum of two expressions: ``a + b``."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a + b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The sum.
        """
        return self.a(V, ca_i) + self.b(V, ca_i)


@dataclass(frozen=True)
class Sub(_Binary):
    """Difference of two expressions: ``a − b``."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a − b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The difference.
        """
        return self.a(V, ca_i) - self.b(V, ca_i)


@dataclass(frozen=True)
class Mul(_Binary):
    """Product of two expressions: ``a · b``."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a · b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The product.
        """
        return self.a(V, ca_i) * self.b(V, ca_i)


@dataclass(frozen=True)
class Div(_Binary):
    """Quotient of two expressions: ``a / b``."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a / b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The quotient.
        """
        return self.a(V, ca_i) / self.b(V, ca_i)


@dataclass(frozen=True)
class Max(_Binary):
    """Larger of two expressions, e.g. a time constant with a floor."""

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``max(a, b)``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The larger operand.
        """
        return max(self.a(V, ca_i), self.b(V, ca_i))


@dataclass(frozen=True)
class InfTauAlpha(_Binary):
    """Forward rate from a steady state and time constant: ``inf / tau``.

    Built by :func:`inf_tau`.  Paired with :class:`InfTauBeta` on the same
    ``inf`` and ``tau``, the gate relaxes toward ``inf`` with time constant
    ``tau``.  The explicit node type keeps that structure visible to tools
    that translate expressions, which can evaluate ``inf`` and ``tau`` once
    for both rates.

    Attributes:
        a: Steady-state open probability ``inf`` (dimensionless).
        b: Time constant ``tau`` in ms.
    """

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``inf / tau``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The forward rate in 1/ms.
        """
        return self.a(V, ca_i) / self.b(V, ca_i)


@dataclass(frozen=True)
class InfTauBeta(_Binary):
    """Backward rate from a steady state and time constant: ``(1 − inf) / tau``.

    Built by :func:`inf_tau`; see :class:`InfTauAlpha`.

    Attributes:
        a: Steady-state open probability ``inf`` (dimensionless).
        b: Time constant ``tau`` in ms.
    """

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``(1 − inf) / tau``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The backward rate in 1/ms.
        """
        return (1.0 - self.a(V, ca_i)) / self.b(V, ca_i)


def inf_tau(inf: Expr, tau: "Expr | float") -> tuple[InfTauAlpha, InfTauBeta]:
    """Build a gate's (alpha, beta) rates from its steady state and time constant.

    Many channel models specify a gate by ``x_inf(V)`` and ``tau_x(V)``
    rather than by rates.  The equivalent rates are ``alpha = inf / tau`` and
    ``beta = (1 − inf) / tau``, so that ``dx/dt = (inf − x) / tau``.

    Args:
        inf: Steady-state open probability expression.
        tau: Time constant in ms, as an expression or a number.

    Returns:
        ``(alpha, beta)`` for a :class:`~neuroclamp.channels.GatingVariable`.
    """
    tau_expr = _as_expr(tau)
    return InfTauAlpha(inf, tau_expr), InfTauBeta(inf, tau_expr)
