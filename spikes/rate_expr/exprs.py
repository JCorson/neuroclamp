"""Gating-variable rate functions as small expression trees.

Every node is a frozen dataclass and a callable :class:`~neuroclamp.rates.Rate`
``(V, ca_i) -> float``, so existing call sites (equilibrium, voltage-clamp
tabulation, tests) keep working. ``to_spec()`` emits a plain nested dict that a
compiled backend can deserialize (e.g. serde + pythonize in Rust).

Arithmetic operators are overloaded so channel definitions read close to the
published equations; plain floats are wrapped in :class:`Const` automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from neuroclamp.rates import Rate
from neuroclamp.utils import safe_cosh, safe_exp

DEFAULT_TOL = 1e-6


class Expr(Rate):
    """Base class for expression-tree rate nodes."""

    @property
    def uses_calcium(self) -> bool:
        """Whether any node in this subtree reads ``ca_i``.

        Returns:
            ``True`` if a calcium leaf (:class:`Hill`) appears in the subtree.
        """
        return any(
            isinstance(child, Expr) and child.uses_calcium for child in self._children()
        )

    def _children(self) -> list[Any]:
        """Return the values of this node's dataclass fields.

        Returns:
            Field values in declaration order.
        """
        return [getattr(self, f.name) for f in fields(self)]  # type: ignore[arg-type]

    def to_spec(self) -> dict[str, Any]:
        """Lower this subtree to a plain nested dict tagged by ``op``.

        Returns:
            A JSON-serializable dict whose ``op`` key names the node type.
        """
        out: dict[str, Any] = {"op": type(self).__name__}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            out[f.name] = value.to_spec() if isinstance(value, Expr) else value
        return out

    def __add__(self, other: Expr | float) -> Expr:
        """Build ``self + other``.

        Args:
            other: Right operand.

        Returns:
            An :class:`Add` node.
        """
        return Add(self, _wrap(other))

    def __radd__(self, other: Expr | float) -> Expr:
        """Build ``other + self``.

        Args:
            other: Left operand.

        Returns:
            An :class:`Add` node.
        """
        return Add(_wrap(other), self)

    def __sub__(self, other: Expr | float) -> Expr:
        """Build ``self - other``.

        Args:
            other: Right operand.

        Returns:
            A :class:`Sub` node.
        """
        return Sub(self, _wrap(other))

    def __rsub__(self, other: Expr | float) -> Expr:
        """Build ``other - self``.

        Args:
            other: Left operand.

        Returns:
            A :class:`Sub` node.
        """
        return Sub(_wrap(other), self)

    def __mul__(self, other: Expr | float) -> Expr:
        """Build ``self * other``.

        Args:
            other: Right operand.

        Returns:
            A :class:`Mul` node.
        """
        return Mul(self, _wrap(other))

    def __rmul__(self, other: Expr | float) -> Expr:
        """Build ``other * self``.

        Args:
            other: Left operand.

        Returns:
            A :class:`Mul` node.
        """
        return Mul(_wrap(other), self)

    def __truediv__(self, other: Expr | float) -> Expr:
        """Build ``self / other``.

        Args:
            other: Right operand.

        Returns:
            A :class:`Div` node.
        """
        return Div(self, _wrap(other))

    def __rtruediv__(self, other: Expr | float) -> Expr:
        """Build ``other / self``.

        Args:
            other: Left operand.

        Returns:
            A :class:`Div` node.
        """
        return Div(_wrap(other), self)


def _wrap(x: Expr | float) -> Expr:
    """Wrap a plain number as :class:`Const`; pass expressions through.

    Args:
        x: An expression or a number.

    Returns:
        An expression node.
    """
    return x if isinstance(x, Expr) else Const(float(x))


# ---------------------------------------------------------------- leaves
@dataclass(frozen=True)
class Const(Expr):
    """A constant value.

    Attributes:
        value: The constant.
    """

    value: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Return the constant.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ in mM (ignored).

        Returns:
            The constant value.
        """
        return self.value


@dataclass(frozen=True)
class Exp(Expr):
    """Exponential: ``k · exp((V − vh) / s)``.

    Attributes:
        k: Amplitude.
        vh: Reference voltage in mV.
        s: Voltage scale in mV (negative for a decreasing curve).
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the exponential.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k * safe_exp((V - self.vh) / self.s)


@dataclass(frozen=True)
class Sigmoid(Expr):
    """Boltzmann sigmoid: ``k / (1 + exp(−(V − vh) / s))``.

    Attributes:
        k: Amplitude.
        vh: Half-point voltage in mV.
        s: Slope in mV; negative gives a decreasing curve.
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the sigmoid.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / (1.0 + safe_exp(-(V - self.vh) / self.s))


@dataclass(frozen=True)
class ExpLinear(Expr):
    """Exp-linear (linoid): ``k · (V − vh) / (1 − exp(−(V − vh) / s))``.

    The removable singularity at ``V = vh`` returns its limit ``k · s``.

    Attributes:
        k: Amplitude.
        vh: Singular-point voltage in mV.
        s: Voltage scale in mV.
        tol: Distance from ``vh`` within which the limit is returned.
    """

    k: float
    vh: float
    s: float
    tol: float = DEFAULT_TOL

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the exp-linear form.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM (ignored).

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

    Attributes:
        k: Amplitude.
        vh: Center voltage in mV.
        s: Voltage scale in mV.
    """

    k: float
    vh: float
    s: float

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the secant form.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / safe_cosh((V - self.vh) / self.s)


@dataclass(frozen=True)
class Hill(Expr):
    """Hill function of intracellular calcium: ``caⁿ / (caⁿ + kdⁿ)``.

    Attributes:
        kd: Half-activation concentration in mM.
        n: Hill coefficient.
    """

    kd: float
    n: int

    @property
    def uses_calcium(self) -> bool:
        """Report that this leaf reads ``ca_i``.

        Returns:
            Always ``True``.
        """
        return True

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the Hill function.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            Fractional activation in [0, 1].
        """
        ca_n = ca_i**self.n
        return ca_n / (ca_n + self.kd**self.n)


# ------------------------------------------------------------ combinators
@dataclass(frozen=True)
class Add(Expr):
    """Sum of two subtrees.

    Attributes:
        a: Left operand.
        b: Right operand.
    """

    a: Expr
    b: Expr

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a + b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The sum.
        """
        return self.a(V, ca_i) + self.b(V, ca_i)


@dataclass(frozen=True)
class Sub(Expr):
    """Difference of two subtrees.

    Attributes:
        a: Left operand.
        b: Right operand.
    """

    a: Expr
    b: Expr

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a - b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The difference.
        """
        return self.a(V, ca_i) - self.b(V, ca_i)


@dataclass(frozen=True)
class Mul(Expr):
    """Product of two subtrees.

    Attributes:
        a: Left operand.
        b: Right operand.
    """

    a: Expr
    b: Expr

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a * b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The product.
        """
        return self.a(V, ca_i) * self.b(V, ca_i)


@dataclass(frozen=True)
class Div(Expr):
    """Quotient of two subtrees.

    Attributes:
        a: Numerator.
        b: Denominator.
    """

    a: Expr
    b: Expr

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``a / b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The quotient.
        """
        return self.a(V, ca_i) / self.b(V, ca_i)


@dataclass(frozen=True)
class Max(Expr):
    """Maximum of two subtrees (used for time-constant floors).

    Attributes:
        a: First operand.
        b: Second operand.
    """

    a: Expr
    b: Expr

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate ``max(a, b)``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The larger operand.
        """
        return max(self.a(V, ca_i), self.b(V, ca_i))


# ---------------------------------------------------------------- helpers
def inf_tau(inf: Expr, tau: Expr | float) -> tuple[Expr, Expr]:
    """Build (alpha, beta) from a steady state and a time constant.

    ``alpha = inf / tau`` and ``beta = (1 − inf) / tau``. The design on #410
    proposes returning a paired ``InfTauAlpha`` / ``InfTauBeta`` so lowering
    can emit an explicit ``inf_tau`` form; this spike expands them to plain
    trees, which duplicates ``inf`` and ``tau`` in the lowered spec.

    Args:
        inf: Steady-state open probability.
        tau: Time constant in ms.

    Returns:
        Tuple of (alpha, beta) expression trees in 1/ms.
    """
    tau_expr = _wrap(tau)
    return inf / tau_expr, (1.0 - inf) / tau_expr


def boltzmann_cosh(
    half: float,
    slope: float,
    tau_scale: float,
    tau_floor: float,
    *,
    inverted: bool = False,
    tau_cosh_scale: float | None = None,
    tau_rate: float = 1.0,
) -> tuple[Expr, Expr]:
    """Expression-tree equivalent of ``boltzmann_cosh_rates``.

    Args:
        half: Half-activation voltage in mV.
        slope: Activation slope in mV.
        tau_scale: Numerator of the time-constant expression in ms.
        tau_floor: Minimum time constant in ms.
        inverted: If True the gate opens at hyperpolarized potentials.
        tau_cosh_scale: Voltage scale in the cosh denominator; defaults to
            ``2 * slope``.
        tau_rate: Extra multiplier in the tau denominator.

    Returns:
        Tuple of (alpha, beta) expression trees in 1/ms.
    """
    c = tau_cosh_scale if tau_cosh_scale is not None else 2.0 * slope
    inf = Sigmoid(1.0, half, -slope if inverted else slope)
    tau = Max(Sech(tau_scale / tau_rate, half, c), Const(tau_floor))
    return inf_tau(inf, tau)


def traub_miles(vt: float) -> dict[str, Expr]:
    """Return the six Traub-Miles rates for a voltage threshold.

    Args:
        vt: Cell-type voltage threshold in mV.

    Returns:
        Dict keyed ``alpha_m``, ``beta_m``, ``alpha_h``, ``beta_h``,
        ``alpha_n``, ``beta_n``.
    """
    return {
        "alpha_m": ExpLinear(0.32, vt + 13, 4),
        "beta_m": ExpLinear(-0.28, vt + 40, -5),
        "alpha_h": Exp(0.128, vt + 17, -18),
        "beta_h": Sigmoid(4.0, vt + 40, 5),
        "alpha_n": ExpLinear(0.032, vt + 15, 5),
        "beta_n": Exp(0.5, vt + 10, -40),
    }
