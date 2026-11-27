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

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import cast

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

#: Distance in mV from a removable exp-linear singularity within which the
#: analytic (L'Hôpital) limit is returned instead of the 0/0 formula.  Shared
#: by :class:`ExpLinear` and the hand-written HH52 and Traub-Miles rates so
#: both agree near the singular point.
SINGULARITY_THRESHOLD: float = 1e-6


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

    An operand can also be a :class:`VoltageOnlyFn` or
    :class:`CalciumDependentFn`, so a term no form covers can be combined
    with the standard forms.

    :attr:`uses_calcium` is derived from the tree: an expression reads ca_i
    exactly when it contains a :class:`Hill` node or a rate (such as a
    :class:`CalciumDependentFn`) that reports ``uses_calcium``.  Because the
    structure is plain data, expressions compare by value, pickle for
    :func:`~neuroclamp.clamp_simulations.simulate_batch`, and can be inspected
    or translated by other tools.
    """

    # The compiled function, set on first call.  This bare annotation adds no
    # class attribute (Expr is not a dataclass), so before the first call the
    # lookup in __call__ raises AttributeError and triggers compilation.
    _compiled: Callable[[float, float], float]

    def children(self) -> tuple[Rate, ...]:
        """Return this node's direct operands.

        Returns:
            The child rates; empty for leaf forms.
        """
        return ()

    @abstractmethod
    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate this node directly from its definition.

        This is the readable reference implementation of each form.  Calling
        the expression runs an equivalent compiled function instead (see
        :meth:`__call__`), which performs the same operations in the same
        order and so returns identical results.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The value of the expression.
        """

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the expression through its compiled function.

        On first use the whole tree is translated into one flat Python
        function, cached on this instance.  Walking the tree node by node
        costs a Python call per node; the compiled function evaluates the
        same arithmetic with none of that overhead, so expressions run as
        fast as hand-written rate functions.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The value of the expression.
        """
        # The common case is a plain attribute call; the cache is built on the
        # first call (or the first after unpickling), when the lookup fails.
        try:
            return self._compiled(V, ca_i)
        except AttributeError:
            object.__setattr__(self, "_compiled", _compile(self))
            return self._compiled(V, ca_i)

    def __getstate__(self) -> dict[str, object]:
        """Return the pickled state, without the compiled-function cache.

        The cached function cannot be pickled; it is rebuilt on first use
        after unpickling.

        Returns:
            The instance dictionary minus the cache.
        """
        state = dict(self.__dict__)
        state.pop("_compiled", None)
        return state

    def _source(self, emit: "_Emitter") -> str:
        """Return Python source evaluating this node, for :func:`_compile`.

        The built-in forms override this with inline arithmetic equivalent
        to their :meth:`evaluate`.  A node whose class overrides ``evaluate``
        without also overriding ``_source`` is not inlined: the compiled
        function calls its ``evaluate`` instead (see :meth:`_Emitter.source`),
        so an ``Expr`` subclass defined outside this module, or a subclass of
        a built-in form, only needs ``evaluate``.  The default here does the
        same.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"{emit.rate(self.evaluate)}(V, ca_i)"

    @property
    def uses_calcium(self) -> bool:
        """Whether any node in this expression reads ca_i.

        Returns:
            ``True`` if a child (or this node) depends on ca_i.
        """
        return any(child.uses_calcium for child in self.children())

    def __add__(self, other: object) -> "Expr":
        """Build ``self + other``.

        Args:
            other: Right operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            An :class:`Add` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Add(self, o)

    def __radd__(self, other: object) -> "Expr":
        """Build ``other + self``.

        Args:
            other: Left operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            An :class:`Add` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Add(o, self)

    def __sub__(self, other: object) -> "Expr":
        """Build ``self - other``.

        Args:
            other: Right operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Sub` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Sub(self, o)

    def __rsub__(self, other: object) -> "Expr":
        """Build ``other - self``.

        Args:
            other: Left operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Sub` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Sub(o, self)

    def __mul__(self, other: object) -> "Expr":
        """Build ``self * other``.

        Args:
            other: Right operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Mul` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Mul(self, o)

    def __rmul__(self, other: object) -> "Expr":
        """Build ``other * self``.

        Args:
            other: Left operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Mul` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Mul(o, self)

    def __truediv__(self, other: object) -> "Expr":
        """Build ``self / other``.

        Args:
            other: Right operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Div` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Div(self, o)

    def __rtruediv__(self, other: object) -> "Expr":
        """Build ``other / self``.

        Args:
            other: Left operand: a :class:`Rate` or a number, which becomes
                a :class:`Const`.

        Returns:
            A :class:`Div` node, or ``NotImplemented`` for any other operand type.
        """
        o = _coerce(other)
        if o is None:
            return NotImplemented
        return Div(o, self)


def _coerce(value: object) -> Rate | None:
    """Convert an operand to a rate node, or ``None`` if it is not one.

    Rates (expressions and the :class:`VoltageOnlyFn` /
    :class:`CalciumDependentFn` adapters) pass through unchanged; real numbers
    become :class:`Const`.

    Args:
        value: The operand to convert.

    Returns:
        A rate, or ``None`` for an unsupported type.
    """
    if isinstance(value, Rate):
        return value
    if isinstance(value, Real):
        return Const(float(value))
    return None


def _as_expr(value: "Rate | float") -> Rate:
    """Convert a rate or number to a rate node, rejecting other types.

    Args:
        value: A rate or a real number.

    Returns:
        ``value`` itself if it is a :class:`Rate`, else a :class:`Const`.

    Raises:
        TypeError: If ``value`` is neither a rate nor a real number.
    """
    rate = _coerce(value)
    if rate is None:
        raise TypeError(f"expected a Rate or a number, got {type(value).__name__}")
    return rate


def _check_scale(s: float) -> None:
    """Reject a zero voltage scale for a leaf form.

    Args:
        s: Voltage scale in mV.

    Raises:
        ValueError: If ``s`` is zero.
    """
    if s == 0.0:
        raise ValueError("voltage scale s must be nonzero")


def _ref(rate: Rate, V: float, ca_i: float) -> float:
    """Evaluate an operand on the reference path.

    Expressions use :meth:`Expr.evaluate`, so a reference evaluation of a tree
    never touches the compiled functions; any other rate is called directly.

    Args:
        rate: The operand.
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM.

    Returns:
        The operand's value.
    """
    return rate.evaluate(V, ca_i) if isinstance(rate, Expr) else rate(V, ca_i)


class _Emitter:
    """Collects the names that an expression's generated source refers to.

    Parameter values are never written into the source text.  Each constant,
    opaque rate, and helper is bound to a generated name (``_c0``, ``_r1``,
    ...) that the compiled function reads as a closure variable, so the
    source contains only generated identifiers and operators, whatever values
    a node holds.
    """

    def __init__(self) -> None:
        """Start with the helper functions every expression may use."""
        self.values: dict[str, object] = {
            "_safe_exp": safe_exp,
            "_safe_cosh": safe_cosh,
            "_max": max,
        }
        self._temps = 0
        self._depth = 0

    def _bind(self, prefix: str, value: object) -> str:
        """Bind a value to a fresh name.

        Args:
            prefix: Name prefix (``_c`` for constants, ``_r`` for rates).
            value: The value the name refers to.

        Returns:
            The generated name.
        """
        name = f"{prefix}{len(self.values)}"
        self.values[name] = value
        return name

    def const(self, value: float) -> str:
        """Bind a numeric parameter.

        Args:
            value: The parameter value.

        Returns:
            The generated name.
        """
        return self._bind("_c", value)

    def rate(self, fn: Callable[[float, float], float]) -> str:
        """Bind a rate that is called as an opaque function.

        Args:
            fn: A ``(V, ca_i) -> float`` callable.

        Returns:
            The generated name.
        """
        return self._bind("_r", fn)

    def temp(self) -> str:
        """Allocate a name for an intermediate value inside the source.

        Returns:
            The generated name.
        """
        self._temps += 1
        return f"_t{self._temps}"

    def source(self, rate: Rate) -> str:
        """Return source evaluating an operand.

        An expression node is inlined when its inline source describes its
        own :meth:`~Expr.evaluate` (see :func:`_inlines`); otherwise its
        ``evaluate`` is called as an opaque function, so a subclass that
        overrides ``evaluate`` is never replaced by its parent's formula.
        Inlining stops :data:`_MAX_INLINE_DEPTH` nodes below the root: a
        deeper subtree is called through its own compiled function, which
        keeps every generated source within the parser's nesting limit.

        Args:
            rate: An expression (inlined) or any other rate (called).

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        if not isinstance(rate, Expr):
            return f"{self.rate(rate)}(V, ca_i)"
        if not _inlines(type(rate)):
            return f"{self.rate(rate.evaluate)}(V, ca_i)"
        if self._depth >= _MAX_INLINE_DEPTH:
            return f"{self.rate(rate)}(V, ca_i)"
        self._depth += 1
        try:
            return rate._source(self)
        finally:
            self._depth -= 1


# How many nested expression nodes one compiled function inlines.  Each node
# adds at most a few levels of parentheses to the generated source, so this
# stays well below the roughly 200 levels Python's tokenizer accepts.
_MAX_INLINE_DEPTH = 32


@functools.cache
def _inlines(cls: type) -> bool:
    """Report whether a node class's inline source matches its ``evaluate``.

    Walks the method resolution order and checks which of ``_source`` and
    ``evaluate`` is defined closer to ``cls``.  A class that overrides
    ``evaluate`` without also overriding ``_source`` would otherwise inherit
    inline source for a different formula.

    Args:
        cls: An :class:`Expr` subclass.

    Returns:
        ``True`` if ``_source`` is defined in ``cls`` or a class nearer to it
        than the nearest definition of ``evaluate``.
    """
    for klass in cls.__mro__:
        if "_source" in vars(klass):
            return True
        if "evaluate" in vars(klass):
            return False
    return False


def _compile(expr: "Expr") -> Callable[[float, float], float]:
    """Translate an expression tree into one flat Python function.

    The generated function performs the same operations, in the same order,
    as :meth:`Expr.evaluate` on each node, so its results are identical.

    Args:
        expr: The expression to compile.

    Returns:
        A ``(V, ca_i) -> float`` function.
    """
    emit = _Emitter()
    body = emit.source(expr)
    names = ", ".join(emit.values)
    # An outer factory receives the bound values as arguments; the inner
    # function reads them as closure variables, which costs nothing per call.
    source = (
        f"def _factory({names}):\n"
        f"    def _rate(V, ca_i):\n"
        f"        return {body}\n"
        f"    return _rate\n"
    )
    namespace: dict[str, object] = {}
    # The source holds only generated names and operators (see _Emitter).
    exec(source, namespace)
    factory = cast(
        "Callable[..., Callable[[float, float], float]]", namespace["_factory"]
    )
    return factory(**emit.values)


@dataclass(frozen=True)
class Const(Expr):
    """A constant value.

    Attributes:
        value: The constant, in the units of the surrounding expression.
    """

    value: float

    def evaluate(self, V: float, ca_i: float) -> float:
        """Return the constant.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            :attr:`value`.
        """
        return self.value

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the constant.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return emit.const(self.value)


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

    def __post_init__(self) -> None:
        """Reject a zero voltage scale, which would divide by zero.

        Raises:
            ValueError: If ``s`` is zero.
        """
        _check_scale(self.s)

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate the exponential (argument clipped by ``safe_exp``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k * safe_exp((V - self.vh) / self.s)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the exponential.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        k, vh, s = emit.const(self.k), emit.const(self.vh), emit.const(self.s)
        return f"({k} * _safe_exp((V - {vh}) / {s}))"


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

    def __post_init__(self) -> None:
        """Reject a zero voltage scale, which would divide by zero.

        Raises:
            ValueError: If ``s`` is zero.
        """
        _check_scale(self.s)

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate the sigmoid (argument clipped by ``safe_exp``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / (1.0 + safe_exp(-(V - self.vh) / self.s))

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the sigmoid.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        k, vh, s = emit.const(self.k), emit.const(self.vh), emit.const(self.s)
        return f"({k} / (1.0 + _safe_exp(-(V - {vh}) / {s})))"


@dataclass(frozen=True)
class ExpLinear(Expr):
    """Exp-linear (linoid) form: ``k · (V − vh) / (1 − exp(−(V − vh) / s))``.

    The classic Hodgkin-Huxley activation rate.  It has a removable
    singularity at ``V = vh``, where its analytic limit ``k · s`` is returned
    when ``|V − vh| < tol``.  A rate written as
    ``k′ · (V − vh) / (exp((V − vh) / s′) − 1)`` is ``ExpLinear(-k′, vh, -s′)``:
    negating ``s`` alone flips the sign of the whole rate, so ``k`` must be
    negated with it.

    Attributes:
        k: Slope of the linear asymptote in 1/(ms·mV).
        vh: Singular-point voltage in mV.
        s: Voltage scale in mV.
        tol: Distance from ``vh`` within which the limit is returned.
    """

    k: float
    vh: float
    s: float
    tol: float = SINGULARITY_THRESHOLD

    def __post_init__(self) -> None:
        """Reject a zero voltage scale, which would divide by zero.

        Raises:
            ValueError: If ``s`` is zero.
        """
        _check_scale(self.s)

    def evaluate(self, V: float, ca_i: float) -> float:
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

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the exp-linear form.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        k, vh, s = emit.const(self.k), emit.const(self.vh), emit.const(self.s)
        tol, x = emit.const(self.tol), emit.temp()
        # The condition runs first, so it binds x before either branch uses it.
        return (
            f"({k} * {s} if abs({x} := V - {vh}) < {tol}"
            f" else {k} * {x} / (1.0 - _safe_exp(-{x} / {s})))"
        )


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

    def __post_init__(self) -> None:
        """Reject a zero voltage scale, which would divide by zero.

        Raises:
            ValueError: If ``s`` is zero.
        """
        _check_scale(self.s)

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate the secant (argument clipped by ``safe_cosh``).

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM (ignored).

        Returns:
            The value at ``V``.
        """
        return self.k / safe_cosh((V - self.vh) / self.s)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the secant.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        k, vh, s = emit.const(self.k), emit.const(self.vh), emit.const(self.s)
        return f"({k} / _safe_cosh((V - {vh}) / {s}))"


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

    def __post_init__(self) -> None:
        """Reject parameters that make the Hill function undefined.

        Raises:
            ValueError: If ``kd`` or ``n`` is not positive.
        """
        if not self.kd > 0.0:
            raise ValueError(f"Hill kd must be positive, got {self.kd}")
        if not self.n > 0:
            raise ValueError(f"Hill coefficient n must be positive, got {self.n}")

    @property
    def uses_calcium(self) -> bool:
        """Report that this node reads ca_i.

        Returns:
            Always ``True``.
        """
        return True

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate the Hill function.

        Args:
            V: Membrane voltage in mV (ignored).
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            Fractional activation in [0, 1].
        """
        ca_n = ca_i**self.n
        return ca_n / (ca_n + self.kd**self.n)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for the Hill function.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        n, ca_n = emit.const(self.n), emit.temp()
        # kd**n is the same constant on every call, so it is computed once.
        kd_n = emit.const(self.kd**self.n)
        return f"(({ca_n} := ca_i ** {n}) / ({ca_n} + {kd_n}))"


@dataclass(frozen=True)
class _Binary(Expr):
    """Abstract base for nodes that combine two rates.

    Attributes:
        a: Left operand.
        b: Right operand.
    """

    a: Rate
    b: Rate

    def children(self) -> tuple[Rate, ...]:
        """Return both operands.

        Returns:
            ``(a, b)``.
        """
        return (self.a, self.b)


@dataclass(frozen=True)
class Add(_Binary):
    """Sum of two expressions: ``a + b``."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``a + b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The sum.
        """
        return _ref(self.a, V, ca_i) + _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``a + b``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"({emit.source(self.a)} + {emit.source(self.b)})"


@dataclass(frozen=True)
class Sub(_Binary):
    """Difference of two expressions: ``a − b``."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``a − b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The difference.
        """
        return _ref(self.a, V, ca_i) - _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``a − b``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"({emit.source(self.a)} - {emit.source(self.b)})"


@dataclass(frozen=True)
class Mul(_Binary):
    """Product of two expressions: ``a · b``."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``a · b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The product.
        """
        return _ref(self.a, V, ca_i) * _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``a · b``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"({emit.source(self.a)} * {emit.source(self.b)})"


@dataclass(frozen=True)
class Div(_Binary):
    """Quotient of two expressions: ``a / b``."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``a / b``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The quotient.
        """
        return _ref(self.a, V, ca_i) / _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``a / b``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"({emit.source(self.a)} / {emit.source(self.b)})"


@dataclass(frozen=True)
class Max(_Binary):
    """Larger of two expressions, e.g. a time constant with a floor."""

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``max(a, b)``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The larger operand.
        """
        return max(_ref(self.a, V, ca_i), _ref(self.b, V, ca_i))

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``max(a, b)``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"_max({emit.source(self.a)}, {emit.source(self.b)})"


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

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``inf / tau``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The forward rate in 1/ms.
        """
        return _ref(self.a, V, ca_i) / _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``inf / tau``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"({emit.source(self.a)} / {emit.source(self.b)})"


@dataclass(frozen=True)
class InfTauBeta(_Binary):
    """Backward rate from a steady state and time constant: ``(1 − inf) / tau``.

    Built by :func:`inf_tau`; see :class:`InfTauAlpha`.

    Attributes:
        a: Steady-state open probability ``inf`` (dimensionless).
        b: Time constant ``tau`` in ms.
    """

    def evaluate(self, V: float, ca_i: float) -> float:
        """Evaluate ``(1 − inf) / tau``.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ concentration in mM.

        Returns:
            The backward rate in 1/ms.
        """
        return (1.0 - _ref(self.a, V, ca_i)) / _ref(self.b, V, ca_i)

    def _source(self, emit: "_Emitter") -> str:
        """Return inline source for ``(1 − inf) / tau``.

        Args:
            emit: Collects the names the generated source refers to.

        Returns:
            A Python expression in ``V`` and ``ca_i``.
        """
        return f"((1.0 - {emit.source(self.a)}) / {emit.source(self.b)})"


def inf_tau(inf: "Rate | float", tau: "Rate | float") -> tuple[InfTauAlpha, InfTauBeta]:
    """Build a gate's (alpha, beta) rates from its steady state and time constant.

    Many channel models specify a gate by ``x_inf(V)`` and ``tau_x(V)``
    rather than by rates.  The equivalent rates are ``alpha = inf / tau`` and
    ``beta = (1 − inf) / tau``, so that ``dx/dt = (inf − x) / tau``.

    Args:
        inf: Steady-state open probability, as a rate or a number.
        tau: Time constant in ms, as a rate or a number.

    Returns:
        ``(alpha, beta)`` for a :class:`~neuroclamp.channels.GatingVariable`.
    """
    inf_expr = _as_expr(inf)
    tau_expr = _as_expr(tau)
    return InfTauAlpha(inf_expr, tau_expr), InfTauBeta(inf_expr, tau_expr)
