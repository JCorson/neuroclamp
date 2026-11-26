"""Compile an expression tree once into a single flat Python function.

Walking a tree node by node costs extra Python calls per rate evaluation.
Generating one ``lambda`` per tree removes that overhead so the Python backend
is no slower than hand-written rate functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import exprs as E

from neuroclamp.rates import Rate
from neuroclamp.utils import safe_cosh, safe_exp

_BINARY_OPS: dict[type, str] = {E.Add: "+", E.Sub: "-", E.Mul: "*", E.Div: "/"}


def _src(node: E.Expr) -> str:
    """Render a subtree as a Python expression in ``V`` and ``ca``.

    Args:
        node: The subtree to render.

    Returns:
        Python source for the subtree.

    Raises:
        TypeError: If the node type is unknown.
    """
    r = repr
    if isinstance(node, E.Const):
        return r(node.value)
    if isinstance(node, E.Exp):
        return f"({r(node.k)}*_exp((V-{r(node.vh)})/{r(node.s)}))"
    if isinstance(node, E.Sigmoid):
        return f"({r(node.k)}/(1.0+_exp(-(V-{r(node.vh)})/{r(node.s)})))"
    if isinstance(node, E.Sech):
        return f"({r(node.k)}/_cosh((V-{r(node.vh)})/{r(node.s)}))"
    if isinstance(node, E.ExpLinear):
        x = f"(V-{r(node.vh)})"
        limit = r(node.k * node.s)
        body = f"{r(node.k)}*{x}/(1.0-_exp(-{x}/{r(node.s)}))"
        return f"({limit} if abs({x})<{r(node.tol)} else {body})"
    if isinstance(node, E.Hill):
        return f"(ca**{node.n}/(ca**{node.n}+{r(node.kd**node.n)}))"
    if type(node) in _BINARY_OPS:
        a, b = node._children()
        return f"({_src(a)}{_BINARY_OPS[type(node)]}{_src(b)})"
    if isinstance(node, E.Max):
        a, b = node._children()
        return f"max({_src(a)},{_src(b)})"
    raise TypeError(f"unknown node {node!r}")


@dataclass(frozen=True, eq=False)
class Compiled(Rate):
    """A rate backed by a tree compiled to a single Python lambda.

    Attributes:
        tree: The source expression tree.
    """

    tree: E.Expr
    _fn: Callable[[float, float], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Generate and compile the lambda for ``tree``."""
        namespace = {"_exp": safe_exp, "_cosh": safe_cosh}
        fn = eval(f"lambda V, ca: {_src(self.tree)}", namespace)  # noqa: S307
        object.__setattr__(self, "_fn", fn)

    @property
    def uses_calcium(self) -> bool:
        """Whether the underlying tree reads ``ca_i``.

        Returns:
            The tree's ``uses_calcium`` flag.
        """
        return self.tree.uses_calcium

    def __call__(self, V: float, ca_i: float) -> float:
        """Evaluate the compiled rate.

        Args:
            V: Membrane voltage in mV.
            ca_i: Intracellular Ca²⁺ in mM.

        Returns:
            The rate in 1/ms.
        """
        return self._fn(V, ca_i)
