"""Compare every module-level rate against its expression tree, pointwise."""

from __future__ import annotations

import numpy as np
from exprs import ExpLinear, Expr
from translate import all_module_rates, translate

CA_GRID = [0.0, 1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 1e-2]


def _singular_points(tree: Expr) -> list[float]:
    """Collect every ExpLinear singular point and its close neighbors.

    Args:
        tree: An expression tree.

    Returns:
        Voltages in mV at and around each removable singularity.
    """
    points: list[float] = []
    stack: list[Expr] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ExpLinear):
            points += [node.vh, node.vh + 1e-8, node.vh - 1e-8, node.vh + 1e-5]
        stack += [c for c in node._children() if isinstance(c, Expr)]
    return points


def main() -> None:
    """Report coverage, worst relative error and Ca²⁺ classification."""
    rates = all_module_rates()
    voltages = list(np.linspace(-150, 100, 5001)) + list(np.arange(-150, 101, 0.5))
    missing: list[str] = []
    for name, rate in rates:
        tree = translate(rate)
        if tree is None:
            missing.append(name)
        else:
            voltages += _singular_points(tree)

    worst, where = 0.0, None
    for name, rate in rates:
        tree = translate(rate)
        if tree is None:
            continue
        for ca in CA_GRID if tree.uses_calcium else [0.0]:
            for v in voltages:
                expected, got = rate(float(v), ca), tree(float(v), ca)
                err = abs(expected - got) / max(abs(expected), 1e-12)
                if err > worst:
                    worst, where = err, (name, float(v), ca)

    print(f"{len(rates)} module-level rates, {len(rates) - len(missing)} translated")
    print(f"missing: {missing}")
    print(f"{len(voltages)} voltages; worst relative error {worst:.2e} at {where}")
    ca_dep = sorted(n for n, r in rates if (t := translate(r)) and t.uses_calcium)
    print(f"Ca²⁺-dependent by tree structure: {ca_dep}")


if __name__ == "__main__":
    main()
