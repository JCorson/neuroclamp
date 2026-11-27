# Rates

Gating-variable rate functions. Every `alpha`/`beta` on a `GatingVariable` is a
`Rate`.

Rate **expressions** describe kinetics as data: a small tree of standard forms
(`Exp`, `Sigmoid`, `ExpLinear`, `Sech`, `Hill`, `Const`) joined with `+`, `-`,
`*`, `/` and `Max`. For gates specified by a steady state and a time constant,
`inf_tau` builds the matching pair of rates. For example, a Na⁺ inactivation
gate:

```python
from neuroclamp.channels import GatingVariable
from neuroclamp.rates import Exp, Sigmoid, inf_tau

h_alpha, h_beta = inf_tau(
    inf=Sigmoid(1.0, -45.5, -6.4),
    tau=1.0 / (Exp(0.128, -38.0, -18.0) + Sigmoid(4.0, -15.0, 5.0)),
)
h = GatingVariable("h", power=1, alpha=h_alpha, beta=h_beta)
```

Kinetics that no expression form covers can wrap a plain function with
`VoltageOnlyFn` or `CalciumDependentFn`, which can also appear inside an
expression.

The first call to an expression compiles the whole tree into one Python
function, so expressions run as fast as hand-written rate functions.
`Expr.evaluate()` computes a node directly from its definition and is the
only method a custom `Expr` subclass needs.

::: neuroclamp.rates
