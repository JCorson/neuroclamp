# Rate functions as expression trees (spike)

Throwaway prototype for [#410](https://github.com/JCorson/patch-sim/issues/410)
(exploring a Rust simulation core). It validates the declarative rate spec and
the `lower(neuron)` step described in the
[design comment](https://github.com/JCorson/patch-sim/issues/410#issuecomment-7194687739).

This directory is **not** part of the `patch_sim` package. Nothing imports it,
and it is outside the pytest and ty paths. When the design lands for real, the
node classes move into `patch_sim/rates.py` and this directory should be deleted.

## Files

| File | Purpose |
|---|---|
| `exprs.py` | Expression-tree nodes (`Const`, `Exp`, `Sigmoid`, `ExpLinear`, `Sech`, `Hill`, `Add`/`Sub`/`Mul`/`Div`/`Max`) and the `inf_tau`, `boltzmann_cosh`, `traub_miles` helpers. Every node is a callable `Rate`. |
| `translate.py` | Maps each existing rate object in `patch_sim.channels` to its tree, and rebuilds a `Neuron` with tree gates. |
| `codegen.py` | Compiles a tree once into a flat Python `lambda`, so tree evaluation adds no overhead on the Python backend. |
| `lower.py` | `lower(neuron)`: flattens a neuron into the JSON-able spec a compiled backend would consume. |
| `verify_rates.py` | Pointwise comparison of all 106 module-level rates against their trees, including the exp-linear singular points. |
| `verify_sims.py` | Runs every preset in current and voltage clamp with tree gates and compares against today's rates. |
| `benchmark.py` | Times today's rates vs. trees vs. codegen, with the `uses_calcium` tabulation fix patched in. |
| `backend.py` | `simulate_current_clamp(neuron, current, backend="python" \| "rust")`. The Rust path lowers the neuron and runs `spikes/patch_sim_core`, returning the library's structured array. |
| `verify_rust.py` | Compares every result field from the two backends on all presets, and times them. Needs the extension built (see `spikes/patch_sim_core/README.md`). |

## Running

`patch_sim` is not installed into the virtualenv, so put the repo root on the path:

```bash
PYTHONPATH=. uv run --frozen python spikes/rate_expr/verify_rates.py
PYTHONPATH=. uv run --frozen python spikes/rate_expr/verify_sims.py
PYTHONPATH=. uv run --frozen python spikes/rate_expr/lower.py
PYTHONPATH=. uv run --frozen python spikes/rate_expr/benchmark.py
PYTHONPATH=. uv run --frozen python spikes/rate_expr/verify_rust.py  # after building the crate
```

## Results (2026-11-26, 4-core container)

- All 106 module-level rates translate; worst pointwise relative error is
  about 2e-11, next to an exp-linear singularity.
- All 9 presets reproduce current-clamp voltage to within about 1e-10 mV, and
  voltage-clamp currents exactly.
- Lowered specs are 1 to 8 KB of JSON.
- Python backend: trees alone cost up to about 20% in current clamp; codegen
  removes that and matches today's speed (within run-to-run noise; one run
  came out about 9% faster on Purkinje, another about 3% slower). Voltage clamp needs the
  `uses_calcium` tabulation fix to avoid a 1.4 to 1.8× slowdown.

## Known shortcuts

- `inf_tau` expands to plain `Div` trees, so `inf` and `tau` appear twice in
  the lowered spec. The design calls for a paired `InfTauAlpha`/`InfTauBeta`
  so lowering can emit an explicit `inf_tau` form.
- `translate.py` keys its registry on object identity and reaches into private
  module names. It exists only to prove coverage, not as a migration path.
