# neuroclamp_core (spike)

A compiled current-clamp integrator for `neuroclamp`, written in Rust with
[PyO3](https://pyo3.rs/). Step 3 of
[#410](https://github.com/JCorson/patch-sim/issues/410). It consumes the model
spec produced by `spikes/rate_expr/lower.py` and is selected with
`backend="rust"` in `spikes/rate_expr/backend.py`.

Like `spikes/rate_expr/`, this is throwaway exploration. `neuroclamp` does not
depend on it, and nothing here is built by CI or the Docker image.

## Reading order

If you are new to Rust, read the source in this order. Each file is short and
commented for a Python reader.

1. `src/expr.rs`: the rate-function expression tree as a Rust `enum`, and
   `eval` as one exhaustive `match`. Compare with `spikes/rate_expr/exprs.py`.
2. `src/spec.rs`: serde structs mirroring the JSON spec, plus validation.
3. `src/current_clamp.rs`: the RK4 integrator, a line-by-line port of
   `_rk4_step_current_clamp` in `neuroclamp/clamp_simulations.py`.
4. `src/python.rs`: the PyO3 bindings. This is the only file that knows about
   Python; the other three build and test without it.

## Building

Requires a Rust toolchain ([rustup](https://rustup.rs/)). From the repo root:

```bash
# Build an optimized extension and install it into the project venv.
VIRTUAL_ENV=.venv uvx maturin develop --release --uv -m spikes/neuroclamp_core/Cargo.toml
```

maturin handles the platform details (including macOS linker flags), so this
works the same on Linux and macOS. The installed package is not in `uv.lock`,
so a later `uv sync` removes it; rerun the command above to reinstall.

## Testing

```bash
cd spikes/neuroclamp_core
cargo test --no-default-features      # pure-Rust unit tests, no Python needed
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# From the repo root, after building: compare against the Python library.
uv run --frozen python spikes/rate_expr/verify_rust.py
```

`--no-default-features` turns off the `python` feature, so the test binary
does not need to link against libpython.

## Results (2026-11-26, 4-core container)

150 ms current step, compared field by field (voltage, every channel current,
`Itotal`, every gate, `ca_i`) against `neuroclamp.simulate_current_clamp`:

- **Squid: bit-identical** (max difference 0.0 in every field).
- All 9 presets, including the five with Ca²⁺ dynamics: voltage within
  1.2e-10 mV.

Speed (squid / Purkinje):

| | Squid | Purkinje |
|---|---|---|
| Python backend, 150 ms | 134 ms | 717 ms |
| Rust backend end to end, 150 ms | 4.4 ms | 19.4 ms |
| Rust integrator alone, per step | 591 ns | 2.9 µs |

That is ~40× per step, below the 100 to 200× first estimated on #410. Swapping
`powf` for `powi` on gate powers saves only ~20% (and breaks bit-identity), so
the remaining cost is probably the ~25 `exp` calls per step and the recursive
tree walk. Profiling is the benchmarking step on #410.

## Deliberate limits

- Current clamp only; voltage clamp is not ported.
- Gates use the `alpha_beta` form. The explicit `inf_tau` form from the design
  is not implemented, so `inf`/`tau` subtrees are evaluated twice for those gates.
- A Ca²⁺-carrying channel without Ca²⁺ dynamics is rejected (`neuroclamp` falls
  back to a 1e-9 mM floor there; no preset does this).
- One simulation per call. Batching sweeps across threads (`rayon`) is future
  work; the bindings already release the GIL with `py.detach`.
- The spec is JSON-encoded on every call (0.05 to 0.4 ms) to key the model
  cache in `backend.py`.
