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

**Agreement** with `neuroclamp.simulate_current_clamp`, compared field by
field (voltage, every channel current, `Itotal`, every gate, `ca_i`):

- All 9 presets, including the five with Ca²⁺ dynamics: voltage within
  ~1e-10 mV on a 150 ms step. Squid differs by 3.8e-12 mV, because powers of
  3 and up use `powi` instead of C `pow` (see `int_pow` in `src/expr.rs`).
- Golden traces (`tests/parity/golden/`): all 90 current-clamp runs pass,
  using at most 2% of the tolerance band. Check with
  `uv run --frozen python spikes/rate_expr/verify_golden.py --rust-only`.

**Per step** (`cargo run --release --no-default-features --example bench`):

| Preset | Python | Rust, first port | Rust now |
|---|---|---|---|
| Squid | ~23 µs | 595 ns | **231 ns** |
| Purkinje | ~132 µs | 2,815 ns | **1,106 ns** |
| CA1 | n/a | 3,296 ns | **1,294 ns** |
| STN | n/a | 3,128 ns | **1,182 ns** |

That is ~100× per step against Python. What changed from the first port,
measured one change at a time:

1. Reuse the k1 currents for the output row, and compute E_Ca once per
   evaluation (bit-identical): squid 595 → 307 ns.
2. Evaluate `inf` and `tau` once for steady-state gates (the `inf_tau` form,
   bit-identical): Purkinje, CA1 and STN −34 to −38%.
3. Skip C `pow` for powers 1 and 2 (bit-identical): heavy presets −30%.
4. `powi` for powers 3 and up (not bit-identical): a further 6 to 15%.

Tried and dropped: the `libm` crate instead of glibc (15 to 35% slower), and
precomputed reciprocal voltage scales (0 to 7%, not worth the semantics). The
remaining cost is mostly `exp` itself and the recursive tree walk.

**Whole protocol families** (`uv run --frozen python spikes/rate_expr/benchmark_batch.py`),
wall time from Python:

| Family | Python serial | `neuroclamp.simulate_batch` | Rust batch (`rayon`) |
|---|---|---|---|
| Squid F-I, 9 sweeps | 0.54 s | 0.19 s | **3.9 ms** |
| Purkinje F-I, 9 sweeps | 3.08 s | 1.03 s | **14.1 ms** |
| CA1 F-I, 11 sweeps | 10.44 s | 2.96 s | **52.6 ms** |
| CA1 hyperpolarization steps, 5 × 900 ms | 12.97 s | 5.09 s | **96.9 ms** |

## Profiling

```bash
cd spikes/neuroclamp_core
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --no-default-features --example profile
valgrind --tool=callgrind target/release/examples/profile ca1
callgrind_annotate --inclusive=no callgrind.out.<pid> | head -30
```

Benchmark fixtures in `bench_fixtures/` are regenerated with
`uv run --frozen python spikes/rate_expr/dump_bench_fixtures.py` whenever
presets or the spec format change.

## Deliberate limits

- Current clamp only; voltage clamp is not ported.
- A Ca²⁺-carrying channel without Ca²⁺ dynamics is rejected (`neuroclamp` falls
  back to a 1e-9 mM floor there; no preset does this).
- The spec is JSON-encoded on every call (0.05 to 0.4 ms) to key the model
  cache in `backend.py`.
