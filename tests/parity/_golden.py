"""Golden-trace references for parity checks: generation, storage, comparison.

A golden trace is a stored copy of what the library produces for one sweep of
one protocol preset on one neuron preset, run exactly as the app runs it
(:func:`neuroclamp.build_protocol_from_preset` with the neuron's protocol
adjustments, then a fresh ``simulate_current_clamp`` or
``simulate_voltage_clamp`` per sweep).

The references are a **baseline, not a definition of correct output**. They
answer one question: does a change that should be behavior-preserving (an
integrator refactor, rate functions expressed as data, an alternative backend)
still reproduce the baseline? A mismatch means the output changed, not that it
is wrong; correctness is checked by the behavioral tests in
``tests/integration``. Intentional changes to presets or kinetics are expected
to make these references stale.

Coverage is every neuron preset × every protocol preset, using the first,
middle, and last sweep of each protocol to keep the runtime manageable.

**Storage.** Each run stores its recorded fields (``voltage`` in current
clamp, ``Itotal``, and ``ca_i`` when the neuron has Ca²⁺ dynamics) sampled
every :data:`STRIDE` steps, plus a per-sample tolerance band. Voltage in
voltage clamp is the command waveform, so it is not stored.

**Tolerance.** Spontaneously firing cells (e.g. the Purkinje pacemaker) carry
phase errors forward from spike to spike, so last-bit arithmetic differences
shift late spikes by about a nanosecond. On a spike upstroke near 400 mV/ms
that is a 5e-4 mV pointwise difference, even though nothing physiological
changed. The band therefore allows a tiny timing shift as well as a small
absolute error::

    band = ATOL[field] + RTOL · |x| + TIME_TOL_MS · max|dx/dt| (nearby samples)

Any real change to the kinetics moves spikes far more than
:data:`TIME_TOL_MS` and changes subthreshold voltages far more than
:data:`ATOL`.

**Workflow.** The baseline must reflect what ``main`` produces. From the work
branch, generate it with ``main``'s code in a temporary worktree, writing into
this branch's ``golden/`` directory, then run the parity tests::

    git worktree add ../neuroclamp-main main
    (cd ../neuroclamp-main && uv run --frozen python -m tests.parity._golden \
        --out "$OLDPWD/tests/parity/golden")
    git worktree remove ../neuroclamp-main
    uv run --frozen -m pytest tests/parity

Generation is byte-reproducible, so the files change only when ``main``'s
output differs from the committed baseline. Committing the refreshed files on
the work branch is correct: they still record ``main``'s output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np

from neuroclamp import (
    NEURON_PRESETS,
    PROTOCOL_PRESETS,
    build_protocol_from_preset,
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from neuroclamp.clamp_simulations import Backend
from neuroclamp.constants import CURRENT_CLAMP, SIM_SAMPLING_FREQ

#: Directory holding one ``.npz`` file per neuron preset plus ``manifest.json``.
GOLDEN_DIR = Path(__file__).parent / "golden"

#: Store every ``STRIDE``-th simulation step (40 → 1 kHz at the 40 kHz rate).
STRIDE = 40

#: Absolute tolerance floor per field, in the field's units (mV, µA/cm², mM).
#: Arithmetic reordering leaves ~1e-6 mV between spikes after 900 ms of
#: Purkinje pacemaking; these floors keep a 10× margin above that.
ATOL: dict[str, float] = {"voltage": 1e-5, "Itotal": 1e-5, "ca_i": 1e-10}

#: Relative tolerance applied to each sample's magnitude.
RTOL = 1e-9

#: Allowed timing shift in ms, converted to an amplitude via the local slope.
TIME_TOL_MS = 1e-4

#: Half-width, in simulation steps, of the window used for the local slope.
SLOPE_HALF_WINDOW = 2

_DT_MS = 1000.0 / SIM_SAMPLING_FREQ


@dataclass(frozen=True)
class GoldenRun:
    """One stored sweep: a neuron preset, a protocol preset, and a sweep index.

    Attributes:
        neuron: Key in :data:`neuroclamp.NEURON_PRESETS`.
        protocol: Key in :data:`neuroclamp.PROTOCOL_PRESETS`.
        sweep: Index of the sweep within the built protocol.
        clamp_mode: ``"Current Clamp"`` or ``"Voltage Clamp"``.
    """

    neuron: str
    protocol: str
    sweep: int
    clamp_mode: str

    @property
    def test_id(self) -> str:
        """Short, readable identifier for pytest parametrization.

        Returns:
            ``"<neuron>|<protocol>|<sweep>"``.
        """
        return f"{self.neuron}|{self.protocol}|{self.sweep}"

    @property
    def key(self) -> str:
        """Key prefix for this run's arrays inside the neuron's ``.npz`` file.

        Returns:
            ``"<protocol_slug>/<sweep>"``.
        """
        return f"{_slug(self.protocol)}/{self.sweep}"


def _slug(name: str) -> str:
    """Turn a preset name into a lowercase, filesystem-safe identifier.

    Args:
        name: A neuron or protocol preset name.

    Returns:
        The name with runs of non-alphanumeric characters replaced by ``_``.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def golden_file(neuron: str) -> Path:
    """Path of the reference file for a neuron preset.

    Args:
        neuron: Key in :data:`neuroclamp.NEURON_PRESETS`.

    Returns:
        Path to ``golden/<neuron_slug>.npz``.
    """
    return GOLDEN_DIR / f"{_slug(neuron)}.npz"


@lru_cache(maxsize=None)
def _protocol(neuron: str, protocol: str) -> np.ndarray:
    """Build (and cache) the sweep array for a neuron/protocol pair.

    Args:
        neuron: Key in :data:`neuroclamp.NEURON_PRESETS`.
        protocol: Key in :data:`neuroclamp.PROTOCOL_PRESETS`.

    Returns:
        Array of shape ``(n_sweeps, n_samples)``.
    """
    return build_protocol_from_preset(protocol, neuron)


def golden_runs() -> list[GoldenRun]:
    """List every run covered by the golden traces.

    Returns:
        One entry per neuron preset × protocol preset × selected sweep (first,
        middle, last), in a stable order.
    """
    runs = []
    for neuron in NEURON_PRESETS:
        for protocol, config in PROTOCOL_PRESETS.items():
            n_sweeps = _protocol(neuron, protocol).shape[0]
            for sweep in sorted({0, n_sweeps // 2, n_sweeps - 1}):
                runs.append(GoldenRun(neuron, protocol, sweep, config["clamp_mode"]))
    return runs


def simulate_run(run: GoldenRun, backend: Backend = "python") -> np.ndarray:
    """Simulate one golden run with the library, as the app would.

    Args:
        run: The run to simulate.
        backend: Simulation backend.

    Returns:
        The structured simulation result.
    """
    neuron = NEURON_PRESETS[run.neuron]()
    stimulus = _protocol(run.neuron, run.protocol)[run.sweep]
    if run.clamp_mode == CURRENT_CLAMP:
        return simulate_current_clamp(neuron, stimulus, backend=backend)
    return simulate_voltage_clamp(neuron, stimulus, backend=backend)


def compared_fields(run: GoldenRun, result: np.ndarray) -> list[str]:
    """Fields stored and compared for a run.

    Args:
        run: The run.
        result: Its structured simulation result.

    Returns:
        ``voltage`` (current clamp only), ``Itotal``, and ``ca_i`` if present.
    """
    names = result.dtype.names or ()
    fields = ["voltage"] if run.clamp_mode == CURRENT_CLAMP else []
    fields.append("Itotal")
    if "ca_i" in names:
        fields.append("ca_i")
    return fields


def stored_fields(run: GoldenRun, stored: dict[str, np.ndarray]) -> list[str]:
    """Fields the baseline holds for a run.

    Args:
        run: The run.
        stored: Every array stored for the run's neuron (see :func:`_load`).

    Returns:
        Field names with a stored value array under ``run.key``, sorted.
    """
    prefix = f"{run.key}/"
    return sorted(
        key.removeprefix(prefix)
        for key in stored
        if key.startswith(prefix) and not key.endswith(".band")
    )


def sample_indices(n_samples: int) -> np.ndarray:
    """Indices of the stored samples: every ``STRIDE``-th step plus the last.

    Args:
        n_samples: Number of simulation steps in the trace.

    Returns:
        Sorted, unique sample indices.
    """
    return np.unique(np.append(np.arange(0, n_samples, STRIDE), n_samples - 1))


def tolerance_band(field: str, trace: np.ndarray) -> np.ndarray:
    """Per-step tolerance for a full-resolution trace (see module docstring).

    Args:
        field: Field name, used to pick the absolute floor from :data:`ATOL`.
        trace: Full-resolution values of the field.

    Returns:
        Allowed absolute deviation at every step.
    """
    slope = np.abs(np.gradient(trace, _DT_MS))
    window = 2 * SLOPE_HALF_WINDOW + 1
    padded = np.pad(slope, SLOPE_HALF_WINDOW, mode="edge")
    local_max = np.lib.stride_tricks.sliding_window_view(padded, window).max(axis=1)
    return ATOL[field] + RTOL * np.abs(trace) + TIME_TOL_MS * local_max


def _reference_arrays(run: GoldenRun) -> dict[str, np.ndarray]:
    """Simulate a run and reduce it to the arrays stored on disk.

    Args:
        run: The run to simulate.

    Returns:
        ``{"<key>/<field>": values, "<key>/<field>.band": band}`` for each
        compared field; values are float64 and bands float32.
    """
    result = simulate_run(run)
    idx = sample_indices(len(result))
    arrays: dict[str, np.ndarray] = {}
    for field in compared_fields(run, result):
        trace = np.asarray(result[field], dtype=np.float64)
        arrays[f"{run.key}/{field}"] = trace[idx]
        band = tolerance_band(field, trace)[idx]
        arrays[f"{run.key}/{field}.band"] = band.astype(np.float32)
    return arrays


@lru_cache(maxsize=None)
def _load(neuron: str) -> dict[str, np.ndarray]:
    """Load (and cache) every stored array for a neuron preset.

    Args:
        neuron: Key in :data:`neuroclamp.NEURON_PRESETS`.

    Returns:
        Mapping of array key to array.

    Raises:
        FileNotFoundError: If the reference file has not been generated.
    """
    with np.load(golden_file(neuron)) as data:
        return {key: data[key] for key in data.files}


def compare(run: GoldenRun, result: np.ndarray) -> list[str]:
    """Compare a simulation result against its stored golden trace.

    The fields checked are the union of those the baseline holds and those the
    result produces, so a field that disappears from the output or appears
    without a reference is reported rather than skipped. Non-finite samples
    always fail.

    Args:
        run: The run that produced *result*.
        result: Structured simulation result (any backend).

    Returns:
        One human-readable message per field that is missing or outside its
        tolerance band; empty when the result matches.
    """
    stored = _load(run.neuron)
    expected_fields = stored_fields(run, stored)
    if not expected_fields:
        return [f"no stored reference for {run.key!r}"]
    produced = set(compared_fields(run, result))
    idx = sample_indices(len(result))
    failures = []
    for field in sorted(produced - set(expected_fields)):
        failures.append(f"{field}: no stored reference ({run.key}/{field!r})")
    for field in expected_fields:
        key = f"{run.key}/{field}"
        if field not in produced:
            failures.append(f"{field}: stored in the baseline but missing from result")
            continue
        if f"{key}.band" not in stored:
            failures.append(f"{field}: stored reference has no band ({key}.band)")
            continue
        expected, band = stored[key], stored[f"{key}.band"].astype(np.float64)
        if len(expected) != len(idx) or len(band) != len(idx):
            failures.append(
                f"{field}: {len(idx)} stored samples expected, "
                f"reference has {len(expected)} values and {len(band)} bands"
            )
            continue
        actual = np.asarray(result[field], dtype=np.float64)[idx]
        excess = np.abs(actual - expected) / band
        # NaN compares False against anything, so test for being inside the
        # band and treat everything else (including NaN and inf) as outside.
        outside = ~(excess <= 1.0)
        if np.any(outside):
            worst = int(np.argmax(np.where(np.isfinite(excess), excess, np.inf)))
            size = (
                f"{excess[worst]:.3g}× the allowed deviation"
                if np.isfinite(excess[worst])
                else "not a finite value"
            )
            failures.append(
                f"{field}: {int(np.sum(outside))}/{len(idx)} samples outside "
                f"tolerance; worst at t={idx[worst] * _DT_MS:.3f} ms: "
                f"got {actual[worst]:.9g}, expected {expected[worst]:.9g}, {size}"
            )
    return failures


def manifest_settings() -> dict[str, object]:
    """Comparison settings recorded in the manifest.

    The stored bands are computed from these at generation time, so the
    manifest's copy must match the constants for the constants to take effect.

    Returns:
        The stride and tolerance settings, JSON-serializable.
    """
    return {
        "stride": STRIDE,
        "atol": ATOL,
        "rtol": RTOL,
        "time_tol_ms": TIME_TOL_MS,
        "slope_half_window": SLOPE_HALF_WINDOW,
    }


def generate(out_dir: Path = GOLDEN_DIR) -> None:
    """Regenerate every reference file and the manifest from the library.

    Args:
        out_dir: Directory to write the ``.npz`` files and ``manifest.json``
            into. Existing ``.npz`` files there are removed first.
    """
    runs = golden_runs()
    with ProcessPoolExecutor() as pool:
        reduced = list(pool.map(_reference_arrays, runs))

    by_neuron: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for run, arrays in zip(runs, reduced, strict=True):
        by_neuron[run.neuron].update(arrays)

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.npz"):
        stale.unlink()
    for neuron, arrays in by_neuron.items():
        path = out_dir / golden_file(neuron).name
        np.savez_compressed(path, allow_pickle=False, **arrays)

    manifest = {
        "neuroclamp_version": version("neuroclamp"),
        **manifest_settings(),
        "runs": [asdict(run) for run in runs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    print(f"Wrote {len(runs)} runs for {len(by_neuron)} neurons ({total / 1e6:.2f} MB)")


def main() -> None:
    """Command-line entry point: regenerate the baseline."""
    parser = argparse.ArgumentParser(description="Regenerate the golden traces.")
    parser.add_argument(
        "--out",
        type=Path,
        default=GOLDEN_DIR,
        help="Directory to write into (default: this checkout's golden/).",
    )
    generate(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
