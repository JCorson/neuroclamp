# Changelog

All notable user-facing changes to neuroclamp are recorded here: new features,
changes in behavior, changes to the `neuroclamp` Python API, and fixes that
change simulation results. Internal refactors, test, CI, and dependency updates
are not listed; see the pull request history for those.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, minor releases may include breaking API changes; these are always
listed under **Changed** or **Removed**.

## [Unreleased]

### Added

- Rate expressions in `neuroclamp.rates` for defining gating kinetics as data
  rather than as Python functions: the forms `Exp`, `Sigmoid`, `ExpLinear`,
  `Sech`, `Hill` (Ca²⁺-dependent) and `Const`, combined with `+`, `-`, `*`,
  `/` and `Max`, plus `inf_tau()` to build a gate's rates from its steady
  state and time constant. Expressions can be used anywhere a rate function
  can, compare by value, and pickle for `simulate_batch`. Each expression
  compiles to a single Python function on first use, so it runs as fast as a
  hand-written rate function. A custom `Expr` subclass only needs to define
  `evaluate()`.
- `simulate_current_clamp()`, `simulate_voltage_clamp()` and their
  `*_from_state()` variants take `backend="rust"` to run on a compiled Rust
  integrator, typically 30 to 90 times faster, with the same result to within
  rounding.
  The default stays `backend="python"`. Every gate must be built from rate
  expressions, as all built-in channels are.
- `simulate_batch()` takes `backend="rust"` too: the whole protocol family runs
  in one call on the compiled core, its sweeps in parallel on a thread pool,
  typically 35 to 80 times faster than the process pool. It covers
  `simulate_current_clamp` and `simulate_voltage_clamp`, and sweeps may differ
  in length. `max_workers` caps its threads.
- `Rate.uses_calcium` reports whether a rate depends on intracellular Ca²⁺.
  Expressions derive it automatically; a custom `Rate` subclass that does not
  declare it is treated as Ca²⁺-dependent.

### Changed

- `neuroclamp` now ships platform-specific wheels that include the compiled
  simulation core, for Linux (x86_64 and aarch64), macOS (Intel and Apple
  silicon) and Windows (x86_64). Each wheel serves Python 3.12 and later.
  Installing from the source distribution on any other platform needs a Rust
  toolchain.
- **Breaking:** every built-in channel's gating rates are now rate
  expressions. Public rate objects such as `neuroclamp.channels.alpha_m`,
  `pospischil_alpha_m` and `mainen_sejnowski_beta_h` are `neuroclamp.rates`
  expression nodes instead of `VoltageOnlyFn` wrappers around Python
  functions, and `boltzmann_cosh_rates()` returns the expressions built by
  `inf_tau()`. Calling a rate as `rate(V, ca_i)` works as before, and
  simulation results are unchanged apart from floating-point rounding. Code
  that reads a built-in rate's `.fn`, checks `isinstance(rate, VoltageOnlyFn)`,
  or reads fields of the old `boltzmann_cosh_rates()` results (`half`,
  `slope`, `is_alpha`, ...) must be updated.
- **Breaking:** `initial_ca_i` in `simulate_voltage_clamp_from_state()` and
  `simulate_current_clamp_from_state()` now defaults to the Ca²⁺ a fresh run
  starts from (`ca_init` or `ca_rest` with Ca²⁺ dynamics, `Ca_in` without)
  instead of 0.0. Calls that omit it on a neuron with Ca²⁺ channels or Ca²⁺
  dynamics give different results; pass `initial_ca_i=0.0` to keep the old
  starting point.

### Removed

- **Breaking:** `neuroclamp.electrochemistry.BoltzmannCoshRate`. Use the rates
  returned by `boltzmann_cosh_rates()`, which are ordinary rate expressions.

### Fixed

- `simulate_batch()` accepts the 2-D array from `build_protocol_from_preset()`
  directly, one sweep per row, instead of raising a `ValueError` about an
  ambiguous truth value. Invalid arguments (`backend`, `max_workers`, or a
  `simulate_fn` the Rust backend cannot run) now raise when `simulate_batch()`
  is called rather than when its results are first read.
- A neuron with Ca²⁺ channels but no `calcium_dynamics` now holds
  intracellular Ca²⁺ at its `Ca_in` instead of at zero. Before, its Ca²⁺
  channels saw an E_Ca of about +286 mV instead of about +125 mV, so their
  inward current was far too large. This changes results for such neurons in
  `simulate_voltage_clamp`, `simulate_current_clamp` and continuous mode in
  the app; the built-in presets are unaffected. `find_zero_current_voltage()`
  and `find_coupled_equilibrium()` use the same `Ca_in`, so the resting
  potential they return no longer drifts when simulated, and
  `find_coupled_equilibrium()` returns `Ca_in` instead of 0.0 as its Ca²⁺.

## [0.1.0] - 2026-11-25

The first release of neuroclamp, a patch-clamp experiment simulator built on
conductance-based (Hodgkin-Huxley) neuron models. It ships as a Python library
(`neuroclamp`) and an interactive web app. Requires Python 3.12 or newer.

### Added

#### Neuron model

- `Neuron`, a single-compartment conductance-based model assembled from a list
  of ion channels, with configurable membrane capacitance, ion concentrations,
  and temperature.
- Reversal potentials computed from ion concentrations with the Nernst equation
  (`nernst_potential`), and with the Goldman-Hodgkin-Katz equation
  (`goldman_potential`) for mixed-cation channels such as Ih.
- Q10 temperature scaling of channel gating rates.
- Optional intracellular calcium dynamics (`CalciumDynamics`), coupled to the
  calcium-permeable channels; the calcium reversal potential tracks the
  intracellular calcium concentration as it changes.
- Resting-state solvers: `find_zero_current_voltage` and
  `find_coupled_equilibrium`.
- Channels beyond the classic Na⁺, K⁺, and leak: Ih, IKA, Kv3.1, INaP, INaR,
  IM, IKir, IKATP, IKCa (SK), ICaL, ICaT, and ICaN, along with cell-type-specific
  Na⁺/K⁺ kinetics from the published models the presets are based on.

#### Neuron presets

- Nine biologically tuned cell types, available from `NEURON_PRESETS` or as
  `make_*` factory functions: Squid Giant Axon (Classic HH), Fast-Spiking
  Interneuron, Cortical Pyramidal Neuron, Purkinje Neuron, SNc Dopaminergic
  Neuron, Thalamic Relay Neuron, Hippocampal CA1 Pyramidal Neuron, Subthalamic
  Nucleus Neuron, and Thalamic Reticular Nucleus Neuron. Each is tuned against
  published action-potential shape, resting potential, and firing pattern
  (tonic pacemaking, spike-frequency adaptation, rebound bursting, and so on).
  Known limitations are listed per preset in the documentation.

#### Protocols and simulation

- Current-clamp protocols: step, ramp, pulse train, sinusoidal, chirp (ZAP),
  and noise.
- Voltage-clamp protocols: step, ramp, pulse train, and two-pulse steady-state
  inactivation.
- Multi-sweep step protocols for F-I, I-V, and hyperpolarization-step series.
- Nine protocol presets (Action Potential, Subthreshold Response, Repetitive
  Firing, F-I Curve, I-V Curve, Na+ Channel Activation, Steady-State
  Inactivation, Frequency Response, and Hyperpolarization Steps), built with
  `build_protocol_from_preset` and automatically adjusted to each neuron
  preset's physiological range.
- `simulate_current_clamp`, `simulate_voltage_clamp`, and `simulate_batch`, plus
  `*_from_state` variants that resume from a saved state. All return a
  `SimulationResult` with voltage, per-channel currents, gating variables, and
  intracellular calcium.

#### Analysis

- `neuroclamp.analysis`, which turns traces into physiological metrics, each with
  a typed result object:
  - Action potentials: threshold, peak, rise time, half-width, AHP depth, ISI,
    and firing rate.
  - F-I curves and rheobase estimation.
  - I-V curves, G-V activation curves with Boltzmann fits, and steady-state
    inactivation (h∞) curves.
  - Activation and inactivation time constants versus voltage (τ-V).
  - Spike-frequency adaptation.
  - Burst detection and burst metrics.
  - Calcium transients: peak, time to peak, and decay τ.
  - Sag and rebound from hyperpolarizing steps.
  - Impedance profile, resonance frequency, and Q factor from chirp protocols.
  - Passive properties (input resistance, membrane time constant, capacitance)
    from a built-in membrane test, reported both per unit area and as absolute
    values (MΩ, pF) for the cell's surface area.

#### Web app

- An interactive simulator inspired by pClamp, with panels for configuring the
  neuron and protocol, a trace plot, and analysis.
- Neuron and protocol preset selectors, conductance sliders for every channel
  in the model, and editable membrane properties and ion concentrations.
- Overlaid multi-sweep traces with click, hover, and keyboard sweep selection;
  per-channel current and gating-variable traces; and stored reference traces
  for comparing runs.
- A continuous mode that re-runs the simulation in a loop, like an oscilloscope.
- An analysis panel that switches with the clamp mode: AP metrics, phase plane,
  and F-I curve; bursts; calcium transients; impedance; and I-V, g-V, h∞, and
  τ-V plots.
- Export of the latest run, with every simulated quantity, to CSV.
- A built-in user guide at `/help`.
- A log viewer, a dark mode toggle, and a Reset button.
- A Docker image and Compose file for running the app without installing
  Python, uv, or Node.

#### Installation

- neuroclamp installs from PyPI: `pip install neuroclamp` for the library, or
  `pip install "neuroclamp[ui]"` for the library and the web app. The library
  ships type annotations (`py.typed`) for type checkers.
- The `neuroclamp ui` command starts the web app from any directory, with
  `--port`, `--host`, `--no-browser`, `--workdir`, `--reset` and `--verbose`
  options. It opens the app in your browser once the server is ready.

#### Documentation

- A library documentation site covering the presets, protocols, and analysis,
  with a generated API reference.

[Unreleased]: https://github.com/JCorson/neuroclamp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JCorson/neuroclamp/releases/tag/v0.1.0
