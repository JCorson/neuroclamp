# Changelog

All notable user-facing changes to patch-sim are recorded here: new features,
changes in behavior, changes to the `patch_sim` Python API, and fixes that
change simulation results. Internal refactors, test, CI, and dependency updates
are not listed; see the pull request history for those.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, minor releases may include breaking API changes; these are always
listed under **Changed** or **Removed**.

## [Unreleased]

## [0.1.0] - Unreleased

The first release of patch-sim, a patch-clamp experiment simulator built on
conductance-based (Hodgkin-Huxley) neuron models. It ships as a Python library
(`patch_sim`) and an interactive web app. Requires Python 3.12 or newer.

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

- `patch_sim.analysis`, which turns traces into physiological metrics, each with
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

#### Documentation

- A library documentation site covering the presets, protocols, and analysis,
  with a generated API reference.

[Unreleased]: https://github.com/JCorson/patch-sim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JCorson/patch-sim/releases/tag/v0.1.0
