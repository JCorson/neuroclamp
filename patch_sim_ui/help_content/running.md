# Running & reading results

## Running a simulation

Click **Run** in the top bar to simulate once with the current settings. While
it computes, the bar shows a *Running…* indicator; when it finishes, the trace
and analysis update.

**Continuous** loops the simulation for a live, oscilloscope-like view (useful
while dragging a slider). Click **Stop** to end it. **Reset** returns every
parameter to the defaults for the current neuron and protocol.

## The trace plot

The center plot shows the recorded response against time (ms):

- **Current clamp** plots membrane **voltage (mV)**.
- **Voltage clamp** plots **current (µA/cm²)**.

![A simulated action-potential trace](/screenshots/trace-plot.png)

Multi-sweep protocols (F-I, I-V, hyperpolarization steps) overlay one colored
trace per sweep:

![A multi-sweep F-I protocol](/screenshots/fi-curve.png)

## Sweep and trace controls

The controls beneath the plot manage what is shown:

![The sweep and trace controls](/screenshots/sweep-manager.png)

- **Traces** (the eye icon) opens a popover of checkboxes to show or hide
  individual traces — the voltage/current, per-channel currents, and gating
  variables available for the current model and mode.
- The **crosshair** icon toggles hover tooltips on the plot.
- **Store** saves the current trace as a translucent reference overlay (labeled
  *Stored 1*, *Stored 2*, …) so you can compare it against later runs.
  **Clear Stored** removes the overlays. Storing is unavailable for multi-sweep
  protocols.
- **Logs** opens a panel of simulation messages (warnings, errors, debug). You
  can filter by level and clear it.
- **Export CSV** downloads the latest result as a CSV file (see below). It is
  disabled until a simulation has run, and while one is running.

## Exporting results

**Export CSV** saves the most recent run as
`patch-sim_<neuron>_<protocol>_<timestamp>.csv`. The file has one row per time
sample and one column per quantity:

- `time_ms` first.
- Current clamp: membrane voltage (`V_mV`) and the injected stimulus
  (`Istim_uA_cm2`). Voltage clamp: the command voltage (`Vcmd_mV`), since the
  recorded voltage equals it.
- Then the total current, each channel's current, the gating variables, and
  intracellular calcium (`ca_i_mM`) when the model tracks it.
- Multi-sweep protocols add a `_s01`, `_s02`, … suffix per sweep, with each
  quantity's sweeps side by side.

Every simulated quantity is exported, whether or not it is visible on the plot.
dV/dt and stored reference overlays are not included. A block of `# key: value`
comment lines above the header records the neuron, protocol, and sampling
settings that produced the data. Most spreadsheet and analysis tools can skip
these lines (for example, `pandas.read_csv(path, comment="#")`).

To save an image of a plot instead, use the camera icon in the plot's toolbar.
