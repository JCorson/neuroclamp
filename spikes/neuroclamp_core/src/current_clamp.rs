//! Fixed-step RK4 current-clamp integrator.
//!
//! This is a line-by-line port of `_simulate_current_clamp_core` and
//! `_rk4_step_current_clamp` in `neuroclamp/clamp_simulations.py`. The order of
//! floating-point operations follows the Python code so the two backends agree
//! to within rounding (~1e-10 mV after hundreds of milliseconds).
//!
//! Integrator contract (shared with the Python backend):
//! - State is `[V, x₁ … xₙ, Ca²⁺]`; gates are in `ModelSpec::gates` order.
//! - V is clamped to `v_clamp_mv` at every sub-step and after the full step.
//! - Gates are clipped to [0, 1] and Ca²⁺ is floored at 0 only after the full
//!   step, never at intermediate sub-steps.
//! - φ (the Q10 factor) multiplies every rate.

use crate::expr::{int_pow, py_max};
use crate::spec::{ChannelSpec, ModelSpec};

/// Simulation output as one row-major block, `n_steps × n_columns`.
///
/// Columns, matching the Python structured array after `time`:
/// `voltage`, one current per channel, `Itotal`, one value per gate, and
/// `ca_i` when the model has calcium dynamics.
#[derive(Debug, Clone)]
pub struct Trace {
    pub n_steps: usize,
    pub n_columns: usize,
    pub data: Vec<f64>,
}

/// A validated model ready to simulate.
#[derive(Debug, Clone)]
pub struct Model {
    spec: ModelSpec,
    /// Whether any channel carries Ca²⁺, so E_Ca must be evaluated.
    has_ca_channels: bool,
}

/// Python's `max(lo, min(hi, x))`, NaN-compatible with the Python backend.
fn py_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    let upper = if x < hi { x } else { hi }; // min(hi, x)
    if upper > lo { upper } else { lo } // max(lo, ·)
}

impl Model {
    /// Build a model from the JSON spec emitted by Python's `lower()`.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let spec = ModelSpec::from_json(json)?;
        let has_ca_channels = spec.channels.iter().any(|ch| ch.carries_calcium);
        Ok(Self {
            spec,
            has_ca_channels,
        })
    }

    /// Number of gating variables in the state vector.
    pub fn n_gates(&self) -> usize {
        self.spec.gates.len()
    }

    /// Number of output columns per time step (see [`Trace`]).
    pub fn n_columns(&self) -> usize {
        let ca = usize::from(self.spec.calcium.is_some());
        1 + self.spec.channels.len() + 1 + self.spec.gates.len() + ca
    }

    /// Live Ca²⁺ Nernst potential in mV, as in `IonChannel.compute_current`.
    fn e_ca(&self, ca: f64) -> f64 {
        // Validation guarantees calcium dynamics exist whenever a channel
        // carries Ca²⁺, so this branch is unreachable for valid specs.
        let Some(c) = &self.spec.calcium else {
            return f64::NAN;
        };
        let ca_in = py_max(ca, c.nernst_floor_mm);
        (c.gas_constant * c.temperature_k / (2.0 * c.faraday)) * 1000.0 * (c.ca_out / ca_in).ln()
    }

    /// One channel's current in µA/cm²: g_max · Π xᵖ · (V − E).
    ///
    /// `e_ca` is the live Ca²⁺ reversal potential, computed once per
    /// evaluation by the caller and used only by Ca²⁺-carrying channels.
    fn channel_current(&self, ch: &ChannelSpec, v: f64, gates: &[f64], e_ca: f64) -> f64 {
        let mut g = ch.g_max;
        for &(idx, power) in &ch.gates {
            g *= int_pow(gates[idx], power);
        }
        let e_rev = if ch.carries_calcium {
            e_ca
        } else {
            ch.e_rev.unwrap_or(f64::NAN)
        };
        g * (v - e_rev)
    }

    /// Evaluate dV/dt, dx/dt for every gate, and dCa/dt.
    ///
    /// Writes gate derivatives into `d_gates` and each channel's current into
    /// `currents` (both preallocated by the caller, so the hot loop never
    /// allocates) and returns `(dV, dCa)`.
    fn derivatives(
        &self,
        v: f64,
        gates: &[f64],
        ca: f64,
        i_ext: f64,
        d_gates: &mut [f64],
        currents: &mut [f64],
    ) -> (f64, f64) {
        let e_ca = if self.has_ca_channels {
            self.e_ca(ca)
        } else {
            f64::NAN
        };
        let mut i_total = 0.0;
        let mut i_ca = 0.0;
        for (ch, out) in self.spec.channels.iter().zip(currents.iter_mut()) {
            let i = self.channel_current(ch, v, gates, e_ca);
            *out = i;
            i_total += i;
            if ch.carries_calcium {
                i_ca += i;
            }
        }
        let dv = (i_ext - i_total) / self.spec.c_m;

        let phi = self.spec.phi;
        for ((gate, &x), d) in self.spec.gates.iter().zip(gates).zip(d_gates.iter_mut()) {
            let (alpha, beta) = gate.kinetics.rates(v, ca);
            *d = phi * (alpha * (1.0 - x) - beta * x);
        }

        let dca = match &self.spec.calcium {
            Some(c) => -c.alpha_ca * i_ca - (ca - c.ca_rest) / c.tau_ca,
            None => 0.0,
        };
        (dv, dca)
    }

    /// Write one output row from a state and its already-computed currents.
    fn record(&self, row: &mut [f64], v: f64, gates: &[f64], ca: f64, currents: &[f64]) {
        let n_ch = currents.len();
        row[0] = v;
        row[1..1 + n_ch].copy_from_slice(currents);
        // Same summation order as `derivatives`, so Itotal matches bit for bit.
        row[1 + n_ch] = currents.iter().fold(0.0, |acc, &i| acc + i);
        row[2 + n_ch..2 + n_ch + gates.len()].copy_from_slice(gates);
        if self.spec.calcium.is_some() {
            row[2 + n_ch + gates.len()] = ca;
        }
    }

    /// Run a current-clamp simulation.
    ///
    /// # Arguments
    /// * `i_ext` - injected current per time step in µA/cm²; its length sets
    ///   the number of steps.
    /// * `v0` - initial membrane voltage in mV.
    /// * `gates0` - initial gate values in `ModelSpec::gates` order.
    /// * `ca0` - initial intracellular Ca²⁺ in mM (ignored without dynamics).
    /// * `dt` - time step in ms.
    pub fn simulate_current_clamp(
        &self,
        i_ext: &[f64],
        v0: f64,
        gates0: &[f64],
        ca0: f64,
        dt: f64,
    ) -> Result<Trace, String> {
        let n_gates = self.n_gates();
        if gates0.len() != n_gates {
            return Err(format!(
                "expected {n_gates} initial gate values, got {}",
                gates0.len()
            ));
        }
        if i_ext.is_empty() {
            return Err("current_external must not be empty".into());
        }
        let (v_lo, v_hi) = self.spec.v_clamp_mv;
        let n_steps = i_ext.len();
        let n_cols = self.n_columns();
        let n_ch = self.spec.channels.len();
        let mut data = vec![0.0; n_steps * n_cols];

        // Current state, plus scratch buffers reused every step.
        let mut v = v0;
        let mut gates = gates0.to_vec();
        let mut ca = ca0;
        let (mut d1, mut d2, mut d3, mut d4) = (
            vec![0.0; n_gates],
            vec![0.0; n_gates],
            vec![0.0; n_gates],
            vec![0.0; n_gates],
        );
        let mut trial = vec![0.0; n_gates];
        // Currents at the current state (from the k1 evaluation, reused for
        // the output row) and a scratch buffer for the k2..k4 sub-steps.
        let mut currents = vec![0.0; n_ch];
        let mut scratch = vec![0.0; n_ch];

        let half = 0.5 * dt;
        let sixth = dt / 6.0;
        for step in 0..n_steps {
            let i = i_ext[step];

            // k1 is evaluated at the current state, which is also the state
            // this step's output row records, so its currents are reused.
            let (dv1, dca1) = self.derivatives(v, &gates, ca, i, &mut d1, &mut currents);
            self.record(
                &mut data[step * n_cols..(step + 1) * n_cols],
                v,
                &gates,
                ca,
                &currents,
            );
            if step + 1 == n_steps {
                break;
            }

            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d1) {
                *t = x + half * d;
            }
            let (dv2, dca2) = self.derivatives(
                py_clamp(v + half * dv1, v_lo, v_hi),
                &trial,
                ca + half * dca1,
                i,
                &mut d2,
                &mut scratch,
            );

            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d2) {
                *t = x + half * d;
            }
            let (dv3, dca3) = self.derivatives(
                py_clamp(v + half * dv2, v_lo, v_hi),
                &trial,
                ca + half * dca2,
                i,
                &mut d3,
                &mut scratch,
            );

            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d3) {
                *t = x + dt * d;
            }
            let (dv4, dca4) = self.derivatives(
                py_clamp(v + dt * dv3, v_lo, v_hi),
                &trial,
                ca + dt * dca3,
                i,
                &mut d4,
                &mut scratch,
            );

            v = py_clamp(v + sixth * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4), v_lo, v_hi);
            for (k, x) in gates.iter_mut().enumerate() {
                let next = *x + sixth * (d1[k] + 2.0 * d2[k] + 2.0 * d3[k] + d4[k]);
                *x = py_clamp(next, 0.0, 1.0);
            }
            ca = py_max(0.0, ca + sixth * (dca1 + 2.0 * dca2 + 2.0 * dca3 + dca4));
        }
        Ok(Trace {
            n_steps,
            n_columns: n_cols,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A passive membrane: one ungated leak channel and no gates.
    fn passive_model(g_leak: f64, e_leak: f64, c_m: f64) -> Model {
        let json = format!(
            r#"{{"version": 2, "c_m": {c_m}, "phi": 1.0, "exp_clip": 100,
                 "v_clamp_mv": [-150, 150], "gates": [], "calcium": null,
                 "channels": [{{"name": "L", "g_max": {g_leak}, "gates": [],
                                "carries_calcium": false, "e_rev": {e_leak}}}]}}"#
        );
        Model::from_json(&json).unwrap()
    }

    #[test]
    fn passive_membrane_relaxes_exponentially() {
        // RC membrane: V(t) = E + (V0 − E) · exp(−g t / C). RK4 with
        // dt = 0.025 ms is accurate to far better than 1e-9 mV here.
        let (g, e, c, v0, dt) = (0.3, -65.0, 1.0, -40.0, 0.025);
        let model = passive_model(g, e, c);
        let trace = model
            .simulate_current_clamp(&vec![0.0; 2001], v0, &[], 0.0, dt)
            .unwrap();
        for step in [0, 100, 1000, 2000] {
            let t = step as f64 * dt;
            let expected = e + (v0 - e) * (-g * t / c).exp();
            let got = trace.data[step * trace.n_columns];
            assert!((got - expected).abs() < 1e-9, "t={t}: {got} vs {expected}");
        }
    }

    #[test]
    fn records_leak_current_and_total() {
        let model = passive_model(0.3, -65.0, 1.0);
        let trace = model
            .simulate_current_clamp(&[0.0], -40.0, &[], 0.0, 0.025)
            .unwrap();
        // Columns: voltage, I_L, Itotal.
        assert_eq!(trace.n_columns, 3);
        assert_eq!(trace.data, vec![-40.0, 0.3 * 25.0, 0.3 * 25.0]);
    }

    #[test]
    fn rejects_wrong_number_of_initial_gates() {
        let model = passive_model(0.3, -65.0, 1.0);
        let err = model
            .simulate_current_clamp(&[0.0], -40.0, &[0.5], 0.0, 0.025)
            .unwrap_err();
        assert!(err.contains("expected 0 initial gate values"), "{err}");
    }

    #[test]
    fn clamp_matches_python_semantics() {
        assert_eq!(py_clamp(200.0, -150.0, 150.0), 150.0);
        assert_eq!(py_clamp(-200.0, -150.0, 150.0), -150.0);
        // Python's max(0.0, min(1.0, nan)) is 1.0.
        assert_eq!(py_clamp(f64::NAN, 0.0, 1.0), 1.0);
    }
}
