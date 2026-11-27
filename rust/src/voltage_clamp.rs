//! Fixed-step RK4 voltage-clamp integrator.
//!
//! A port of `_simulate_voltage_clamp_core` and `_rk4_step_voltage_clamp` in
//! `neuroclamp/clamp_simulations.py`, following its order of floating-point
//! operations so the two backends agree to within rounding.
//!
//! Integrator contract (shared with the Python backend):
//! - V is prescribed: step `i` advances the state with every RK4 stage at
//!   `v_protocol[i]`, and records the currents at that voltage.
//! - The Q10 factor multiplies each rate before the gate equation:
//!   dx/dt = (φα)(1 − x) − (φβ)x.
//! - Ca²⁺ evolves only with Ca²⁺ dynamics, driven by the Ca²⁺ channels'
//!   current at the prescribed voltage; otherwise it stays at its initial
//!   value.
//! - Gates are clipped to `gate_bounds` and Ca²⁺ is floored at `ca_floor_mm`
//!   only after the full step.

use rayon::prelude::*;

use crate::current_clamp::{Model, Trace, py_clamp};
use crate::expr::py_max;

/// φ-scaled `(α, β)` for one gate at one voltage, before any Ca²⁺ update.
type Rates = (f64, f64);

impl Model {
    /// Evaluate every voltage-only gate's φ-scaled rates at `v`.
    ///
    /// V is constant across a voltage-clamp step's RK4 stages, so these are
    /// computed once per step and reused by all four stages, the same values
    /// the Python backend reads from its precomputed tables. Entries for
    /// Ca²⁺-dependent gates are left untouched; they are evaluated per stage.
    fn voltage_only_rates(&self, v: f64, out: &mut [Rates]) {
        let phi = self.spec.phi;
        for (gate, rates) in self.spec.gates.iter().zip(out.iter_mut()) {
            if !gate.uses_calcium {
                let (alpha, beta) = gate.kinetics.rates(v, 0.0);
                *rates = (phi * alpha, phi * beta);
            }
        }
    }

    /// Gate derivatives and dCa/dt at prescribed voltage `v`.
    ///
    /// Writes each gate's derivative into `d_gates` and returns dCa/dt.
    fn voltage_clamp_derivatives(
        &self,
        v: f64,
        gates: &[f64],
        ca: f64,
        voltage_only: &[Rates],
        d_gates: &mut [f64],
    ) -> f64 {
        let phi = self.spec.phi;
        for (k, (gate, (&x, d))) in self
            .spec
            .gates
            .iter()
            .zip(gates.iter().zip(d_gates.iter_mut()))
            .enumerate()
        {
            let (a, b) = if gate.uses_calcium {
                let (alpha, beta) = gate.kinetics.rates(v, ca);
                (phi * alpha, phi * beta)
            } else {
                voltage_only[k]
            };
            *d = a * (1.0 - x) - b * x;
        }
        match &self.spec.calcium_dynamics {
            Some(c) => {
                let e_ca = if self.has_ca_channels {
                    self.e_ca(ca)
                } else {
                    f64::NAN
                };
                let mut i_ca = 0.0;
                for ch in self.spec.channels.iter().filter(|ch| ch.carries_calcium) {
                    i_ca += self.channel_current(ch, v, gates, e_ca);
                }
                -c.alpha_ca * i_ca - (ca - c.ca_rest) / c.tau_ca
            }
            None => 0.0,
        }
    }

    /// Every channel's current at `v` for the given state, written to `currents`.
    fn currents_at(&self, v: f64, gates: &[f64], ca: f64, currents: &mut [f64]) {
        let e_ca = if self.has_ca_channels {
            self.e_ca(ca)
        } else {
            f64::NAN
        };
        for (ch, out) in self.spec.channels.iter().zip(currents.iter_mut()) {
            *out = self.channel_current(ch, v, gates, e_ca);
        }
    }

    /// Run a voltage-clamp simulation.
    ///
    /// Output rows have the same columns as [`Model::simulate_current_clamp`]
    /// (see [`Trace`]), with the prescribed voltage in the first column.
    ///
    /// # Arguments
    /// * `v_protocol` - prescribed membrane voltage per time step in mV; its
    ///   length sets the number of steps.
    /// * `gates0` - initial gate values in `ModelSpec::gates` order.
    /// * `ca0` - initial intracellular Ca²⁺ in mM; held constant without
    ///   Ca²⁺ dynamics.
    /// * `dt` - time step in ms.
    pub fn simulate_voltage_clamp(
        &self,
        v_protocol: &[f64],
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
        if v_protocol.is_empty() {
            return Err("voltage_protocol must not be empty".into());
        }
        let (x_lo, x_hi) = self.spec.gate_bounds;
        let ca_floor = self.spec.ca_floor_mm;
        let n_steps = v_protocol.len();
        let n_cols = self.n_columns();
        let mut data = vec![0.0; n_steps * n_cols];

        let mut gates = gates0.to_vec();
        let mut ca = ca0;
        let (mut d1, mut d2, mut d3, mut d4) = (
            vec![0.0; n_gates],
            vec![0.0; n_gates],
            vec![0.0; n_gates],
            vec![0.0; n_gates],
        );
        let mut trial = vec![0.0; n_gates];
        let mut voltage_only: Vec<Rates> = vec![(0.0, 0.0); n_gates];
        let mut currents = vec![0.0; self.spec.channels.len()];

        self.currents_at(v_protocol[0], &gates, ca, &mut currents);
        self.record(&mut data[..n_cols], v_protocol[0], &gates, ca, &currents);

        let half = 0.5 * dt;
        let sixth = dt / 6.0;
        for step in 1..n_steps {
            let v = v_protocol[step];
            self.voltage_only_rates(v, &mut voltage_only);

            let dca1 = self.voltage_clamp_derivatives(v, &gates, ca, &voltage_only, &mut d1);
            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d1) {
                *t = x + half * d;
            }
            let dca2 =
                self.voltage_clamp_derivatives(v, &trial, ca + half * dca1, &voltage_only, &mut d2);
            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d2) {
                *t = x + half * d;
            }
            let dca3 =
                self.voltage_clamp_derivatives(v, &trial, ca + half * dca2, &voltage_only, &mut d3);
            for ((t, &x), &d) in trial.iter_mut().zip(&gates).zip(&d3) {
                *t = x + dt * d;
            }
            let dca4 =
                self.voltage_clamp_derivatives(v, &trial, ca + dt * dca3, &voltage_only, &mut d4);

            for (k, x) in gates.iter_mut().enumerate() {
                let next = *x + sixth * (d1[k] + 2.0 * d2[k] + 2.0 * d3[k] + d4[k]);
                *x = py_clamp(next, x_lo, x_hi);
            }
            ca = py_max(
                ca_floor,
                ca + sixth * (dca1 + 2.0 * dca2 + 2.0 * dca3 + dca4),
            );

            self.currents_at(v, &gates, ca, &mut currents);
            self.record(
                &mut data[step * n_cols..(step + 1) * n_cols],
                v,
                &gates,
                ca,
                &currents,
            );
        }
        Ok(Trace {
            n_steps,
            n_columns: n_cols,
            data,
        })
    }

    /// Run one voltage-clamp sweep per protocol, in parallel across CPU cores.
    ///
    /// Every sweep starts from the same initial state; results come back in
    /// protocol order.
    pub fn simulate_voltage_clamp_batch(
        &self,
        protocols: &[&[f64]],
        gates0: &[f64],
        ca0: f64,
        dt: f64,
    ) -> Result<Vec<Trace>, String> {
        protocols
            .par_iter()
            .map(|v_protocol| self.simulate_voltage_clamp(v_protocol, gates0, ca0, dt))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One K⁺ channel with a single gate of constant rates α = 0.1, β = 0.2.
    fn one_gate_model() -> Model {
        let json = r#"{
            "version": 3, "c_m": 1.0, "phi": 1.0, "exp_clip": 100,
            "v_clamp_mv": [-150, 150], "gate_bounds": [0, 1], "ca_floor_mm": 0,
            "gates": [{"name": "n", "uses_calcium": false,
                       "kinetics": {"form": "alpha_beta",
                                    "alpha": {"op": "Const", "value": 0.1},
                                    "beta": {"op": "Const", "value": 0.2}}}],
            "channels": [{"name": "K", "g_max": 2.0, "gates": [[0, 1]],
                          "carries_calcium": false, "e_rev": -80.0}],
            "calcium_nernst": null, "calcium_dynamics": null
        }"#;
        Model::from_json(json).unwrap()
    }

    #[test]
    fn gate_relaxes_to_steady_state_exponentially() {
        // dn/dt = α(1 − n) − βn relaxes to α/(α+β) with time constant 1/(α+β).
        let model = one_gate_model();
        let dt = 0.025;
        let trace = model
            .simulate_voltage_clamp(&vec![0.0; 4001], &[0.0], 0.0, dt)
            .unwrap();
        let (n_inf, tau) = (0.1 / 0.3, 1.0 / 0.3);
        for step in [0, 400, 4000] {
            let t = step as f64 * dt;
            let expected = n_inf * (1.0 - (-t / tau).exp());
            // Columns: voltage, I_K, Itotal, n.
            let got = trace.data[step * trace.n_columns + 3];
            assert!((got - expected).abs() < 1e-10, "t={t}: {got} vs {expected}");
        }
    }

    #[test]
    fn records_prescribed_voltage_and_current() {
        let model = one_gate_model();
        let trace = model
            .simulate_voltage_clamp(&[-60.0, 10.0], &[0.5], 0.0, 0.025)
            .unwrap();
        assert_eq!(trace.data[0], -60.0);
        assert_eq!(trace.data[1], 2.0 * 0.5 * 20.0);
        assert_eq!(trace.data[trace.n_columns], 10.0);
    }

    #[test]
    fn batch_matches_sequential_runs() {
        let model = one_gate_model();
        let protocols: Vec<Vec<f64>> = (0..5).map(|k| vec![-80.0 + 20.0 * k as f64; 300]).collect();
        let refs: Vec<&[f64]> = protocols.iter().map(Vec::as_slice).collect();
        let batch = model
            .simulate_voltage_clamp_batch(&refs, &[0.3], 0.0, 0.025)
            .unwrap();
        for (trace, v) in batch.iter().zip(&protocols) {
            let single = model.simulate_voltage_clamp(v, &[0.3], 0.0, 0.025).unwrap();
            assert_eq!(trace.data, single.data);
        }
    }
}
