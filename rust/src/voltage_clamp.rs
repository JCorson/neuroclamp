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
    /// Each sweep takes its own initial state, because a voltage-clamp run
    /// starts from the steady state at its protocol's first voltage: sweeps
    /// that hold at different potentials start from different gate values.
    /// Results come back in protocol order.
    ///
    /// # Arguments
    /// * `protocols` - prescribed voltages in mV, one slice per sweep.
    /// * `gates0` - initial gate values for each sweep, in
    ///   `ModelSpec::gates` order.
    /// * `ca0` - initial intracellular Ca²⁺ for each sweep in mM.
    /// * `dt` - time step in ms.
    pub fn simulate_voltage_clamp_batch(
        &self,
        protocols: &[&[f64]],
        gates0: &[&[f64]],
        ca0: &[f64],
        dt: f64,
    ) -> Result<Vec<Trace>, String> {
        if protocols.is_empty() {
            return Err("voltage_protocol batch must have at least one sweep".into());
        }
        let n_sweeps = protocols.len();
        if gates0.len() != n_sweeps || ca0.len() != n_sweeps {
            return Err(format!(
                "expected one initial state per sweep ({n_sweeps}), got {} gate \
                 rows and {} Ca²⁺ values",
                gates0.len(),
                ca0.len()
            ));
        }
        protocols
            .par_iter()
            .zip(gates0.par_iter())
            .zip(ca0.par_iter())
            .map(|((v_protocol, gates), &ca)| {
                self.simulate_voltage_clamp(v_protocol, gates, ca, dt)
            })
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
    fn rejects_wrong_number_of_initial_gates() {
        let err = one_gate_model()
            .simulate_voltage_clamp(&[0.0], &[], 0.0, 0.025)
            .unwrap_err();
        assert!(err.contains("expected 1 initial gate values"), "{err}");
    }

    #[test]
    fn rejects_empty_protocol() {
        let err = one_gate_model()
            .simulate_voltage_clamp(&[], &[0.3], 0.0, 0.025)
            .unwrap_err();
        assert!(err.contains("must not be empty"), "{err}");
    }

    #[test]
    fn empty_batch_is_rejected() {
        let err = one_gate_model()
            .simulate_voltage_clamp_batch(&[], &[], &[], 0.025)
            .unwrap_err();
        assert!(err.contains("at least one sweep"), "{err}");
    }

    #[test]
    fn batch_rejects_mismatched_initial_states() {
        let model = one_gate_model();
        let protocol = [0.0; 10];
        let err = model
            .simulate_voltage_clamp_batch(&[&protocol, &protocol], &[&[0.3]], &[0.0, 0.0], 0.025)
            .unwrap_err();
        assert!(err.contains("one initial state per sweep"), "{err}");
        let err = model
            .simulate_voltage_clamp_batch(&[&protocol], &[&[0.3]], &[0.0, 0.0], 0.025)
            .unwrap_err();
        assert!(err.contains("one initial state per sweep"), "{err}");
    }

    #[test]
    fn batch_matches_sequential_runs_from_per_sweep_states() {
        let model = one_gate_model();
        let protocols: Vec<Vec<f64>> = (0..5).map(|k| vec![-80.0 + 20.0 * k as f64; 300]).collect();
        let gates: Vec<Vec<f64>> = (0..5).map(|k| vec![0.1 + 0.15 * k as f64]).collect();
        let ca0: Vec<f64> = (0..5).map(|k| 1e-4 * k as f64).collect();
        let refs: Vec<&[f64]> = protocols.iter().map(Vec::as_slice).collect();
        let gate_refs: Vec<&[f64]> = gates.iter().map(Vec::as_slice).collect();
        let batch = model
            .simulate_voltage_clamp_batch(&refs, &gate_refs, &ca0, 0.025)
            .unwrap();
        for (k, trace) in batch.iter().enumerate() {
            let single = model
                .simulate_voltage_clamp(&protocols[k], &gates[k], ca0[k], 0.025)
                .unwrap();
            assert_eq!(trace.data, single.data);
            // Each sweep starts from its own gate value, not the first sweep's.
            assert_eq!(trace.data[3], gates[k][0]);
        }
    }

    /// A voltage-gated Ca²⁺ channel (gate `m`) and a Ca²⁺-activated K⁺
    /// channel (gate `q`), with Ca²⁺ dynamics and φ = 3.
    ///
    /// `m` has voltage-only rates α = 0.1·exp(V/20), β = 0.1; `q` has
    /// Ca²⁺-only rates α = Ca/(Ca + 0.001), β = 0.2.
    fn calcium_model() -> Model {
        let json = r#"{
            "version": 3, "c_m": 1.0, "phi": 3.0, "exp_clip": 100,
            "v_clamp_mv": [-150, 150], "gate_bounds": [0, 1], "ca_floor_mm": 1e-7,
            "gates": [
                {"name": "m", "uses_calcium": false,
                 "kinetics": {"form": "alpha_beta",
                              "alpha": {"op": "Exp", "k": 0.1, "vh": 0.0, "s": 20.0},
                              "beta": {"op": "Const", "value": 0.1}}},
                {"name": "q", "uses_calcium": true,
                 "kinetics": {"form": "alpha_beta",
                              "alpha": {"op": "Hill", "kd": 0.001, "n": 1},
                              "beta": {"op": "Const", "value": 0.2}}}
            ],
            "channels": [
                {"name": "CaL", "g_max": 1.0, "gates": [[0, 1]],
                 "carries_calcium": true, "e_rev": null},
                {"name": "KCa", "g_max": 2.0, "gates": [[1, 1]],
                 "carries_calcium": false, "e_rev": -80.0}
            ],
            "calcium_nernst": {"ca_out": 2.0, "temperature_k": 310.0, "floor_mm": 1e-6,
                               "gas_constant": 8.314, "faraday": 96485.0},
            "calcium_dynamics": {"alpha_ca": 0.002, "tau_ca": 80.0, "ca_rest": 5e-5}
        }"#;
        Model::from_json(json).unwrap()
    }

    #[test]
    fn calcium_dependent_gates_follow_stage_calcium() {
        // A reference RK4 written out from the equations: every stage
        // re-evaluates q's rates at that stage's Ca²⁺, and dCa/dt is driven
        // by the Ca²⁺ current at the prescribed voltage with the live E_Ca.
        let model = calcium_model();
        let (v, dt, phi) = (-20.0_f64, 0.025_f64, 3.0_f64);
        let e_ca = |ca: f64| (8.314 * 310.0 / (2.0 * 96485.0)) * 1000.0 * (2.0 / ca.max(1e-6)).ln();
        let derivs = |m: f64, q: f64, ca: f64| {
            let (am, bm) = (phi * (0.1 * (v / 20.0).exp()), phi * 0.1);
            let (aq, bq) = (phi * (ca / (ca + 0.001)), phi * 0.2);
            let i_ca = 1.0 * m * (v - e_ca(ca));
            let dca = -0.002 * i_ca - (ca - 5e-5) / 80.0;
            (am * (1.0 - m) - bm * m, aq * (1.0 - q) - bq * q, dca)
        };
        let (mut m, mut q, mut ca) = (0.2_f64, 0.05_f64, 5e-5_f64);
        let n_steps = 200;
        let trace = model
            .simulate_voltage_clamp(&vec![v; n_steps], &[m, q], ca, dt)
            .unwrap();
        // Columns: voltage, I_CaL, I_KCa, Itotal, m, q, ca.
        assert_eq!(trace.n_columns, 7);
        for step in 1..n_steps {
            let (m1, q1, c1) = derivs(m, q, ca);
            let (m2, q2, c2) = derivs(m + 0.5 * dt * m1, q + 0.5 * dt * q1, ca + 0.5 * dt * c1);
            let (m3, q3, c3) = derivs(m + 0.5 * dt * m2, q + 0.5 * dt * q2, ca + 0.5 * dt * c2);
            let (m4, q4, c4) = derivs(m + dt * m3, q + dt * q3, ca + dt * c3);
            m += dt / 6.0 * (m1 + 2.0 * m2 + 2.0 * m3 + m4);
            q += dt / 6.0 * (q1 + 2.0 * q2 + 2.0 * q3 + q4);
            ca += dt / 6.0 * (c1 + 2.0 * c2 + 2.0 * c3 + c4);
            let row = &trace.data[step * trace.n_columns..(step + 1) * trace.n_columns];
            for (got, want, name) in [(row[4], m, "m"), (row[5], q, "q"), (row[6], ca, "ca")] {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "step {step} {name}: {got} vs {want}"
                );
            }
            let i_ca = m * (v - e_ca(ca));
            let i_k = 2.0 * q * (v + 80.0);
            assert!(
                (row[1] - i_ca).abs() <= 1e-12 * i_ca.abs(),
                "step {step} I_CaL"
            );
            assert!(
                (row[2] - i_k).abs() <= 1e-12 * i_k.abs(),
                "step {step} I_KCa"
            );
        }
        // The inward Ca²⁺ current raises Ca²⁺, which opens q.
        assert!(ca > 5e-4, "Ca²⁺ should accumulate: {ca}");
        assert!(q > 0.3, "q should open with Ca²⁺: {q}");
    }

    #[test]
    fn calcium_stays_constant_without_dynamics() {
        // Without Ca²⁺ dynamics there is no Ca²⁺ column and q relaxes at the
        // fixed initial Ca²⁺.
        let json = r#"{
            "version": 3, "c_m": 1.0, "phi": 1.0, "exp_clip": 100,
            "v_clamp_mv": [-150, 150], "gate_bounds": [0, 1], "ca_floor_mm": 0,
            "gates": [{"name": "q", "uses_calcium": true,
                       "kinetics": {"form": "alpha_beta",
                                    "alpha": {"op": "Hill", "kd": 0.001, "n": 1},
                                    "beta": {"op": "Const", "value": 0.2}}}],
            "channels": [{"name": "KCa", "g_max": 2.0, "gates": [[0, 1]],
                          "carries_calcium": false, "e_rev": -80.0}],
            "calcium_nernst": null, "calcium_dynamics": null
        }"#;
        let model = Model::from_json(json).unwrap();
        let ca = 0.001;
        let trace = model
            .simulate_voltage_clamp(&vec![0.0; 8001], &[0.0], ca, 0.025)
            .unwrap();
        // Columns: voltage, I_KCa, Itotal, q.
        assert_eq!(trace.n_columns, 4);
        let alpha = ca / (ca + 0.001);
        let q_inf = alpha / (alpha + 0.2);
        let q_end = trace.data[8000 * trace.n_columns + 3];
        assert!((q_end - q_inf).abs() < 1e-9, "{q_end} vs {q_inf}");
    }
}
