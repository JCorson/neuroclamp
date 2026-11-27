//! The flat model description produced by Python's `neuroclamp._spec.lower()`.
//!
//! These structs mirror the JSON spec field for field; the format itself is
//! documented in `neuroclamp/_spec.py`. `#[derive(Deserialize)]` generates the
//! parsing code, and [`ModelSpec::from_json`] then checks the invariants the
//! integrator relies on, so a bad spec fails loudly at load time instead of
//! producing a silently wrong trace.

use serde::Deserialize;

use crate::expr::{EXP_CLIP, Expr};

/// Spec format version this crate understands; matches `SPEC_VERSION` in
/// `neuroclamp/_spec.py`.
pub const SPEC_VERSION: u32 = 3;

/// How a gate's forward and backward rates are defined.
///
/// `#[serde(tag = "form")]` picks the variant from the JSON `"form"` key:
/// `{"form": "alpha_beta", "alpha": …, "beta": …}` or
/// `{"form": "inf_tau", "inf": …, "tau": …}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "form", rename_all = "snake_case")]
pub enum Kinetics {
    /// Rates given directly.
    AlphaBeta { alpha: Expr, beta: Expr },
    /// Steady state and time constant: α = inf / τ, β = (1 − inf) / τ.
    /// Evaluating `inf` and `tau` once serves both rates.
    InfTau { inf: Expr, tau: Expr },
}

impl Kinetics {
    /// Evaluate `(α, β)` in 1/ms at voltage `v` (mV) and Ca²⁺ `ca` (mM).
    pub fn rates(&self, v: f64, ca: f64) -> (f64, f64) {
        match self {
            Kinetics::AlphaBeta { alpha, beta } => (alpha.eval(v, ca), beta.eval(v, ca)),
            Kinetics::InfTau { inf, tau } => {
                let (inf, tau) = (inf.eval(v, ca), tau.eval(v, ca));
                // The same operations as the expanded `Div` trees, so the
                // result is bit-identical to evaluating α and β separately.
                (inf / tau, (1.0 - inf) / tau)
            }
        }
    }
}

/// A single gating variable: dx/dt = φ · (α(1 − x) − βx).
#[derive(Debug, Clone, Deserialize)]
pub struct GateSpec {
    pub name: String,
    pub kinetics: Kinetics,
    /// Whether either rate reads Ca²⁺. The voltage-clamp integrator
    /// evaluates the other gates' rates once per step, since V is fixed
    /// across its RK4 stages; current clamp evaluates every rate every stage.
    pub uses_calcium: bool,
}

/// An ion channel: I = g_max · Π xᵢ^pᵢ · (V − E).
#[derive(Debug, Clone, Deserialize)]
pub struct ChannelSpec {
    pub name: String,
    pub g_max: f64,
    /// `(gate_index, power)` pairs; indices point into [`ModelSpec::gates`].
    pub gates: Vec<(usize, i32)>,
    pub carries_calcium: bool,
    /// Fixed reversal potential in mV. `None` for Ca²⁺ channels, whose E_Ca
    /// is recomputed from the live intracellular Ca²⁺ every sub-step.
    pub e_rev: Option<f64>,
}

/// What the live Ca²⁺ Nernst potential needs; present when a channel carries Ca²⁺.
#[derive(Debug, Clone, Deserialize)]
pub struct CalciumNernstSpec {
    pub ca_out: f64,
    pub temperature_k: f64,
    pub floor_mm: f64,
    pub gas_constant: f64,
    pub faraday: f64,
}

/// Intracellular Ca²⁺ dynamics: d[Ca]/dt = −α_Ca · I_Ca − (Ca − Ca_rest) / τ_Ca.
#[derive(Debug, Clone, Deserialize)]
pub struct CalciumDynamicsSpec {
    pub alpha_ca: f64,
    pub tau_ca: f64,
    pub ca_rest: f64,
}

/// The whole model, as lowered from a `neuroclamp.Neuron`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub version: u32,
    /// Membrane capacitance in µF/cm².
    pub c_m: f64,
    /// Q10 temperature factor multiplying every rate.
    pub phi: f64,
    pub exp_clip: f64,
    /// Voltage safety range `(min, max)` in mV applied at every RK4 sub-step.
    pub v_clamp_mv: (f64, f64),
    /// Range every gate is clipped to after each full RK4 step.
    pub gate_bounds: (f64, f64),
    /// Floor applied to Ca²⁺ (mM) after each full RK4 step.
    pub ca_floor_mm: f64,
    pub gates: Vec<GateSpec>,
    pub channels: Vec<ChannelSpec>,
    pub calcium_nernst: Option<CalciumNernstSpec>,
    pub calcium_dynamics: Option<CalciumDynamicsSpec>,
}

impl ModelSpec {
    /// Parse and validate a spec from its JSON text.
    ///
    /// Returns `Err` with a human-readable message on malformed JSON or a
    /// spec this integrator cannot run. `?` propagates an error to the
    /// caller early, like raising in Python.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let spec: ModelSpec =
            serde_json::from_str(json).map_err(|e| format!("invalid model spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Check the invariants the integrator assumes.
    fn validate(&self) -> Result<(), String> {
        if self.version != SPEC_VERSION {
            return Err(format!(
                "spec version {} is not supported (expected {SPEC_VERSION})",
                self.version
            ));
        }
        if self.exp_clip != EXP_CLIP {
            return Err(format!(
                "spec exp_clip {} does not match the compiled EXP_CLIP {EXP_CLIP}",
                self.exp_clip
            ));
        }
        if self.c_m <= 0.0 {
            return Err(format!("c_m must be positive, got {}", self.c_m));
        }
        let mut carries_calcium = false;
        for ch in &self.channels {
            for &(idx, _) in &ch.gates {
                if idx >= self.gates.len() {
                    return Err(format!(
                        "channel {:?} refers to gate {idx}, but there are only {} gates",
                        ch.name,
                        self.gates.len()
                    ));
                }
            }
            if !ch.carries_calcium && ch.e_rev.is_none() {
                return Err(format!("channel {:?} has no reversal potential", ch.name));
            }
            carries_calcium |= ch.carries_calcium;
        }
        if carries_calcium != self.calcium_nernst.is_some() {
            return Err(
                "calcium_nernst must be present exactly when a channel carries Ca²⁺".into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "version": 3, "c_m": 1.0, "phi": 1.0, "exp_clip": 100,
        "v_clamp_mv": [-150, 150], "gate_bounds": [0, 1], "ca_floor_mm": 0,
        "gates": [{"name": "n", "uses_calcium": false,
                   "kinetics": {"form": "alpha_beta",
                                "alpha": {"op": "Const", "value": 0.1},
                                "beta": {"op": "Const", "value": 0.2}}}],
        "channels": [{"name": "K", "g_max": 36.0, "gates": [[GATE, 4]],
                      "carries_calcium": false, "e_rev": -77.0}],
        "calcium_nernst": null,
        "calcium_dynamics": null
    }"#;

    #[test]
    fn accepts_valid_spec() {
        let spec = ModelSpec::from_json(&MINIMAL.replace("GATE", "0")).unwrap();
        assert_eq!(spec.channels[0].gates, vec![(0, 4)]);
    }

    #[test]
    fn rejects_out_of_range_gate_index() {
        let err = ModelSpec::from_json(&MINIMAL.replace("GATE", "3")).unwrap_err();
        assert!(err.contains("refers to gate 3"), "{err}");
    }

    #[test]
    fn rejects_unknown_version() {
        let json = MINIMAL
            .replace("GATE", "0")
            .replace("\"version\": 3", "\"version\": 2");
        let err = ModelSpec::from_json(&json).unwrap_err();
        assert!(err.contains("version 2"), "{err}");
    }

    #[test]
    fn rejects_calcium_channel_without_nernst_constants() {
        let json = MINIMAL.replace("GATE", "0").replace(
            "\"carries_calcium\": false, \"e_rev\": -77.0",
            "\"carries_calcium\": true, \"e_rev\": null",
        );
        let err = ModelSpec::from_json(&json).unwrap_err();
        assert!(err.contains("calcium_nernst"), "{err}");
    }

    #[test]
    fn inf_tau_matches_expanded_alpha_beta_bit_for_bit() {
        let inf = r#"{"op": "Sigmoid", "k": 1.0, "vh": -40.0, "s": 8.0}"#;
        let tau = r#"{"op": "Sech", "k": 4.0, "vh": -40.0, "s": 16.0}"#;
        let packed: Kinetics = serde_json::from_str(&format!(
            r#"{{"form": "inf_tau", "inf": {inf}, "tau": {tau}}}"#
        ))
        .unwrap();
        // alpha = inf / tau, beta = (1 - inf) / tau, as Python's inf_tau builds.
        let expanded: Kinetics = serde_json::from_str(&format!(
            r#"{{"form": "alpha_beta",
                 "alpha": {{"op": "Div", "a": {inf}, "b": {tau}}},
                 "beta": {{"op": "Div",
                           "a": {{"op": "Sub", "a": {{"op": "Const", "value": 1.0}}, "b": {inf}}},
                           "b": {tau}}}}}"#
        ))
        .unwrap();
        for v in [-120.0, -65.0, -40.0, 0.0, 35.0] {
            assert_eq!(packed.rates(v, 0.0), expanded.rates(v, 0.0), "v = {v}");
        }
    }
}
