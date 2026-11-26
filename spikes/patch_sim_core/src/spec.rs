//! The flat model description produced by Python's `lower(neuron)`.
//!
//! These structs mirror the JSON spec field for field. `#[derive(Deserialize)]`
//! generates the parsing code; [`ModelSpec::from_json`] then checks the
//! invariants the integrator relies on, so a bad spec fails loudly at load
//! time instead of producing a silently wrong trace.

use serde::Deserialize;

use crate::expr::{EXP_CLIP, Expr};

/// Spec format version this crate understands.
pub const SPEC_VERSION: u32 = 1;

/// A single gating variable: dx/dt = φ · (α(1 − x) − βx).
#[derive(Debug, Clone, Deserialize)]
pub struct GateSpec {
    pub name: String,
    pub alpha: Expr,
    pub beta: Expr,
    /// Informational: whether either rate reads Ca²⁺. Not needed by the
    /// current-clamp integrator, which evaluates every rate every sub-step.
    #[allow(dead_code)]
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

/// Intracellular Ca²⁺ dynamics plus what the live E_Ca Nernst potential needs.
#[derive(Debug, Clone, Deserialize)]
pub struct CalciumSpec {
    pub alpha_ca: f64,
    pub tau_ca: f64,
    pub ca_rest: f64,
    pub ca_out: f64,
    pub temperature_k: f64,
    pub nernst_floor_mm: f64,
    pub gas_constant: f64,
    pub faraday: f64,
}

/// The whole model, as lowered from a `patch_sim.Neuron`.
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
    pub gates: Vec<GateSpec>,
    pub channels: Vec<ChannelSpec>,
    pub calcium: Option<CalciumSpec>,
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
            match (ch.carries_calcium, ch.e_rev) {
                (false, None) => {
                    return Err(format!("channel {:?} has no reversal potential", ch.name));
                }
                (true, _) if self.calcium.is_none() => {
                    // patch_sim falls back to a 1e-9 mM floor here; no preset
                    // does this, so the prototype rejects it instead.
                    return Err(format!(
                        "channel {:?} carries Ca²⁺ but the model has no calcium dynamics",
                        ch.name
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "version": 1, "c_m": 1.0, "phi": 1.0, "exp_clip": 100,
        "v_clamp_mv": [-150, 150],
        "gates": [{"name": "n", "uses_calcium": false,
                   "alpha": {"op": "Const", "value": 0.1},
                   "beta": {"op": "Const", "value": 0.2}}],
        "channels": [{"name": "K", "g_max": 36.0, "gates": [[GATE, 4]],
                      "carries_calcium": false, "e_rev": -77.0}],
        "calcium": null
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
            .replace("\"version\": 1", "\"version\": 2");
        assert!(
            ModelSpec::from_json(&json)
                .unwrap_err()
                .contains("version 2")
        );
    }
}
