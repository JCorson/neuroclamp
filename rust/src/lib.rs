//! Compiled simulation core for neuroclamp, imported as `neuroclamp._core`.
//!
//! Layout:
//! - [`expr`]: rate-function expression trees and their evaluator.
//! - [`spec`]: the flat model description produced by `neuroclamp._spec.lower()`.
//! - [`current_clamp`]: the model and the fixed-step RK4 current-clamp integrator.
//! - [`voltage_clamp`]: the RK4 voltage-clamp integrator for the same model.
//! - `python` (feature `python`): PyO3 bindings exposing the `Model` class.
//!
//! All but `python` are pure Rust with no Python dependency, the same
//! split as `neuroclamp` (no Reflex) vs. `neuroclamp_ui`.

pub mod current_clamp;
pub mod expr;
pub mod spec;
pub mod voltage_clamp;

#[cfg(feature = "python")]
mod python;
