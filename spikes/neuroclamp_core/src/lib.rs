//! Compiled current-clamp core for neuroclamp (spike for issue #410).
//!
//! Layout:
//! - [`expr`]: rate-function expression trees and their evaluator.
//! - [`spec`]: the flat model description produced by Python's `lower()`.
//! - [`current_clamp`]: the fixed-step RK4 integrator.
//! - `python` (feature `python`): PyO3 bindings exposing `neuroclamp_core.Model`.
//!
//! The first three modules are pure Rust with no Python dependency, the same
//! split as `neuroclamp` (no Reflex) vs. `neuroclamp_ui`.

pub mod current_clamp;
pub mod expr;
pub mod spec;

#[cfg(feature = "python")]
mod python;
