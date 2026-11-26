//! Compiled current-clamp core for patch_sim (spike for issue #410).
//!
//! Layout:
//! - [`expr`]: rate-function expression trees and their evaluator.
//! - [`spec`]: the flat model description produced by Python's `lower()`.
//! - [`current_clamp`]: the fixed-step RK4 integrator.
//! - `python` (feature `python`): PyO3 bindings exposing `patch_sim_core.Model`.
//!
//! The first three modules are pure Rust with no Python dependency, the same
//! split as `patch_sim` (no Reflex) vs. `patch_sim_ui`.

pub mod current_clamp;
pub mod expr;
pub mod spec;

#[cfg(feature = "python")]
mod python;
