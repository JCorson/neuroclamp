//! Compiled current-clamp core for neuroclamp, imported as `neuroclamp._core`.
//!
//! Layout:
//! - [`expr`]: rate-function expression trees and their evaluator.
//! - [`spec`]: the flat model description produced by `neuroclamp._spec.lower()`.
//! - [`current_clamp`]: the fixed-step RK4 integrator.
//! - `python` (feature `python`): PyO3 bindings exposing the `Model` class.
//!
//! The first three modules are pure Rust with no Python dependency, the same
//! split as `neuroclamp` (no Reflex) vs. `neuroclamp_ui`.

pub mod current_clamp;
pub mod expr;
pub mod spec;

#[cfg(feature = "python")]
mod python;
