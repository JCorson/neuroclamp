//! Rate-function expression trees.
//!
//! This is the Rust mirror of `spikes/rate_expr/exprs.py`. Python lowers each
//! rate to a nested dict such as
//! `{"op": "ExpLinear", "k": 0.1, "vh": -40, "s": 10, "tol": 1e-6}`, and serde
//! turns that straight into an [`Expr`] value: the `op` key picks the enum
//! variant (`#[serde(tag = "op")]`), the remaining keys fill its fields.
//!
//! Every formula matches its Python counterpart operation for operation, so
//! results agree with the Python backend to the last few bits.

use serde::Deserialize;

/// Largest |x| passed to `exp`/`cosh`; matches `neuroclamp.utils.SAFE_EXP_CLIP_MAX`.
pub const EXP_CLIP: f64 = 100.0;

/// `exp(x)` with the input clipped to `[-EXP_CLIP, EXP_CLIP]`.
///
/// NaN passes through: both comparisons are false for NaN, so it reaches
/// `x.exp()`, which returns NaN, exactly like `neuroclamp.utils.safe_exp`.
pub fn safe_exp(x: f64) -> f64 {
    if x > EXP_CLIP {
        EXP_CLIP.exp()
    } else if x < -EXP_CLIP {
        (-EXP_CLIP).exp()
    } else {
        x.exp()
    }
}

/// `cosh(x)` with the input clipped to `[-EXP_CLIP, EXP_CLIP]`.
pub fn safe_cosh(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if !(-EXP_CLIP..=EXP_CLIP).contains(&x) {
        // cosh is symmetric, so both tails clip to cosh(EXP_CLIP).
        EXP_CLIP.cosh()
    } else {
        x.cosh()
    }
}

/// `x` raised to an integer power, matching Python's `x ** n` on floats.
///
/// Python calls C `pow`. For `n = 1` that returns `x` exactly, and for
/// `n = 2` a single multiplication is the correctly rounded result, so these
/// fast paths are bit-identical; other powers still call `pow`.
pub fn int_pow(x: f64, n: i32) -> f64 {
    match n {
        1 => x,
        2 => x * x,
        _ => x.powf(f64::from(n)),
    }
}

/// Python's `max(a, b)`: returns `a` unless `b` is strictly greater.
///
/// Rust's `f64::max` treats NaN differently, so this keeps the two backends
/// identical even on pathological inputs.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// A rate expression in membrane voltage `v` (mV) and intracellular Ca²⁺ `ca` (mM).
///
/// Leaves hold their parameters directly. Combinators hold their children in
/// a `Box`, because an enum cannot contain itself by value (its size would be
/// infinite); a `Box` is a pointer to a heap-allocated child.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "op")]
pub enum Expr {
    /// A constant value.
    Const { value: f64 },
    /// `k · exp((v − vh) / s)`
    Exp { k: f64, vh: f64, s: f64 },
    /// `k / (1 + exp(−(v − vh) / s))`
    Sigmoid { k: f64, vh: f64, s: f64 },
    /// `k · (v − vh) / (1 − exp(−(v − vh) / s))`, with limit `k · s` near `vh`.
    ExpLinear { k: f64, vh: f64, s: f64, tol: f64 },
    /// `k / cosh((v − vh) / s)`
    Sech { k: f64, vh: f64, s: f64 },
    /// `caⁿ / (caⁿ + kdⁿ)`
    Hill { kd: f64, n: i32 },
    /// `a + b`
    Add { a: Box<Expr>, b: Box<Expr> },
    /// `a − b`
    Sub { a: Box<Expr>, b: Box<Expr> },
    /// `a · b`
    Mul { a: Box<Expr>, b: Box<Expr> },
    /// `a / b`
    Div { a: Box<Expr>, b: Box<Expr> },
    /// `max(a, b)` with Python semantics (see [`py_max`]).
    Max { a: Box<Expr>, b: Box<Expr> },
}

impl Expr {
    /// Evaluate the expression at voltage `v` (mV) and Ca²⁺ `ca` (mM).
    ///
    /// `match` is Rust's exhaustive pattern match: the compiler refuses to
    /// build if a variant is missing, so adding a node type forces every
    /// evaluator to handle it.
    pub fn eval(&self, v: f64, ca: f64) -> f64 {
        match self {
            Expr::Const { value } => *value,
            Expr::Exp { k, vh, s } => k * safe_exp((v - vh) / s),
            Expr::Sigmoid { k, vh, s } => k / (1.0 + safe_exp(-(v - vh) / s)),
            Expr::ExpLinear { k, vh, s, tol } => {
                let x = v - vh;
                if x.abs() < *tol {
                    k * s
                } else {
                    k * x / (1.0 - safe_exp(-x / s))
                }
            }
            Expr::Sech { k, vh, s } => k / safe_cosh((v - vh) / s),
            Expr::Hill { kd, n } => {
                let ca_n = int_pow(ca, *n);
                ca_n / (ca_n + int_pow(*kd, *n))
            }
            Expr::Add { a, b } => a.eval(v, ca) + b.eval(v, ca),
            Expr::Sub { a, b } => a.eval(v, ca) - b.eval(v, ca),
            Expr::Mul { a, b } => a.eval(v, ca) * b.eval(v, ca),
            Expr::Div { a, b } => a.eval(v, ca) / b.eval(v, ca),
            Expr::Max { a, b } => py_max(a.eval(v, ca), b.eval(v, ca)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Expr {
        serde_json::from_str(json).expect("valid expression JSON")
    }

    #[test]
    fn deserializes_nested_tree_from_python_spec() {
        // Python emits integer literals like -40; serde accepts them as f64.
        let e = parse(
            r#"{"op": "Div",
                "a": {"op": "Const", "value": 1.0},
                "b": {"op": "Sigmoid", "k": 1.0, "vh": -40, "s": 8}}"#,
        );
        // At the half point the sigmoid is k/2, so 1 / 0.5 = 2.
        assert_eq!(e.eval(-40.0, 0.0), 2.0);
    }

    #[test]
    fn exp_linear_returns_limit_at_singularity() {
        // HH alpha_n: 0.01 (V + 55) / (1 − exp(−(V + 55) / 10)), limit 0.1.
        let alpha_n = Expr::ExpLinear {
            k: 0.01,
            vh: -55.0,
            s: 10.0,
            tol: 1e-6,
        };
        assert_eq!(alpha_n.eval(-55.0, 0.0), 0.01 * 10.0);
        // Just outside the tolerance the formula is continuous with the limit.
        assert!((alpha_n.eval(-55.0 + 1e-5, 0.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn safe_exp_clips_large_inputs() {
        assert_eq!(safe_exp(1e6), EXP_CLIP.exp());
        assert_eq!(safe_exp(-1e6), (-EXP_CLIP).exp());
        assert!(safe_exp(f64::NAN).is_nan());
    }

    #[test]
    fn hill_is_half_activated_at_kd() {
        let sk = Expr::Hill { kd: 3e-4, n: 4 };
        assert!((sk.eval(0.0, 3e-4) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn max_matches_python_semantics() {
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_max(2.0, 1.0), 2.0);
        // Python: max(nan, 1.0) is nan because 1.0 > nan is False.
        assert!(py_max(f64::NAN, 1.0).is_nan());
    }
}
