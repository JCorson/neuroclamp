//! Python bindings (compiled only with the default `python` feature).
//!
//! Exposes one class to Python:
//!
//! ```python
//! import patch_sim_core
//! model = patch_sim_core.Model(spec_json)
//! out = model.simulate_current_clamp(i_ext, v0, gates0, ca0, dt)  # 2-D ndarray
//! ```
//!
//! Everything numeric lives in the pure-Rust modules; this file only converts
//! between Python and Rust types.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::current_clamp::Model;

/// A compiled patch_sim model, built once from a lowered JSON spec.
#[pyclass(name = "Model", frozen)]
pub struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    /// Parse and validate a model spec (the JSON form of `lower(neuron)`).
    #[new]
    fn new(spec_json: &str) -> PyResult<Self> {
        Model::from_json(spec_json)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    /// Number of output columns per time step.
    #[getter]
    fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// Run a current-clamp simulation and return an `(n_steps, n_columns)` array.
    fn simulate_current_clamp<'py>(
        &self,
        py: Python<'py>,
        i_ext: PyReadonlyArray1<'py, f64>,
        v0: f64,
        gates0: Vec<f64>,
        ca0: f64,
        dt: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let i_ext: Vec<f64> = i_ext.as_array().to_vec();
        // `detach` releases the GIL while Rust works, so other Python threads
        // keep running (and batched sweeps can later run in parallel).
        let trace = py
            .detach(|| {
                self.inner
                    .simulate_current_clamp(&i_ext, v0, &gates0, ca0, dt)
            })
            .map_err(PyValueError::new_err)?;
        let array = Array2::from_shape_vec((trace.n_steps, trace.n_columns), trace.data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

/// Compiled simulation core for patch_sim (spike for issue #410).
#[pymodule]
mod patch_sim_core {
    #[pymodule_export]
    use super::PyModel;
}
