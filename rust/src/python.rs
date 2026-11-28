//! Python bindings (compiled only with the default `python` feature).
//!
//! Exposes one class to Python:
//!
//! ```python
//! from neuroclamp import _core
//! model = _core.Model(spec_json)
//! out = model.simulate_current_clamp(i_ext, v0, gates0, ca0, dt)  # 2-D ndarray
//! ```
//!
//! Everything numeric lives in the pure-Rust modules; this file only converts
//! between Python and Rust types.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::current_clamp::Model;

/// A compiled neuroclamp model, built once from a lowered JSON spec.
#[pyclass(name = "Model", module = "neuroclamp._core", frozen)]
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

    /// Run one sweep per row of `i_ext` in parallel.
    ///
    /// Returns an `(n_sweeps, n_steps, n_columns)` array. Every sweep starts
    /// from the same initial state.
    fn simulate_current_clamp_batch<'py>(
        &self,
        py: Python<'py>,
        i_ext: PyReadonlyArray2<'py, f64>,
        v0: f64,
        gates0: Vec<f64>,
        ca0: f64,
        dt: f64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let protocols: Vec<Vec<f64>> = i_ext
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let traces = py
            .detach(|| {
                let refs: Vec<&[f64]> = protocols.iter().map(Vec::as_slice).collect();
                self.inner
                    .simulate_current_clamp_batch(&refs, v0, &gates0, ca0, dt)
            })
            .map_err(PyValueError::new_err)?;
        let (n_steps, n_cols) = traces.first().map_or((0, 0), |t| (t.n_steps, t.n_columns));
        let data: Vec<f64> = traces.into_iter().flat_map(|t| t.data).collect();
        let array = Array3::from_shape_vec((protocols.len(), n_steps, n_cols), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

/// Compiled current-clamp core for neuroclamp.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::PyModel;
}
