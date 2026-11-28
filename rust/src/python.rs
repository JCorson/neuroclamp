//! Python bindings (compiled only with the default `python` feature).
//!
//! Exposes one class to Python:
//!
//! ```python
//! from neuroclamp import _core
//! model = _core.Model(spec_json)
//! out = model.simulate_current_clamp(i_ext, v0, gates0, ca0, dt)  # 2-D ndarray
//! out = model.simulate_voltage_clamp(v_protocol, gates0, ca0, dt)
//! ```
//!
//! Everything numeric lives in the pure-Rust modules; this file only converts
//! between Python and Rust types.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::current_clamp::{Model, Trace};

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
        // keep running.
        let trace = py
            .detach(|| {
                self.inner
                    .simulate_current_clamp(&i_ext, v0, &gates0, ca0, dt)
            })
            .map_err(PyValueError::new_err)?;
        to_array(py, trace)
    }

    /// Run a voltage-clamp simulation and return an `(n_steps, n_columns)` array.
    fn simulate_voltage_clamp<'py>(
        &self,
        py: Python<'py>,
        v_protocol: PyReadonlyArray1<'py, f64>,
        gates0: Vec<f64>,
        ca0: f64,
        dt: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let v_protocol: Vec<f64> = v_protocol.as_array().to_vec();
        let trace = py
            .detach(|| {
                self.inner
                    .simulate_voltage_clamp(&v_protocol, &gates0, ca0, dt)
            })
            .map_err(PyValueError::new_err)?;
        to_array(py, trace)
    }

    /// Run one voltage-clamp sweep per array in `v_protocol`, in parallel.
    ///
    /// Sweeps may differ in length, and each has its own initial state: a
    /// fresh voltage-clamp run starts at steady state at its first voltage.
    /// Returns one `(n_steps, n_columns)` array per sweep, in order.
    /// `num_threads` caps the threads the sweeps run on; `None` uses one
    /// per CPU core.
    #[pyo3(signature = (v_protocol, gates0, ca0, dt, num_threads=None))]
    fn simulate_voltage_clamp_batch<'py>(
        &self,
        py: Python<'py>,
        v_protocol: Vec<PyReadonlyArray1<'py, f64>>,
        gates0: Vec<Vec<f64>>,
        ca0: Vec<f64>,
        dt: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let protocols: Vec<Vec<f64>> = v_protocol.iter().map(|a| a.as_array().to_vec()).collect();
        let traces = py
            .detach(|| {
                in_pool(num_threads, || {
                    let refs: Vec<&[f64]> = protocols.iter().map(Vec::as_slice).collect();
                    let gate_refs: Vec<&[f64]> = gates0.iter().map(Vec::as_slice).collect();
                    self.inner
                        .simulate_voltage_clamp_batch(&refs, &gate_refs, &ca0, dt)
                })
            })
            .map_err(PyValueError::new_err)?;
        traces.into_iter().map(|t| to_array(py, t)).collect()
    }

    /// Run one current-clamp sweep per array in `i_ext`, in parallel.
    ///
    /// Sweeps may differ in length. Every sweep starts from the same initial
    /// state, as every fresh current-clamp run does. Returns one
    /// `(n_steps, n_columns)` array per sweep, in order. `num_threads` caps
    /// the threads the sweeps run on; `None` uses one per CPU core.
    #[pyo3(signature = (i_ext, v0, gates0, ca0, dt, num_threads=None))]
    // The arguments are the Python call's, one per keyword; `py` is pyo3's.
    #[allow(clippy::too_many_arguments)]
    fn simulate_current_clamp_batch<'py>(
        &self,
        py: Python<'py>,
        i_ext: Vec<PyReadonlyArray1<'py, f64>>,
        v0: f64,
        gates0: Vec<f64>,
        ca0: f64,
        dt: f64,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let protocols: Vec<Vec<f64>> = i_ext.iter().map(|a| a.as_array().to_vec()).collect();
        let traces = py
            .detach(|| {
                in_pool(num_threads, || {
                    let refs: Vec<&[f64]> = protocols.iter().map(Vec::as_slice).collect();
                    self.inner
                        .simulate_current_clamp_batch(&refs, v0, &gates0, ca0, dt)
                })
            })
            .map_err(PyValueError::new_err)?;
        traces.into_iter().map(|t| to_array(py, t)).collect()
    }
}

/// Run `f` on a dedicated pool of `num_threads` threads, or on rayon's global
/// pool (one thread per CPU core) when `num_threads` is `None`.
///
/// Rayon's parallel iterators inside `f` use whichever pool `install` puts
/// them on, so a dedicated pool caps the batch's CPU use.
fn in_pool<T: Send>(
    num_threads: Option<usize>,
    f: impl FnOnce() -> Result<T, String> + Send,
) -> Result<T, String> {
    match num_threads {
        None => f(),
        Some(0) => Err("num_threads must be at least 1".into()),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| e.to_string())?
            .install(f),
    }
}

/// Convert one trace into an `(n_steps, n_columns)` NumPy array.
fn to_array(py: Python<'_>, trace: Trace) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((trace.n_steps, trace.n_columns), trace.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Compiled simulation core for neuroclamp.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::PyModel;
}
