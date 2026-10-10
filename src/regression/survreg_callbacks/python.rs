//! One Python attachment and NumPy argument batch per callback invocation.
use super::*;
use crate::internal::numpy_utils::{FloatMatrix, FloatVec};
use crate::regression::survreg_distributions::{
    SurvregDistribution, SurvregFamily, SurvregTransform,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::cell::RefCell;

thread_local! {
    // Fitting keeps its existing SurvivalError contract. Query calls retain
    // the original Python exception, including nested query callbacks.
    static DPQR_CALLBACK_ERRORS: RefCell<Vec<Option<(String, PyErr)>>> = const { RefCell::new(Vec::new()) };
}

struct DpqrPythonCall;

impl Drop for DpqrPythonCall {
    fn drop(&mut self) {
        DPQR_CALLBACK_ERRORS.with(|errors| {
            errors.borrow_mut().pop();
        });
    }
}

fn dpqr_python_call<T>(operation: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    DPQR_CALLBACK_ERRORS.with(|errors| errors.borrow_mut().push(None));
    let call = DpqrPythonCall;
    let result = operation();
    let callback_error =
        DPQR_CALLBACK_ERRORS.with(|errors| errors.borrow_mut().last_mut().and_then(Option::take));
    drop(call);
    match (result, callback_error) {
        // A callback can catch an error from a nested fitting operation and
        // then return normally. Restore only the surrogate that escaped from
        // the query callback, never an unrelated later arithmetic warning.
        (Err(mapped), Some((signature, original))) if mapped.to_string() == signature => {
            Err(original)
        }
        (result, _) => result,
    }
}

pub(crate) fn dpqr_python_warning(warning: DpqrWarning) -> PyResult<()> {
    Python::attach(|py| {
        py.import("warnings")?.getattr("warn")?.call1((
            warning.message(),
            py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            2,
        ))?;
        Ok(())
    })
}

struct PythonCallbacks {
    init: Py<PyAny>,
    density: Py<PyAny>,
    deviance: Py<PyAny>,
    quantile: Py<PyAny>,
    variance: Option<Py<PyAny>>,
    fitting_variance: Option<Py<PyAny>>,
    parm_names: Vec<String>,
}

struct PythonTransform {
    trans: Py<PyAny>,
    dtrans: Py<PyAny>,
    itrans: Py<PyAny>,
}

fn failure(name: &str, err: PyErr) -> SurvivalError {
    let message = format!("{name} callback failed: {err}");
    DPQR_CALLBACK_ERRORS.with(|errors| {
        if let Some(current) = errors.borrow_mut().last_mut() {
            *current = Some((format!("RuntimeError: {message}"), err));
        }
    });
    SurvivalError::computation(message)
}

fn array<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyAny> {
    FloatVec::from(values.to_vec())
        .into_pyobject(py)
        .unwrap()
        .into_any()
}

impl PythonCallbacks {
    fn call<'py>(
        &self,
        py: Python<'py>,
        callback: &Py<PyAny>,
        mut args: Vec<Bound<'py, PyAny>>,
        parms: &[f64],
    ) -> PyResult<Bound<'py, PyAny>> {
        if !parms.is_empty() {
            let values = if self.parm_names.is_empty() {
                array(py, parms)
            } else {
                let values = PyDict::new(py);
                for (name, value) in self.parm_names.iter().zip(parms) {
                    values.set_item(name, value)?;
                }
                values.into_any()
            };
            args.push(values);
        }
        callback.bind(py).call1(PyTuple::new(py, args)?)
    }
}

impl SurvregCallbacks for PythonCallbacks {
    fn init(&self, y: &[f64], weights: &[f64], parms: &[f64]) -> SurvivalResult<[f64; 2]> {
        Python::attach(|py| -> PyResult<[f64; 2]> {
            let result = self
                .call(
                    py,
                    &self.init,
                    vec![array(py, y), array(py, weights)],
                    parms,
                )?
                .extract::<FloatVec>()?;
            if result.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "init must return location and variance",
                ));
            }
            Ok([result[0], result[1]])
        })
        .map_err(|e| failure("init", e))
    }

    fn density(&self, z: &[f64], parms: &[f64]) -> SurvivalResult<Vec<SurvregDensity>> {
        Python::attach(|py| -> PyResult<Vec<SurvregDensity>> {
            let matrix = self
                .call(py, &self.density, vec![array(py, z)], parms)?
                .extract::<FloatMatrix>()?
                .into_inner();
            if matrix.ncols() != 5 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "density must return a five-column matrix",
                ));
            }
            Ok(matrix
                .outer_iter()
                .map(|r| SurvregDensity {
                    cdf: r[0],
                    survival: r[1],
                    pdf: r[2],
                    score: r[3],
                    curvature: r[4],
                })
                .collect())
        })
        .map_err(|e| failure("density", e))
    }

    fn quantile(&self, p: &[f64], parms: &[f64]) -> SurvivalResult<Vec<f64>> {
        Python::attach(|py| -> PyResult<Vec<f64>> {
            Ok(self
                .call(py, &self.quantile, vec![array(py, p)], parms)?
                .extract::<FloatVec>()?
                .into_inner())
        })
        .map_err(|e| failure("quantile", e))
    }

    fn deviance(
        &self,
        y1: &[f64],
        y2: &[f64],
        status: &[i32],
        scale: &[f64],
        parms: &[f64],
    ) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        Python::attach(|py| -> PyResult<(Vec<f64>, Vec<f64>)> {
            let intervals = status.contains(&3);
            let ncol = if intervals { 3 } else { 2 };
            let mut y = Vec::with_capacity(y1.len() * ncol);
            for i in 0..y1.len() {
                y.push(y1[i]);
                if intervals {
                    y.push(y2[i]);
                }
                y.push(f64::from(status[i]));
            }
            let y = FloatMatrix::from_flat(y, ncol)?
                .into_pyobject(py)?
                .into_any();
            let result = self.call(py, &self.deviance, vec![y, array(py, scale)], parms)?;
            let (center, loglik) = if let Ok(result) = result.cast::<PyDict>() {
                let get = |key: &str| -> PyResult<FloatVec> {
                    result
                        .get_item(key)?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "deviance result is missing {key}"
                            ))
                        })?
                        .extract()
                };
                (get("center")?, get("loglik")?)
            } else {
                result.extract::<(FloatVec, FloatVec)>()?
            };
            Ok((center.into_inner(), loglik.into_inner()))
        })
        .map_err(|e| failure("deviance", e))
    }

    fn fitting_variance(&self, scale_squared: f64, parms: &[f64]) -> SurvivalResult<f64> {
        let Some(callback) = &self.fitting_variance else {
            let value = self.variance(parms)?;
            if !value.is_finite() || value <= 0.0 {
                return Err(SurvivalError::invalid_input(
                    "variance callback must return a finite positive value",
                ));
            }
            return Ok(value);
        };
        Python::attach(|py| {
            self.call(
                py,
                callback,
                vec![scale_squared.into_pyobject(py)?.into_any()],
                parms,
            )?
            .extract::<f64>()
        })
        .map_err(|e| failure("fitting_variance", e))
    }

    fn variance(&self, parms: &[f64]) -> SurvivalResult<f64> {
        let callback = self.variance.as_ref().ok_or_else(|| {
            SurvivalError::invalid_input("custom distribution has no variance callback")
        })?;
        Python::attach(|py| self.call(py, callback, vec![], parms)?.extract::<f64>())
            .map_err(|e| failure("variance", e))
    }
}

fn transform_call(callback: &Py<PyAny>, values: &[f64], name: &str) -> SurvivalResult<Vec<f64>> {
    Python::attach(|py| -> PyResult<Vec<f64>> {
        Ok(callback
            .bind(py)
            .call1((array(py, values),))?
            .extract::<FloatVec>()?
            .into_inner())
    })
    .map_err(|e| failure(name, e))
}

impl SurvregTransformCallbacks for PythonTransform {
    fn transform(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        transform_call(&self.trans, y, "transform")
    }
    fn derivative(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        transform_call(&self.dtrans, y, "transform derivative")
    }
    fn inverse(&self, y: &[f64]) -> SurvivalResult<Vec<f64>> {
        transform_call(&self.itrans, y, "inverse transform")
    }
}

fn callable(py: Python<'_>, value: &Py<PyAny>, name: &str) -> PyResult<()> {
    if value.bind(py).is_callable() {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Missing or invalid {name} function"
        )))
    }
}

#[pymethods]
impl SurvregDistribution {
    /// Original-scale density with vector recycling at each arithmetic stage.
    #[pyo3(name = "pdf_values")]
    fn pdf_values_py(
        &self,
        py: Python<'_>,
        x: FloatVec,
        mean: FloatVec,
        scale: FloatVec,
    ) -> PyResult<Vec<f64>> {
        dpqr_python_call(|| {
            py.detach(|| self.pdf_values_with_warnings(&x, &mean, &scale, &mut dpqr_python_warning))
        })
    }

    /// Original-scale CDF evaluated with one density callback batch.
    #[pyo3(name = "cdf_values")]
    fn cdf_values_py(
        &self,
        py: Python<'_>,
        q: FloatVec,
        mean: FloatVec,
        scale: FloatVec,
    ) -> PyResult<Vec<f64>> {
        dpqr_python_call(|| {
            py.detach(|| self.cdf_values_with_warnings(&q, &mean, &scale, &mut dpqr_python_warning))
        })
    }

    /// Original-scale quantiles evaluated with one quantile callback batch.
    #[pyo3(name = "quantile_values")]
    fn quantile_values_py(
        &self,
        py: Python<'_>,
        p: FloatVec,
        mean: FloatVec,
        scale: FloatVec,
    ) -> PyResult<Vec<f64>> {
        dpqr_python_call(|| {
            py.detach(|| {
                self.quantile_values_with_warnings(&p, &mean, &scale, &mut dpqr_python_warning)
            })
        })
    }

    /// Draw random observations; a supplied seed uses R's uniform stream.
    #[pyo3(name = "sample", signature = (n, mean, scale, seed=None))]
    fn sample_py(
        &self,
        py: Python<'_>,
        n: usize,
        mean: FloatVec,
        scale: FloatVec,
        seed: Option<i32>,
    ) -> PyResult<Vec<f64>> {
        dpqr_python_call(|| {
            py.detach(|| {
                self.sample_with_warnings(n, &mean, &scale, seed, &mut dpqr_python_warning)
            })
        })
    }

    /// Pickle and copy support (see `internal::pickle`).
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        self.reduce_callbacks(py)
    }

    /// `survreg.distributions[[name]]` with optional `parms` (see
    /// [`SurvregDistribution::from_name`]).
    #[new]
    #[pyo3(signature = (name, parms=None))]
    fn new(name: &str, parms: Option<Vec<f64>>) -> PyResult<Self> {
        Ok(Self::from_name(name, parms.as_deref())?)
    }

    /// Exact named DPQR lookup, with deferred query-only parameter handling.
    #[staticmethod]
    #[pyo3(name = "for_query", signature = (name, parms=None, *, _parms_null=false))]
    fn for_query_py(name: &str, parms: Option<FloatVec>, _parms_null: bool) -> PyResult<Self> {
        let parms = parms.map(FloatVec::into_inner);
        let mut result = Self::for_query(name, parms.as_deref())?;
        if result.family == SurvregFamily::T && _parms_null {
            result.query_parms = QueryParms::Null;
        }
        Ok(result)
    }

    #[pyo3(name = "with_query_parms", signature = (parms, *, _parms_null=false))]
    fn with_query_parms_py(&self, parms: FloatVec, _parms_null: bool) -> Self {
        let mut result = self.clone();
        result.parms = parms.into_inner();
        result.query_parms = if result.family == SurvregFamily::T && _parms_null {
            QueryParms::Null
        } else {
            QueryParms::Values
        };
        result
    }

    /// A user-defined distribution built from a base family, a response
    /// transform, an optional fixed scale and parameters (see
    /// [`SurvregDistribution::custom`]).
    #[staticmethod]
    #[pyo3(name = "custom", signature = (name, family, transform, scale=None, parms=None))]
    fn custom_py(
        name: &str,
        family: SurvregFamily,
        transform: SurvregTransform,
        scale: Option<f64>,
        parms: Option<Vec<f64>>,
    ) -> Self {
        Self::custom(name, family, transform, scale, parms.as_deref())
    }

    /// `survregDtest(dlist, verbose = TRUE)`: the problems with this
    /// definition (empty when it is legal).
    #[pyo3(name = "dtest")]
    fn dtest_py(&self) -> Vec<String> {
        self.dtest()
    }

    /// `variance(parms)` of the standardised base distribution.
    #[pyo3(name = "variance")]
    fn variance_py(&self) -> PyResult<f64> {
        Ok(self.variance()?)
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvregDistribution(name='{}', family={:?}, transform={:?}, scale={:?}, parms={:?})",
            self.name, self.family, self.transform, self.scale, self.parms
        )
    }
    /// Define a custom family with vectorized NumPy callbacks. Parameters,
    /// when nonempty, are passed as a final argument (a dict when named).
    #[staticmethod]
    #[pyo3(name = "from_callbacks", signature = (name, init, density, deviance, quantile, variance=None, transform=None, scale=None, parms=None, parm_names=None, fitting_variance=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_callbacks_py(
        py: Python<'_>,
        name: String,
        init: Py<PyAny>,
        density: Py<PyAny>,
        deviance: Py<PyAny>,
        quantile: Py<PyAny>,
        variance: Option<Py<PyAny>>,
        transform: Option<SurvregTransform>,
        scale: Option<f64>,
        parms: Option<FloatVec>,
        parm_names: Option<Vec<String>>,
        fitting_variance: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        for (name, value) in [
            ("init", &init),
            ("density", &density),
            ("deviance", &deviance),
            ("quantile", &quantile),
        ] {
            callable(py, value, name)?;
        }
        if let Some(v) = &variance {
            callable(py, v, "variance")?;
        }
        if let Some(v) = &fitting_variance {
            callable(py, v, "fitting_variance")?;
        }
        let parms = parms.map(FloatVec::into_inner).unwrap_or_default();
        let parm_names = parm_names.unwrap_or_default();
        if !parm_names.is_empty()
            && (parm_names.len() != parms.len()
                || parm_names.iter().any(String::is_empty)
                || parm_names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != parm_names.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "parameter names must be unique and match parms",
            ));
        }
        Ok(Self::from_callbacks(
            name,
            Arc::new(PythonCallbacks {
                init,
                density,
                deviance,
                quantile,
                variance,
                fitting_variance,
                parm_names,
            }),
            transform.unwrap_or(SurvregTransform::Identity),
            scale,
            parms,
        )?)
    }

    /// Replace the response transform with vectorized trans/dtrans/itrans.
    #[pyo3(name = "with_transform")]
    fn with_transform_py(
        &self,
        py: Python<'_>,
        trans: Py<PyAny>,
        dtrans: Py<PyAny>,
        itrans: Py<PyAny>,
    ) -> PyResult<Self> {
        for (name, value) in [("trans", &trans), ("dtrans", &dtrans), ("itrans", &itrans)] {
            callable(py, value, name)?;
        }
        Ok(self.clone().with_transform(Arc::new(PythonTransform {
            trans,
            dtrans,
            itrans,
        })))
    }

    #[pyo3(name = "with_parms")]
    fn with_parms_py(&self, parms: FloatVec) -> PyResult<Self> {
        Ok(self.with_parms(parms.into_inner())?)
    }

    #[pyo3(name = "derived", signature = (name, transform, scale=None))]
    fn derived_py(
        &self,
        name: String,
        transform: SurvregTransform,
        scale: Option<f64>,
    ) -> PyResult<Self> {
        Ok(self.derived(name, transform, scale)?)
    }

    /// Names used for Python callback parameters, or df for Student-t.
    #[getter]
    fn parm_names(&self) -> Vec<String> {
        if let Some(c) = self.python_callbacks() {
            c.parm_names.clone()
        } else if self.family == SurvregFamily::T {
            vec!["df".to_string()]
        } else {
            vec![]
        }
    }

    /// Restore metadata and separately pickled callables.
    #[staticmethod]
    #[pyo3(signature = (state, callbacks, transform, query_parms=None))]
    fn _from_callback_state(
        py: Python<'_>,
        state: &[u8],
        callbacks: Option<&Bound<'_, PyDict>>,
        transform: Option<&Bound<'_, PyDict>>,
        query_parms: Option<&str>,
    ) -> PyResult<Self> {
        let mut result: Self = crate::internal::pickle::decode(py, state)?;
        if let Some(c) = callbacks {
            let required = |name: &str| -> PyResult<Py<PyAny>> {
                Ok(c.get_item(name)?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!("missing {name} callback"))
                    })?
                    .unbind())
            };
            let rebuilt = Self::from_callbacks_py(
                py,
                result.name.clone(),
                required("init")?,
                required("density")?,
                required("deviance")?,
                required("quantile")?,
                c.get_item("variance")?
                    .filter(|v| !v.is_none())
                    .map(Bound::unbind),
                Some(SurvregTransform::Identity),
                result.scale,
                Some(result.parms.clone().into()),
                Some(
                    c.get_item("parm_names")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("missing parameter names")
                        })?
                        .extract()?,
                ),
                c.get_item("fitting_variance")?
                    .filter(|v| !v.is_none())
                    .map(Bound::unbind),
            )?;
            result.callbacks = rebuilt.callbacks;
        }
        if let Some(t) = transform {
            let get = |name: &str| -> PyResult<Py<PyAny>> {
                Ok(t.get_item(name)?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!("missing {name} callback"))
                    })?
                    .unbind())
            };
            result = result.with_transform_py(py, get("trans")?, get("dtrans")?, get("itrans")?)?;
        }
        if let Some(marker) = query_parms {
            if result.family != SurvregFamily::T {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "query parameter marker requires a Student-t distribution",
                ));
            }
            result.query_parms = match marker {
                "values" => QueryParms::Values,
                "missing" => QueryParms::Missing,
                "null" => QueryParms::Null,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "invalid query parameter marker",
                    ));
                }
            };
        } else {
            result.validate()?;
        }
        Ok(result)
    }
}

impl SurvregDistribution {
    fn python_callbacks(&self) -> Option<&PythonCallbacks> {
        self.callbacks
            .as_ref()
            .and_then(|c| (&*c.0 as &dyn Any).downcast_ref())
    }

    pub(crate) fn reduce_callbacks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let query_marker = match self.query_parms {
            QueryParms::Missing => Some("missing"),
            QueryParms::Null => Some("null"),
            QueryParms::Values
                if self.family == SurvregFamily::T
                    && !matches!(self.parms.as_slice(), [df] if df.is_finite() && *df > 2.0) =>
            {
                Some("values")
            }
            _ => None,
        };
        if self.callbacks.is_none() && self.transform_callbacks.is_none() && query_marker.is_none()
        {
            return crate::internal::pickle::reduce(py, self)?.into_pyobject(py);
        }
        let callbacks = if self.callbacks.is_some() {
            let c = self.python_callbacks().ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("native Rust callbacks cannot be pickled")
            })?;
            let d = PyDict::new(py);
            for (name, v) in [
                ("init", &c.init),
                ("density", &c.density),
                ("deviance", &c.deviance),
                ("quantile", &c.quantile),
            ] {
                d.set_item(name, v)?;
            }
            d.set_item("variance", &c.variance)?;
            d.set_item("fitting_variance", &c.fitting_variance)?;
            d.set_item("parm_names", &c.parm_names)?;
            Some(d)
        } else {
            None
        };
        let transform = if let Some(t) = &self.transform_callbacks {
            let t = (&*t.0 as &dyn Any)
                .downcast_ref::<PythonTransform>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "native Rust transforms cannot be pickled",
                    )
                })?;
            let d = PyDict::new(py);
            for (name, v) in [
                ("trans", &t.trans),
                ("dtrans", &t.dtrans),
                ("itrans", &t.itrans),
            ] {
                d.set_item(name, v)?;
            }
            Some(d)
        } else {
            None
        };
        let state =
            bincode::serde::encode_to_vec(self.detached_callbacks(), bincode::config::standard())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let restore = py.get_type::<Self>().getattr("_from_callback_state")?;
        match query_marker {
            None => (restore, (PyBytes::new(py, &state), callbacks, transform)).into_pyobject(py),
            Some(marker) => (
                restore,
                (PyBytes::new(py, &state), callbacks, transform, marker),
            )
                .into_pyobject(py),
        }
    }
}
