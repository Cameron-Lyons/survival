
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DeepSurv {
    weights: StoredWeights,
    hidden_layers: Vec<usize>,
    activation: Activation,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub unique_times: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    n_vars: usize,
}

#[pymethods]
impl DeepSurv {
    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_vars, time, status, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        config: &DeepSurvConfig,
    ) -> PyResult<Self> {
        if x.len() != n_obs * n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x length must equal n_obs * n_vars",
            ));
        }
        if time.len() != n_obs || status.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and status must have length n_obs",
            ));
        }

        let config = config.clone();
        Ok(py.detach(move || fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config)))
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        if x_new.len() != n_new * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_new dimensions don't match",
            ));
        }

        Ok(predict_with_weights(
            &x_new,
            n_new,
            self.n_vars,
            &self.weights,
            self.activation,
        ))
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let survival: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| (-h * risk.exp()).exp())
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_cumulative_hazard(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let cumhaz: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| h * risk.exp())
                    .collect()
            })
            .collect();

        Ok(cumhaz)
    }

    #[pyo3(signature = (x_new, n_new, percentile=0.5))]
    pub fn predict_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        percentile: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        if !(0.0..=1.0).contains(&percentile) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "percentile must be between 0 and 1",
            ));
        }

        let survival = self.predict_survival(x_new, n_new)?;

        let times: Vec<Option<f64>> = survival
            .par_iter()
            .map(|surv| {
                for (i, &s) in surv.iter().enumerate() {
                    if s <= percentile && i < self.unique_times.len() {
                        return Some(self.unique_times[i]);
                    }
                }
                None
            })
            .collect();

        Ok(times)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_median_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        self.predict_survival_time(x_new, n_new, 0.5)
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.n_vars
    }

    #[getter]
    pub fn get_hidden_layers(&self) -> Vec<usize> {
        self.hidden_layers.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config=None))]
pub fn deep_surv(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: Option<&DeepSurvConfig>,
) -> PyResult<DeepSurv> {
    let cfg = match config.cloned() {
        Some(cfg) => cfg,
        None => DeepSurvConfig::new(
            None,
            Activation::SELU,
            0.2,
            0.001,
            256,
            100,
            0.0001,
            None,
            None,
            0.1,
        )?,
    };

    DeepSurv::fit(py, x, n_obs, n_vars, time, status, &cfg)
}

