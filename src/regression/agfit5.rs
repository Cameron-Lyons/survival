use crate::constants::normal_ci_95;
use crate::internal::matrix::{lu_solve, matrix_inverse};
use crate::internal::statistical::normal_cdf;
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::borrow::Cow;

#[derive(Debug)]
pub(crate) struct CoxResult {
    pub coefficients: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub p_values: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub log_likelihood: f64,
    pub score: f64,
    pub wald_test: f64,
    pub iterations: i32,
    pub converged: bool,
    pub variance_matrix: Vec<Vec<f64>>,
}

pub(crate) struct CoxModelData<'a> {
    pub nused: usize,
    pub nvar: usize,
    pub nfrail: usize,
    pub yy: &'a [f64],
    pub covar: &'a [f64],
    pub offset: &'a [f64],
    pub weights: &'a [f64],
    pub strata: &'a [i32],
    pub sort: &'a [i32],
    pub frail: &'a [i32],
}

pub(crate) struct CoxFitParams {
    pub max_iter: i32,
    pub eps: f64,
}

struct CoxState<'a> {
    covar: Vec<Vec<f64>>,
    a: Vec<f64>,
    a2: Vec<f64>,
    offset: &'a [f64],
    weights: &'a [f64],
    event: Vec<i32>,
    frail: &'a [i32],
    score: Vec<f64>,
    risk_weighted: Vec<f64>,
    strata: &'a [i32],
}
impl<'a> CoxState<'a> {
    fn new(data: &'a CoxModelData<'a>, _params: &CoxFitParams) -> Self {
        let CoxModelData {
            nused,
            nvar,
            nfrail,
            yy,
            covar: covar2,
            offset: offset2,
            weights: weights2,
            strata,
            sort: _sort,
            frail: frail2,
        } = *data;
        let mut covar = vec![vec![0.0; nused]; nvar];
        let mut k = 0;
        for covar_row in covar.iter_mut().take(nvar) {
            for covar_elem in covar_row.iter_mut().take(nused) {
                *covar_elem = covar2[k];
                k += 1;
            }
        }
        let mut state = CoxState {
            covar,
            a: vec![0.0; 4 * (nvar + nfrail) + 5 * nused],
            a2: vec![0.0; nvar + nfrail],
            offset: offset2,
            weights: weights2,
            event: yy[2 * nused..3 * nused].iter().map(|&x| x as i32).collect(),
            frail: frail2,
            score: vec![0.0; nused],
            risk_weighted: vec![0.0; nused],
            strata,
        };
        for i in 0..nvar {
            let mean = state.covar[i].iter().sum::<f64>() / nused as f64;
            for val in &mut state.covar[i] {
                *val -= mean;
            }
        }
        state
    }

    fn update(&mut self, beta: &mut [f64], u: &mut [f64], imat: &mut [f64], loglik: &mut f64) {
        let nvar = self.covar.len();
        let has_frailty = beta.len() > nvar;
        let nvar2 = beta.len();
        self.a.fill(0.0);
        self.a2.fill(0.0);
        u.fill(0.0);
        imat.fill(0.0);
        for person in 0..self.weights.len() {
            let mut zbeta = self.offset[person];
            for (i, beta_val) in beta.iter().enumerate().take(nvar) {
                zbeta += beta_val * self.covar[i][person];
            }
            if has_frailty {
                zbeta += beta[nvar] * self.frail[person] as f64;
            }
            self.score[person] = zbeta;
            self.risk_weighted[person] = self.weights[person] * zbeta.exp();
        }
        *loglik = 0.0;
        let mut istrat = 0;
        while istrat < self.strata.len() {
            let current_stratum = self.strata[istrat];
            let mut stratum_end = istrat + 1;
            while stratum_end < self.strata.len() && self.strata[stratum_end] == current_stratum {
                stratum_end += 1;
            }

            let mut risk_sum = 0.0;
            for person in istrat..stratum_end {
                risk_sum += self.risk_weighted[person];
            }
            for person in istrat..stratum_end {
                if self.event[person] == 1 {
                    *loglik += self.weights[person] * self.score[person];
                    *loglik -= self.weights[person] * risk_sum.ln();
                    for (i, u_elem) in u.iter_mut().enumerate().take(nvar) {
                        let mut temp = 0.0;
                        for j in person..stratum_end {
                            temp += self.risk_weighted[j] * self.covar[i][j];
                        }
                        *u_elem += self.weights[person] * (self.covar[i][person] - temp / risk_sum);
                    }
                    if has_frailty {
                        let mut temp = 0.0;
                        for j in person..stratum_end {
                            temp += self.risk_weighted[j] * self.frail[j] as f64;
                        }
                        u[nvar] +=
                            self.weights[person] * (self.frail[person] as f64 - temp / risk_sum);
                    }
                    for i in 0..nvar {
                        for j in i..nvar {
                            let mut temp = 0.0;
                            for k in person..stratum_end {
                                temp += self.risk_weighted[k] * self.covar[i][k] * self.covar[j][k];
                            }
                            let idx = i * nvar2 + j;
                            imat[idx] += self.weights[person]
                                * (temp / risk_sum
                                    - (self.a[i] * self.a[j]) / (risk_sum * risk_sum));
                        }
                    }
                    if has_frailty {
                        for i in 0..nvar {
                            let mut temp = 0.0;
                            for k in person..stratum_end {
                                temp +=
                                    self.risk_weighted[k] * self.covar[i][k] * self.frail[k] as f64;
                            }
                            let idx = i * nvar2 + nvar;
                            imat[idx] += self.weights[person]
                                * (temp / risk_sum
                                    - (self.a[i] * self.a[nvar]) / (risk_sum * risk_sum));
                        }
                        let mut temp = 0.0;
                        for k in person..stratum_end {
                            temp += self.risk_weighted[k] * (self.frail[k] as f64).powi(2);
                        }
                        let idx = nvar * nvar2 + nvar;
                        imat[idx] += self.weights[person]
                            * (temp / risk_sum
                                - (self.a[nvar] * self.a[nvar]) / (risk_sum * risk_sum));
                    }
                }
                if person + 1 < stratum_end {
                    risk_sum -= self.risk_weighted[person];
                }
            }
            istrat = stratum_end;
        }
    }
}
fn normalize_covariates(covariates: Vec<Vec<f64>>, nused: usize) -> PyResult<Vec<Vec<f64>>> {
    if covariates.is_empty() {
        return Err(PyRuntimeError::new_err("No covariates provided"));
    }
    if covariates.iter().all(|covariate| covariate.len() == nused) {
        return Ok(covariates);
    }
    if covariates.len() == nused {
        let nvar = covariates[0].len();
        if nvar == 0 {
            return Err(PyRuntimeError::new_err("No covariates provided"));
        }
        if covariates.iter().any(|row| row.len() != nvar) {
            return Err(PyRuntimeError::new_err(
                "Covariate rows must all have the same length",
            ));
        }
        let mut transposed = vec![vec![0.0; nused]; nvar];
        for (row_idx, row) in covariates.iter().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                transposed[col_idx][row_idx] = *value;
            }
        }
        return Ok(transposed);
    }
    Err(PyRuntimeError::new_err(
        "Covariates must be n_observations rows or n_covariates vectors",
    ))
}

fn validate_optional_len<T>(name: &str, values: &Option<Vec<T>>, nused: usize) -> PyResult<()> {
    if let Some(values) = values
        && values.len() != nused
    {
        return Err(PyRuntimeError::new_err(format!(
            "{name} vector length does not match time vector"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    offset=None,
    weights=None,
    strata=None,
    frail=None,
    max_iter=None,
    eps=None,
))]
pub fn perform_cox_regression_frailty(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    offset: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    frail: Option<Vec<i32>>,
    max_iter: Option<i32>,
    eps: Option<f64>,
) -> PyResult<Py<PyAny>> {
    validate_optional_len("Offset", &offset, time.len())?;
    validate_optional_len("Weights", &weights, time.len())?;
    validate_optional_len("Strata", &strata, time.len())?;
    validate_optional_len("Frailty", &frail, time.len())?;
    let config = CoxRegressionConfig {
        offset,
        weights,
        strata,
        frail,
        max_iter,
        eps,
    };
    perform_cox_regression(time, event, covariates, config)
}
pub(crate) fn agfit5(
    data: &CoxModelData<'_>,
    params: &CoxFitParams,
) -> Result<CoxResult, Box<dyn std::error::Error>> {
    let mut state = CoxState::new(data, params);
    let nvar2 = data.nvar + data.nfrail;
    let mut beta = vec![0.0; nvar2];
    let mut u = vec![0.0; nvar2];
    let mut imat = vec![0.0; nvar2 * nvar2];
    let mut loglik = 0.0;
    let mut iter = 0;
    let mut converged = false;
    while iter < params.max_iter {
        let old_loglik = loglik;
        state.update(&mut beta, &mut u, &mut imat, &mut loglik);
        if (loglik - old_loglik).abs() < params.eps {
            converged = true;
            break;
        }
        let mut imat_array = Array2::from_shape_vec((nvar2, nvar2), imat.clone())?;
        let u_array = Array1::from_vec(u.clone());
        for i in 0..nvar2 {
            imat_array[[i, i]] += 1e-8;
        }
        match lu_solve(&imat_array, &u_array) {
            Some(delta) => {
                for i in 0..nvar2 {
                    beta[i] += delta[i];
                }
            }
            None => {
                return Err("Failed to solve linear system".into());
            }
        }
        iter += 1;
    }
    state.update(&mut beta, &mut u, &mut imat, &mut loglik);
    let mut variance_matrix = vec![vec![0.0; nvar2]; nvar2];
    let imat_array = Array2::from_shape_vec((nvar2, nvar2), imat)?;
    match matrix_inverse(&imat_array) {
        Some(inv_imat) => {
            for i in 0..nvar2 {
                for j in 0..nvar2 {
                    variance_matrix[i][j] = inv_imat[[i, j]];
                }
            }
        }
        None => {
            return Err("Failed to invert information matrix".into());
        }
    }
    let standard_errors: Vec<f64> = (0..nvar2).map(|i| variance_matrix[i][i].sqrt()).collect();
    let p_values: Vec<f64> = (0..nvar2)
        .map(|i| {
            if standard_errors[i] > 0.0 {
                let z = beta[i] / standard_errors[i];
                2.0 * (1.0 - normal_cdf(z.abs()))
            } else {
                1.0
            }
        })
        .collect();
    let confidence_intervals: Vec<(f64, f64)> = (0..nvar2)
        .map(|i| {
            let se = standard_errors[i];
            let coef = beta[i];
            normal_ci_95(coef, se)
        })
        .collect();
    let score: f64 = u.iter().map(|&x| x * x).sum();
    let wald_test: f64 = beta
        .iter()
        .zip(standard_errors.iter())
        .map(|(&coef, &se)| if se > 0.0 { (coef / se).powi(2) } else { 0.0 })
        .sum();
    Ok(CoxResult {
        coefficients: beta,
        standard_errors,
        p_values,
        confidence_intervals,
        log_likelihood: loglik,
        score,
        wald_test,
        iterations: iter,
        converged,
        variance_matrix,
    })
}
#[derive(Clone, Default)]
struct CoxRegressionConfig {
    offset: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    frail: Option<Vec<i32>>,
    max_iter: Option<i32>,
    eps: Option<f64>,
}
fn perform_cox_regression(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: CoxRegressionConfig,
) -> PyResult<Py<PyAny>> {
    let nused = time.len();
    if nused == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }
    let covariates = normalize_covariates(covariates, nused)?;
    let nvar = covariates.len();
    if event.len() != nused {
        return Err(PyRuntimeError::new_err(
            "Event vector length does not match time vector",
        ));
    }
    let offset = config
        .offset
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![0.0; nused]));
    let weights = config
        .weights
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![1.0; nused]));
    let strata = config
        .strata
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![1; nused]));
    let frail = config
        .frail
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![0; nused]));
    let max_iter = config.max_iter.unwrap_or(20);
    let eps = config.eps.unwrap_or(1e-6);
    let mut yy = Vec::with_capacity(3 * nused);
    yy.extend_from_slice(&time);
    yy.extend_from_slice(&time);
    yy.extend(event.iter().map(|&x| x as f64));
    let mut covar = Vec::with_capacity(nvar * nused);
    for covariate in &covariates {
        covar.extend(covariate);
    }
    let sort: Vec<i32> = (1..=nused as i32).collect();
    let nfrail = if frail.iter().any(|&x| x != 0) { 1 } else { 0 };
    let model_data = CoxModelData {
        nused,
        nvar,
        nfrail,
        yy: &yy,
        covar: &covar,
        offset: offset.as_ref(),
        weights: weights.as_ref(),
        strata: strata.as_ref(),
        sort: &sort,
        frail: frail.as_ref(),
    };
    let fit_params = CoxFitParams { max_iter, eps };
    match agfit5(&model_data, &fit_params) {
        Ok(result) => Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("coefficients", result.coefficients)?;
            dict.set_item("standard_errors", result.standard_errors)?;
            dict.set_item("p_values", result.p_values)?;
            dict.set_item("confidence_intervals", result.confidence_intervals)?;
            dict.set_item("log_likelihood", result.log_likelihood)?;
            dict.set_item("score", result.score)?;
            dict.set_item("wald_test", result.wald_test)?;
            dict.set_item("iterations", result.iterations)?;
            dict.set_item("converged", result.converged)?;
            dict.set_item("variance_matrix", result.variance_matrix)?;
            Ok(dict.into())
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Cox regression failed: {}",
            e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cox_state_update_caches_weighted_risk_scores() {
        let yy = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0];
        let covar = vec![0.0, 1.0, 2.0];
        let offset = vec![0.0; 3];
        let weights = vec![1.0, 2.0, 1.0];
        let strata = vec![1, 1, 1];
        let sort = vec![1, 2, 3];
        let frail = vec![0, 0, 0];
        let data = CoxModelData {
            nused: 3,
            nvar: 1,
            nfrail: 0,
            yy: &yy,
            covar: &covar,
            offset: &offset,
            weights: &weights,
            strata: &strata,
            sort: &sort,
            frail: &frail,
        };
        let params = CoxFitParams {
            max_iter: 1,
            eps: 1e-6,
        };
        let mut state = CoxState::new(&data, &params);
        let mut beta = vec![0.0];
        let mut u = vec![0.0];
        let mut imat = vec![0.0];
        let mut loglik = 0.0;

        state.update(&mut beta, &mut u, &mut imat, &mut loglik);

        assert_eq!(state.risk_weighted, weights);
        assert!(u.iter().all(|value| value.is_finite()));
        assert!(imat.iter().all(|value| value.is_finite()));
        assert!(loglik.is_finite());
    }

    #[test]
    fn cox_state_update_processes_all_strata() {
        let yy = vec![
            1.0, 2.0, 1.0, 2.0, // start
            1.0, 2.0, 1.0, 2.0, // stop
            1.0, 0.0, 1.0, 0.0, // event
        ];
        let covar = vec![0.0, 0.0, 0.0, 0.0];
        let offset = vec![0.0; 4];
        let weights = vec![1.0; 4];
        let strata = vec![1, 1, 2, 2];
        let sort = vec![1, 2, 3, 4];
        let frail = vec![0, 0, 0, 0];
        let data = CoxModelData {
            nused: 4,
            nvar: 1,
            nfrail: 0,
            yy: &yy,
            covar: &covar,
            offset: &offset,
            weights: &weights,
            strata: &strata,
            sort: &sort,
            frail: &frail,
        };
        let params = CoxFitParams {
            max_iter: 1,
            eps: 1e-6,
        };
        let mut state = CoxState::new(&data, &params);
        let mut beta = vec![0.0];
        let mut u = vec![0.0];
        let mut imat = vec![0.0];
        let mut loglik = 0.0;

        state.update(&mut beta, &mut u, &mut imat, &mut loglik);

        assert!((loglik + 2.0 * 2.0_f64.ln()).abs() < 1e-12);
    }
}
