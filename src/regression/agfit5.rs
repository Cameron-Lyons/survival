use crate::utilities::matrix::{lu_solve, matrix_inverse};
use crate::utilities::statistical::normal_cdf;
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug)]
pub struct CoxResult {
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

pub struct CoxModelData<'a> {
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

pub struct CoxFitParams {
    pub max_iter: i32,
    pub eps: f64,
}

struct CoxState {
    covar: Vec<Vec<f64>>,
    a: Vec<f64>,
    a2: Vec<f64>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    event: Vec<i32>,
    frail: Vec<i32>,
    score: Vec<f64>,
    strata: Vec<i32>,
}
impl CoxState {
    fn new(data: &CoxModelData<'_>, _params: &CoxFitParams) -> Self {
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
            offset: offset2.to_vec(),
            weights: weights2.to_vec(),
            event: yy[2 * nused..3 * nused].iter().map(|&x| x as i32).collect(),
            frail: frail2.to_vec(),
            score: vec![0.0; nused],
            strata: strata.to_vec(),
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
        let nvar = beta.len();
        let nfrail = self.frail.len();
        let nvar2 = nvar + nfrail;
        self.a.fill(0.0);
        self.a2.fill(0.0);
        u.fill(0.0);
        imat.fill(0.0);
        for person in 0..self.weights.len() {
            let mut zbeta = self.offset[person];
            for (i, beta_val) in beta.iter().enumerate().take(nvar) {
                zbeta += beta_val * self.covar[i][person];
            }
            if nfrail > 0 {
                zbeta += beta[nvar] * self.frail[person] as f64;
            }
            self.score[person] = zbeta;
        }
        *loglik = 0.0;
        let mut istrat = 0;
        let _indx2 = 0;
        while istrat < self.strata.len() {
            let _denom = 0.0;
            let mut risk_sum = 0.0;
            for person in istrat..self.weights.len() {
                if self.strata[person] != self.strata[istrat] {
                    break;
                }
                let risk_score = self.score[person].exp();
                risk_sum += self.weights[person] * risk_score;
                let _ = self.weights[person] * risk_score * risk_score;
            }
            for person in istrat..self.weights.len() {
                if self.strata[person] != self.strata[istrat] {
                    break;
                }
                if self.event[person] == 1 {
                    *loglik += self.weights[person] * self.score[person];
                    *loglik -= self.weights[person] * risk_sum.ln();
                    for (i, u_elem) in u.iter_mut().enumerate().take(nvar) {
                        let mut temp = 0.0;
                        for j in person..self.weights.len() {
                            if self.strata[j] == self.strata[person] {
                                temp += self.weights[j] * self.score[j].exp() * self.covar[i][j];
                            }
                        }
                        *u_elem += self.weights[person] * (self.covar[i][person] - temp / risk_sum);
                    }
                    if nfrail > 0 {
                        let mut temp = 0.0;
                        for j in person..self.weights.len() {
                            if self.strata[j] == self.strata[person] {
                                temp +=
                                    self.weights[j] * self.score[j].exp() * self.frail[j] as f64;
                            }
                        }
                        u[nvar] +=
                            self.weights[person] * (self.frail[person] as f64 - temp / risk_sum);
                    }
                    for i in 0..nvar {
                        for j in i..nvar {
                            let mut temp = 0.0;
                            for k in person..self.weights.len() {
                                if self.strata[k] == self.strata[person] {
                                    temp += self.weights[k]
                                        * self.score[k].exp()
                                        * self.covar[i][k]
                                        * self.covar[j][k];
                                }
                            }
                            let idx = i * nvar2 + j;
                            imat[idx] += self.weights[person]
                                * (temp / risk_sum
                                    - (self.a[i] * self.a[j]) / (risk_sum * risk_sum));
                        }
                    }
                    if nfrail > 0 {
                        for i in 0..nvar {
                            let mut temp = 0.0;
                            for k in person..self.weights.len() {
                                if self.strata[k] == self.strata[person] {
                                    temp += self.weights[k]
                                        * self.score[k].exp()
                                        * self.covar[i][k]
                                        * self.frail[k] as f64;
                                }
                            }
                            let idx = i * nvar2 + nvar;
                            imat[idx] += self.weights[person]
                                * (temp / risk_sum
                                    - (self.a[i] * self.a[nvar]) / (risk_sum * risk_sum));
                        }
                        let mut temp = 0.0;
                        for k in person..self.weights.len() {
                            if self.strata[k] == self.strata[person] {
                                temp += self.weights[k]
                                    * self.score[k].exp()
                                    * (self.frail[k] as f64).powi(2);
                            }
                        }
                        let idx = nvar * nvar2 + nvar;
                        imat[idx] += self.weights[person]
                            * (temp / risk_sum
                                - (self.a[nvar] * self.a[nvar]) / (risk_sum * risk_sum));
                    }
                }
                if person < self.weights.len() - 1 && self.strata[person + 1] == self.strata[person]
                {
                    let risk_score = self.score[person].exp();
                    risk_sum -= self.weights[person] * risk_score;
                    let _ = self.weights[person] * risk_score * risk_score;
                }
            }
            while istrat < self.strata.len() {
                istrat += 1;
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[pyfunction]
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
pub fn agfit5(
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
            (coef - 1.96 * se, coef + 1.96 * se)
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
    let nvar = covariates.len();
    if nvar == 0 {
        return Err(PyRuntimeError::new_err("No covariates provided"));
    }
    if event.len() != nused {
        return Err(PyRuntimeError::new_err(
            "Event vector length does not match time vector",
        ));
    }
    for cov in &covariates {
        if cov.len() != nused {
            return Err(PyRuntimeError::new_err(
                "Covariate vector length does not match time vector",
            ));
        }
    }
    let offset = config.offset.unwrap_or_else(|| vec![0.0; nused]);
    let weights = config.weights.unwrap_or_else(|| vec![1.0; nused]);
    let strata = config.strata.unwrap_or_else(|| vec![1; nused]);
    let frail = config.frail.unwrap_or_else(|| vec![0; nused]);
    let max_iter = config.max_iter.unwrap_or(20);
    let eps = config.eps.unwrap_or(1e-6);
    let mut yy = Vec::with_capacity(3 * nused);
    yy.extend_from_slice(&time);
    yy.extend_from_slice(&time);
    yy.extend(event.iter().map(|&x| x as f64));
    let mut covar = Vec::with_capacity(nvar * nused);
    for i in 0..nused {
        for covariate_row in covariates.iter().take(nvar) {
            covar.push(covariate_row[i]);
        }
    }
    let sort: Vec<i32> = (1..=nused as i32).collect();
    let nfrail = if frail.iter().any(|&x| x != 0) { 1 } else { 0 };
    let model_data = CoxModelData {
        nused,
        nvar,
        nfrail,
        yy: &yy,
        covar: &covar,
        offset: &offset,
        weights: &weights,
        strata: &strata,
        sort: &sort,
        frail: &frail,
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
