use crate::utilities::matrix::lu_solve;
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum PSplineError {
    UnsupportedMethod(String),
    MatrixCreationError {
        rows: usize,
        cols: usize,
        reason: String,
    },
    LinearSolveError,
}

impl fmt::Display for PSplineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PSplineError::UnsupportedMethod(method) => write!(
                f,
                "Unsupported penalty method: {}. Supported methods are: GCV, UBRE, REML, AIC, BIC",
                method
            ),
            PSplineError::MatrixCreationError { rows, cols, reason } => {
                write!(
                    f,
                    "Failed to create matrix with shape ({}, {}): {}",
                    rows, cols, reason
                )
            }
            PSplineError::LinearSolveError => {
                write!(
                    f,
                    "Failed to solve linear system: matrix may be singular or ill-conditioned"
                )
            }
        }
    }
}

impl std::error::Error for PSplineError {}
impl From<PSplineError> for PyErr {
    fn from(err: PSplineError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
#[pyclass]
pub struct PSpline {
    x: Vec<f64>,
    df: u32,
    theta: f64,
    nterm: u32,
    degree: u32,
    eps: f64,
    method: String,
    boundary_knots: (f64, f64),
    intercept: bool,
    penalty: bool,
    #[pyo3(get)]
    coefficients: Option<Vec<f64>>,
    #[pyo3(get)]
    fitted: bool,
}

#[pyclass]
pub struct PSplineBuilder {
    x: Vec<f64>,
    df: u32,
    theta: f64,
    eps: f64,
    method: String,
    boundary_knots: Option<(f64, f64)>,
    intercept: bool,
    penalty: bool,
}

#[pymethods]
impl PSplineBuilder {
    #[new]
    pub fn new(x: Vec<f64>) -> Self {
        Self {
            x,
            df: 4,
            theta: 1.0,
            eps: 1e-5,
            method: "GCV".to_string(),
            boundary_knots: None,
            intercept: true,
            penalty: true,
        }
    }

    pub fn df(mut self_: PyRefMut<'_, Self>, df: u32) -> PyRefMut<'_, Self> {
        self_.df = df;
        self_
    }

    pub fn theta(mut self_: PyRefMut<'_, Self>, theta: f64) -> PyRefMut<'_, Self> {
        self_.theta = theta;
        self_
    }

    pub fn eps(mut self_: PyRefMut<'_, Self>, eps: f64) -> PyRefMut<'_, Self> {
        self_.eps = eps;
        self_
    }

    pub fn method(mut self_: PyRefMut<'_, Self>, method: String) -> PyRefMut<'_, Self> {
        self_.method = method;
        self_
    }

    pub fn boundary_knots(
        mut self_: PyRefMut<'_, Self>,
        lower: f64,
        upper: f64,
    ) -> PyRefMut<'_, Self> {
        self_.boundary_knots = Some((lower, upper));
        self_
    }

    pub fn intercept(mut self_: PyRefMut<'_, Self>, intercept: bool) -> PyRefMut<'_, Self> {
        self_.intercept = intercept;
        self_
    }

    pub fn penalty(mut self_: PyRefMut<'_, Self>, penalty: bool) -> PyRefMut<'_, Self> {
        self_.penalty = penalty;
        self_
    }

    pub fn build(&self) -> PyResult<PSpline> {
        let boundary_knots = self.boundary_knots.unwrap_or_else(|| {
            let min = self.x.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = self.x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        });

        Ok(PSpline::new(
            self.x.clone(),
            self.df,
            self.theta,
            self.eps,
            self.method.clone(),
            boundary_knots,
            self.intercept,
            self.penalty,
        ))
    }
}
#[pymethods]
impl PSpline {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Vec<f64>,
        df: u32,
        theta: f64,
        eps: f64,
        method: String,
        boundary_knots: (f64, f64),
        intercept: bool,
        penalty: bool,
    ) -> Self {
        let nterm = df + 1;
        let degree = 3;
        PSpline {
            x,
            df,
            theta,
            nterm,
            degree,
            eps,
            method,
            boundary_knots,
            intercept,
            penalty,
            coefficients: None,
            fitted: false,
        }
    }
    pub fn fit(&mut self) -> PyResult<Vec<f64>> {
        let basis = self.create_basis();
        let penalized_basis = self
            .apply_penalty(basis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let coefficients = self
            .optimize_fit(penalized_basis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.fitted = true;
        self.coefficients = Some(coefficients.clone());
        Ok(coefficients)
    }
    pub fn predict(&self, new_x: Vec<f64>) -> PyResult<Vec<f64>> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;
        let mut predictions = Vec::with_capacity(new_x.len());
        for x_val in &new_x {
            let mut pred = 0.0;
            for (j, coef) in coefficients.iter().enumerate() {
                pred += coef * self.basis_function(*x_val, j as u32);
            }
            predictions.push(pred);
        }
        Ok(predictions)
    }
    #[getter]
    pub fn get_df(&self) -> u32 {
        self.df
    }
    #[getter]
    pub fn get_eps(&self) -> f64 {
        self.eps
    }
    fn create_basis(&self) -> Vec<Vec<f64>> {
        let n = self.x.len();
        let mut basis = vec![vec![0.0; self.nterm as usize]; n];
        for (i, x_val) in self.x.iter().enumerate() {
            for j in 0..self.nterm {
                basis[i][j as usize] = self.basis_function(*x_val, j);
            }
        }
        basis
    }
    fn basis_function(&self, x: f64, j: u32) -> f64 {
        let mut b = 0.0;
        if self.intercept {
            b += 1.0;
        }
        if j == 0 {
            return b;
        }
        let (a, b) = self.boundary_knots;
        let t = (x - a) / (b - a);
        let knots = self.knots();
        let mut d = vec![0.0; self.degree as usize + 1];
        d[0] = 1.0;
        for k in 1..=self.degree {
            for i in 0..=(self.degree - k) {
                let i_usize = i as usize;
                let k_usize = k as usize;
                let denom = knots[i_usize + k_usize] - knots[i_usize];
                let w = if denom.abs() > 1e-10 {
                    ((t - knots[i_usize]) / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                d[i_usize] = (1.0 - w) * d[i_usize] + w * d[i_usize + 1];
            }
        }
        b * d[0]
    }
    fn knots(&self) -> Vec<f64> {
        let (a, b) = self.boundary_knots;
        let mut knots = vec![0.0; self.nterm as usize + self.degree as usize + 1];
        for i in 0..self.degree + 1 {
            knots[i as usize] = a;
            knots[(self.nterm + self.degree - i) as usize] = b;
        }
        knots
    }
    fn apply_penalty(&self, basis: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PSplineError> {
        if !self.penalty {
            return Ok(basis);
        }
        let mut penalty = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                penalty[i as usize][j as usize] = self.penalty_function(i, j)?;
            }
        }
        let mut result = basis.clone();
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                for k in 0..self.nterm {
                    result[i as usize][j as usize] -= self.theta
                        * penalty[i as usize][k as usize]
                        * basis[k as usize][j as usize];
                }
            }
        }
        Ok(result)
    }
    fn penalty_function(&self, i: u32, j: u32) -> Result<f64, PSplineError> {
        match self.method.as_str() {
            "GCV" => Ok(self.gcv(i, j)),
            "UBRE" => Ok(self.ubre(i, j)),
            "REML" => Ok(self.reml(i, j)),
            "AIC" => Ok(self.aic(i, j)),
            "BIC" => Ok(self.bic(i, j)),
            _ => Err(PSplineError::UnsupportedMethod(self.method.clone())),
        }
    }
    fn reml(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        let n = self.nterm as f64;
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        let edf = trace / n;
        df / (1.0 - edf).powi(2) + (n - edf).ln()
    }
    fn aic(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        let n = self.nterm as f64;
        let edf = trace / n;
        df / (1.0 - edf).powi(2) + 2.0 * edf
    }
    fn bic(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        let n = self.nterm as f64;
        let edf = trace / n;
        df / (1.0 - edf).powi(2) + n.ln() * edf
    }
    fn gcv(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        df / (1.0 - trace / self.nterm as f64).powi(2)
    }
    fn ubre(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        df / (1.0 - trace / self.nterm as f64).powi(2)
    }
    fn optimize_fit(&self, basis: Vec<Vec<f64>>) -> Result<Vec<f64>, PSplineError> {
        let mut a = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        let mut b = vec![0.0; self.nterm as usize];
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                for k in 0..self.nterm {
                    a[i as usize][j as usize] +=
                        basis[k as usize][i as usize] * basis[k as usize][j as usize];
                }
            }
        }
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                b[i as usize] += basis[j as usize][i as usize];
            }
        }
        self.solve(a, b)
    }
    fn solve(&self, a: Vec<Vec<f64>>, b: Vec<f64>) -> Result<Vec<f64>, PSplineError> {
        let n = a.len();
        let a_array =
            Array2::from_shape_vec((n, n), a.into_iter().flatten().collect()).map_err(|e| {
                PSplineError::MatrixCreationError {
                    rows: n,
                    cols: n,
                    reason: e.to_string(),
                }
            })?;
        let b_array = Array1::from_vec(b);
        let x = lu_solve(&a_array, &b_array).ok_or(PSplineError::LinearSolveError)?;
        Ok(x.to_vec())
    }
}
