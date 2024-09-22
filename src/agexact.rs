use itertools::Itertools;
use ndarray::prelude::*;
use pyo3::prelude::*;

/// Cox proportional hazards model.
#[pyclass]
struct CoxModel {
    max_iter: usize,
    n_used: usize,
    n_var: usize,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<bool>,
    covar: Array2<f64>,
    strata: Vec<usize>,
    offset: Array1<f64>,
    eps: f64,
    tol_chol: f64,
    means: Array1<f64>,
    beta: Array1<f64>,
    u: Array1<f64>,
    imat: Array2<f64>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter_used: usize,
}

#[pymethods]
impl CoxModel {
    /// Creates a new CoxModel instance.
    #[new]
    pub fn new(
        max_iter: usize,
        n_used: usize,
        n_var: usize,
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<bool>,
        covar: Vec<Vec<f64>>,
        strata: Vec<usize>,
        offset: Vec<f64>,
        eps: f64,
        tol_chol: f64,
        initial_beta: Vec<f64>,
    ) -> PyResult<Self> {
        let covar_flat: Vec<f64> = covar.into_iter().flatten().collect();
        let covar_array = Array2::from_shape_vec((n_used, n_var), covar_flat).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid covariate matrix dimensions")
        })?;
        let offset_array = Array1::from(offset);
        let means = Array1::zeros(n_var);
        let beta = Array1::from(initial_beta);
        let u = Array1::zeros(n_var);
        let imat = Array2::zeros((n_var, n_var));
        let loglik = [0.0, 0.0];

        Ok(CoxModel {
            max_iter,
            n_used,
            n_var,
            start,
            stop,
            event,
            covar: covar_array,
            strata,
            offset: offset_array,
            eps,
            tol_chol,
            means,
            beta,
            u,
            imat,
            loglik,
            sctest: 0.0,
            flag: 0,
            iter_used: 0,
        })
    }

    /// Fits the Cox model using the Newton-Raphson algorithm.
    pub fn compute(&mut self) -> PyResult<()> {
        for iter in 0..self.max_iter {
            self.score_and_info()?;
            let delta_beta = self.solve_system()?;
            self.beta += &delta_beta;

            if self.has_converged(&delta_beta) {
                self.iter_used = iter + 1;
                self.flag = 0;
                break;
            }
        }

        if self.flag != 0 {
            self.flag = 1000;
        }
        self.finalize_statistics()?;
        Ok(())
    }

    /// Returns the fitted beta coefficients.
    #[getter]
    fn beta(&self) -> Vec<f64> {
        self.beta.to_vec()
    }

    /// Returns the log-likelihood of the model.
    #[getter]
    fn loglik(&self) -> [f64; 2] {
        self.loglik
    }

    /// Returns the score test statistic.
    #[getter]
    fn sctest(&self) -> f64 {
        self.sctest
    }
}

// Private methods (not exposed to Python)
impl CoxModel {
    /// Checks if the algorithm has converged based on the change in beta.
    fn has_converged(&self, delta_beta: &Array1<f64>) -> bool {
        delta_beta.iter().all(|&change| change.abs() <= self.eps)
    }

    /// Computes the score vector and information matrix for the Cox model.
    fn score_and_info(&mut self) -> PyResult<()> {
        let mut score = Array1::<f64>::zeros(self.n_var);
        let mut info_matrix = Array2::<f64>::zeros((self.n_var, self.n_var));

        let beta = &self.beta;
        let covar = &self.covar;
        let offset = &self.offset;

        let lin_pred = covar.dot(beta) + offset;
        let exp_lin_pred = lin_pred.mapv(f64::exp);

        // Group indices by strata
        let strata_indices: Vec<_> = self
            .strata
            .iter()
            .enumerate()
            .chunk_by(|&(_, &stratum1), &(_, &stratum2)| stratum1 == stratum2)
            .map(|chunk| chunk.map(|(idx, _)| idx).collect::<Vec<_>>())
            .collect();

        for indices in strata_indices {
            let mut risk_set_sum = Array1::<f64>::zeros(self.n_var);
            let mut denominator = 0.0;

            // Temporary accumulators for this stratum
            let mut stratum_score = Array1::<f64>::zeros(self.n_var);
            let mut stratum_info_matrix = Array2::<f64>::zeros((self.n_var, self.n_var));

            for &i in indices.iter().rev() {
                let exp_lp = exp_lin_pred[i];
                denominator += exp_lp;
                risk_set_sum += covar.row(i).to_owned() * exp_lp;

                if self.event[i] {
                    stratum_score += covar.row(i).to_owned();
                    let weighted_mean = &risk_set_sum / denominator;

                    let diff = covar.row(i).to_owned() - &weighted_mean;
                    let outer = diff.insert_axis(Axis(1)).dot(&diff.insert_axis(Axis(0)));

                    stratum_info_matrix += &outer;
                }
            }

            score += stratum_score - info_matrix.dot(beta);
            info_matrix += stratum_info_matrix;
        }

        self.u = score;
        self.imat = info_matrix;

        Ok(())
    }

    /// Solves the linear system to find the change in beta coefficients.
    fn solve_system(&self) -> PyResult<Array1<f64>> {
        let cholesky = self.imat.clone().cholesky_into().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Cholesky decomposition failed")
        })?;
        let delta_beta = cholesky.solve(&self.u).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to solve linear system")
        })?;
        Ok(delta_beta)
    }

    /// Finalizes the statistics after fitting the model.
    fn finalize_statistics(&mut self) -> PyResult<()> {
        self.compute_log_likelihood()?;
        self.compute_score_test()?;
        Ok(())
    }

    /// Computes the log-likelihood of the fitted model.
    fn compute_log_likelihood(&mut self) -> PyResult<()> {
        let beta = &self.beta;
        let covar = &self.covar;
        let offset = &self.offset;

        let lin_pred = covar.dot(beta) + offset;
        let exp_lin_pred = lin_pred.mapv(f64::exp);

        let mut loglik = 0.0;

        // Group indices by strata
        let strata_indices: Vec<_> = self
            .strata
            .iter()
            .enumerate()
            .chunk_by(|&(_, &stratum1), &(_, &stratum2)| stratum1 == stratum2)
            .map(|chunk| chunk.map(|(idx, _)| idx).collect::<Vec<_>>())
            .collect();

        for indices in strata_indices {
            let mut denominator = 0.0;

            for &i in indices.iter().rev() {
                let exp_lp = exp_lin_pred[i];
                denominator += exp_lp;

                if self.event[i] {
                    loglik += lin_pred[i] - denominator.ln();
                }
            }
        }

        self.loglik[1] = loglik;
        Ok(())
    }

    /// Computes the score test statistic.
    fn compute_score_test(&mut self) -> PyResult<()> {
        let imat_inv = self.imat.clone().inv().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix inversion failed")
        })?;
        let score_test_statistic = self.u.dot(&imat_inv.dot(&self.u));
        self.sctest = score_test_statistic;
        Ok(())
    }
}

#[pymodule]
fn py_cox_model(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CoxModel>()?;
    Ok(())
}
