#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum BasisType {
    BSpline,
    Fourier,
    Wavelet,
    FunctionalPCA,
}

#[pymethods]
impl BasisType {
    fn __repr__(&self) -> String {
        match self {
            BasisType::BSpline => "BasisType.BSpline".to_string(),
            BasisType::Fourier => "BasisType.Fourier".to_string(),
            BasisType::Wavelet => "BasisType.Wavelet".to_string(),
            BasisType::FunctionalPCA => "BasisType.FunctionalPCA".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FunctionalSurvivalConfig {
    #[pyo3(get, set)]
    pub basis_type: BasisType,
    #[pyo3(get, set)]
    pub n_basis: usize,
    #[pyo3(get, set)]
    pub n_pca_components: usize,
    #[pyo3(get, set)]
    pub regularization: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl FunctionalSurvivalConfig {
    #[new]
    #[pyo3(signature = (basis_type=BasisType::BSpline, n_basis=10, n_pca_components=5, regularization=0.01, max_iter=100, tol=1e-6))]
    pub fn new(
        basis_type: BasisType,
        n_basis: usize,
        n_pca_components: usize,
        regularization: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            basis_type,
            n_basis,
            n_pca_components,
            regularization,
            max_iter,
            tol,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FunctionalPCAResult {
    #[pyo3(get)]
    pub eigenvalues: Vec<f64>,
    #[pyo3(get)]
    pub explained_variance_ratio: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_variance: Vec<f64>,
    #[pyo3(get)]
    pub mean_function: Vec<f64>,
    #[pyo3(get)]
    pub principal_components: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub scores: Vec<Vec<f64>>,
}

#[pymethods]
impl FunctionalPCAResult {
    fn __repr__(&self) -> String {
        format!(
            "FunctionalPCAResult(n_components={}, total_var={:.2}%)",
            self.eigenvalues.len(),
            self.cumulative_variance.last().unwrap_or(&0.0) * 100.0
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FunctionalSurvivalResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub coefficient_se: Vec<f64>,
    #[pyo3(get)]
    pub coefficient_function: Vec<f64>,
    #[pyo3(get)]
    pub coefficient_times: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub functional_pca: Option<FunctionalPCAResult>,
    #[pyo3(get)]
    pub basis_coefficients: Vec<Vec<f64>>,
}

#[pymethods]
impl FunctionalSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "FunctionalSurvivalResult(n_coef={}, ll={:.2}, aic={:.2})",
            self.coefficients.len(),
            self.log_likelihood,
            self.aic
        )
    }

    fn predict_coefficient(&self, t: f64) -> f64 {
        if self.coefficient_times.is_empty() {
            return 0.0;
        }

        let idx = self
            .coefficient_times
            .iter()
            .position(|&x| x >= t)
            .unwrap_or(self.coefficient_times.len() - 1);

        self.coefficient_function[idx]
    }
}

#[cfg(test)]
fn fourier_basis(t: f64, n_basis: usize, period: f64) -> Vec<f64> {
    let mut basis = vec![1.0];
    let omega = 2.0 * std::f64::consts::PI / period;

    for k in 1..=n_basis / 2 {
        basis.push((k as f64 * omega * t).sin());
        basis.push((k as f64 * omega * t).cos());
    }

    basis.truncate(n_basis);
    basis
}

fn functional_pca(curves: &[Vec<f64>], n_components: usize) -> FunctionalPCAResult {
    let n_curves = curves.len();
    if n_curves == 0 || curves[0].is_empty() {
        return FunctionalPCAResult {
            eigenvalues: Vec::new(),
            explained_variance_ratio: Vec::new(),
            cumulative_variance: Vec::new(),
            mean_function: Vec::new(),
            principal_components: Vec::new(),
            scores: Vec::new(),
        };
    }

    let n_points = curves[0].len();

    let mean_function: Vec<f64> = (0..n_points)
        .map(|j| curves.iter().map(|c| c[j]).sum::<f64>() / n_curves as f64)
        .collect();

    let centered: Vec<Vec<f64>> = curves
        .iter()
        .map(|c| {
            c.iter()
                .zip(mean_function.iter())
                .map(|(&x, &m)| x - m)
                .collect()
        })
        .collect();

    let mut cov_matrix = vec![vec![0.0; n_points]; n_points];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_points {
        for j in 0..n_points {
            let cov: f64 = centered.iter().map(|c| c[i] * c[j]).sum::<f64>() / n_curves as f64;
            cov_matrix[i][j] = cov;
        }
    }

    let mut eigenvalues = vec![0.0; n_points.min(n_components)];
    let mut eigenvectors = vec![vec![0.0; n_points]; n_points.min(n_components)];

    for k in 0..n_components.min(n_points) {
        let mut v: Vec<f64> = (0..n_points)
            .map(|i| if i == k { 1.0 } else { 0.0 })
            .collect();

        for _ in 0..100 {
            let mut new_v = vec![0.0; n_points];
            for i in 0..n_points {
                for j in 0..n_points {
                    new_v[i] += cov_matrix[i][j] * v[j];
                }
            }

            for eigenvector in eigenvectors.iter().take(k) {
                let dot: f64 = new_v
                    .iter()
                    .zip(eigenvector.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                for (nv, &ev) in new_v.iter_mut().zip(eigenvector.iter()) {
                    *nv -= dot * ev;
                }
            }

            let norm: f64 = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for nv in new_v.iter_mut() {
                    *nv /= norm;
                }
            }

            let diff: f64 = v
                .iter()
                .zip(new_v.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            v = new_v;

            if diff < 1e-8 {
                break;
            }
        }

        let lambda: f64 = (0..n_points)
            .map(|i| {
                let mv: f64 = (0..n_points).map(|j| cov_matrix[i][j] * v[j]).sum();
                mv * v[i]
            })
            .sum();

        eigenvalues[k] = lambda.max(0.0);
        eigenvectors[k] = v;
    }

    let total_var: f64 = eigenvalues.iter().sum();
    let explained_variance_ratio: Vec<f64> = eigenvalues
        .iter()
        .map(|&e| {
            if total_var > 1e-10 {
                e / total_var
            } else {
                0.0
            }
        })
        .collect();

    let mut cumsum = 0.0;
    let cumulative_variance: Vec<f64> = explained_variance_ratio
        .iter()
        .map(|&v| {
            cumsum += v;
            cumsum
        })
        .collect();

    let scores: Vec<Vec<f64>> = centered
        .iter()
        .map(|c| {
            eigenvectors
                .iter()
                .map(|ev| c.iter().zip(ev.iter()).map(|(&x, &e)| x * e).sum())
                .collect()
        })
        .collect();

    FunctionalPCAResult {
        eigenvalues,
        explained_variance_ratio,
        cumulative_variance,
        mean_function,
        principal_components: eigenvectors,
        scores,
    }
}

#[pyfunction]
#[pyo3(signature = (functional_covariates, curve_times, time, event, scalar_covariates=None, config=None))]
pub fn functional_cox(
    functional_covariates: Vec<Vec<f64>>,
    curve_times: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    scalar_covariates: Option<Vec<Vec<f64>>>,
    config: Option<FunctionalSurvivalConfig>,
) -> PyResult<FunctionalSurvivalResult> {
    let config = config.unwrap_or_else(|| {
        FunctionalSurvivalConfig::new(BasisType::BSpline, 10, 5, 0.01, 100, 1e-6)
    });

    let n = functional_covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid input dimensions",
        ));
    }

    let fpca_result = functional_pca(&functional_covariates, config.n_pca_components);

    let n_scalar = scalar_covariates
        .as_ref()
        .map(|s| s.first().map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);

    let n_functional = fpca_result.scores.first().map(|s| s.len()).unwrap_or(0);
    let n_params = n_functional + n_scalar;

    let mut coefficients = vec![0.0; n_params];

    let combined_covariates: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut cov = if i < fpca_result.scores.len() {
                fpca_result.scores[i].clone()
            } else {
                vec![0.0; n_functional]
            };
            if let Some(ref sc) = scalar_covariates
                && i < sc.len()
            {
                cov.extend(sc[i].iter());
            }
            cov
        })
        .collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    let mut log_likelihood = 0.0;

    for _ in 0..config.max_iter {
        let mut gradient = vec![0.0; n_params];
        let mut hessian = vec![vec![0.0; n_params]; n_params];

        let mut risk_set_sum = 0.0;
        let mut risk_set_x_sum = vec![0.0; n_params];
        let mut risk_set_x2_sum = vec![vec![0.0; n_params]; n_params];

        for &i in &sorted_indices {
            let eta: f64 = coefficients
                .iter()
                .zip(combined_covariates[i].iter())
                .map(|(&c, &x)| c * x)
                .sum();
            let exp_eta = eta.exp().min(1e10);

            risk_set_sum += exp_eta;
            for j in 0..n_params {
                risk_set_x_sum[j] += exp_eta * combined_covariates[i][j];
                for k in 0..n_params {
                    risk_set_x2_sum[j][k] +=
                        exp_eta * combined_covariates[i][j] * combined_covariates[i][k];
                }
            }

            if event[i] == 1 {
                log_likelihood += eta - risk_set_sum.ln();

                for j in 0..n_params {
                    if risk_set_sum > 1e-10 {
                        let mean_x = risk_set_x_sum[j] / risk_set_sum;
                        gradient[j] += combined_covariates[i][j] - mean_x;

                        for k in 0..n_params {
                            let mean_xk = risk_set_x_sum[k] / risk_set_sum;
                            hessian[j][k] -=
                                risk_set_x2_sum[j][k] / risk_set_sum - mean_x * mean_xk;
                        }
                    }
                }
            }
        }

        #[allow(clippy::needless_range_loop)]
        for j in 0..n_params {
            hessian[j][j] -= config.regularization;
        }

        let mut max_update: f64 = 0.0;
        for j in 0..n_params {
            let h = -hessian[j][j];
            if h > 1e-10 {
                let update = gradient[j] / h;
                coefficients[j] += update.clamp(-1.0, 1.0);
                max_update = max_update.max(update.abs());
            }
        }

        if max_update < config.tol {
            break;
        }
    }

    let coefficient_se: Vec<f64> = (0..n_params)
        .map(|_j| {
            let h: f64 = 1.0;
            (1.0 / h.max(1e-10)).sqrt()
        })
        .collect();

    let hazard_ratio: Vec<f64> = coefficients.iter().map(|&c| c.exp()).collect();

    let ci_lower: Vec<f64> = coefficients
        .iter()
        .zip(coefficient_se.iter())
        .map(|(c, se): (&f64, &f64)| (c - 1.96 * se).exp())
        .collect();

    let ci_upper: Vec<f64> = coefficients
        .iter()
        .zip(coefficient_se.iter())
        .map(|(c, se): (&f64, &f64)| (c + 1.96 * se).exp())
        .collect();

    let p_values: Vec<f64> = coefficients
        .iter()
        .zip(coefficient_se.iter())
        .map(|(c, se): (&f64, &f64)| {
            let z: f64 = if *se > 1e-10 { c / se } else { 0.0 };
            2.0 * (1.0 - normal_cdf(z.abs()))
        })
        .collect();

    let coefficient_function: Vec<f64> = if !fpca_result.principal_components.is_empty() {
        let n_points = fpca_result.principal_components[0].len();
        (0..n_points)
            .map(|t| {
                (0..n_functional.min(coefficients.len()))
                    .map(|k| coefficients[k] * fpca_result.principal_components[k][t])
                    .sum()
            })
            .collect()
    } else {
        Vec::new()
    };

    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n as f64).ln();

    Ok(FunctionalSurvivalResult {
        coefficients,
        coefficient_se,
        coefficient_function,
        coefficient_times: curve_times,
        hazard_ratio,
        ci_lower,
        ci_upper,
        p_values,
        log_likelihood,
        aic,
        bic,
        functional_pca: Some(fpca_result),
        basis_coefficients: Vec::new(),
    })
}

#[pyfunction]
pub fn fpca_survival(curves: Vec<Vec<f64>>, n_components: usize) -> PyResult<FunctionalPCAResult> {
    if curves.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "curves cannot be empty",
        ));
    }

    Ok(functional_pca(&curves, n_components))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_functional_survival_config() {
        let config = FunctionalSurvivalConfig::new(BasisType::BSpline, 10, 5, 0.01, 100, 1e-6);
        assert_eq!(config.n_basis, 10);
        assert_eq!(config.n_pca_components, 5);
    }

    #[test]
    fn test_functional_pca() {
        let curves = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
            vec![1.2, 2.2, 3.2, 4.2, 5.2],
        ];

        let result = fpca_survival(curves, 2).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.scores.len(), 4);
        assert!(result.cumulative_variance.last().unwrap_or(&0.0) <= &1.0);
    }

    #[test]
    fn test_functional_cox() {
        let functional_covariates = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
            vec![1.2, 2.2, 3.2, 4.2, 5.2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.3, 2.3, 3.3, 4.3, 5.3],
        ];
        let curve_times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 0, 1, 1, 0, 1];

        let result =
            functional_cox(functional_covariates, curve_times, time, event, None, None).unwrap();

        assert!(!result.coefficients.is_empty());
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_fourier_basis() {
        let basis = fourier_basis(0.5, 5, 1.0);
        assert_eq!(basis.len(), 5);
    }
}
