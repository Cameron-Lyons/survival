//! Population marginal means (Yates' weighted means) and their tests.
//!
//! Port of the linear-predictor branch of R survival `R/yates.R`
//! (`yates`, `estfun`, `testfun`, `qform`, `gsolve`, `cmatrix`'s contrast
//! matrices).  R builds one model matrix per level of the term of interest
//! over the chosen population (`population = "data"`, `"factorial"`,
//! `"sas"` or a data frame), averages its rows into a contrast matrix
//! `Cmat` and evaluates `Cmat %*% beta` with variance `Cmat V Cmat'`.
//! Building those model matrices needs the formula machinery (`model.matrix`,
//! factor levels, `xlevels`) and stays with the caller; [`population_means`]
//! does the averaging and [`yates`] the estimates and tests.
//!
//! For a Cox model the caller passes `Cmat` restricted to the coefficient
//! columns (R drops the intercept and strata columns) and the offset
//! `-sum(fit$means * beta)` that recentres the predictions.
//!
//! [`yates_risk`] evaluates marginal relative risks with Monte-Carlo
//! coefficient covariance, using a seeded R-compatible normal stream.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;
mod rng;

/// Monte-Carlo covariance for marginal risk predictions. Normal draws use R's
/// default generator and matrix fill order for reproducible cross-language runs.
pub fn yates_risk(
    xmatlist: &[Vec<Vec<f64>>],
    beta: &[f64],
    vmat: &[Vec<f64>],
    means: &[f64],
    nsim: usize,
    seed: u32,
    test: YatesTest,
) -> SurvivalResult<YatesResult> {
    if nsim < 2 {
        return Err(SurvivalError::invalid_input("nsim must be at least two"));
    }
    validate_length(beta.len(), means.len(), "means")?;
    validate_finite(means, "means")?;
    let cmat = population_means(xmatlist, None)?;
    yates(&YatesInput {
        cmat: &cmat,
        beta,
        vmat,
        offset: 0.0,
        sigma2: None,
        estimable: None,
        test,
    })?;
    let (values, vectors) = symmetric_eigen(vmat);
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    if values
        .iter()
        .any(|&value| value < -f64::EPSILON.sqrt() * largest)
    {
        return Err(SurvivalError::invalid_input(
            "coefficient covariance must be positive semidefinite",
        ));
    }
    let p = beta.len();
    let root: Vec<Vec<f64>> = (0..p)
        .map(|i| {
            (0..p)
                .map(|j| {
                    (0..p)
                        .map(|k| vectors[i][k] * vectors[j][k] * values[k].max(0.0).sqrt())
                        .sum()
                })
                .collect()
        })
        .collect();
    let mut rng = rng::RNormal::new(seed);
    let mut z = vec![vec![0.0; p]; nsim];
    for j in 0..p {
        for row in &mut z {
            row[j] = rng.normal();
        }
    }
    let centered: Vec<Vec<Vec<f64>>> = xmatlist
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| row.iter().zip(means).map(|(x, mean)| x - mean).collect())
                .collect()
        })
        .collect();
    let predict = |coef: &[f64]| -> Vec<f64> {
        centered
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|row| row.iter().zip(coef).map(|(x, b)| x * b).sum::<f64>().exp())
                    .sum::<f64>()
                    / rows.len() as f64
            })
            .collect()
    };
    let estimates = predict(beta);
    let mut sample_mean = vec![0.0; estimates.len()];
    let mut covariance = vec![vec![0.0; estimates.len()]; estimates.len()];
    for (draw, z) in z.iter().enumerate() {
        let coef: Vec<f64> = (0..p)
            .map(|j| beta[j] + (0..p).map(|k| z[k] * root[k][j]).sum::<f64>())
            .collect();
        let predictions = predict(&coef);
        validate_finite(&predictions, "simulated risk")?;
        let delta: Vec<f64> = predictions
            .iter()
            .zip(&sample_mean)
            .map(|(x, m)| x - m)
            .collect();
        for (mean, change) in sample_mean.iter_mut().zip(&delta) {
            *mean += change / (draw + 1) as f64;
        }
        for i in 0..estimates.len() {
            for j in 0..estimates.len() {
                covariance[i][j] += delta[i] * (predictions[j] - sample_mean[j]);
            }
        }
    }
    for row in &mut covariance {
        for value in row {
            *value /= (nsim - 1) as f64;
        }
    }
    let identity: Vec<Vec<f64>> = (0..estimates.len())
        .map(|i| (0..estimates.len()).map(|j| f64::from(i == j)).collect())
        .collect();
    let mut result = yates(&YatesInput {
        cmat: &identity,
        beta: &estimates,
        vmat: &covariance,
        offset: 0.0,
        sigma2: None,
        estimable: None,
        test,
    })?;
    result.cmat.clear();
    Ok(result)
}

#[pyfunction(name = "yates_risk")]
#[pyo3(signature=(xmatlist, beta, vmat, means, nsim=200, seed=0, test="global", term=None))]
#[allow(clippy::too_many_arguments)]
pub fn yates_risk_py(
    py: Python<'_>,
    xmatlist: Vec<Vec<Vec<f64>>>,
    beta: Vec<f64>,
    vmat: Vec<Vec<f64>>,
    means: Vec<f64>,
    nsim: usize,
    seed: u32,
    test: &str,
    term: Option<&str>,
) -> PyResult<YatesResult> {
    let test = YatesTest::parse(test)?;
    let mut result = py.detach(|| yates_risk(&xmatlist, &beta, &vmat, &means, nsim, seed, test))?;
    if let Some(term) = term {
        for row in &mut result.test {
            if row.name == "global" {
                row.name = term.to_owned();
            }
        }
    }
    Ok(result)
}

/// Which contrasts of the population marginal means to test (R `test`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YatesTest {
    /// All levels equal to the last one (one chi-square on `k - 1` df).
    Global,
    /// Every pair of levels, one test each.
    Pairwise,
    /// Every level against the mean of all levels, one test each.
    Mean,
}

impl YatesTest {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "global" => Ok(Self::Global),
            "pairwise" => Ok(Self::Pairwise),
            "mean" => Ok(Self::Mean),
            "trend" => Err(SurvivalError::invalid_input(
                "test = \"trend\" is not supported",
            )),
            other => Err(SurvivalError::invalid_input(format!(
                "unknown yates test {other:?}"
            ))),
        }
    }
}

/// Inputs of [`yates`].
#[derive(Debug, Clone)]
pub struct YatesInput<'a> {
    /// Population-averaged design rows, one per level of the term, over the
    /// coefficient columns (R's `Cmat`).
    pub cmat: &'a [Vec<f64>],
    pub beta: &'a [f64],
    /// Variance matrix of `beta`.
    pub vmat: &'a [Vec<f64>],
    /// Added to every estimate (`-sum(fit$means * beta)` for `coxph`, 0
    /// otherwise).
    pub offset: f64,
    /// Residual variance of a linear model, which adds R's `ss` column.
    pub sigma2: Option<f64>,
    /// Levels whose estimate is estimable; `None` for all.
    pub estimable: Option<&'a [bool]>,
    pub test: YatesTest,
}

/// One tested contrast: R's `test` matrix row.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesContrast {
    pub name: String,
    pub chisq: f64,
    pub df: usize,
    /// Sum of squares (`chisq * sigma2`), linear models only.
    pub ss: Option<f64>,
}

/// One level's population marginal mean and standard error (R's
/// `estimate` data frame); `NaN` for a non-estimable level.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesEstimate {
    pub pmm: f64,
    pub std: f64,
}

/// R's `yates` object (linear predictor scale).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesResult {
    pub estimate: Vec<YatesEstimate>,
    pub test: Vec<YatesContrast>,
    /// Variance matrix of the estimable marginal means (R `mvar`).
    pub mvar: Vec<Vec<f64>>,
    pub cmat: Vec<Vec<f64>>,
}

/// Average the rows of each level's model matrix, optionally with case
/// weights (R's `meanfun`): `Cmat[i, ] = colMeans(xmatlist[[i]])`.
pub fn population_means(
    xmatlist: &[Vec<Vec<f64>>],
    weights: Option<&[f64]>,
) -> SurvivalResult<Vec<Vec<f64>>> {
    let mut cmat = Vec::with_capacity(xmatlist.len());
    for (level, rows) in xmatlist.iter().enumerate() {
        if rows.is_empty() {
            return Err(SurvivalError::invalid_input(format!(
                "population matrix {level} has no rows"
            )));
        }
        let width = rows[0].len();
        if let Some(weights) = weights {
            validate_length(rows.len(), weights.len(), "weights")?;
        }
        let mut sums = vec![0.0; width];
        let mut total = 0.0;
        for (i, row) in rows.iter().enumerate() {
            validate_length(width, row.len(), "population matrix row")?;
            validate_finite(row, "population matrix")?;
            let weight = weights.map_or(1.0, |w| w[i]);
            total += weight;
            for (sum, value) in sums.iter_mut().zip(row) {
                *sum += weight * value;
            }
        }
        cmat.push(sums.into_iter().map(|s| s / total).collect());
    }
    Ok(cmat)
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations;
/// returns the eigenvalues and the eigenvectors as columns of `v`.
fn symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(i == j)).collect())
        .collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let akp = row[p];
                    let akq = row[q];
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (row_p, row_q) = {
                    let (head, tail) = a.split_at_mut(q);
                    (&mut head[p], &mut tail[0])
                };
                for (apk, aqk) in row_p.iter_mut().zip(row_q.iter_mut()) {
                    let (old_p, old_q) = (*apk, *aqk);
                    *apk = c * old_p - s * old_q;
                    *aqk = s * old_p + c * old_q;
                }
                for row in v.iter_mut() {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// R's `qform`: `b' V^- b` with the generalised inverse of `gsolve` (the
/// singular-value decomposition with values below `sqrt(eps)` times the
/// largest set to zero), and the rank as degrees of freedom.
fn quadratic_form(var: &[Vec<f64>], b: &[f64]) -> (f64, usize) {
    let (values, vectors) = symmetric_eigen(var);
    let largest = values.iter().copied().fold(f64::MIN, f64::max);
    let eps = f64::EPSILON.sqrt();
    let threshold = (largest * eps).max(0.0);
    let n = b.len();
    let mut statistic = 0.0;
    let mut rank = 0;
    for (k, &value) in values.iter().enumerate() {
        if value <= threshold {
            continue;
        }
        rank += 1;
        let projection: f64 = (0..n).map(|i| vectors[i][k] * b[i]).sum();
        statistic += projection * projection / value;
    }
    (statistic, rank)
}

fn matrix_product(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let inner = b.len();
    let width = b.first().map_or(0, Vec::len);
    a.iter()
        .map(|row| {
            (0..width)
                .map(|j| (0..inner).map(|k| row[k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let width = a.first().map_or(0, Vec::len);
    (0..width)
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

/// R's `estfun`: estimates `C beta` and their variance `C V C'`.
fn estimates(cmat: &[Vec<f64>], beta: &[f64], vmat: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let estimate = cmat
        .iter()
        .map(|row| row.iter().zip(beta).map(|(c, b)| c * b).sum())
        .collect();
    let var = matrix_product(&matrix_product(cmat, vmat), &transpose(cmat));
    (estimate, var)
}

/// R's `testfun`: the chi-square test of `contrast %*% Cmat %*% beta = 0`.
fn contrast_test(
    name: &str,
    contrast: &[Vec<f64>],
    cmat: &[Vec<f64>],
    beta: &[f64],
    vmat: &[Vec<f64>],
    sigma2: Option<f64>,
) -> YatesContrast {
    let rows = matrix_product(contrast, cmat);
    let (estimate, var) = estimates(&rows, beta, vmat);
    let (chisq, df) = quadratic_form(&var, &estimate);
    YatesContrast {
        name: name.to_string(),
        chisq,
        df,
        ss: sigma2.map(|s| chisq * s),
    }
}

/// The contrast matrices of R's `cmatrix` for `nlev` levels.
fn contrasts(test: YatesTest, nlev: usize) -> SurvivalResult<Vec<(String, Vec<Vec<f64>>)>> {
    match test {
        YatesTest::Global => {
            let rows = (0..nlev.saturating_sub(1))
                .map(|i| {
                    let mut row = vec![0.0; nlev];
                    row[i] = 1.0;
                    row[nlev - 1] = -1.0;
                    row
                })
                .collect();
            Ok(vec![("global".to_string(), rows)])
        }
        YatesTest::Pairwise => {
            if nlev < 2 {
                return Err(SurvivalError::invalid_input(
                    "pairwise tests need at least 2 groups",
                ));
            }
            let mut out = Vec::with_capacity(nlev * (nlev - 1) / 2);
            for i in 0..nlev - 1 {
                for j in i + 1..nlev {
                    let mut row = vec![0.0; nlev];
                    row[i] = 1.0;
                    row[j] = -1.0;
                    out.push((format!("{} vs {}", i + 1, j + 1), vec![row]));
                }
            }
            Ok(out)
        }
        YatesTest::Mean => Ok((0..nlev)
            .map(|k| {
                let mut row = vec![-1.0 / nlev as f64; nlev];
                row[k] = (nlev as f64 - 1.0) / nlev as f64;
                (format!("{} vs mean", k + 1), vec![row])
            })
            .collect()),
    }
}

fn validate(input: &YatesInput<'_>) -> SurvivalResult<()> {
    let p = input.beta.len();
    if input.cmat.is_empty() {
        return Err(SurvivalError::invalid_input("cmat must have rows"));
    }
    validate_finite(input.beta, "beta")?;
    for row in input.cmat {
        validate_length(p, row.len(), "cmat columns")?;
        validate_finite(row, "cmat")?;
    }
    validate_length(p, input.vmat.len(), "vmat rows")?;
    for row in input.vmat {
        validate_length(p, row.len(), "vmat columns")?;
        validate_finite(row, "vmat")?;
    }
    if let Some(estimable) = input.estimable {
        validate_length(input.cmat.len(), estimable.len(), "estimable")?;
    }
    if !input.offset.is_finite() {
        return Err(SurvivalError::invalid_input("offset must be finite"));
    }
    Ok(())
}

/// Population marginal means of the levels of a term, their standard
/// errors, variance matrix and the requested contrast tests.
pub fn yates(input: &YatesInput<'_>) -> SurvivalResult<YatesResult> {
    validate(input)?;
    let nlev = input.cmat.len();
    let estimable: Vec<bool> = input
        .estimable
        .map_or_else(|| vec![true; nlev], <[bool]>::to_vec);
    let kept: Vec<usize> = (0..nlev).filter(|&i| estimable[i]).collect();
    let mut estimate = vec![
        YatesEstimate {
            pmm: f64::NAN,
            std: f64::NAN,
        };
        nlev
    ];
    let mut mvar = Vec::new();
    if !kept.is_empty() {
        let rows: Vec<Vec<f64>> = kept.iter().map(|&i| input.cmat[i].clone()).collect();
        let (values, var) = estimates(&rows, input.beta, input.vmat);
        for (k, &i) in kept.iter().enumerate() {
            estimate[i] = YatesEstimate {
                pmm: values[k] + input.offset,
                std: var[k][k].sqrt(),
            };
        }
        mvar = var;
    }
    let test = contrasts(input.test, nlev)?
        .into_iter()
        .map(|(name, contrast)| {
            // nafun: a contrast touching a non-estimable level is NA
            let uses_missing =
                (0..nlev).any(|j| !estimable[j] && contrast.iter().any(|row| row[j] != 0.0));
            if uses_missing {
                return YatesContrast {
                    name,
                    chisq: f64::NAN,
                    df: 0,
                    ss: input.sigma2.map(|_| f64::NAN),
                };
            }
            contrast_test(
                &name,
                &contrast,
                input.cmat,
                input.beta,
                input.vmat,
                input.sigma2,
            )
        })
        .collect();
    Ok(YatesResult {
        estimate,
        test,
        mvar,
        cmat: input.cmat.to_vec(),
    })
}

/// Python entry point: `yates(cmat, beta, vmat, offset=0.0, sigma2=None,
/// estimable=None, test="global")`.
#[pyfunction(name = "yates")]
#[pyo3(signature = (cmat, beta, vmat, offset=0.0, sigma2=None, estimable=None, test="global"))]
pub fn yates_py(
    cmat: Vec<Vec<f64>>,
    beta: Vec<f64>,
    vmat: Vec<Vec<f64>>,
    offset: f64,
    sigma2: Option<f64>,
    estimable: Option<Vec<bool>>,
    test: &str,
) -> PyResult<YatesResult> {
    Ok(yates(&YatesInput {
        cmat: &cmat,
        beta: &beta,
        vmat: &vmat,
        offset,
        sigma2,
        estimable: estimable.as_deref(),
        test: YatesTest::parse(test)?,
    })?)
}

/// Python entry point for [`population_means`]: `xmatlist` is a list of
/// model matrices (nested rows), one per level.
#[pyfunction(name = "yates_population_means")]
#[pyo3(signature = (xmatlist, weights=None))]
pub fn population_means_py(
    xmatlist: Vec<Vec<Vec<f64>>>,
    weights: Option<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(population_means(&xmatlist, weights.as_deref())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigen_recovers_a_known_spectrum() {
        let (values, vectors) = symmetric_eigen(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 3.0).abs() < 1e-12);
        for column in 0..2 {
            let norm: f64 = vectors.iter().map(|row| row[column] * row[column]).sum();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn quadratic_form_uses_the_generalised_inverse() {
        let (stat, df) = quadratic_form(&[vec![2.0, 0.0], vec![0.0, 4.0]], &[2.0, 2.0]);
        assert!((stat - 3.0).abs() < 1e-12);
        assert_eq!(df, 2);
        // rank-one matrix: only the projected component counts
        let (stat, df) = quadratic_form(&[vec![1.0, 1.0], vec![1.0, 1.0]], &[1.0, 1.0]);
        assert!((stat - 1.0).abs() < 1e-12);
        assert_eq!(df, 1);
    }

    #[test]
    fn marginal_means_and_global_test_follow_r() {
        // Two-level factor plus a covariate: Cmat rows are the level's
        // dummy with the covariate at its population mean.
        let cmat = vec![vec![0.0, 3.0], vec![1.0, 3.0]];
        let beta = [0.5, 0.2];
        let vmat = vec![vec![0.04, 0.01], vec![0.01, 0.02]];
        let result = yates(&YatesInput {
            cmat: &cmat,
            beta: &beta,
            vmat: &vmat,
            offset: -0.6,
            sigma2: None,
            estimable: None,
            test: YatesTest::Global,
        })
        .unwrap();
        assert!((result.estimate[0].pmm - 0.0).abs() < 1e-12);
        assert!((result.estimate[1].pmm - 0.5).abs() < 1e-12);
        // var of row 0 = 9 * 0.02 = 0.18
        assert!((result.estimate[0].std - 0.18f64.sqrt()).abs() < 1e-12);
        assert_eq!(result.test.len(), 1);
        assert_eq!(result.test[0].name, "global");
        assert_eq!(result.test[0].df, 1);
        // contrast (1, -1): difference -0.5, variance 0.04
        assert!((result.test[0].chisq - 0.25 / 0.04).abs() < 1e-9);
        assert_eq!(result.mvar.len(), 2);
    }

    #[test]
    fn pairwise_and_mean_contrasts_and_missing_levels() {
        let cmat = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let beta = [0.5, 1.0];
        let vmat = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let pairwise = yates(&YatesInput {
            cmat: &cmat,
            beta: &beta,
            vmat: &vmat,
            offset: 0.0,
            sigma2: Some(2.0),
            estimable: None,
            test: YatesTest::Pairwise,
        })
        .unwrap();
        assert_eq!(pairwise.test.len(), 3);
        assert_eq!(pairwise.test[0].name, "1 vs 2");
        assert!((pairwise.test[0].chisq - 2.5).abs() < 1e-9);
        assert_eq!(pairwise.test[0].ss, Some(5.0));
        let mean = yates(&YatesInput {
            cmat: &cmat,
            beta: &beta,
            vmat: &vmat,
            offset: 0.0,
            sigma2: None,
            estimable: Some(&[true, true, false]),
            test: YatesTest::Mean,
        })
        .unwrap();
        assert!(mean.estimate[2].pmm.is_nan());
        assert!(mean.test.iter().all(|t| t.chisq.is_nan()));
        assert_eq!(mean.mvar.len(), 2);
    }

    #[test]
    fn population_means_average_rows() {
        let xmatlist = vec![
            vec![vec![1.0, 2.0], vec![1.0, 4.0]],
            vec![vec![0.0, 2.0], vec![0.0, 4.0]],
        ];
        let cmat = population_means(&xmatlist, None).unwrap();
        assert_eq!(cmat, vec![vec![1.0, 3.0], vec![0.0, 3.0]]);
        let weighted = population_means(&xmatlist, Some(&[3.0, 1.0])).unwrap();
        assert!((weighted[0][1] - 2.5).abs() < 1e-12);
        assert!(YatesTest::parse("trend").is_err());
    }
}
