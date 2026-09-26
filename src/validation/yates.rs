//! Population marginal means (Yates' weighted means) and their tests.
//!
//! Port of R survival `R/yates.R` (`yates`, `estfun`, `testfun`, `qform`,
//! `gsolve`, `nafun`, `cmatrix`'s contrast matrices and the prediction
//! functions of `yates_setup.coxph`).  R builds one model matrix per level of
//! the term of interest over the chosen population (`population = "data"`,
//! `"factorial"`, `"sas"` or a data frame).  Building those model matrices
//! needs the formula machinery (`model.matrix`, factor levels, `xlevels`) and
//! stays with the caller.  [`population_means`] averages them into R's
//! contrast matrix `Cmat`; [`yates_estimable`] flags, for a fit with aliased
//! coefficients, the levels outside the row space of the fit's design;
//! [`yates`] evaluates `Cmat %*% beta` with variance `Cmat V Cmat'` and the
//! contrast tests; [`yates_simulate`] averages a nonlinear prediction
//! (`predict = "risk"` or `"survival"`) over the population and estimates its
//! variance from simulated coefficients.
//!
//! For a Cox model the caller passes `Cmat` restricted to the non-aliased
//! coefficient columns (R drops the intercept, strata and `NA` columns) and
//! the offset `-sum(fit$means * beta)` that recentres the predictions.

use std::collections::HashSet;

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::qnorm;
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;
mod rng;

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
    /// Levels whose estimate is estimable (see [`yates_estimable`]); `None` for all.
    pub estimable: Option<&'a [bool]>,
    pub test: YatesTest,
}

/// One tested contrast: R's `test` matrix row.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesContrast {
    pub name: String,
    /// `NaN` for a contrast that uses a non-estimable level (R's `NA`).
    pub chisq: f64,
    /// `None` for a contrast that uses a non-estimable level (R's `NA`).
    pub df: Option<usize>,
    /// Sum of squares (`chisq * sigma2`), linear models only.
    pub ss: Option<f64>,
}

/// One level's population marginal mean and standard error (R's
/// `estimate` data frame); `pmm` is `NaN` for a non-estimable level.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesEstimate {
    pub pmm: f64,
    pub std: f64,
}

/// The survival curves of `predict = "survival"` (R's `summary` component):
/// the simulated mean survival of each level at every time of the baseline
/// curve, one row per time and one column per level.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesCurves {
    pub surv: Vec<Vec<f64>>,
    /// `-log(surv)`.
    pub cumhaz: Vec<Vec<f64>>,
    /// Standard deviation of the simulated survival over `surv`.
    pub std_err: Vec<Vec<f64>>,
    pub lower: Vec<Vec<f64>>,
    pub upper: Vec<Vec<f64>>,
}

/// R's `yates` object.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct YatesResult {
    pub estimate: Vec<YatesEstimate>,
    pub test: Vec<YatesContrast>,
    /// Variance matrix of the marginal means (R `mvar`): over the estimable
    /// levels on the linear predictor scale, over every level for a
    /// simulated prediction; empty when no level is estimable.
    pub mvar: Vec<Vec<f64>>,
    /// R's `Cmat`; empty for a simulated prediction or when no level is
    /// estimable.
    pub cmat: Vec<Vec<f64>>,
    /// The curves of `predict = "survival"`.
    pub summary: Option<YatesCurves>,
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

/// R's estimability check of `yates` for a fit with aliased coefficients.
///
/// The columns of `t(unique(X))`, with a leading row of ones when
/// `intercept` (R's `cbind(1, Xu)` for a Cox model, whose population rows
/// carry the intercept column), span the estimable linear combinations.  A
/// level is estimable when every row of its population model matrix lies in
/// that span: every `abs(qr.resid(qr(t(Xu)), t(xmat)))` is below
/// `sqrt(.Machine$double.eps)`.
pub fn yates_estimable(
    xmatlist: &[Vec<Vec<f64>>],
    x: &[Vec<f64>],
    intercept: bool,
) -> SurvivalResult<Vec<bool>> {
    let Some(first) = x.first() else {
        return Err(SurvivalError::invalid_input("x must have rows"));
    };
    let width = first.len() + usize::from(intercept);
    // unique(X): the first occurrence of every distinct row, in order
    let mut seen = HashSet::with_capacity(x.len());
    let mut columns = Vec::new();
    for row in x {
        validate_length(first.len(), row.len(), "x columns")?;
        validate_finite(row, "x")?;
        // `+ 0.0` makes -0 and 0 the same row, as R's comparison does
        let key: Vec<u64> = row.iter().map(|value| (value + 0.0).to_bits()).collect();
        if seen.insert(key) {
            let mut column = Vec::with_capacity(width);
            if intercept {
                column.push(1.0);
            }
            column.extend_from_slice(row);
            columns.push(column);
        }
    }
    let qr = LinpackQr::new(columns, width);
    let eps = f64::EPSILON.sqrt();
    xmatlist
        .iter()
        .map(|rows| {
            for row in rows {
                validate_length(width, row.len(), "population matrix columns")?;
                validate_finite(row, "population matrix")?;
                if qr.residual(row).iter().any(|value| value.abs() >= eps) {
                    return Ok(false);
                }
            }
            Ok(true)
        })
        .collect()
}

/// R's `qr()` (LINPACK `dqrdc2` with `tol = 1e-7`) of an `n`-row matrix
/// given by its columns, and `qr.resid()` (`dqrsl`) against it.
struct LinpackQr {
    /// The Householder vector of each of the first `min(rank, n - 1)`
    /// columns: column `j` of R's `qr$qr` from row `j` on, with `qraux[j]` in
    /// place of its diagonal.
    householder: Vec<Vec<f64>>,
    rank: usize,
}

impl LinpackQr {
    /// `dqrdc2`: Householder QR with LINPACK's limited pivoting, which moves
    /// a column whose norm has fallen below `tol` times its original norm to
    /// the end; `rank` counts the columns left in front.
    fn new(mut x: Vec<Vec<f64>>, n: usize) -> Self {
        const TOL: f64 = 1e-7;
        let p = x.len();
        let mut qraux: Vec<f64> = x.iter().map(|column| norm(column)).collect();
        let mut original: Vec<f64> = qraux
            .iter()
            .map(|&value| if value == 0.0 { 1.0 } else { value })
            .collect();
        // LINPACK's 1-based `k`: one past the last non-negligible column
        let mut k = p + 1;
        for l in 0..n.min(p) {
            while l + 1 < k && qraux[l] < original[l] * TOL {
                x[l..].rotate_left(1);
                qraux[l..].rotate_left(1);
                original[l..].rotate_left(1);
                k -= 1;
            }
            if l + 1 == n {
                continue;
            }
            let (head, tail) = x.split_at_mut(l + 1);
            let xl = &mut head[l];
            let mut nrmxl = norm(&xl[l..]);
            if nrmxl == 0.0 {
                continue;
            }
            if xl[l] != 0.0 {
                nrmxl = nrmxl.copysign(xl[l]);
            }
            let scale = 1.0 / nrmxl;
            for value in &mut xl[l..] {
                *value *= scale;
            }
            xl[l] += 1.0;
            for (offset, xj) in tail.iter_mut().enumerate() {
                let j = l + 1 + offset;
                let t = -dot(&xl[l..], &xj[l..]) / xl[l];
                for (value, h) in xj[l..].iter_mut().zip(&xl[l..]) {
                    *value += t * h;
                }
                if qraux[j] != 0.0 {
                    let tt = (1.0 - (xj[l].abs() / qraux[j]).powi(2)).max(0.0);
                    if tt < 1e-6 {
                        qraux[j] = norm(&xj[l + 1..]);
                    } else {
                        qraux[j] *= tt.sqrt();
                    }
                }
            }
            qraux[l] = xl[l];
            xl[l] = -nrmxl;
        }
        let rank = (k - 1).min(n);
        let householder = (0..rank.min(n.saturating_sub(1)))
            .map(|j| {
                let mut vector = x[j][j..].to_vec();
                vector[0] = qraux[j];
                vector
            })
            .collect();
        Self { householder, rank }
    }

    /// `qr.resid()`: `y` minus its projection on the span of the first
    /// `rank` columns (`dqrsl` computing `Q'y`, zeroing its first `rank`
    /// entries and applying `Q`).
    fn residual(&self, y: &[f64]) -> Vec<f64> {
        let mut rsd = y.to_vec();
        if self.rank == 0 {
            return rsd;
        }
        if self.householder.is_empty() {
            // one row, of full rank
            rsd[0] = 0.0;
            return rsd;
        }
        for (j, vector) in self.householder.iter().enumerate() {
            reflect(vector, &mut rsd[j..]);
        }
        rsd[..self.rank].fill(0.0);
        for (j, vector) in self.householder.iter().enumerate().rev() {
            reflect(vector, &mut rsd[j..]);
        }
        rsd
    }
}

/// Applies one Householder reflection of `dqrsl` (skipped when its
/// `qraux` is zero) to `y`.
fn reflect(vector: &[f64], y: &mut [f64]) {
    if vector[0] == 0.0 {
        return;
    }
    let t = -dot(vector, y) / vector[0];
    for (value, h) in y.iter_mut().zip(vector) {
        *value += t * h;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
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
    let estimate = cmat.iter().map(|row| dot(row, beta)).collect();
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
        df: Some(df),
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

/// `beta` finite and `vmat` a finite `p x p` matrix.
fn validate_coefficients(beta: &[f64], vmat: &[Vec<f64>]) -> SurvivalResult<()> {
    let p = beta.len();
    validate_finite(beta, "beta")?;
    validate_length(p, vmat.len(), "vmat rows")?;
    for row in vmat {
        validate_length(p, row.len(), "vmat columns")?;
        validate_finite(row, "vmat")?;
    }
    Ok(())
}

fn validate(input: &YatesInput<'_>) -> SurvivalResult<()> {
    if input.cmat.is_empty() {
        return Err(SurvivalError::invalid_input("cmat must have rows"));
    }
    validate_coefficients(input.beta, input.vmat)?;
    for row in input.cmat {
        validate_length(input.beta.len(), row.len(), "cmat columns")?;
        validate_finite(row, "cmat")?;
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
                    df: None,
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
        cmat: if kept.is_empty() {
            Vec::new()
        } else {
            input.cmat.to_vec()
        },
        summary: None,
    })
}

/// What [`yates_simulate`] predicts for each population row: the prediction
/// functions of R's `yates_setup.coxph`.
#[derive(Debug, Clone, Copy)]
pub enum YatesPredictor<'a> {
    /// `predict = "risk"`: `exp(eta)`.
    Risk,
    /// `predict = "survival"`: the curve `exp(-exp(eta) * cumhaz)` of the
    /// baseline `survfit(fit, censor = FALSE)` (`time`, `cumhaz`) from time 0
    /// on, preceded by its mean restricted to `rmean`.  `conf_int` is the
    /// baseline's confidence level, which the summary curves use.
    Survival {
        time: &'a [f64],
        cumhaz: &'a [f64],
        rmean: f64,
        conf_int: f64,
    },
}

/// Inputs of [`yates_simulate`].
#[derive(Debug, Clone)]
pub struct YatesSimulation<'a> {
    /// One model matrix per level over the non-aliased coefficient columns
    /// (R's `xmatlist` without the intercept and `NA` columns).
    pub xmatlist: &'a [Vec<Vec<f64>>],
    pub beta: &'a [f64],
    pub vmat: &'a [Vec<f64>],
    /// `fit$means` of the coefficients: every linear predictor is
    /// `x %*% b - sum(means * b)`.
    pub means: &'a [f64],
    /// Levels whose estimate is estimable (see [`yates_estimable`]); `None` for all.
    pub estimable: Option<&'a [bool]>,
    pub predictor: YatesPredictor<'a>,
    pub nsim: usize,
    /// Seed of R's generator (`set.seed`), so the draws are R's `rnorm`.
    pub seed: u32,
    pub test: YatesTest,
}

/// A [`YatesPredictor`] ready to evaluate: the risk, or the restricted mean
/// followed by the survival curve from time 0 on.
enum Prediction {
    Risk,
    /// `cumhaz = c(0, baseline$cumhaz)`; `widths = c(diff(c(0, pmin(rmean,
    /// baseline$time))), 0)` weight the curve into the restricted mean.
    Survival {
        cumhaz: Vec<f64>,
        widths: Vec<f64>,
    },
}

impl Prediction {
    fn new(predictor: YatesPredictor<'_>) -> SurvivalResult<Self> {
        let YatesPredictor::Survival {
            time,
            cumhaz,
            rmean,
            conf_int,
        } = predictor
        else {
            return Ok(Self::Risk);
        };
        if time.is_empty() {
            return Err(SurvivalError::invalid_input(
                "the baseline curve has no times",
            ));
        }
        validate_length(time.len(), cumhaz.len(), "cumhaz")?;
        validate_finite(time, "time")?;
        validate_finite(cumhaz, "cumhaz")?;
        if rmean.is_nan() {
            return Err(SurvivalError::invalid_input("rmean must not be NaN"));
        }
        if !(conf_int > 0.0 && conf_int < 1.0) {
            return Err(SurvivalError::invalid_input(
                "conf_int must be between 0 and 1",
            ));
        }
        let mut widths = Vec::with_capacity(time.len() + 1);
        let mut previous = 0.0;
        for &t in time {
            let t = t.min(rmean);
            widths.push(t - previous);
            previous = t;
        }
        widths.push(0.0);
        Ok(Self::Survival {
            cumhaz: std::iter::once(0.0).chain(cumhaz.iter().copied()).collect(),
            widths,
        })
    }

    fn width(&self) -> usize {
        match self {
            Self::Risk => 1,
            Self::Survival { cumhaz, .. } => cumhaz.len() + 1,
        }
    }

    /// Adds the prediction at linear predictor `eta` to `out`.
    fn accumulate(&self, eta: f64, out: &mut [f64]) {
        let risk = eta.exp();
        match self {
            Self::Risk => out[0] += risk,
            Self::Survival { cumhaz, widths } => {
                let mut mean = 0.0;
                for ((value, hazard), width) in out[1..].iter_mut().zip(cumhaz).zip(widths) {
                    let surv = (-risk * hazard).exp();
                    *value += surv;
                    mean += width * surv;
                }
                out[0] += mean;
            }
        }
    }

    /// Each level's prediction averaged over its population rows (R's
    /// `rowsum(predfun(eta), index) / n1`) at coefficients `coef`.
    fn population(&self, xmatlist: &[Vec<Vec<f64>>], means: &[f64], coef: &[f64]) -> Vec<Vec<f64>> {
        let center = dot(means, coef);
        xmatlist
            .iter()
            .map(|rows| {
                let mut out = vec![0.0; self.width()];
                for row in rows {
                    self.accumulate(dot(row, coef) - center, &mut out);
                }
                let n = rows.len() as f64;
                for value in &mut out {
                    *value /= n;
                }
                out
            })
            .collect()
    }
}

fn validate_simulation(input: &YatesSimulation<'_>) -> SurvivalResult<()> {
    if input.nsim < 2 {
        return Err(SurvivalError::invalid_input("nsim must be at least two"));
    }
    if input.xmatlist.is_empty() {
        return Err(SurvivalError::invalid_input(
            "xmatlist must have a matrix per level",
        ));
    }
    let p = input.beta.len();
    validate_coefficients(input.beta, input.vmat)?;
    validate_length(p, input.means.len(), "means")?;
    validate_finite(input.means, "means")?;
    for (level, rows) in input.xmatlist.iter().enumerate() {
        if rows.is_empty() {
            return Err(SurvivalError::invalid_input(format!(
                "population matrix {level} has no rows"
            )));
        }
        for row in rows {
            validate_length(p, row.len(), "population matrix columns")?;
            validate_finite(row, "population matrix")?;
        }
    }
    if let Some(estimable) = input.estimable {
        validate_length(input.xmatlist.len(), estimable.len(), "estimable")?;
    }
    Ok(())
}

/// R's `Rmat`: the symmetric square root `V diag(sqrt(d)) V'` of the
/// coefficient variance, which turns standard normal draws into draws with
/// that variance.
fn covariance_root(vmat: &[Vec<f64>]) -> SurvivalResult<Vec<Vec<f64>>> {
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
    let p = vmat.len();
    Ok((0..p)
        .map(|i| {
            (0..p)
                .map(|j| {
                    (0..p)
                        .map(|k| vectors[i][k] * vectors[j][k] * values[k].max(0.0).sqrt())
                        .sum()
                })
                .collect()
        })
        .collect())
}

/// Population marginal means of a nonlinear prediction (R's `yates` with a
/// `yates_setup` prediction function): the prediction at `beta` averaged
/// over each level's population rows, the variance of the first prediction
/// (the risk, or the restricted mean survival) estimated from `nsim`
/// coefficient vectors drawn from `N(beta, vmat)`, and its tests.
///
/// As in R, a non-estimable level's `pmm` is `NaN` while its `std` and
/// `mvar` entries keep their simulated values; a test that uses it is `NA`.
pub fn yates_simulate(input: &YatesSimulation<'_>) -> SurvivalResult<YatesResult> {
    validate_simulation(input)?;
    let prediction = Prediction::new(input.predictor)?;
    let root = covariance_root(input.vmat)?;
    let nlev = input.xmatlist.len();
    let p = input.beta.len();
    let width = prediction.width();
    let estimates = prediction.population(input.xmatlist, input.means, input.beta);

    // R's matrix(rnorm(nsim * p), nrow = nsim) fills the draws by column
    let mut rng = rng::RNormal::new(input.seed);
    let mut z = vec![vec![0.0; p]; input.nsim];
    for j in 0..p {
        for row in &mut z {
            row[j] = rng.normal();
        }
    }
    // running means and sums of squared deviations of every prediction, and
    // the co-moments of the first prediction across levels
    let mut mean = vec![vec![0.0; width]; nlev];
    let mut squares = vec![vec![0.0; width]; nlev];
    let mut comoment = vec![vec![0.0; nlev]; nlev];
    for (draw, z) in z.iter().enumerate() {
        let coef: Vec<f64> = (0..p)
            .map(|j| input.beta[j] + (0..p).map(|k| z[k] * root[k][j]).sum::<f64>())
            .collect();
        let sims = prediction.population(input.xmatlist, input.means, &coef);
        let first: Vec<f64> = sims.iter().map(|row| row[0]).collect();
        validate_finite(&first, "simulated prediction")?;
        let delta: Vec<f64> = first
            .iter()
            .zip(&mean)
            .map(|(value, running)| value - running[0])
            .collect();
        let count = (draw + 1) as f64;
        for ((running, square), sim) in mean.iter_mut().zip(&mut squares).zip(&sims) {
            for ((m, s), &value) in running.iter_mut().zip(square.iter_mut()).zip(sim) {
                let change = value - *m;
                *m += change / count;
                *s += change * (value - *m);
            }
        }
        for (row, d) in comoment.iter_mut().zip(&delta) {
            for (value, (sim, running)) in row.iter_mut().zip(first.iter().zip(&mean)) {
                *value += d * (sim - running[0]);
            }
        }
    }
    let denominator = (input.nsim - 1) as f64;
    let mvar: Vec<Vec<f64>> = comoment
        .into_iter()
        .map(|row| row.into_iter().map(|value| value / denominator).collect())
        .collect();

    let first: Vec<f64> = estimates.iter().map(|row| row[0]).collect();
    let identity: Vec<Vec<f64>> = (0..nlev)
        .map(|i| (0..nlev).map(|j| f64::from(i == j)).collect())
        .collect();
    let mut result = yates(&YatesInput {
        cmat: &identity,
        beta: &first,
        vmat: &mvar,
        offset: 0.0,
        sigma2: None,
        estimable: input.estimable,
        test: input.test,
    })?;
    for (i, (estimate, row)) in result.estimate.iter_mut().zip(&mvar).enumerate() {
        estimate.std = row[i].sqrt();
    }
    result.cmat.clear();
    if input.estimable.is_none_or(|flags| flags.contains(&true)) {
        result.mvar = mvar;
        if let YatesPredictor::Survival { conf_int, .. } = input.predictor {
            let variance: Vec<Vec<f64>> = squares
                .into_iter()
                .map(|row| row.into_iter().map(|value| value / denominator).collect())
                .collect();
            result.summary = Some(survival_curves(&mean, &variance, conf_int));
        }
    }
    Ok(result)
}

/// R's `summary` function of `yates_setup.coxph`: from the simulation means
/// and variances of each level's curve (the columns after the restricted
/// mean and the time-0 value), `surv`, `chaz = -log(surv)`,
/// `std.err = std / surv` and the limits `exp(-(chaz -/+ z * std))`.
fn survival_curves(mean: &[Vec<f64>], variance: &[Vec<f64>], conf_int: f64) -> YatesCurves {
    let z = -qnorm((1.0 - conf_int) / 2.0, true, false);
    let ntime = mean[0].len() - 2;
    let per_time = |f: &dyn Fn(f64, f64) -> f64| -> Vec<Vec<f64>> {
        (0..ntime)
            .map(|t| {
                mean.iter()
                    .zip(variance)
                    .map(|(m, v)| f(m[t + 2], v[t + 2].sqrt()))
                    .collect()
            })
            .collect()
    };
    YatesCurves {
        surv: per_time(&|surv, _| surv),
        cumhaz: per_time(&|surv, _| -surv.ln()),
        std_err: per_time(&|surv, std| std / surv),
        lower: per_time(&|surv, std| (-(-surv.ln() + z * std)).exp()),
        upper: per_time(&|surv, std| (-(-surv.ln() - z * std)).exp()),
    }
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

/// Runs [`yates_simulate`] without the GIL; R names the tests of a
/// simulated prediction after the term instead of `global`.
fn simulate_py(
    py: Python<'_>,
    input: &YatesSimulation<'_>,
    term: Option<&str>,
) -> PyResult<YatesResult> {
    let mut result = py.detach(|| yates_simulate(input))?;
    if let Some(term) = term {
        for row in &mut result.test {
            if row.name == "global" {
                row.name = term.to_owned();
            }
        }
    }
    Ok(result)
}

/// Python entry point for `predict = "risk"`: see [`yates_simulate`].
#[pyfunction(name = "yates_risk")]
#[pyo3(signature = (xmatlist, beta, vmat, means, estimable=None, nsim=200, seed=0, test="global", term=None))]
#[allow(clippy::too_many_arguments)]
pub fn yates_risk_py(
    py: Python<'_>,
    xmatlist: Vec<Vec<Vec<f64>>>,
    beta: Vec<f64>,
    vmat: Vec<Vec<f64>>,
    means: Vec<f64>,
    estimable: Option<Vec<bool>>,
    nsim: usize,
    seed: u32,
    test: &str,
    term: Option<&str>,
) -> PyResult<YatesResult> {
    let input = YatesSimulation {
        xmatlist: &xmatlist,
        beta: &beta,
        vmat: &vmat,
        means: &means,
        estimable: estimable.as_deref(),
        predictor: YatesPredictor::Risk,
        nsim,
        seed,
        test: YatesTest::parse(test)?,
    };
    simulate_py(py, &input, term)
}

/// Python entry point for `predict = "survival"`: see [`yates_simulate`];
/// `time` and `cumhaz` are the baseline `survfit(fit, censor = FALSE)`.
#[pyfunction(name = "yates_survival")]
#[pyo3(signature = (xmatlist, beta, vmat, means, time, cumhaz, rmean, conf_int=0.95, estimable=None, nsim=200, seed=0, test="global", term=None))]
#[allow(clippy::too_many_arguments)]
pub fn yates_survival_py(
    py: Python<'_>,
    xmatlist: Vec<Vec<Vec<f64>>>,
    beta: Vec<f64>,
    vmat: Vec<Vec<f64>>,
    means: Vec<f64>,
    time: Vec<f64>,
    cumhaz: Vec<f64>,
    rmean: f64,
    conf_int: f64,
    estimable: Option<Vec<bool>>,
    nsim: usize,
    seed: u32,
    test: &str,
    term: Option<&str>,
) -> PyResult<YatesResult> {
    let input = YatesSimulation {
        xmatlist: &xmatlist,
        beta: &beta,
        vmat: &vmat,
        means: &means,
        estimable: estimable.as_deref(),
        predictor: YatesPredictor::Survival {
            time: &time,
            cumhaz: &cumhaz,
            rmean,
            conf_int,
        },
        nsim,
        seed,
        test: YatesTest::parse(test)?,
    };
    simulate_py(py, &input, term)
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

/// Python entry point for [`yates_estimable`]: `x` is the fit's model matrix.
#[pyfunction(name = "yates_estimable")]
#[pyo3(signature = (xmatlist, x, intercept=false))]
pub fn yates_estimable_py(
    xmatlist: Vec<Vec<Vec<f64>>>,
    x: Vec<Vec<f64>>,
    intercept: bool,
) -> PyResult<Vec<bool>> {
    Ok(yates_estimable(&xmatlist, &x, intercept)?)
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
        assert_eq!(result.test[0].df, Some(1));
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
        let mean = YatesInput {
            cmat: &cmat,
            beta: &beta,
            vmat: &vmat,
            offset: 0.0,
            sigma2: None,
            estimable: Some(&[true, true, false]),
            test: YatesTest::Mean,
        };
        let partial = yates(&mean).unwrap();
        assert!(partial.estimate[2].pmm.is_nan());
        assert!(
            partial
                .test
                .iter()
                .all(|t| t.chisq.is_nan() && t.df.is_none())
        );
        assert_eq!(partial.mvar.len(), 2);
        // R keeps no mvar or Cmat when no level is estimable
        let none = yates(&YatesInput {
            estimable: Some(&[false, false, false]),
            ..mean
        })
        .unwrap();
        assert!(none.mvar.is_empty() && none.cmat.is_empty());
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

    #[test]
    fn qr_residual_matches_r() {
        // R: X <- cbind(1, 1:4, 2 * (1:4)); q <- qr(X)
        // q$rank = 2; qr.resid(q, c(1, 3, 2, 5)) = -0.1 0.8 -1.3 0.6
        let qr = LinpackQr::new(
            vec![
                vec![1.0, 1.0, 1.0, 1.0],
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2.0, 4.0, 6.0, 8.0],
            ],
            4,
        );
        assert_eq!(qr.rank, 2);
        let residual = qr.residual(&[1.0, 3.0, 2.0, 5.0]);
        for (value, expected) in residual.iter().zip([-0.1, 0.8, -1.3, 0.6]) {
            assert!((value - expected).abs() < 1e-12, "{residual:?}");
        }
        let inside = qr.residual(&[3.0, 5.0, 7.0, 9.0]);
        assert!(inside.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn estimable_levels_lie_in_the_row_space() {
        // design rows (a, b) with b = 2a: with the Cox intercept, only rows
        // (1, a, 2a) are estimable
        let x = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![1.0, 2.0]];
        let xmatlist = vec![
            vec![vec![1.0, 3.0, 6.0]],
            vec![vec![1.0, 3.0, 6.0], vec![1.0, 3.0, 5.0]],
        ];
        assert_eq!(
            yates_estimable(&xmatlist, &x, true).unwrap(),
            vec![true, false]
        );
        assert!(yates_estimable(&xmatlist, &x, false).is_err());
    }

    #[test]
    fn simulated_risk_keeps_non_estimable_spread() {
        let xmatlist = vec![
            vec![vec![0.0, 1.0], vec![0.0, 2.0]],
            vec![vec![1.0, 1.0], vec![1.0, 2.0]],
            vec![vec![2.0, 1.0], vec![2.0, 2.0]],
        ];
        let beta = [0.3, -0.2];
        let vmat = vec![vec![0.04, 0.0], vec![0.0, 0.01]];
        let means = [1.0, 1.5];
        let result = yates_simulate(&YatesSimulation {
            xmatlist: &xmatlist,
            beta: &beta,
            vmat: &vmat,
            means: &means,
            estimable: Some(&[true, true, false]),
            predictor: YatesPredictor::Risk,
            nsim: 50,
            seed: 7,
            test: YatesTest::Pairwise,
        })
        .unwrap();
        assert!(result.estimate[2].pmm.is_nan());
        assert!(result.estimate[2].std > 0.0);
        assert_eq!(result.mvar.len(), 3);
        assert!(result.test[0].df.is_some());
        assert!(result.test[1].df.is_none() && result.test[2].df.is_none());
        assert!(result.cmat.is_empty() && result.summary.is_none());
        // exp(eta) at beta averaged over the rows: eta = x b - sum(means * b) = -0.2, -0.4
        let point = ((-0.2f64).exp() + (-0.4f64).exp()) / 2.0;
        assert!((result.estimate[0].pmm - point).abs() < 1e-12);
    }

    #[test]
    fn simulated_survival_restricts_the_mean_and_aligns_the_curves() {
        let xmatlist = vec![vec![vec![0.0]], vec![vec![1.0]]];
        let beta = [0.5];
        let vmat = vec![vec![0.01]];
        let time = [1.0, 2.0, 4.0];
        let cumhaz = [0.1, 0.3, 0.6];
        let result = yates_simulate(&YatesSimulation {
            xmatlist: &xmatlist,
            beta: &beta,
            vmat: &vmat,
            means: &[0.0],
            estimable: None,
            predictor: YatesPredictor::Survival {
                time: &time,
                cumhaz: &cumhaz,
                rmean: 3.0,
                conf_int: 0.95,
            },
            nsim: 20,
            seed: 1,
            test: YatesTest::Global,
        })
        .unwrap();
        // widths 1, 1, 1, 0 over the curve 1, S(1), S(2), S(4)
        let mean = 1.0 + (-0.1f64).exp() + (-0.3f64).exp();
        assert!((result.estimate[0].pmm - mean).abs() < 1e-12);
        let summary = result.summary.unwrap();
        assert_eq!(summary.surv.len(), 3);
        assert_eq!(summary.surv[0].len(), 2);
        for (surv, cumhaz) in summary.surv.iter().zip(&summary.cumhaz) {
            assert!((surv[1] - (-cumhaz[1]).exp()).abs() < 1e-12);
        }
        assert!(summary.lower[2][1] < summary.surv[2][1]);
        assert!(summary.surv[2][1] < summary.upper[2][1]);
    }
}
