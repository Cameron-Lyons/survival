//! The location-scale distributions of R's `survreg`.
//!
//! Ports `R/survreg.distributions.R` (the `survreg.distributions` list),
//! `R/survregDtest.R` and `R/dsurvreg.R` from the CRAN `survival` package.
//! A [`SurvregDistribution`] plays the role of one entry of
//! `survreg.distributions`: a base [`SurvregFamily`] (R's `init`,
//! `deviance`, `density` and `quantile` functions), an optional response
//! [`SurvregTransform`] (`trans`/`dtrans`/`itrans`), an optional fixed
//! `scale` and the distribution `parms`.  Distribution names are parsed once,
//! in [`SurvregDistribution::from_name`]; everything downstream works with the
//! struct.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::{dnorm, erf, pnorm, qnorm};
use crate::internal::rng::Rng;
use crate::internal::statistical::{erfc, student_t_cdf, student_t_inverse_cdf, student_t_pdf};
use crate::internal::validation::{validate_equal_len, validate_finite, validate_positive};
use pyo3::prelude::*;
use std::f64::consts::{PI, SQRT_2};

/// `sqrt(2 * pi)` as `#define SPI` in `survregc1.c`.
const SPI: f64 = 2.506628274631001;
/// `#define SMALL -200`: `exvalue_d` clamps `z` to `[SMALL, -SMALL]` before
/// exponentiating so a wild Newton step never produces an infinite answer.
const KERNEL_CLAMP: f64 = 200.0;

/// A distribution with its own `init`/`deviance`/`density`/`quantile`
/// definition in `survreg.distributions` (`extreme`, `logistic`, `gaussian`
/// and `t`); the remaining entries are transforms of one of these.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregFamily {
    ExtremeValue,
    Logistic,
    Gaussian,
    T,
}

/// The `trans`/`dtrans`/`itrans` triple of a derived distribution.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregTransform {
    /// No transform: the response is modelled directly.
    Identity,
    /// `trans = log`, `dtrans = 1/y`, `itrans = exp` (Weibull, log-normal,
    /// log-logistic, exponential, Rayleigh).
    Log,
}

impl SurvregTransform {
    /// `trans(y)`.
    pub fn apply(self, y: f64) -> f64 {
        match self {
            Self::Identity => y,
            Self::Log => y.ln(),
        }
    }

    /// `dtrans(y)`, the derivative of the transform.
    pub fn derivative(self, y: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Log => 1.0 / y,
        }
    }

    /// `itrans(x)`, the inverse transform.
    pub fn inverse(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Log => x.exp(),
        }
    }
}

/// Which block of `survregc1.c`'s `sreg_gg(z, ans, j)` to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelCase {
    /// `j = 1`: `ans = [_, f, f'/f, f''/f]`, used for exact observations.
    Density,
    /// `j = 2`: `ans = [F, 1 - F, f, f']`, used for censored observations.
    Distribution,
}

/// The five columns of `survreg.distributions$<family>$density(z, parms)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvregDensity {
    /// `F(z)`.
    pub cdf: f64,
    /// `1 - F(z)`, evaluated directly so the upper tail keeps its accuracy.
    pub survival: f64,
    /// `f(z)`.
    pub pdf: f64,
    /// `f'(z) / f(z)`.
    pub score: f64,
    /// `f''(z) / f(z)`.
    pub curvature: f64,
}

/// One entry of R's `survreg.distributions`.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregDistribution {
    /// R's `name` component ("Weibull", "Log Normal", ...).
    #[pyo3(get)]
    pub name: String,
    /// The base distribution whose `density`/`quantile` are used.
    #[pyo3(get)]
    pub family: SurvregFamily,
    /// The response transform (R's `trans`/`dtrans`/`itrans`).
    #[pyo3(get)]
    pub transform: SurvregTransform,
    /// A fixed scale (R's `scale` component: 1 for the exponential, 0.5 for
    /// the Rayleigh); `None` when the scale is estimated.
    #[pyo3(get)]
    pub scale: Option<f64>,
    /// Distribution parameters (R's `parms`): the degrees of freedom for
    /// the `t` family, empty otherwise.
    #[pyo3(get)]
    pub parms: Vec<f64>,
}

/// `names(survreg.distributions)`, in R's order, with each entry's
/// definition.
const BUILTIN_DISTRIBUTIONS: [(&str, &str, SurvregFamily, SurvregTransform, Option<f64>); 10] = [
    (
        "extreme",
        "Extreme value",
        SurvregFamily::ExtremeValue,
        SurvregTransform::Identity,
        None,
    ),
    (
        "logistic",
        "Logistic",
        SurvregFamily::Logistic,
        SurvregTransform::Identity,
        None,
    ),
    (
        "gaussian",
        "Gaussian",
        SurvregFamily::Gaussian,
        SurvregTransform::Identity,
        None,
    ),
    (
        "weibull",
        "Weibull",
        SurvregFamily::ExtremeValue,
        SurvregTransform::Log,
        None,
    ),
    (
        "exponential",
        "Exponential",
        SurvregFamily::ExtremeValue,
        SurvregTransform::Log,
        Some(1.0),
    ),
    (
        "rayleigh",
        "Rayleigh",
        SurvregFamily::ExtremeValue,
        SurvregTransform::Log,
        Some(0.5),
    ),
    (
        "loggaussian",
        "Log Normal",
        SurvregFamily::Gaussian,
        SurvregTransform::Log,
        None,
    ),
    (
        "lognormal",
        "Log Normal",
        SurvregFamily::Gaussian,
        SurvregTransform::Log,
        None,
    ),
    (
        "loglogistic",
        "Log logistic",
        SurvregFamily::Logistic,
        SurvregTransform::Log,
        None,
    ),
    (
        "t",
        "Student-t",
        SurvregFamily::T,
        SurvregTransform::Identity,
        None,
    ),
];

/// R's default `parms = c(df = 4)` for the `t` family.
const DEFAULT_T_DF: f64 = 4.0;

fn invalid(message: impl Into<String>) -> SurvivalError {
    SurvivalError::invalid_input(message)
}

/// `match.arg(dist, names(survreg.distributions))`: an exact match wins,
/// otherwise a unique prefix.  Names are case-folded (as `dsurvreg` does)
/// and `extreme_value`/`extreme-value` are accepted for `extreme` because
/// that is how [`SurvregFamily::ExtremeValue`] is spelled.
fn resolve_builtin(name: &str) -> SurvivalResult<usize> {
    let key = name.trim().to_lowercase();
    let key = match key.as_str() {
        "extreme_value" | "extreme-value" | "extreme value" => "extreme".to_string(),
        _ => key,
    };
    if key.is_empty() {
        return Err(invalid("distribution name must not be empty"));
    }
    if let Some(index) = BUILTIN_DISTRIBUTIONS
        .iter()
        .position(|(builtin, ..)| *builtin == key)
    {
        return Ok(index);
    }
    let matches: Vec<usize> = BUILTIN_DISTRIBUTIONS
        .iter()
        .enumerate()
        .filter(|(_, (builtin, ..))| builtin.starts_with(key.as_str()))
        .map(|(index, _)| index)
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        _ => Err(invalid(format!(
            "'{name}' should be one of {}",
            BUILTIN_DISTRIBUTIONS
                .iter()
                .map(|(builtin, ..)| format!("\"{builtin}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

impl SurvregDistribution {
    /// `survreg.distributions[[dist]]` for a character `dist`, with the
    /// `parms` handling of `survreg()`: parameters are only accepted by
    /// families that define them, and unspecified parameters take their
    /// default (`df = 4` for `t`).
    pub fn from_name(name: &str, parms: Option<&[f64]>) -> SurvivalResult<Self> {
        let (_, display, family, transform, scale) = BUILTIN_DISTRIBUTIONS[resolve_builtin(name)?];
        let parms = match (family, parms) {
            (SurvregFamily::T, Some(values)) => {
                if values.len() != 1 {
                    return Err(invalid(
                        "Student-t distribution takes a single parameter (the degrees of freedom)",
                    ));
                }
                values.to_vec()
            }
            (SurvregFamily::T, None) => vec![DEFAULT_T_DF],
            (_, Some(values)) if !values.is_empty() => {
                return Err(invalid(format!(
                    "{display} distribution has no optional parameters"
                )));
            }
            (_, _) => Vec::new(),
        };
        let distribution = Self {
            name: display.to_string(),
            family,
            transform,
            scale,
            parms,
        };
        distribution.validate()?;
        Ok(distribution)
    }

    /// A user-defined distribution in the shape of an entry of
    /// `survreg.distributions` that references a base distribution
    /// (`dist = "gaussian"`, `trans`/`dtrans`/`itrans`, optional `scale`).
    pub fn custom(
        name: impl Into<String>,
        family: SurvregFamily,
        transform: SurvregTransform,
        scale: Option<f64>,
        parms: Option<&[f64]>,
    ) -> Self {
        let parms = match parms {
            Some(values) => values.to_vec(),
            None if family == SurvregFamily::T => vec![DEFAULT_T_DF],
            None => Vec::new(),
        };
        Self {
            name: name.into(),
            family,
            transform,
            scale,
            parms,
        }
    }

    /// `survregDtest(dlist, verbose = TRUE)`: every problem with the
    /// definition, or an empty list when it is legal.  The structural checks
    /// of the R version (functions present, `trans`/`itrans` inverse of each
    /// other) hold by construction here; what remains are the checks
    /// `survreg` itself performs on the name, parameters and fixed scale.
    pub fn dtest(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.name.trim().is_empty() {
            issues.push("Missing a name".to_string());
        }
        match self.family {
            SurvregFamily::T => match self.parms.as_slice() {
                [df] if df.is_finite() && *df > 2.0 => {}
                [df] if df.is_finite() => {
                    issues.push("Degrees of freedom must be >=3".to_string());
                }
                _ => issues.push(
                    "Student-t distribution needs a single finite degrees-of-freedom parameter"
                        .to_string(),
                ),
            },
            _ if !self.parms.is_empty() => {
                issues.push(format!(
                    "{} distribution has no optional parameters",
                    self.name
                ));
            }
            _ => {}
        }
        if let Some(scale) = self.scale
            && !(scale.is_finite() && scale > 0.0)
        {
            issues.push("Invalid scale value".to_string());
        }
        issues
    }

    /// `if (!survregDtest(dlist)) stop("Invalid distribution object")`.
    pub fn validate(&self) -> SurvivalResult<()> {
        let issues = self.dtest();
        if issues.is_empty() {
            Ok(())
        } else {
            Err(invalid(format!(
                "Invalid distribution object: {}",
                issues.join("; ")
            )))
        }
    }

    /// The degrees of freedom of the `t` family (`parms["df"]`).
    fn df(&self) -> f64 {
        self.parms.first().copied().unwrap_or(DEFAULT_T_DF)
    }

    /// `variance(parms)`: the variance of the standardised base distribution.
    pub fn variance(&self) -> f64 {
        match self.family {
            SurvregFamily::ExtremeValue => PI * PI / 6.0,
            SurvregFamily::Logistic => PI * PI / 3.0,
            SurvregFamily::Gaussian => 1.0,
            SurvregFamily::T => {
                let df = self.df();
                df / (df - 2.0)
            }
        }
    }

    /// `init(x, weights, parms)`: `c(location, variance)` starting values
    /// from the weighted mean and variance of the (transformed) response.
    pub(crate) fn init(&self, y: &[f64], weights: &[f64]) -> SurvivalResult<[f64; 2]> {
        let total: f64 = weights.iter().sum();
        let mean = y.iter().zip(weights).map(|(y, w)| y * w).sum::<f64>() / total;
        let var = y
            .iter()
            .zip(weights)
            .map(|(y, w)| w * (y - mean).powi(2))
            .sum::<f64>()
            / total;
        Ok(match self.family {
            SurvregFamily::ExtremeValue => [mean + 0.572, var / 1.64],
            SurvregFamily::Logistic => [mean, var / 3.2],
            SurvregFamily::Gaussian => [mean, var],
            SurvregFamily::T => {
                let df = self.df();
                if df <= 2.0 {
                    return Err(invalid("Degrees of freedom must be >=3"));
                }
                [mean, var * (df - 2.0) / df]
            }
        })
    }

    /// `deviance(y, scale, parms)` for one observation on the transformed
    /// scale: the `center` of the saturated model and its log-likelihood.
    /// `y2` is only read for an interval-censored (`status == 3`) row.
    ///
    /// Two slips of the R code are corrected rather than copied: for the `t`
    /// family R's `center` is `rowMeans(y)`, which averages the status column
    /// in, and its `loglik` is `log(1 - 2*pt(width/2))`, the log of a
    /// negative number; the Gaussian formulas (`(y1 + y2)/2` and
    /// `log(2*pt(width/2) - 1)`) are what was meant.
    pub(crate) fn deviance(&self, y1: f64, y2: f64, status: i32, scale: f64) -> (f64, f64) {
        let interval = status == 3;
        match self.family {
            SurvregFamily::ExtremeValue => {
                let width = if interval { (y2 - y1) / scale } else { 1.0 };
                let temp = width / (width.exp() - 1.0);
                let center = if interval { y1 - temp.ln() } else { y1 };
                let temp3 = -temp + (1.0 - (-width.exp()).exp()).ln();
                let loglik = if status == 1 {
                    -(1.0 + scale.ln())
                } else if interval {
                    temp3
                } else {
                    0.0
                };
                (center, loglik)
            }
            SurvregFamily::Logistic => {
                let width = if interval { (y2 - y1) / scale } else { 0.0 };
                let center = if interval { (y1 + y2) / 2.0 } else { y1 };
                let temp2 = if interval { (width / 2.0).exp() } else { 2.0 };
                let temp3 = ((temp2 - 1.0) / (temp2 + 1.0)).ln();
                let loglik = if status == 1 {
                    -(4.0 * scale).ln()
                } else if interval {
                    temp3
                } else {
                    0.0
                };
                (center, loglik)
            }
            SurvregFamily::Gaussian => {
                let width = if interval { (y2 - y1) / scale } else { 0.0 };
                let center = if interval { (y1 + y2) / 2.0 } else { y1 };
                let temp2 = (2.0 * pnorm(width / 2.0, true, false) - 1.0).ln();
                let loglik = if status == 1 {
                    -(SPI * scale).ln()
                } else if interval {
                    temp2
                } else {
                    0.0
                };
                (center, loglik)
            }
            SurvregFamily::T => {
                let df = self.df();
                let width = if interval { (y2 - y1) / scale } else { 0.0 };
                let center = if interval { (y1 + y2) / 2.0 } else { y1 };
                let temp2 = (2.0 * student_t_cdf(width / 2.0, df) - 1.0).ln();
                let loglik = if status == 1 {
                    -(student_t_pdf(0.0, df) * scale).ln()
                } else if interval {
                    temp2
                } else {
                    0.0
                };
                (center, loglik)
            }
        }
    }

    /// `density(x, parms)`: the five-column summary
    /// `cbind(F, 1 - F, f, f'/f, f''/f)` of the base distribution.
    pub fn density(&self, z: f64) -> SurvregDensity {
        match self.family {
            SurvregFamily::ExtremeValue => {
                let w = z.exp();
                let ww = (-w).exp();
                SurvregDensity {
                    cdf: 1.0 - ww,
                    survival: ww,
                    pdf: w * ww,
                    score: 1.0 - w,
                    curvature: w * (w - 3.0) + 1.0,
                }
            }
            SurvregFamily::Logistic => {
                let w = z.exp();
                let denom = 1.0 + w;
                SurvregDensity {
                    cdf: w / denom,
                    survival: 1.0 / denom,
                    pdf: w / (denom * denom),
                    score: (1.0 - w) / denom,
                    curvature: (w * (w - 4.0) + 1.0) / (denom * denom),
                }
            }
            SurvregFamily::Gaussian => SurvregDensity {
                cdf: pnorm(z, true, false),
                survival: pnorm(-z, true, false),
                pdf: dnorm(z, false),
                score: -z,
                curvature: z * z - 1.0,
            },
            SurvregFamily::T => {
                let df = self.df();
                let denom = df + z * z;
                SurvregDensity {
                    cdf: student_t_cdf(z, df),
                    survival: student_t_cdf(-z, df),
                    pdf: student_t_pdf(z, df),
                    score: -(df + 1.0) * z / denom,
                    curvature: (df + 1.0) * (z * z * (df + 3.0) / denom - 1.0) / denom,
                }
            }
        }
    }

    /// `quantile(p, parms)` of the base distribution.
    pub fn quantile(&self, p: f64) -> f64 {
        match self.family {
            SurvregFamily::ExtremeValue => (-(1.0 - p).ln()).ln(),
            SurvregFamily::Logistic => (p / (1.0 - p)).ln(),
            SurvregFamily::Gaussian => qnorm(p, true, false),
            SurvregFamily::T => student_t_inverse_cdf(p, self.df()),
        }
    }

    /// The distribution evaluation used by the fitting kernel: `exvalue_d`,
    /// `logistic_d` and `gauss_d` of `survregc1.c` for the built-in
    /// families, and for the `t` family the R-level [`Self::density`] the
    /// way `survregc2.c` consumes it (`f' = f * f'/f`).
    ///
    /// Returns `[_, f, f'/f, f''/f]` for [`KernelCase::Density`] and
    /// `[F, 1 - F, f, f']` for [`KernelCase::Distribution`].
    pub(crate) fn kernel(&self, z: f64, case: KernelCase) -> [f64; 4] {
        match self.family {
            SurvregFamily::ExtremeValue => {
                let w = z.clamp(-KERNEL_CLAMP, KERNEL_CLAMP).exp();
                let temp = (-w).exp();
                match case {
                    KernelCase::Density => [0.0, w * temp, 1.0 - w, w * (w - 3.0) + 1.0],
                    KernelCase::Distribution => [1.0 - temp, temp, w * temp, w * temp * (1.0 - w)],
                }
            }
            SurvregFamily::Logistic => {
                // The symmetry of the logistic lets the C code never take
                // exp(large number).
                let (w, sign, positive) = if z > 0.0 {
                    ((-z).exp(), -1.0, true)
                } else {
                    (z.exp(), 1.0, false)
                };
                let temp = 1.0 + w;
                match case {
                    KernelCase::Density => [
                        0.0,
                        w / (temp * temp),
                        sign * (1.0 - w) / temp,
                        (w * w - 4.0 * w + 1.0) / (temp * temp),
                    ],
                    KernelCase::Distribution => {
                        let (cdf, survival) = if positive {
                            (1.0 / temp, w / temp)
                        } else {
                            (w / temp, 1.0 / temp)
                        };
                        let pdf = w / (temp * temp);
                        [cdf, survival, pdf, sign * pdf * (1.0 - w) / temp]
                    }
                }
            }
            SurvregFamily::Gaussian => {
                let f = (-z * z / 2.0).exp() / SPI;
                match case {
                    KernelCase::Density => [0.0, f, -z, z * z - 1.0],
                    KernelCase::Distribution => {
                        let (cdf, survival) = if z > 0.0 {
                            ((1.0 + erf(z / SQRT_2)) / 2.0, erfc(z / SQRT_2) / 2.0)
                        } else {
                            (erfc(-z / SQRT_2) / 2.0, (1.0 + erf(-z / SQRT_2)) / 2.0)
                        };
                        [cdf, survival, f, -z * f]
                    }
                }
            }
            SurvregFamily::T => {
                let d = self.density(z);
                match case {
                    KernelCase::Density => [0.0, d.pdf, d.score, d.curvature],
                    KernelCase::Distribution => [d.cdf, d.survival, d.pdf, d.pdf * d.score],
                }
            }
        }
    }

    /// `dsurvreg(x, mean, scale, distribution, parms)` for one value.
    pub fn pdf(&self, x: f64, mean: f64, scale: f64) -> f64 {
        let dx = self.transform.derivative(x);
        let z = (self.transform.apply(x) - mean) / scale;
        self.density(z).pdf * dx / scale
    }

    /// `psurvreg(q, mean, scale, distribution, parms)` for one value.
    pub fn cdf(&self, q: f64, mean: f64, scale: f64) -> f64 {
        let z = (self.transform.apply(q) - mean) / scale;
        self.density(z).cdf
    }

    /// `qsurvreg(p, mean, scale, distribution, parms)` for one value.
    pub fn quantile_at(&self, p: f64, mean: f64, scale: f64) -> f64 {
        self.transform.inverse(self.quantile(p) * scale + mean)
    }
}

#[pymethods]
impl SurvregDistribution {
    /// `survreg.distributions[[name]]` with optional `parms` (see
    /// [`SurvregDistribution::from_name`]).
    #[new]
    #[pyo3(signature = (name, parms=None))]
    fn new(name: &str, parms: Option<Vec<f64>>) -> PyResult<Self> {
        Ok(Self::from_name(name, parms.as_deref())?)
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
    fn variance_py(&self) -> f64 {
        self.variance()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvregDistribution(name='{}', family={:?}, transform={:?}, scale={:?}, parms={:?})",
            self.name, self.family, self.transform, self.scale, self.parms
        )
    }
}

/// `survregDtest(dlist, verbose = TRUE)` as a function: the problems with
/// `distribution`, empty when it is a legal definition.
#[pyfunction]
pub fn survreg_dtest(distribution: &SurvregDistribution) -> Vec<String> {
    distribution.dtest()
}

/// Shared argument checking of `dsurvreg`/`psurvreg`/`qsurvreg`: `mean` and
/// `scale` must be finite, positive scales, and either match the length of
/// the values or be a single number that recycles.
fn recycled<'a>(
    values: &'a [f64],
    name: &str,
    n: usize,
) -> SurvivalResult<impl Fn(usize) -> f64 + 'a> {
    if values.len() != 1 {
        validate_equal_len(&[("x", n), (name, values.len())])?;
    }
    validate_finite(values, name)?;
    let single = values.len() == 1;
    Ok(move |index: usize| if single { values[0] } else { values[index] })
}

fn distribution_values(
    values: &[f64],
    mean: &[f64],
    scale: &[f64],
    distribution: &SurvregDistribution,
    f: impl Fn(&SurvregDistribution, f64, f64, f64) -> f64,
) -> SurvivalResult<Vec<f64>> {
    let mean = recycled(mean, "mean", values.len())?;
    validate_positive(scale, "scale")?;
    let scale = recycled(scale, "scale", values.len())?;
    Ok(values
        .iter()
        .enumerate()
        .map(|(index, &value)| f(distribution, value, mean(index), scale(index)))
        .collect())
}

/// `dsurvreg(x, mean, scale, distribution, parms)`: the density of the
/// distribution on the original response scale.
#[pyfunction]
#[pyo3(signature = (x, mean, scale, distribution="weibull", parms=None))]
pub fn dsurvreg(
    x: Vec<f64>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    distribution: &str,
    parms: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let distribution = SurvregDistribution::from_name(distribution, parms.as_deref())?;
    Ok(distribution_values(
        &x,
        &mean,
        &scale,
        &distribution,
        |d, x, m, s| d.pdf(x, m, s),
    )?)
}

/// `psurvreg(q, mean, scale, distribution, parms)`: the distribution function.
#[pyfunction]
#[pyo3(signature = (q, mean, scale, distribution="weibull", parms=None))]
pub fn psurvreg(
    q: Vec<f64>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    distribution: &str,
    parms: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let distribution = SurvregDistribution::from_name(distribution, parms.as_deref())?;
    Ok(distribution_values(
        &q,
        &mean,
        &scale,
        &distribution,
        |d, q, m, s| d.cdf(q, m, s),
    )?)
}

/// `qsurvreg(p, mean, scale, distribution, parms)`: the quantile function.
#[pyfunction]
#[pyo3(signature = (p, mean, scale, distribution="weibull", parms=None))]
pub fn qsurvreg(
    p: Vec<f64>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    distribution: &str,
    parms: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let distribution = SurvregDistribution::from_name(distribution, parms.as_deref())?;
    Ok(distribution_values(
        &p,
        &mean,
        &scale,
        &distribution,
        |d, p, m, s| d.quantile_at(p, m, s),
    )?)
}

/// `rsurvreg(n, mean, scale, distribution, parms)`: `qsurvreg(runif(n), ...)`
/// drawn from the crate's generator (`seed` makes the draw reproducible; the
/// stream is not R's).
#[pyfunction]
#[pyo3(signature = (n, mean, scale, distribution="weibull", parms=None, seed=None))]
pub fn rsurvreg(
    n: usize,
    mean: Vec<f64>,
    scale: Vec<f64>,
    distribution: &str,
    parms: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<Vec<f64>> {
    let distribution = SurvregDistribution::from_name(distribution, parms.as_deref())?;
    let mut rng = seed.map_or_else(Rng::new, Rng::with_seed);
    let uniform: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
    Ok(distribution_values(
        &uniform,
        &mean,
        &scale,
        &distribution,
        |d, p, m, s| d.quantile_at(p, m, s),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn names_resolve_like_match_arg() {
        assert_eq!(
            SurvregDistribution::from_name("weibull", None)
                .unwrap()
                .name,
            "Weibull"
        );
        assert_eq!(
            SurvregDistribution::from_name("exp", None).unwrap().name,
            "Exponential"
        );
        assert_eq!(
            SurvregDistribution::from_name("LogNormal", None)
                .unwrap()
                .family,
            SurvregFamily::Gaussian
        );
        assert_eq!(
            SurvregDistribution::from_name("extreme_value", None)
                .unwrap()
                .transform,
            SurvregTransform::Identity
        );
        assert!(SurvregDistribution::from_name("log", None).is_err());
        assert!(SurvregDistribution::from_name("mystery", None).is_err());
        assert!(SurvregDistribution::from_name("", None).is_err());
    }

    #[test]
    fn parms_are_only_accepted_by_the_t_family() {
        let t = SurvregDistribution::from_name("t", None).unwrap();
        assert_eq!(t.parms, vec![4.0]);
        let t8 = SurvregDistribution::from_name("t", Some(&[8.0])).unwrap();
        assert_eq!(t8.parms, vec![8.0]);
        assert!(SurvregDistribution::from_name("t", Some(&[2.0])).is_err());
        assert!(SurvregDistribution::from_name("weibull", Some(&[3.0])).is_err());
        assert!(SurvregDistribution::from_name("weibull", Some(&[])).is_ok());
    }

    #[test]
    fn fixed_scales_follow_the_r_table() {
        assert_eq!(
            SurvregDistribution::from_name("exponential", None)
                .unwrap()
                .scale,
            Some(1.0)
        );
        assert_eq!(
            SurvregDistribution::from_name("rayleigh", None)
                .unwrap()
                .scale,
            Some(0.5)
        );
        assert_eq!(
            SurvregDistribution::from_name("weibull", None)
                .unwrap()
                .scale,
            None
        );
    }

    #[test]
    fn dtest_reports_problems() {
        let bad = SurvregDistribution::custom(
            "",
            SurvregFamily::T,
            SurvregTransform::Log,
            Some(-1.0),
            Some(&[1.0]),
        );
        let issues = bad.dtest();
        assert_eq!(issues.len(), 3, "{issues:?}");
        assert!(bad.validate().is_err());
        let good = SurvregDistribution::custom(
            "log-t",
            SurvregFamily::T,
            SurvregTransform::Log,
            None,
            None,
        );
        assert!(good.dtest().is_empty());
        assert_eq!(good.parms, vec![4.0]);
    }

    #[test]
    fn density_columns_match_the_r_definitions() {
        let gaussian = SurvregDistribution::from_name("gaussian", None).unwrap();
        let d = gaussian.density(0.3);
        assert_close(d.cdf, pnorm(0.3, true, false), 1e-15);
        assert_close(d.survival, pnorm(-0.3, true, false), 1e-15);
        assert_close(d.pdf, dnorm(0.3, false), 1e-15);
        assert_close(d.score, -0.3, 1e-15);
        assert_close(d.curvature, 0.09 - 1.0, 1e-15);

        let logistic = SurvregDistribution::from_name("logistic", None).unwrap();
        let d = logistic.density(-1.2);
        let w = (-1.2f64).exp();
        assert_close(d.cdf, w / (1.0 + w), 1e-15);
        assert_close(d.pdf, w / (1.0 + w).powi(2), 1e-15);

        let extreme = SurvregDistribution::from_name("extreme", None).unwrap();
        let d = extreme.density(0.4);
        let w = 0.4f64.exp();
        assert_close(d.survival, (-w).exp(), 1e-15);
        assert_close(d.score, 1.0 - w, 1e-15);
    }

    #[test]
    fn kernel_agrees_with_density_for_every_family() {
        for name in ["extreme", "logistic", "gaussian", "t"] {
            let dist = SurvregDistribution::from_name(name, None).unwrap();
            for z in [-2.5, -0.3, 0.0, 0.7, 3.1] {
                let d = dist.density(z);
                let density = dist.kernel(z, KernelCase::Density);
                let distribution = dist.kernel(z, KernelCase::Distribution);
                assert_close(density[1], d.pdf, 1e-13);
                assert_close(density[2], d.score, 1e-13);
                assert_close(density[3], d.curvature, 1e-13);
                assert_close(distribution[0], d.cdf, 1e-13);
                assert_close(distribution[1], d.survival, 1e-13);
                assert_close(distribution[2], d.pdf, 1e-13);
                assert_close(distribution[3], d.pdf * d.score, 1e-13);
            }
        }
    }

    #[test]
    fn kernel_extreme_value_is_clamped() {
        let dist = SurvregDistribution::from_name("extreme", None).unwrap();
        let far = dist.kernel(1e6, KernelCase::Distribution);
        assert!(far.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dpq_match_r_reference_values() {
        // dsurvreg(c(1, 2), 0.5, 1.2), psurvreg(...), qsurvreg(c(.25, .5, .75), 0.5, 1.2)
        let d = dsurvreg(vec![1.0, 2.0], vec![0.5], vec![1.2], "weibull", None).unwrap();
        assert_close(d[0], 0.2841569, 1e-7);
        assert_close(d[1], 0.1512009, 1e-7);
        let p = psurvreg(vec![1.0, 2.0], vec![0.5], vec![1.2], "weibull", None).unwrap();
        assert_close(p[0], 0.4827560, 1e-7);
        assert_close(p[1], 0.6910677, 1e-7);
        let q = qsurvreg(vec![0.25, 0.5, 0.75], vec![0.5], vec![1.2], "weibull", None).unwrap();
        assert_close(q[0], 0.3696942, 1e-7);
        assert_close(q[1], 1.0620325, 1e-7);
        assert_close(q[2], 2.4399099, 1e-7);

        // Student t with parms = 5, mean 0, scale 1.
        let d = dsurvreg(vec![1.0, 2.0], vec![0.0], vec![1.0], "t", Some(vec![5.0])).unwrap();
        assert_close(d[0], 0.2196798, 1e-7);
        assert_close(d[1], 0.06509031, 1e-7);
        let p = psurvreg(vec![1.0, 2.0], vec![0.0], vec![1.0], "t", Some(vec![5.0])).unwrap();
        assert_close(p[0], 0.8183913, 1e-7);
        assert_close(p[1], 0.9490303, 1e-7);
        let q = qsurvreg(
            vec![0.0, 0.25, 0.5, 1.0],
            vec![0.0],
            vec![1.0],
            "t",
            Some(vec![5.0]),
        )
        .unwrap();
        assert_eq!(q[0], f64::NEG_INFINITY);
        assert_close(q[1], -0.7266868, 1e-7);
        assert_eq!(q[2], 0.0);
        assert_eq!(q[3], f64::INFINITY);
    }

    #[test]
    fn dpq_reject_bad_arguments() {
        assert!(dsurvreg(vec![1.0], vec![0.0, 1.0], vec![1.0], "weibull", None).is_err());
        assert!(dsurvreg(vec![1.0], vec![0.0], vec![0.0], "weibull", None).is_err());
        assert!(qsurvreg(vec![0.5], vec![0.0], vec![1.0], "t", None).is_ok());
        assert!(qsurvreg(vec![0.5], vec![0.0], vec![1.0], "t", Some(vec![1.0])).is_err());
    }

    #[test]
    fn rsurvreg_is_reproducible_with_a_seed() {
        let a = rsurvreg(5, vec![1.0], vec![0.5], "weibull", None, Some(7)).unwrap();
        let b = rsurvreg(5, vec![1.0], vec![0.5], "weibull", None, Some(7)).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn deviance_saturated_loglik_matches_r() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let (center, loglik) = weibull.deviance(0.0, 2.0f64.ln(), 3, 1.0);
        // width = log 2, temp = width/(2 - 1), center = -log(temp)
        let width = 2.0f64.ln();
        assert_close(center, -(width / (width.exp() - 1.0)).ln(), 1e-14);
        assert_close(
            loglik,
            -width / (width.exp() - 1.0) + (1.0 - (-width.exp()).exp()).ln(),
            1e-14,
        );
        let (center, loglik) = weibull.deviance(1.5, 0.0, 1, 0.7);
        assert_eq!(center, 1.5);
        assert_close(loglik, -(1.0 + 0.7f64.ln()), 1e-14);
        assert_eq!(weibull.deviance(1.5, 0.0, 0, 0.7).1, 0.0);

        let gaussian = SurvregDistribution::from_name("gaussian", None).unwrap();
        let (center, loglik) = gaussian.deviance(1.0, 3.0, 3, 2.0);
        assert_eq!(center, 2.0);
        assert_close(loglik, (2.0 * pnorm(0.5, true, false) - 1.0).ln(), 1e-14);
    }

    #[test]
    fn init_uses_weighted_moments() {
        let extreme = SurvregDistribution::from_name("extreme", None).unwrap();
        let [mean, var] = extreme.init(&[1.0, 3.0], &[1.0, 3.0]).unwrap();
        assert_close(mean, 2.5 + 0.572, 1e-14);
        assert_close(var, 0.75 / 1.64, 1e-14);
        let t = SurvregDistribution::custom(
            "t2",
            SurvregFamily::T,
            SurvregTransform::Identity,
            None,
            Some(&[2.0]),
        );
        assert!(t.init(&[1.0, 3.0], &[1.0, 1.0]).is_err());
    }
}
