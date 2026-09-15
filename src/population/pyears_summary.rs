//! The data side of R's `summary.pyears` and `print.pyears`
//! (`R/summary.pyears.R`, `R/print.pyears.R`): totals, event rates,
//! observed/expected ratios and their Poisson confidence limits for the
//! tables produced by [`super::pyears::pyears`].  Layout and printing stay
//! with the caller.

use super::pyears::PyearsResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::statistical::gamma_inverse_cdf;
use pyo3::prelude::*;

/// The logical switches of `summary.pyears`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyearsSummaryOptions {
    /// Append marginal totals (`pytot`).
    pub totals: bool,
    /// `scale * event / pyears`.
    pub rate: bool,
    /// Poisson confidence limits of the rate.
    pub ci_r: bool,
    /// `event / expected`.
    pub rr: bool,
    /// Poisson confidence limits of the ratio.
    pub ci_rr: bool,
    pub conf_level: f64,
    pub scale: f64,
}

impl Default for PyearsSummaryOptions {
    fn default() -> Self {
        Self {
            totals: false,
            rate: false,
            ci_r: false,
            rr: true,
            ci_rr: false,
            conf_level: 0.95,
            scale: 1.0,
        }
    }
}

/// The tables of `summary.pyears`, each in column-major order over `dims`
/// (which include the "Total" margins when they were requested).  A
/// missing entry (R's `NA`) is `NaN`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct PyearsSummary {
    #[pyo3(get)]
    pub dims: Vec<usize>,
    #[pyo3(get)]
    pub n: Vec<f64>,
    #[pyo3(get)]
    pub event: Option<Vec<f64>>,
    #[pyo3(get)]
    pub pyears: Vec<f64>,
    #[pyo3(get)]
    pub expected: Option<Vec<f64>>,
    #[pyo3(get)]
    pub rate: Option<Vec<f64>>,
    #[pyo3(get)]
    pub ci_r_lower: Option<Vec<f64>>,
    #[pyo3(get)]
    pub ci_r_upper: Option<Vec<f64>>,
    #[pyo3(get)]
    pub rr: Option<Vec<f64>>,
    #[pyo3(get)]
    pub ci_rr_lower: Option<Vec<f64>>,
    #[pyo3(get)]
    pub ci_rr_upper: Option<Vec<f64>>,
    /// `print.pyears`: `sum(x$event)`, `NaN` without events.
    #[pyo3(get)]
    pub total_events: f64,
    /// `print.pyears`: `sum(x$pyears)`.
    #[pyo3(get)]
    pub total_pyears: f64,
    #[pyo3(get)]
    pub offtable: f64,
    #[pyo3(get)]
    pub observations: usize,
}

/// Column-major offset of a multi-index.
fn offset(index: &[usize], dims: &[usize]) -> usize {
    let mut result = 0;
    let mut stride = 1;
    for (&i, &n) in index.iter().zip(dims) {
        result += i * stride;
        stride *= n;
    }
    result
}

/// R's `pytot`: append a "Total" margin to the first two dimensions (or
/// the only one).  With `na` the totals are `NA`, which `summary.pyears`
/// uses for `n` when time-dependent cuts make it meaningless.
fn pytot(x: &[f64], dims: &[usize], na: bool) -> (Vec<f64>, Vec<usize>) {
    let total = |values: &[f64]| if na { f64::NAN } else { values.iter().sum() };
    if dims.len() == 1 {
        let mut out = x.to_vec();
        out.push(total(x));
        return (out, vec![dims[0] + 1]);
    }
    let mut new_dims = dims.to_vec();
    new_dims[0] += 1;
    new_dims[1] += 1;
    let n_cells: usize = new_dims.iter().product();
    let mut out = vec![0.0; n_cells];
    let outer: usize = dims[2..].iter().product();
    for slab in 0..outer {
        // Decompose the slab into the trailing indices (shared by x and out).
        let mut trailing = Vec::with_capacity(dims.len() - 2);
        let mut rest = slab;
        for &n in &dims[2..] {
            trailing.push(rest % n);
            rest /= n;
        }
        let at = |i: usize, j: usize, dims: &[usize]| {
            let mut index = vec![i, j];
            index.extend_from_slice(&trailing);
            offset(&index, dims)
        };
        let mut grand = 0.0;
        let mut col_sums = vec![0.0; dims[1]];
        for i in 0..dims[0] {
            let mut row_sum = 0.0;
            for j in 0..dims[1] {
                let value = x[at(i, j, dims)];
                out[at(i, j, &new_dims)] = value;
                row_sum += value;
                col_sums[j] += value;
                grand += value;
            }
            out[at(i, dims[1], &new_dims)] = if na { f64::NAN } else { row_sum };
        }
        for (j, col_sum) in col_sums.into_iter().enumerate() {
            out[at(dims[0], j, &new_dims)] = if na { f64::NAN } else { col_sum };
        }
        out[at(dims[0], dims[1], &new_dims)] = if na { f64::NAN } else { grand };
    }
    (out, new_dims)
}

/// R's `cipoisson(k, time, p, method = "exact")`: the gamma-quantile
/// limits of a Poisson count `k` observed over `time` (`R/cipoisson.R`).
fn cipoisson_exact(k: f64, time: f64, p: f64) -> (f64, f64) {
    let alpha = (1.0 - p) / 2.0;
    let lower = if k == 0.0 {
        0.0
    } else {
        gamma_inverse_cdf(alpha, k)
    };
    let upper = gamma_inverse_cdf(1.0 - alpha, k + 1.0);
    (lower / time, upper / time)
}

/// R's `summary.pyears` on a [`PyearsResult`]; `tcut` says whether any
/// term was a `tcut`, in which case totals of `n` are `NA`.
pub fn summary_pyears(
    result: &PyearsResult,
    tcut: bool,
    options: PyearsSummaryOptions,
) -> SurvivalResult<PyearsSummary> {
    if !(options.conf_level > 0.0 && options.conf_level < 1.0) {
        return Err(SurvivalError::invalid_input(
            "conf.level must be a single numeric between 0 and 1",
        ));
    }
    if options.scale.is_nan() || options.scale <= 0.0 || !options.scale.is_finite() {
        return Err(SurvivalError::invalid_input("scale must be a value > 0"));
    }
    let dims = if result.dims.is_empty() {
        vec![1]
    } else {
        result.dims.clone()
    };
    let n_cells: usize = dims.iter().product();
    for (name, len) in [
        ("pyears", result.pyears.len()),
        ("n", result.n.len()),
        ("event", result.event.as_ref().map_or(n_cells, Vec::len)),
        (
            "expected",
            result.expected.as_ref().map_or(n_cells, Vec::len),
        ),
    ] {
        if len != n_cells {
            return Err(SurvivalError::invalid_input(format!(
                "{name} must have prod(dims) = {n_cells} cells"
            )));
        }
    }
    let has_event = result.event.is_some();
    let has_expected = result.expected.is_some();
    let rate = options.rate && has_event;
    let ci_r = options.ci_r && has_event;
    let rr = options.rr && has_event && has_expected;
    let ci_rr = options.ci_rr && has_event && has_expected;

    let total_events = result.event.as_ref().map_or(f64::NAN, |e| e.iter().sum());
    let total_pyears = result.pyears.iter().sum();

    let (n, pyears, event, expected, out_dims) = if options.totals {
        let (n, out_dims) = pytot(&result.n, &dims, tcut);
        let (pyears, _) = pytot(&result.pyears, &dims, false);
        let event = result.event.as_ref().map(|e| pytot(e, &dims, false).0);
        let expected = result.expected.as_ref().map(|e| pytot(e, &dims, false).0);
        (n, pyears, event, expected, out_dims)
    } else {
        (
            result.n.clone(),
            result.pyears.clone(),
            result.event.clone(),
            result.expected.clone(),
            dims,
        )
    };

    let ratio = |num: &[f64], den: &[f64]| -> Vec<f64> {
        num.iter().zip(den).map(|(a, b)| a / b).collect()
    };
    let limits = |num: &[f64], den: &[f64], scale: f64| -> (Vec<f64>, Vec<f64>) {
        num.iter()
            .zip(den)
            .map(|(&k, &time)| {
                let (lower, upper) = cipoisson_exact(k, time, options.conf_level);
                (lower * scale, upper * scale)
            })
            .unzip()
    };
    let event_values = event.as_deref().unwrap_or_default();
    let expected_values = expected.as_deref().unwrap_or_default();
    let rate_values = rate.then(|| {
        ratio(event_values, &pyears)
            .into_iter()
            .map(|r| r * options.scale)
            .collect()
    });
    let (ci_r_lower, ci_r_upper) = ci_r
        .then(|| limits(event_values, &pyears, options.scale))
        .unzip();
    let rr_values = rr.then(|| ratio(event_values, expected_values));
    let (ci_rr_lower, ci_rr_upper) = ci_rr
        .then(|| limits(event_values, expected_values, 1.0))
        .unzip();

    Ok(PyearsSummary {
        dims: out_dims,
        n,
        event,
        pyears,
        expected,
        rate: rate_values,
        ci_r_lower,
        ci_r_upper,
        rr: rr_values,
        ci_rr_lower,
        ci_rr_upper,
        total_events,
        total_pyears,
        offtable: result.offtable,
        observations: result.observations,
    })
}

/// Python entry point of [`summary_pyears`].
#[pyfunction(name = "summary_pyears")]
#[pyo3(signature = (result, tcut=false, totals=false, rate=false, ci_r=false, rr=true, ci_rr=false, conf_level=0.95, scale=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn summary_pyears_py(
    result: &PyearsResult,
    tcut: bool,
    totals: bool,
    rate: bool,
    ci_r: bool,
    rr: bool,
    ci_rr: bool,
    conf_level: f64,
    scale: f64,
) -> PyResult<PyearsSummary> {
    Ok(summary_pyears(
        result,
        tcut,
        PyearsSummaryOptions {
            totals,
            rate,
            ci_r,
            rr,
            ci_rr,
            conf_level,
            scale,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(dims: Vec<usize>, event: bool, expected: bool) -> PyearsResult {
        let cells: usize = dims.iter().product();
        PyearsResult {
            pyears: (1..=cells).map(|c| c as f64 * 10.0).collect(),
            n: (1..=cells).map(|c| c as f64).collect(),
            event: event.then(|| (0..cells).map(|c| c as f64).collect()),
            expected: expected.then(|| (1..=cells).map(|c| c as f64 / 2.0).collect()),
            offtable: 0.5,
            dims,
            observations: 7,
        }
    }

    #[test]
    fn totals_append_margins_to_the_first_two_dimensions() {
        let (one, dims) = pytot(&[1.0, 2.0, 3.0], &[3], false);
        assert_eq!(one, vec![1.0, 2.0, 3.0, 6.0]);
        assert_eq!(dims, vec![4]);
        assert!(pytot(&[1.0, 2.0], &[2], true).0[2].is_nan());

        // 2 x 2 column-major: [[1, 3], [2, 4]] in R's display.
        let (two, dims) = pytot(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert_eq!(dims, vec![3, 3]);
        assert_eq!(two, vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 4.0, 6.0, 10.0]);

        // 2 x 2 x 2: each slab gets its own margins.
        let x: Vec<f64> = (1..=8).map(f64::from).collect();
        let (three, dims) = pytot(&x, &[2, 2, 2], false);
        assert_eq!(dims, vec![3, 3, 2]);
        assert_eq!(&three[..9], &[1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 4.0, 6.0, 10.0]);
        assert_eq!(
            &three[9..],
            &[5.0, 6.0, 11.0, 7.0, 8.0, 15.0, 12.0, 14.0, 26.0]
        );
    }

    #[test]
    fn rates_ratios_and_limits_follow_r() {
        let summary = summary_pyears(
            &result(vec![2], true, true),
            false,
            PyearsSummaryOptions {
                totals: true,
                rate: true,
                ci_r: true,
                rr: true,
                ci_rr: true,
                conf_level: 0.95,
                scale: 100.0,
            },
        )
        .unwrap();
        assert_eq!(summary.dims, vec![3]);
        assert_eq!(summary.n, vec![1.0, 2.0, 3.0]);
        assert_eq!(summary.pyears, vec![10.0, 20.0, 30.0]);
        assert_eq!(summary.event, Some(vec![0.0, 1.0, 1.0]));
        assert_eq!(summary.expected, Some(vec![0.5, 1.0, 1.5]));
        assert_eq!(summary.rate, Some(vec![0.0, 5.0, 100.0 / 30.0]));
        assert_eq!(summary.rr, Some(vec![0.0, 1.0, 1.0 / 1.5]));
        let ci_r_lower = summary.ci_r_lower.unwrap();
        let ci_r_upper = summary.ci_r_upper.unwrap();
        assert_eq!(ci_r_lower[0], 0.0);
        // cipoisson(1, 20) * 100: qgamma(0.025, 1) / 20 and qgamma(0.975, 2) / 20.
        assert!((ci_r_lower[1] - 0.1265890399).abs() < 1e-6);
        assert!((ci_r_upper[1] - 27.8582169547).abs() < 1e-6);
        let ci_rr_upper = summary.ci_rr_upper.unwrap();
        assert!((ci_rr_upper[0] - 7.377758908).abs() < 1e-6);
        assert_eq!(summary.total_events, 1.0);
        assert_eq!(summary.total_pyears, 30.0);
        assert_eq!(summary.offtable, 0.5);
        assert_eq!(summary.observations, 7);
    }

    #[test]
    fn missing_components_switch_their_statistics_off() {
        let summary = summary_pyears(
            &result(vec![2, 2], false, false),
            true,
            PyearsSummaryOptions {
                totals: true,
                rate: true,
                ci_r: true,
                rr: true,
                ci_rr: true,
                ..PyearsSummaryOptions::default()
            },
        )
        .unwrap();
        assert!(summary.rate.is_none() && summary.rr.is_none());
        assert!(summary.ci_r_lower.is_none() && summary.ci_rr_lower.is_none());
        assert!(summary.total_events.is_nan());
        assert_eq!(summary.dims, vec![3, 3]);
        assert!(summary.n[2].is_nan() && summary.n[8].is_nan());
        assert_eq!(summary.pyears[8], 100.0);
    }

    #[test]
    fn rejects_bad_options_and_shapes() {
        let bad_level = PyearsSummaryOptions {
            conf_level: 1.0,
            ..PyearsSummaryOptions::default()
        };
        assert!(summary_pyears(&result(vec![2], true, true), false, bad_level).is_err());
        let bad_scale = PyearsSummaryOptions {
            scale: 0.0,
            ..PyearsSummaryOptions::default()
        };
        assert!(summary_pyears(&result(vec![2], true, true), false, bad_scale).is_err());
        let mut short = result(vec![2], true, true);
        short.n.pop();
        assert!(summary_pyears(&short, false, PyearsSummaryOptions::default()).is_err());
    }
}
