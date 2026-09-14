//! R's `tcut` (`R/tcut.R`): a time-dependent cut for `pyears`.  Unlike
//! `cut`, the values are kept as they are; the breakpoints and labels ride
//! along as attributes and `pyears` splits each subject's follow-up across
//! the intervals as time advances.

use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

/// A `tcut` object.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct TcutResult {
    /// The (scaled) values, unchanged otherwise.
    #[pyo3(get)]
    pub values: Vec<f64>,
    /// The (scaled) breakpoints, one more than the number of intervals.
    #[pyo3(get)]
    pub cutpoints: Vec<f64>,
    /// One label per interval (`levels()` of the object).
    #[pyo3(get)]
    pub labels: Vec<String>,
}

/// R's `seq(from, to, length.out = n)`: interior points are
/// `from + k * (to - from) / (n - 1)` and the last is exactly `to`.
fn seq_length(from: f64, to: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![from],
        2 => vec![from, to],
        _ => {
            let n1 = (n - 1) as f64;
            let mut out = Vec::with_capacity(n);
            out.push(from);
            for k in 1..n - 1 {
                out.push(from + k as f64 * ((to - from) / n1));
            }
            out.push(to);
            out
        }
    }
}

/// Build a `tcut`.  A single `breaks` value is the number of intervals,
/// spread evenly over the range of the data plus 1% at either end.
pub fn tcut(
    x: &[f64],
    breaks: &[f64],
    labels: Option<&[String]>,
    scale: f64,
) -> SurvivalResult<TcutResult> {
    if !scale.is_finite() {
        return Err(SurvivalError::invalid_input("scale must be finite"));
    }
    let (cutpoints, labels) = if let [count] = breaks {
        if count.is_nan() || *count < 1.0 {
            return Err(SurvivalError::invalid_input(
                "Must specify at least one interval",
            ));
        }
        let n_intervals = count.ceil() as usize;
        let labels = match labels {
            None => (1..=n_intervals).map(|i| format!("Range {i}")).collect(),
            Some(labels) if labels.len() == n_intervals => labels.to_vec(),
            Some(_) => {
                return Err(SurvivalError::invalid_input(
                    "Number of labels must equal number of intervals",
                ));
            }
        };
        // range(x[!is.na(x)]), with NA (no data) replaced by 1.
        let finite: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
        let (mut low, mut high) = if finite.is_empty() {
            (1.0, 1.0)
        } else {
            (
                finite.iter().copied().fold(f64::INFINITY, f64::min),
                finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        if !low.is_finite() || !high.is_finite() {
            return Err(SurvivalError::invalid_input(
                "x must be finite when breaks is an interval count",
            ));
        }
        let mut width = high - low;
        if width == 0.0 {
            high = low + 1.0;
            width = 1.0;
        }
        low -= 0.01 * width;
        high += 0.01 * width;
        (seq_length(low, high, n_intervals + 1), labels)
    } else {
        if breaks.len() < 2
            || breaks.iter().any(|b| b.is_nan())
            || breaks.windows(2).any(|w| w[1] < w[0])
        {
            return Err(SurvivalError::invalid_input(
                "breaks must be given in ascending order and contain no NA's",
            ));
        }
        let labels = match labels {
            None => {
                let lower = format_numbers(&breaks[..breaks.len() - 1]);
                let upper = format_numbers(&breaks[1..]);
                lower
                    .iter()
                    .zip(&upper)
                    .map(|(a, b)| format!("{a}+ thru {b}"))
                    .collect()
            }
            Some(labels) if labels.len() == breaks.len() - 1 => labels.to_vec(),
            Some(_) => {
                return Err(SurvivalError::invalid_input(
                    "Number of labels must be 1 less than number of break points",
                ));
            }
        };
        (breaks.to_vec(), labels)
    };
    Ok(TcutResult {
        values: x.iter().map(|v| v * scale).collect(),
        cutpoints: cutpoints.iter().map(|c| c * scale).collect(),
        labels,
    })
}

/// Python entry point of [`tcut`].
#[pyfunction(name = "tcut")]
#[pyo3(signature = (x, breaks, labels=None, scale=1.0))]
pub fn tcut_py(
    x: Vec<f64>,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    scale: f64,
) -> PyResult<TcutResult> {
    Ok(tcut(&x, &breaks, labels.as_deref(), scale)?)
}

// ---------------------------------------------------------------------------
// R's format() for a numeric vector (src/main/format.c, formatReal with
// digits = 7), which the default tcut labels are built from.
// ---------------------------------------------------------------------------

/// `R_print.digits`.
const DIGITS: i32 = 7;
/// Exact powers of ten up to `1e22`.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `scientific()`: sign, decimal exponent, significant digits needed (at
/// most `DIGITS`) and whether rounding to `DIGITS` widens the number.
fn scientific(x: f64) -> (bool, i32, i32, bool) {
    if x == 0.0 {
        return (false, 0, 1, false);
    }
    let neg = x < 0.0;
    let r = x.abs();
    let mut kp = r.log10().floor() as i32 - DIGITS + 1;
    let mut r_prec = if kp.abs() < 10 {
        if kp > 0 {
            r / POW10[kp as usize]
        } else {
            r * POW10[(-kp) as usize]
        }
    } else if kp <= -308 {
        // Denormals: 10^kp would underflow.
        (r * 1e303) / 10f64.powi(kp + 303)
    } else {
        r / 10f64.powi(kp)
    };
    if r_prec < POW10[(DIGITS - 1) as usize] {
        r_prec *= 10.0;
        kp -= 1;
    }
    let mut alpha = r_prec.round();
    let mut nsig = DIGITS;
    for _ in 1..=DIGITS {
        alpha /= 10.0;
        if alpha == alpha.floor() {
            nsig -= 1;
        } else {
            break;
        }
    }
    if nsig == 0 {
        nsig = 1;
        kp += 1;
    }
    let kpower = kp + DIGITS - 1;
    let rgt = (DIGITS - kpower).clamp(0, 22);
    let fuzz = 0.5 / POW10[rgt as usize];
    let widens = kpower > 0 && kpower <= 22 && r < POW10[kpower as usize] - fuzz;
    (neg, kpower, nsig, widens)
}

/// `format(x)` of a numeric vector: a common fixed or scientific layout,
/// padded to a common width.
pub fn format_numbers(x: &[f64]) -> Vec<String> {
    let mut neg = false;
    let mut rgt = i32::MIN;
    let mut mxl = i32::MIN;
    let mut mnl = i32::MAX;
    let mut mxsl = i32::MIN;
    let mut mxns = i32::MIN;
    let mut any_finite = false;
    for &value in x {
        if !value.is_finite() {
            continue;
        }
        any_finite = true;
        let (neg_i, kpower, nsig, widens) = scientific(value);
        let mut left = kpower + 1;
        if widens {
            left -= 1;
        }
        let sleft = i32::from(neg_i) + if left <= 0 { 1 } else { left };
        let right = nsig - left;
        neg |= neg_i;
        rgt = rgt.max(right);
        mxl = mxl.max(left);
        mnl = mnl.min(left);
        mxsl = mxsl.max(sleft);
        mxns = mxns.max(nsig);
    }
    let (fixed, width, decimals) = if any_finite {
        if mxl < 0 {
            mxsl = 1 + i32::from(neg);
        }
        rgt = rgt.max(0);
        let w_fixed = mxsl + rgt + i32::from(rgt != 0);
        let exponent_digits = if mxl > 100 || mnl <= -99 { 2 } else { 1 };
        let d = mxns - 1;
        let w_sci = i32::from(neg) + i32::from(d > 0) + d + 4 + exponent_digits;
        if w_fixed <= w_sci {
            (true, w_fixed, rgt)
        } else {
            (false, w_sci, d)
        }
    } else {
        (true, 0, 0)
    };
    let mut out: Vec<String> = x
        .iter()
        .map(|&value| {
            if value.is_nan() {
                "NaN".to_string()
            } else if value.is_infinite() {
                if value > 0.0 { "Inf" } else { "-Inf" }.to_string()
            } else if fixed {
                format!("{value:.*}", decimals as usize)
            } else {
                format_scientific(value, decimals as usize)
            }
        })
        .collect();
    let width = out
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(0)
        .max(width.max(0) as usize);
    for s in out.iter_mut() {
        if s.len() < width {
            *s = format!("{}{s}", " ".repeat(width - s.len()));
        }
    }
    out
}

/// C's `%.*e` with R's two-digit minimum exponent (`1e+05`).
fn format_scientific(value: f64, decimals: usize) -> String {
    let formatted = format!("{value:.*e}", decimals);
    let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn default_labels_use_r_number_formatting() {
        let result = tcut(
            &[10.0, 25.0, 40.0, 55.0, 70.0],
            &[0.0, 20.0, 50.0, 100.0],
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(result.values, vec![10.0, 25.0, 40.0, 55.0, 70.0]);
        assert_eq!(result.cutpoints, vec![0.0, 20.0, 50.0, 100.0]);
        assert_eq!(
            result.labels,
            strings(&[" 0+ thru  20", "20+ thru  50", "50+ thru 100"])
        );
        let days = tcut(
            &[1.0],
            &[0.0, 60.0 * 365.25, 70.0 * 365.25, 100.0 * 365.25],
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(
            days.labels,
            strings(&[
                "    0.0+ thru 21915.0",
                "21915.0+ thru 25567.5",
                "25567.5+ thru 36525.0"
            ])
        );
        let dates = tcut(&[1.0], &[9131.0, 9862.0, 10592.0], None, 1.0).unwrap();
        assert_eq!(
            dates.labels,
            strings(&["9131+ thru  9862", "9862+ thru 10592"])
        );
    }

    #[test]
    fn explicit_labels_and_scale() {
        let result = tcut(
            &[5.0, 15.0],
            &[0.0, 10.0, 20.0],
            Some(&strings(&["young", "old"])),
            2.0,
        )
        .unwrap();
        assert_eq!(result.labels, strings(&["young", "old"]));
        assert_eq!(result.values, vec![10.0, 30.0]);
        assert_eq!(result.cutpoints, vec![0.0, 20.0, 40.0]);
        assert!(tcut(&[5.0], &[0.0, 10.0], Some(&strings(&["a", "b"])), 1.0).is_err());
    }

    #[test]
    fn interval_count_spreads_breaks_over_the_padded_range() {
        let result = tcut(&[10.0, 25.0, 40.0, 55.0, 70.0], &[3.0], None, 1.0).unwrap();
        // R's seq(9.4, 70.6, length = 4), including its rounding.
        assert_eq!(
            result.cutpoints,
            vec![9.4, 29.799999999999997, 50.199999999999996, 70.6]
        );
        assert_eq!(result.labels, strings(&["Range 1", "Range 2", "Range 3"]));
        let constant = tcut(&[5.0, 5.0, f64::NAN], &[2.0], None, 1.0).unwrap();
        assert_eq!(constant.cutpoints, vec![4.99, 5.5, 6.01]);
        let empty = tcut(&[f64::NAN], &[1.0], None, 1.0).unwrap();
        assert_eq!(empty.cutpoints, vec![0.99, 2.01]);
        assert!(tcut(&[1.0], &[0.5], None, 1.0).is_err());
        assert!(tcut(&[1.0], &[2.0], Some(&strings(&["only"])), 1.0).is_err());
    }

    #[test]
    fn rejects_unsorted_or_missing_breaks() {
        assert!(tcut(&[0.5], &[0.0, f64::NAN], None, 1.0).is_err());
        assert!(tcut(&[0.5], &[2.0, 1.0], None, 1.0).is_err());
        assert!(tcut(&[0.5], &[], None, 1.0).is_err());
        assert!(tcut(&[0.5], &[0.0, 1.0], None, f64::NAN).is_err());
        // Equal adjacent breaks are allowed, as in R.
        assert!(tcut(&[0.5], &[0.0, 1.0, 1.0], None, 1.0).is_ok());
    }

    #[test]
    fn number_formatting_matches_r() {
        assert_eq!(
            format_numbers(&[0.5, 100.0, 1e6]),
            strings(&["5e-01", "1e+02", "1e+06"])
        );
        assert_eq!(format_numbers(&[1e-10, 1.0]), strings(&["1e-10", "1e+00"]));
        assert_eq!(
            format_numbers(&[123456789.0, 1.0]),
            strings(&["123456789", "        1"])
        );
        assert_eq!(format_numbers(&[-1.5, 2.25]), strings(&["-1.50", " 2.25"]));
        assert_eq!(
            format_numbers(&[0.1 + 0.2, 1.0 / 3.0]),
            strings(&["0.3000000", "0.3333333"])
        );
        assert_eq!(format_numbers(&[1e5]), strings(&["1e+05"]));
        assert_eq!(format_numbers(&[123456.7]), strings(&["123456.7"]));
        assert_eq!(format_numbers(&[1e-5, 1.0]), strings(&["1e-05", "1e+00"]));
        assert_eq!(format_numbers(&[0.0, 1e15]), strings(&["0e+00", "1e+15"]));
        assert_eq!(
            format_numbers(&[f64::NAN, 1.5, f64::INFINITY]),
            strings(&["NaN", "1.5", "Inf"])
        );
        assert_eq!(format_numbers(&[9999999.7]), strings(&["1e+07"]));
        assert_eq!(format_numbers(&[0.00012345678]), strings(&["0.0001234568"]));
        assert_eq!(format_numbers(&[0.001, 0.01]), strings(&["0.001", "0.010"]));
        assert_eq!(
            format_numbers(&[1234567.8, 1.0]),
            strings(&["1234568", "      1"])
        );
        assert_eq!(format_numbers(&[-0.5, 10.0]), strings(&["-0.5", "10.0"]));
        assert_eq!(
            format_numbers(&[99999.99, 0.5]),
            strings(&["99999.99", "    0.50"])
        );
        assert_eq!(format_numbers(&[2f64.sqrt()]), strings(&["1.414214"]));
        assert_eq!(
            format_numbers(&[1e6 / 3.0, 2e6 / 3.0]),
            strings(&["333333.3", "666666.7"])
        );
        assert_eq!(
            format_numbers(&[12.5, 1200.0]),
            strings(&["  12.5", "1200.0"])
        );
        assert_eq!(format_numbers(&[5e-324]), strings(&["4.940656e-324"]));
        assert_eq!(
            format_numbers(&[1e300, 1.0]),
            strings(&["1e+300", " 1e+00"])
        );
        assert_eq!(format_numbers(&[]), Vec::<String>::new());
    }
}
