use crate::constants::ITERATIVE_MAX_ITER;
use std::f64::consts::SQRT_2;

#[inline]
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[inline]
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

#[inline]
#[allow(clippy::excessive_precision)]
pub fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[inline]
pub fn gamma_cdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(a, x)
}

#[inline]
pub fn gamma_inverse_cdf(p: f64, a: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let mut x = if a > 1.0 {
        let d = 1.0 / (9.0 * a);
        let z = normal_inverse_cdf(p);
        a * (1.0 - d + z * d.sqrt()).powi(3).max(0.001)
    } else {
        (p * ln_gamma(a).exp() * a).powf(1.0 / a).max(0.001)
    };

    let eps = 1e-10;
    let max_iter = 50;
    for _ in 0..max_iter {
        let cdf = gamma_cdf(x, a);
        let pdf = gamma_pdf(x, a);
        if pdf < 1e-300 {
            break;
        }
        let delta = (cdf - p) / pdf;
        x -= delta;
        x = x.max(1e-10);
        if delta.abs() < eps * x {
            break;
        }
    }
    x
}

#[inline]
fn gamma_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

#[inline]
pub fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    let k = df as f64 / 2.0;
    let x_half = x / 2.0;
    1.0 - lower_incomplete_gamma(k, x_half)
}

#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[inline]
pub fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

#[inline]
pub fn gamma_series(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = ITERATIVE_MAX_ITER;
    let mut sum = 1.0 / a;
    let mut term = sum;
    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

#[inline]
pub fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = ITERATIVE_MAX_ITER;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_sf_basic() {
        assert!((chi2_sf(0.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(-1.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(1.0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_gamma() {
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_inverse_cdf() {
        let result = gamma_inverse_cdf(0.475, 5.0);
        assert!(
            result > 4.0 && result < 5.0,
            "Expected ~4.5, got {}",
            result
        );

        let result2 = gamma_inverse_cdf(0.525, 6.0);
        assert!(
            result2 > 5.0 && result2 < 7.0,
            "Expected ~6, got {}",
            result2
        );
    }
}
