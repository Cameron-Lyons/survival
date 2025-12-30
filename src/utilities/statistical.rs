#[inline]
pub fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    let k = df as f64 / 2.0;
    let x_half = x / 2.0;
    let ln_gamma_k = ln_gamma(k);
    let regularized_gamma = lower_incomplete_gamma(k, x_half) / ln_gamma_k.exp();
    1.0 - regularized_gamma
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
    let gamma_a = ln_gamma(a).exp();
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        gamma_a * (1.0 - gamma_continued_fraction(a, x))
    }
}

#[inline]
pub fn gamma_series(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = 100;
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
    let max_iter = 100;
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
}
