//! Statistical distribution functions ported from R's `nmath` library.
//!
//! Every public function mirrors the R function of the same name (`pnorm`,
//! `qnorm`, `lgammafn`, `pgamma`, `qgamma`, `pbeta`, `pt`, `qt`, ...) and
//! follows the same algorithm, so results agree with R to roughly 1e-15
//! relative error instead of the ~1e-7 of the textbook approximations they
//! replace.  The `lower_tail` / `log_p` flags have the same meaning as R's
//! `lower.tail` / `log.p` arguments.
//!
//! Two deliberate simplifications: `pbeta` evaluates TOMS 708's `bfrac`
//! continued fraction with the `brcomp` prefactor for every argument instead
//! of porting the whole 2000-line `bratio` dispatcher, and `dpois_raw` uses
//! the single-double deviance `bd0` of R <= 4.0 rather than the table-driven
//! `ebd0`.  Both are verified against R in the tests below; the latter costs
//! at most ~2e-13 relative accuracy, and only in tails below 1e-100.
#![allow(clippy::excessive_precision)]

use std::f64::consts::{FRAC_1_PI, FRAC_PI_2, LN_2, PI, SQRT_2, TAU};

/// `log(sqrt(2*pi))`
const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;
/// `log(sqrt(pi/2))`
const M_LN_SQRT_PID2: f64 = 0.225791352644727432363097614947;
/// `1/sqrt(2*pi)`
const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934;
/// `sqrt(32)`
const M_SQRT_32: f64 = 5.656854249492380195206754896838;
/// `2*pi`
const M_2PI: f64 = TAU;
/// `log(2*pi)`
const M_LN_2PI: f64 = 1.837877066409345483560659472811;
/// `sqrt(2*pi)`
const M_SQRT_2PI: f64 = 2.506628274631000502415765284811;
const DBL_EPSILON: f64 = f64::EPSILON;
const DBL_MIN: f64 = f64::MIN_POSITIVE;

// ---------------------------------------------------------------------------
// R's dpq.h helpers (`R_D__0`, `R_DT_qIv`, ...).
// ---------------------------------------------------------------------------

#[inline]
fn r_d_0(log_p: bool) -> f64 {
    if log_p { f64::NEG_INFINITY } else { 0.0 }
}

#[inline]
fn r_d_1(log_p: bool) -> f64 {
    if log_p { 0.0 } else { 1.0 }
}

#[inline]
fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_0(log_p)
    } else {
        r_d_1(log_p)
    }
}

#[inline]
fn r_dt_1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_1(log_p)
    } else {
        r_d_0(log_p)
    }
}

#[inline]
fn r_d_lval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail { p } else { 0.5 - p + 0.5 }
}

#[inline]
fn r_d_cval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail { 0.5 - p + 0.5 } else { p }
}

#[inline]
fn r_d_exp(x: f64, log_p: bool) -> f64 {
    if log_p { x } else { x.exp() }
}

#[inline]
fn r_d_log(p: f64, log_p: bool) -> f64 {
    if log_p { p } else { p.ln() }
}

/// `log(1 - exp(x))` in a numerically stable form (R's `R_Log1_Exp`).
#[inline]
fn r_log1_exp(x: f64) -> f64 {
    if x > -LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

#[inline]
fn r_d_lexp(x: f64, log_p: bool) -> f64 {
    if log_p { r_log1_exp(x) } else { (-x).ln_1p() }
}

/// Lower-tail probability on the plain scale (R's `R_DT_qIv`).
#[inline]
fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail { p.exp() } else { -p.exp_m1() }
    } else {
        r_d_lval(p, lower_tail)
    }
}

/// Upper-tail probability on the plain scale (R's `R_DT_CIv`).
#[inline]
fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail { -p.exp_m1() } else { p.exp() }
    } else {
        r_d_cval(p, lower_tail)
    }
}

#[inline]
fn r_dt_log(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_log(p, log_p)
    } else {
        r_d_lexp(p, log_p)
    }
}

#[inline]
fn r_dt_clog(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_lexp(p, log_p)
    } else {
        r_d_log(p, log_p)
    }
}

/// R's `R_Q_P01_boundaries`: `Some(value)` when `p` is invalid (NaN) or on a
/// boundary where the quantile is `left`/`right`.
#[inline]
fn q_p01_boundaries(p: f64, left: f64, right: f64, lower_tail: bool, log_p: bool) -> Option<f64> {
    if log_p {
        if p > 0.0 {
            return Some(f64::NAN);
        }
        if p == 0.0 {
            return Some(if lower_tail { right } else { left });
        }
        if p == f64::NEG_INFINITY {
            return Some(if lower_tail { left } else { right });
        }
    } else {
        if !(0.0..=1.0).contains(&p) {
            return Some(f64::NAN);
        }
        if p == 0.0 {
            return Some(if lower_tail { left } else { right });
        }
        if p == 1.0 {
            return Some(if lower_tail { right } else { left });
        }
    }
    None
}

/// `sin(pi * x)`, exact at half-integers (R's `sinpi`).
fn sinpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut x = x % 2.0;
    if x <= -1.0 {
        x += 2.0;
    } else if x > 1.0 {
        x -= 2.0;
    }
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    if x == 0.5 {
        return 1.0;
    }
    if x == -0.5 {
        return -1.0;
    }
    (PI * x).sin()
}

/// `tan(pi * x)`, exact at quarter-integers (R's `tanpi`).
fn tanpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut x = x % 1.0;
    if x <= -0.5 {
        x += 1.0;
    } else if x > 0.5 {
        x -= 1.0;
    }
    if x == 0.0 {
        0.0
    } else if x == 0.5 {
        f64::NAN
    } else if x == 0.25 {
        1.0
    } else if x == -0.25 {
        -1.0
    } else {
        (PI * x).tan()
    }
}

// ---------------------------------------------------------------------------
// Normal distribution.
// ---------------------------------------------------------------------------

/// R's `pnorm(x, 0, 1, lower.tail, log.p)`: Cody (1993) rational Chebyshev
/// approximations, as in `nmath/pnorm.c`.
pub(crate) fn pnorm(x: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if !x.is_finite() {
        return if x < 0.0 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    let (cum, ccum) = pnorm_both(x, lower_tail, !lower_tail, log_p);
    if lower_tail { cum } else { ccum }
}

/// Cody's central rational approximation: `pnorm(x) - 0.5` for
/// `|x| <= 0.67448975`, computed without the `0.5 + ... - 0.5` round trip.
fn pnorm_central_offset(x: f64) -> f64 {
    const A: [f64; 5] = [
        2.2352520354606839287,
        161.02823106855587881,
        1067.6894854603709582,
        18154.981253343561249,
        0.065682337918207449113,
    ];
    const B: [f64; 4] = [
        47.20258190468824187,
        976.09855173777669322,
        10260.932208618978205,
        45507.789335026729956,
    ];
    let eps = DBL_EPSILON * 0.5;
    let (xnum, xden) = if x.abs() > eps {
        let xsq = x * x;
        let mut xnum = A[4] * xsq;
        let mut xden = xsq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * xsq;
            xden = (xden + B[i]) * xsq;
        }
        (xnum, xden)
    } else {
        (0.0, 0.0)
    };
    x * (xnum + A[3]) / (xden + B[3])
}

/// R's `pnorm_both`: returns `(P[X <= x], P[X > x])`; only the tails
/// requested through `lower`/`upper` are guaranteed to be computed.
fn pnorm_both(x: f64, lower: bool, upper: bool, log_p: bool) -> (f64, f64) {
    const C: [f64; 9] = [
        0.39894151208813466764,
        8.8831497943883759412,
        93.506656132177855979,
        597.27027639480026226,
        2494.5375852903726711,
        6848.1904505362823326,
        11602.651437647350124,
        9842.7148383839780218,
        1.0765576773720192317e-8,
    ];
    const D: [f64; 8] = [
        22.266688044328115691,
        235.38790178262499861,
        1519.377599407554805,
        6485.558298266760755,
        18615.571640885098091,
        34900.952721145977266,
        38912.003286093271411,
        19685.429676859990727,
    ];
    const P: [f64; 6] = [
        0.21589853405795699,
        0.1274011611602473639,
        0.022235277870649807,
        0.001421619193227893466,
        2.9112874951168792e-5,
        0.02307344176494017303,
    ];
    const Q: [f64; 5] = [
        1.28426009614491121,
        0.468238212480865118,
        0.0659881378689285515,
        0.00378239633202758244,
        7.29751555083966205e-5,
    ];

    let y = x.abs();
    let mut cum = f64::NAN;
    let mut ccum = f64::NAN;

    // R's `do_del(X)` followed by `swap_tail`.
    let del_and_swap = |big_x: f64, temp: f64| -> (f64, f64) {
        let xsq = (big_x * 16.0).trunc() / 16.0;
        let del = (big_x - xsq) * (big_x + xsq);
        let (mut cum, mut ccum);
        if log_p {
            cum = (-xsq * (xsq * 0.5)) - del * 0.5 + temp.ln();
            ccum = f64::NAN;
            if (lower && x > 0.0) || (upper && x <= 0.0) {
                ccum = (-(-xsq * (xsq * 0.5)).exp() * (-del * 0.5).exp() * temp).ln_1p();
            }
        } else {
            cum = (-xsq * (xsq * 0.5)).exp() * (-del * 0.5).exp() * temp;
            ccum = 1.0 - cum;
        }
        if x > 0.0 {
            let temp = cum;
            if lower {
                cum = ccum;
            }
            ccum = temp;
        }
        (cum, ccum)
    };

    if y <= 0.67448975 {
        // qnorm(3/4) = .6744....
        let temp = pnorm_central_offset(x);
        if lower {
            cum = 0.5 + temp;
        }
        if upper {
            ccum = 0.5 - temp;
        }
        if log_p {
            if lower {
                cum = cum.ln();
            }
            if upper {
                ccum = ccum.ln();
            }
        }
    } else if y <= M_SQRT_32 {
        // qnorm(3/4) < |x| <= sqrt(32) ~= 5.657
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let temp = (xnum + C[7]) / (xden + D[7]);
        (cum, ccum) = del_and_swap(y, temp);
    } else if (log_p && y < 1e170)
        || (lower && -38.4674 < x && x < 8.2924)
        || (upper && -8.2924 < x && x < 38.4674)
    {
        // |x| > sqrt(32): asymptotic expansion; the bounds are where the
        // non-log tail underflows even the denormal range.
        let xsq = 1.0 / (x * x);
        let mut xnum = P[5] * xsq;
        let mut xden = xsq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * xsq;
            xden = (xden + Q[i]) * xsq;
        }
        let temp = xsq * (xnum + P[4]) / (xden + Q[4]);
        let temp = (M_1_SQRT_2PI - temp) / y;
        (cum, ccum) = del_and_swap(x, temp);
    } else if x > 0.0 {
        cum = r_d_1(log_p);
        ccum = r_d_0(log_p);
    } else {
        cum = r_d_0(log_p);
        ccum = r_d_1(log_p);
    }
    (cum, ccum)
}

/// R's `qnorm(p, 0, 1, lower.tail, log.p)`: Wichura's AS 241 (`PPND16`) with
/// R's asymptotic extension for `log.p = TRUE` and extremely small `p`, as
/// in `nmath/qnorm.c`.
pub(crate) fn qnorm(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if let Some(bound) = q_p01_boundaries(p, f64::NEG_INFINITY, f64::INFINITY, lower_tail, log_p) {
        return bound;
    }

    let p_ = r_dt_qiv(p, lower_tail, log_p);
    let q = p_ - 0.5;

    if q.abs() <= 0.425 {
        // 0.075 <= p~ <= 0.925
        let r = 0.180625 - q * q;
        return q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }

    // p~ = min(p, 1-p) < 0.075 :  r := sqrt(-log(p~))
    let lp = if log_p && ((lower_tail && q <= 0.0) || (!lower_tail && q > 0.0)) {
        p
    } else {
        (if q > 0.0 {
            r_dt_civ(p, lower_tail, log_p)
        } else {
            p_
        })
        .ln()
    };
    let r = (-lp).sqrt();
    let mut val = if r <= 5.0 {
        // min(p,1-p) >= exp(-25) ~= 1.3888e-11
        let r = r - 1.6;
        (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else if r <= 27.0 {
        // min(p,1-p) >= exp(-729) ~= 2.5e-317
        let r = r - 5.0;
        (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    } else if r >= 6.4e8 {
        // p is *very extremely* close to 0 or 1: zeroth-order asymptotic.
        r * SQRT_2
    } else {
        let s2 = -2.0 * lp;
        let mut x2 = s2 - (M_2PI * s2).ln();
        if r < 36000.0 {
            x2 = s2 - (M_2PI * x2).ln() - 2.0 / (2.0 + x2);
            if r < 840.0 {
                x2 =
                    s2 - (M_2PI * x2).ln() + 2.0 * (-(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2)).ln_1p();
                if r < 109.0 {
                    x2 = s2 - (M_2PI * x2).ln()
                        + 2.0
                            * (-(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2)).ln_1p();
                    if r < 55.0 {
                        x2 = s2 - (M_2PI * x2).ln()
                            + 2.0
                                * (-(1.0
                                    - (1.0 - (5.0 - 9.0 / (8.0 + x2)) / (6.0 + x2)) / (4.0 + x2))
                                    / (2.0 + x2))
                                    .ln_1p();
                    }
                }
            }
        }
        x2.sqrt()
    };
    if q < 0.0 {
        val = -val;
    }
    val
}

/// Error function `erf(x) = 2 pnorm(x sqrt 2) - 1`, evaluated through the
/// tails of `pnorm` so it keeps relative accuracy for both small and large
/// `|x|` (R itself has no `erf`; this is the identity its documentation
/// gives).
pub(crate) fn erf(x: f64) -> f64 {
    let z = x * SQRT_2;
    if z.abs() <= 0.67448975 {
        2.0 * pnorm_central_offset(z)
    } else if x > 0.0 {
        1.0 - 2.0 * pnorm(z, false, false)
    } else {
        2.0 * pnorm(z, true, false) - 1.0
    }
}

/// Complementary error function `erfc(x) = 2 pnorm(x sqrt 2, lower = FALSE)`.
pub(crate) fn erfc(x: f64) -> f64 {
    2.0 * pnorm(x * SQRT_2, false, false)
}

/// R's `dnorm(x, 0, 1, log)` as in `nmath/dnorm.c` (Welinder's split of `x`
/// for accuracy in the far tail).
pub(crate) fn dnorm(x: f64, give_log: bool) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if !x.is_finite() {
        return r_d_0(give_log);
    }
    let x = x.abs();
    if x >= 2.0 * f64::MAX.sqrt() {
        return r_d_0(give_log);
    }
    if give_log {
        return -(M_LN_SQRT_2PI + 0.5 * x * x);
    }
    if x < 5.0 {
        return M_1_SQRT_2PI * (-0.5 * x * x).exp();
    }
    // Underflow boundary (denormals included) ~= 38.57.
    if x > (-2.0 * LN_2 * ((f64::MIN_EXP + 1 - f64::MANTISSA_DIGITS as i32) as f64)).sqrt() {
        return 0.0;
    }
    // Split x = x1 + x2 with |x2| <= 2^-16 so x1*x1 is error free.
    let x1 = (x * 65536.0).round_ties_even() / 65536.0;
    let x2 = x - x1;
    M_1_SQRT_2PI * ((-0.5 * x1 * x1).exp() * ((-0.5 * x2 - x1) * x2).exp())
}

// ---------------------------------------------------------------------------
// Gamma function family.
// ---------------------------------------------------------------------------

/// R's `chebyshev_eval` (Clenshaw recurrence).
fn chebyshev_eval(x: f64, a: &[f64], n: usize) -> f64 {
    if !(1..=1000).contains(&n) || !(-1.1..=1.1).contains(&x) {
        return f64::NAN;
    }
    let twox = x * 2.0;
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for i in 1..=n {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

/// R's `lgammacor(x)`: the Stirling-series correction
/// `log(gamma(x)) - ((x - 0.5) log(x) - x + log(sqrt(2 pi)))` for `x >= 10`.
fn lgammacor(x: f64) -> f64 {
    const ALGMCS: [f64; 15] = [
        0.1666389480451863247205729650822e+0,
        -0.1384948176067563840732986059135e-4,
        0.9810825646924729426157171547487e-8,
        -0.1809129475572494194263306266719e-10,
        0.6221098041892605227126015543416e-13,
        -0.3399615005417721944303330599666e-15,
        0.2683181998482698748957538846666e-17,
        -0.2868042435334643284144622399999e-19,
        0.3962837061046434803679306666666e-21,
        -0.6831888753985766870111999999999e-23,
        0.1429227355942498147573333333333e-24,
        -0.3547598158101070547199999999999e-26,
        0.1025680058010470912000000000000e-27,
        -0.3401102254316748799999999999999e-29,
        0.1276642195630062933333333333333e-30,
    ];
    const NALGM: usize = 5;
    const XBIG: f64 = 94906265.62425156;

    if x < 10.0 {
        f64::NAN
    } else if x < XBIG {
        let tmp = 10.0 / x;
        chebyshev_eval(tmp * tmp * 2.0 - 1.0, &ALGMCS, NALGM) / x
    } else {
        1.0 / (x * 12.0)
    }
}

/// R's `gammafn(x)` (SLATEC `gamma` Chebyshev series on [-10, 10], Stirling
/// with `lgammacor`/`stirlerr` beyond), as in `nmath/gamma.c`.
pub(crate) fn gammafn(x: f64) -> f64 {
    const GAMCS: [f64; 42] = [
        0.8571195590989331421920062399942e-2,
        0.4415381324841006757191315771652e-2,
        0.5685043681599363378632664588789e-1,
        -0.4219835396418560501012500186624e-2,
        0.1326808181212460220584006796352e-2,
        -0.1893024529798880432523947023886e-3,
        0.3606925327441245256578082217225e-4,
        -0.6056761904460864218485548290365e-5,
        0.1055829546302283344731823509093e-5,
        -0.1811967365542384048291855891166e-6,
        0.3117724964715322277790254593169e-7,
        -0.5354219639019687140874081024347e-8,
        0.9193275519859588946887786825940e-9,
        -0.1577941280288339761767423273953e-9,
        0.2707980622934954543266540433089e-10,
        -0.4646818653825730144081661058933e-11,
        0.7973350192007419656460767175359e-12,
        -0.1368078209830916025799499172309e-12,
        0.2347319486563800657233471771688e-13,
        -0.4027432614949066932766570534699e-14,
        0.6910051747372100912138336975257e-15,
        -0.1185584500221992907052387126192e-15,
        0.2034148542496373955201026051932e-16,
        -0.3490054341717405849274012949108e-17,
        0.5987993856485305567135051066026e-18,
        -0.1027378057872228074490069778431e-18,
        0.1762702816060529824942759660748e-19,
        -0.3024320653735306260958772112042e-20,
        0.5188914660218397839717833550506e-21,
        -0.8902770842456576692449251601066e-22,
        0.1527474068493342602274596891306e-22,
        -0.2620731256187362900257328332799e-23,
        0.4496464047830538670331046570666e-24,
        -0.7714712731336877911703901525333e-25,
        0.1323635453126044036486572714666e-25,
        -0.2270999412942928816702313813333e-26,
        0.3896418998003991449320816639999e-27,
        -0.6685198115125953327792127999999e-28,
        0.1146998663140024384347613866666e-28,
        -0.1967938586345134677295103999999e-29,
        0.3376448816585338090334890666666e-30,
        -0.5793070335782135784625493333333e-31,
    ];
    const NGAM: usize = 22;
    const XMIN: f64 = -170.5674972726612;
    const XMAX: f64 = 171.61447887182298;
    const XSML: f64 = 2.2474362225598545e-308;

    if x.is_nan() {
        return x;
    }
    if x == 0.0 || (x < 0.0 && x == x.round()) {
        return f64::NAN;
    }

    let y = x.abs();
    if y <= 10.0 {
        // Reduce to gamma(1 + y) for 0 <= y < 1.
        let mut n = x.trunc() as i32;
        if x < 0.0 {
            n -= 1;
        }
        let y = x - n as f64; // n = floor(x)  ==>  y in [0, 1)
        n -= 1;
        let mut value = chebyshev_eval(y * 2.0 - 1.0, &GAMCS, NGAM) + 0.9375;
        if n == 0 {
            return value; // x = 1.dddd = 1+y
        }
        if n < 0 {
            // gamma(x) for -10 <= x < 1
            if y < XSML {
                return if x > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
            for i in 0..(-n) {
                value /= x + i as f64;
            }
            value
        } else {
            // gamma(x) for 2 <= x <= 10
            for i in 1..=n {
                value *= y + i as f64;
            }
            value
        }
    } else {
        if x > XMAX {
            return f64::INFINITY;
        }
        if x < XMIN {
            return 0.0;
        }
        let value = if y <= 50.0 && y == y.trunc() {
            // (n - 1)! exactly
            let mut value = 1.0;
            let mut i = 2.0;
            while i < y {
                value *= i;
                i += 1.0;
            }
            value
        } else {
            // R writes `(2*y == (int)2*y) ? stirlerr(y) : lgammacor(y)`, and
            // since the cast binds to the literal 2 the test is always true.
            ((y - 0.5) * y.ln() - y + M_LN_SQRT_2PI + stirlerr(y)).exp()
        };
        if x > 0.0 {
            return value;
        }
        let sinpiy = sinpi(y);
        if sinpiy == 0.0 {
            return f64::INFINITY;
        }
        -PI / (y * sinpiy * value)
    }
}

/// R's `lgammafn(x)` = `log|gamma(x)|`, as in `nmath/lgamma.c`.
pub(crate) fn lgammafn(x: f64) -> f64 {
    const XMAX: f64 = 2.5327372760800758e+305;

    if x.is_nan() {
        return x;
    }
    if x <= 0.0 && x == x.trunc() {
        return f64::INFINITY; // lgamma(-n) = log|gamma(-n)| = +Inf
    }
    let y = x.abs();
    if y < 1e-306 {
        return -y.ln();
    }
    if y <= 10.0 {
        return gammafn(x).abs().ln();
    }
    if y > XMAX {
        return f64::INFINITY;
    }
    if x > 0.0 {
        if x > 1e17 {
            return x * (x.ln() - 1.0);
        }
        if x > 4934720.0 {
            return M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x;
        }
        return M_LN_SQRT_2PI + (x - 0.5) * x.ln() - x + lgammacor(x);
    }
    // x < -10; y = -x
    let sinpiy = sinpi(y).abs();
    if sinpiy == 0.0 {
        return f64::NAN;
    }
    M_LN_SQRT_PID2 + (x - 0.5) * y.ln() - x - sinpiy.ln() - lgammacor(y)
}

/// Continued fraction for `sum_{k>=0} x^k / (i + k d)`; auxiliary of
/// `log1pmx` and `lgamma1p` (R's `logcf`).
fn logcf(x: f64, i: f64, d: f64, eps: f64) -> f64 {
    let mut c1 = 2.0 * d;
    let mut c2 = i + d;
    let mut c4 = c2 + d;
    let mut a1 = c2;
    let mut b1 = i * (c2 - i * x);
    let mut b2 = d * d * x;
    let mut a2 = c4 * c2 - b2;

    b2 = c4 * b1 - i * b2;

    while (a2 * b1 - a1 * b2).abs() > (eps * b1 * b2).abs() {
        let mut c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if b2.abs() > SCALEFACTOR {
            a1 /= SCALEFACTOR;
            b1 /= SCALEFACTOR;
            a2 /= SCALEFACTOR;
            b2 /= SCALEFACTOR;
        } else if b2.abs() < 1.0 / SCALEFACTOR {
            a1 *= SCALEFACTOR;
            b1 *= SCALEFACTOR;
            a2 *= SCALEFACTOR;
            b2 *= SCALEFACTOR;
        }
    }
    a2 / b2
}

/// `2^256`, R's `scalefactor` for continued fractions.
const SCALEFACTOR: f64 = f64::from_bits(0x4FF0_0000_0000_0000);

/// Accurate `log(1 + x) - x`, particularly for small `x` (R's `log1pmx`).
pub(crate) fn log1pmx(x: f64) -> f64 {
    const MIN_LOG1_VALUE: f64 = -0.79149064;
    if !(MIN_LOG1_VALUE..=1.0).contains(&x) {
        x.ln_1p() - x
    } else {
        // Expand in y = [x/(2+x)]^2: log(1+x) - x = x/(2+x) * [2 y S(y) - x]
        let r = x / (2.0 + x);
        let y = r * r;
        if x.abs() < 1e-2 {
            r * ((((2.0 / 9.0 * y + 2.0 / 7.0) * y + 2.0 / 5.0) * y + 2.0 / 3.0) * y - x)
        } else {
            r * (2.0 * y * logcf(y, 3.0, 2.0, 1e-14) - x)
        }
    }
}

/// Accurate `log(gamma(1 + a))` for small `|a|` (R's `lgamma1p`).
pub(crate) fn lgamma1p(a: f64) -> f64 {
    if a.abs() >= 0.5 {
        return lgammafn(a + 1.0);
    }
    const EULERS_CONST: f64 = 0.5772156649015328606065120900824024;
    /// `coeffs[i]` holds `(zeta(i+2) - 1) / (i+2)`.
    const COEFFS: [f64; 40] = [
        0.3224670334241132182362075833230126e-0,
        0.6735230105319809513324605383715000e-1,
        0.2058080842778454787900092413529198e-1,
        0.7385551028673985266273097291406834e-2,
        0.2890510330741523285752988298486755e-2,
        0.1192753911703260977113935692828109e-2,
        0.5096695247430424223356548135815582e-3,
        0.2231547584535793797614188036013401e-3,
        0.9945751278180853371459589003190170e-4,
        0.4492623673813314170020750240635786e-4,
        0.2050721277567069155316650397830591e-4,
        0.9439488275268395903987425104415055e-5,
        0.4374866789907487804181793223952411e-5,
        0.2039215753801366236781900709670839e-5,
        0.9551412130407419832857179772951265e-6,
        0.4492469198764566043294290331193655e-6,
        0.2120718480555466586923135901077628e-6,
        0.1004322482396809960872083050053344e-6,
        0.4769810169363980565760193417246730e-7,
        0.2271109460894316491031998116062124e-7,
        0.1083865921489695409107491757968159e-7,
        0.5183475041970046655121248647057669e-8,
        0.2483674543802478317185008663991718e-8,
        0.1192140140586091207442548202774640e-8,
        0.5731367241678862013330194857961011e-9,
        0.2759522885124233145178149692816341e-9,
        0.1330476437424448948149715720858008e-9,
        0.6422964563838100022082448087644648e-10,
        0.3104424774732227276239215783404066e-10,
        0.1502138408075414217093301048780668e-10,
        0.7275974480239079662504549924814047e-11,
        0.3527742476575915083615072228655483e-11,
        0.1711991790559617908601084114443031e-11,
        0.8315385841420284819798357793954418e-12,
        0.4042200525289440065536008957032895e-12,
        0.1966475631096616490411045679010286e-12,
        0.9573630387838555763782200936508615e-13,
        0.4664076026428374224576492565974577e-13,
        0.2273736960065972320633279596737272e-13,
        0.1109139947083452201658320007192334e-13,
    ];
    const N: usize = 40;
    /// `zeta(N + 2) - 1`
    const C: f64 = 0.2273736845824652515226821577978691e-12;
    const TOL_LOGCF: f64 = 1e-14;

    // Abramowitz & Stegun 6.1.33 with a convergence acceleration for the
    // tail of the zeta series.
    let mut lgam = C * logcf(-a / 2.0, (N + 2) as f64, 1.0, TOL_LOGCF);
    for &coeff in COEFFS.iter().rev() {
        lgam = coeff - a * lgam;
    }
    (a * lgam - EULERS_CONST) * a - log1pmx(a)
}

/// Evaluates R's nested Stirling series
/// `(S0 - (S1 - (S2 - ... - S_k / nn) / nn ...) / nn) / n` for the leading
/// coefficients in `coeffs`, in exactly R's operation order.
fn stirlerr_series(n: f64, nn: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &coeff in coeffs.iter().rev() {
        acc = coeff - acc / nn;
    }
    acc / n
}

/// R's `stirlerr(n)` = `log(n!) - log(sqrt(2 pi n) (n/e)^n)`, the error of
/// Stirling's formula, as in `nmath/stirlerr.c` (R >= 4.4).
fn stirlerr(n: f64) -> f64 {
    /// `S[k]` is the k-th Stirling series coefficient `B_{2k+2} / ((2k+1)(2k+2))`:
    /// 1/12, 1/360, 1/1260, 1/1680, 1/1188, 691/360360, 1/156, 3617/122400, ...
    const S: [f64; 17] = [
        0.083333333333333333333,
        0.00277777777777777777778,
        0.00079365079365079365079365,
        0.000595238095238095238095238,
        0.0008417508417508417508417508,
        0.0019175269175269175269175262,
        0.0064102564102564102564102561,
        0.029550653594771241830065352,
        0.17964437236883057316493850,
        1.3924322169059011164274315,
        13.402864044168391994478957,
        156.84828462600201730636509,
        2193.1033333333333333333333,
        36108.771253724989357173269,
        691472.26885131306710839498,
        15238221.539407416192283370,
        382900751.39141414141414141,
    ];

    /// Exact values for n = 0, 0.5, 1.0, ..., 14.5, 15.0.
    const SFERR_HALVES: [f64; 31] = [
        0.0, // n=0 - wrong, place holder only
        0.1534264097200273452913848,
        0.0810614667953272582196702,
        0.0548141210519176538961390,
        0.0413406959554092940938221,
        0.03316287351993628748511048,
        0.02767792568499833914878929,
        0.02374616365629749597132920,
        0.02079067210376509311152277,
        0.01848845053267318523077934,
        0.01664469118982119216319487,
        0.01513497322191737887351255,
        0.01387612882307074799874573,
        0.01281046524292022692424986,
        0.01189670994589177009505572,
        0.01110455975820691732662991,
        0.010411265261972096497478567,
        0.009799416126158803298389475,
        0.009255462182712732917728637,
        0.008768700134139385462952823,
        0.008330563433362871256469318,
        0.007934114564314020547248100,
        0.007573675487951840794972024,
        0.007244554301320383179543912,
        0.006942840107209529865664152,
        0.006665247032707682442354394,
        0.006408994188004207068439631,
        0.006171712263039457647532867,
        0.005951370112758847735624416,
        0.005746216513010115682023589,
        0.005554733551962801371038690,
    ];

    if n <= 23.5 {
        let nn = n + n;
        if n <= 15.0 && nn == nn.trunc() {
            return SFERR_HALVES[nn as usize];
        }
        if n <= 5.25 {
            if n >= 1.0 {
                let l_n = n.ln();
                return lgammafn(n) + n * (1.0 - l_n) + (l_n - M_LN_2PI) * 0.5;
            }
            return lgamma1p(n) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
        }
        // 5.25 < n <= 23.5: the number of series terms R uses per range.
        let terms = if n > 12.8 {
            7
        } else if n > 12.3 {
            8
        } else if n > 8.9 {
            9
        } else if n > 7.3 {
            11
        } else if n > 6.6 {
            13
        } else if n > 6.1 {
            15
        } else {
            17
        };
        stirlerr_series(n, n * n, &S[..terms])
    } else {
        let terms = if n > 15.7e6 {
            1
        } else if n > 6180.0 {
            2
        } else if n > 205.0 {
            3
        } else if n > 86.0 {
            4
        } else if n > 27.0 {
            5
        } else {
            6
        };
        stirlerr_series(n, n * n, &S[..terms])
    }
}

/// R's `bd0(x, np)` = `x log(x/np) + np - x`, the deviance term, evaluated
/// by a Taylor series when `x` is close to `np` (as in `nmath/bd0.c`).
fn bd0(x: f64, np: f64) -> f64 {
    if !x.is_finite() || !np.is_finite() || np == 0.0 {
        return f64::NAN;
    }
    if (x - np).abs() < 0.1 * (x + np) {
        let d = x - np;
        let mut v = d / (x + np);
        if d != 0.0 && v == 0.0 {
            // v underflowed to 0 because x + np overflowed.
            let x_ = x * 0.25;
            let n_ = np * 0.25;
            v = (x_ - n_) / (x_ + n_);
        }
        let mut s = (d * 0.5) * v;
        if (s * 2.0).abs() < DBL_MIN {
            return s * 2.0;
        }
        let mut ej = x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v; // = x v^(2j+1)
            let s_ = s;
            s += ej / ((2 * j + 1) as f64);
            if s == s_ {
                return s * 2.0;
            }
        }
    }
    // |x - np| is not too small
    let lg_x_n = if (x / np).is_finite() {
        (x / np).ln()
    } else {
        x.ln() - np.ln()
    };
    if x > np {
        x * (lg_x_n - 1.0) + np
    } else {
        x * lg_x_n + np - x
    }
}

/// R's `dpois_raw(x, lambda, log)`: the Poisson density kernel
/// `lambda^x e^-lambda / gamma(x+1)` for real `x >= 0`, via `stirlerr` and
/// `bd0` as in `nmath/dpois.c` (R <= 4.0 form: single-double deviance).
fn dpois_raw(x: f64, lambda: f64, give_log: bool) -> f64 {
    /// `2^1023 / pi`: above this `2 pi x` overflows.
    const X_LRG: f64 = 2.86111748575702815380240589208115399625e+307;

    if lambda == 0.0 {
        return if x == 0.0 {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }
    if !lambda.is_finite() {
        return r_d_0(give_log);
    }
    if x < 0.0 {
        return r_d_0(give_log);
    }
    if x <= lambda * DBL_MIN {
        return r_d_exp(-lambda, give_log);
    }
    if lambda < x * DBL_MIN {
        if !x.is_finite() {
            return r_d_0(give_log);
        }
        return r_d_exp(-lambda + x * lambda.ln() - lgammafn(x + 1.0), give_log);
    }
    let lrg_x = x >= X_LRG;
    let r = if lrg_x {
        M_SQRT_2PI * x.sqrt()
    } else {
        M_2PI * x
    };
    let y = -stirlerr(x) - bd0(x, lambda);
    if give_log {
        y - if lrg_x { r.ln() } else { 0.5 * r.ln() }
    } else {
        y.exp() / if lrg_x { r } else { r.sqrt() }
    }
}

/// R's `dpois_wrap(x_plus_1, lambda, log)` from `nmath/pgamma.c`.
fn dpois_wrap(x_plus_1: f64, lambda: f64, give_log: bool) -> f64 {
    /// `M_LN2 * DBL_MAX_EXP / DBL_EPSILON` = 3.196577e18
    const M_CUTOFF: f64 = LN_2 * (f64::MAX_EXP as f64) / DBL_EPSILON;

    if !lambda.is_finite() {
        return r_d_0(give_log);
    }
    if x_plus_1 > 1.0 {
        return dpois_raw(x_plus_1 - 1.0, lambda, give_log);
    }
    if lambda > (x_plus_1 - 1.0).abs() * M_CUTOFF {
        return r_d_exp(-lambda - lgammafn(x_plus_1), give_log);
    }
    let d = dpois_raw(x_plus_1, lambda, give_log);
    if give_log {
        d + (x_plus_1 / lambda).ln()
    } else {
        d * (x_plus_1 / lambda)
    }
}

/// Abramowitz & Stegun 6.5.29 series for `x < 1` (R's `pgamma_smallx`).
fn pgamma_smallx(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut sum = 0.0;
    let mut c = alph;
    let mut n = 0.0;
    loop {
        n += 1.0;
        c *= -x / n;
        let term = c / (alph + n);
        sum += term;
        if term.abs() <= DBL_EPSILON * sum.abs() {
            break;
        }
    }

    if lower_tail {
        let f1 = if log_p { sum.ln_1p() } else { 1.0 + sum };
        let f2 = if alph > 1.0 {
            let f2 = dpois_raw(alph, x, log_p);
            if log_p { f2 + x } else { f2 * x.exp() }
        } else if log_p {
            alph * x.ln() - lgamma1p(alph)
        } else {
            x.powf(alph) / lgamma1p(alph).exp()
        };
        if log_p { f1 + f2 } else { f1 * f2 }
    } else {
        let lf2 = alph * x.ln() - lgamma1p(alph);
        if log_p {
            r_log1_exp(sum.ln_1p() + lf2)
        } else {
            let f1m1 = sum;
            let f2m1 = lf2.exp_m1();
            -(f1m1 + f2m1 + f1m1 * f2m1)
        }
    }
}

/// `sum_{n>=1} x^n / (y (y+1) ... (y+n-1))` (R's `pd_upper_series`).
fn pd_upper_series(x: f64, y: f64, log_p: bool) -> f64 {
    let mut y = y;
    let mut term = x / y;
    let mut sum = term;
    loop {
        y += 1.0;
        term *= x / y;
        sum += term;
        if term <= sum * DBL_EPSILON {
            break;
        }
    }
    if log_p { sum.ln() } else { sum }
}

/// Continued fraction for the scaled upper-tail gamma integral
/// `~ (y/d) [1 + (1-y)/d + O(((1-y)/d)^2)]` (R's `pd_lower_cf`).
fn pd_lower_cf(y: f64, d: f64) -> f64 {
    const MAX_IT: usize = 200_000;

    if y == 0.0 {
        return 0.0;
    }
    let mut f0 = y / d;
    // Needed, e.g. for pgamma(10^c(100,295), shape = 1.1, log = TRUE)
    if (y - 1.0).abs() < d.abs() * DBL_EPSILON {
        return f0;
    }
    if f0 > 1.0 {
        f0 = 1.0;
    }
    let mut c2 = y;
    let mut c4 = d; // original (y, d), *not* potentially scaled ones
    let mut a1 = 0.0;
    let mut b1 = 1.0;
    let mut a2 = y;
    let mut b2 = d;

    while b2 > SCALEFACTOR {
        a1 /= SCALEFACTOR;
        b1 /= SCALEFACTOR;
        a2 /= SCALEFACTOR;
        b2 /= SCALEFACTOR;
    }

    let mut i = 0.0;
    let mut of = -1.0; // far away
    let mut f = 0.0;
    let mut it = 0;
    while it < MAX_IT {
        it += 2;
        i += 1.0;
        c2 -= 1.0;
        let mut c3 = i * c2;
        c4 += 2.0;
        // c2 = y - i,  c3 = i(y - i),  c4 = d + 2i,  for i odd
        a1 = c4 * a2 + c3 * a1;
        b1 = c4 * b2 + c3 * b1;

        i += 1.0;
        c2 -= 1.0;
        c3 = i * c2;
        c4 += 2.0;
        // for i even
        a2 = c4 * a1 + c3 * a2;
        b2 = c4 * b1 + c3 * b2;

        if b2 > SCALEFACTOR {
            a1 /= SCALEFACTOR;
            b1 /= SCALEFACTOR;
            a2 /= SCALEFACTOR;
            b2 /= SCALEFACTOR;
        }

        if b2 != 0.0 {
            f = a2 / b2;
            // convergence check: relative; "absolute" for very small f
            if (f - of).abs() <= DBL_EPSILON * f0.max(f.abs()) {
                return f;
            }
            of = f;
        }
    }
    f // non-convergence; should not happen
}

/// `sum_{n>=0} y (y-1) ... (y-n) / lambda^(n+1)` (R's `pd_lower_series`).
fn pd_lower_series(lambda: f64, y: f64) -> f64 {
    let mut y = y;
    let mut term = 1.0;
    let mut sum = 0.0;
    while y >= 1.0 && term > sum * DBL_EPSILON {
        term *= y / lambda;
        sum += term;
        y -= 1.0;
    }
    if y != y.floor() {
        // The series does not converge as the terms start getting bigger
        // (besides flipping sign) for y < -lambda.
        let f = pd_lower_cf(y, lambda + 1.0 - y);
        sum += term * f;
    }
    sum
}

/// `dnorm(x) / pnorm(x, lower_tail)` computed accurately (R's `dpnorm`);
/// `lp` must be `pnorm(x, lower_tail, log_p = TRUE)`.
fn dpnorm(x: f64, lower_tail: bool, lp: f64) -> f64 {
    let (x, lower_tail) = if x < 0.0 {
        (-x, !lower_tail)
    } else {
        (x, lower_tail)
    };
    if x > 10.0 && !lower_tail {
        // Abramowitz & Stegun 26.2.12
        let mut term = 1.0 / x;
        let mut sum = term;
        let x2 = x * x;
        let mut i = 1.0;
        loop {
            term *= -i / x2;
            sum += term;
            i += 2.0;
            if term.abs() <= DBL_EPSILON * sum {
                break;
            }
        }
        1.0 / sum
    } else {
        dnorm(x, false) / lp.exp()
    }
}

/// Asymptotic expansion for `P[Poisson(lambda) <= x]` when `x` and `lambda`
/// are both large and close (R's `ppois_asymp`).
fn ppois_asymp(x: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    const COEFS_A: [f64; 8] = [
        -1e99, // placeholder used for 1-indexing
        2.0 / 3.0,
        -4.0 / 135.0,
        8.0 / 2835.0,
        16.0 / 8505.0,
        -8992.0 / 12629925.0,
        -334144.0 / 492567075.0,
        698752.0 / 1477701225.0,
    ];
    const COEFS_B: [f64; 8] = [
        -1e99, // placeholder
        1.0 / 12.0,
        1.0 / 288.0,
        -139.0 / 51840.0,
        -571.0 / 2488320.0,
        163879.0 / 209018880.0,
        5246819.0 / 75246796800.0,
        -534703531.0 / 902961561600.0,
    ];

    let dfm = lambda - x;
    // If lambda is large, the distribution is highly concentrated about
    // lambda, so representation error in x or lambda can lead to
    // arbitrarily large values of pt_.
    let pt_ = -log1pmx(dfm / x);
    let mut s2pt = (2.0 * x * pt_).sqrt();
    if dfm < 0.0 {
        s2pt = -s2pt;
    }

    let mut res12 = 0.0;
    let mut res1_term = x.sqrt();
    let mut res1_ig = res1_term;
    let mut res2_term = s2pt;
    let mut res2_ig = res2_term;
    for i in 1..8 {
        res12 += res1_ig * COEFS_A[i];
        res12 += res2_ig * COEFS_B[i];
        res1_term *= pt_ / i as f64;
        res2_term *= 2.0 * pt_ / (2 * i + 1) as f64;
        res1_ig = res1_ig / x + res1_term;
        res2_ig = res2_ig / x + res2_term;
    }

    let mut elfb = x;
    let mut elfb_term = 1.0;
    for coef in COEFS_B.iter().skip(1) {
        elfb += elfb_term * coef;
        elfb_term /= x;
    }
    if !lower_tail {
        elfb = -elfb;
    }
    let f = res12 / elfb;

    let np = pnorm(s2pt, !lower_tail, log_p);
    if log_p {
        let n_d_over_p = dpnorm(s2pt, !lower_tail, np);
        np + (f * n_d_over_p).ln_1p()
    } else {
        let nd = dnorm(s2pt, false);
        np + f * nd
    }
}

/// R's `pgamma_raw(x, alph, lower_tail, log_p)` for scale 1: Morten
/// Welinder's implementation choosing between a small-`x` series, the upper
/// series, the lower continued fraction and a Poisson asymptotic expansion.
fn pgamma_raw(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x <= 0.0 {
        return r_dt_0(lower_tail, log_p);
    }
    if x >= f64::INFINITY {
        return r_dt_1(lower_tail, log_p);
    }

    let res = if x < 1.0 {
        pgamma_smallx(x, alph, lower_tail, log_p)
    } else if x <= alph - 1.0 && x < 0.8 * (alph + 50.0) {
        // incl. large alph compared to x
        let sum = pd_upper_series(x, alph, log_p); // = x/alph + o(x/alph)
        let d = dpois_wrap(alph, x, log_p);
        if !lower_tail {
            if log_p {
                r_log1_exp(d + sum)
            } else {
                1.0 - d * sum
            }
        } else if log_p {
            sum + d
        } else {
            sum * d
        }
    } else if alph - 1.0 < x && alph < 0.8 * (x + 50.0) {
        // incl. large x compared to alph
        let d = dpois_wrap(alph, x, log_p);
        let sum = if alph < 1.0 {
            if x * DBL_EPSILON > 1.0 - alph {
                r_d_1(log_p)
            } else {
                let f = pd_lower_cf(alph, x - (alph - 1.0)) * x / alph;
                // = [alph/(x - alph+1) + o(alph/(x-alph+1))] * x/alph = 1 + o(1)
                if log_p { f.ln() } else { f }
            }
        } else {
            let sum = pd_lower_series(x, alph - 1.0); // = (alph-1)/x + o((alph-1)/x)
            if log_p { sum.ln_1p() } else { 1.0 + sum }
        };
        if !lower_tail {
            if log_p { sum + d } else { sum * d }
        } else if log_p {
            r_log1_exp(d + sum)
        } else {
            1.0 - d * sum
        }
    } else {
        // x >= 1 and x fairly near alph.
        ppois_asymp(alph - 1.0, x, !lower_tail, log_p)
    };

    // We lose a fair amount of accuracy to underflow in the cases where the
    // final result is very close to DBL_MIN; redo those via log space.
    if !log_p && res < DBL_MIN / DBL_EPSILON {
        pgamma_raw(x, alph, lower_tail, true).exp()
    } else {
        res
    }
}

/// R's `pgamma(x, shape, scale, lower.tail, log.p)`.
pub(crate) fn pgamma(x: f64, shape: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return f64::NAN;
    }
    if shape < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    let x = x / scale;
    if x.is_nan() {
        return x;
    }
    if shape == 0.0 {
        // limit case; useful e.g. in pnchisq()
        return if x <= 0.0 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    pgamma_raw(x, shape, lower_tail, log_p)
}

/// R's `dgamma(x, shape, scale, log)`.
pub(crate) fn dgamma(x: f64, shape: f64, scale: f64, give_log: bool) -> f64 {
    if x.is_nan() || shape.is_nan() || scale.is_nan() {
        return f64::NAN;
    }
    if shape < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return r_d_0(give_log);
    }
    if shape == 0.0 {
        // point mass at 0
        return if x == 0.0 {
            f64::INFINITY
        } else {
            r_d_0(give_log)
        };
    }
    if x == 0.0 {
        if shape < 1.0 {
            return f64::INFINITY;
        }
        if shape > 1.0 {
            return r_d_0(give_log);
        }
        return if give_log { -scale.ln() } else { 1.0 / scale };
    }
    if shape < 1.0 {
        let pr = dpois_raw(shape, x / scale, give_log);
        return if give_log {
            pr + if (shape / x).is_finite() {
                (shape / x).ln()
            } else {
                shape.ln() - x.ln()
            }
        } else {
            pr * shape / x
        };
    }
    let pr = dpois_raw(shape - 1.0, x / scale, give_log);
    if give_log {
        pr - scale.ln()
    } else {
        pr / scale
    }
}

/// Starting approximation for `qgamma` (AS 91 / Wilson-Hilferty), R's
/// `qchisq_appr(p, nu, g = lgamma(nu/2), lower_tail, log_p, tol)`.
fn qchisq_appr(p: f64, nu: f64, g: f64, lower_tail: bool, log_p: bool, tol: f64) -> f64 {
    const C7: f64 = 4.67;
    const C8: f64 = 6.66;
    const C9: f64 = 6.73;
    const C10: f64 = 13.32;

    if p.is_nan() || nu.is_nan() {
        return f64::NAN;
    }
    if (log_p && p > 0.0) || (!log_p && !(0.0..=1.0).contains(&p)) {
        return f64::NAN;
    }
    if nu <= 0.0 {
        return f64::NAN;
    }

    let alpha = 0.5 * nu; // = [pq]gamma() shape
    let c = alpha - 1.0;

    let p1 = r_dt_log(p, lower_tail, log_p);
    if nu < -1.24 * p1 {
        // for small chi-squared: log(alpha) + g = lgamma(alpha + 1) suffers
        // from catastrophic cancellation when alpha << 1
        let lgam1pa = if alpha < 0.5 {
            lgamma1p(alpha)
        } else {
            alpha.ln() + g
        };
        ((lgam1pa + p1) / alpha + LN_2).exp()
    } else if nu > 0.32 {
        // Wilson and Hilferty estimate
        let x = qnorm(p, lower_tail, log_p);
        let p1 = 2.0 / (9.0 * nu);
        let mut ch = nu * (x * p1.sqrt() + 1.0 - p1).powi(3);
        // approximation for p tending to 1:
        if ch > 2.2 * nu + 6.0 {
            ch = -2.0 * (r_dt_clog(p, lower_tail, log_p) - c * (0.5 * ch).ln() + g);
        }
        ch
    } else {
        // "small nu": 1.24*(-log(p)) <= nu <= 0.32
        let mut ch = 0.4;
        let a = r_dt_clog(p, lower_tail, log_p) + g + c * LN_2;
        loop {
            let q = ch;
            let p1 = 1.0 / (1.0 + ch * (C7 + ch));
            let p2 = ch * (C9 + ch * (C8 + ch));
            let t = -0.5 + (C7 + 2.0 * ch) * p1 - (C9 + ch * (C10 + 3.0 * ch)) / p2;
            ch -= (1.0 - (a + 0.5 * ch).exp() * p2 * p1) / t;
            if (q - ch).abs() <= tol * ch.abs() {
                break;
            }
        }
        ch
    }
}

/// R's `qgamma(p, shape, scale, lower.tail, log.p)`: AS 91 (Best & Roberts
/// 1975) seven-term Taylor iteration from `qchisq_appr`, followed by Newton
/// steps on the log scale for full double precision.
pub(crate) fn qgamma(p: f64, shape: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    const EPS1: f64 = 1e-2;
    const EPS2: f64 = 5e-7; // final precision of AS 91
    const EPS_N: f64 = 1e-15; // precision of Newton step / iterations
    const MAXIT: usize = 1000;
    const P_MIN: f64 = 1e-100;
    const P_MAX: f64 = 1.0 - 1e-14;
    const I420: f64 = 1.0 / 420.0;
    const I2520: f64 = 1.0 / 2520.0;
    const I5040: f64 = 1.0 / 5040.0;

    if p.is_nan() || shape.is_nan() || scale.is_nan() {
        return f64::NAN;
    }
    if let Some(bound) = q_p01_boundaries(p, 0.0, f64::INFINITY, lower_tail, log_p) {
        return bound;
    }
    if shape < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if shape == 0.0 {
        // all mass at 0
        return 0.0;
    }

    let mut max_it_newton = if shape < 1e-10 { 7 } else { 1 };
    let mut p_ = r_dt_qiv(p, lower_tail, log_p); // lower_tail prob (in any case)
    let g = lgammafn(shape); // log Gamma(v/2)

    // Phase I: starting approximation
    let mut ch = qchisq_appr(p, 2.0 * shape, g, lower_tail, log_p, EPS1);
    let skip_iteration = if !ch.is_finite() {
        max_it_newton = 0;
        true
    } else if ch < EPS2 || !(P_MIN..=P_MAX).contains(&p_) {
        // Corrected according to AS 91; do Newton steps only.
        max_it_newton = 20;
        true
    } else {
        false
    };

    if !skip_iteration {
        // Phase II: iteration — call pgamma() [AS 239] and calculate a
        // seven-term Taylor series
        let c = shape - 1.0;
        let s6 = (120.0 + c * (346.0 + 127.0 * c)) * I5040;
        let ch0 = ch; // save initial approx.
        for _ in 1..=MAXIT {
            let q = ch;
            let p1 = 0.5 * ch;
            let p2 = p_ - pgamma_raw(p1, shape, true, false);
            if !p2.is_finite() || ch <= 0.0 {
                ch = ch0;
                max_it_newton = 27;
                break;
            }
            let t = p2 * (shape * LN_2 + g + p1 - c * ch.ln()).exp();
            let b = t / ch;
            let a = 0.5 * t - b * c;
            let s1 =
                (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) * I420;
            let s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) * I2520;
            let s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) * I2520;
            let s4 =
                (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a))) * I5040;
            let s5 = (84.0 + 2264.0 * a + c * (1175.0 + 606.0 * a)) * I2520;

            ch += t
                * (1.0 + 0.5 * t * s1
                    - b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
            if (q - ch).abs() < EPS2 * ch {
                break;
            }
            if (q - ch).abs() > 0.1 * ch {
                // diverging? -- also forces ch > 0
                ch = if ch < q { 0.9 * q } else { 1.1 * q };
            }
        }
    }

    // Final Newton step(s) on the log scale (PR#2214, Morten Welinder).
    let mut x = 0.5 * scale * ch;
    if max_it_newton > 0 {
        let p = if log_p { p } else { p.ln() };
        if x == 0.0 {
            let one_p = 1.0 + 1e-7;
            let one_m = 1.0 - 1e-7;
            x = DBL_MIN;
            p_ = pgamma(x, shape, scale, lower_tail, true);
            if (lower_tail && p_ > p * one_p) || (!lower_tail && p_ < p * one_m) {
                return 0.0;
            }
            // else: continue, using x = DBL_MIN instead of 0
        } else {
            p_ = pgamma(x, shape, scale, lower_tail, true);
        }
        if p_ == f64::NEG_INFINITY {
            return 0.0; // PR#14710
        }
        for i in 1..=max_it_newton {
            let p1 = p_ - p;
            if p1.abs() < (EPS_N * p).abs() {
                break;
            }
            let g = dgamma(x, shape, scale, true);
            if g == f64::NEG_INFINITY {
                break;
            }
            // delta x = f(x)/f'(x) with f(x) = log P(x) - p, f'(x) = P'/P
            let t = p1 * (p_ - g).exp();
            let t = if lower_tail { x - t } else { x + t };
            p_ = pgamma(t, shape, scale, lower_tail, true);
            if (p_ - p).abs() > p1.abs() || (i > 1 && (p_ - p).abs() == p1.abs()) {
                // no improvement (the second clause guards against flip-flop)
                break;
            }
            x = t;
        }
    }
    x
}

/// R's `pchisq(x, df, lower.tail, log.p)` = `pgamma(x, df/2, 2)`.
#[inline]
pub(crate) fn pchisq(x: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    pgamma(x, df / 2.0, 2.0, lower_tail, log_p)
}

/// R's `qchisq(p, df, lower.tail, log.p)` = `qgamma(p, df/2, 2)`.
#[allow(dead_code)] // part of the R distribution surface; exercised by the tests
#[inline]
pub(crate) fn qchisq(p: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    qgamma(p, 0.5 * df, 2.0, lower_tail, log_p)
}

// ---------------------------------------------------------------------------
// Beta and Student t.
// ---------------------------------------------------------------------------

/// R's `lbeta(a, b)` = `log(beta(a, b))`, using `lgammacor` differences for
/// large arguments to avoid cancellation (as in `nmath/lbeta.c`).
pub(crate) fn lbeta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let p = a.min(b);
    let q = a.max(b);
    if p < 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if !q.is_finite() {
        return f64::NEG_INFINITY;
    }
    if p >= 10.0 {
        // p and q are big.
        let corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
        q.ln() * -0.5
            + M_LN_SQRT_2PI
            + corr
            + (p - 0.5) * (p / (p + q)).ln()
            + q * (-p / (p + q)).ln_1p()
    } else if q >= 10.0 {
        // p is small, but q is big.
        let corr = lgammacor(q) - lgammacor(p + q);
        lgammafn(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p()
    } else if p < 1e-306 {
        lgammafn(p) + (lgammafn(q) - lgammafn(p + q))
    } else {
        // p and q are small: p <= q < 10.
        (gammafn(p) * (gammafn(q) / gammafn(p + q))).ln()
    }
}

/// `log(x^a y^b / beta(a, b))` with `y = 1 - x`, evaluated as in TOMS 708's
/// `brcomp`: for `a, b >= 10` via the relative deviances `log1pmx` and the
/// Stirling corrections so no large terms cancel.
fn log_beta_front(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if a.min(b) >= 10.0 {
        let (x0, y0, lambda) = if a <= b {
            let h = a / b;
            (h / (h + 1.0), 1.0 / (h + 1.0), a - (a + b) * x)
        } else {
            let h = b / a;
            (1.0 / (h + 1.0), h / (h + 1.0), (a + b) * y - b)
        };
        let e = -lambda / a;
        let u = if e.abs() > 0.6 {
            e - (x / x0).ln()
        } else {
            -log1pmx(e)
        };
        let e = lambda / b;
        let v = if e.abs() <= 0.6 {
            -log1pmx(e)
        } else {
            e - (y / y0).ln()
        };
        let bcorr = lgammacor(a) + lgammacor(b) - lgammacor(a + b);
        -M_LN_SQRT_2PI + 0.5 * (b * x0).ln() - (a * u + b * v) - bcorr
    } else {
        let (lnx, lny) = if x <= 0.375 {
            (x.ln(), (-x).ln_1p())
        } else if y > 0.375 {
            (x.ln(), y.ln())
        } else {
            ((-y).ln_1p(), y.ln())
        };
        a * lnx + b * lny - lbeta(a, b)
    }
}

/// Continued fraction for `I_x(a, b) / (x^a y^b / B(a, b))`, in the form of
/// TOMS 708's `bfrac` (Didonato & Morris 1992).  It is written in terms of
/// `y = 1 - x` and `lambda = (a + b) y - b >= 0`, so no `1 - x` cancellation
/// occurs when `x` is close to 1 and `a` is large.
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> f64 {
    const MAX_IT: usize = 100_000;

    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;

    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;

    for _ in 0..MAX_IT {
        n += 1.0;
        let mut t = n / a;
        let w = n * (b - n) * x;
        let mut e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        e = (t + 1.0) / (c1 + t + t);
        let beta = n + w / s + e * (c + n * yp1);
        p = t + 1.0;
        s += 2.0;

        // update an, bn, anp1, and bnp1
        t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;

        let r0 = r;
        r = anp1 / bnp1;
        if (r - r0).abs() <= DBL_EPSILON * r {
            break;
        }

        // rescale an, bn, anp1, and bnp1
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    r
}

/// R's `pbeta(x, a, b, lower.tail, log.p)` (regularized incomplete beta
/// function), including R's limit cases for `a == 0`, `b == 0`, infinities.
/// As in TOMS 708's `bratio`, the tail below the mean `a/(a+b)` is evaluated
/// directly (continued fraction times the `brcomp` prefactor) and the other
/// tail by complement, so both tails keep relative accuracy.
pub(crate) fn pbeta(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    if x >= 1.0 {
        return r_dt_1(lower_tail, log_p);
    }
    if a == 0.0 || b == 0.0 || !a.is_finite() || !b.is_finite() {
        // NB: 0 <= x < 1
        if a == 0.0 && b == 0.0 {
            // point mass 1/2 at each of {0,1}
            return if log_p { -LN_2 } else { 0.5 };
        }
        if a == 0.0 || a / b == 0.0 {
            // point mass 1 at 0 ==> P(X <= x) = 1, all x >= 0
            return r_dt_1(lower_tail, log_p);
        }
        if b == 0.0 || b / a == 0.0 {
            // point mass 1 at 1 ==> P(X <= x) = 0, all x < 1
            return r_dt_0(lower_tail, log_p);
        }
        // remaining case: a = b = Inf: point mass 1 at 1/2
        return if x < 0.5 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    if x <= 0.0 {
        return r_dt_0(lower_tail, log_p);
    }
    // Now: 0 < a < Inf; 0 < b < Inf and 0 < x < 1
    let y = 0.5 - x + 0.5;
    // lambda = a - (a+b) x = (a+b) y - b, computed the way that does not cancel.
    let lambda = if a > b {
        (a + b) * y - b
    } else {
        a - (a + b) * x
    };
    let swap = lambda < 0.0;
    let (aa, bb, xx, yy, lambda) = if swap {
        (b, a, y, x, -lambda)
    } else {
        (a, b, x, y, lambda)
    };
    let log_direct = log_beta_front(aa, bb, xx, yy) + bfrac(aa, bb, xx, yy, lambda).ln();
    // `log_direct` is log of the lower tail when !swap, of the upper tail when swap.
    if lower_tail != swap {
        r_d_exp(log_direct, log_p)
    } else if log_p {
        r_log1_exp(log_direct)
    } else {
        -log_direct.exp_m1()
    }
}

/// R's `dt(x, n, log)` (Catherine Loader's saddle-point form), as in
/// `nmath/dt.c`.
pub(crate) fn dt(x: f64, n: f64, give_log: bool) -> f64 {
    if x.is_nan() || n.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return r_d_0(give_log);
    }
    if !n.is_finite() {
        return dnorm(x, give_log);
    }

    let t = -bd0(n / 2.0, (n + 1.0) / 2.0) + stirlerr((n + 1.0) / 2.0) - stirlerr(n / 2.0);
    let x2n = x * x / n; // in [0, Inf]
    let lrg_x2n = x2n > 1.0 / DBL_EPSILON;
    let mut ax = 0.0;
    let (l_x2n, u) = if lrg_x2n {
        // large x^2/n
        ax = x.abs();
        let l_x2n = ax.ln() - n.ln() / 2.0; // = log(x2n)/2
        (l_x2n, n * l_x2n)
    } else if x2n > 0.2 {
        let l_x2n = (1.0 + x2n).ln() / 2.0;
        (l_x2n, n * l_x2n)
    } else {
        (
            x2n.ln_1p() / 2.0,
            -bd0(n / 2.0, (n + x * x) / 2.0) + x * x / 2.0,
        )
    };

    if give_log {
        return t - u - (M_LN_SQRT_2PI + l_x2n);
    }
    let i_sqrt = if lrg_x2n {
        n.sqrt() / ax
    } else {
        (-l_x2n).exp()
    };
    (t - u).exp() * M_1_SQRT_2PI * i_sqrt
}

/// R's `pt(x, n, lower.tail, log.p)` via `pbeta`, as in `nmath/pt.c`.
pub(crate) fn pt(x: f64, n: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || n.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return if x < 0.0 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    if !n.is_finite() {
        return pnorm(x, lower_tail, log_p);
    }

    let nx = 1.0 + (x / n) * x;
    let val = if nx > 1e100 {
        // x*x > 1e100 * n: danger of underflow, use Abramowitz & Stegun 26.5.4
        let lval = -0.5 * n * (2.0 * x.abs().ln() - n.ln()) - lbeta(0.5 * n, 0.5) - (0.5 * n).ln();
        if log_p { lval } else { lval.exp() }
    } else if n > x * x {
        pbeta(x * x / (n + x * x), 0.5, n / 2.0, false, log_p)
    } else {
        pbeta(1.0 / nx, n / 2.0, 0.5, true, log_p)
    };

    // Use "1 - v" if lower_tail and x > 0 (but not both):
    let lower_tail = if x <= 0.0 { !lower_tail } else { lower_tail };
    if log_p {
        if lower_tail {
            (-0.5 * val.exp()).ln_1p()
        } else {
            val - LN_2 // = log(.5 * pbeta(....))
        }
    } else {
        let val = val / 2.0;
        r_d_cval(val, lower_tail)
    }
}

/// R's `qt(p, ndf, lower.tail, log.p)`: Hill's (1970) algorithm 396 with
/// the Taylor-series improvement of Hill (1981), closed forms for `df` 1 and
/// 2, and bisection for `df < 1`, as in `nmath/qt.c`.
pub(crate) fn qt(p: f64, ndf: f64, lower_tail: bool, log_p: bool) -> f64 {
    const EPS: f64 = 1e-12;

    if p.is_nan() || ndf.is_nan() {
        return f64::NAN;
    }
    if let Some(bound) = q_p01_boundaries(p, f64::NEG_INFINITY, f64::INFINITY, lower_tail, log_p) {
        return bound;
    }
    if ndf <= 0.0 {
        return f64::NAN;
    }

    if ndf < 1.0 {
        // based on qnt: bracket then bisect pt()
        const ACCU: f64 = 1e-13;
        const EPS_BRACKET: f64 = 1e-11; // must be > ACCU
        let p = r_dt_qiv(p, lower_tail, log_p);
        if p > 1.0 - DBL_EPSILON {
            return f64::INFINITY;
        }
        let pp = (1.0 - DBL_EPSILON).min(p * (1.0 + EPS_BRACKET));
        let mut ux = 1.0;
        while ux < f64::MAX && pt(ux, ndf, true, false) < pp {
            ux *= 2.0;
        }
        let pp = p * (1.0 - EPS_BRACKET);
        let mut lx = -1.0;
        while lx > -f64::MAX && pt(lx, ndf, true, false) > pp {
            lx *= 2.0;
        }
        let mut iter = 0;
        loop {
            let nx = 0.5 * (lx + ux);
            if pt(nx, ndf, true, false) > p {
                ux = nx;
            } else {
                lx = nx;
            }
            iter += 1;
            if (ux - lx) / nx.abs() <= ACCU || iter >= 1000 {
                break;
            }
        }
        return 0.5 * (lx + ux);
    }

    if ndf > 1e20 {
        return qnorm(p, lower_tail, log_p);
    }

    let mut big_p = if log_p { p.exp() } else { p }; // if exp(p) underflows, we fix below
    let neg = (!lower_tail || big_p < 0.5) && (lower_tail || big_p > 0.5);
    let is_neg_lower = lower_tail == neg; // both TRUE or FALSE == !xor
    big_p = if neg {
        2.0 * if log_p {
            if lower_tail { big_p } else { -p.exp_m1() }
        } else {
            r_d_lval(p, lower_tail)
        }
    } else {
        2.0 * if log_p {
            if lower_tail { -p.exp_m1() } else { big_p }
        } else {
            r_d_cval(p, lower_tail)
        }
    };
    // 0 <= P <= 1 ; P = 2*min(P', 1 - P')  in all cases

    let q = if (ndf - 2.0).abs() < EPS {
        // df ~= 2
        if big_p > DBL_MIN {
            if 3.0 * big_p < DBL_EPSILON {
                1.0 / big_p.sqrt()
            } else if big_p > 0.9 {
                (1.0 - big_p) * (2.0 / (big_p * (2.0 - big_p))).sqrt()
            } else {
                (2.0 / (big_p * (2.0 - big_p)) - 2.0).sqrt()
            }
        } else if log_p {
            // P << 1, q = 1/sqrt(P)
            if is_neg_lower {
                (-p / 2.0).exp() / SQRT_2
            } else {
                1.0 / (-p.exp_m1()).sqrt()
            }
        } else {
            f64::INFINITY
        }
    } else if ndf < 1.0 + EPS {
        // df ~= 1 (df < 1 excluded above): Cauchy
        if big_p == 1.0 {
            0.0
        } else if big_p > 0.0 {
            1.0 / tanpi(big_p / 2.0) // == - tan((P+1) * M_PI_2) -- suffers for P ~= 0
        } else if log_p {
            // P = 0, but maybe = 2*exp(p): 1/tan(e) ~ 1/e
            if is_neg_lower {
                FRAC_1_PI * (-p).exp()
            } else {
                -1.0 / (PI * p.exp_m1())
            }
        } else {
            f64::INFINITY
        }
    } else {
        // usual case; including, e.g., df = 1.1
        let mut x = 0.0;
        let mut log_p2 = 0.0;
        let a = 1.0 / (ndf - 0.5);
        let b = 48.0 / (a * a);
        let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * FRAC_PI_2).sqrt() * ndf;

        let p_ok1 = big_p > DBL_MIN || !log_p;
        let mut p_ok = p_ok1; // when true, use "normal scale": log_p = FALSE
        let mut y = 0.0;
        if p_ok1 {
            y = (d * big_p).powf(2.0 / ndf);
            p_ok = y >= DBL_EPSILON;
        }
        if !p_ok {
            // log.p && P very small  ||  (d*P)^(2/df) =: y < eps_c
            log_p2 = if is_neg_lower {
                r_d_log(p, log_p)
            } else {
                r_d_lexp(p, log_p)
            }; // == log(P / 2)
            x = (d.ln() + LN_2 + log_p2) / ndf;
            y = (2.0 * x).exp();
        }

        let mut q = if (ndf < 2.1 && big_p > 0.5) || y > 0.05 + a {
            // P > P0(df): asymptotic inverse expansion about normal
            x = if p_ok {
                qnorm(0.5 * big_p, true, false)
            } else {
                // log_p && P underflowed
                qnorm(log_p2, lower_tail, true)
            };
            y = x * x;
            if ndf < 5.0 {
                c += 0.3 * (ndf - 4.5) * (x + 0.6);
            }
            c += (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = (a * y * y).exp_m1();
            (ndf * y).sqrt()
        } else if !p_ok && x < -LN_2 * f64::MANTISSA_DIGITS as f64 {
            // 0.5 * log(DBL_EPSILON): y above might have underflown
            ndf.sqrt() * (-x).exp()
        } else {
            // re-use 'y' from above
            y = ((1.0 / (((ndf + 6.0) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2.0) * 3.0)
                + 0.5 / (ndf + 4.0))
                * y
                - 1.0)
                * (ndf + 1.0)
                / (ndf + 2.0)
                + 1.0 / y;
            (ndf * y).sqrt()
        };

        // Now apply 2-term Taylor expansion improvement (1-term = Newton):
        // as by Hill (1981)
        if p_ok1 {
            let big_m = ((f64::MAX / 2.0).sqrt() - ndf).abs();
            let mut it = 0;
            loop {
                it += 1;
                if it > 10 {
                    break;
                }
                let y = dt(q, ndf, false);
                if y <= 0.0 {
                    break;
                }
                let x = (pt(q, ndf, false, false) - big_p / 2.0) / y;
                if !x.is_finite() || x.abs() <= 1e-14 * q.abs() {
                    break;
                }
                let f = if q.abs() < big_m {
                    q * (ndf + 1.0) / (2.0 * (q * q + ndf))
                } else {
                    (ndf + 1.0) / (2.0 * (q + ndf / q))
                };
                let del_q = x * (1.0 + x * f);
                if del_q.is_finite() && (q + del_q).is_finite() {
                    q += del_q;
                } else if x.is_finite() && (q + x).is_finite() {
                    q += x;
                } else {
                    break; // cannot improve q with a Newton/Taylor step
                }
            }
        }
        q
    };
    if neg { -q } else { q }
}

#[cfg(test)]
mod tests {
    // The reference tables below are R's own 17-digit output.
    #![allow(clippy::approx_constant, clippy::excessive_precision)]

    use super::*;

    /// Relative tolerance against R for ordinary values.
    const TOL: f64 = 1e-13;
    /// Relative tolerance in the far tails (values below 1e-100 or log
    /// probabilities below -230), where the last digits depend on the
    /// single-double deviance term `bd0` (R >= 4.1 uses a double-double one).
    const TAIL_TOL: f64 = 1e-12;

    fn assert_matches_r(what: &str, actual: f64, expected: f64) {
        if expected == 0.0 || !expected.is_finite() {
            assert!(
                actual == expected || (actual.is_nan() && expected.is_nan()),
                "{what}: got {actual:e}, R gives {expected:e}"
            );
            return;
        }
        let tol = if expected.abs() < 1e-100 || expected < -230.0 {
            TAIL_TOL
        } else {
            TOL
        };
        let rel = ((actual - expected) / expected).abs();
        // The absolute fallback only matters for quantiles that are
        // mathematically zero (e.g. qt(0.5, df)), where R returns noise too.
        assert!(
            rel <= tol || (actual - expected).abs() <= 1e-15,
            "{what}: got {actual:.17e}, R gives {expected:.17e} (rel err {rel:.2e})"
        );
    }

    // pnorm: (x, lower, upper, log_lower, log_upper)
    #[rustfmt::skip]
    const PNORM_REF: &[(f64, f64, f64, f64, f64)] = &[
        (-40.0, 0.0, 1.0, -804.6084420137538, -0.0),
        (-38.460000000000001, 4.9406564584124654e-324, 1.0, -744.15503218861841, -4.9406564584124654e-324),
        (-37.0, 5.7255712225245771e-300, 1.0, -689.03058557689064, -5.7255712225245771e-300),
        (-20.0, 2.7536241186062337e-89, 1.0, -203.91715537109727, -2.7536241186062337e-89),
        (-10.0, 7.6198530241605269e-24, 1.0, -53.23128515051247, -7.6198530241605269e-24),
        (-8.3000000000000007, 5.2055697448902552e-17, 1.0, -37.494217423748253, -5.2055697448902552e-17),
        (-5.657, 7.7020881721775998e-09, 0.99999999229791181, -18.681774353660561, -7.7020882018386803e-09),
        (-3.0, 0.0013498980316300946, 0.9986501019683699, -6.6077262215103492, -0.0013508099647481938),
        (-1.0, 0.15865525393145705, 0.84134474606854293, -1.8410216450092636, -0.17275377902344988),
        (-0.68000000000000005, 0.24825223045357053, 0.75174776954642941, -1.3933099913899847, -0.28535442413295908),
        (-0.10000000000000001, 0.46017216272297101, 0.53982783727702899, -0.77615459273027332, -0.61650501011502623),
        (-1e-08, 0.49999999601057721, 0.50000000398942279, -0.69314718853879087, -0.6931471725810997),
        (0.0, 0.5, 0.5, -0.69314718055994529, -0.69314718055994529),
        (0.10000000000000001, 0.53982783727702899, 0.46017216272297101, -0.61650501011502623, -0.77615459273027332),
        (0.68000000000000005, 0.75174776954642941, 0.24825223045357053, -0.28535442413295908, -1.3933099913899847),
        (1.0, 0.84134474606854293, 0.15865525393145705, -0.17275377902344988, -1.8410216450092636),
        (3.0, 0.9986501019683699, 0.0013498980316300946, -0.0013508099647481938, -6.6077262215103492),
        (5.657, 0.99999999229791181, 7.7020881721775998e-09, -7.7020882018386803e-09, -18.681774353660561),
        (8.3000000000000007, 1.0, 5.2055697448902552e-17, -5.2055697448902552e-17, -37.494217423748253),
        (10.0, 1.0, 7.6198530241605269e-24, -7.6198530241605269e-24, -53.23128515051247),
        (20.0, 1.0, 2.7536241186062337e-89, -2.7536241186062337e-89, -203.91715537109727),
        (37.0, 1.0, 5.7255712225245771e-300, -5.7255712225245771e-300, -689.03058557689064),
        (38.460000000000001, 1.0, 4.9406564584124654e-324, -4.9406564584124654e-324, -744.15503218861841),
        (40.0, 1.0, 0.0, -0.0, -804.6084420137538),
        (-10000000000.0, 0.0, 1.0, -5e19, -0.0),
        (1e170, 1.0, 0.0, 0.0, f64::NEG_INFINITY),
    ];

    // qnorm: (p, lower, upper)
    #[rustfmt::skip]
    const QNORM_REF: &[(f64, f64, f64)] = &[
        (1e-300, -37.047096299361201, 37.047096299361201),
        (1e-100, -21.273453560965319, 21.273453560965319),
        (9.9999999999999995e-21, -9.2623400897984052, 9.2623400897984052),
        (9.9999999999999998e-13, -7.0344838253011321, 7.0344838253011321),
        (1.0000000000000001e-05, -4.2648907939228256, 4.2648907939228256),
        (0.001, -3.0902323061678132, 3.0902323061678132),
        (0.025000000000000001, -1.9599639845400538, 1.9599639845400538),
        (0.050000000000000003, -1.6448536269514726, 1.6448536269514726),
        (0.074999999999999997, -1.4395314709384557, 1.4395314709384555),
        (0.10000000000000001, -1.2815515655446008, 1.2815515655446008),
        (0.25, -0.67448975019608171, 0.67448975019608171),
        (0.5, 0.0, 0.0),
        (0.75, 0.67448975019608171, -0.67448975019608171),
        (0.90000000000000002, 1.2815515655446008, -1.2815515655446008),
        (0.92500000000000004, 1.4395314709384559, -1.4395314709384559),
        (0.94999999999999996, 1.6448536269514715, -1.6448536269514715),
        (0.97499999999999998, 1.9599639845400536, -1.9599639845400536),
        (0.98999999999999999, 2.3263478740408408, -2.3263478740408408),
        (0.99999000000000005, 4.2648907939238399, -4.2648907939238399),
        (0.99999999999900002, 7.0344869100478356, -7.0344869100478356),
        (0.99999999999999989, 8.2095361516013856, -8.2095361516013856),
    ];

    // qnorm with log.p = TRUE: (log_p, lower, upper)
    #[rustfmt::skip]
    const QNORM_LOG_REF: &[(f64, f64, f64)] = &[
        (-100000.0, -447.19789367852508, 447.19789367852508),
        (-1000.0, -44.6157477319694, 44.6157477319694),
        (-750.0, -38.611574423848019, 38.611574423848019),
        (-730.0, -38.090429516011881, 38.090429516011881),
        (-700.0, -37.295079632647408, 37.295079632647408),
        (-100.0, -13.888476033003888, 13.888476033003888),
        (-10.0, -3.9139462405318932, 3.9139462405318932),
        (-1.0, -0.33747496376420238, 0.33747496376420238),
        (-0.001, 3.090380786917045, -3.090380786917045),
        (-9.9999999999999995e-21, 9.2623400897984052, -9.2623400897984052),
    ];

    // lgamma: (x, value)
    #[rustfmt::skip]
    const LGAMMA_REF: &[(f64, f64)] = &[
        (1e-300, 690.77552789821368),
        (1e-10, 23.025850929882736),
        (0.001, 6.9071788853838534),
        (0.01, 4.5994798780420219),
        (0.10000000000000001, 2.252712651734206),
        (0.29999999999999999, 1.0957979948180756),
        (0.5, 0.57236494292470008),
        (0.69999999999999996, 0.2608672465316666),
        (0.90000000000000002, 0.066376239734742881),
        (1.2, -0.085374090003315806),
        (1.5, -0.12078223763524518),
        (1.8, -0.07108387291437214),
        (2.2000000000000002, 0.096947466790638856),
        (2.5, 0.28468287047291918),
        (3.7000000000000002, 1.4280723266653881),
        (7.5, 7.5343642367587327),
        (9.9000000000000004, 12.577179904219879),
        (10.5, 13.940625219403763),
        (15.0, 25.191221182738683),
        (25.0, 54.784729398112312),
        (49.5, 142.61728282114601),
        (100.0, 359.1342053695754),
        (1000.0, 5905.2204232091808),
        (10000.0, 82099.717496442376),
        (4934720.0, 71118222.943517447),
        (100000.0, 1051287.7089736569),
        (1000000.0, 12815504.569147611),
        (1e18, 4.0446531673892823e19),
        (1.0000000000000001e300, 6.8977552789821374e302),
        (-0.5, 1.2655121234846454),
        (-2.5, -0.056243716497674033),
        (-9.5, -12.795895333554363),
        (-10.5, -15.147270590717842),
        (-100.3, -363.76620618267623),
    ];

    // gamma: (x, value)
    #[rustfmt::skip]
    const GAMMA_REF: &[(f64, f64)] = &[
        (0.001, 999.42377248459547),
        (0.10000000000000001, 9.5135076986687306),
        (0.5, 1.7724538509055161),
        (1.5, 0.88622692545275805),
        (4.2000000000000002, 7.7566895357931784),
        (9.9000000000000004, 289867.70384010958),
        (10.5, 1133278.3889487833),
        (20.0, 1.21645100408832e17),
        (50.5, 4.2904629123520087e63),
        (100.7, 2.3417900214543555e157),
        (171.0, 7.2574156153088822e306),
        (-0.5, -3.5449077018110322),
        (-9.9000000000000004, 3.5426845530808427e-06),
        (-20.5, -2.834656574391336e-19),
    ];

    // pchisq: (x, df, lower, upper, log_lower, log_upper)
    #[rustfmt::skip]
    const PCHISQ_REF: &[(f64, f64, f64, f64, f64, f64)] = &[
        (0.001, 1.0, 0.025227120630039609, 0.97477287936996038, -3.6798356476917045, -0.025550779355950615),
        (0.5, 1.0, 0.52049987781304652, 0.47950012218695348, -0.65296562567633121, -0.73501112983708439),
        (5.0, 1.0, 0.97465268132253169, 0.025347318677468304, -0.025674095731815012, -3.6750823266311876),
        (30.0, 1.0, 0.99999995679536946, 4.3204630578274968e-08, -4.3204631511595082e-08, -16.957318158128789),
        (100.0, 1.0, 1.0, 1.5239706048321054e-23, -1.5239706048320983e-23, -52.53813796995253),
        (500.0, 1.0, 1.0, 9.5053977665540927e-111, -9.5053977665541562e-111, -253.33508549913603),
        (1000.0, 1.0, 1.0, 1.7958327848007259e-219, -1.7958327848007187e-219, -503.68066650438169),
        (10000.0, 1.0, 1.0, 0.0, -0.0, -5004.8310615136461),
        (0.001, 2.0, 0.0004998750208307295, 0.99950012497916929, -7.6011524491254159, -0.00050000000000000001),
        (0.5, 2.0, 0.22119921692859512, 0.77880078307140488, -1.508691549446032, -0.25),
        (5.0, 2.0, 0.91791500137610116, 0.0820849986238988, -0.08565048374203818, -2.5),
        (30.0, 2.0, 0.99999969409767953, 3.0590232050182579e-07, -3.0590236728995019e-07, -15.0),
        (100.0, 2.0, 1.0, 1.9287498479639178e-22, -1.9287498479639178e-22, -50.0),
        (500.0, 2.0, 1.0, 2.6691902155412764e-109, -2.6691902155412764e-109, -250.0),
        (1000.0, 2.0, 1.0, 7.1245764067412855e-218, -7.1245764067412855e-218, -500.0),
        (10000.0, 2.0, 1.0, 0.0, -0.0, -5000.0),
        (0.001, 5.0, 1.6814877189706274e-09, 0.99999999831851227, -20.203586888390344, -1.6814877203843298e-09),
        (0.5, 5.0, 0.0078767067673704057, 0.9921232932326296, -4.8438453853890104, -0.0079078918874270245),
        (5.0, 5.0, 0.58411981300449201, 0.41588018699550794, -0.53764915794192458, -0.87735807223433282),
        (30.0, 5.0, 0.99998525141896155, 1.4748581038443054e-05, -1.4748689799833744e-05, -11.12436368058956),
        (100.0, 5.0, 1.0, 5.28514836094324e-20, -5.2851483609432719e-20, -44.386801168872637),
        (500.0, 5.0, 1.0, 7.9846611105628018e-106, -7.9846611105628473e-106, -241.99649751735856),
        (1000.0, 5.0, 1.0, 6.0100776879208033e-214, -6.0100776879209666e-214, -490.95977222581945),
        (10000.0, 5.0, 1.0, 0.0, -0.0, -4987.5085930983514),
        (0.001, 30.0, 2.3326354880142099e-62, 1.0, -141.91327702654129, -2.3326354880141822e-62),
        (0.5, 30.0, 5.6345587204509913e-22, 1.0, -48.92795321192267, -5.6345587204509528e-22),
        (5.0, 30.0, 6.9153138669928861e-08, 0.99999993084686134, -16.48694239056811, -6.915314106100721e-08),
        (30.0, 30.0, 0.53434629105599041, 0.46565370894400965, -0.62671116507398339, -0.76431303495651126),
        (100.0, 30.0, 0.99999999814319762, 1.8568023365102387e-09, -1.8568023382340961e-09, -20.104410002615904),
        (500.0, 30.0, 1.0, 1.2079559936813978e-86, -1.2079559936814144e-86, -197.83338832771054),
        (1000.0, 30.0, 1.0, 5.131435321572533e-191, -5.1314353215722875e-191, -438.15836735203374),
        (10000.0, 30.0, 1.0, 0.0, -0.0, -4905.9477131385011),
        (0.001, 100.0, 2.9188545552513494e-230, 1.0, -528.52338012490952, -2.9188545552512359e-230),
        (0.5, 100.0, 2.029952461864641e-95, 1.0, -218.03757145945551, -2.0299524618647152e-95),
        (5.0, 100.0, 2.2386989282288958e-46, 1.0, -105.1130194162216, -2.2386989282288965e-46),
        (30.0, 100.0, 9.0561255431481326e-13, 0.99999999999909439, -27.730164824577717, -9.0561255431522523e-13),
        (100.0, 100.0, 0.51880831547204331, 0.48119168452795669, -0.65622079838756853, -0.73148957572927076),
        (500.0, 100.0, 1.0, 1.7201210053695373e-54, -1.7201210053695388e-54, -123.79720038136865),
        (1000.0, 100.0, 1.0, 2.3060767380353981e-148, -2.3060767380354051e-148, -339.94704606427337),
        (10000.0, 100.0, 1.0, 0.0, -0.0, -4727.2134312290664),
        (0.001, 200.0, 0.0, 1.0, -1123.8301165592645, -0.0),
        (0.5, 200.0, 5.2059487067099018e-219, 1.0, -502.61633341189793, -5.2059487067097099e-219),
        (5.0, 200.0, 5.6123327362440568e-120, 1.0, -274.58524470866672, -5.612332736244001e-120),
        (30.0, 200.0, 1.5645857689298614e-47, 1.0, -107.77387826615299, -1.5645857689298612e-47),
        (100.0, 200.0, 3.2000653245851258e-10, 0.99999999967999342, -21.862679706410287, -3.2000653250971433e-10),
        (500.0, 200.0, 1.0, 1.1737017704487873e-27, -1.1737017704487925e-27, -62.009634850282595),
        (1000.0, 200.0, 1.0, 1.5008794119250894e-106, -1.5008794119251201e-106, -243.66796864643575),
        (10000.0, 200.0, 1.0, 0.0, -0.0, -4515.9120848927378),
    ];

    // qchisq: (p, df, lower, upper)
    #[rustfmt::skip]
    const QCHISQ_REF: &[(f64, f64, f64, f64)] = &[
        (1e-100, 1.0, 1.5707963267948823e-200, 453.94308223879898),
        (1e-08, 1.0, 1.5707963267948893e-16, 32.841253361236788),
        (0.001, 1.0, 1.5707971492624904e-06, 10.827566170662729),
        (0.10000000000000001, 1.0, 0.015790774093431229, 2.7055434540954155),
        (0.5, 1.0, 0.45493642311957283, 0.45493642311957283),
        (0.97499999999999998, 1.0, 5.0238861873148846, 0.00098206911717525769),
        (0.99999998999999995, 1.0, 32.841253351468858, 1.5707963425806362e-16),
        (1e-100, 2.0, 1.9999999999999815e-100, 460.51701859880916),
        (1e-08, 2.0, 2.0000000099999939e-08, 36.841361487904734),
        (0.001, 2.0, 0.002001000667167067, 13.815510557964274),
        (0.10000000000000001, 2.0, 0.21072103131565262, 4.6051701859880918),
        (0.5, 2.0, 1.3862943611198906, 1.3862943611198906),
        (0.97499999999999998, 2.0, 7.3777589082278707, 0.050635615968579795),
        (0.99999998999999995, 2.0, 36.841361477855216, 2.0000000200495157e-08),
        (1e-100, 5.0, 3.2334077805831046e-40, 476.37943706416274),
        (1e-08, 5.0, 0.0020407372249365316, 45.794587123084568),
        (0.001, 5.0, 0.21021260262921918, 20.515005652432876),
        (0.10000000000000001, 5.0, 1.6103079869623227, 9.2363568997811196),
        (0.5, 5.0, 4.3514601910955264, 4.3514601910955264),
        (0.97499999999999998, 5.0, 12.832501994030025, 0.83121161348666273),
        (0.99999998999999995, 5.0, 45.794587112362635, 0.0020407372290394115),
        (1e-100, 30.0, 2.7677700613391894e-06, 568.42757044912014),
        (1e-08, 30.0, 4.3012028318871689, 95.328831531183496),
        (0.001, 30.0, 11.587951045645056, 59.70306430442993),
        (0.10000000000000001, 30.0, 20.599234614585349, 40.256023738711797),
        (0.5, 30.0, 29.336031516661588, 29.336031516661588),
        (0.97499999999999998, 30.0, 46.979242243671152, 16.790772265566627),
        (0.99999998999999995, 30.0, 95.328831517115688, 4.3012028335494676),
        (1e-100, 200.0, 7.901597617080129, 966.43906044512391),
        (1e-08, 200.0, 107.24289708616725, 333.2597044265313),
        (0.001, 200.0, 143.84279499000078, 267.54052782275721),
        (0.10000000000000001, 200.0, 174.83527299918737, 226.02104771968897),
        (0.5, 200.0, 199.33372983863086, 199.33372983863086),
        (0.97499999999999998, 200.0, 241.05789550631096, 162.72798250184633),
        (0.99999998999999995, 200.0, 333.25970440226263, 107.24289709752081),
        (1e-100, 10000.0, 7284.7615761801671, 13316.709073059299),
        (1e-08, 10000.0, 9226.5632927022561, 10814.092977078823),
        (0.001, 10000.0, 9568.6684950939689, 10442.730565410178),
        (0.10000000000000001, 10000.0, 9819.19488184482, 10181.661613830378),
        (0.5, 10000.0, 9999.333341235144, 9999.333341235144),
        (0.97499999999999998, 10000.0, 10279.070179887591, 9724.7183773897978),
        (0.99999998999999995, 10000.0, 10814.092976949323, 9226.5632928187479),
    ];

    // pgamma (scale 1): (x, shape, lower, upper)
    #[rustfmt::skip]
    const PGAMMA_REF: &[(f64, f64, f64, f64)] = &[
        (0.0001, 0.001, 0.99140311966744343, 0.0085968803325566414),
        (0.29999999999999999, 0.001, 0.99909416191169331, 0.000905838088306647),
        (0.94999999999999996, 0.001, 0.99976102739542605, 0.0002389726045740358),
        (1.0, 0.001, 0.99978039164241439, 0.00021960835758555607),
        (1.05, 0.001, 0.9997979132836744, 0.00020208671632562228),
        (2.0, 0.001, 0.99995102308216899, 4.8976917830981505e-05),
        (5.0, 0.001, 0.99999884901866032, 1.1509813397308614e-06),
        (0.0001, 0.5, 0.011283415555849618, 0.98871658444415034),
        (0.29999999999999999, 0.5, 0.56142197391900006, 0.43857802608099994),
        (0.94999999999999996, 0.5, 0.83192168096502961, 0.16807831903497031),
        (1.0, 0.5, 0.84270079294971556, 0.15729920705028447),
        (1.05, 0.5, 0.85270086137732393, 0.14729913862267613),
        (2.0, 0.5, 0.95449973610364158, 0.045500263896358473),
        (5.0, 0.5, 0.9984345977419975, 0.0015654022580025519),
        (0.00011000000000000002, 1.1000000000000001, 4.2244830423212085e-05, 0.99995775516957686),
        (0.33000000000000002, 1.1000000000000001, 0.23848826801465098, 0.76151173198534905),
        (1.0449999999999999, 1.1000000000000001, 0.60623386730728668, 0.39376613269271332),
        (1.1000000000000001, 1.1000000000000001, 0.6261553270785416, 0.37384467292145834),
        (1.1550000000000002, 1.1000000000000001, 0.64510526506873866, 0.35489473493126134),
        (2.2000000000000002, 1.1000000000000001, 0.86962729021365281, 0.13037270978634713),
        (5.5, 1.1000000000000001, 0.99482478039747579, 0.0051752196025242588),
        (0.001, 10.0, 2.7532278594284623e-37, 1.0),
        (3.0, 10.0, 0.0011024881301154794, 0.99889751186988451),
        (9.5, 10.0, 0.47817397776279247, 0.52182602223720753),
        (10.0, 10.0, 0.54207028552814773, 0.45792971447185227),
        (10.5, 10.0, 0.60286740064918942, 0.39713259935081058),
        (20.0, 10.0, 0.99500458769169242, 0.0049954123083075872),
        (50.0, 10.0, 0.99999999999874034, 1.259608459166091e-12),
        (0.01, 100.0, 0.0, 1.0),
        (30.0, 100.0, 7.3384686328783343e-24, 1.0),
        (95.0, 100.0, 0.31735681116979975, 0.68264318883020025),
        (100.0, 100.0, 0.51329879827914882, 0.48670120172085113),
        (105.0, 100.0, 0.70024534239115632, 0.29975465760884362),
        (200.0, 100.0, 0.99999999999999811, 1.8438936497115737e-15),
        (500.0, 100.0, 1.0, 1.5008794119250894e-106),
        (10.0, 100000.0, 0.0, 1.0),
        (30000.0, 100000.0, 0.0, 1.0),
        (95000.0, 100000.0, 1.7109743250860559e-58, 1.0),
        (100000.0, 100000.0, 0.50042052211036514, 0.4995794778896348),
        (105000.0, 100000.0, 1.0, 7.2049596948348962e-55),
        (200000.0, 100000.0, 1.0, 0.0),
        (500000.0, 100000.0, 1.0, 0.0),
        (9.9998886718268301e-321, 0.5, 1.1283728860584653e-160, 1.0),
        (1.0000000000000001e300, 3.0, 1.0, 0.0),
    ];

    // pgamma with log.p = TRUE: (x, shape, log_lower, log_upper)
    #[rustfmt::skip]
    const PGAMMA_LOG_REF: &[(f64, f64, f64, f64)] = &[
        (1e-10, 0.10000000000000001, -2.2527126517432969, -0.11105860856547681),
        (5.0, 0.10000000000000001, -0.00014394932605384876, -8.8461211965239173),
        (500.0, 0.10000000000000001, -2.7833376533568768e-221, -507.84765474938638),
        (100000.0, 0.10000000000000001, -0.0, -100012.61435457008),
        (1e-10, 1.1000000000000001, -25.373873761531375, -9.5557909641976187e-12),
        (5.0, 1.1000000000000001, -0.0084992001084677731, -4.7720298150060954),
        (500.0, 1.1000000000000001, -1.3944471831399667e-217, -499.32846712746351),
        (100000.0, 1.1000000000000001, -0.0, -99998.798834012254),
        (1e-10, 10.0, -245.36292187257101, -2.7557319221480277e-107),
        (5.0, 10.0, -3.4474070729719002, -0.032345580729534869),
        (500.0, 10.0, -3.9047966391213617e-199, -456.85222780092909),
        (100000.0, 10.0, -0.0, -99909.185408292193),
        (1e-10, 1000.0, -28937.979108428717, -0.0),
        (5.0, 1000.0, -4307.6852585574825, -0.0),
        (500.0, 1000.0, -196.82891906086252, -3.2982727970671435e-86),
        (100000.0, 1000.0, -0.0, -94403.797843570952),
    ];

    // qgamma (scale 1): (p, shape, lower, upper)
    #[rustfmt::skip]
    const QGAMMA_REF: &[(f64, f64, f64, f64)] = &[
        (1e-50, 0.001, 0.0, 103.57689269435197),
        (0.00069999999999999999, 0.001, 0.0, 0.40158047294152122),
        (0.10000000000000001, 0.001, 0.0, 9.8216596440664843e-47),
        (0.5, 0.001, 5.2442064082777997e-302, 5.2442064082777997e-302),
        (0.98999999999999999, 0.001, 2.4259428385578427e-05, 0.0),
        (0.99999999989999999, 0.001, 13.454595433853402, 0.0),
        (1e-50, 0.5, 7.8539816339744978e-101, 112.19237415939827),
        (0.00069999999999999999, 0.5, 3.8484519880194989e-07, 5.7446233770019282),
        (0.10000000000000001, 0.5, 0.0078953870467156143, 1.3527717270477078),
        (0.5, 0.5, 0.22746821155978642, 0.22746821155978642),
        (0.98999999999999999, 0.5, 3.3174483005106064, 7.8543928954851136e-05),
        (0.99999999989999999, 0.5, 20.910728101491305, 7.8539829336572272e-21),
        (1e-50, 0.90000000000000002, 2.6646035646005264e-56, 114.58787936804968),
        (0.00069999999999999999, 0.90000000000000002, 0.0002990955159362689, 6.9909779764068816),
        (0.10000000000000001, 0.90000000000000002, 0.077196721093799284, 2.1266600892875078),
        (0.5, 0.90000000000000002, 0.59674304895539454, 0.59674304895539454),
        (0.98999999999999999, 0.90000000000000002, 4.3722706800919795, 0.0057581294033880468),
        (0.99999999989999999, 0.90000000000000002, 22.643260096460821, 7.4144183834978951e-12),
        (1e-50, 2.5, 1.6167038902915651e-20, 122.06364901373752),
        (0.00069999999999999999, 2.5, 0.090760962579975096, 10.667660666358072),
        (0.10000000000000001, 2.5, 0.80515399348116135, 4.6181784498905598),
        (0.5, 2.5, 2.1757300955477632, 2.1757300955477632),
        (0.98999999999999999, 2.5, 7.5431362346944937, 0.27714903836413868),
        (0.99999999989999999, 2.5, 27.781199171832373, 0.00016167786266364607),
        (1e-50, 100.0, 13.767539159236735, 330.65575904365477),
        (0.00069999999999999999, 100.0, 71.088831634681085, 135.03926106361769),
        (0.10000000000000001, 100.0, 87.417636499593684, 113.01052385984448),
        (0.5, 100.0, 99.666864919315429, 99.666864919315429),
        (0.98999999999999999, 100.0, 124.72256149072079, 78.215983053795824),
        (0.99999999989999999, 100.0, 177.30050452110291, 48.883092131540522),
        (1e-50, 10000.0, 8579.7529536585516, 11568.212885526242),
        (0.00069999999999999999, 10000.0, 9683.6006025640272, 10322.53650349421),
        (0.10000000000000001, 10000.0, 9872.0608750497358, 10128.367373674177),
        (0.5, 10000.0, 9999.6666686420467, 9999.6666686420467),
        (0.98999999999999999, 10000.0, 10234.104379158054, 9768.836856696591),
        (0.99999999989999999, 10000.0, 10649.348143058445, 9376.961683261572),
    ];

    // pbeta: (x, a, b, lower, upper)
    #[rustfmt::skip]
    const PBETA_REF: &[(f64, f64, f64, f64, f64)] = &[
        (0.001, 0.5, 0.5, 0.020135041633377489, 0.97986495836662257),
        (0.10000000000000001, 0.5, 0.5, 0.20483276469913345, 0.79516723530086653),
        (0.5, 0.5, 0.5, 0.49999999999999956, 0.50000000000000044),
        (0.90000000000000002, 0.5, 0.5, 0.79516723530086664, 0.20483276469913339),
        (0.999, 0.5, 0.5, 0.97986495836662257, 0.020135041633377496),
        (0.001, 1.0, 1.0, 0.0010000000000000002, 0.999),
        (0.10000000000000001, 1.0, 1.0, 0.10000000000000002, 0.89999999999999991),
        (0.5, 1.0, 1.0, 0.5, 0.5),
        (0.90000000000000002, 1.0, 1.0, 0.90000000000000002, 0.099999999999999978),
        (0.999, 1.0, 1.0, 0.999, 0.0010000000000000011),
        (0.001, 2.0, 3.0, 5.9920030000000003e-06, 0.999994007997),
        (0.10000000000000001, 2.0, 3.0, 0.052300000000000027, 0.94769999999999999),
        (0.5, 2.0, 3.0, 0.6875, 0.31250000000000006),
        (0.90000000000000002, 2.0, 3.0, 0.99629999999999996, 0.0036999999999999993),
        (0.999, 2.0, 3.0, 0.99999999600299994, 3.9970000000000225e-09),
        (0.001, 0.5, 500.0, 0.68268955269028253, 0.31731044730971752),
        (0.10000000000000001, 0.5, 500.0, 1.0, 1.0453677938354537e-24),
        (0.5, 0.5, 500.0, 1.0, 1.0887202471752057e-152),
        (0.90000000000000002, 0.5, 500.0, 1.0, 0.0),
        (0.999, 0.5, 500.0, 1.0, 0.0),
        (0.001, 500.0, 0.5, 0.0, 1.0),
        (0.10000000000000001, 500.0, 0.5, 0.0, 1.0),
        (0.5, 500.0, 0.5, 1.0887202471752057e-152, 1.0),
        (0.90000000000000002, 500.0, 0.5, 1.0453677938354686e-24, 1.0),
        (0.999, 500.0, 0.5, 0.31731044730971736, 0.68268955269028264),
        (0.001, 15.0, 0.5, 1.4453221364280131e-46, 1.0),
        (0.10000000000000001, 15.0, 0.5, 1.5175525817575614e-16, 0.99999999999999978),
        (0.5, 15.0, 0.5, 6.0551763840959078e-06, 0.99999394482361592),
        (0.90000000000000002, 15.0, 0.5, 0.077858670314667824, 0.92214132968533224),
        (0.999, 15.0, 0.5, 0.86358607508951313, 0.13641392491048684),
        (0.001, 2.5, 100.0, 0.00090341520334406573, 0.99909658479665597),
        (0.10000000000000001, 2.5, 100.0, 0.99926657838877486, 0.00073342161122510032),
        (0.5, 2.5, 100.0, 1.0, 2.1694116821877815e-28),
        (0.90000000000000002, 2.5, 100.0, 1.0, 6.5544047860931392e-98),
        (0.999, 2.5, 100.0, 1.0, 7.6525773343392215e-298),
        (0.001, 100.0, 100.0, 4.104506907877898e-242, 1.0),
        (0.10000000000000001, 100.0, 100.0, 1.4990328239114312e-46, 1.0),
        (0.5, 100.0, 100.0, 0.50000000000000022, 0.49999999999999978),
        (0.90000000000000002, 100.0, 100.0, 1.0, 1.499032823911389e-46),
        (0.999, 100.0, 100.0, 1.0, 4.1045069078783658e-242),
        (0.001, 1000.0, 1000.0, 0.0, 1.0),
        (0.10000000000000001, 1000.0, 1000.0, 0.0, 1.0),
        (0.5, 1000.0, 1000.0, 0.5, 0.5),
        (0.90000000000000002, 1000.0, 1000.0, 1.0, 0.0),
        (0.999, 1000.0, 1000.0, 1.0, 0.0),
        (0.001, 0.01, 0.02, 0.62237591132777537, 0.37762408867222458),
        (0.10000000000000001, 0.01, 0.02, 0.65236731431385087, 0.34763268568614919),
        (0.5, 0.01, 0.02, 0.66671946104127244, 0.33328053895872756),
        (0.90000000000000002, 0.01, 0.02, 0.6809152058570811, 0.3190847941429189),
        (0.999, 0.01, 0.02, 0.70957968101247249, 0.29042031898752751),
    ];

    // pt: (x, df, lower, upper, log_lower)
    #[rustfmt::skip]
    const PT_REF: &[(f64, f64, f64, f64, f64)] = &[
        (-50.0, 0.5, 0.045352606402966393, 0.95464739359703366, -3.0932876310192214),
        (-3.0, 0.5, 0.18365407799297176, 0.81634592200702827, -1.6947013017194772),
        (-1.0, 0.5, 0.30112161084132205, 0.69887838915867795, -1.2002410730998223),
        (-0.10000000000000001, 0.5, 0.47316581056430318, 0.52683418943569682, -0.74830940104290156),
        (0.5, 0.5, 0.62134096353528168, 0.37865903646471832, -0.47587529211217083),
        (1.5, 0.5, 0.74604910467594576, 0.25395089532405424, -0.29296385698228439),
        (10.0, 0.5, 0.89866132361433437, 0.10133867638566559, -0.1068490411687345),
        (10000000000.0, 0.5, 0.9999967929902458, 3.2070097541422295e-06, -3.2070148966090067e-06),
        (-50.0, 1.0, 0.0063653491009727971, 0.99363465089902725, -5.0568862015054341),
        (-3.0, 1.0, 0.10241638234956672, 0.89758361765043326, -2.2787085952902988),
        (-1.0, 1.0, 0.24999999999999978, 0.75000000000000022, -1.3862943611198917),
        (-0.10000000000000001, 1.0, 0.46827448256944643, 0.53172551743055352, -0.75870065377234364),
        (0.5, 1.0, 0.64758361765043326, 0.35241638234956679, -0.43450735451772826),
        (1.5, 1.0, 0.81283295818900125, 0.1871670418109988, -0.20722965402700266),
        (10.0, 1.0, 0.96827448256944648, 0.03172551743055356, -0.032239675526675296),
        (10000000000.0, 1.0, 0.99999999996816902, 3.1830988618379065e-11, -3.1830988618885694e-11),
        (-50.0, 2.0, 0.00019988007994404028, 0.99980011992005591, -8.5177929715281717),
        (-3.0, 2.0, 0.047732983133354563, 0.95226701686664539, -3.0421326497423489),
        (-1.0, 2.0, 0.21132486540518713, 0.78867513459481287, -1.5543586830764358),
        (-0.10000000000000001, 2.0, 0.46473271920707004, 0.53526728079292996, -0.76629283605828913),
        (0.5, 2.0, 0.66666666666666663, 0.33333333333333337, -0.40546510810816438),
        (1.5, 2.0, 0.86380343755449951, 0.13619656244550046, -0.1464100388912164),
        (10.0, 2.0, 0.99507377148833709, 0.0049262285116628462, -0.0049384023726074741),
        (10000000000.0, 2.0, 1.0, 4.9999999999999997e-21, -4.999999999999996e-21),
        (-50.0, 5.0, 3.0238788133006171e-08, 0.99999996976121186, -17.314140361404828),
        (-3.0, 5.0, 0.015049623948731284, 0.98495037605126878, -4.1964022748946901),
        (-1.0, 5.0, 0.18160873382456127, 0.81839126617543867, -1.705900720223827),
        (-0.10000000000000001, 5.0, 0.4621150705773302, 0.53788492942266974, -0.77194134839960737),
        (0.5, 5.0, 0.68085056417953549, 0.31914943582046451, -0.38441243277190862),
        (1.5, 5.0, 0.90304815987876341, 0.096951840121236574, -0.10197939378824261),
        (10.0, 5.0, 0.99991452621212851, 8.5473787871481787e-05, -8.5477440963852261e-05),
        (10000000000.0, 5.0, 1.0, 9.4901672455624471e-50, -9.4901672455623673e-50),
        (-50.0, 30.0, 9.3577088296114218e-31, 1.0, -69.143937405451382),
        (-3.0, 30.0, 0.0026949820328259705, 0.99730501796717408, -5.9163637414983281),
        (-1.0, 30.0, 0.16265430771301492, 0.83734569228698508, -1.8161281418575999),
        (-0.10000000000000001, 30.0, 0.46050480589513554, 0.53949519410486446, -0.77543198708686678),
        (0.5, 30.0, 0.68963849755743634, 0.31036150244256366, -0.37158773526650601),
        (1.5, 30.0, 0.92796703543567693, 0.072032964564323038, -0.074759068986694197),
        (10.0, 30.0, 0.99999999997712374, 2.2876257041148048e-11, -2.2876257041409767e-11),
        (10000000000.0, 30.0, 1.0, 1.0364534652561783e-279, -1.0364534652561315e-279),
        (-50.0, 1000.0, 1.3793362061624403e-274, 1.0, -630.58671310694899),
        (-3.0, 1000.0, 0.001383354522119095, 0.99861664547788087, -6.5832439163304208),
        (-1.0, 1000.0, 0.15877620904233633, 0.84122379095766364, -1.8402595585065575),
        (-0.10000000000000001, 1000.0, 0.46018218451180209, 0.53981781548819785, -0.77613281462087103),
        (0.5, 1000.0, 0.69140745958306249, 0.30859254041693751, -0.36902596245379471),
        (1.5, 1000.0, 0.93303498058895684, 0.066965019411043103, -0.069312586244957411),
        (10.0, 1000.0, 1.0, 8.3353514793000405e-23, -8.3353514793000229e-23),
        (10000000000.0, 1000.0, 1.0, 0.0, -0.0),
        (-50.0, 1000000.0, 0.0, 1.0, -1253.2702122421226),
        (-3.0, 1000000.0, 0.0013499312707108962, 0.99865006872928919, -6.6077015984123086),
        (-1.0, 1000000.0, 0.15865537491678908, 0.84134462508321095, -1.8410208824421068),
        (-0.10000000000000001, 1000000.0, 0.46017217274602162, 0.53982782725397838, -0.77615457094918505),
        (0.5, 1000000.0, 0.69146240626381417, 0.30853759373618583, -0.36894649484496522),
        (1.5, 1000000.0, 0.93319264088160359, 0.066807359118396414, -0.069143624762223063),
        (10.0, 1000000.0, 1.0, 7.6393053840891504e-24, -7.6393053840891548e-24),
        (10000000000.0, 1000000.0, 1.0, 0.0, -0.0),
    ];

    // qt: (p, df, lower, upper)
    #[rustfmt::skip]
    const QT_REF: &[(f64, f64, f64, f64)] = &[
        (1e-100, 0.5, -1.0284911563163961e199, f64::INFINITY),
        (1e-10, 0.5, -1.0284911563163566e19, 1.0284926989044548e19),
        (0.001, 0.5, -102849.11563017513, 102849.11563018258),
        (0.025000000000000001, 0.5, -164.55767348049446, 164.55767348049446),
        (0.29999999999999999, 0.5, -1.0095258786071923, 1.0095258786071923),
        (0.69999999999999996, 0.5, 1.0095258786071923, -1.0095258786071923),
        (0.97499999999999998, 0.5, 164.55767348049446, -164.55767348049446),
        (0.99999999989999999, 0.5, 1.0284926989044548e19, -1.028490986120885e19),
        (1e-100, 1.0, -3.1830988618379069e99, 3.1830988618379069e99),
        (1e-10, 1.0, -3183098861.8379064, 3183098861.8379064),
        (0.001, 1.0, -318.30883898555044, 318.30883898555044),
        (0.025000000000000001, 1.0, -12.706204736174707, 12.706204736174707),
        (0.29999999999999999, 1.0, -0.72654252800536101, 0.72654252800536101),
        (0.69999999999999996, 1.0, 0.72654252800536079, -0.72654252800536079),
        (0.97499999999999998, 1.0, 12.706204736174694, -12.706204736174694),
        (0.99999999989999999, 1.0, 3183098598.4671478, -3183098598.4671478),
        (1e-100, 2.0, -7.0710678118654751e49, 7.0710678118654751e49),
        (1e-10, 2.0, -70710.678108048145, 70710.678108048145),
        (0.001, 2.0, -22.327124770119873, 22.327124770119873),
        (0.025000000000000001, 2.0, -4.3026527297494637, 4.3026527297494637),
        (0.29999999999999999, 2.0, -0.61721339984836765, 0.61721339984836765),
        (0.69999999999999996, 2.0, 0.61721339984836765, -0.61721339984836765),
        (0.97499999999999998, 2.0, 4.3026527297494619, -4.3026527297494619),
        (0.99999999989999999, 2.0, 70710.675182734471, -70710.675182734471),
        (1e-100, 5.0, -1.5683925590993401e20, 1.5683925590993401e20),
        (1e-10, 5.0, -156.82559270889428, 156.82559270889428),
        (0.001, 5.0, -5.8934295313560101, 5.8934295313560101),
        (0.025000000000000001, 5.0, -2.570581835636315, 2.570581835636315),
        (0.29999999999999999, 5.0, -0.55942964446936105, 0.55942964446936105),
        (0.69999999999999996, 5.0, 0.55942964446936083, -0.55942964446936083),
        (0.97499999999999998, 5.0, 2.5705818356363137, -2.5705818356363137),
        (0.99999999989999999, 5.0, 156.82559011328078, -156.82559011328078),
        (1e-100, 30.0, -10810.645001144016, 10810.645001144016),
        (1e-10, 30.0, -9.377489780407144, 9.377489780407144),
        (0.001, 30.0, -3.3851848668293054, 3.3851848668293054),
        (0.025000000000000001, 30.0, -2.0422724563012382, 2.0422724563012382),
        (0.29999999999999999, 30.0, -0.53001900390650458, 0.53001900390650458),
        (0.69999999999999996, 30.0, 0.53001900390650436, -0.53001900390650436),
        (0.97499999999999998, 30.0, 2.0422724563012378, -2.0422724563012378),
        (0.99999999989999999, 30.0, 9.3774897460797746, -9.3774897460797746),
        (1e-100, 1000.0, -23.930617087826445, 23.930617087826445),
        (1e-10, 1000.0, -6.4278762831342062, 6.4278762831342062),
        (0.001, 1000.0, -3.098402163912922, 3.098402163912922),
        (0.025000000000000001, 1000.0, -1.9623390808264078, 1.9623390808264078),
        (0.29999999999999999, 1000.0, -0.52456770730922675, 0.52456770730922675),
        (0.69999999999999996, 1000.0, 0.52456770730922675, -0.52456770730922675),
        (0.97499999999999998, 1000.0, 1.9623390808264076, -1.9623390808264076),
        (0.99999999989999999, 1000.0, 6.4278762700330212, -6.4278762700330212),
        (1e-100, 1000000.0, -21.275865985490611, 21.275865985490611),
        (1e-10, 1000000.0, -6.3614068488767419, 6.3614068488767419),
        (0.001, 1000000.0, -3.0902404563165193, 3.0902404563165193),
        (0.025000000000000001, 1000000.0, -1.9599663568141066, 1.9599663568141066),
        (0.29999999999999999, 1000000.0, -0.52440067986020866, 0.52440067986020866),
        (0.69999999999999996, 1000000.0, 0.52440067986020866, -0.52440067986020866),
        (0.97499999999999998, 1000000.0, 1.9599663568141066, -1.9599663568141066),
        (0.99999999989999999, 1000000.0, 6.3614068361697171, -6.3614068361697171),
    ];

    // qt with log.p = TRUE: (lp, df, lower)
    #[rustfmt::skip]
    const QT_LOG_REF: &[(f64, f64, f64)] = &[
        (-1000.0, 1.0, f64::NEG_INFINITY),
        (-100.0, 1.0, -8.5565426146019082e42),
        (-1.0, 1.0, -0.44067109149597056),
        (-1000.0, 2.0, -9.9248957526440727e216),
        (-100.0, 2.0, -3.666140437719302e21),
        (-1.0, 2.0, -0.38746518251861761),
        (-1000.0, 5.0, -1.1333163510624135e87),
        (-100.0, 5.0, -760929482.41469133),
        (-1.0, 5.0, -0.3568658837283194),
        (-1000.0, 30.0, -1503145624621950.0),
        (-100.0, 30.0, -140.55535599026089),
        (-1.0, 30.0, -0.34062621237207513),
    ];

    // dnorm: (x, value, log_value)
    #[rustfmt::skip]
    const DNORM_REF: &[(f64, f64, f64)] = &[
        (0.0, 0.3989422804014327, -0.91893853320467278),
        (0.5, 0.35206532676429952, -1.0439385332046727),
        (2.5, 0.01752830049356854, -4.0439385332046731),
        (4.9900000000000002, 1.5628671089492902e-06, -13.368988533204673),
        (5.0, 1.4867195147342977e-06, -13.418938533204672),
        (10.0, 7.6945986267064199e-23, -50.918938533204674),
        (37.0, 2.1200065515246056e-298, -685.41893853320471),
        (38.5, 5.434722104253712e-323, -742.04393853320471),
        (39.0, 0.0, -761.41893853320471),
        (-30.0, 1.4736461348785476e-196, -450.91893853320465),
    ];

    // dt: (x, df, value, log_value)
    #[rustfmt::skip]
    const DT_REF: &[(f64, f64, f64, f64)] = &[
        (0.0, 1.0, 0.31830988618379069, -1.1447298858494002),
        (0.29999999999999999, 1.0, 0.29202741851723918, -1.2309075820904525),
        (2.5, 1.0, 0.043904811887419404, -3.1257313547159837),
        (10000.0, 1.0, 3.1830988300069238e-09, -19.565410639801762),
        (0.0, 2.0, 0.35355339059327379, -1.0397207708399181),
        (0.29999999999999999, 2.0, 0.33096385830912667, -1.1057460989650796),
        (2.5, 2.0, 0.04220064386804797, -3.1653198005198844),
        (10000.0, 2.0, 9.9999996999999669e-13, -27.63102114592855),
        (0.0, 30.0, 0.39563218489409779, -0.92727032537884568),
        (0.29999999999999999, 30.0, 0.37768275260924272, -0.97370071456572205),
        (2.5, 30.0, 0.02105701922062166, -3.8605213197760349),
        (10000.0, 30.0, 3.1093459372785308e-102, -233.72926709087642),
        (0.0, 1000000.0, 0.39894218066587506, -0.91893878320467282),
        (0.29999999999999999, 1000000.0, 0.38138770372344594, -0.96393882617967086),
        (2.5, 1000000.0, 0.01752841251017976, -4.043932142610597),
        (10000.0, 1000000.0, 0.0, -2307563.4849196714),
    ];

    // dgamma (scale 1): (x, shape, value, log_value)
    #[rustfmt::skip]
    const DGAMMA_REF: &[(f64, f64, f64, f64)] = &[
        (0.001, 0.5, 17.823408838013965, 2.8805126965663681),
        (1.0, 0.5, 0.20755374871029733, -1.5723649429247),
        (90.0, 0.5, 4.8730451768405759e-41, -92.822269778089833),
        (500.0, 0.5, 1.7976250437466468e-219, -503.67966899213582),
        (0.001, 1.0, 0.99900049983337502, -0.001),
        (1.0, 1.0, 0.36787944117144233, -1.0),
        (90.0, 1.0, 8.1940126239905147e-40, -90.0),
        (500.0, 1.0, 7.1245764067412855e-218, -500.0),
        (0.001, 2.5, 2.3764545117351958e-05, -10.647315788946123),
        (1.0, 2.5, 0.27673833161372979, -1.2846828704729192),
        (90.0, 2.5, 5.2628887909878227e-37, -83.53496836497753),
        (500.0, 2.5, 5.9920834791554886e-214, -490.96277072283959),
        (0.001, 100.0, 0.0, -1043.0029779888068),
        (1.0, 100.0, 3.941866060050479e-157, -360.1342053695754),
        (90.0, 100.0, 0.025912028250157558, -3.6530480068791569),
        (500.0, 100.0, 1.2044418102251225e-106, -243.8880036257784),
    ];

    // lbeta: (a, b, value)
    #[rustfmt::skip]
    const LBETA_REF: &[(f64, f64, f64)] = &[
        (0.5, 0.5, 1.1447298858494004),
        (2.0, 3.0, -2.4849066497880004),
        (0.5, 500.0, -2.5346891063280625),
        (15.0, 0.5, -0.77332836545223194),
        (100.0, 100.0, -139.66525908670664),
        (1000.0, 1000.0, -1388.482601635902),
        (0.01, 0.02, 5.0103133506979853),
        (1e-08, 5.0, 18.420680723119034),
    ];

    // z_score_for_confidence: (conf, qnorm((1 + conf) / 2))
    #[rustfmt::skip]
    const Z_REF: &[(f64, f64)] = &[
        (0.5, 0.67448975019608171),
        (0.80000000000000004, 1.2815515655446008),
        (0.90000000000000002, 1.6448536269514715),
        (0.94999999999999996, 1.9599639845400536),
        (0.98999999999999999, 2.5758293035488999),
        (0.999, 3.2905267314919255),
    ];

    #[test]
    fn pnorm_matches_r() {
        for &(x, lower, upper, log_lower, log_upper) in PNORM_REF {
            assert_matches_r(&format!("pnorm({x})"), pnorm(x, true, false), lower);
            assert_matches_r(&format!("pnorm({x}, upper)"), pnorm(x, false, false), upper);
            assert_matches_r(&format!("pnorm({x}, log)"), pnorm(x, true, true), log_lower);
            assert_matches_r(
                &format!("pnorm({x}, upper, log)"),
                pnorm(x, false, true),
                log_upper,
            );
        }
        assert!(pnorm(f64::NAN, true, false).is_nan());
        assert_eq!(pnorm(f64::NEG_INFINITY, true, false), 0.0);
        assert_eq!(pnorm(f64::INFINITY, false, true), f64::NEG_INFINITY);
    }

    #[test]
    fn qnorm_matches_r() {
        for &(p, lower, upper) in QNORM_REF {
            assert_matches_r(&format!("qnorm({p})"), qnorm(p, true, false), lower);
            assert_matches_r(&format!("qnorm({p}, upper)"), qnorm(p, false, false), upper);
        }
        for &(lp, lower, upper) in QNORM_LOG_REF {
            assert_matches_r(&format!("qnorm({lp}, log)"), qnorm(lp, true, true), lower);
            assert_matches_r(
                &format!("qnorm({lp}, upper, log)"),
                qnorm(lp, false, true),
                upper,
            );
        }
        assert_eq!(qnorm(0.0, true, false), f64::NEG_INFINITY);
        assert_eq!(qnorm(1.0, true, false), f64::INFINITY);
        assert_eq!(qnorm(0.0, false, false), f64::INFINITY);
        assert_eq!(qnorm(0.0, true, true), f64::INFINITY);
        assert!(qnorm(1.5, true, false).is_nan());
        assert!(qnorm(0.5, true, true).is_nan());
        assert!(qnorm(f64::NAN, true, false).is_nan());
    }

    #[test]
    fn erf_keeps_relative_accuracy() {
        // erf(1e-8) = 2/sqrt(pi) * 1e-8 (1 - 1e-16/3); erf(3) = 1 - erfc(3).
        assert!((erf(1e-8) / 1.1283791670955126e-08 - 1.0).abs() < 1e-15);
        assert!((erf(-1e-8) / -1.1283791670955126e-08 - 1.0).abs() < 1e-15);
        assert!((erfc(3.0) / 2.2090496998585394e-05 - 1.0).abs() < 1e-15);
        assert!((erf(3.0) - 0.99997790950300141).abs() < 1.2e-16);
        assert!((erfc(-3.0) - (2.0 - 2.2090496998585394e-05)).abs() < 1e-15);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erfc(0.0), 1.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
    }

    #[test]
    fn gamma_functions_match_r() {
        for &(x, value) in LGAMMA_REF {
            assert_matches_r(&format!("lgamma({x})"), lgammafn(x), value);
        }
        for &(x, value) in GAMMA_REF {
            assert_matches_r(&format!("gamma({x})"), gammafn(x), value);
        }
        assert_eq!(lgammafn(1.0), 0.0);
        assert_eq!(lgammafn(2.0), 0.0);
        assert_eq!(lgammafn(0.0), f64::INFINITY);
        assert_eq!(lgammafn(-3.0), f64::INFINITY);
        assert!(gammafn(-3.0).is_nan());
        assert!(lgammafn(f64::NAN).is_nan());
        // lgamma1p(a) = lgamma(1 + a) = -gamma a + zeta(2)/2 a^2 - ... without
        // the cancellation of log(gamma(1 + a)) for tiny a.
        assert!((lgamma1p(1e-10) / -5.7721566481928617e-11 - 1.0).abs() < 1e-13);
        assert!((lgamma1p(0.3) - lgammafn(1.3)).abs() < 1e-16);
        // log1pmx(x) = log(1 + x) - x = -x^2/2 + x^3/3 - ...
        assert!((log1pmx(1e-8) / -4.9999999666666669e-17 - 1.0).abs() < 1e-13);
        assert!((log1pmx(0.5) - (1.5f64.ln() - 0.5)).abs() < 1e-16);
    }

    #[test]
    fn chisq_matches_r() {
        for &(x, df, lower, upper, log_lower, log_upper) in PCHISQ_REF {
            assert_matches_r(
                &format!("pchisq({x}, {df})"),
                pchisq(x, df, true, false),
                lower,
            );
            assert_matches_r(
                &format!("pchisq({x}, {df}, upper)"),
                pchisq(x, df, false, false),
                upper,
            );
            assert_matches_r(
                &format!("pchisq({x}, {df}, log)"),
                pchisq(x, df, true, true),
                log_lower,
            );
            assert_matches_r(
                &format!("pchisq({x}, {df}, upper, log)"),
                pchisq(x, df, false, true),
                log_upper,
            );
        }
        for &(p, df, lower, upper) in QCHISQ_REF {
            assert_matches_r(
                &format!("qchisq({p}, {df})"),
                qchisq(p, df, true, false),
                lower,
            );
            assert_matches_r(
                &format!("qchisq({p}, {df}, upper)"),
                qchisq(p, df, false, false),
                upper,
            );
        }
        assert_eq!(pchisq(0.0, 3.0, true, false), 0.0);
        assert_eq!(pchisq(-1.0, 3.0, false, false), 1.0);
        assert_eq!(qchisq(0.0, 3.0, true, false), 0.0);
        assert_eq!(qchisq(1.0, 3.0, true, false), f64::INFINITY);
    }

    #[test]
    fn gamma_distribution_matches_r() {
        for &(x, shape, lower, upper) in PGAMMA_REF {
            assert_matches_r(
                &format!("pgamma({x}, {shape})"),
                pgamma(x, shape, 1.0, true, false),
                lower,
            );
            assert_matches_r(
                &format!("pgamma({x}, {shape}, upper)"),
                pgamma(x, shape, 1.0, false, false),
                upper,
            );
        }
        for &(x, shape, log_lower, log_upper) in PGAMMA_LOG_REF {
            assert_matches_r(
                &format!("pgamma({x}, {shape}, log)"),
                pgamma(x, shape, 1.0, true, true),
                log_lower,
            );
            assert_matches_r(
                &format!("pgamma({x}, {shape}, upper, log)"),
                pgamma(x, shape, 1.0, false, true),
                log_upper,
            );
        }
        for &(p, shape, lower, upper) in QGAMMA_REF {
            assert_matches_r(
                &format!("qgamma({p}, {shape})"),
                qgamma(p, shape, 1.0, true, false),
                lower,
            );
            assert_matches_r(
                &format!("qgamma({p}, {shape}, upper)"),
                qgamma(p, shape, 1.0, false, false),
                upper,
            );
        }
        for &(x, shape, value, log_value) in DGAMMA_REF {
            assert_matches_r(
                &format!("dgamma({x}, {shape})"),
                dgamma(x, shape, 1.0, false),
                value,
            );
            assert_matches_r(
                &format!("dgamma({x}, {shape}, log)"),
                dgamma(x, shape, 1.0, true),
                log_value,
            );
        }
        // Scale is a plain rescaling of x.
        assert_eq!(
            pgamma(6.0, 2.5, 2.0, true, false),
            pgamma(3.0, 2.5, 1.0, true, false)
        );
        assert_eq!(
            qgamma(0.3, 2.5, 2.0, true, false),
            2.0 * qgamma(0.3, 2.5, 1.0, true, false)
        );
        assert!(pgamma(1.0, -1.0, 1.0, true, false).is_nan());
        assert_eq!(pgamma(1.0, 0.0, 1.0, true, false), 1.0);
        assert_eq!(qgamma(0.5, 0.0, 1.0, true, false), 0.0);
    }

    #[test]
    fn pbeta_matches_r() {
        for &(x, a, b, lower, upper) in PBETA_REF {
            assert_matches_r(
                &format!("pbeta({x}, {a}, {b})"),
                pbeta(x, a, b, true, false),
                lower,
            );
            assert_matches_r(
                &format!("pbeta({x}, {a}, {b}, upper)"),
                pbeta(x, a, b, false, false),
                upper,
            );
            let log_lower = pbeta(x, a, b, true, true);
            assert_matches_r(
                &format!("pbeta({x}, {a}, {b}, log)"),
                log_lower.exp(),
                lower,
            );
        }
        for &(a, b, value) in LBETA_REF {
            assert_matches_r(&format!("lbeta({a}, {b})"), lbeta(a, b), value);
        }
        assert_eq!(pbeta(0.0, 2.0, 3.0, true, false), 0.0);
        assert_eq!(pbeta(1.0, 2.0, 3.0, true, false), 1.0);
        assert_eq!(pbeta(0.3, 0.0, 0.0, true, false), 0.5);
        assert_eq!(pbeta(0.3, 0.0, 2.0, true, false), 1.0);
        assert_eq!(pbeta(0.3, 2.0, 0.0, true, false), 0.0);
        assert_eq!(pbeta(0.3, f64::INFINITY, f64::INFINITY, true, false), 0.0);
        assert!(pbeta(0.3, -1.0, 2.0, true, false).is_nan());
    }

    #[test]
    fn student_t_matches_r() {
        for &(x, df, lower, upper, log_lower) in PT_REF {
            assert_matches_r(&format!("pt({x}, {df})"), pt(x, df, true, false), lower);
            assert_matches_r(
                &format!("pt({x}, {df}, upper)"),
                pt(x, df, false, false),
                upper,
            );
            assert_matches_r(
                &format!("pt({x}, {df}, log)"),
                pt(x, df, true, true),
                log_lower,
            );
        }
        for &(p, df, lower, upper) in QT_REF {
            assert_matches_r(&format!("qt({p}, {df})"), qt(p, df, true, false), lower);
            assert_matches_r(
                &format!("qt({p}, {df}, upper)"),
                qt(p, df, false, false),
                upper,
            );
        }
        for &(lp, df, lower) in QT_LOG_REF {
            assert_matches_r(
                &format!("qt({lp}, {df}, log)"),
                qt(lp, df, true, true),
                lower,
            );
        }
        for &(x, df, value, log_value) in DT_REF {
            assert_matches_r(&format!("dt({x}, {df})"), dt(x, df, false), value);
            assert_matches_r(&format!("dt({x}, {df}, log)"), dt(x, df, true), log_value);
        }
        assert_eq!(pt(f64::INFINITY, 3.0, true, false), 1.0);
        assert_eq!(pt(f64::NEG_INFINITY, 3.0, true, false), 0.0);
        assert_eq!(pt(1.5, f64::INFINITY, true, false), pnorm(1.5, true, false));
        assert_eq!(dt(f64::INFINITY, 3.0, false), 0.0);
        assert_eq!(dt(0.3, f64::INFINITY, false), dnorm(0.3, false));
        assert_eq!(qt(0.0, 3.0, true, false), f64::NEG_INFINITY);
        assert_eq!(qt(1.0, 3.0, true, false), f64::INFINITY);
        assert_eq!(qt(0.5, 3.0, true, false), 0.0);
        assert!(qt(0.5, 0.0, true, false).is_nan());
        assert!(pt(0.5, -1.0, true, false).is_nan());
    }

    #[test]
    fn dnorm_matches_r() {
        for &(x, value, log_value) in DNORM_REF {
            assert_matches_r(&format!("dnorm({x})"), dnorm(x, false), value);
            assert_matches_r(&format!("dnorm({x}, log)"), dnorm(x, true), log_value);
        }
        assert_eq!(dnorm(f64::INFINITY, false), 0.0);
        assert!(dnorm(f64::NAN, false).is_nan());
    }

    #[test]
    fn z_score_for_confidence_is_exact_qnorm() {
        for &(conf, z) in Z_REF {
            assert_matches_r(
                &format!("z_score_for_confidence({conf})"),
                crate::constants::z_score_for_confidence(conf),
                z,
            );
        }
        assert_eq!(crate::constants::Z_SCORE_90, qnorm(0.95, true, false));
        assert_eq!(crate::constants::Z_SCORE_95, qnorm(0.975, true, false));
        assert_eq!(crate::constants::Z_SCORE_99, qnorm(0.995, true, false));
    }
}
