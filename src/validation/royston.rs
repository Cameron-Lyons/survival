use crate::data_types::{FloatVec, IntVec};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::qnorm;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::coxph::{CoxPHFit, CoxphData, CoxphOptions};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

const DIRECT_THRESHOLD: usize = 256;
const PARALLEL_BLOCK_THRESHOLD: usize = 4_096;
const BLOCK_WIDTH: f64 = 0.5;
const FAR_GAP: f64 = 2.0;
const SAME_ORDER: usize = 16;
const PAIR_ORDER: usize = 22;
const EXP_TERMS: usize = 18;

#[derive(Debug, Clone)]
pub struct RoystonInput<'a> {
    pub eta: &'a [f64],
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub entry_times: Option<&'a [f64]>,
    pub loglik: [f64; 2],
    pub n_event: usize,
    pub n_coef: usize,
    pub ties: bool,
    pub adjust: bool,
}

#[doc = " R's named result vector."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object, get_all)]
pub struct RoystonResult {
    pub d: f64,
    pub se_d: f64,
    pub r_d: f64,
    pub r_ko: f64,
    pub r_n: f64,
    pub c_gh: f64,
}

fn normal_scores(eta: &[f64], ties: bool) -> Vec<f64> {
    let n = eta.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eta[a].total_cmp(&eta[b]).then_with(|| a.cmp(&b)));
    let blom = |rank: f64| qnorm((rank - 3.0 / 8.0) / (n as f64 + 0.25), true, false);
    let mut scores = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && eta[order[end]] == eta[order[start]] {
            end += 1;
        }
        if ties {
            let mean =
                (start..end).map(|k| blom((k + 1) as f64)).sum::<f64>() / (end - start) as f64;
            for &idx in &order[start..end] {
                scores[idx] = mean;
            }
        } else {
            let average_rank = ((start + 1) + end) as f64 / 2.0;
            for &idx in &order[start..end] {
                scores[idx] = blom(average_rank);
            }
        }
        start = end;
    }
    scores
}

/// Taylor coefficients `a[r] = g^(r)(z) / r!` of `g(z) = 1 / (1 + exp(z))`
/// via the logistic ODE recurrence `g'(z) = g(z)^2 - g(z)`.
#[inline]
fn logistic_taylor<const N: usize>(z: f64) -> [f64; N] {
    let a0 = if z >= 0.0 {
        let e = (-z).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + z.exp())
    };
    let mut a = [0.0; N];
    a[0] = a0;
    for r in 0..N - 1 {
        let mut conv = 0.0;
        for k in 0..=r {
            conv += a[k] * a[r - k];
        }
        a[r + 1] = (conv - a[r]) / (r + 1) as f64;
    }
    a
}

const fn binomial_table<const N: usize>() -> [[f64; N]; N] {
    let mut table = [[0.0; N]; N];
    let mut n = 0;
    while n < N {
        table[n][0] = 1.0;
        let mut k = 1;
        while k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}

const PAIR_BINOM: [[f64; PAIR_ORDER]; PAIR_ORDER] = binomial_table::<PAIR_ORDER>();

struct GhBlock {
    start: usize,
    end: usize,
    center: f64,
    moments: [f64; PAIR_ORDER],
    exp_plus: [f64; EXP_TERMS],
    exp_minus: [f64; EXP_TERMS],
    same_sum: f64,
}

fn build_gh_block(sorted: &[f64], start: usize, end: usize) -> GhBlock {
    let min_x = sorted[start];
    let max_x = sorted[end - 1];
    let center = 0.5 * (min_x + max_x);
    let len = end - start;
    let mut moments = [0.0; PAIR_ORDER];
    let mut exp_plus = [0.0; EXP_TERMS];
    let mut exp_minus = [0.0; EXP_TERMS];

    if min_x == max_x {
        moments[0] = len as f64;
        exp_plus.fill(len as f64);
        exp_minus.fill(len as f64);
        let same_sum = 0.5 * (len as f64) * ((len - 1) as f64) * 0.5;
        return GhBlock {
            start,
            end,
            center,
            moments,
            exp_plus,
            exp_minus,
            same_sum,
        };
    }

    let mut suffix_m = [0.0; SAME_ORDER];
    let mut same_sum = 0.0;
    for &x in sorted[start..end].iter().rev() {
        let u = x - center;
        let a = logistic_taylor::<SAME_ORDER>(-u);
        let mut dot = 0.0;
        for r in 0..SAME_ORDER {
            dot += a[r] * suffix_m[r];
        }
        same_sum += dot;

        let mut pow = 1.0;
        for r in 0..SAME_ORDER {
            suffix_m[r] += pow;
            moments[r] += pow;
            pow *= u;
        }
        for m in moments.iter_mut().take(PAIR_ORDER).skip(SAME_ORDER) {
            *m += pow;
            pow *= u;
        }

        let ep = u.exp();
        let em = 1.0 / ep;
        let mut p = ep;
        let mut q = em;
        for k in 0..EXP_TERMS {
            exp_plus[k] += p;
            exp_minus[k] += q;
            p *= ep;
            q *= em;
        }
    }

    GhBlock {
        start,
        end,
        center,
        moments,
        exp_plus,
        exp_minus,
        same_sum,
    }
}

fn gonen_heller_direct(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    let total: f64 = (0..n - 1)
        .map(|i| {
            let xi = sorted[i];
            sorted[i + 1..]
                .iter()
                .map(|&later| 1.0 / (1.0 + (xi - later).exp()))
                .sum::<f64>()
        })
        .sum();
    total * 2.0 / (n as f64 * (n - 1) as f64)
}

fn gonen_heller(eta: &[f64]) -> f64 {
    let n = eta.len();
    let mut sorted = eta.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    if n <= DIRECT_THRESHOLD {
        return gonen_heller_direct(&sorted);
    }

    // Partition sorted predictors into contiguous blocks of width <= 0.5.
    // Within each block |x - center| <= 0.25, so the Taylor expansion of
    // g(d) = 1 / (1 + exp(d)) converges geometrically with ratio <= 0.5 / pi.
    // For pairs separated by >= 2.0, g(d) = sum_{m>=1} (-1)^{m-1} exp(-m d)
    // converges to < 1e-16 in 18 terms without overflow.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    while start < n {
        let limit = sorted[start] + BLOCK_WIDTH;
        let mut end = start + 1;
        while end < n && sorted[end] <= limit {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }

    let blocks: Vec<GhBlock> = if n >= PARALLEL_BLOCK_THRESHOLD && ranges.len() > 1 {
        ranges
            .par_iter()
            .map(|&(s, e)| build_gh_block(&sorted, s, e))
            .collect()
    } else {
        ranges
            .iter()
            .map(|&(s, e)| build_gh_block(&sorted, s, e))
            .collect()
    };

    let mut sum_g: f64 = blocks.iter().map(|b| b.same_sum).sum();
    let n_blocks = blocks.len();
    let mut far_suffix = [0.0; EXP_TERMS];
    let mut far_start = n_blocks;
    let mut ref_center = blocks[n_blocks - 1].center;

    for a_idx in (0..n_blocks).rev() {
        let a = &blocks[a_idx];
        let max_a = sorted[a.end - 1];

        let shift = ref_center - a.center;
        if shift > 0.0 && far_start < n_blocks {
            let e = (-shift).exp();
            let mut em = e;
            for slot in &mut far_suffix {
                *slot *= em;
                em *= e;
            }
        }
        ref_center = a.center;

        while far_start > a_idx + 1 && sorted[blocks[far_start - 1].start] - max_a >= FAR_GAP {
            far_start -= 1;
            let b = &blocks[far_start];
            let delta = b.center - a.center;
            let e = (-delta).exp();
            let mut em = e;
            for (slot, &exp_minus) in far_suffix.iter_mut().zip(&b.exp_minus) {
                *slot += exp_minus * em;
                em *= e;
            }
        }

        if far_start < n_blocks {
            let mut far_contrib = 0.0;
            for (k, (&exp_plus, &suffix)) in a.exp_plus.iter().zip(&far_suffix).enumerate() {
                let term = exp_plus * suffix;
                if k % 2 == 0 {
                    far_contrib += term;
                } else {
                    far_contrib -= term;
                }
            }
            sum_g += far_contrib;
        }

        for b in &blocks[a_idx + 1..far_start] {
            let coeffs = logistic_taylor::<PAIR_ORDER>(b.center - a.center);
            let mut pair_sum = 0.0;
            for (m, (&coeff, binom_row)) in coeffs.iter().zip(&PAIR_BINOM).enumerate() {
                let mut inner = 0.0;
                for (p, &binom) in binom_row[..=m].iter().enumerate() {
                    let term = binom * a.moments[p] * b.moments[m - p];
                    if p % 2 == 0 {
                        inner += term;
                    } else {
                        inner -= term;
                    }
                }
                pair_sum += coeff * inner;
            }
            sum_g += pair_sum;
        }
    }

    let pairs = (n as f64) * ((n - 1) as f64) * 0.5;
    (pairs - sum_g) / pairs
}

fn validate(input: &RoystonInput<'_>) -> SurvivalResult<()> {
    let n = input.eta.len();
    if n < 2 {
        return Err(SurvivalError::invalid_input(
            "royston needs at least two observations",
        ));
    }
    validate_length(n, input.time.len(), "time")?;
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.eta, "eta")?;
    validate_finite(input.time, "time")?;
    validate_binary_i32(input.status, "status")?;
    if let Some(entry) = input.entry_times {
        validate_length(n, entry.len(), "entry_times")?;
        validate_finite(entry, "entry_times")?;
    }
    if input.loglik.iter().any(|v| !v.is_finite()) {
        return Err(SurvivalError::invalid_input("loglik must be finite"));
    }
    if input.adjust && input.n_event <= input.n_coef {
        return Err(SurvivalError::invalid_input(
            "adjust requires more events than coefficients",
        ));
    }
    Ok(())
}

pub fn royston(input: &RoystonInput<'_>) -> SurvivalResult<RoystonResult> {
    validate(input)?;
    let eta = input.eta;
    let n = eta.len() as f64;
    let mean = eta.iter().sum::<f64>() / n;
    let var_eta = eta.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let r_ko = var_eta / (PI * PI / 6.0 + var_eta);

    let qhat = normal_scores(eta, input.ties);
    let n_obs = qhat.len();
    let data = CoxphData::try_new(
        input.time.to_vec(),
        input.entry_times.map(<[f64]>::to_vec),
        input.status.to_vec(),
        Array2::from_shape_vec((n_obs, 1), qhat)
            .map_err(|err| SurvivalError::computation(err.to_string()))?,
        None,
        None,
        None,
    )?;
    let refit = CoxPHFit::fit(data, CoxphOptions::default())
        .map_err(|err| SurvivalError::computation(format!("coxph(y ~ qhat): {err}")))?;
    let beta = refit.coefficients[0];
    let var = refit.var[[0, 0]];
    let mut d = beta * (8.0 / PI).sqrt();
    let mut se_d = (var * 8.0 / PI).sqrt();
    let mut r_d = beta * beta / (PI * PI / 6.0 + beta * beta);
    let r_i = beta * beta / (1.0 + beta * beta);
    if input.adjust {
        let r = input.n_event as f64 / (input.n_event - input.n_coef) as f64;
        let temp = (1.0 + beta * beta - r) / r;
        d = beta.signum() * temp.signum() * (temp.abs() * 8.0 / PI).sqrt();
        se_d = se_d * beta.abs() / (r * temp.abs().sqrt());
        r_d = 1.0 - r * (1.0 - r_i);
    }
    let c_gh = gonen_heller(eta);
    let logtest = -2.0 * (input.loglik[0] - input.loglik[1]);
    let r_n = (1.0 - (-logtest / n).exp()) / (1.0 - (2.0 * input.loglik[0] / n).exp());
    Ok(RoystonResult {
        d,
        se_d,
        r_d,
        r_ko,
        r_n,
        c_gh,
    })
}

#[doc = " Python entry point: `royston(eta, time, status, loglik, n_event,"]
#[doc = " n_coef, entry_times=None, ties=True, adjust=False)`."]
#[pyfunction(name = "royston")]
#[pyo3(signature = (eta, time, status, loglik, n_event, n_coef, entry_times=None, ties=true, adjust=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn royston_py(
    py: Python<'_>,
    eta: FloatVec,
    time: FloatVec,
    status: IntVec,
    loglik: [f64; 2],
    n_event: usize,
    n_coef: usize,
    entry_times: Option<FloatVec>,
    ties: bool,
    adjust: bool,
) -> PyResult<RoystonResult> {
    Ok(py.detach(|| {
        royston(&RoystonInput {
            eta: &eta,
            time: &time,
            status: &status,
            entry_times: entry_times.as_deref(),
            loglik,
            n_event,
            n_coef,
            ties,
            adjust,
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_scores_average_over_ties_when_requested() {
        let eta = [0.3, 0.1, 0.3, 0.7];
        let tied = normal_scores(&eta, true);
        assert_eq!(tied[0], tied[2]);
        let z2 = qnorm((2.0 - 0.375) / 4.25, true, false);
        let z3 = qnorm((3.0 - 0.375) / 4.25, true, false);
        assert!((tied[0] - (z2 + z3) / 2.0).abs() < 1e-12);
        let untied = normal_scores(&eta, false);
        assert!((untied[0] - qnorm((2.5 - 0.375) / 4.25, true, false)).abs() < 1e-12);
        assert!(tied[1] < tied[0] && tied[0] < tied[3]);
    }

    #[test]
    fn gonen_heller_is_half_for_a_constant_predictor() {
        assert!((gonen_heller(&[0.2; 5]) - 0.5).abs() < 1e-12);
        assert!(gonen_heller(&[-1.0, 0.0, 1.0, 2.0]) > 0.5);
    }

    #[test]
    fn gonen_heller_fast_matches_direct_across_scales_and_ties() {
        let mut state: u64 = 0x5EED_CAFE_1234_5678;
        let mut next_f64 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let cases: Vec<Vec<f64>> = vec![
            (0..600).map(|_| (next_f64() - 0.5) * 6.0).collect(),
            (0..600).map(|_| (next_f64() - 0.5) * 0.2).collect(),
            (0..600)
                .map(|i| {
                    if i % 3 == 0 {
                        -50.0
                    } else {
                        (next_f64() - 0.5) * 20.0
                    }
                })
                .collect(),
            (0..600).map(|i| (i % 7) as f64 * 0.35 - 1.0).collect(),
            vec![42.0; 400],
            (0..400).map(|i| (i as f64 - 200.0) * 1e5).collect(),
        ];
        for eta in cases {
            let mut sorted = eta.clone();
            sorted.sort_unstable_by(f64::total_cmp);
            let direct = gonen_heller_direct(&sorted);
            let fast = gonen_heller(&eta);
            assert!(
                (fast - direct).abs() < 1e-13,
                "fast {fast} != direct {direct} (diff={})",
                (fast - direct).abs()
            );
        }
    }

    #[test]
    fn a_stronger_predictor_gives_a_larger_d() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = [1, 1, 1, 1, 1, 1, 1, 0];
        let strong = [1.0, 0.9, 0.8, 0.7, -0.7, -0.8, -0.9, -1.0];
        let weak = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.1, -0.3];
        let run = |eta: &[f64]| {
            royston(&RoystonInput {
                eta,
                time: &time,
                status: &status,
                entry_times: None,
                loglik: [-10.0, -8.0],
                n_event: 7,
                n_coef: 1,
                ties: true,
                adjust: false,
            })
            .unwrap()
        };
        let strong = run(&strong);
        let weak = run(&weak);
        assert!(strong.d > weak.d.abs());
        assert!(strong.r_d > weak.r_d);
        assert!(strong.se_d > 0.0);
        assert!((0.0..=1.0).contains(&strong.c_gh));
    }

    #[test]
    fn inputs_are_validated() {
        let bad = royston(&RoystonInput {
            eta: &[0.1, f64::NAN],
            time: &[1.0, 2.0],
            status: &[1, 1],
            entry_times: None,
            loglik: [-1.0, -1.0],
            n_event: 2,
            n_coef: 1,
            ties: true,
            adjust: false,
        });
        assert!(bad.is_err());
        let adjust = royston(&RoystonInput {
            eta: &[0.1, 0.2],
            time: &[1.0, 2.0],
            status: &[1, 1],
            entry_times: None,
            loglik: [-1.0, -1.0],
            n_event: 1,
            n_coef: 1,
            ties: true,
            adjust: true,
        });
        assert!(adjust.is_err());
    }
}
