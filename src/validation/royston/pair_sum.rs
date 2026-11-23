//! Gönen and Heller's pair sum, `Σ_{i<j} 1 / (1 + exp(s_i - s_j))` over sorted `s`,
//! in O(n p log n) work instead of n²/2 exponentials.
//!
//! The logistic kernel `g(d) = 1 / (1 + exp(-d))` is analytic in the strip
//! `|Im d| < π` and bounded by `2/√3` on `|Im d| ≤ 2π/3`, so over two value ranges of
//! width at most one, Chebyshev interpolation in each argument separates `g(x - y)`
//! to near machine precision with at most 19 nodes per range. The sum halves the
//! sorted index range: pairs within each half recurse, and pairs across the halves
//! are
//!
//! * exactly one each once the halves are [`SATURATION`] apart, where the kernel
//!   rounds to one,
//! * summed directly when there are at most [`DIRECT_PAIRS`] of them,
//! * summed through the separated expansions once both value ranges are at most
//!   [`MAX_WIDTH`] wide,
//!
//! and otherwise the wider side halves again. Every split is fixed by the input, so
//! the result does not depend on the thread count.

use rayon::join;
use std::f64::consts::PI;

/// `exp(-37) < 2^-53`, so `1 / (1 + exp(-d))` rounds to exactly one for `d ≥ 37`.
const SATURATION: f64 = 37.0;
/// Widest value range that a separated expansion covers.
const MAX_WIDTH: f64 = 1.0;
/// Index ranges at most this long sum their pairs directly.
const LEAF: usize = 64;
/// Cross sums with at most this many pairs are summed directly.
const DIRECT_PAIRS: usize = 4096;
/// Index ranges at least this long split their work across threads.
const PARALLEL_LEN: usize = 4096;
/// Semi-minor axis of the Bernstein ellipse the interpolation bound uses: inside the
/// strip `|Im d| ≤ 2π/3`, `|1 + exp(-d)| ≥ √3/2`.
const STRIP: f64 = 2.0 * PI / 3.0;
/// `max |g|` on that strip, `2/√3`.
const BOUND: f64 = 1.154_700_538_379_251_5;
/// Interpolation error allowed per pair, `2^-56`.
const TARGET: f64 = 1.387_778_780_781_445_7e-17;
const MAX_NODES: usize = 32;

/// The kernel as the direct sum evaluates it, for `lower ≤ upper`.
#[inline]
fn kernel(lower: f64, upper: f64) -> f64 {
    1.0 / (1.0 + (lower - upper).exp())
}

/// `Σ_{i<j} 1 / (1 + exp(sorted[i] - sorted[j]))` for ascending finite `sorted`.
pub(super) fn logistic_pair_sum(sorted: &[f64]) -> f64 {
    within(sorted)
}

fn within(s: &[f64]) -> f64 {
    if s.len() <= LEAF {
        return direct_within(s);
    }
    let (a, b) = s.split_at(s.len() / 2);
    if s.len() >= PARALLEL_LEN {
        let ((lower, upper), across) = join(|| join(|| within(a), || within(b)), || cross(a, b));
        lower + upper + across
    } else {
        within(a) + within(b) + cross(a, b)
    }
}

/// Neumaier's compensated sum. Tied or clustered predictors give long runs of equal
/// terms, whose plain running sum rounds the same way at every step.
#[derive(Clone, Copy, Default)]
struct Compensated {
    sum: f64,
    correction: f64,
}

impl Compensated {
    #[inline]
    fn add(&mut self, value: f64) {
        let total = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - total) + value
        } else {
            (value - total) + self.sum
        };
        self.sum = total;
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

fn direct_within(s: &[f64]) -> f64 {
    let mut total = Compensated::default();
    for (i, &lower) in s.iter().enumerate() {
        for &upper in &s[i + 1..] {
            total.add(kernel(lower, upper));
        }
    }
    total.value()
}

fn direct_cross(a: &[f64], b: &[f64]) -> f64 {
    let mut total = Compensated::default();
    for &lower in a {
        for &upper in b {
            total.add(kernel(lower, upper));
        }
    }
    total.value()
}

/// The pairs `(i, j)` with `i` in `a` and `j` in `b`, every value of `a` at most every
/// value of `b`.
fn cross(a: &[f64], b: &[f64]) -> f64 {
    let (a_lo, a_hi) = (a[0], a[a.len() - 1]);
    let (b_lo, b_hi) = (b[0], b[b.len() - 1]);
    if b_lo - a_hi >= SATURATION {
        return a.len() as f64 * b.len() as f64;
    }
    if a.len().saturating_mul(b.len()) <= DIRECT_PAIRS {
        return direct_cross(a, b);
    }
    let (a_width, b_width) = (a_hi - a_lo, b_hi - b_lo);
    if a_width <= MAX_WIDTH && b_width <= MAX_WIDTH {
        return Expansion::new(a).cross(&Expansion::new(b));
    }
    let parallel = a.len() + b.len() >= PARALLEL_LEN;
    let (first, second) = if a_width >= b_width {
        let (a1, a2) = a.split_at(a.len() / 2);
        if parallel {
            join(|| cross(a1, b), || cross(a2, b))
        } else {
            (cross(a1, b), cross(a2, b))
        }
    } else {
        let (b1, b2) = b.split_at(b.len() / 2);
        if parallel {
            join(|| cross(a, b1), || cross(a, b2))
        } else {
            (cross(a, b1), cross(a, b2))
        }
    };
    first + second
}

/// Chebyshev nodes for a sorted value range, relative to its centre, and the weights
/// `W_k = Σ_i ℓ_k(s_i)` that turn `Σ_i f(s_i)` into `Σ_k W_k f(centre + node_k)`.
/// Local coordinates keep the nodes distinct when the range is narrow next to the
/// magnitude of its values.
struct Expansion {
    centre: f64,
    nodes: [f64; MAX_NODES],
    weights: [f64; MAX_NODES],
    len: usize,
}

impl Expansion {
    fn new(values: &[f64]) -> Self {
        let (lo, hi) = (values[0], values[values.len() - 1]);
        let centre = 0.5 * lo + 0.5 * hi;
        let radius = 0.5 * (hi - lo);
        let mut expansion = Self {
            centre,
            nodes: [0.0; MAX_NODES],
            weights: [0.0; MAX_NODES],
            len: node_count(radius),
        };
        if expansion.len == 1 {
            expansion.weights[0] = values.len() as f64;
            return expansion;
        }
        let p = expansion.len;
        let mut barycentric = [0.0; MAX_NODES];
        for (k, (node, weight)) in expansion.nodes[..p]
            .iter_mut()
            .zip(&mut barycentric[..p])
            .enumerate()
        {
            let theta = (2 * k + 1) as f64 * PI / (2 * p) as f64;
            *node = radius * theta.cos();
            *weight = if k % 2 == 0 {
                theta.sin()
            } else {
                -theta.sin()
            };
        }
        let mut terms = [0.0; MAX_NODES];
        let mut weights = [Compensated::default(); MAX_NODES];
        for &value in values {
            let local = value - centre;
            let mut total = 0.0;
            let mut exact = None;
            for k in 0..p {
                let diff = local - expansion.nodes[k];
                if diff == 0.0 {
                    exact = Some(k);
                    break;
                }
                terms[k] = barycentric[k] / diff;
                total += terms[k];
            }
            match exact {
                Some(k) => weights[k].add(1.0),
                None => {
                    for (weight, term) in weights[..p].iter_mut().zip(&terms[..p]) {
                        weight.add(term / total);
                    }
                }
            }
        }
        for (weight, total) in expansion.weights[..p].iter_mut().zip(&weights) {
            *weight = total.value();
        }
        expansion
    }

    /// `Σ_k Σ_m W_k V_m g(x_m - y_k)` for this expansion's `y` below `upper`'s `x`.
    fn cross(&self, upper: &Self) -> f64 {
        let gap = upper.centre - self.centre;
        let mut total = Compensated::default();
        for (&y, &weight) in self.nodes[..self.len].iter().zip(&self.weights) {
            for (&x, &other) in upper.nodes[..upper.len].iter().zip(&upper.weights) {
                total.add(weight * other / (1.0 + (-(gap + (x - y))).exp()));
            }
        }
        total.value()
    }
}

/// Nodes for a range of half-width `radius`: the interpolant of degree `n` errs by at
/// most `4 M ρ^-n / (ρ - 1)` with `ρ = (b + √(b² + r²)) / r` for the ellipse of
/// semi-minor axis `b` = [`STRIP`] on which `|g| ≤ M` = [`BOUND`].
fn node_count(radius: f64) -> usize {
    if radius < f64::MIN_POSITIVE {
        return 1;
    }
    let rho = (STRIP + STRIP.hypot(radius)) / radius;
    let degree = (4.0 * BOUND / ((rho - 1.0) * TARGET)).ln() / rho.ln();
    (degree.ceil().max(0.0) as usize + 1).clamp(2, MAX_NODES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::dist::qnorm;

    /// Every pair's kernel, summed with compensation: within a few ulps of the exact
    /// sum of the rounded terms.
    fn direct(sorted: &[f64]) -> f64 {
        direct_within(sorted)
    }

    /// Deterministic standard normal scores.
    fn normals(n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = (phase + i as f64 * 0.618_033_988_749_894_9).fract();
                qnorm(u.clamp(1e-12, 1.0 - 1e-12), true, false)
            })
            .collect()
    }

    fn sorted(mut values: Vec<f64>) -> Vec<f64> {
        values.sort_by(f64::total_cmp);
        values
    }

    fn cases() -> Vec<(&'static str, Vec<f64>)> {
        let n = 3000;
        let z = normals(n, 0.1);
        let mut dominant = vec![0.3; n - 20];
        dominant.extend(normals(20, 0.7).iter().map(|v| 40.0 * v));
        let mut clusters: Vec<f64> = z[..n / 2].iter().map(|v| 0.1 * v).collect();
        clusters.extend(z[n / 2..].iter().map(|v| 50.0 + 0.1 * v));
        let gaps: Vec<f64> = (0..n)
            .map(|i| [0.0, 36.9, 37.1, 74.0][i % 4] + 1e-3 * z[i])
            .collect();
        vec![
            ("normal sd 0.5", z.iter().map(|v| 0.5 * v).collect()),
            ("normal sd 3", z.iter().map(|v| 3.0 * v).collect()),
            ("normal sd 30", z.iter().map(|v| 30.0 * v).collect()),
            (
                "uniform +-200",
                (0..n)
                    .map(|i| 400.0 * (i as f64 * 0.754_877_666_246_692_7).fract() - 200.0)
                    .collect(),
            ),
            (
                "five tied values",
                (0..n).map(|i| (i * 7 % 5) as f64 * 0.7).collect(),
            ),
            (
                "four close tied values",
                (0..2 * n)
                    .map(|i| [-0.02, 0.01, 0.035, 0.06][i % 4])
                    .collect(),
            ),
            ("all equal", vec![1.25; n]),
            ("two far clusters", clusters),
            ("dominant cluster and outliers", dominant),
            ("lognormal", z.iter().map(|v| (1.5 * v).exp()).collect()),
            ("large offset", z.iter().map(|v| 1e4 + v).collect()),
            (
                "narrow at a large offset",
                z.iter().map(|v| 1e4 + 1e-11 * v).collect(),
            ),
            ("tiny spread", z.iter().map(|v| 1e-9 * v).collect()),
            ("gaps near saturation", gaps),
            ("short", vec![-0.4, 0.2]),
            ("one leaf and one", normals(LEAF + 1, 0.3)),
            ("parallel split", normals(PARALLEL_LEN + 3, 0.9)),
        ]
    }

    #[test]
    fn matches_the_direct_pair_sum() {
        for (label, values) in cases() {
            let values = sorted(values);
            let expected = direct(&values);
            let actual = logistic_pair_sum(&values);
            let error = (actual - expected).abs() / expected;
            assert!(error <= 1e-14, "{label}: relative error {error:e}");
        }
    }

    #[test]
    fn ties_and_saturated_halves_are_exact() {
        let tied = vec![2.5; 5000];
        assert_eq!(logistic_pair_sum(&tied), 0.5 * (5000.0 * 4999.0 / 2.0));
        let mut apart = vec![-20.0; 3000];
        apart.extend(vec![20.0; 3000]);
        let within_halves = 2.0 * 0.5 * (3000.0 * 2999.0 / 2.0);
        assert_eq!(logistic_pair_sum(&apart), within_halves + 3000.0 * 3000.0);
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let values = sorted(normals(20_000, 0.2));
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| logistic_pair_sum(&values))
        };
        let one = run(1);
        assert_eq!(one.to_bits(), run(4).to_bits());
        assert_eq!(one.to_bits(), run(7).to_bits());
    }

    #[test]
    fn node_counts_shrink_with_the_range() {
        assert_eq!(node_count(0.0), 1);
        assert_eq!(node_count(f64::MIN_POSITIVE / 2.0), 1);
        assert_eq!(node_count(1e-300), 2);
        let counts: Vec<usize> = [1e-6, 1e-3, 0.05, 0.25, 0.5]
            .iter()
            .map(|&radius| node_count(radius))
            .collect();
        assert!(counts.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(counts[4] <= 19, "{counts:?}");
    }
}
