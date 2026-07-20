pub const CHOLESKY_TOL: f64 = 1e-10;
pub const RIDGE_REGULARIZATION: f64 = 1e-6;
pub const NEAR_ZERO_MATRIX: f64 = 1e-10;
pub const TIME_EPSILON: f64 = 1e-9;
pub const PYEARS_TIME_EPSILON: f64 = 1e-8;
pub const CONVERGENCE_EPSILON: f64 = 1e-6;
pub const STRICT_EPSILON: f64 = 1e-5;
pub const CLOGIT_TOLERANCE: f64 = 1e-6;
pub const DIVISION_FLOOR: f64 = 1e-10;
pub const GAUSSIAN_ELIMINATION_TOL: f64 = 1e-12;
pub const DEFAULT_RANDOM_SEED: u64 = 42;
pub const DEFAULT_MAX_ITER: usize = 30;
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
pub const DEFAULT_BOOTSTRAP_SAMPLES: usize = 1000;

pub const Z_SCORE_80: f64 = 1.28;
pub const Z_SCORE_90: f64 = 1.645;
pub const Z_SCORE_95: f64 = 1.96;
pub const Z_SCORE_99: f64 = 2.576;

pub const TIED_PAIR_WEIGHT: f64 = 0.5;
pub const DEFAULT_CONCORDANCE: f64 = 0.5;

#[inline]
pub fn z_score_for_confidence(confidence_level: f64) -> f64 {
    if confidence_level >= 0.99 {
        Z_SCORE_99
    } else if confidence_level >= 0.95 {
        Z_SCORE_95
    } else if confidence_level >= 0.90 {
        Z_SCORE_90
    } else {
        Z_SCORE_80
    }
}

#[inline]
pub fn normal_ci(estimate: f64, standard_error: f64, z_score: f64) -> (f64, f64) {
    (
        estimate - z_score * standard_error,
        estimate + z_score * standard_error,
    )
}

#[inline]
pub fn clamped_normal_ci(
    estimate: f64,
    standard_error: f64,
    z_score: f64,
    lower_bound: f64,
    upper_bound: f64,
) -> (f64, f64) {
    let (lower, upper) = normal_ci(estimate, standard_error, z_score);
    (
        lower.clamp(lower_bound, upper_bound),
        upper.clamp(lower_bound, upper_bound),
    )
}

#[inline]
pub fn normal_ci_95(estimate: f64, standard_error: f64) -> (f64, f64) {
    normal_ci(estimate, standard_error, Z_SCORE_95)
}

#[inline]
pub fn clamped_normal_ci_95(
    estimate: f64,
    standard_error: f64,
    lower_bound: f64,
    upper_bound: f64,
) -> (f64, f64) {
    clamped_normal_ci(
        estimate,
        standard_error,
        Z_SCORE_95,
        lower_bound,
        upper_bound,
    )
}

#[inline]
pub fn exp_ci(log_estimate: f64, standard_error: f64, z_score: f64) -> (f64, f64) {
    let (lower, upper) = normal_ci(log_estimate, standard_error, z_score);
    (lower.exp(), upper.exp())
}

#[inline]
pub fn exp_ci_95(log_estimate: f64, standard_error: f64) -> (f64, f64) {
    exp_ci(log_estimate, standard_error, Z_SCORE_95)
}

#[inline]
pub fn normal_ci_bounds_95(estimates: &[f64], standard_errors: &[f64]) -> (Vec<f64>, Vec<f64>) {
    normal_ci_bounds(estimates, standard_errors, Z_SCORE_95)
}

#[inline]
pub fn normal_ci_bounds(
    estimates: &[f64],
    standard_errors: &[f64],
    z_score: f64,
) -> (Vec<f64>, Vec<f64>) {
    estimates
        .iter()
        .zip(standard_errors.iter())
        .map(|(&estimate, &standard_error)| normal_ci(estimate, standard_error, z_score))
        .unzip()
}

#[inline]
pub fn clamped_normal_ci_bounds_95(
    estimates: &[f64],
    standard_errors: &[f64],
    lower_bound: f64,
    upper_bound: f64,
) -> (Vec<f64>, Vec<f64>) {
    clamped_normal_ci_bounds(
        estimates,
        standard_errors,
        Z_SCORE_95,
        lower_bound,
        upper_bound,
    )
}

#[inline]
pub fn clamped_normal_ci_bounds(
    estimates: &[f64],
    standard_errors: &[f64],
    z_score: f64,
    lower_bound: f64,
    upper_bound: f64,
) -> (Vec<f64>, Vec<f64>) {
    estimates
        .iter()
        .zip(standard_errors.iter())
        .map(|(&estimate, &standard_error)| {
            clamped_normal_ci(estimate, standard_error, z_score, lower_bound, upper_bound)
        })
        .unzip()
}

#[inline]
pub fn exp_ci_bounds_95(log_estimates: &[f64], standard_errors: &[f64]) -> (Vec<f64>, Vec<f64>) {
    exp_ci_bounds(log_estimates, standard_errors, Z_SCORE_95)
}

#[inline]
pub fn exp_ci_bounds(
    log_estimates: &[f64],
    standard_errors: &[f64],
    z_score: f64,
) -> (Vec<f64>, Vec<f64>) {
    log_estimates
        .iter()
        .zip(standard_errors.iter())
        .map(|(&log_estimate, &standard_error)| exp_ci(log_estimate, standard_error, z_score))
        .unzip()
}

pub const PARALLEL_THRESHOLD_SMALL: usize = 100;
pub const PARALLEL_THRESHOLD_MEDIUM: usize = 500;
pub const PARALLEL_THRESHOLD_LARGE: usize = 1000;
pub const PARALLEL_THRESHOLD_XLARGE: usize = 10000;

pub const COX_MAX_ITER: usize = 20;
pub const COX_CONVERGENCE_TOLERANCE: f64 = 1e-9;
/// Relative information-matrix pivot tolerance used by Cox regression.
///
/// This is `f64::EPSILON.powf(0.75)`, matching the reference Cox control.
pub const COX_RANK_TOLERANCE: f64 = 1.818_989_403_545_856_5e-12;
pub const ITERATIVE_MAX_ITER: usize = 100;
pub const LINEAR_PRED_CLAMP_MIN: f64 = -20.0;
pub const LINEAR_PRED_CLAMP_MAX: f64 = 20.0;

pub const EXP_CLAMP_MIN: f64 = -100.0;
pub const EXP_CLAMP_MAX: f64 = 100.0;

#[inline]
pub fn exp_clamped(value: f64) -> f64 {
    value.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp()
}

#[inline]
pub fn exp_clamped_ci(log_estimate: f64, standard_error: f64, z_score: f64) -> (f64, f64) {
    let (lower, upper) = normal_ci(log_estimate, standard_error, z_score);
    (exp_clamped(lower), exp_clamped(upper))
}

pub const CONVERGENCE_FLAG: i32 = 1000;

pub const DEFAULT_CONFORMAL_COVERAGE: f64 = 0.9;
pub const DEFAULT_IPCW_TRIM: f64 = 0.01;
pub const IPCW_SURVIVAL_FLOOR: f64 = DEFAULT_IPCW_TRIM;
pub const DEFAULT_MIN_GROUP_SIZE: usize = 10;
pub const DEFAULT_WEIGHT_TRIM: f64 = 0.01;
pub const MAX_WEIGHT_RATIO: f64 = 100.0;

pub const LCG64_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
pub const LCG64_INCREMENT: u64 = 1;

pub const DEFAULT_ALPHA: f64 = 0.05;
pub const DEFAULT_POWER: f64 = 0.8;
pub const DEFAULT_ALLOCATION_RATIO: f64 = 1.0;
pub const DEFAULT_SIDED: usize = 2;

pub const CONCORDANCE_COUNT_SIZE: usize = 5;
pub const CONCORDANCE_COUNT_SIZE_EXTENDED: usize = 6;

pub const MAX_HALVING_ITERATIONS: usize = 10;
pub const STEP_HALVE_FACTOR: f64 = 0.5;
pub const STEP_DOUBLE_FACTOR: f64 = 2.0;

pub const HARTLEY_A1: f64 = 0.2316419;
pub const HARTLEY_NORM: f64 = 0.3989423;
pub const HARTLEY_B1: f64 = 0.3193815;
pub const HARTLEY_B2: f64 = -0.3565638;
pub const HARTLEY_B3: f64 = 1.781478;
pub const HARTLEY_B4: f64 = -1.821256;
pub const HARTLEY_B5: f64 = 1.330274;

pub const ROYSTON_KAPPA_FACTOR: f64 = 8.0;
pub const ROYSTON_VARIANCE_FACTOR: f64 = 6.0;

#[inline]
pub fn same_time(left: f64, right: f64) -> bool {
    (left - right).abs() < TIME_EPSILON
}
