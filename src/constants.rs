pub const CHOLESKY_TOL: f64 = 1e-10;
pub const RIDGE_REGULARIZATION: f64 = 1e-6;
pub const SYMMETRY_TOL: f64 = 1e-10;
pub const NEAR_ZERO_MATRIX: f64 = 1e-10;
pub const TIME_EPSILON: f64 = 1e-9;
pub const PYEARS_TIME_EPSILON: f64 = 1e-8;
pub const CONVERGENCE_EPSILON: f64 = 1e-6;
pub const STRICT_EPSILON: f64 = 1e-5;
pub const CLOGIT_TOLERANCE: f64 = 1e-6;
pub const DIVISION_FLOOR: f64 = 1e-10;
pub const GAUSSIAN_ELIMINATION_TOL: f64 = 1e-12;
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

pub const PARALLEL_THRESHOLD_SMALL: usize = 100;
pub const PARALLEL_THRESHOLD_MEDIUM: usize = 500;
pub const PARALLEL_THRESHOLD_LARGE: usize = 1000;
pub const PARALLEL_THRESHOLD_XLARGE: usize = 10000;

pub const COX_MAX_ITER: usize = 25;
pub const ITERATIVE_MAX_ITER: usize = 100;

pub const EXP_CLAMP_MIN: f64 = -100.0;
pub const EXP_CLAMP_MAX: f64 = 100.0;

pub const CONVERGENCE_FLAG: i32 = 1000;

#[cfg(test)]
pub const TEST_STRICT_TOL: f64 = 1e-4;

#[cfg(test)]
pub const TEST_STANDARD_TOL: f64 = 1e-3;

#[cfg(test)]
pub const TEST_LOOSE_TOL: f64 = 1e-2;

pub const DEFAULT_CONFORMAL_COVERAGE: f64 = 0.9;
pub const DEFAULT_IPCW_TRIM: f64 = 0.01;
