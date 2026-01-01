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
pub const CF_FLOOR: f64 = 1e-30;
pub const GAMMA_EPSILON: f64 = 1e-10;
pub const QR_TOLERANCE: f64 = 1e-7;
pub const DEFAULT_MAX_ITER: usize = 30;
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
pub const DEFAULT_BOOTSTRAP_SAMPLES: usize = 1000;

#[cfg(test)]
pub const TEST_STRICT_TOL: f64 = 1e-4;

#[cfg(test)]
pub const TEST_STANDARD_TOL: f64 = 1e-3;

#[cfg(test)]
pub const TEST_LOOSE_TOL: f64 = 1e-2;
