//! Public re-exports of the crate's vectorised reductions.
//!
//! The implementation lives in `crate::internal::simd`; these aliases keep
//! the historical `*_simd` names used by callers and benchmarks.

pub use crate::internal::simd::{
    dot_product as dot_product_simd, mean as mean_simd, subtract_scalar as subtract_scalar_simd,
    sum_f64 as sum_simd, sum_of_squares as sum_of_squares_simd, variance as variance_simd,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_forward_to_internal_simd() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sum_simd(&values), crate::internal::simd::sum_f64(&values));
        assert_eq!(mean_simd(&values), 3.0);
        assert_eq!(sum_of_squares_simd(&values), 55.0);
        assert_eq!(dot_product_simd(&values, &values), 55.0);
        assert_eq!(variance_simd(&values), 2.5);
        assert_eq!(
            subtract_scalar_simd(&values, 1.0),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }
}
