#![allow(dead_code)]
use pulp::{Arch, Simd, WithSimd};

pub fn sum_f64(data: &[f64]) -> f64 {
    struct Sum<'a>(&'a [f64]);

    impl WithSimd for Sum<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (head, tail) = S::as_simd_f64s(self.0);
            let mut acc = simd.splat_f64s(0.0);
            for &chunk in head {
                acc = simd.add_f64s(acc, chunk);
            }
            simd.reduce_sum_f64s(acc) + tail.iter().sum::<f64>()
        }
    }

    Arch::new().dispatch(Sum(data))
}

pub fn weighted_squared_diff_sum(predictions: &[f64], outcomes: &[f64], weights: &[f64]) -> f64 {
    struct WeightedSquaredDiff<'a>(&'a [f64], &'a [f64], &'a [f64]);

    impl WithSimd for WeightedSquaredDiff<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (pred_head, pred_tail) = S::as_simd_f64s(self.0);
            let (out_head, out_tail) = S::as_simd_f64s(self.1);
            let (wt_head, wt_tail) = S::as_simd_f64s(self.2);

            let mut acc = simd.splat_f64s(0.0);
            for ((&p, &o), &w) in pred_head.iter().zip(out_head.iter()).zip(wt_head.iter()) {
                let diff = simd.sub_f64s(p, o);
                let sq = simd.mul_f64s(diff, diff);
                acc = simd.mul_add_f64s(sq, w, acc);
            }

            let mut scalar_sum = simd.reduce_sum_f64s(acc);
            for ((&p, &o), &w) in pred_tail.iter().zip(out_tail.iter()).zip(wt_tail.iter()) {
                let diff = p - o;
                scalar_sum += w * diff * diff;
            }
            scalar_sum
        }
    }

    Arch::new().dispatch(WeightedSquaredDiff(predictions, outcomes, weights))
}

pub fn squared_diff_sum(predictions: &[f64], outcomes: &[f64]) -> f64 {
    struct SquaredDiff<'a>(&'a [f64], &'a [f64]);

    impl WithSimd for SquaredDiff<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (pred_head, pred_tail) = S::as_simd_f64s(self.0);
            let (out_head, out_tail) = S::as_simd_f64s(self.1);

            let mut acc = simd.splat_f64s(0.0);
            for (&p, &o) in pred_head.iter().zip(out_head.iter()) {
                let diff = simd.sub_f64s(p, o);
                acc = simd.mul_add_f64s(diff, diff, acc);
            }

            let mut scalar_sum = simd.reduce_sum_f64s(acc);
            for (&p, &o) in pred_tail.iter().zip(out_tail.iter()) {
                let diff = p - o;
                scalar_sum += diff * diff;
            }
            scalar_sum
        }
    }

    Arch::new().dispatch(SquaredDiff(predictions, outcomes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        struct DotProduct<'a>(&'a [f64], &'a [f64]);

        impl WithSimd for DotProduct<'_> {
            type Output = f64;

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let (a_head, a_tail) = S::as_simd_f64s(self.0);
                let (b_head, b_tail) = S::as_simd_f64s(self.1);

                let mut acc = simd.splat_f64s(0.0);
                for (&a_chunk, &b_chunk) in a_head.iter().zip(b_head.iter()) {
                    acc = simd.mul_add_f64s(a_chunk, b_chunk, acc);
                }

                let mut scalar_sum = simd.reduce_sum_f64s(acc);
                for (&a_val, &b_val) in a_tail.iter().zip(b_tail.iter()) {
                    scalar_sum += a_val * b_val;
                }
                scalar_sum
            }
        }

        Arch::new().dispatch(DotProduct(a, b))
    }

    fn min_max_f64(data: &[f64]) -> (f64, f64) {
        struct MinMax<'a>(&'a [f64]);

        impl WithSimd for MinMax<'_> {
            type Output = (f64, f64);

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                if self.0.is_empty() {
                    return (f64::INFINITY, f64::NEG_INFINITY);
                }

                let (head, tail) = S::as_simd_f64s(self.0);

                let mut min_acc = simd.splat_f64s(f64::INFINITY);
                let mut max_acc = simd.splat_f64s(f64::NEG_INFINITY);

                for &chunk in head {
                    min_acc = simd.min_f64s(min_acc, chunk);
                    max_acc = simd.max_f64s(max_acc, chunk);
                }

                let mut min_val = simd.reduce_min_f64s(min_acc);
                let mut max_val = simd.reduce_max_f64s(max_acc);

                for &val in tail {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }

                (min_val, max_val)
            }
        }

        Arch::new().dispatch(MinMax(data))
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((sum_f64(&data) - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        assert!((dot_product(&a, &b) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        assert!((squared_diff_sum(&a, &b) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (min, max) = min_max_f64(&data);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 9.0).abs() < 1e-10);
    }
}
