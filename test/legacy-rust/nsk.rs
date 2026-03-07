use approx::assert_abs_diff_eq;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix};
use rand::prelude::*;
use rand_distr::Normal;

type DynMatrix = OMatrix<f64, Dynamic, Dynamic>;
type DynVector = DVector<f64>;

fn main() {
    let mut rng = thread_rng();
    let n = 500;
    let xx: Vec<f64> = (0..n).map(|_| rng.gen_range(1.0..100.0)).collect();
    let yy: Vec<f64> = xx
        .iter()
        .map(|&x| 10.0 * x.ln() + rng.sample(Normal::new(0.0, 2.0).unwrap()))
        .collect();

    let df = 4;
    let ns_design = design_matrix_ns(&xx, df);
    let fit1_coef = fit_linear_regression(&ns_design, &yy);

    let b = 0.0;
    let nsk_design = design_matrix_nsk(&xx, df, b);
    let fit2_coef = fit_linear_regression(&nsk_design, &yy);

    let pred1 = predict(&ns_design, &fit1_coef);
    let pred2 = predict(&nsk_design, &fit2_coef);
    assert_predictions_equal(&pred1, &pred2, 1e-6);

    let all_knots = compute_all_knots(&xx, df);
    let pred_knot = predict_knots(&all_knots, &fit1_coef, df);
    let expected_coef = compute_expected_coefficients(&pred_knot);
    assert_coefficients_equal(&expected_coef, &fit2_coef.as_slice()[1..], 1e-6);
}

fn design_matrix_ns(x: &[f64], df: usize) -> DynMatrix {
    let n = x.len();
    let mut matrix = DMatrix::zeros(n, df + 1); 
    matrix.column_mut(0).fill(1.0);
    for (i, &xi) in x.iter().enumerate() {
        for j in 1..=df {
            matrix[(i, j)] = xi.powi(j as i32);
        }
    }
    matrix
}

fn design_matrix_nsk(x: &[f64], df: usize, b: f64) -> DynMatrix {
    let n = x.len();
    let mut matrix = DMatrix::zeros(n, df + 1);
    matrix.column_mut(0).fill(1.0);
    for (i, &xi) in x.iter().enumerate() {
        for j in 1..=df {
            matrix[(i, j)] = (xi - b).powi(j as i32);
        }
    }
    matrix
}

fn fit_linear_regression(x: &DynMatrix, y: &[f64]) -> DynVector {
    let y_vec = DVector::from_vec(y.to_vec());
    let x_t = x.transpose();
    let x_t_x = &x_t * x;
    let x_t_y = x_t * y_vec;
    x_t_x.try_inverse().unwrap() * x_t_y
}

fn predict(x: &DynMatrix, coef: &DynVector) -> Vec<f64> {
    (x * coef).iter().cloned().collect()
}

fn compute_all_knots(x: &[f64], df: usize) -> Vec<f64> {
    let min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    vec![min, max]
}

fn predict_knots(knots: &[f64], coef: &DynVector, df: usize) -> Vec<f64> {
    knots
        .iter()
        .map(|&k| {
            let mut sum = coef[0];
            for j in 1..=df {
                sum += coef[j] * k.powi(j as i32);
            }
            sum
        })
        .collect()
}

fn compute_expected_coefficients(pred_knot: &[f64]) -> Vec<f64> {
    pred_knot
        .iter()
        .skip(1)
        .map(|&p| p - pred_knot[0])
        .collect()
}

fn assert_predictions_equal(pred1: &[f64], pred2: &[f64], epsilon: f64) {
    for (a, b) in pred1.iter().zip(pred2.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = epsilon);
    }
}

fn assert_coefficients_equal(coef1: &[f64], coef2: &[f64], epsilon: f64) {
    for (a, b) in coef1.iter().zip(coef2.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = epsilon);
    }
}
