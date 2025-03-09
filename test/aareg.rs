use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use survival::aareg::{AalenOptions, AalenResult};
use survival::data::{Surv, TimeType};

fn assert_aeq(actual: &Array1<f64>, expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(a, e, epsilon = tol);
    }
}

fn assert_matrix_aeq(actual: &Array2<f64>, expected: &[&[f64]], tol: f64) {
    assert_eq!(actual.dim(), (expected.len(), expected[0].len()));
    for (row_idx, row) in expected.iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            assert_abs_diff_eq!(actual[(row_idx, col_idx)], val, epsilon = tol);
        }
    }
}

#[test]
fn test_aareg_basic() {
    let time = vec![4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
    let status = vec![1, 0, 1, 0, 1, 1, 0];
    let x = vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let wt = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let surv = Surv::new(time.clone(), status.clone());

    let options = AalenOptions::default().test_type("aalen");
    let fit = surv
        .aareg(&x, Some(&options))
        .expect("Model fitting failed");

    assert_eq!(fit.times(), &[1.0, 2.0, 2.0]);
    assert_eq!(fit.nrisk(), &[6, 4, 4]);

    let expected_coef = array![[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0], [1.0, -1.0 / 3.0]];
    assert_matrix_aeq(fit.coefficients(), &expected_coef, 1e-6);

    let expected_tweight = array![[3.0, 3.0], [3.0 / 2.0, 3.0 / 4.0], [3.0 / 4.0, 3.0 / 4.0]];
    assert_matrix_aeq(fit.tweights(), &expected_tweight, 1e-6);

    assert_aeq(fit.test_statistic(), &[1.0, 1.0], 1e-6);

    let options = AalenOptions::default().weights(wt).test_type("aalen");
    let fit_weighted = surv
        .aareg(&x, Some(&options))
        .expect("Weighted model failed");

    assert_eq!(fit_weighted.times(), &[1.0, 2.0, 2.0]);
    assert_eq!(fit_weighted.nrisk(), &[21, 16, 16]);
}

#[test]
fn test_aareg_lung_data() {}

#[test]
fn test_dfbeta_calculations() {
    let time = vec![4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
    let status = vec![1, 0, 1, 0, 1, 1, 0];
    let x = vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0];

    let surv = Surv::new(time, status);
    let options = AalenOptions::default().dfbeta(true);
    let fit = surv
        .aareg(&x, Some(&options))
        .expect("DFBETA calculation failed");

    let expected_dfbeta = array![
        [0.0, 0.0],
        [0.0, 0.0],
        [2.0 / 9.0, -1.0 / 9.0],
        [-1.0 / 9.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ];
    assert_matrix_aeq(fit.dfbeta(), &expected_dfbeta, 1e-6);
}

#[test]
fn test_summary_consistency() {
    let time = vec![4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0];
    let status = vec![1, 0, 1, 0, 1, 1, 0];
    let x = vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0];

    let surv = Surv::new(time, status);
    let fit = surv.aareg(&x, None).expect("Model fitting failed");
    let summary = fit.summary();

    assert_aeq(fit.test_statistic(), summary.test_statistic(), 1e-6);
    assert_aeq(fit.test_variance(), summary.test_variance(), 1e-6);
}
