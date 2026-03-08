use approx::assert_abs_diff_eq;
use rust_survival::{
    coxph::{CoxPHModel, ResidualType},
    survival::Surv,
};

fn aeq(x: &[f64], y: &[f64]) {
    assert_eq!(x.len(), y.len(), "Vectors have different lengths");
    for (i, (&a, &b)) in x.iter().zip(y.iter()).enumerate() {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6, "Mismatch at index {}", i);
    }
}

fn main() {
    let testw1 = vec![
        (1.0, 1, 2.0, 1.0),
        (1.0, 0, 0.0, 2.0),
        (2.0, 1, 1.0, 3.0),
        (2.0, 1, 1.0, 4.0),
        (2.0, 1, 0.0, 3.0),
        (2.0, 0, 1.0, 2.0),
        (3.0, 0, 0.0, 1.0),
        (4.0, 1, 1.0, 2.0),
        (5.0, 0, 0.0, 1.0),
    ];

    let mut testw2 = Vec::new();
    for (i, &(time, status, x, wt)) in testw1.iter().enumerate() {
        for _ in 0..(wt as usize) {
            testw2.push((time, status, x, i + 1));
        }
    }

    let mut fit0 = CoxPHModel::new()
        .entry(Surv::new(
            testw1.iter().map(|&(t, ..)| t).collect(),
            testw1.iter().map(|&(_, s, ..)| s).collect(),
        ))
        .covariate(testw1.iter().map(|&(_, _, x, _)| x).collect())
        .weights(testw1.iter().map(|&(_, _, _, wt)| wt).collect())
        .ties("breslow")
        .max_iter(0)
        .fit()
        .unwrap();

    let mut fit0b = CoxPHModel::new()
        .entry(Surv::new(
            testw2.iter().map(|&(t, ..)| t).collect(),
            testw2.iter().map(|&(_, s, ..)| s).collect(),
        ))
        .covariate(testw2.iter().map(|&(_, _, x, _)| x).collect())
        .ties("breslow")
        .max_iter(0)
        .fit()
        .unwrap();

    let texp_0 = texp(0.0);
    let statuses: Vec<f64> = testw1.iter().map(|&(_, s, ..)| s as f64).collect();
    let expected_martingale: Vec<f64> = statuses
        .iter()
        .zip(texp_0.iter())
        .map(|(&s, &e)| s - e)
        .collect();

    let martingale_resid = fit0.residuals(ResidualType::Martingale);
    aeq(&martingale_resid, &expected_martingale);
}

fn texp(beta: f64) -> Vec<f64> {
    let r = beta.exp();
    let t1 = 1.0 / (r.powi(2) + 11.0 * r + 7.0);
    let t2 = 10.0 / (11.0 * r + 5.0);
    let t3 = 2.0 / (2.0 * r + 1.0);

    let temp = vec![t1, t1 + t2, t1 + t2 + t3];
    let temp_indices = vec![0, 0, 1, 1, 1, 1, 1, 2, 2];

    let factors = vec![r.powi(2), 1.0, r, r, 1.0, r, 1.0, r, 1.0];

    temp_indices
        .iter()
        .zip(factors.iter())
        .map(|(&i, &f)| f * temp[i])
        .collect()
}
