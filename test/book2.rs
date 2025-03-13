use approx::assert_abs_diff_eq;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
struct Observation {
    time: f64,
    status: Option<f64>,
    x: f64,
}

#[derive(Debug)]
struct ByhandResult {
    loglik: f64,
    u: f64,
    imat: f64,
    xbar: Array1<f64>,
    haz: Array1<f64>,
    mart: Array1<f64>,
    score: Array1<f64>,
    scho: Array1<f64>,
    surv: Array1<f64>,
    var: Array1<f64>,
}

fn byhand(beta: f64, newx: f64) -> ByhandResult {
    let r = beta.exp();

    let term1 = 3.0 * r + 3.0;
    let term2 = (r + 5.0) / 2.0;
    let term3 = r + 3.0;
    let loglik = 2.0 * beta - (term1.ln() + term2.ln() + term3.ln());

    let numerator_u = 30.0 + 23.0 * r - r.powi(3);
    let denominator_u = (r + 1.0) * (r + 3.0) * (r + 5.0);
    let u = numerator_u / denominator_u;

    let tfun = |x: f64| x - x.powi(2);
    let imat = tfun(r / (r + 1.0)) + tfun(r / (r + 3.0)) + tfun(r / (r + 5.0));

    let mut wtmat = Array2::<f64>::zeros((6, 4));
    wtmat
        .column_mut(0)
        .assign(&array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    wtmat
        .column_mut(1)
        .assign(&array![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    wtmat
        .column_mut(2)
        .assign(&array![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
    wtmat
        .column_mut(3)
        .assign(&array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

    let diag = array![r, r, r, 1.0, 1.0, 1.0];
    let wtmat_scaled = &wtmat * &diag.slice(s![.., NewAxis]);

    let x = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let status = array![1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let xbar: Array1<f64> = (0..4)
        .map(|col| {
            let column = wtmat_scaled.column(col);
            let numerator = x.dot(&column);
            let denominator = column.sum();
            numerator / denominator
        })
        .collect();

    let haz: Array1<f64> = (0..4)
        .map(|col| 1.0 / wtmat_scaled.column(col).sum())
        .collect();

    let hazmat = &wtmat_scaled * &haz.slice(s![NewAxis, ..]);

    let mart = &status - hazmat.sum_axis(Axis(1));

    let a = r + 1.0;
    let b = r + 3.0;
    let d = r + 5.0;

    let score1 = (2.0 * r + 3.0) / (3.0 * a.powi(2));
    let score2 = -r / (3.0 * a.powi(2));
    let score3_numerator = 675.0 + r * (1305.0 + r * (756.0 + r * (-4.0 + r * (-79.0 - 13.0 * r))));
    let score3_denominator = 3.0 * (a * b * d).powi(2);
    let score3 = score3_numerator / score3_denominator;
    let score4 = r * (1.0 / (3.0 * a.powi(2)) - a / (2.0 * b.powi(2)) - b / (2.0 * d.powi(2)));
    let score5_6_numerator = 2.0 * r * (177.0 + r * (282.0 + r * (182.0 + r * (50.0 + 5.0 * r))));
    let score5_6_denominator = 3.0 * (a * b * d).powi(2);
    let score5 = score5_6_numerator / score5_6_denominator;
    let score6 = score5_6_numerator / score5_6_denominator;
    let score = array![score1, score2, score3, score4, score5, score6];

    let d_mean = (xbar[1] + xbar[2]) / 2.0;
    let scho = array![1.0 / (r + 1.0), 1.0 - d_mean, 0.0 - d_mean, 0.0];

    let cum_haz: Array1<f64> = haz
        .iter()
        .scan(0.0, |acc, &h| {
            *acc += h;
            Some(*acc)
        })
        .collect();
    let surv_exp = (-&cum_haz * beta.exp() * newx).exp();
    let surv = array![surv_exp[0], surv_exp[2], surv_exp[3]];

    let varhaz_g: Array1<f64> = haz
        .iter()
        .scan(0.0, |acc, &h| {
            *acc += h.powi(2);
            Some(*acc)
        })
        .collect();
    let varhaz_d: Array1<f64> = haz
        .iter()
        .zip(xbar.iter())
        .scan(0.0, |acc, (&h, &xb)| {
            *acc += (newx - xb) * h;
            Some(*acc)
        })
        .collect();
    let varhaz = (&varhaz_g + (&varhaz_d.mapv(|d| d.powi(2)) / imat)) * (2.0 * beta * newx).exp();
    let var = array![varhaz[0], varhaz[2], varhaz[3]];

    ByhandResult {
        loglik,
        u,
        imat,
        xbar,
        haz,
        mart,
        score,
        scho,
        surv,
        var,
    }
}

fn main() {
    let truth0 = byhand(0.0, 0.0);

    let expected_loglik = 2.0 * 0.0 - (6.0_f64.ln() + 3.0_f64.ln() + 4.0_f64.ln());
    assert_abs_diff_eq!(truth0.loglik, expected_loglik, epsilon = 1e-10);

    let r = 1.0;
    let tfun = |x: f64| x - x.powi(2);
    let expected_imat = tfun(r / (r + 1.0)) + tfun(r / (r + 3.0)) + tfun(r / (r + 5.0));
    assert_abs_diff_eq!(truth0.imat, expected_imat, epsilon = 1e-10);

    let expected_mart = array![-0.75, 0.0, 5.0 / 6.0, -1.0 / 6.0, 5.0 / 12.0, 5.0 / 12.0];
    for (actual, expected) in truth0.mart.iter().zip(expected_mart.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }

    println!("All tests passed!");
}
