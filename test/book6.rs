use ndarray::{array, Array1, Array2, ArrayView1, Axis};

#[derive(Debug)]
struct ByHandResults {
    loglik: f64,
    imat: f64,
    hazard: Array1<f64>,
    xbar: Array1<f64>,
    mart: Array1<f64>,
    expected: Array1<f64>,
    score: Array1<f64>,
    schoen: Array1<f64>,
    varhaz: Array1<f64>,
}

fn byhand(beta: f64, newx: f64) -> ByHandResults {
    let r = beta.exp();
    let x = array![2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let status = array![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let wt = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0];

    let a = 7.0 * r + 3.0;
    let b = 4.0 * r + 2.0;
    let loglik = 11.0 * beta
        - ((r.powi(2) + 11.0 * r + 7.0).ln()
            + 10.0 * (11.0 * r + 5.0).ln() / 3.0
            + 10.0 * (a * 2.0 / 3.0 + b).ln() / 3.0
            + 10.0 * (a / 3.0 + b).ln() / 3.0
            + 2.0 * (2.0 * r + 1.0).ln());

    let hazard = array![
        1.0 / (r.powi(2) + 11.0 * r + 7.0),
        10.0 / (3.0 * (11.0 * r + 5.0)),
        10.0 / (3.0 * (a * 2.0 / 3.0 + b)),
        10.0 / (3.0 * (a / 3.0 + b)),
        2.0 / (2.0 * r + 1.0)
    ];

    let temp = array![
        hazard[0],
        hazard[0] + hazard[1] + hazard[2] * 2.0 / 3.0 + hazard[3] / 3.0,
        hazard.sum() - hazard[4],
        hazard.sum()
    ];

    let risk = array![r.powi(2), 1.0, r, r, 1.0, r, 1.0, r, 1.0];
    let expected = risk
        * array![temp[0], temp[0], temp[1], temp[1], temp[1], temp[2], temp[2], temp[3], temp[3]];

    let riskmat = array![
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        [1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        [1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let wtmat = riskmat * wt.slice(ndarray::s![.., ndarray::NewAxis]);

    let x_weighted = x.clone().insert_axis(Axis(1)) * &wtmat;
    let xbar = x_weighted.sum_axis(Axis(0)) / wtmat.sum_axis(Axis(0));

    let imat = (4.0 * r.powi(2) + 11.0 * r) * hazard[0] - xbar[0].powi(2)
        + 10.0 * (xbar[1..4].mean().unwrap() - xbar[1..4].mapv(|v| v.powi(2)).mean().unwrap())
        + 2.0 * (xbar[4] - xbar[4].powi(2));

    let hazmat = riskmat.dot(&Array2::from_diag(
        &(&array![1.0, 10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0, 2.0] / wtmat.sum_axis(Axis(0))),
    ));
    let mut dM = -risk.insert_axis(Axis(1)) * hazmat;
    dM[[0, 0]] += 1.0;
    for i in 1..4 {
        dM.slice_mut(s![2..5, i]).mapv_inplace(|v| v + 1.0 / 3.0);
    }
    dM[[7, 4]] += 1.0;
    let mart = status.clone() - expected.clone();

    let resid = &dM * &x.mapv(|xi| xi - xbar.slice(s![..]));

    let var_g =
        hazard.mapv(|h| h.powi(2)) * array![1.0, 3.0 / 10.0, 3.0 / 10.0, 3.0 / 10.0, 1.0 / 2.0];
    let var_d = (&xbar - newx) * &hazard;
    let varhaz = (var_g + var_d.mapv(|v| v.powi(2)) / imat) * (2.0 * beta * newx).exp();

    ByHandResults {
        loglik,
        imat,
        hazard,
        xbar,
        mart,
        expected,
        score: resid.sum_axis(Axis(1)),
        schoen: array![2.0, 1.0, 1.0, 0.0, 1.0]
            - array![xbar[0], xbar[1..4].mean().unwrap(), xbar[4]],
        varhaz: varhaz.select(ndarray::indices(&[0, 3, 4])),
    }
}

fn aeq(a: &Array1<f64>, b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
}

fn main() {
    let temp = byhand(0.0, 0.0);
    assert!(aeq(
        &temp.xbar,
        &[
            13.0 / 19.0,
            11.0 / 16.0,
            26.0 / 38.0,
            19.0 / 28.0,
            2.0 / 3.0
        ],
        1e-9
    ));
    assert!(aeq(
        &temp.hazard,
        &[1.0 / 19.0, 5.0 / 24.0, 5.0 / 19.0, 5.0 / 14.0, 2.0 / 3.0],
        1e-9
    ));

    let truth0 = byhand(0.0, std::f64::consts::PI);
    let sfit_var = truth0.varhaz.clone();
    let sfit_surv = (-truth0.hazard).mapv(|h| h.exp());
    assert!(aeq(&sfit_var, &[0.0, 0.0, 0.0], 1e-6)); 

    println!("All tests passed!");
}
