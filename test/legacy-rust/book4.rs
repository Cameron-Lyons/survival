use ndarray::prelude::*;
use std::f64::EPSILON;

#[derive(Debug)]
struct Observation {
    start: f64,
    stop: f64,
    event: u8,
    x: f64,
}

#[derive(Debug)]
struct ByhandResult {
    loglik: f64,
    u: f64,
    imat: f64,
    mart: Vec<f64>,
    score: Vec<f64>,
    scho: Vec<f64>,
}

fn main() {
    let test2 = vec![
        Observation {
            start: 1.0,
            stop: 2.0,
            event: 1,
            x: 1.0,
        },
        Observation {
            start: 2.0,
            stop: 3.0,
            event: 1,
            x: 0.0,
        },
        Observation {
            start: 5.0,
            stop: 6.0,
            event: 1,
            x: 0.0,
        },
        Observation {
            start: 2.0,
            stop: 7.0,
            event: 1,
            x: 1.0,
        },
        Observation {
            start: 1.0,
            stop: 8.0,
            event: 1,
            x: 0.0,
        },
        Observation {
            start: 7.0,
            stop: 9.0,
            event: 1,
            x: 1.0,
        },
        Observation {
            start: 3.0,
            stop: 9.0,
            event: 1,
            x: 1.0,
        },
        Observation {
            start: 4.0,
            stop: 9.0,
            event: 0,
            x: 1.0,
        },
        Observation {
            start: 8.0,
            stop: 14.0,
            event: 0,
            x: 0.0,
        },
        Observation {
            start: 8.0,
            stop: 17.0,
            event: 0,
            x: 0.0,
        },
    ];

    let truth0 = byhand(0.0, &test2);
    println!("Truth0 loglik: {}", truth0.loglik);
    println!("Truth0 score: {}", truth0.u);
    println!("Truth0 imat: {}", truth0.imat);

    let mut beta = 0.0;
    let mut iter = 0;
    let max_iter = 100;
    let tolerance = 1e-8;
    loop {
        let result = byhand(beta, &test2);
        let delta = result.u / result.imat;
        beta += delta;
        iter += 1;
        if delta.abs() < tolerance || iter >= max_iter {
            break;
        }
    }
    println!("Estimated beta: {}", beta);
    let truth = byhand(beta, &test2);
    println!("Converged loglik: {}", truth.loglik);
}

fn byhand(beta: f64, data: &[Observation]) -> ByhandResult {
    let r = beta.exp();
    let x: Vec<f64> = data.iter().map(|obs| obs.x).collect();

    let loglik = 4.0 * beta
        - ((r + 1.0).ln()
            + (r + 2.0).ln()
            + 2.0 * (3.0 * r + 2.0).ln()
            + 2.0 * (3.0 * r + 1.0).ln()
            + (2.0 * r + 2.0).ln());

    let u = 1.0 / (r + 1.0)
        + 1.0 / (3.0 * r + 1.0)
        + 2.0 * (1.0 / (3.0 * r + 2.0) + 1.0 / (2.0 * r + 2.0))
        - (r / (r + 2.0) + 3.0 * r / (3.0 * r + 2.0) + 3.0 * r / (3.0 * r + 1.0));

    let imat = r
        * (1.0 / (r + 1.0).powi(2)
            + 2.0 / (r + 2.0).powi(2)
            + 6.0 / (3.0 * r + 2.0).powi(2)
            + 6.0 / (3.0 * r + 1.0).powi(2)
            + 6.0 / (3.0 * r + 2.0).powi(2)
            + 4.0 / (2.0 * r + 2.0).powi(2));

    let hazard = vec![
        1.0 / (r + 1.0),
        1.0 / (r + 2.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (2.0 * r + 2.0),
    ];

    let wtmat = create_wtmat(r);

    let wtmat_sum = wtmat.sum_axis(Axis(0));
    let wtmat_x = wtmat.t().dot(&Array1::from(x.clone()));
    let xbar: Vec<f64> = wtmat_x
        .iter()
        .zip(wtmat_sum.iter())
        .map(|(x_sum, wt_sum)| x_sum / wt_sum)
        .collect();

    let mut dM = -&wtmat * &Array2::from_shape_fn(wtmat.dim(), |(i, j)| hazard[j]);
    for i in 0..5 {
        dM[(i, i)] += 1.0;
    }
    for i in 5..7 {
        dM[(i, i)] += 0.5;
    }
    let mart = dM.sum_axis(Axis(1)).to_vec();

    let x_diff = Array1::from(x) - Array1::from(xbar.clone());
    let resid = &dM * &x_diff.broadcast((10, 7)).unwrap();
    let score = resid.sum_axis(Axis(1)).to_vec();

    let mut scho = resid.sum_axis(Axis(0)).to_vec();
    let avg = (scho[5] + scho[6]) / 2.0;
    scho[5] = avg;
    scho[6] = avg;

    ByhandResult {
        loglik,
        u,
        imat,
        mart,
        score,
        scho,
    }
}

fn create_wtmat(r: f64) -> Array2<f64> {
    let mut wtmat = Array2::zeros((10, 7));
    wtmat[[0, 0]] = 1.0;
    wtmat[[4, 0]] = 1.0;
    wtmat[[1, 1]] = 1.0;
    wtmat[[3, 1]] = 1.0;
    wtmat[[4, 1]] = 1.0;
    wtmat[[2, 2]] = 1.0;
    wtmat[[3, 2]] = 1.0;
    wtmat[[4, 2]] = 1.0;
    wtmat[[6, 2]] = 1.0;
    wtmat[[7, 2]] = 1.0;
    wtmat[[3, 3]] = 1.0;
    wtmat[[4, 3]] = 1.0;
    wtmat[[6, 3]] = 1.0;
    wtmat[[7, 3]] = 1.0;
    wtmat[[4, 4]] = 1.0;
    wtmat[[5, 4]] = 1.0;
    wtmat[[6, 4]] = 1.0;
    wtmat[[7, 4]] = 1.0;
    wtmat[[5, 5]] = 1.0;
    wtmat[[6, 5]] = 1.0;
    wtmat[[7, 5]] = 1.0;
    wtmat[[8, 5]] = 1.0;
    wtmat[[9, 5]] = 1.0;
    wtmat[[5, 6]] = 0.5;
    wtmat[[6, 6]] = 0.5;
    wtmat[[7, 6]] = 1.0;
    wtmat[[8, 6]] = 1.0;
    wtmat[[9, 6]] = 1.0;

    let row_weights = array![r, 1.0, 1.0, r, 1.0, r, r, r, 1.0, 1.0];
    for i in 0..10 {
        let weight = row_weights[i];
        for j in 0..7 {
            wtmat[[i, j]] *= weight;
        }
    }
    wtmat
}

fn aeq(x: &[f64], y: &[f64]) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-9)
}
