struct ByHandResults {
    loglik: f64,
    U: f64,
    imat: f64,
    hazard: Vec<f64>,
    xbar: Vec<f64>,
    mart: Vec<f64>,
    expected: Vec<f64>,
    score: Vec<f64>,
    schoen: Vec<f64>,
    varhaz: Vec<f64>,
}

fn byhand(beta: f64, newx: f64) -> ByHandResults {
    let r = beta.exp();

    let loglik = 11.0 * beta
        - ((r.powi(2) + 11.0 * r + 7.0).ln()
            + 10.0 * (11.0 * r + 5.0).ln()
            + 2.0 * (2.0 * r + 1.0).ln());

    let hazard = vec![
        1.0 / (r.powi(2) + 11.0 * r + 7.0),
        10.0 / (11.0 * r + 5.0),
        2.0 / (2.0 * r + 1.0),
    ];

    let xbar = vec![
        (2.0 * r.powi(2) + 11.0 * r) * hazard[0],
        (11.0 * r * hazard[1]) / 10.0,
        r * hazard[2],
    ];

    let U = 11.0 - (xbar[0] + 10.0 * xbar[1] + 2.0 * xbar[2]);

    let imat = (4.0 * r.powi(2) + 11.0 * r) * hazard[0] - xbar[0].powi(2)
        + 10.0 * (xbar[1] - xbar[1].powi(2))
        + 2.0 * (xbar[2] - xbar[2].powi(2));

    let cumulative_hazard = vec![
        hazard[0],
        hazard[0] + 3.0 * hazard[1],
        hazard[0] + 3.0 * hazard[1] + hazard[2],
    ];
    let risk = vec![r.powi(2), 1.0, r, r, 1.0, r, 1.0, r, 1.0];
    let expected = vec![
        risk[0] * cumulative_hazard[0],
        risk[1] * cumulative_hazard[0],
        risk[2] * cumulative_hazard[1],
        risk[3] * cumulative_hazard[1],
        risk[4] * cumulative_hazard[1],
        risk[5] * cumulative_hazard[1],
        risk[6] * cumulative_hazard[1],
        risk[7] * cumulative_hazard[2],
        risk[8] * cumulative_hazard[2],
    ];

    let status = vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let mart = status
        .iter()
        .zip(expected.iter())
        .map(|(s, e)| s - e)
        .collect::<Vec<f64>>();

    let score = vec![
        2.0 - xbar[0],
        0.0 - xbar[0],
        1.0 - xbar[1],
        1.0 - xbar[1],
        0.0 - xbar[1],
        1.0 - xbar[1],
        0.0 - xbar[1],
        1.0 - xbar[2],
        0.0 - xbar[2],
    ];

    let schoen = vec![
        2.0 - xbar[0],
        1.0 - xbar[1],
        1.0 - xbar[1],
        0.0 - xbar[1],
        1.0 - xbar[2],
    ];

    let var_g = vec![
        hazard[0].powi(2) / 1.0,
        hazard[1].powi(2) / 10.0,
        hazard[2].powi(2) / 2.0,
    ];
    let var_d = vec![
        (xbar[0] - newx) * hazard[0],
        (xbar[1] - newx) * hazard[1],
        (xbar[2] - newx) * hazard[2],
    ];
    let varhaz = vec![
        (var_g[0] + var_d[0].powi(2) / imat) * (2.0 * beta * newx).exp(),
        (var_g[0] + var_g[1] + (var_d[0] + var_d[1]).powi(2) / imat) * (2.0 * beta * newx).exp(),
        (var_g[0] + var_g[1] + var_g[2] + (var_d[0] + var_d[1] + var_d[2]).powi(2) / imat)
            * (2.0 * beta * newx).exp(),
    ];

    ByHandResults {
        loglik,
        U,
        imat,
        hazard,
        xbar,
        mart,
        expected,
        score,
        schoen,
        varhaz,
    }
}

fn aeq(x: &[f64], y: &[f64], tolerance: f64) -> bool {
    x.iter()
        .zip(y.iter())
        .all(|(a, b)| (a - b).abs() < tolerance)
}

fn main() {
    let truth0 = byhand(0.0, std::f64::consts::PI);
    let expected_expected = vec![
        1.0 / 19.0,
        1.0 / 19.0,
        103.0 / 152.0,
        103.0 / 152.0,
        103.0 / 152.0,
        103.0 / 152.0,
        103.0 / 152.0,
        613.0 / 456.0,
        613.0 / 456.0,
    ];
    assert!(aeq(&truth0.expected, &expected_expected, 1e-9));

    println!("All tests passed!");
}
