struct ByHandResults {
    loglik: f64,
    u: f64,
    imat: f64,
    mart: Vec<f64>,
}

fn byhand7(beta: f64) -> ByHandResults {
    let r = beta.exp();

    let loglik = 2.0 * (beta - (3.0 * r + 3.0).ln());

    let u = 2.0 / (r + 1.0);
    let imat = 2.0 * r / (r + 1.0).powi(2);

    let haz = [1.0 / (3.0 * r + 3.0), 2.0 / (r + 3.0), 0.0, 1.0];
    let cum_haz = [
        haz[0],
        haz[0] + haz[1],
        haz[0] + haz[1] + haz[2],
        haz[0] + haz[1] + haz[2] + haz[3],
    ];

    let ties = [0, 0, 1, 1, 2, 3]; 
    let wt = [r, r, r, 1.0, 1.0, 1.0];
    let status = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let mart = status
        .iter()
        .zip(wt.iter())
        .zip(ties.iter())
        .map(|((&s, &w), &t)| s - w * cum_haz[t])
        .collect();

    ByHandResults {
        loglik,
        u,
        imat,
        mart,
    }
}

fn aeq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

fn main() {
    let truth0 = byhand7(0.0);

    assert!((truth0.loglik - (-3.583519)).abs() < 1e-4);

    assert!((1.0 / truth0.imat - 2.0).abs() < 1e-6);

    let expected_mart = vec![
        5.0 / 6.0,
        -1.0 / 6.0,
        1.0 / 3.0,
        1.0 / 3.0,
        -2.0 / 3.0,
        -2.0 / 3.0,
    ];
    assert!(aeq(&truth0.mart, &expected_mart, 1e-6));

    let truth1 = byhand7(1.0);

    let expected_u = 2.0 / (1.0f64.exp() + 1.0);
    assert!((truth1.u - expected_u).abs() < 1e-6);

    println!("All tests passed!");
}
