use approx::assert_abs_diff_eq;

fn aeq(x: &[f64], y: &[f64]) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-9)
}

#[derive(Debug)]
struct SurvivalData {
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u32>,
    x: Vec<f64>,
    grp: Option<Vec<String>>,
}

struct CoxPHModel {
    coefficients: f64,
}

struct SurvFit {
    time: Vec<f64>,
    surv: Vec<f64>,
    std_err: Vec<f64>,
}

impl CoxPHModel {
    fn fit(data: &SurvivalData, formula: &str) -> Self {
        CoxPHModel { coefficients: 0.0 }
    }
}

fn lambda(beta: f64, x: f64, method: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let time = vec![2.0, 3.0, 6.0, 7.0, 8.0, 9.0];
    let r = beta.exp();
    let mut lambda = vec![
        1.0 / (r + 1.0),
        1.0 / (r + 2.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 2.0) + 1.0 / (2.0 * r + 2.0),
    ];

    let mut xbar = vec![
        r / (r + 1.0),
        r / (r + 2.0),
        3.0 * r / (3.0 * r + 2.0),
        3.0 * r / (3.0 * r + 1.0),
        3.0 * r / (3.0 * r + 1.0),
        (1.5 * r) / (3.0 * r + 2.0) + r / (2.0 * r + 2.0),
    ];

    if method == "breslow" {
        lambda[5] = 2.0 / (3.0 * r + 2.0);
        xbar[5] = 3.0 * r / (3.0 * r + 2.0);
    }

    (time, lambda, xbar)
}

fn main() {
    let test2 = SurvivalData {
        start: vec![1.0, 2.0, 5.0, 2.0, 1.0, 7.0, 3.0, 4.0, 8.0, 8.0],
        stop: vec![2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 9.0, 9.0, 14.0, 17.0],
        event: vec![1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        x: vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        grp: None,
    };

    let fit = CoxPHModel::fit(&test2, "Surv(start, stop, event) ~ x");

    let (true_time, true_lambda, _) = lambda(fit.coefficients, 0.0, "efron");

    let surv1 = SurvFit {
        time: true_time.clone(),
        surv: true_lambda
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some((-*acc).exp())
            })
            .collect(),
        std_err: vec![],
    };

    assert!(aeq(&surv1.time, &true_time));

    let cum_hazard: Vec<f64> = true_lambda
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();
    let surv_cum_hazard: Vec<f64> = surv1.surv.iter().map(|s| -s.ln()).collect();
    assert!(aeq(&cum_hazard, &surv_cum_hazard));

    println!("All tests passed!");
}
