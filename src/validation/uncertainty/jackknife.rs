#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    config
))]
pub fn jackknife_plus_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: JackknifePlusConfig,
) -> PyResult<JackknifePlusResult> {
    let n = time.len();

    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 observations for jackknife",
        ));
    }

    let full_predictions = simple_cox_predictions(&time, &event, &covariates);

    let mut loo_residuals = vec![0.0; n];
    let p = covariates.first().map_or(0, |row| row.len());

    let mut loo_time = vec![0.0; n - 1];
    let mut loo_event = vec![0; n - 1];
    let mut loo_cov = vec![vec![0.0; p]; n - 1];

    for i in 0..n {
        let mut dst = 0;
        for src in 0..n {
            if src == i {
                continue;
            }
            loo_time[dst] = time[src];
            loo_event[dst] = event[src];
            if p > 0 {
                loo_cov[dst].copy_from_slice(&covariates[src]);
            }
            dst += 1;
        }

        let loo_predictions = simple_cox_predictions(&loo_time, &loo_event, &loo_cov);

        let pred_i = if !loo_predictions.is_empty() {
            kernel_weighted_prediction(&covariates[i], &loo_cov, &loo_predictions)
        } else {
            full_predictions[i]
        };

        loo_residuals[i] = (time[i] - pred_i).abs();
    }

    let quantile_idx = if config.plus_variant {
        ((n as f64 + 1.0) * (1.0 - config.alpha)).ceil() as usize
    } else {
        ((n as f64) * (1.0 - config.alpha)).ceil() as usize
    };

    let mut sorted_residuals = loo_residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = sorted_residuals[quantile_idx.min(n - 1)];

    let lower_bounds: Vec<f64> = full_predictions.iter().map(|p| (p - q).max(0.0)).collect();
    let upper_bounds: Vec<f64> = full_predictions.iter().map(|p| p + q).collect();

    let coverage = compute_coverage(&lower_bounds, &upper_bounds, &time);

    Ok(JackknifePlusResult {
        lower_bounds,
        upper_bounds,
        point_predictions: full_predictions,
        coverage,
        residuals: loo_residuals,
    })
}
