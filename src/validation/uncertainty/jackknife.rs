fn validate_jackknife_covariates(covariates: &[Vec<f64>], n: usize) -> PyResult<()> {
    if covariates.len() != n {
        return Err(uncertainty_value_error(format!(
            "covariates must have one row per observation; got {} rows for {n} observations",
            covariates.len()
        )));
    }

    let p = covariates.first().map_or(0, |row| row.len());
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != p {
            return Err(uncertainty_value_error(format!(
                "covariates must be rectangular; row {row_idx} has {} values, expected {p}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(uncertainty_value_error(format!(
                    "covariates contains non-finite value {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }

    Ok(())
}

fn validate_jackknife_plus_inputs(
    time: &[f64],
    event: &[i32],
    covariates: &[Vec<f64>],
    config: &JackknifePlusConfig,
) -> PyResult<()> {
    if time.len() < 2 {
        return Err(uncertainty_value_error(
            "Need at least 2 observations for jackknife",
        ));
    }
    crate::internal::validation::validate_length(time.len(), event.len(), "event")?;
    crate::internal::validation::validate_finite(time, "time")?;
    crate::internal::validation::validate_non_negative(time, "time")?;
    crate::internal::validation::validate_binary_i32(event, "event")?;
    validate_jackknife_covariates(covariates, time.len())?;
    validate_probability_open(config.alpha, "alpha")?;

    if config.cv_folds == 0 {
        return Err(uncertainty_value_error("cv_folds must be positive"));
    }

    Ok(())
}

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
    validate_jackknife_plus_inputs(&time, &event, &covariates, &config)?;

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
    sorted_residuals.sort_by(f64::total_cmp);
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
