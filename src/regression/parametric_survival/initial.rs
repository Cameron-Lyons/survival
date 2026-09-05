use super::{ComputeSurvregInput, DistributionType, SurvivalFitComputed, compute_survreg};
use crate::regression::coxph_wtest_module::coxph_wtest_core;
use crate::regression::survregc1::survreg_location_derivatives;
use ndarray::{Array1, Array2, ArrayView1};

/// The coordinate change used by R only when starts are omitted and the
/// first design column is an intercept. Binary columns retain their coding.
pub(super) struct SurvregRescaling {
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl SurvregRescaling {
    pub(super) fn apply(
        x: &mut Array2<f64>,
        weights: &Array1<f64>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let p = x.nrows();
        let n = weights.iter().filter(|&&weight| weight > 0.0).count();
        if p <= 1
            || !x
                .row(0)
                .iter()
                .zip(weights)
                .all(|(&value, &weight)| weight == 0.0 || value == 1.0)
        {
            return Ok(None);
        }
        let mut center = vec![0.0; p];
        let mut scale = vec![1.0; p];
        let mut changed = false;
        for column in 1..p {
            let mut values = x.row_mut(column);
            if values
                .iter()
                .zip(weights)
                .all(|(&value, &weight)| weight == 0.0 || value == 0.0 || value == 1.0)
            {
                continue;
            }
            changed = true;
            // Zero-weight observations do not determine the working coordinates.
            center[column] = values
                .iter()
                .zip(weights)
                .filter_map(|(&value, &weight)| (weight > 0.0).then_some(value))
                .sum::<f64>()
                / n as f64;
            scale[column] = (values
                .iter()
                .zip(weights)
                .filter(|(_, weight)| **weight > 0.0)
                .map(|(value, _)| (value - center[column]).powi(2))
                .sum::<f64>()
                / (n - 1) as f64)
                .sqrt();
            if !center[column].is_finite() || !scale[column].is_finite() || scale[column] == 0.0 {
                return Err("initial iteration failed: cannot rescale a constant or non-finite covariate (use starting estimates?)".into());
            }
            values.mapv_inplace(|value| (value - center[column]) / scale[column]);
        }
        Ok(changed.then_some(Self { center, scale }))
    }

    pub(super) fn restore(self, fit: &mut SurvivalFitComputed) {
        let p = self.center.len();
        let parameters = fit.variance_matrix.nrows();
        let mut transform = Array2::<f64>::eye(parameters);
        for column in 0..p {
            transform[[column, column]] = 1.0 / self.scale[column];
            if column > 0 {
                transform[[0, column]] = -self.center[column] / self.scale[column];
            }
        }
        let coefficients = transform.dot(&ArrayView1::from(&fit.coefficients[..parameters]));
        fit.coefficients[..parameters].copy_from_slice(coefficients.as_slice().unwrap());
        fit.variance_matrix = transform.dot(&fit.variance_matrix).dot(&transform.t());
        // R returns its optimizer score in the working design coordinates,
        // even though coefficients and covariance are restored to input units.
    }
}

fn initial_location_coefficients(
    input: &ComputeSurvregInput<'_>,
    covariates: &Array2<f64>,
    midpoint: &[f64],
    log_scales: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let p = covariates.nrows();
    if p == 0 {
        return Ok(Vec::new());
    }
    let distribution = input
        .distribution
        .likelihood_distribution(input.distribution_parameter)?;
    let transform = |time: f64| {
        if input.distribution.uses_log_time() {
            time.ln()
        } else {
            time
        }
    };
    let mut information = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    let status_column = input.y.ncols() - 1;
    for (person, &eta) in midpoint.iter().enumerate() {
        let weight = input.weights[person];
        if weight == 0.0 {
            continue;
        }
        let time = transform(input.y[[person, 0]]);
        let time2 = (status_column == 2).then(|| transform(input.y[[person, 1]]));
        let (dg, ddg) = survreg_location_derivatives(
            time,
            time2,
            input.y[[person, status_column]] as i32,
            eta,
            log_scales[input.strata[person]].exp(),
            distribution,
        )?;
        let curvature = -ddg * weight;
        let response = curvature * (eta - input.offsets[person]) + weight * dg;
        for column in 0..p {
            let x = covariates[[column, person]];
            rhs[column] += x * response;
            for previous in 0..=column {
                information[column][previous] += curvature * x * covariates[[previous, person]];
            }
        }
    }
    // coxph.wtest has a scalar fast path without a Cholesky rank cutoff.
    let coefficients = if p == 1 {
        vec![rhs[0] / information[0][0]]
    } else {
        let (_, _, solved) = coxph_wtest_core(&information, &[rhs], input.tol_chol)
            .map_err(|error| error.to_string())?;
        solved.into_iter().map(|row| row[0]).collect()
    };
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err("initial iteration failed (use starting estimates?)".into());
    }
    Ok(coefficients)
}

pub(super) fn is_mean_only(covariates: &Array2<f64>, weights: &Array1<f64>) -> bool {
    covariates.nrows() == 1
        && covariates
            .iter()
            .zip(weights)
            .all(|(&value, &weight)| weight == 0.0 || value == 1.0)
}

pub(super) fn initialize_survreg(
    input: &ComputeSurvregInput<'_>,
    initial_location: Option<&[f64]>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let transform = |time: f64| {
        if input.distribution.uses_log_time() {
            time.ln()
        } else {
            time
        }
    };
    let status_column = input.y.ncols() - 1;
    let midpoint: Vec<f64> = input
        .y
        .outer_iter()
        .map(|row| {
            let lower = transform(row[0]);
            if row[status_column] == 3.0 {
                // Center intervals after transforming to the fitting time scale.
                0.5 * lower + 0.5 * transform(row[1])
            } else {
                lower
            }
        })
        .collect();
    let log_scale = if let Some(scale) = input.fixed_scale {
        scale.ln()
    } else {
        let total_weight = input.weights.sum();
        let mean = midpoint
            .iter()
            .zip(input.weights)
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            / total_weight;
        let variance = midpoint
            .iter()
            .zip(input.weights)
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(value, weight)| weight * (value - mean).powi(2))
            .sum::<f64>()
            / total_weight;
        let variance = match input.distribution {
            DistributionType::ExtremeValue | DistributionType::Weibull => variance / 1.64,
            DistributionType::Logistic | DistributionType::LogLogistic => variance / 3.2,
            DistributionType::Gaussian | DistributionType::LogNormal => variance,
            DistributionType::StudentT => {
                let df = input
                    .distribution_parameter
                    .ok_or("Student-t degrees of freedom are missing")?;
                variance * (df - 2.0) / df
            }
        };
        0.5 * (4.0 * variance).ln()
    };
    if !log_scale.is_finite() {
        return Err("initial iteration failed: response scale is not finite and positive (use starting estimates?)".into());
    }
    let mut log_scales = vec![log_scale; input.nstrat];
    let mean_only = is_mean_only(input.covariates, input.weights);
    if !mean_only && input.fixed_scale.is_none() {
        let intercept = Array2::ones((1, input.y.nrows()));
        let mut beta = initial_location_coefficients(input, &intercept, &midpoint, &log_scales)?;
        beta.extend_from_slice(&log_scales);
        let fit = compute_survreg(ComputeSurvregInput {
            max_iter: 20,
            nvar: 1,
            y: input.y,
            covariates: &intercept,
            weights: input.weights,
            offsets: input.offsets,
            beta,
            nstrat: input.nstrat,
            strata: input.strata,
            eps: input.eps,
            tol_chol: input.tol_chol,
            distribution: input.distribution,
            distribution_parameter: input.distribution_parameter,
            fixed_scale: None,
        })?;
        if fit.coefficients.iter().any(|value| !value.is_finite())
            || !fit.log_likelihood.is_finite()
            || fit.variance_matrix.iter().any(|value| !value.is_finite())
        {
            return Err("initial iteration failed (use starting estimates?)".into());
        }
        log_scales.copy_from_slice(&fit.coefficients[1..]);
    }
    // A location-only start is already in the user's design coordinates.
    // Only its missing scales come from the preliminary intercept fit.
    let mut beta = match initial_location {
        Some(values) => values.to_vec(),
        None => initial_location_coefficients(input, input.covariates, &midpoint, &log_scales)?,
    };
    if input.fixed_scale.is_none() {
        beta.extend(log_scales);
    }
    Ok(beta)
}
