use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass(from_py_object)]
pub struct AaregFitResult {
    #[pyo3(get)]
    pub n: Vec<usize>,
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub coefficient: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub test_statistic: Vec<f64>,
    #[pyo3(get)]
    pub test_variance: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub test: String,
    #[pyo3(get)]
    pub time_weights: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dfbeta: Option<Vec<Vec<Vec<f64>>>>,
    #[pyo3(get)]
    pub robust_test_variance: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl AaregFitResult {
    fn __repr__(&self) -> String {
        format!(
            "AaregFitResult(n_obs={}, n_times={}, n_coefficients={})",
            self.n.first().copied().unwrap_or(0),
            self.n.get(1).copied().unwrap_or(0),
            self.coefficient.first().map_or(0, Vec::len),
        )
    }
}

#[derive(Clone, Debug)]
struct RiskMoment {
    time: f64,
    events: Vec<usize>,
    risk: f64,
    mean: Vec<f64>,
    covariance: Vec<f64>,
    inverse: Vec<f64>,
    time_weight: Vec<f64>,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_inputs(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    weights: Option<&[f64]>,
    cluster: Option<&[i32]>,
    test_cluster: Option<&[i32]>,
) -> PyResult<usize> {
    let n = stop.len();
    if n == 0 {
        return Err(value_error("stop must not be empty"));
    }
    if status.len() != n || covariates.len() != n {
        return Err(value_error(
            "stop, status, and covariates must have the same length",
        ));
    }
    let nvar = covariates.first().map_or(0, Vec::len);
    for (idx, &value) in stop.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "stop contains non-finite value at index {idx}"
            )));
        }
    }
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(value_error(format!(
                "status must contain only 0 and 1; got {value} at index {idx}"
            )));
        }
    }
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != nvar {
            return Err(value_error(format!(
                "covariate row {row_idx} has {} columns; expected {nvar}",
                row.len()
            )));
        }
        for (column_idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "covariates contains non-finite value at row {row_idx}, column {column_idx}"
                )));
            }
        }
    }
    if let Some(values) = start {
        if values.len() != n {
            return Err(value_error("start must have the same length as stop"));
        }
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "start contains non-finite value at index {idx}"
                )));
            }
            if value >= stop[idx] {
                return Err(value_error(format!(
                    "start[{idx}] must be less than stop[{idx}]"
                )));
            }
        }
    }
    if let Some(values) = weights {
        if values.len() != n {
            return Err(value_error("weights must have the same length as stop"));
        }
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(value_error(format!(
                    "weights must contain positive finite values; got {value} at index {idx}"
                )));
            }
        }
    }
    if let Some(values) = cluster {
        if values.len() != n {
            return Err(value_error("cluster must have the same length as stop"));
        }
        if let Some((idx, value)) = values.iter().enumerate().find(|(_, value)| **value < 0) {
            return Err(value_error(format!(
                "cluster codes must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    if let Some(values) = test_cluster {
        if values.len() != n {
            return Err(value_error(
                "test_cluster must have the same length as stop",
            ));
        }
        if let Some((idx, value)) = values.iter().enumerate().find(|(_, value)| **value < 0) {
            return Err(value_error(format!(
                "test_cluster codes must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    if !status.contains(&1) {
        return Err(value_error("aareg requires at least one event"));
    }
    Ok(nvar)
}

fn validate_fit_options(qrtol: f64, taper: &[f64]) -> PyResult<()> {
    if !qrtol.is_finite() || qrtol <= 0.0 {
        return Err(value_error("qrtol must be finite and positive"));
    }
    if taper.is_empty()
        || taper
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(value_error("taper must contain positive finite values"));
    }
    Ok(())
}

fn event_groups(stop: &[f64], status: &[i32]) -> Vec<(f64, Vec<usize>)> {
    let mut events: Vec<usize> = (0..stop.len()).filter(|&idx| status[idx] == 1).collect();
    events.sort_by(|&left, &right| stop[left].total_cmp(&stop[right]).then(left.cmp(&right)));
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for idx in events {
        if let Some((time, members)) = groups.last_mut()
            && *time == stop[idx]
        {
            members.push(idx);
        } else {
            groups.push((stop[idx], vec![idx]));
        }
    }
    groups
}

fn update_risk_sums(
    idx: usize,
    sign: f64,
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    s0: &mut f64,
    s1: &mut [f64],
    s2: &mut [f64],
) {
    let weight = sign * weights.map_or(1.0, |values| values[idx]);
    *s0 += weight;
    let row = &covariates[idx];
    let nvar = row.len();
    for column in 0..nvar {
        s1[column] += weight * row[column];
        for inner in 0..nvar {
            s2[column * nvar + inner] += weight * row[column] * row[inner];
        }
    }
}

fn risk_moments(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    weights: Option<&[f64]>,
    nvar: usize,
) -> Vec<RiskMoment> {
    let groups = event_groups(stop, status);
    let n = stop.len();
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&left, &right| stop[left].total_cmp(&stop[right]).then(left.cmp(&right)));
    let mut start_order: Vec<usize> = (0..n).collect();
    if let Some(values) = start {
        start_order.sort_by(|&left, &right| {
            values[left]
                .total_cmp(&values[right])
                .then(left.cmp(&right))
        });
    }

    let mut s0 = 0.0;
    let mut s1 = vec![0.0; nvar];
    let mut s2 = vec![0.0; nvar * nvar];
    let mut start_pos = 0;
    if start.is_none() {
        for idx in 0..n {
            update_risk_sums(idx, 1.0, covariates, weights, &mut s0, &mut s1, &mut s2);
        }
        start_pos = n;
    }
    let mut stop_pos = 0;
    let mut moments = Vec::with_capacity(groups.len());

    for (time, events) in groups {
        if let Some(values) = start {
            while start_pos < n && values[start_order[start_pos]] < time {
                update_risk_sums(
                    start_order[start_pos],
                    1.0,
                    covariates,
                    weights,
                    &mut s0,
                    &mut s1,
                    &mut s2,
                );
                start_pos += 1;
            }
        }
        while stop_pos < n && stop[stop_order[stop_pos]] < time {
            update_risk_sums(
                stop_order[stop_pos],
                -1.0,
                covariates,
                weights,
                &mut s0,
                &mut s1,
                &mut s2,
            );
            stop_pos += 1;
        }

        let mut mean = vec![0.0; nvar];
        let mut covariance = vec![0.0; nvar * nvar];
        if s0 > 0.0 {
            for column in 0..nvar {
                mean[column] = s1[column] / s0;
            }
            for row in 0..nvar {
                for column in 0..nvar {
                    let value = s2[row * nvar + column] / s0 - mean[row] * mean[column];
                    covariance[row * nvar + column] = if value.abs() < 1e-15 { 0.0 } else { value };
                }
            }
        }
        moments.push(RiskMoment {
            time,
            events,
            risk: s0,
            mean,
            covariance,
            inverse: Vec::new(),
            time_weight: Vec::new(),
        });
    }
    moments
}

fn taper_covariances(moments: &mut [RiskMoment], taper: &[f64], nvar: usize) {
    if taper.len() == 1 || moments.is_empty() || nvar == 0 {
        return;
    }
    let original: Vec<Vec<f64>> = moments
        .iter()
        .map(|moment| moment.covariance.clone())
        .collect();
    for (time_idx, moment) in moments.iter_mut().enumerate() {
        let window = taper.len().min(time_idx + 1);
        let first_time = time_idx + 1 - window;
        let first_weight = taper.len() - window;
        let denominator: f64 = taper[first_weight..].iter().sum();
        moment.covariance.fill(0.0);
        for offset in 0..window {
            let weight = taper[first_weight + offset] / denominator;
            for (value, original_value) in moment
                .covariance
                .iter_mut()
                .zip(&original[first_time + offset])
            {
                *value += original_value * weight;
            }
        }
    }
}

fn invert_matrix(values: &[f64], n: usize, tolerance: f64) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let mut augmented = vec![0.0; n * 2 * n];
    let width = 2 * n;
    for row in 0..n {
        for column in 0..n {
            augmented[row * width + column] = values[row * n + column];
        }
        augmented[row * width + n + row] = 1.0;
    }
    for column in 0..n {
        let mut pivot_row = column;
        for row in (column + 1)..n {
            if augmented[row * width + column].abs() > augmented[pivot_row * width + column].abs() {
                pivot_row = row;
            }
        }
        let pivot = augmented[pivot_row * width + column];
        if !pivot.is_finite() || pivot.abs() <= tolerance * scale {
            return None;
        }
        if pivot_row != column {
            for idx in 0..width {
                augmented.swap(column * width + idx, pivot_row * width + idx);
            }
        }
        let pivot = augmented[column * width + column];
        for idx in 0..width {
            augmented[column * width + idx] /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = augmented[row * width + column];
            for idx in 0..width {
                augmented[row * width + idx] -= factor * augmented[column * width + idx];
            }
        }
    }
    let mut inverse = vec![0.0; n * n];
    for row in 0..n {
        inverse[row * n..(row + 1) * n]
            .copy_from_slice(&augmented[row * width + n..row * width + 2 * n]);
    }
    Some(inverse)
}

fn matrix_vector_product(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[row * n + column] * vector[column])
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

fn prepare_retained_moments(
    moments: &mut [RiskMoment],
    nvar: usize,
    nmin: Option<usize>,
    qrtol: f64,
) -> PyResult<usize> {
    let threshold = nmin.unwrap_or(3 * nvar) as f64;
    let mut retained = moments
        .iter()
        .filter(|moment| {
            moment.risk >= threshold
                && (nvar != 1 || moment.covariance.first().is_some_and(|value| *value > 0.0))
        })
        .count();
    if nvar > 1 {
        while retained > 0
            && invert_matrix(&moments[retained - 1].covariance, nvar, qrtol).is_none()
        {
            retained -= 1;
        }
    }
    if retained <= 1 {
        return Err(value_error(
            "the nmin threshold is too high; no Aalen model can be fit",
        ));
    }
    for (time_idx, moment) in moments.iter_mut().take(retained).enumerate() {
        moment.inverse = invert_matrix(&moment.covariance, nvar, qrtol).ok_or_else(|| {
            value_error(format!(
                "risk covariance is rank deficient at event time {} (index {time_idx})",
                moment.time
            ))
        })?;
        moment.time_weight = if nvar == 0 {
            vec![moment.risk]
        } else {
            let inverse_mean = matrix_vector_product(&moment.inverse, &moment.mean, nvar);
            let mut weights = Vec::with_capacity(nvar + 1);
            weights.push(moment.risk / (1.0 + dot(&moment.mean, &inverse_mean)));
            for column in 0..nvar {
                weights.push(moment.risk / moment.inverse[column * nvar + column]);
            }
            weights
        };
    }
    Ok(retained)
}

fn coefficient_row(
    moment: &RiskMoment,
    event_idx: usize,
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
) -> Vec<f64> {
    let nvar = moment.mean.len();
    let event_weight = weights.map_or(1.0, |values| values[event_idx]);
    if nvar == 0 {
        return vec![event_weight / moment.risk];
    }
    let rhs: Vec<f64> = (0..nvar)
        .map(|column| {
            event_weight * (covariates[event_idx][column] - moment.mean[column]) / moment.risk
        })
        .collect();
    let slopes = matrix_vector_product(&moment.inverse, &rhs, nvar);
    let mut row = Vec::with_capacity(nvar + 1);
    row.push(event_weight / moment.risk - dot(&moment.mean, &slopes));
    row.extend(slopes);
    row
}

fn add_outer_product(matrix: &mut [f64], values: &[f64]) {
    let width = values.len();
    for row in 0..width {
        for column in 0..width {
            matrix[row * width + column] += values[row] * values[column];
        }
    }
}

fn nested_square(values: Vec<f64>, width: usize) -> Vec<Vec<f64>> {
    if width == 0 {
        return Vec::new();
    }
    values.chunks(width).map(<[f64]>::to_vec).collect()
}

fn model_test(
    moments: &[RiskMoment],
    coefficients: &[Vec<Vec<f64>>],
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    test: &str,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let nvar = covariates.first().map_or(0, Vec::len);
    if test == "variance" && nvar > 1 {
        let mut statistic = vec![0.0; nvar];
        let mut variance = vec![0.0; nvar * nvar];
        for (moment, group_coefficients) in moments.iter().zip(coefficients) {
            for (&event_idx, _) in moment.events.iter().zip(group_coefficients) {
                let event_weight = weights.map_or(1.0, |values| values[event_idx]);
                let contribution: Vec<f64> = (0..nvar)
                    .map(|column| {
                        event_weight * (covariates[event_idx][column] - moment.mean[column])
                    })
                    .collect();
                for column in 0..nvar {
                    statistic[column] += contribution[column];
                }
                add_outer_product(&mut variance, &contribution);
            }
        }
        return (statistic, nested_square(variance, nvar));
    }

    let width = nvar + 1;
    let mut statistic = vec![0.0; width];
    let mut variance = vec![0.0; width * width];
    for (moment, group_coefficients) in moments.iter().zip(coefficients) {
        for coefficient in group_coefficients {
            let contribution: Vec<f64> = coefficient
                .iter()
                .enumerate()
                .map(|(column, value)| {
                    value
                        * if test == "nrisk" {
                            moment.risk
                        } else {
                            moment.time_weight[column]
                        }
                })
                .collect();
            for column in 0..width {
                statistic[column] += contribution[column];
            }
            add_outer_product(&mut variance, &contribution);
        }
    }
    (statistic, nested_square(variance, width))
}

fn row_at_risk(start: Option<&[f64]>, stop: &[f64], idx: usize, time: f64) -> bool {
    stop[idx] >= time && start.is_none_or(|values| values[idx] < time)
}

#[allow(clippy::too_many_arguments)]
fn influence_values_rowwise(
    moments: &[RiskMoment],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    stop: &[f64],
    weights: Option<&[f64]>,
    cluster: Option<&[i32]>,
    test_cluster: Option<&[i32]>,
    test: &str,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let n = stop.len();
    let nvar = covariates.first().map_or(0, Vec::len);
    let width = nvar + 1;
    let cluster_count = cluster
        .and_then(|values| values.iter().max().copied())
        .map_or(n, |value| value as usize + 1);
    let test_cluster = test_cluster.or(cluster);
    let test_cluster_count = test_cluster
        .and_then(|values| values.iter().max().copied())
        .map_or(n, |value| value as usize + 1);
    let mut dfbeta = vec![vec![vec![0.0; moments.len()]; width]; cluster_count];
    let mut test_influence = vec![vec![0.0; width]; test_cluster_count];
    let mut event_rows = vec![false; n];
    let mut event_rhs = vec![0.0; nvar];
    let mut summed_slopes = vec![0.0; nvar];
    let mut row_slopes = vec![0.0; nvar];

    for (time_idx, moment) in moments.iter().enumerate() {
        for &event_idx in &moment.events {
            event_rows[event_idx] = true;
        }
        let total_event_weight: f64 = moment
            .events
            .iter()
            .map(|&idx| weights.map_or(1.0, |values| values[idx]))
            .sum();
        summed_slopes.fill(0.0);
        for &event_idx in &moment.events {
            let event_weight = weights.map_or(1.0, |values| values[event_idx]);
            let event_covariates = &covariates[event_idx];
            for column in 0..nvar {
                event_rhs[column] =
                    event_weight * (event_covariates[column] - moment.mean[column]) / moment.risk;
            }
            for (row, slope) in summed_slopes.iter_mut().enumerate() {
                *slope += (0..nvar)
                    .map(|column| moment.inverse[row * nvar + column] * event_rhs[column])
                    .sum::<f64>();
            }
        }

        for row_idx in 0..n {
            let row_covariates = &covariates[row_idx];
            let at_risk = row_at_risk(start, stop, row_idx, moment.time);
            let event_indicator = usize::from(event_rows[row_idx]) as f64;
            let row_weight = if at_risk {
                weights.map_or(1.0, |values| values[row_idx])
            } else {
                0.0
            };
            let predicted = if at_risk {
                total_event_weight / moment.risk
                    + (0..nvar)
                        .map(|column| {
                            (row_covariates[column] - moment.mean[column]) * summed_slopes[column]
                        })
                        .sum::<f64>()
            } else {
                0.0
            };
            let residual = event_indicator - predicted;
            let normalized_score = residual * row_weight / moment.risk;
            let intercept = if nvar == 0 {
                normalized_score
            } else {
                for (row, slope) in row_slopes.iter_mut().enumerate() {
                    *slope = (0..nvar)
                        .map(|column| {
                            moment.inverse[row * nvar + column]
                                * normalized_score
                                * (row_covariates[column] - moment.mean[column])
                        })
                        .sum();
                }
                normalized_score - dot(&row_slopes, &moment.mean)
            };
            let cluster_idx = cluster.map_or(row_idx, |values| values[row_idx] as usize);
            let test_cluster_idx = test_cluster.map_or(row_idx, |values| values[row_idx] as usize);
            for column in 0..width {
                let influence = if column == 0 {
                    intercept
                } else {
                    row_slopes[column - 1]
                };
                dfbeta[cluster_idx][column][time_idx] += influence;
                let test_value = if test == "nrisk" {
                    influence * moment.risk
                } else if test == "variance" && nvar > 1 && column > 0 {
                    residual * row_weight * (row_covariates[column - 1] - moment.mean[column - 1])
                } else {
                    influence * moment.time_weight[column]
                };
                test_influence[test_cluster_idx][column] += test_value;
            }
        }
        for &event_idx in &moment.events {
            event_rows[event_idx] = false;
        }
    }

    let mut robust = vec![0.0; width * width];
    for row in &test_influence {
        add_outer_product(&mut robust, row);
    }
    (dfbeta, nested_square(robust, width))
}

#[derive(Debug)]
struct ClusterRiskSums {
    nvar: usize,
    weight: Vec<f64>,
    covariate: Vec<f64>,
    outer: Vec<f64>,
}

impl ClusterRiskSums {
    fn new(cluster_count: usize, nvar: usize) -> Self {
        Self {
            nvar,
            weight: vec![0.0; cluster_count],
            covariate: vec![0.0; cluster_count * nvar],
            outer: vec![0.0; cluster_count * nvar * nvar],
        }
    }

    fn update(
        &mut self,
        row_idx: usize,
        sign: f64,
        clusters: &[i32],
        covariates: &[Vec<f64>],
        weights: Option<&[f64]>,
    ) {
        let cluster_idx = clusters[row_idx] as usize;
        let weight = sign * weights.map_or(1.0, |values| values[row_idx]);
        self.weight[cluster_idx] += weight;
        let covariate_base = cluster_idx * self.nvar;
        let outer_base = cluster_idx * self.nvar * self.nvar;
        let row_covariates = &covariates[row_idx];
        for (row, &row_value) in row_covariates.iter().enumerate() {
            self.covariate[covariate_base + row] += weight * row_value;
            for (column, &column_value) in row_covariates.iter().enumerate() {
                self.outer[outer_base + row * self.nvar + column] +=
                    weight * row_value * column_value;
            }
        }
    }
}

#[derive(Debug)]
struct ClusterInfluence {
    dfbeta: Vec<f64>,
    centered_score: Vec<f64>,
}

fn cluster_count(clusters: &[i32]) -> usize {
    clusters.iter().max().map_or(0, |value| *value as usize + 1)
}

fn cluster_influence_at_time(
    moment: &RiskMoment,
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    clusters: &[i32],
    risk_sums: &ClusterRiskSums,
    total_event_weight: f64,
) -> Vec<ClusterInfluence> {
    let nvar = moment.mean.len();
    let width = nvar + 1;
    let cluster_count = risk_sums.weight.len();
    let mut event_weight = vec![0.0; cluster_count];
    let mut event_covariate = vec![0.0; cluster_count * nvar];
    for &event_idx in &moment.events {
        let cluster_idx = clusters[event_idx] as usize;
        let weight = weights.map_or(1.0, |values| values[event_idx]);
        event_weight[cluster_idx] += weight;
        let base = cluster_idx * nvar;
        for column in 0..nvar {
            event_covariate[base + column] += weight * covariates[event_idx][column];
        }
    }

    let mut summed_slopes = vec![0.0; nvar];
    for &event_idx in &moment.events {
        let coefficient = coefficient_row(moment, event_idx, covariates, weights);
        for column in 0..nvar {
            summed_slopes[column] += coefficient[column + 1];
        }
    }
    let event_rate = total_event_weight / moment.risk;

    (0..cluster_count)
        .map(|cluster_idx| {
            let cluster_weight = risk_sums.weight[cluster_idx];
            let covariate_base = cluster_idx * nvar;
            let outer_base = cluster_idx * nvar * nvar;
            let mut centered_sum = vec![0.0; nvar];
            let mut centered_event_sum = vec![0.0; nvar];
            for column in 0..nvar {
                centered_sum[column] = risk_sums.covariate[covariate_base + column]
                    - moment.mean[column] * cluster_weight;
                centered_event_sum[column] = event_covariate[covariate_base + column]
                    - moment.mean[column] * event_weight[cluster_idx];
            }

            let mut centered_score = vec![0.0; nvar];
            for row in 0..nvar {
                let mut projected_second_moment = 0.0;
                for (column, &slope) in summed_slopes.iter().enumerate() {
                    let centered_outer = risk_sums.outer[outer_base + row * nvar + column]
                        - moment.mean[row] * risk_sums.covariate[covariate_base + column]
                        - risk_sums.covariate[covariate_base + row] * moment.mean[column]
                        + cluster_weight * moment.mean[row] * moment.mean[column];
                    projected_second_moment += centered_outer * slope;
                }
                centered_score[row] = centered_event_sum[row]
                    - event_rate * centered_sum[row]
                    - projected_second_moment;
            }

            let projected_mean = dot(&centered_sum, &summed_slopes);
            let intercept_score =
                event_weight[cluster_idx] - event_rate * cluster_weight - projected_mean;
            let normalized_score: Vec<f64> = centered_score
                .iter()
                .map(|value| value / moment.risk)
                .collect();
            let slopes = matrix_vector_product(&moment.inverse, &normalized_score, nvar);
            let mut dfbeta = vec![0.0; width];
            dfbeta[0] = intercept_score / moment.risk - dot(&slopes, &moment.mean);
            dfbeta[1..].copy_from_slice(&slopes);
            ClusterInfluence {
                dfbeta,
                centered_score,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn update_cluster_risk_sums(
    row_idx: usize,
    sign: f64,
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    clusters: &[i32],
    risk_sums: &mut ClusterRiskSums,
    test_clusters: &[i32],
    test_risk_sums: Option<&mut ClusterRiskSums>,
) {
    risk_sums.update(row_idx, sign, clusters, covariates, weights);
    if let Some(values) = test_risk_sums {
        values.update(row_idx, sign, test_clusters, covariates, weights);
    }
}

#[allow(clippy::too_many_arguments)]
fn influence_values_clustered(
    moments: &[RiskMoment],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    stop: &[f64],
    weights: Option<&[f64]>,
    clusters: &[i32],
    test_cluster: Option<&[i32]>,
    test: &str,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let n = stop.len();
    let nvar = covariates.first().map_or(0, Vec::len);
    let width = nvar + 1;
    let test_clusters = test_cluster.unwrap_or(clusters);
    let same_clusters = clusters == test_clusters;
    let mut risk_sums = ClusterRiskSums::new(cluster_count(clusters), nvar);
    let mut test_risk_sums =
        (!same_clusters).then(|| ClusterRiskSums::new(cluster_count(test_clusters), nvar));
    let mut dfbeta = vec![vec![vec![0.0; moments.len()]; width]; risk_sums.weight.len()];
    let mut test_influence =
        vec![
            vec![0.0; width];
            test_risk_sums
                .as_ref()
                .map_or(risk_sums.weight.len(), |values| { values.weight.len() })
        ];

    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&left, &right| stop[left].total_cmp(&stop[right]).then(left.cmp(&right)));
    let mut start_order: Vec<usize> = (0..n).collect();
    if let Some(values) = start {
        start_order.sort_by(|&left, &right| {
            values[left]
                .total_cmp(&values[right])
                .then(left.cmp(&right))
        });
    }
    let mut start_pos = 0;
    if start.is_none() {
        for row_idx in 0..n {
            update_cluster_risk_sums(
                row_idx,
                1.0,
                covariates,
                weights,
                clusters,
                &mut risk_sums,
                test_clusters,
                test_risk_sums.as_mut(),
            );
        }
        start_pos = n;
    }
    let mut stop_pos = 0;

    for (time_idx, moment) in moments.iter().enumerate() {
        if let Some(values) = start {
            while start_pos < n && values[start_order[start_pos]] < moment.time {
                update_cluster_risk_sums(
                    start_order[start_pos],
                    1.0,
                    covariates,
                    weights,
                    clusters,
                    &mut risk_sums,
                    test_clusters,
                    test_risk_sums.as_mut(),
                );
                start_pos += 1;
            }
        }
        while stop_pos < n && stop[stop_order[stop_pos]] < moment.time {
            update_cluster_risk_sums(
                stop_order[stop_pos],
                -1.0,
                covariates,
                weights,
                clusters,
                &mut risk_sums,
                test_clusters,
                test_risk_sums.as_mut(),
            );
            stop_pos += 1;
        }

        let total_event_weight: f64 = moment
            .events
            .iter()
            .map(|&idx| weights.map_or(1.0, |values| values[idx]))
            .sum();
        let cluster_values = cluster_influence_at_time(
            moment,
            covariates,
            weights,
            clusters,
            &risk_sums,
            total_event_weight,
        );
        for (cluster_idx, values) in cluster_values.iter().enumerate() {
            for (column_values, &value) in dfbeta[cluster_idx].iter_mut().zip(&values.dfbeta) {
                column_values[time_idx] = value;
            }
        }

        let test_values = if let Some(test_sums) = &test_risk_sums {
            cluster_influence_at_time(
                moment,
                covariates,
                weights,
                test_clusters,
                test_sums,
                total_event_weight,
            )
        } else {
            cluster_values
        };
        for (cluster_idx, values) in test_values.iter().enumerate() {
            for (column, (test_influence, &dfbeta)) in test_influence[cluster_idx]
                .iter_mut()
                .zip(&values.dfbeta)
                .enumerate()
            {
                let test_value = if test == "nrisk" {
                    dfbeta * moment.risk
                } else if test == "variance" && nvar > 1 && column > 0 {
                    values.centered_score[column - 1]
                } else {
                    dfbeta * moment.time_weight[column]
                };
                *test_influence += test_value;
            }
        }
    }

    let mut robust = vec![0.0; width * width];
    for row in &test_influence {
        add_outer_product(&mut robust, row);
    }
    (dfbeta, nested_square(robust, width))
}

#[allow(clippy::too_many_arguments)]
fn influence_values(
    moments: &[RiskMoment],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    stop: &[f64],
    weights: Option<&[f64]>,
    cluster: Option<&[i32]>,
    test_cluster: Option<&[i32]>,
    test: &str,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    if let Some(clusters) = cluster {
        influence_values_clustered(
            moments,
            covariates,
            start,
            stop,
            weights,
            clusters,
            test_cluster,
            test,
        )
    } else {
        influence_values_rowwise(
            moments,
            covariates,
            start,
            stop,
            weights,
            None,
            test_cluster,
            test,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    stop,
    status,
    covariates,
    start=None,
    weights=None,
    cluster=None,
    qrtol=1e-7,
    nmin=None,
    dfbeta=false,
    taper=None,
    test="aalen".to_string(),
    test_cluster=None
))]
#[allow(clippy::too_many_arguments)]
pub fn aareg_fit(
    stop: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    cluster: Option<Vec<i32>>,
    qrtol: f64,
    nmin: Option<usize>,
    dfbeta: bool,
    taper: Option<Vec<f64>>,
    test: String,
    test_cluster: Option<Vec<i32>>,
) -> PyResult<AaregFitResult> {
    let taper = taper.unwrap_or_else(|| vec![1.0]);
    let nvar = validate_inputs(
        &stop,
        &status,
        &covariates,
        start.as_deref(),
        weights.as_deref(),
        cluster.as_deref(),
        test_cluster.as_deref(),
    )?;
    validate_fit_options(qrtol, &taper)?;
    let normalized_test = test.to_ascii_lowercase();
    if !matches!(normalized_test.as_str(), "aalen" | "variance" | "nrisk") {
        return Err(value_error("test must be one of aalen, variance, or nrisk"));
    }

    let mut moments = risk_moments(
        &stop,
        &status,
        &covariates,
        start.as_deref(),
        weights.as_deref(),
        nvar,
    );
    let total_times = moments.len();
    taper_covariances(&mut moments, &taper, nvar);
    let retained = prepare_retained_moments(&mut moments, nvar, nmin, qrtol)?;
    moments.truncate(retained);

    let grouped_coefficients: Vec<Vec<Vec<f64>>> = moments
        .iter()
        .map(|moment| {
            moment
                .events
                .iter()
                .map(|&event_idx| {
                    coefficient_row(moment, event_idx, &covariates, weights.as_deref())
                })
                .collect()
        })
        .collect();
    let (test_statistic, test_variance) = model_test(
        &moments,
        &grouped_coefficients,
        &covariates,
        weights.as_deref(),
        &normalized_test,
    );
    let (dfbeta_values, robust_test_variance) = if dfbeta {
        let (values, variance) = influence_values(
            &moments,
            &covariates,
            start.as_deref(),
            &stop,
            weights.as_deref(),
            cluster.as_deref(),
            test_cluster.as_deref(),
            &normalized_test,
        );
        (Some(values), Some(variance))
    } else {
        (None, None)
    };

    let mut times = Vec::new();
    let mut n_risk = Vec::new();
    let mut coefficient = Vec::new();
    let mut time_weights = Vec::new();
    for (moment, rows) in moments.iter().zip(grouped_coefficients) {
        for row in rows {
            times.push(moment.time);
            n_risk.push(moment.risk);
            coefficient.push(row);
            time_weights.push(moment.time_weight.clone());
        }
    }

    Ok(AaregFitResult {
        n: vec![stop.len(), retained, total_times],
        times,
        n_risk,
        coefficient,
        test_statistic,
        test_variance,
        test: normalized_test,
        time_weights,
        dfbeta: dfbeta_values,
        robust_test_variance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-11,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn right_censored_fit_matches_reference_values() {
        let result = aareg_fit(
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
            vec![1, 1, 1, 1, 0, 1],
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, -1.0],
                vec![-1.0, 0.0],
            ],
            None,
            None,
            None,
            1e-7,
            Some(1),
            false,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("fit should succeed");

        assert_eq!(result.n, vec![6, 3, 4]);
        assert_eq!(result.times, vec![1.0, 2.0, 2.0, 3.0]);
        assert_eq!(result.n_risk, vec![6.0, 5.0, 5.0, 3.0]);
        let expected = [
            [0.225490196078431, -0.088235294117647, 0.0588235294117647],
            [0.278538812785388, -0.0365296803652968, -0.0867579908675799],
            [0.00456621004566216, 0.114155251141552, 0.146118721461187],
            [0.1, 0.1, 0.4],
        ];
        for (actual, expected_row) in result.coefficient.iter().zip(expected) {
            for (&actual_value, expected_value) in actual.iter().zip(expected_row) {
                assert_close(actual_value, expected_value);
            }
        }
        let expected_statistic = [1.45943838575418, 0.549950049950051, 2.26212121212121];
        for (&actual, expected) in result.test_statistic.iter().zip(expected_statistic) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn counting_weighted_fit_matches_reference_values() {
        let result = aareg_fit(
            vec![1.0, 3.0, 3.0, 4.0, 4.0, 2.0],
            vec![1, 1, 0, 1, 0, 1],
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, -1.0],
                vec![-1.0, 0.0],
            ],
            Some(vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0]),
            Some(vec![1.0, 2.0, 0.5, 1.5, 1.0, 3.0]),
            None,
            1e-7,
            Some(1),
            false,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("fit should succeed");

        assert_eq!(result.n, vec![6, 3, 4]);
        assert_eq!(result.times, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.n_risk, vec![4.5, 7.0, 5.0]);
        let expected = [
            [1.0, -1.0, 0.0],
            [0.53030303030303, -0.439393939393939, -0.0151515151515152],
            [1.69411764705882, -0.705882352941176, -0.470588235294118],
        ];
        for (actual, expected_row) in result.coefficient.iter().zip(expected) {
            for (&actual_value, expected_value) in actual.iter().zip(expected_row) {
                assert_close(actual_value, expected_value);
            }
        }
    }

    #[test]
    fn clustered_influence_matches_reference_values() {
        let result = aareg_fit(
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
            vec![1, 1, 1, 1, 0, 1],
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![1.0],
                vec![3.0],
                vec![-1.0],
            ],
            Some(vec![0.0; 6]),
            None,
            Some(vec![0, 0, 1, 1, 2, 2]),
            1e-7,
            Some(1),
            true,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("fit should succeed");

        let dfbeta = result.dfbeta.expect("dfbeta should be present");
        let expected = [
            [
                [0.167777777777778, 0.139462809917355, 0.0, 0.0],
                [-0.0733333333333333, -0.0139462809917355, 0.0, 0.0],
            ],
            [
                [
                    -0.0322222222222222,
                    -0.0382231404958678,
                    0.222222222222222,
                    0.0,
                ],
                [-0.00666666666666667, 0.0583677685950413, 0.0, 0.0],
            ],
            [
                [
                    -0.135555555555556,
                    -0.101239669421488,
                    -0.222222222222222,
                    0.0,
                ],
                [0.08, -0.0444214876033058, 0.0, 0.0],
            ],
        ];
        for (actual_cluster, expected_cluster) in dfbeta.iter().zip(expected) {
            for (actual_column, expected_column) in actual_cluster.iter().zip(expected_cluster) {
                for (&actual, expected) in actual_column.iter().zip(expected_column) {
                    assert_close(actual, expected);
                }
            }
        }
        let robust = result
            .robust_test_variance
            .expect("robust test variance should be present");
        let expected_robust = [
            [2.70951324322773, -1.27139864554637],
            [-1.27139864554637, 1.09997704315886],
        ];
        for (actual_row, expected_row) in robust.iter().zip(expected_robust) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn tapered_fit_matches_reference_values() {
        let result = aareg_fit(
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0],
            vec![1, 1, 1, 1, 0, 1, 1, 1],
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, -1.0],
                vec![-1.0, 0.0],
                vec![0.25, 0.5],
                vec![1.5, -0.5],
            ],
            None,
            Some(vec![1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 1.25, 0.75]),
            None,
            1e-7,
            Some(1),
            false,
            Some(vec![1.0, 2.0, 4.0]),
            "aalen".to_string(),
            None,
        )
        .expect("tapered fit should succeed");

        assert_eq!(result.n, vec![8, 5, 6]);
        let expected_statistic = [6.03082880188496, -1.10314578979036, 3.14780371938323];
        for (&actual, expected) in result.test_statistic.iter().zip(expected_statistic) {
            assert_close(actual, expected);
        }
        let expected_last_coefficient = [0.590576698540064, -0.0637179490991195, 0.641764618999422];
        for (&actual, expected) in result
            .coefficient
            .last()
            .expect("fit should retain coefficients")
            .iter()
            .zip(expected_last_coefficient)
        {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn variance_test_matches_reference_values() {
        let result = aareg_fit(
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
            vec![1, 1, 1, 1, 0, 1],
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, -1.0],
                vec![-1.0, 0.0],
            ],
            None,
            Some(vec![1.0, 2.0, 0.5, 1.5, 1.0, 3.0]),
            None,
            1e-7,
            Some(1),
            false,
            None,
            "variance".to_string(),
            None,
        )
        .expect("variance test should succeed");

        let expected_statistic = [2.18465909090909, 2.78440656565657];
        for (&actual, expected) in result.test_statistic.iter().zip(expected_statistic) {
            assert_close(actual, expected);
        }
        let expected_variance = [
            [2.72230920712810, 2.07232570735767],
            [2.07232570735767, 6.90703924105321],
        ];
        for (actual_row, expected_row) in result.test_variance.iter().zip(expected_variance) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn distinct_test_clusters_match_counting_process_reference_order() {
        let result = aareg_fit(
            vec![1.0, 3.0, 3.0, 4.0, 4.0, 2.0],
            vec![1, 1, 0, 1, 0, 1],
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, -1.0],
                vec![-1.0, 0.0],
            ],
            Some(vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0]),
            Some(vec![1.0, 2.0, 0.5, 1.5, 1.0, 3.0]),
            Some(vec![0, 0, 1, 1, 2, 2]),
            1e-7,
            Some(1),
            true,
            Some(vec![1.0, 2.0]),
            "aalen".to_string(),
            Some(vec![0, 1, 1, 2, 2, 0]),
        )
        .expect("fit should succeed");

        let robust = result
            .robust_test_variance
            .expect("robust test variance should be present");
        let expected = [
            [0.637362888575849, -0.223925492768809, -0.320475445723996],
            [-0.223925492768809, 0.41881161043852, -0.107835817285858],
            [-0.320475445723996, -0.107835817285858, 0.305063630729396],
        ];
        for (actual_row, expected_row) in robust.iter().zip(expected) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn clustered_risk_sweep_matches_rowwise_influence() {
        let n = 180;
        let nvar = 3;
        let stop: Vec<f64> = (0..n)
            .map(|row| 1.0 + (row % 31) as f64 * 0.4 + (row / 31) as f64 * 0.003)
            .collect();
        let status: Vec<i32> = (0..n)
            .map(|row| i32::from(row % 4 != 0 && row % 9 != 0))
            .collect();
        let covariates: Vec<Vec<f64>> = (0..n)
            .map(|row| {
                (0..nvar)
                    .map(|column| {
                        (row % (13 + column)) as f64 * 0.07
                            + (row * (column + 3) % 17) as f64 * 0.013
                            - column as f64 * 0.2
                    })
                    .collect()
            })
            .collect();
        let weights: Vec<f64> = (0..n).map(|row| 0.6 + (row % 7) as f64 * 0.15).collect();
        let clusters: Vec<i32> = (0..n).map(|row| (row % 11) as i32).collect();
        let test_clusters: Vec<i32> = (0..n).map(|row| (row % 7) as i32).collect();
        let counting_start: Vec<f64> = stop
            .iter()
            .enumerate()
            .map(|(row, value)| value - 0.5 - (row % 6) as f64 * 0.2)
            .collect();

        for start in [None, Some(counting_start.as_slice())] {
            let mut moments =
                risk_moments(&stop, &status, &covariates, start, Some(&weights), nvar);
            let retained = prepare_retained_moments(&mut moments, nvar, Some(12), 1e-7)
                .expect("deterministic input should retain full-rank moments");
            moments.truncate(retained);

            for test in ["aalen", "variance", "nrisk"] {
                for test_cluster in [None, Some(test_clusters.as_slice())] {
                    let expected = influence_values_rowwise(
                        &moments,
                        &covariates,
                        start,
                        &stop,
                        Some(&weights),
                        Some(&clusters),
                        test_cluster,
                        test,
                    );
                    let actual = influence_values_clustered(
                        &moments,
                        &covariates,
                        start,
                        &stop,
                        Some(&weights),
                        &clusters,
                        test_cluster,
                        test,
                    );

                    for (actual_cluster, expected_cluster) in actual.0.iter().zip(&expected.0) {
                        for (actual_column, expected_column) in
                            actual_cluster.iter().zip(expected_cluster)
                        {
                            for (&actual_value, &expected_value) in
                                actual_column.iter().zip(expected_column)
                            {
                                let tolerance = 1e-9 * expected_value.abs().max(1.0);
                                assert!(
                                    (actual_value - expected_value).abs() <= tolerance,
                                    "{test} dfbeta mismatch: expected {expected_value}, got {actual_value}"
                                );
                            }
                        }
                    }
                    for (actual_row, expected_row) in actual.1.iter().zip(&expected.1) {
                        for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row)
                        {
                            let tolerance = 1e-9 * expected_value.abs().max(1.0);
                            assert!(
                                (actual_value - expected_value).abs() <= tolerance,
                                "{test} variance mismatch: expected {expected_value}, got {actual_value}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_survival_inputs() {
        let error = aareg_fit(
            vec![1.0],
            vec![2],
            vec![vec![0.0]],
            None,
            None,
            None,
            1e-7,
            None,
            false,
            None,
            "aalen".to_string(),
            None,
        )
        .expect_err("non-binary status should fail");
        assert!(
            error
                .to_string()
                .contains("status must contain only 0 and 1")
        );
    }
}
