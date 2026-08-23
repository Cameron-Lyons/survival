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
    risk_count: usize,
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

fn reference_risk_moment(
    time: f64,
    order: &[usize],
    stop: &[f64],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    weights: Option<&[f64]>,
    center: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let nvar = center.len();
    let mut risk = 0.0;
    let mut first = vec![0.0; nvar];
    let mut second = vec![0.0; nvar * nvar];
    for &idx in order {
        if stop[idx] < time || start.is_some_and(|values| values[idx] >= time) {
            continue;
        }
        let weight = weights.map_or(1.0, |values| values[idx]);
        risk += weight;
        for row in 0..nvar {
            let centered_row = covariates[idx][row] - center[row];
            first[row] = weight.mul_add(centered_row, first[row]);
            for column in 0..=row {
                let centered_column = covariates[idx][column] - center[column];
                let offset = row * nvar + column;
                second[offset] = (weight * centered_row).mul_add(centered_column, second[offset]);
            }
        }
    }

    debug_assert!(risk > 0.0);
    let mut mean = vec![0.0; nvar];
    let mut covariance = vec![0.0; nvar * nvar];
    for row in 0..nvar {
        let centered_mean = first[row] / risk;
        mean[row] = center[row] + centered_mean;
        for column in 0..=row {
            let value = (-centered_mean).mul_add(first[column], second[row * nvar + column]) / risk;
            covariance[row * nvar + column] = value;
            covariance[column * nvar + row] = value;
        }
    }
    (mean, covariance)
}

fn risk_moments(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    weights: Option<&[f64]>,
    nvar: usize,
    qrtol: f64,
) -> Vec<RiskMoment> {
    let groups = event_groups(stop, status);
    let n = stop.len();
    // Preserve the upstream stop/status order for cancellation-dominated
    // or reduced-rank moments without replacing the ordinary sorted sweep.
    let risk_moment_reference = (nvar > 0).then(|| {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| {
            stop[left]
                .total_cmp(&stop[right])
                .then_with(|| status[right].cmp(&status[left]))
                .then(left.cmp(&right))
        });
        let mut center = vec![0.0; nvar];
        for &idx in &order {
            for column in 0..nvar {
                center[column] += covariates[idx][column];
            }
        }
        for value in &mut center {
            *value /= n as f64;
        }
        (center, order)
    });
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
    let mut risk_count = 0;
    let mut start_pos = 0;
    if start.is_none() {
        for idx in 0..n {
            update_risk_sums(idx, 1.0, covariates, weights, &mut s0, &mut s1, &mut s2);
        }
        risk_count = n;
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
                risk_count += 1;
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
            risk_count -= 1;
            stop_pos += 1;
        }

        let mut mean = vec![0.0; nvar];
        let mut covariance = vec![0.0; nvar * nvar];
        let mut precomputed_inverse = Vec::new();
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
            if let Some((center, order)) = &risk_moment_reference {
                let cancellation_dominated = (0..nvar).any(|column| {
                    let scale =
                        (s2[column * nvar + column] / s0).abs() + mean[column].abs().powi(2);
                    covariance[column * nvar + column].abs() <= 1e-10 * scale
                });
                let sweep_inverse = (nvar > 1)
                    .then(|| invert_matrix(&covariance, nvar, qrtol))
                    .flatten();
                let reduced_rank = nvar > 1 && sweep_inverse.is_none();
                let ill_conditioned = nvar > 1
                    && !reduced_rank
                    && sweep_inverse.as_ref().is_some_and(|inverse| {
                        let covariance_scale = covariance
                            .iter()
                            .map(|value| value.abs())
                            .fold(0.0_f64, f64::max);
                        let inverse_scale = inverse
                            .iter()
                            .map(|value| value.abs())
                            .fold(0.0_f64, f64::max);
                        covariance_scale * inverse_scale >= 1e5
                    });
                if cancellation_dominated || reduced_rank || ill_conditioned {
                    (mean, covariance) = reference_risk_moment(
                        time, order, stop, covariates, start, weights, center,
                    );
                } else if let Some(inverse) = sweep_inverse {
                    precomputed_inverse = inverse;
                }
            }
        }
        moments.push(RiskMoment {
            time,
            events,
            risk_count,
            risk: s0,
            mean,
            covariance,
            inverse: precomputed_inverse,
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
        moment.inverse.clear();
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

struct QrCoefficientMatrix {
    rank: usize,
    coefficients: Vec<f64>,
}

fn euclidean_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |norm, value| norm.hypot(*value))
}

fn qr_coefficient_matrix(values: &[f64], n: usize, tolerance: f64) -> QrCoefficientMatrix {
    if n == 0 {
        return QrCoefficientMatrix {
            rank: 0,
            coefficients: Vec::new(),
        };
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut active_columns = Vec::with_capacity(n);
    let mut triangular = vec![0.0; n * n];
    for column in 0..n {
        let original: Vec<f64> = (0..n).map(|row| values[row * n + column]).collect();
        let original_norm = euclidean_norm(&original);
        let mut residual = original.clone();
        let mut projections = Vec::with_capacity(basis.len());
        for direction in &basis {
            let projection = dot(direction, &residual);
            projections.push(projection);
            for row in 0..n {
                residual[row] -= projection * direction[row];
            }
        }
        let residual_norm = euclidean_norm(&residual);
        if original_norm == 0.0
            || !residual_norm.is_finite()
            || residual_norm < tolerance * original_norm
        {
            continue;
        }

        let position = basis.len();
        for (row, projection) in projections.into_iter().enumerate() {
            triangular[row * n + position] = projection;
        }
        triangular[position * n + position] = residual_norm;
        for value in &mut residual {
            *value /= residual_norm;
        }
        basis.push(residual);
        active_columns.push(column);
    }

    let rank = basis.len();
    let mut coefficients = vec![f64::NAN; n * n];
    for rhs in 0..n {
        let mut solution: Vec<f64> = basis.iter().map(|direction| direction[rhs]).collect();
        for row in (0..rank).rev() {
            for column in (row + 1)..rank {
                solution[row] -= triangular[row * n + column] * solution[column];
            }
            solution[row] /= triangular[row * n + row];
        }
        for (position, &column) in active_columns.iter().enumerate() {
            coefficients[column * n + rhs] = solution[position];
        }
    }

    QrCoefficientMatrix { rank, coefficients }
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
            && moments[retained - 1].inverse.len() != nvar * nvar
            && qr_coefficient_matrix(&moments[retained - 1].covariance, nvar, qrtol).rank < nvar
        {
            retained -= 1;
        }
    }
    if retained <= 1 {
        return Err(value_error(
            "the nmin threshold is too high; no Aalen model can be fit",
        ));
    }
    for moment in moments.iter_mut().take(retained) {
        if moment.inverse.len() != nvar * nvar {
            let qr = qr_coefficient_matrix(&moment.covariance, nvar, qrtol);
            moment.inverse = if qr.rank == nvar {
                invert_matrix(&moment.covariance, nvar, qrtol).unwrap_or(qr.coefficients)
            } else {
                qr.coefficients
            };
        }
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
        .map(|column| event_weight * (covariates[event_idx][column] - moment.mean[column]))
        .collect();
    let mut slopes = matrix_vector_product(&moment.inverse, &rhs, nvar);
    for value in &mut slopes {
        *value /= moment.risk;
    }
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

#[derive(Clone)]
struct GroupRiskMoment {
    weight: f64,
    first: Vec<f64>,
    second: Vec<f64>,
}

impl GroupRiskMoment {
    fn new(nvar: usize) -> Self {
        Self {
            weight: 0.0,
            first: vec![0.0; nvar],
            second: vec![0.0; nvar * nvar],
        }
    }

    fn update(&mut self, row: &[f64], weight: f64, direction: f64) {
        let signed_weight = direction * weight;
        self.weight += signed_weight;
        for row_idx in 0..row.len() {
            self.first[row_idx] += signed_weight * row[row_idx];
            for column_idx in 0..row.len() {
                self.second[row_idx * row.len() + column_idx] +=
                    signed_weight * row[row_idx] * row[column_idx];
            }
        }
    }
}

fn update_group_risk_moments(
    groups: &mut [GroupRiskMoment],
    group_codes: &[i32],
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    row_idx: usize,
    direction: f64,
) {
    let group_idx = group_codes[row_idx] as usize;
    let weight = weights.map_or(1.0, |values| values[row_idx]);
    groups[group_idx].update(&covariates[row_idx], weight, direction);
}

fn event_group_moments(
    moment: &RiskMoment,
    group_codes: &[i32],
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
    group_count: usize,
    nvar: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut event_weight = vec![0.0; group_count];
    let mut event_first = vec![vec![0.0; nvar]; group_count];
    for &row_idx in &moment.events {
        let group_idx = group_codes[row_idx] as usize;
        let weight = weights.map_or(1.0, |values| values[row_idx]);
        event_weight[group_idx] += weight;
        for column in 0..nvar {
            event_first[group_idx][column] += weight * covariates[row_idx][column];
        }
    }
    (event_weight, event_first)
}

fn grouped_influence_at_time(
    moment: &RiskMoment,
    risk: &GroupRiskMoment,
    event_weight: f64,
    event_first: &[f64],
    total_event_weight: f64,
    summed_slopes: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let nvar = moment.mean.len();
    let mut centered_first = vec![0.0; nvar];
    for (column, value) in centered_first.iter_mut().enumerate() {
        *value = risk.first[column] - risk.weight * moment.mean[column];
    }

    let predicted_intercept = total_event_weight / moment.risk;
    let qsum =
        event_weight - predicted_intercept * risk.weight - dot(&centered_first, summed_slopes);
    if nvar == 0 {
        return (vec![qsum / moment.risk], Vec::new());
    }

    let mut qcenter = vec![0.0; nvar];
    for row in 0..nvar {
        let mut predicted_slope = 0.0;
        for (column, slope) in summed_slopes.iter().enumerate() {
            let centered_second = risk.second[row * nvar + column]
                - moment.mean[row] * risk.first[column]
                - risk.first[row] * moment.mean[column]
                + risk.weight * moment.mean[row] * moment.mean[column];
            predicted_slope += centered_second * slope;
        }
        qcenter[row] = event_first[row]
            - event_weight * moment.mean[row]
            - predicted_intercept * centered_first[row]
            - predicted_slope;
    }

    let mut slopes = matrix_vector_product(&moment.inverse, &qcenter, nvar);
    for value in &mut slopes {
        *value /= moment.risk;
    }
    let mut temp = Vec::with_capacity(nvar + 1);
    temp.push(qsum / moment.risk - dot(&slopes, &moment.mean));
    temp.extend_from_slice(&slopes);
    (temp, qcenter)
}

fn add_group_test_influence(
    influence: &mut [f64],
    temp: &[f64],
    qcenter: &[f64],
    moment: &RiskMoment,
    test: &str,
) {
    let nvar = moment.mean.len();
    for column in 0..influence.len() {
        let test_value = if test == "nrisk" {
            temp[column] * moment.risk
        } else if test == "variance" && nvar > 1 && column > 0 {
            qcenter[column - 1]
        } else {
            temp[column] * moment.time_weight[column]
        };
        influence[column] += test_value;
    }
}

fn singleton_influence_at_time(
    moment: &RiskMoment,
    event_idx: usize,
    covariates: &[Vec<f64>],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let nvar = moment.mean.len();
    let event_weight = weights.map_or(1.0, |values| values[event_idx]);
    let coefficient = coefficient_row(moment, event_idx, covariates, weights);
    let centered: Vec<f64> = (0..nvar)
        .map(|column| covariates[event_idx][column] - moment.mean[column])
        .collect();
    let predicted = event_weight / moment.risk + dot(&centered, &coefficient[1..]);
    let residual = 1.0 - predicted;
    let qcenter: Vec<f64> = centered
        .iter()
        .map(|value| residual * event_weight * value)
        .collect();
    let mut slopes = matrix_vector_product(&moment.inverse, &qcenter, nvar);
    for value in &mut slopes {
        *value /= moment.risk;
    }
    let mut temp = Vec::with_capacity(nvar + 1);
    temp.push(residual * event_weight / moment.risk - dot(&slopes, &moment.mean));
    temp.extend(slopes);
    (temp, qcenter)
}

#[allow(clippy::too_many_arguments)]
fn influence_values_grouped(
    moments: &[RiskMoment],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    stop: &[f64],
    weights: Option<&[f64]>,
    cluster: &[i32],
    test_cluster: &[i32],
    test: &str,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let n = stop.len();
    let nvar = covariates.first().map_or(0, Vec::len);
    let width = nvar + 1;
    let cluster_count = cluster.iter().max().copied().unwrap_or(0) as usize + 1;
    let test_cluster_count = test_cluster.iter().max().copied().unwrap_or(0) as usize + 1;
    let mut dfbeta = vec![vec![vec![0.0; moments.len()]; width]; cluster_count];
    let mut test_influence = vec![vec![0.0; width]; test_cluster_count];
    let mut cluster_risk = vec![GroupRiskMoment::new(nvar); cluster_count];
    let shared_groups = cluster == test_cluster;
    let mut test_cluster_risk =
        (!shared_groups).then(|| vec![GroupRiskMoment::new(nvar); test_cluster_count]);
    let mut active = vec![false; n];

    let mut start_order: Vec<usize> = (0..n).collect();
    if let Some(values) = start {
        start_order.sort_by(|&left, &right| {
            values[left]
                .total_cmp(&values[right])
                .then(left.cmp(&right))
        });
    }
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&left, &right| stop[left].total_cmp(&stop[right]).then(left.cmp(&right)));
    let mut start_cursor = 0;
    let mut stop_cursor = 0;

    if start.is_none() {
        for (row_idx, is_active) in active.iter_mut().enumerate() {
            update_group_risk_moments(
                &mut cluster_risk,
                cluster,
                covariates,
                weights,
                row_idx,
                1.0,
            );
            if let Some(test_risk) = &mut test_cluster_risk {
                update_group_risk_moments(
                    test_risk,
                    test_cluster,
                    covariates,
                    weights,
                    row_idx,
                    1.0,
                );
            }
            *is_active = true;
        }
        start_cursor = n;
    }

    for (time_idx, moment) in moments.iter().enumerate() {
        if let Some(values) = start {
            while start_cursor < n && values[start_order[start_cursor]] < moment.time {
                let row_idx = start_order[start_cursor];
                update_group_risk_moments(
                    &mut cluster_risk,
                    cluster,
                    covariates,
                    weights,
                    row_idx,
                    1.0,
                );
                if let Some(test_risk) = &mut test_cluster_risk {
                    update_group_risk_moments(
                        test_risk,
                        test_cluster,
                        covariates,
                        weights,
                        row_idx,
                        1.0,
                    );
                }
                active[row_idx] = true;
                start_cursor += 1;
            }
        }
        while stop_cursor < n && stop[stop_order[stop_cursor]] < moment.time {
            let row_idx = stop_order[stop_cursor];
            if active[row_idx] {
                update_group_risk_moments(
                    &mut cluster_risk,
                    cluster,
                    covariates,
                    weights,
                    row_idx,
                    -1.0,
                );
                if let Some(test_risk) = &mut test_cluster_risk {
                    update_group_risk_moments(
                        test_risk,
                        test_cluster,
                        covariates,
                        weights,
                        row_idx,
                        -1.0,
                    );
                }
                active[row_idx] = false;
            }
            stop_cursor += 1;
        }

        if moment.inverse.iter().any(|value| value.is_nan()) {
            for group in &mut dfbeta {
                for column in group {
                    column[time_idx] = f64::NAN;
                }
            }
            for group in &mut test_influence {
                group.fill(f64::NAN);
            }
            continue;
        }

        if moment.risk_count == 1 {
            // The one-covariate branch has exactly zero singleton influence
            // upstream. Multivariable fits retain row-level roundoff here.
            if nvar == 1 {
                continue;
            }
            let event_idx = moment.events[0];
            let (temp, qcenter) =
                singleton_influence_at_time(moment, event_idx, covariates, weights);
            let cluster_idx = cluster[event_idx] as usize;
            for column in 0..width {
                dfbeta[cluster_idx][column][time_idx] = temp[column];
            }
            let test_cluster_idx = test_cluster[event_idx] as usize;
            add_group_test_influence(
                &mut test_influence[test_cluster_idx],
                &temp,
                &qcenter,
                moment,
                test,
            );
            continue;
        }

        let total_event_weight: f64 = moment
            .events
            .iter()
            .map(|&idx| weights.map_or(1.0, |values| values[idx]))
            .sum();
        let mut summed_slopes = vec![0.0; nvar];
        for &event_idx in &moment.events {
            let coefficient = coefficient_row(moment, event_idx, covariates, weights);
            for column in 0..nvar {
                summed_slopes[column] += coefficient[column + 1];
            }
        }

        let (cluster_event_weight, cluster_event_first) =
            event_group_moments(moment, cluster, covariates, weights, cluster_count, nvar);
        for group_idx in 0..cluster_count {
            let (temp, qcenter) = grouped_influence_at_time(
                moment,
                &cluster_risk[group_idx],
                cluster_event_weight[group_idx],
                &cluster_event_first[group_idx],
                total_event_weight,
                &summed_slopes,
            );
            for column in 0..width {
                dfbeta[group_idx][column][time_idx] = temp[column];
            }
            if shared_groups {
                add_group_test_influence(
                    &mut test_influence[group_idx],
                    &temp,
                    &qcenter,
                    moment,
                    test,
                );
            }
        }

        if let Some(test_risk) = &test_cluster_risk {
            let (test_event_weight, test_event_first) = event_group_moments(
                moment,
                test_cluster,
                covariates,
                weights,
                test_cluster_count,
                nvar,
            );
            for group_idx in 0..test_cluster_count {
                let (temp, qcenter) = grouped_influence_at_time(
                    moment,
                    &test_risk[group_idx],
                    test_event_weight[group_idx],
                    &test_event_first[group_idx],
                    total_event_weight,
                    &summed_slopes,
                );
                add_group_test_influence(
                    &mut test_influence[group_idx],
                    &temp,
                    &qcenter,
                    moment,
                    test,
                );
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
fn influence_values_scanned(
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

    for (time_idx, moment) in moments.iter().enumerate() {
        for &event_idx in &moment.events {
            event_rows[event_idx] = true;
        }
        if moment.inverse.iter().any(|value| value.is_nan()) {
            for group in &mut dfbeta {
                for column in group {
                    column[time_idx] = f64::NAN;
                }
            }
            for group in &mut test_influence {
                group.fill(f64::NAN);
            }
            for &event_idx in &moment.events {
                event_rows[event_idx] = false;
            }
            continue;
        }
        if moment.risk_count == 1 {
            if nvar > 1 {
                let event_idx = moment.events[0];
                let (temp, qcenter) =
                    singleton_influence_at_time(moment, event_idx, covariates, weights);
                let cluster_idx = cluster.map_or(event_idx, |values| values[event_idx] as usize);
                let test_cluster_idx =
                    test_cluster.map_or(event_idx, |values| values[event_idx] as usize);
                for column in 0..width {
                    dfbeta[cluster_idx][column][time_idx] = temp[column];
                }
                add_group_test_influence(
                    &mut test_influence[test_cluster_idx],
                    &temp,
                    &qcenter,
                    moment,
                    test,
                );
            }
            for &event_idx in &moment.events {
                event_rows[event_idx] = false;
            }
            continue;
        }
        let total_event_weight: f64 = moment
            .events
            .iter()
            .map(|&idx| weights.map_or(1.0, |values| values[idx]))
            .sum();
        let mut summed_slopes = vec![0.0; nvar];
        for &event_idx in &moment.events {
            let coefficient = coefficient_row(moment, event_idx, covariates, weights);
            for column in 0..nvar {
                summed_slopes[column] += coefficient[column + 1];
            }
        }

        for row_idx in 0..n {
            let at_risk = row_at_risk(start, stop, row_idx, moment.time);
            let event_indicator = usize::from(event_rows[row_idx]) as f64;
            let row_weight = if at_risk {
                weights.map_or(1.0, |values| values[row_idx])
            } else {
                0.0
            };
            let centered: Vec<f64> = (0..nvar)
                .map(|column| covariates[row_idx][column] - moment.mean[column])
                .collect();
            let predicted = if at_risk {
                total_event_weight / moment.risk + dot(&centered, &summed_slopes)
            } else {
                0.0
            };
            let residual = event_indicator - predicted;
            let mut temp = vec![0.0; width];
            if nvar == 0 {
                temp[0] = residual * row_weight / moment.risk;
            } else {
                let score: Vec<f64> = centered
                    .iter()
                    .map(|value| residual * row_weight * value)
                    .collect();
                let mut slopes = matrix_vector_product(&moment.inverse, &score, nvar);
                for value in &mut slopes {
                    *value /= moment.risk;
                }
                temp[0] = residual * row_weight / moment.risk - dot(&slopes, &moment.mean);
                temp[1..].copy_from_slice(&slopes);
            }
            let cluster_idx = cluster.map_or(row_idx, |values| values[row_idx] as usize);
            let test_cluster_idx = test_cluster.map_or(row_idx, |values| values[row_idx] as usize);
            for column in 0..width {
                dfbeta[cluster_idx][column][time_idx] += temp[column];
                let test_value = if test == "nrisk" {
                    temp[column] * moment.risk
                } else if test == "variance" && nvar > 1 && column > 0 {
                    residual * row_weight * centered[column - 1]
                } else {
                    temp[column] * moment.time_weight[column]
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
    if let Some(cluster_values) = cluster {
        let test_cluster_values = test_cluster.unwrap_or(cluster_values);
        let cluster_count = cluster_values.iter().max().copied().unwrap_or(0) as usize + 1;
        let test_cluster_count =
            test_cluster_values.iter().max().copied().unwrap_or(0) as usize + 1;
        let scanned_work = stop.len().saturating_mul(moments.len());
        let grouped_work = stop.len().saturating_add(
            moments
                .len()
                .saturating_mul(cluster_count.saturating_add(test_cluster_count)),
        );
        if scanned_work > grouped_work.saturating_mul(2) {
            return influence_values_grouped(
                moments,
                covariates,
                start,
                stop,
                weights,
                cluster_values,
                test_cluster_values,
                test,
            );
        }
    }
    influence_values_scanned(
        moments,
        covariates,
        start,
        stop,
        weights,
        cluster,
        test_cluster,
        test,
    )
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
        qrtol,
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
    fn single_covariate_retains_positive_reference_roundoff_variance() {
        let result = aareg_fit(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 1, 1, 1],
            vec![
                vec![-0.626453810742332],
                vec![0.183643324222082],
                vec![-0.835628612410047],
                vec![1.59528080213779],
                vec![0.32950777181536],
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
        .expect("fit should retain the final reference event");

        assert_eq!(result.n, vec![5, 5, 5]);
        assert_eq!(result.times, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.coefficient[4], vec![1.0, 0.0]);
        assert_close(result.time_weights[4][0], 7.92486993001903e-18);
        assert_close(result.time_weights[4][1], 8.60445698220756e-19);

        let clustered = aareg_fit(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 1, 1, 1],
            vec![
                vec![-0.626453810742332],
                vec![0.183643324222082],
                vec![-0.835628612410047],
                vec![1.59528080213779],
                vec![0.32950777181536],
            ],
            None,
            None,
            Some(vec![0, 1, 2, 3, 4]),
            1e-7,
            Some(1),
            true,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("clustered fit should retain the final reference event");
        let influence = clustered.dfbeta.expect("clustered influence");
        for cluster in influence {
            for column in cluster {
                assert_eq!(column[4], 0.0);
            }
        }
    }

    #[test]
    fn multivariate_fit_retains_early_reduced_rank_moments() {
        let result = aareg_fit(
            vec![
                8.0, 10.0, 1.0, 7.0, 11.0, 5.0, 3.0, 12.0, 1.0, 11.0, 5.0, 8.0, 9.0, 12.0, 1.0, 9.0,
            ],
            vec![0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            vec![
                vec![0.573908926345705, -0.998654664215969],
                vec![0.557383157775316, 0.0747227936381808],
                vec![-0.048318399946546, 0.351874693602948],
                vec![1.77926496475137, -0.8879259478384],
                vec![-0.994176155979485, 0.537219149435208],
                vec![0.640347136457419, -0.837719990909545],
                vec![0.14103345783844, -1.66434120525646],
                vec![-0.54381139957941, 0.227724188582013],
                vec![0.481044344724356, 0.795642091749136],
                vec![0.515125707267044, 0.318483478317856],
                vec![0.273248501950473, -0.189220036134785],
                vec![-0.783292341519974, -0.881718032267188],
                vec![-0.236489544926223, 0.358906236331899],
                vec![0.112112836045669, 0.963535718907729],
                vec![-1.27275904597986, 0.672340443408467],
                vec![0.990232610649446, -0.091722475837281],
            ],
            Some(vec![
                7.0, 5.0, 0.0, 6.0, 7.0, 4.0, 1.0, 9.0, 0.0, 7.0, 4.0, 7.0, 7.0, 11.0, 0.0, 5.0,
            ]),
            Some(vec![
                2.0, 3.0, 3.0, 1.5, 1.0, 2.0, 2.0, 1.0, 1.5, 2.0, 3.0, 0.5, 3.0, 3.0, 3.0, 1.0,
            ]),
            Some((0..16).collect()),
            1e-7,
            Some(0),
            true,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("fit should retain estimable rows before later full-rank moments");

        assert_eq!(result.n, vec![16, 8, 8]);
        assert_eq!(
            result.times,
            vec![1.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0]
        );
        assert_eq!(result.coefficient[2], vec![1.0, 0.0, 0.0]);
        assert!(result.coefficient[3][0].is_nan());
        assert_close(result.coefficient[3][1], -2.72406352408014);
        assert!(result.coefficient[3][2].is_nan());
        assert_close(result.time_weights[2][0], 2.20713894501395e-19);
        assert_close(result.time_weights[2][1], 4.82036442220766e-21);
        assert_close(result.time_weights[2][2], -1.30549597135516e-16);
        assert!(result.time_weights[3][0].is_nan());
        assert_close(result.time_weights[3][1], 0.666376318559675);
        assert!(result.time_weights[3][2].is_nan());
        assert!(result.test_statistic[0].is_nan());
        assert_close(result.test_statistic[1], 0.384207314024795);
        assert!(result.test_statistic[2].is_nan());
        let influence = result.dfbeta.expect("clustered influence");
        assert!(
            influence
                .iter()
                .all(|group| group.iter().all(|column| column[2].is_nan()))
        );
    }

    #[test]
    fn ill_conditioned_counting_moment_uses_reference_order() {
        let result = aareg_fit(
            vec![10.0, 12.0, 6.0, 2.0, 3.0, 6.0, 4.0, 4.0],
            vec![1, 0, 1, 1, 1, 1, 0, 1],
            vec![
                vec![0.302864245707098, -0.619339229584789],
                vec![0.643637278746447, -0.998442891497977],
                vec![-0.485867527842945, -0.426723367421423],
                vec![0.134741789184607, -0.842220842557035],
                vec![0.427573412166526, -0.813835914125232],
                vec![-1.70747127299651, -1.02980266064102],
                vec![0.563768815226198, -0.254829711379971],
                vec![-0.619311246208519, -0.887182144937102],
            ],
            Some(vec![5.0, 7.0, 2.0, 0.0, 0.0, 1.0, 3.0, 2.0]),
            Some(vec![0.5, 2.0, 1.0, 1.0, 3.0, 0.5, 0.5, 1.5]),
            Some(vec![1, 0, 0, 0, 0, 0, 1, 0]),
            1e-7,
            Some(3),
            true,
            None,
            "aalen".to_string(),
            None,
        )
        .expect("ill-conditioned fit should succeed");

        assert_eq!(result.n, vec![8, 3, 5]);
        assert_eq!(result.times, vec![2.0, 3.0, 4.0]);
        assert!((result.coefficient[0][0] - 693.465242967597).abs() < 1e-8);
        assert!((result.coefficient[0][1] - -81.8427155699274).abs() < 1e-8);
        assert!((result.coefficient[0][2] - 809.096111859553).abs() < 1e-8);
        let influence = result.dfbeta.expect("clustered influence");
        assert!((influence[0][0][0] - -4.08197157166326e-9).abs() < 1e-8);
        assert!((influence[0][1][0] - 4.8194566051338e-10).abs() < 1e-8);
        assert!((influence[0][2][0] - -4.76416887602925e-9).abs() < 1e-8);
    }

    #[test]
    fn multivariate_singleton_influence_preserves_reference_roundoff() {
        let x = [
            0x3fd7_574d_8210_7103,
            0x4002_37e8_d13d_5376,
            0xbfaf_1ba0_9cb1_dbbe,
            0x3fe2_71ab_0122_54a3,
            0xbfa4_a30c_733b_1cdd,
            0x3fe5_fccc_3dc3_cc19,
            0xbfe1_9c20_a5bc_d130,
            0xbfe5_fc55_42b2_3161,
            0xbfa6_2345_37bd_f792,
        ]
        .map(f64::from_bits);
        let z = [
            0xc000_1ada_d701_1a1d,
            0x3fce_108d_7399_60eb,
            0xbfc9_0bb6_ad82_f500,
            0xbfe9_1e0d_4f14_e16f,
            0x3ff9_a385_0922_6376,
            0x3fe3_fbdf_2e1b_edb2,
            0xbfcb_3f12_cc82_a54b,
            0x3fd1_89be_9b75_b29f,
            0x3ff8_15bc_ab45_24bf,
        ]
        .map(f64::from_bits);
        let covariates: Vec<Vec<f64>> = x
            .into_iter()
            .zip(z)
            .map(|(left, right)| vec![left, right])
            .collect();
        let result = aareg_fit(
            vec![4.0, 7.0, 3.0, 10.0, 12.0, 4.0, 1.0, 8.0, 11.0],
            vec![1, 1, 1, 1, 1, 0, 1, 0, 0],
            covariates.clone(),
            None,
            Some(vec![1.5, 1.5, 1.0, 1.0, 3.0, 0.5, 1.0, 3.0, 0.5]),
            Some(vec![0, 1, 2, 1, 0, 2, 2, 1, 1]),
            1e-7,
            Some(0),
            true,
            None,
            "nrisk".to_string(),
            Some(vec![2, 0, 1, 2, 1, 1, 0, 2, 1]),
        )
        .expect("terminal singleton fit should succeed");

        assert_eq!(result.n, vec![9, 6, 6]);
        assert_eq!(result.times, vec![1.0, 3.0, 4.0, 7.0, 10.0, 12.0]);
        let influence = result.dfbeta.expect("clustered influence");
        let expected_terminal = [
            1.2357462015422309e-8,
            -4.6083184432151884e-8,
            -8.870281696752633e-9,
        ];
        for (column, expected) in influence[0].iter().zip(expected_terminal) {
            assert_close(column[5], expected);
        }
        for group in &influence[1..] {
            for column in group {
                assert_eq!(column[5], 0.0);
            }
        }
        let robust = result
            .robust_test_variance
            .expect("robust variance should be present");
        let expected_robust = [
            [7.859538616305543, -3.0156466390232524, -3.261453899192717],
            [-3.0156466390232524, 1.1589204868909564, 1.2569946031746024],
            [-3.261453899192717, 1.2569946031746024, 1.3704417044925377],
        ];
        for (actual_row, expected_row) in robust.iter().zip(expected_robust) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(actual, expected);
            }
        }

        let grouped = aareg_fit(
            vec![4.0, 7.0, 3.0, 10.0, 12.0, 4.0, 1.0, 8.0, 11.0],
            vec![1, 1, 1, 1, 1, 0, 1, 0, 0],
            covariates,
            None,
            Some(vec![1.5, 1.5, 1.0, 1.0, 3.0, 0.5, 1.0, 3.0, 0.5]),
            Some(vec![0; 9]),
            1e-7,
            Some(0),
            true,
            None,
            "nrisk".to_string(),
            Some(vec![0; 9]),
        )
        .expect("grouped singleton fit should succeed");
        let grouped_influence = grouped.dfbeta.expect("grouped influence");
        for (column, expected) in grouped_influence[0].iter().zip(expected_terminal) {
            assert_close(column[5], expected);
        }
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
    fn grouped_influence_sweep_matches_row_scan() {
        let stop = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1];
        let start = vec![0.0, 0.0, 0.5, 0.0, 1.0, 1.5, 0.0, 2.0, 1.0, 0.0, 3.0, 2.0];
        let covariates: Vec<Vec<f64>> = (0..stop.len())
            .map(|idx| {
                let value = idx as f64;
                vec![
                    (value % 5.0) - 2.0,
                    ((idx * idx + 3 * idx) % 11) as f64 - 4.0,
                ]
            })
            .collect();
        let weights: Vec<f64> = (0..stop.len())
            .map(|idx| 0.75 + (idx % 4) as f64 * 0.25)
            .collect();
        let cluster: Vec<i32> = (0..stop.len()).map(|idx| (idx % 3) as i32).collect();
        let test_cluster: Vec<i32> = (0..stop.len()).map(|idx| (idx % 4) as i32).collect();

        let mut moments = risk_moments(
            &stop,
            &status,
            &covariates,
            Some(&start),
            Some(&weights),
            2,
            1e-7,
        );
        let retained = prepare_retained_moments(&mut moments, 2, Some(1), 1e-7)
            .expect("comparison inputs should retain multiple event times");
        moments.truncate(retained);

        for selected_test_cluster in [&cluster, &test_cluster] {
            for test in ["aalen", "variance", "nrisk"] {
                let (scanned_dfbeta, scanned_variance) = influence_values_scanned(
                    &moments,
                    &covariates,
                    Some(&start),
                    &stop,
                    Some(&weights),
                    Some(&cluster),
                    Some(selected_test_cluster),
                    test,
                );
                let (grouped_dfbeta, grouped_variance) = influence_values_grouped(
                    &moments,
                    &covariates,
                    Some(&start),
                    &stop,
                    Some(&weights),
                    &cluster,
                    selected_test_cluster,
                    test,
                );

                for (actual_cluster, expected_cluster) in grouped_dfbeta.iter().zip(&scanned_dfbeta)
                {
                    for (actual_column, expected_column) in
                        actual_cluster.iter().zip(expected_cluster)
                    {
                        for (&actual, &expected) in actual_column.iter().zip(expected_column) {
                            assert!(
                                (actual - expected).abs() < 1e-10,
                                "{test} dfbeta mismatch: expected {expected}, got {actual}"
                            );
                        }
                    }
                }
                for (actual_row, expected_row) in grouped_variance.iter().zip(&scanned_variance) {
                    for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                        assert!(
                            (actual - expected).abs() < 1e-9,
                            "{test} variance mismatch: expected {expected}, got {actual}"
                        );
                    }
                }
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
