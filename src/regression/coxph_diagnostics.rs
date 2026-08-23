use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN, TIME_EPSILON, same_time};
use crate::internal::matrix::{lu_solve, matrix_inverse};
use crate::internal::statistical::{chi2_sf, chi2_sf_continuous};
use crate::regression::cox_optimizer::{Method as CoxMethod, ProductAccumulator};
use crate::regression::coxph::CoxPHFit;
use crate::regression::coxph_detail_module::{
    CoxphDetail, CoxphDetailOptions, compute_coxph_detail_with_options, coxph_detail,
};
use crate::regression::coxph_support::{ActiveRiskSet, CoxSweepRow, StratifiedBaselineLookup};
use crate::regression::exact_ties::{exact_inclusion_probabilities, exact_tied_moments};
use crate::residuals::agmart_module::{AgmartData, compute_agmart};
use crate::residuals::coxmart_module::{CoxMartSurvivalData, CoxMartWeights, compute_coxmart};
use crate::scoring::coxscore2::{CoxScoreData, CoxScoreParams, compute_cox_score_residuals};
use crate::validation::ProportionalityTest;
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

pub(crate) type CoxZphMatrix = Vec<Vec<f64>>;
pub(crate) type CoxZphSurfaceDiagnostics = (CoxZphMatrix, CoxZphMatrix, ProportionalityTest);
type CoxZphSurface = (CoxZphMatrix, CoxZphMatrix);

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn validate_finite_slice(values: &[f64], name: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_matrix_width(matrix: &[Vec<f64>], width: usize, name: &str) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != width {
            return Err(value_error(format!(
                "{name} row {row_idx} has length {}, expected {width}",
                row.len()
            )));
        }
        validate_finite_slice(row, name)?;
    }
    Ok(())
}

#[pyfunction]
pub fn chi_square_survival(statistic: f64, degrees_of_freedom: f64) -> PyResult<f64> {
    if !statistic.is_finite() || statistic < 0.0 {
        return Err(value_error("statistic must be a finite non-negative value"));
    }
    if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return Err(value_error(
            "degrees_of_freedom must be a finite positive value",
        ));
    }
    Ok(chi2_sf_continuous(statistic, degrees_of_freedom).clamp(0.0, 1.0))
}

fn validate_square_matrix(matrix: &[Vec<f64>], width: usize, name: &str) -> PyResult<()> {
    if matrix.len() != width {
        return Err(value_error(format!("{name} length must be {width}")));
    }
    validate_matrix_width(matrix, width, name)
}

fn validate_column_groups(groups: &[Vec<usize>], width: usize) -> PyResult<()> {
    for (group_idx, columns) in groups.iter().enumerate() {
        if columns.is_empty() {
            return Err(value_error(format!("groups[{group_idx}] cannot be empty")));
        }
        for &col_idx in columns {
            if col_idx >= width {
                return Err(value_error(format!(
                    "groups[{group_idx}] contains column {col_idx}, expected < {width}"
                )));
            }
        }
    }
    Ok(())
}

fn invert_square_rows(matrix: &[Vec<f64>], name: &str) -> PyResult<Vec<Vec<f64>>> {
    let width = matrix.len();
    validate_square_matrix(matrix, width, name)?;
    let values = matrix.iter().flatten().copied().collect::<Vec<_>>();
    let array = Array2::from_shape_vec((width, width), values)
        .map_err(|_| value_error(format!("failed to construct {name}")))?;
    let inverse =
        matrix_inverse(&array).ok_or_else(|| value_error(format!("{name} is singular")))?;
    Ok((0..width)
        .map(|row| (0..width).map(|column| inverse[[row, column]]).collect())
        .collect())
}

fn validate_cluster_codes(codes: &[usize], nrows: usize, name: &str) -> PyResult<usize> {
    if codes.len() != nrows {
        return Err(value_error(format!("{name} length must match row count")));
    }
    Ok(codes
        .iter()
        .copied()
        .max()
        .map_or(0, |max_code| max_code + 1))
}

fn collapse_weighted_rows_by_cluster(
    rows: &[Vec<f64>],
    weights: &[f64],
    cluster: &[usize],
    width: usize,
    name: &str,
) -> PyResult<Vec<Vec<f64>>> {
    validate_matrix_width(rows, width, name)?;
    if weights.len() != rows.len() {
        return Err(value_error("weights length must match row count"));
    }
    validate_finite_slice(weights, "weights")?;
    let cluster_count = validate_cluster_codes(cluster, rows.len(), "cluster")?;
    let mut collapsed = vec![vec![0.0; width]; cluster_count];
    for ((row, &weight), &cluster_idx) in rows.iter().zip(weights).zip(cluster) {
        let target = &mut collapsed[cluster_idx];
        for (col_idx, value) in row.iter().enumerate() {
            target[col_idx] += weight * value;
        }
    }
    Ok(collapsed)
}

fn row_crossprod(rows: &[Vec<f64>], width: usize, name: &str) -> PyResult<Vec<Vec<f64>>> {
    validate_matrix_width(rows, width, name)?;
    let mut result = vec![vec![0.0; width]; width];
    for row in rows {
        for (left_idx, &left) in row.iter().enumerate() {
            for (right_idx, &right) in row.iter().enumerate() {
                result[left_idx][right_idx] += left * right;
            }
        }
    }
    Ok(result)
}

fn sandwich_from_meat(variance: &[Vec<f64>], meat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let width = variance.len();
    let mut left = vec![vec![0.0; width]; width];
    for (row_idx, variance_row) in variance.iter().enumerate() {
        for col_idx in 0..width {
            left[row_idx][col_idx] = variance_row
                .iter()
                .enumerate()
                .map(|(inner_idx, &value)| value * meat[inner_idx][col_idx])
                .sum();
        }
    }

    let mut result = vec![vec![0.0; width]; width];
    for (row_idx, left_row) in left.iter().enumerate() {
        for col_idx in 0..width {
            result[row_idx][col_idx] = left_row
                .iter()
                .enumerate()
                .map(|(inner_idx, &value)| value * variance[inner_idx][col_idx])
                .sum();
        }
    }
    result
}

fn quadratic_form(row: &[f64], variance: &[Vec<f64>]) -> f64 {
    row.iter()
        .enumerate()
        .map(|(left_idx, &left)| {
            row.iter()
                .enumerate()
                .map(|(right_idx, &right)| left * variance[left_idx][right_idx] * right)
                .sum::<f64>()
        })
        .sum()
}

fn grouped_quadratic_form(row: &[f64], variance: &[Vec<f64>], columns: &[usize]) -> f64 {
    columns
        .iter()
        .map(|&left_idx| {
            columns
                .iter()
                .map(|&right_idx| row[left_idx] * variance[left_idx][right_idx] * row[right_idx])
                .sum::<f64>()
        })
        .sum()
}

fn diagnostic_order(strata: &[i32], event_times: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..event_times.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        strata[lhs]
            .cmp(&strata[rhs])
            .then_with(|| event_times[lhs].total_cmp(&event_times[rhs]))
            .then_with(|| lhs.cmp(&rhs))
    });
    order
}

#[pyfunction]
#[pyo3(signature = (time, status, strata=None))]
pub fn cox_event_indices(
    time: Vec<f64>,
    status: Vec<i32>,
    strata: Option<Vec<i32>>,
) -> PyResult<Vec<usize>> {
    let n = time.len();
    if status.len() != n {
        return Err(value_error("status length must match time length"));
    }
    validate_finite_slice(&time, "time")?;
    let strata = strata.as_deref();
    if strata.is_some_and(|values| values.len() != n) {
        return Err(value_error("strata length must match time length"));
    }
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(value_error(format!(
                "status must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    if let Some(values) = strata {
        order.sort_by(|&left, &right| {
            values[left]
                .cmp(&values[right])
                .then_with(|| time[left].total_cmp(&time[right]))
                .then_with(|| left.cmp(&right))
        });
    } else {
        order.sort_by(|&left, &right| {
            time[left]
                .total_cmp(&time[right])
                .then_with(|| left.cmp(&right))
        });
    }
    Ok(order.into_iter().filter(|&idx| status[idx] == 1).collect())
}

#[pyfunction]
pub fn scale_schoenfeld_residuals(
    raw: Vec<Vec<f64>>,
    beta: Vec<f64>,
    information_matrix: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    scale_schoenfeld_residuals_impl(raw, &beta, &information_matrix)
}

fn scale_schoenfeld_residuals_impl(
    raw: Vec<Vec<f64>>,
    beta: &[f64],
    information_matrix: &[Vec<f64>],
) -> PyResult<Vec<Vec<f64>>> {
    let nvar = beta.len();
    if nvar == 0 || raw.is_empty() {
        return Ok(raw);
    }
    validate_finite_slice(beta, "beta")?;
    validate_matrix_width(&raw, nvar, "raw")?;
    validate_square_matrix(information_matrix, nvar, "information_matrix")?;
    let event_count = raw.len() as f64;
    Ok(raw
        .iter()
        .map(|row| {
            (0..nvar)
                .map(|col_idx| {
                    beta[col_idx]
                        + event_count
                            * (0..nvar)
                                .map(|inner_idx| {
                                    row[inner_idx] * information_matrix[inner_idx][col_idx]
                                })
                                .sum::<f64>()
                })
                .collect()
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (score, information_matrix, scaled=false))]
pub fn cox_dfbeta_from_score_residuals(
    score: Vec<Vec<f64>>,
    information_matrix: Vec<Vec<f64>>,
    scaled: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let nvar = information_matrix.len();
    if nvar == 0 {
        return Ok(score);
    }
    validate_square_matrix(&information_matrix, nvar, "information_matrix")?;
    validate_matrix_width(&score, nvar, "score")?;
    let scales: Vec<f64> = if scaled {
        (0..nvar)
            .map(|idx| {
                information_matrix[idx][idx]
                    .abs()
                    .sqrt()
                    .max(crate::constants::DIVISION_FLOOR)
            })
            .collect()
    } else {
        vec![1.0; nvar]
    };

    Ok(score
        .iter()
        .map(|row| {
            (0..nvar)
                .map(|col_idx| {
                    (0..nvar)
                        .map(|inner_idx| information_matrix[col_idx][inner_idx] * row[inner_idx])
                        .sum::<f64>()
                        / scales[col_idx]
                })
                .collect()
        })
        .collect())
}

#[pyfunction]
pub fn cox_zph_term_matrix(
    scaled: Vec<Vec<f64>>,
    groups: Vec<Vec<usize>>,
    beta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let nvar = beta.len();
    validate_finite_slice(&beta, "beta")?;
    validate_matrix_width(&scaled, nvar, "scaled")?;
    validate_column_groups(&groups, nvar)?;
    Ok(scaled
        .iter()
        .map(|row| {
            groups
                .iter()
                .map(|columns| {
                    if columns.len() == 1 {
                        row[columns[0]]
                    } else {
                        columns
                            .iter()
                            .map(|&col_idx| row[col_idx] * beta[col_idx])
                            .sum()
                    }
                })
                .collect()
        })
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cox_zph_tests(
    event_scores: Vec<Vec<f64>>,
    event_information: Vec<Vec<Vec<f64>>>,
    transformed_time: Vec<f64>,
    event_counts: Vec<usize>,
    groups: Vec<Vec<usize>>,
    beta: Vec<f64>,
    single_df: bool,
    global_test: bool,
) -> PyResult<ProportionalityTest> {
    cox_zph_tests_with_penalty(
        event_scores,
        event_information,
        transformed_time,
        event_counts,
        groups,
        beta,
        single_df,
        global_test,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn cox_zph_tests_with_penalty(
    event_scores: Vec<Vec<f64>>,
    event_information: Vec<Vec<Vec<f64>>>,
    transformed_time: Vec<f64>,
    event_counts: Vec<usize>,
    groups: Vec<Vec<usize>>,
    beta: Vec<f64>,
    single_df: bool,
    global_test: bool,
    penalty_matrix: Option<&[Vec<f64>]>,
) -> PyResult<ProportionalityTest> {
    let nvar = beta.len();
    let ntime = event_scores.len();
    if event_information.len() != ntime
        || transformed_time.len() != ntime
        || event_counts.len() != ntime
    {
        return Err(value_error(
            "event_scores, event_information, transformed_time, and event_counts must have the same length",
        ));
    }
    if ntime == 0 {
        return Err(value_error("at least one event time is required"));
    }
    if event_counts.contains(&0) {
        return Err(value_error("event_counts must be positive"));
    }
    validate_finite_slice(&transformed_time, "transformed_time")?;
    validate_finite_slice(&beta, "beta")?;
    validate_matrix_width(&event_scores, nvar, "event_scores")?;
    for (time_idx, information) in event_information.iter().enumerate() {
        validate_square_matrix(information, nvar, &format!("event_information[{time_idx}]"))?;
    }
    validate_column_groups(&groups, nvar)?;
    if let Some(penalty) = penalty_matrix {
        validate_square_matrix(penalty, nvar, "penalty_matrix")?;
    }

    let total_events: usize = event_counts.iter().sum();
    let mean_time = transformed_time
        .iter()
        .zip(&event_counts)
        .map(|(&time, &count)| time * count as f64)
        .sum::<f64>()
        / total_events as f64;
    let centered_time: Vec<f64> = transformed_time
        .iter()
        .map(|&time| time - mean_time)
        .collect();

    let mut time_score = vec![0.0; nvar];
    let mut full_information = vec![vec![0.0; 2 * nvar]; 2 * nvar];
    for ((score, information), &time) in event_scores
        .iter()
        .zip(&event_information)
        .zip(&centered_time)
    {
        for row in 0..nvar {
            time_score[row] += time * score[row];
            for col in 0..nvar {
                let value = information[row][col];
                full_information[row][col] += value;
                full_information[row][nvar + col] += time * value;
                full_information[nvar + row][col] += time * value;
                full_information[nvar + row][nvar + col] += time * time * value;
            }
        }
    }
    if let Some(penalty) = penalty_matrix {
        for row in 0..nvar {
            for col in 0..nvar {
                full_information[row][col] += penalty[row][col];
                full_information[nvar + row][nvar + col] += penalty[row][col];
            }
        }
    }

    let term_system = |columns: &[usize]| -> PyResult<(Array2<f64>, Vec<f64>)> {
        let width = nvar + columns.len();
        let mut matrix = vec![vec![0.0; width]; width];
        for row in 0..nvar {
            matrix[row][..nvar].copy_from_slice(&full_information[row][..nvar]);
            for (group_col, &column) in columns.iter().enumerate() {
                matrix[row][nvar + group_col] = full_information[row][nvar + column];
                matrix[nvar + group_col][row] = full_information[nvar + column][row];
            }
        }
        for (group_row, &row) in columns.iter().enumerate() {
            for (group_col, &col) in columns.iter().enumerate() {
                matrix[nvar + group_row][nvar + group_col] =
                    full_information[nvar + row][nvar + col];
            }
        }

        let flat = matrix.iter().flatten().copied().collect::<Vec<_>>();
        let array = Array2::from_shape_vec((width, width), flat)
            .map_err(|_| value_error("failed to construct Cox zph information matrix"))?;
        let mut score = vec![0.0; width];
        for (group_idx, &column) in columns.iter().enumerate() {
            score[nvar + group_idx] = time_score[column];
        }
        Ok((array, score))
    };

    let score_test = |columns: &[usize]| -> PyResult<f64> {
        let (array, score) = term_system(columns)?;
        let solution = lu_solve(&array, &Array1::from_vec(score.clone()))
            .ok_or_else(|| value_error("Cox zph information matrix is singular"))?;
        let chi2 = score
            .iter()
            .zip(solution.iter())
            .map(|(&left, &right)| left * right)
            .sum::<f64>()
            .max(0.0);
        Ok(chi2)
    };

    let (chi2_values, df_values) = if single_df {
        let values = groups
            .iter()
            .map(|columns| {
                if columns.len() == 1 {
                    return score_test(columns);
                }
                let (array, _score) = term_system(columns)?;
                let inverse = matrix_inverse(&array)
                    .ok_or_else(|| value_error("Cox zph information matrix is singular"))?;
                let term_score = columns
                    .iter()
                    .map(|&column| beta[column] * time_score[column])
                    .sum::<f64>();
                let loading_variance = columns
                    .iter()
                    .enumerate()
                    .map(|(row_idx, &row)| {
                        columns
                            .iter()
                            .enumerate()
                            .map(|(col_idx, &col)| {
                                beta[row] * inverse[[nvar + row_idx, nvar + col_idx]] * beta[col]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                Ok((term_score * term_score * loading_variance).max(0.0))
            })
            .collect::<PyResult<Vec<_>>>()?;
        (values, vec![1; groups.len()])
    } else {
        (
            groups
                .iter()
                .map(|columns| score_test(columns))
                .collect::<PyResult<Vec<_>>>()?,
            groups.iter().map(Vec::len).collect(),
        )
    };
    let p_values = chi2_values
        .iter()
        .zip(&df_values)
        .map(|(&chi2, &df)| chi2_sf(chi2, df))
        .collect();
    let global_chi2 = if global_test {
        let all_columns = (0..nvar).collect::<Vec<_>>();
        score_test(&all_columns)?
    } else {
        0.0
    };

    Ok(ProportionalityTest {
        variable_names: (0..groups.len()).map(|idx| format!("var{idx}")).collect(),
        chi2_values,
        p_values,
        global_chi2,
        global_df: nvar,
        global_p_value: if global_test {
            chi2_sf(global_chi2, nvar)
        } else {
            1.0
        },
    })
}

fn transformed_detail_times(
    detail: &CoxphDetail,
    transformed_events: &[f64],
) -> PyResult<Vec<f64>> {
    validate_finite_slice(transformed_events, "transformed_events")?;
    let mut transformed_time = Vec::with_capacity(detail.rows.len());
    let mut cursor = 0usize;
    for row in &detail.rows {
        let end = cursor
            .checked_add(row.n_event)
            .filter(|&end| end <= transformed_events.len())
            .ok_or_else(|| {
                value_error("Cox detail event counts do not match transformed event times")
            })?;
        let tied = &transformed_events[cursor..end];
        let first = tied.first().copied().ok_or_else(|| {
            value_error("Cox detail event counts do not match transformed event times")
        })?;
        if tied
            .iter()
            .skip(1)
            .any(|&value| (value - first).abs() > TIME_EPSILON)
        {
            return Err(value_error(
                "tied events have inconsistent transformed times",
            ));
        }
        transformed_time.push(first);
        cursor = end;
    }
    if cursor != transformed_events.len() {
        return Err(value_error(
            "Cox detail event counts do not match transformed event times",
        ));
    }
    Ok(transformed_time)
}

fn validate_selected_columns(columns: &[usize], full_width: usize) -> PyResult<()> {
    if columns.is_empty() {
        return Err(value_error("selected Cox detail columns cannot be empty"));
    }
    let mut seen = vec![false; full_width];
    for &column in columns {
        if column >= full_width {
            return Err(value_error(format!(
                "selected Cox detail column {column} must be less than {full_width}"
            )));
        }
        if seen[column] {
            return Err(value_error(format!(
                "selected Cox detail column {column} is duplicated"
            )));
        }
        seen[column] = true;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cox_zph_tests_from_detail(
    detail: CoxphDetail,
    transformed_events: &[f64],
    columns: &[usize],
    groups: Vec<Vec<usize>>,
    beta: Vec<f64>,
    single_df: bool,
    global_test: bool,
    penalty_matrix: Option<Vec<Vec<f64>>>,
) -> PyResult<ProportionalityTest> {
    let full_width = detail.n_covariates;
    if beta.len() != columns.len() {
        return Err(value_error(
            "selected Cox detail columns must match coefficient width",
        ));
    }
    validate_selected_columns(columns, full_width)?;

    let transformed_time = transformed_detail_times(&detail, transformed_events)?;
    let mut event_scores = Vec::with_capacity(detail.rows.len());
    let mut event_information = Vec::with_capacity(detail.rows.len());
    let mut event_counts = Vec::with_capacity(detail.rows.len());
    for row in detail.rows {
        event_scores.push(columns.iter().map(|&column| row.score[column]).collect());
        event_information.push(
            columns
                .iter()
                .map(|&row_idx| {
                    columns
                        .iter()
                        .map(|&col_idx| row.imat[row_idx][col_idx])
                        .collect()
                })
                .collect(),
        );
        event_counts.push(row.n_event);
    }
    cox_zph_tests_with_penalty(
        event_scores,
        event_information,
        transformed_time,
        event_counts,
        groups,
        beta,
        single_df,
        global_test,
        penalty_matrix.as_deref(),
    )
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    covariates,
    coefficients,
    transformed_events,
    groups,
    single_df,
    global_test,
    weights=None,
    entry_times=None,
    strata=None,
    offset=None,
    method="breslow".to_string()
))]
#[allow(clippy::too_many_arguments)]
pub fn cox_zph_tests_from_data(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    transformed_events: Vec<f64>,
    groups: Vec<Vec<usize>>,
    single_df: bool,
    global_test: bool,
    weights: Option<Vec<f64>>,
    entry_times: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    offset: Option<Vec<f64>>,
    method: String,
) -> PyResult<ProportionalityTest> {
    let detail = coxph_detail(
        time,
        status,
        covariates,
        coefficients.clone(),
        weights,
        entry_times,
        strata,
        offset,
        method,
        0.0,
        false,
    )?;
    let columns = (0..coefficients.len()).collect::<Vec<_>>();
    cox_zph_tests_from_detail(
        detail,
        &transformed_events,
        &columns,
        groups,
        coefficients,
        single_df,
        global_test,
        None,
    )
}

#[pyfunction]
pub fn cox_zph_group_variance(
    information_matrix: Vec<Vec<f64>>,
    groups: Vec<Vec<usize>>,
    beta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let nvar = beta.len();
    validate_finite_slice(&beta, "beta")?;
    validate_square_matrix(&information_matrix, nvar, "information_matrix")?;
    validate_column_groups(&groups, nvar)?;
    let mut result = vec![vec![0.0; groups.len()]; groups.len()];
    for (left_idx, left) in groups.iter().enumerate() {
        for (right_idx, right) in groups.iter().enumerate() {
            let mut value = 0.0;
            for &row in left {
                let left_loading = if left.len() > 1 { beta[row] } else { 1.0 };
                for &col in right {
                    let right_loading = if right.len() > 1 { beta[col] } else { 1.0 };
                    value += left_loading * information_matrix[row][col] * right_loading;
                }
            }
            result[left_idx][right_idx] = value;
        }
    }
    Ok(result)
}

#[pyfunction]
pub fn clustered_sandwich_variance(
    rows: Vec<Vec<f64>>,
    weights: Vec<f64>,
    cluster: Vec<usize>,
    variance: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let width = variance.len();
    validate_square_matrix(&variance, width, "variance")?;
    let collapsed = collapse_weighted_rows_by_cluster(&rows, &weights, &cluster, width, "rows")?;
    let meat = row_crossprod(&collapsed, width, "clustered rows")?;
    Ok(sandwich_from_meat(&variance, &meat))
}

#[pyfunction]
#[pyo3(signature = (rows, weights, cluster, width=None))]
pub fn clustered_crossprod(
    rows: Vec<Vec<f64>>,
    weights: Vec<f64>,
    cluster: Vec<usize>,
    width: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let width = width.unwrap_or_else(|| rows.first().map_or(0, Vec::len));
    let collapsed = collapse_weighted_rows_by_cluster(&rows, &weights, &cluster, width, "rows")?;
    row_crossprod(&collapsed, width, "clustered rows")
}

#[pyfunction]
pub fn prediction_se_from_variance(
    rows: Vec<Vec<f64>>,
    variance: Vec<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let width = variance.len();
    validate_square_matrix(&variance, width, "variance")?;
    validate_matrix_width(&rows, width, "rows")?;
    Ok(rows
        .iter()
        .map(|row| quadratic_form(row, &variance).max(0.0).sqrt())
        .collect())
}

#[pyfunction]
pub fn term_prediction_se_from_variance(
    rows: Vec<Vec<f64>>,
    variance: Vec<Vec<f64>>,
    groups: Vec<Vec<usize>>,
) -> PyResult<Vec<Vec<f64>>> {
    let width = variance.len();
    validate_square_matrix(&variance, width, "variance")?;
    validate_matrix_width(&rows, width, "rows")?;
    validate_column_groups(&groups, width)?;
    Ok(rows
        .iter()
        .map(|row| {
            groups
                .iter()
                .map(|columns| {
                    grouped_quadratic_form(row, &variance, columns)
                        .max(0.0)
                        .sqrt()
                })
                .collect()
        })
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cox_interval_cumulative_hazard_se(
    centered_rows: Vec<Vec<f64>>,
    start_hazard: Vec<f64>,
    start_varhaz: Vec<f64>,
    start_xbar: Vec<Vec<f64>>,
    stop_hazard: Vec<f64>,
    stop_varhaz: Vec<f64>,
    stop_xbar: Vec<Vec<f64>>,
    risk: Vec<f64>,
    variance: Vec<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let width = variance.len();
    let n = centered_rows.len();
    validate_square_matrix(&variance, width, "variance")?;
    validate_matrix_width(&centered_rows, width, "centered_rows")?;
    validate_matrix_width(&start_xbar, width, "start_xbar")?;
    validate_matrix_width(&stop_xbar, width, "stop_xbar")?;

    let lengths = [
        ("start_hazard", start_hazard.len()),
        ("start_varhaz", start_varhaz.len()),
        ("start_xbar", start_xbar.len()),
        ("stop_hazard", stop_hazard.len()),
        ("stop_varhaz", stop_varhaz.len()),
        ("stop_xbar", stop_xbar.len()),
        ("risk", risk.len()),
    ];
    for (name, len) in lengths {
        if len != n {
            return Err(value_error(format!(
                "{name} length must match centered_rows length"
            )));
        }
    }
    validate_finite_slice(&start_hazard, "start_hazard")?;
    validate_finite_slice(&start_varhaz, "start_varhaz")?;
    validate_finite_slice(&stop_hazard, "stop_hazard")?;
    validate_finite_slice(&stop_varhaz, "stop_varhaz")?;
    validate_finite_slice(&risk, "risk")?;

    Ok((0..n)
        .map(|row_idx| {
            let hazard_delta = stop_hazard[row_idx] - start_hazard[row_idx];
            let interval_delta: Vec<f64> = (0..width)
                .map(|col_idx| {
                    hazard_delta * centered_rows[row_idx][col_idx]
                        - (stop_xbar[row_idx][col_idx] - start_xbar[row_idx][col_idx])
                })
                .collect();
            let variance_value = stop_varhaz[row_idx] - start_varhaz[row_idx]
                + quadratic_form(&interval_delta, &variance);
            variance_value.max(0.0).sqrt() * risk[row_idx]
        })
        .collect())
}

impl CoxPHFit {
    fn counting_process_expected_events(
        &self,
        entry_times: &[f64],
        method: i32,
    ) -> PyResult<Vec<f64>> {
        let n = self.event_times.len();
        if entry_times.len() != n
            || self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(value_error(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&lhs, &rhs| {
            self.strata[lhs]
                .cmp(&self.strata[rhs])
                .then_with(|| self.event_times[lhs].total_cmp(&self.event_times[rhs]))
                .then_with(|| self.status[rhs].cmp(&self.status[lhs]))
                .then_with(|| lhs.cmp(&rhs))
        });
        let start: Vec<f64> = order.iter().map(|&idx| entry_times[idx]).collect();
        let stop: Vec<f64> = order.iter().map(|&idx| self.event_times[idx]).collect();
        let event: Vec<i32> = order.iter().map(|&idx| self.status[idx]).collect();
        let score: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
            })
            .collect();
        let weights: Vec<f64> = order.iter().map(|&idx| self.weights[idx]).collect();
        let mut strata = vec![0; n];
        for sorted_idx in 0..n {
            if sorted_idx + 1 == n
                || self.strata[order[sorted_idx + 1]] != self.strata[order[sorted_idx]]
            {
                strata[sorted_idx] = 1;
            }
        }
        let residuals = compute_agmart(
            method,
            AgmartData {
                start: &start,
                stop: &stop,
                event: &event,
                score: &score,
                wt: &weights,
                strata: &strata,
            },
        );

        let mut expected = vec![0.0; n];
        for (sorted_idx, &original_idx) in order.iter().enumerate() {
            expected[original_idx] = self.status[original_idx] as f64 - residuals[sorted_idx];
        }
        Ok(expected)
    }

    fn right_censored_expected_events(&self, method: i32) -> PyResult<Vec<f64>> {
        let n = self.event_times.len();
        if self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(value_error(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }

        let order = diagnostic_order(&self.strata, &self.event_times);
        let time: Vec<f64> = order.iter().map(|&idx| self.event_times[idx]).collect();
        let status: Vec<i32> = order.iter().map(|&idx| self.status[idx]).collect();
        let score: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
            })
            .collect();
        let weights: Vec<f64> = order.iter().map(|&idx| self.weights[idx]).collect();
        let mut strata = vec![0; n];
        for sorted_idx in 0..n {
            if sorted_idx + 1 == n
                || self.strata[order[sorted_idx + 1]] != self.strata[order[sorted_idx]]
            {
                strata[sorted_idx] = 1;
            }
        }
        let mut residuals = vec![0.0; n];
        compute_coxmart(
            n,
            method,
            CoxMartSurvivalData {
                time: &time,
                status: &status,
                strata: &strata,
            },
            CoxMartWeights {
                score: &score,
                wt: &weights,
            },
            &mut residuals,
        );

        let mut expected = vec![0.0; n];
        for (sorted_idx, &original_idx) in order.iter().enumerate() {
            expected[original_idx] = self.status[original_idx] as f64 - residuals[sorted_idx];
        }
        Ok(expected)
    }

    pub(crate) fn expected_events_internal(&self) -> PyResult<Vec<f64>> {
        match (self.entry_times.as_deref(), self.tie_method()) {
            (None, CoxMethod::Breslow) => return self.right_censored_expected_events(0),
            (None, CoxMethod::Efron) => return self.right_censored_expected_events(1),
            (Some(entry_times), CoxMethod::Breslow) => {
                return self.counting_process_expected_events(entry_times, 0);
            }
            (Some(entry_times), CoxMethod::Efron) => {
                return self.counting_process_expected_events(entry_times, 1);
            }
            (_, CoxMethod::Exact) => {}
        }

        let (times, hazards, hazard_strata) = self.basehaz_with_strata_internal(false)?;
        let baseline = StratifiedBaselineLookup::from_components(&times, &hazards, &hazard_strata);
        let entry_times = self.entry_times.as_deref();
        let row_strata = self.row_strata_cow();

        Ok(self
            .event_times
            .iter()
            .enumerate()
            .map(|(idx, &stop)| {
                let start_hazard = entry_times
                    .map(|starts| baseline.cumulative_hazard_at(row_strata[idx], starts[idx]))
                    .unwrap_or(0.0);
                let stop_hazard = baseline.cumulative_hazard_at(row_strata[idx], stop);
                let interval_hazard = (stop_hazard - start_hazard).max(0.0);
                let risk_multiplier = self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                interval_hazard * risk_multiplier
            })
            .collect())
    }

    pub(crate) fn tie_method(&self) -> CoxMethod {
        match self.method.as_str() {
            "exact" => CoxMethod::Exact,
            "efron" => CoxMethod::Efron,
            _ => CoxMethod::Breslow,
        }
    }

    pub(crate) fn score_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let n = self.event_times.len();
        if nvar == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let tie_method = self.tie_method();
        let method = match tie_method {
            CoxMethod::Breslow => 0,
            CoxMethod::Efron => 1,
            CoxMethod::Exact => 2,
        };
        if self.covariates.len() != n
            || self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }
        if self.entry_times.is_some() {
            return self.score_residuals_counting_process(nvar, method);
        }

        let order = diagnostic_order(&self.strata, &self.event_times);

        let mut y = Vec::with_capacity(2 * n);
        y.extend(order.iter().map(|&idx| self.event_times[idx]));
        y.extend(order.iter().map(|&idx| self.status[idx] as f64));
        let strata: Vec<i32> = order.iter().map(|&idx| self.strata[idx]).collect();
        let weights: Vec<f64> = order.iter().map(|&idx| self.weights[idx]).collect();
        if matches!(tie_method, CoxMethod::Exact) {
            let log_risk: Vec<f64> = order
                .iter()
                .map(|&idx| self.linear_predictors[idx] + self.weights[idx].ln())
                .collect();
            return Ok(self.score_residuals_exact_right_censored(nvar, &order, &log_risk));
        }
        let score: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
            })
            .collect();
        let mut covar = Vec::with_capacity(n * nvar);
        for &idx in &order {
            covar.extend(self.covariates[idx].iter().copied());
        }
        let flat = compute_cox_score_residuals(
            CoxScoreData {
                y: &y,
                strata: &strata,
                covar: &covar,
                score: &score,
                weights: &weights,
            },
            CoxScoreParams { method, n, nvar },
        );
        let mut residuals = vec![vec![0.0; nvar]; n];
        for (sorted_idx, &original_idx) in order.iter().enumerate() {
            for col_idx in 0..nvar {
                residuals[original_idx][col_idx] = flat[sorted_idx * nvar + col_idx];
            }
        }
        Ok(residuals)
    }

    fn score_residuals_exact_right_censored(
        &self,
        nvar: usize,
        order: &[usize],
        log_risk: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }
            let mut risk_indices: Vec<usize> = Vec::new();
            let mut deaths: Vec<usize> = Vec::new();
            let mut time_pos = stratum_end;
            loop {
                let event_time = self.event_times[order[time_pos]];
                let mut time_start = time_pos;
                while time_start > stratum_start
                    && same_time(self.event_times[order[time_start - 1]], event_time)
                {
                    time_start -= 1;
                }
                for sorted_idx in time_start..=time_pos {
                    risk_indices.push(sorted_idx);
                }
                deaths.clear();
                deaths.extend((time_start..=time_pos).filter(|&idx| self.status[order[idx]] == 1));
                if !deaths.is_empty() {
                    for &sorted_idx in &deaths {
                        let original_idx = order[sorted_idx];
                        let weight = self.weights[original_idx];
                        for (col_idx, residual) in
                            residuals[original_idx].iter_mut().enumerate().take(nvar)
                        {
                            *residual += weight * self.covariates[original_idx][col_idx];
                        }
                    }
                    if let Some(inclusion_weights) =
                        exact_inclusion_probabilities(&risk_indices, deaths.len(), log_risk)
                    {
                        for (sorted_idx, inclusion_weight) in inclusion_weights {
                            let original_idx = order[sorted_idx];
                            for (col_idx, residual) in
                                residuals[original_idx].iter_mut().enumerate().take(nvar)
                            {
                                *residual -=
                                    inclusion_weight * self.covariates[original_idx][col_idx];
                            }
                        }
                    }
                }
                if time_start == stratum_start {
                    break;
                }
                time_pos = time_start - 1;
            }
            stratum_start = stratum_end + 1;
        }
        residuals
    }

    fn score_residuals_counting_process(
        &self,
        nvar: usize,
        method: i32,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n = self.event_times.len();
        let Some(entry_times) = self.entry_times.as_ref() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "counting-process score residuals require entry times",
            ));
        };
        if entry_times.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model entry times have inconsistent length",
            ));
        }

        let order = diagnostic_order(&self.strata, &self.event_times);

        if method == 2 {
            let log_risk: Vec<f64> = (0..n)
                .map(|idx| self.linear_predictors[idx] + self.weights[idx].ln())
                .collect();
            return Ok(self.score_residuals_counting_process_by_scan(
                nvar,
                method,
                &log_risk,
                &order,
                entry_times,
            ));
        }

        let risk: Vec<f64> = (0..n)
            .map(|idx| self.linear_predictors[idx].exp() * self.weights[idx])
            .collect();
        Ok(self.score_residuals_counting_process_sweep(nvar, method, &risk, &order, entry_times))
    }

    pub(crate) fn score_residuals_counting_process_by_scan(
        &self,
        nvar: usize,
        method: i32,
        risk: &[f64],
        order: &[usize],
        entry_times: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let scores = (method != 2).then(|| {
            self.linear_predictors
                .iter()
                .map(|&value| value.exp())
                .collect::<Vec<_>>()
        });
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }
            let stratum_indices = &order[stratum_start..=stratum_end];
            let mut deaths: Vec<usize> = Vec::new();
            let mut risk_indices: Vec<usize> = Vec::new();
            let mut is_death = vec![false; n];
            let mut time_start = stratum_start;
            while time_start <= stratum_end {
                let event_time = self.event_times[order[time_start]];
                let mut time_end = time_start;
                while time_end < stratum_end
                    && same_time(self.event_times[order[time_end + 1]], event_time)
                {
                    time_end += 1;
                }

                deaths.clear();
                deaths.extend(
                    (time_start..=time_end)
                        .map(|idx| order[idx])
                        .filter(|&idx| self.status[idx] == 1),
                );
                if !deaths.is_empty() {
                    for &idx in &deaths {
                        is_death[idx] = true;
                    }
                    risk_indices.clear();
                    risk_indices.extend(stratum_indices.iter().copied().filter(|&idx| {
                        entry_times[idx] < event_time && self.event_times[idx] >= event_time
                    }));
                    if method == 2 {
                        for &idx in &deaths {
                            for (col_idx, residual) in
                                residuals[idx].iter_mut().enumerate().take(nvar)
                            {
                                *residual += self.weights[idx] * self.covariates[idx][col_idx];
                            }
                        }
                        if let Some(inclusion_weights) =
                            exact_inclusion_probabilities(&risk_indices, deaths.len(), risk)
                        {
                            for (idx, inclusion_weight) in inclusion_weights {
                                for (col_idx, residual) in
                                    residuals[idx].iter_mut().enumerate().take(nvar)
                                {
                                    *residual -= inclusion_weight * self.covariates[idx][col_idx];
                                }
                            }
                        }
                    } else {
                        let denom: f64 = risk_indices.iter().map(|&idx| risk[idx]).sum();
                        if denom > 0.0 {
                            let scores = scores
                                .as_ref()
                                .expect("non-exact score residuals have risk scores");
                            if method == 0 || deaths.len() == 1 {
                                let deadwt: f64 = deaths.iter().map(|&idx| self.weights[idx]).sum();
                                let hazard = deadwt / denom;
                                let mut mean = vec![0.0; nvar];
                                for &idx in &risk_indices {
                                    for (col_idx, value) in mean.iter_mut().enumerate() {
                                        *value += risk[idx] * self.covariates[idx][col_idx];
                                    }
                                }
                                for value in &mut mean {
                                    *value /= denom;
                                }
                                for &idx in &risk_indices {
                                    let score = scores[idx];
                                    for (col_idx, residual) in residuals[idx].iter_mut().enumerate()
                                    {
                                        *residual += score
                                            * hazard
                                            * (mean[col_idx] - self.covariates[idx][col_idx]);
                                    }
                                }
                                for &idx in &deaths {
                                    for (col_idx, residual) in residuals[idx].iter_mut().enumerate()
                                    {
                                        *residual += self.covariates[idx][col_idx] - mean[col_idx];
                                    }
                                }
                            } else {
                                let death_count = deaths.len();
                                let deaths_f = death_count as f64;
                                let deadwt: f64 = deaths.iter().map(|&idx| self.weights[idx]).sum();
                                let weight_average = deadwt / deaths_f;
                                let death_risk: f64 = deaths.iter().map(|&idx| risk[idx]).sum();
                                let mut risk_sum = vec![0.0; nvar];
                                let mut death_risk_sum = vec![0.0; nvar];
                                for &idx in &risk_indices {
                                    for (col_idx, value) in risk_sum.iter_mut().enumerate() {
                                        *value += risk[idx] * self.covariates[idx][col_idx];
                                    }
                                }
                                for &idx in &deaths {
                                    for (col_idx, value) in death_risk_sum.iter_mut().enumerate() {
                                        *value += risk[idx] * self.covariates[idx][col_idx];
                                    }
                                }
                                for step in 0..death_count {
                                    let fraction = step as f64 / deaths_f;
                                    let step_denom = denom - fraction * death_risk;
                                    if step_denom <= 0.0 {
                                        continue;
                                    }
                                    let hazard = weight_average / step_denom;
                                    let mean: Vec<f64> = risk_sum
                                        .iter()
                                        .zip(&death_risk_sum)
                                        .map(|(&total, &death_total)| {
                                            (total - fraction * death_total) / step_denom
                                        })
                                        .collect();
                                    for &idx in &risk_indices {
                                        let score = scores[idx];
                                        let multiplier =
                                            if is_death[idx] { 1.0 - fraction } else { 1.0 };
                                        for (col_idx, residual) in
                                            residuals[idx].iter_mut().enumerate()
                                        {
                                            *residual += score
                                                * hazard
                                                * multiplier
                                                * (mean[col_idx] - self.covariates[idx][col_idx]);
                                        }
                                    }
                                    for &idx in &deaths {
                                        for (col_idx, residual) in
                                            residuals[idx].iter_mut().enumerate()
                                        {
                                            *residual += (self.covariates[idx][col_idx]
                                                - mean[col_idx])
                                                / deaths_f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for &idx in &deaths {
                        is_death[idx] = false;
                    }
                }

                if time_end == stratum_end {
                    break;
                }
                time_start = time_end + 1;
            }
            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn score_residuals_counting_process_sweep(
        &self,
        nvar: usize,
        method: i32,
        risk: &[f64],
        order: &[usize],
        entry_times: &[f64],
    ) -> Vec<Vec<f64>> {
        // The reference counting sweep is compiled with multiply-add
        // contraction for both scalar and multivariate fits.
        let contracted = self.counting_roundoff_compatibility;
        let arithmetic = ProductAccumulator::new(contracted);
        let n = self.event_times.len();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let scores: Vec<f64> = self
            .linear_predictors
            .iter()
            .map(|&value| value.exp())
            .collect();
        let mut score_order = order.to_vec();
        score_order.sort_by(|&lhs, &rhs| {
            self.strata[lhs]
                .cmp(&self.strata[rhs])
                .then_with(|| self.event_times[lhs].total_cmp(&self.event_times[rhs]))
                .then_with(|| self.status[rhs].cmp(&self.status[lhs]))
                .then_with(|| lhs.cmp(&rhs))
        });
        let mut stratum_start = 0usize;
        while stratum_start < score_order.len() {
            let stratum = self.strata[score_order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < score_order.len()
                && self.strata[score_order[stratum_end + 1]] == stratum
            {
                stratum_end += 1;
            }
            let stratum_order = &score_order[stratum_start..=stratum_end];
            let rows: Vec<CoxSweepRow> = stratum_order
                .iter()
                .map(|&idx| CoxSweepRow {
                    original_idx: idx,
                    stop: self.event_times[idx],
                    entry: entry_times[idx],
                    risk: risk[idx],
                    weight: self.weights[idx],
                    status: self.status[idx],
                })
                .collect();
            let mut entry_order = (0..rows.len()).collect::<Vec<_>>();
            entry_order.sort_by(|&lhs, &rhs| {
                rows[lhs]
                    .entry
                    .total_cmp(&rows[rhs].entry)
                    .then_with(|| lhs.cmp(&rhs))
            });
            let mut entry_pos = entry_order.len();
            let mut person = rows.len();
            let mut denom = 0.0;
            let mut cumulative_hazard = 0.0_f64;
            let mut risk_covariates = vec![0.0; nvar];
            let mut cumulative_xhazard = vec![0.0; nvar];
            let mut death_covariates = vec![0.0; nvar];
            let mut mean = vec![0.0; nvar];
            let mut hazard_fraction = vec![0.0; nvar];
            let mut mean_hazard_fraction = vec![0.0; nvar];
            let mut mean_sum = vec![0.0; nvar];
            let mut deaths = Vec::new();

            while person > 0 {
                let event_time = rows[person - 1].stop;
                while entry_pos > 0 && rows[entry_order[entry_pos - 1]].entry >= event_time {
                    entry_pos -= 1;
                    let row_idx = entry_order[entry_pos];
                    let row = rows[row_idx];
                    let original_idx = row.original_idx;
                    denom -= row.risk;
                    for col_idx in 0..nvar {
                        let covariate = self.covariates[original_idx][col_idx];
                        let hazard_difference =
                            cumulative_hazard.mul_add(covariate, -cumulative_xhazard[col_idx]);
                        residuals[original_idx][col_idx] = (-scores[original_idx])
                            .mul_add(hazard_difference, residuals[original_idx][col_idx]);
                        risk_covariates[col_idx] =
                            arithmetic.subtract(risk_covariates[col_idx], row.risk, covariate);
                    }
                }

                let mut time_start = person - 1;
                while time_start > 0 && same_time(rows[time_start - 1].stop, event_time) {
                    time_start -= 1;
                }
                let mut death_risk = 0.0;
                let mut death_weight = 0.0;
                death_covariates.fill(0.0);
                deaths.clear();
                for row_idx in (time_start..person).rev() {
                    let row = rows[row_idx];
                    let original_idx = row.original_idx;
                    for col_idx in 0..nvar {
                        let covariate = self.covariates[original_idx][col_idx];
                        residuals[original_idx][col_idx] = scores[original_idx]
                            * covariate.mul_add(cumulative_hazard, -cumulative_xhazard[col_idx]);
                        risk_covariates[col_idx] =
                            arithmetic.add(risk_covariates[col_idx], row.risk, covariate);
                    }
                    denom += row.risk;
                    if row.status == 1 {
                        deaths.push(row_idx);
                        death_risk += row.risk;
                        death_weight += row.weight;
                        for (col_idx, value) in death_covariates.iter_mut().enumerate() {
                            *value = arithmetic.add(
                                *value,
                                row.risk,
                                self.covariates[original_idx][col_idx],
                            );
                        }
                    }
                }

                if !deaths.is_empty() {
                    if method == 0 || deaths.len() == 1 {
                        let hazard = death_weight / denom;
                        cumulative_hazard += hazard;
                        for col_idx in 0..nvar {
                            mean[col_idx] = risk_covariates[col_idx] / denom;
                            cumulative_xhazard[col_idx] =
                                arithmetic.add(cumulative_xhazard[col_idx], mean[col_idx], hazard);
                        }
                        for &row_idx in &deaths {
                            let original_idx = rows[row_idx].original_idx;
                            for col_idx in 0..nvar {
                                residuals[original_idx][col_idx] +=
                                    self.covariates[original_idx][col_idx] - mean[col_idx];
                            }
                        }
                    } else {
                        hazard_fraction.fill(0.0);
                        mean_hazard_fraction.fill(0.0);
                        mean_sum.fill(0.0);
                        let death_count = deaths.len() as f64;
                        let mean_death_weight = death_weight / death_count;
                        for step in 0..deaths.len() {
                            let fraction = step as f64 / death_count;
                            let step_denom = arithmetic.subtract(denom, fraction, death_risk);
                            let hazard = mean_death_weight / step_denom;
                            cumulative_hazard += hazard;
                            for col_idx in 0..nvar {
                                mean[col_idx] = arithmetic.subtract(
                                    risk_covariates[col_idx],
                                    fraction,
                                    death_covariates[col_idx],
                                ) / step_denom;
                                cumulative_xhazard[col_idx] = arithmetic.add(
                                    cumulative_xhazard[col_idx],
                                    mean[col_idx],
                                    hazard,
                                );
                                hazard_fraction[col_idx] =
                                    arithmetic.add(hazard_fraction[col_idx], hazard, fraction);
                                mean_hazard_fraction[col_idx] = arithmetic.add(
                                    mean_hazard_fraction[col_idx],
                                    mean[col_idx] * hazard,
                                    fraction,
                                );
                                mean_sum[col_idx] += mean[col_idx] / death_count;
                            }
                        }
                        for &row_idx in &deaths {
                            let original_idx = rows[row_idx].original_idx;
                            for col_idx in 0..nvar {
                                let covariate = self.covariates[original_idx][col_idx];
                                let correction = covariate.mul_add(
                                    hazard_fraction[col_idx],
                                    -mean_hazard_fraction[col_idx],
                                );
                                if contracted {
                                    let death_increment = scores[original_idx]
                                        .mul_add(correction, covariate - mean_sum[col_idx]);
                                    residuals[original_idx][col_idx] += death_increment;
                                } else {
                                    let death_correction = residuals[original_idx][col_idx]
                                        + covariate
                                        - mean_sum[col_idx];
                                    residuals[original_idx][col_idx] =
                                        scores[original_idx].mul_add(correction, death_correction);
                                }
                            }
                        }
                    }
                }
                person = time_start;
            }

            while entry_pos > 0 {
                entry_pos -= 1;
                let row = rows[entry_order[entry_pos]];
                let original_idx = row.original_idx;
                for col_idx in 0..nvar {
                    let hazard_difference = self.covariates[original_idx][col_idx]
                        .mul_add(cumulative_hazard, -cumulative_xhazard[col_idx]);
                    residuals[original_idx][col_idx] = (-scores[original_idx])
                        .mul_add(hazard_difference, residuals[original_idx][col_idx]);
                }
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn dfbeta_from_score_residuals(&self, scaled: bool) -> PyResult<Vec<Vec<f64>>> {
        let score_residuals = self.score_residuals_internal()?;
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        if nvar == 0 {
            return Ok(score_residuals);
        }
        if self.information_matrix.len() != nvar
            || self.information_matrix.iter().any(|row| row.len() != nvar)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model information matrix does not match coefficient width",
            ));
        }
        let scale: Vec<f64> = if scaled {
            (0..nvar)
                .map(|idx| {
                    self.information_matrix[idx][idx]
                        .abs()
                        .sqrt()
                        .max(crate::constants::DIVISION_FLOOR)
                })
                .collect()
        } else {
            vec![1.0; nvar]
        };
        let arithmetic = ProductAccumulator::new(self.counting_roundoff_compatibility && nvar > 1);
        Ok(score_residuals
            .iter()
            .map(|row| {
                (0..nvar)
                    .map(|col_idx| {
                        let value = (0..nvar).fold(0.0, |sum, inner_idx| {
                            arithmetic.add(
                                sum,
                                self.information_matrix[col_idx][inner_idx],
                                row[inner_idx],
                            )
                        });
                        value / scale[col_idx]
                    })
                    .collect()
            })
            .collect())
    }

    pub(crate) fn scaled_schoenfeld_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        self.scaled_schoenfeld_residuals_with_variance_internal(&self.information_matrix)
    }

    pub(crate) fn scaled_schoenfeld_residuals_with_variance_internal(
        &self,
        information_matrix: &[Vec<f64>],
    ) -> PyResult<Vec<Vec<f64>>> {
        let schoenfeld = self.schoenfeld_residuals_internal()?;
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        scale_schoenfeld_residuals_impl(schoenfeld, beta, information_matrix)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cox_zph_diagnostics_internal(
        &self,
        transformed_events: Vec<f64>,
        active_columns: Vec<usize>,
        groups: Vec<Vec<usize>>,
        information_matrix: Vec<Vec<f64>>,
        single_df: bool,
        global_test: bool,
        penalty_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<(Vec<Vec<f64>>, ProportionalityTest)> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        validate_selected_columns(&active_columns, nvar)?;
        validate_column_groups(&groups, active_columns.len())?;
        validate_square_matrix(&information_matrix, nvar, "information_matrix")?;
        if let Some(penalty) = penalty_matrix.as_ref() {
            validate_square_matrix(penalty, nvar, "penalty_matrix")?;
        }

        let scaled_full =
            self.scaled_schoenfeld_residuals_with_variance_internal(&information_matrix)?;
        let scaled = scaled_full
            .into_iter()
            .map(|row| active_columns.iter().map(|&column| row[column]).collect())
            .collect::<Vec<Vec<f64>>>();

        let n = self.event_times.len();
        if self.status.len() != n
            || self.covariates.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(value_error(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        validate_matrix_width(&self.covariates, nvar, "covariates")?;
        let method = if matches!(self.tie_method(), CoxMethod::Efron) {
            "efron"
        } else {
            "breslow"
        };
        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &self.event_times,
            status: &self.status,
            covariates: &self.covariates,
            coefficients: beta,
            weights: Some(&self.weights),
            entry_times: self.entry_times.as_deref(),
            strata: Some(&self.strata),
            offset: None,
            linear_predictors: Some(&self.linear_predictors),
            method,
            center: 0.0,
            include_riskmat: false,
        })?;
        let active_beta: Vec<f64> = active_columns.iter().map(|&column| beta[column]).collect();
        let active_penalty = penalty_matrix.map(|penalty| {
            active_columns
                .iter()
                .map(|&row| {
                    active_columns
                        .iter()
                        .map(|&column| penalty[row][column])
                        .collect()
                })
                .collect()
        });
        let grouped = cox_zph_term_matrix(scaled, groups.clone(), active_beta.clone())?;
        let test = cox_zph_tests_from_detail(
            detail,
            &transformed_events,
            &active_columns,
            groups,
            active_beta,
            single_df,
            global_test,
            active_penalty,
        )?;
        Ok((grouped, test))
    }

    pub(crate) fn cox_zph_diagnostics_with_surface_internal(
        &self,
        transformed_events: Vec<f64>,
        active_columns: Vec<usize>,
        groups: Vec<Vec<usize>>,
        single_df: bool,
        global_test: bool,
    ) -> PyResult<CoxZphSurfaceDiagnostics> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        validate_selected_columns(&active_columns, nvar)?;
        validate_column_groups(&groups, active_columns.len())?;

        let n = self.event_times.len();
        if self.status.len() != n
            || self.covariates.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(value_error(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        validate_matrix_width(&self.covariates, nvar, "covariates")?;

        let method = if matches!(self.tie_method(), CoxMethod::Efron) {
            "efron"
        } else {
            "breslow"
        };
        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &self.event_times,
            status: &self.status,
            covariates: &self.covariates,
            coefficients: beta,
            weights: Some(&self.weights),
            entry_times: self.entry_times.as_deref(),
            strata: Some(&self.strata),
            offset: None,
            linear_predictors: Some(&self.linear_predictors),
            method,
            center: 0.0,
            include_riskmat: false,
        })?;
        let active_beta = active_columns
            .iter()
            .map(|&column| beta[column])
            .collect::<Vec<_>>();
        let raw_full = self.schoenfeld_residuals_internal()?;
        let (surface, variance) = self.cox_zph_residual_surface(
            raw_full,
            &detail,
            &active_columns,
            &groups,
            &active_beta,
        )?;
        let test = cox_zph_tests_from_detail(
            detail,
            &transformed_events,
            &active_columns,
            groups,
            active_beta,
            single_df,
            global_test,
            None,
        )?;
        Ok((surface, variance, test))
    }

    fn cox_zph_residual_surface(
        &self,
        raw_full: Vec<Vec<f64>>,
        detail: &CoxphDetail,
        active_columns: &[usize],
        groups: &[Vec<usize>],
        active_beta: &[f64],
    ) -> PyResult<CoxZphSurface> {
        let nvar = self.coefficients.first().map_or(0, Vec::len);
        validate_matrix_width(&raw_full, nvar, "Schoenfeld residuals")?;
        let active_width = active_columns.len();
        let raw = raw_full
            .into_iter()
            .map(|row| active_columns.iter().map(|&column| row[column]).collect())
            .collect::<Vec<Vec<f64>>>();

        let mut strata_levels = self.strata.clone();
        strata_levels.sort_unstable();
        strata_levels.dedup();
        let mut strata_rows = vec![Vec::new(); strata_levels.len()];
        for (row, &stratum) in self.strata.iter().enumerate() {
            let stratum_index = strata_levels
                .binary_search(&stratum)
                .expect("fitted stratum must be present in its level set");
            strata_rows[stratum_index].push(row);
        }

        let mut used = vec![vec![0usize; active_width]; strata_levels.len()];
        for (stratum_index, rows) in strata_rows.iter().enumerate() {
            let event_count = rows.iter().filter(|&&row| self.status[row] == 1).count();
            if rows.is_empty() || event_count == 0 {
                continue;
            }
            for (dense_column, &column) in active_columns.iter().enumerate() {
                let first = self.covariates[rows[0]][column];
                if rows
                    .iter()
                    .skip(1)
                    .any(|&row| self.covariates[row][column] != first)
                {
                    used[stratum_index][dense_column] = event_count;
                }
            }
        }

        for columns in groups {
            if columns.len() > 1
                && used
                    .iter()
                    .any(|row| columns.iter().any(|&column| row[column] == 0))
            {
                for row in &mut used {
                    let maximum = columns.iter().map(|&column| row[column]).max().unwrap_or(0);
                    for &column in columns {
                        row[column] = maximum;
                    }
                }
            }
        }

        let mut information = vec![vec![0.0; active_width]; active_width];
        for detail_row in &detail.rows {
            validate_square_matrix(&detail_row.imat, nvar, "Cox detail information matrix")?;
            for (dense_row, &row) in active_columns.iter().enumerate() {
                for (dense_column, &column) in active_columns.iter().enumerate() {
                    information[dense_row][dense_column] += detail_row.imat[row][column];
                }
            }
        }

        let mut weight = vec![vec![0.0; active_width]; active_width];
        for stratum_used in &used {
            for row in 0..active_width {
                for column in 0..active_width {
                    weight[row][column] += stratum_used[row].min(stratum_used[column]) as f64;
                }
            }
        }
        let mean_information = (0..active_width)
            .map(|row| {
                (0..active_width)
                    .map(|column| {
                        information[row][column]
                            / if weight[row][column] == 0.0 {
                                1.0
                            } else {
                                weight[row][column]
                            }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let group_count = groups.len();
        let mut loadings = vec![vec![0.0; group_count]; active_width];
        for (group_index, columns) in groups.iter().enumerate() {
            if columns.len() == 1 {
                loadings[columns[0]][group_index] = 1.0;
            } else {
                for &column in columns {
                    loadings[column][group_index] = active_beta[column];
                }
            }
        }
        let grouped_raw = cox_zph_term_matrix(raw, groups.to_vec(), active_beta.to_vec())?;
        let grouped_mean_information = (0..group_count)
            .map(|left_group| {
                (0..group_count)
                    .map(|right_group| {
                        (0..active_width)
                            .map(|row| {
                                (0..active_width)
                                    .map(|column| {
                                        loadings[row][left_group]
                                            * mean_information[row][column]
                                            * loadings[column][right_group]
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let grouped_used = used
            .iter()
            .map(|row| {
                groups
                    .iter()
                    .map(|columns| row[columns[0]])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let event_indices = diagnostic_order(&self.strata, &self.event_times)
            .into_iter()
            .filter(|&row| self.status[row] == 1)
            .collect::<Vec<_>>();
        if event_indices.len() != grouped_raw.len() {
            return Err(value_error(
                "fitted Cox model event order does not match Schoenfeld residuals",
            ));
        }
        let mut surface = vec![vec![f64::NAN; group_count]; grouped_raw.len()];
        for (stratum_index, &stratum) in strata_levels.iter().enumerate() {
            let active_groups = grouped_used[stratum_index]
                .iter()
                .enumerate()
                .filter_map(|(group, &count)| (count > 0).then_some(group))
                .collect::<Vec<_>>();
            if active_groups.is_empty() {
                continue;
            }
            let stratum_mean = active_groups
                .iter()
                .map(|&row| {
                    active_groups
                        .iter()
                        .map(|&column| grouped_mean_information[row][column])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let inverse = invert_square_rows(&stratum_mean, "Cox zph stratum information matrix")?;
            for (event_row, &source_row) in event_indices.iter().enumerate() {
                if self.strata[source_row] != stratum {
                    continue;
                }
                for (output_column, &group) in active_groups.iter().enumerate() {
                    surface[event_row][group] = active_groups
                        .iter()
                        .enumerate()
                        .map(|(input_column, &input_group)| {
                            grouped_raw[event_row][input_group]
                                * inverse[input_column][output_column]
                        })
                        .sum();
                }
            }
        }
        for row in &mut surface {
            for (group, columns) in groups.iter().enumerate() {
                row[group] += if columns.len() == 1 {
                    active_beta[columns[0]]
                } else {
                    1.0
                };
            }
        }
        let variance = invert_square_rows(
            &grouped_mean_information,
            "Cox zph grouped information matrix",
        )?;
        Ok((surface, variance))
    }

    pub(crate) fn schoenfeld_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let n = self.event_times.len();
        if nvar == 0 {
            return Ok(Vec::new());
        }
        if self.covariates.len() != n
            || self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }
        let entry_times = self.entry_times.as_ref();
        if let Some(values) = entry_times
            && values.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model entry times have inconsistent length",
            ));
        }

        let order = diagnostic_order(&self.strata, &self.event_times);

        let method = self.tie_method();
        if matches!(method, CoxMethod::Exact) {
            Ok(self.schoenfeld_residuals_by_scan(nvar, &order, entry_times, method))
        } else {
            Ok(self.schoenfeld_residuals_sweep(nvar, &order, entry_times, method))
        }
    }

    pub(crate) fn schoenfeld_residuals_by_scan(
        &self,
        nvar: usize,
        order: &[usize],
        entry_times: Option<&Vec<f64>>,
        method: CoxMethod,
    ) -> Vec<Vec<f64>> {
        let n = order.len();
        let sorted_time: Vec<f64> = order.iter().map(|&idx| self.event_times[idx]).collect();
        let sorted_start: Vec<f64> = order
            .iter()
            .map(|&idx| entry_times.map(|values| values[idx]).unwrap_or(0.0))
            .collect();
        let sorted_status: Vec<i32> = order.iter().map(|&idx| self.status[idx]).collect();
        let sorted_strata: Vec<i32> = order.iter().map(|&idx| self.strata[idx]).collect();
        let sorted_risk: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
                    * self.weights[idx]
            })
            .collect();
        let sorted_log_risk: Vec<f64> = order
            .iter()
            .map(|&idx| self.linear_predictors[idx] + self.weights[idx].ln())
            .collect();
        let mut covar = Array2::<f64>::zeros((n, nvar));
        for (row_idx, &source_idx) in order.iter().enumerate() {
            for col_idx in 0..nvar {
                covar[(row_idx, col_idx)] = self.covariates[source_idx][col_idx];
            }
        }

        let event_count = sorted_status.iter().filter(|&&status| status == 1).count();
        let mut residuals = Vec::with_capacity(event_count);
        let mut stratum_start = 0usize;
        while stratum_start < n {
            let stratum = sorted_strata[stratum_start];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < n && sorted_strata[stratum_end + 1] == stratum {
                stratum_end += 1;
            }

            let mut death_indices: Vec<usize> = Vec::new();
            let mut risk_indices: Vec<usize> = Vec::new();
            let mut mean = vec![0.0; nvar];
            let mut risk_weighted_covariates = vec![0.0; nvar];
            let mut death_weighted_covariates = vec![0.0; nvar];
            let mut time_start = stratum_start;
            while time_start <= stratum_end {
                let event_time = sorted_time[time_start];
                let mut time_end = time_start;
                while time_end < stratum_end && same_time(sorted_time[time_end + 1], event_time) {
                    time_end += 1;
                }

                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&idx| sorted_status[idx] == 1));
                if !death_indices.is_empty() {
                    risk_indices.clear();
                    risk_indices.extend((stratum_start..=stratum_end).filter(|&idx| {
                        sorted_start[idx] < event_time && sorted_time[idx] >= event_time
                    }));
                    mean.fill(0.0);
                    if matches!(method, CoxMethod::Exact) {
                        let moments = exact_tied_moments(
                            &risk_indices,
                            death_indices.len(),
                            &sorted_log_risk,
                            &covar,
                        );
                        for (value, &expected_sum) in mean.iter_mut().zip(&moments.mean) {
                            *value = expected_sum / death_indices.len() as f64;
                        }
                    } else {
                        let mut denom = 0.0;
                        let mut death_denom = 0.0;
                        risk_weighted_covariates.fill(0.0);
                        death_weighted_covariates.fill(0.0);
                        for &idx in &risk_indices {
                            let risk = sorted_risk[idx];
                            denom += risk;
                            for col_idx in 0..nvar {
                                let value = covar[(idx, col_idx)];
                                risk_weighted_covariates[col_idx] += risk * value;
                                if same_time(sorted_time[idx], event_time)
                                    && sorted_status[idx] == 1
                                {
                                    death_weighted_covariates[col_idx] += risk * value;
                                }
                            }
                            if same_time(sorted_time[idx], event_time) && sorted_status[idx] == 1 {
                                death_denom += risk;
                            }
                        }
                        if matches!(method, CoxMethod::Efron) && death_indices.len() > 1 {
                            let deaths = death_indices.len() as f64;
                            for step in 0..death_indices.len() {
                                let fraction = step as f64 / deaths;
                                let step_denom = denom - fraction * death_denom;
                                if step_denom > 0.0 {
                                    for col_idx in 0..nvar {
                                        mean[col_idx] += (risk_weighted_covariates[col_idx]
                                            - fraction * death_weighted_covariates[col_idx])
                                            / step_denom
                                            / deaths;
                                    }
                                }
                            }
                        } else if denom > 0.0 {
                            for col_idx in 0..nvar {
                                mean[col_idx] = risk_weighted_covariates[col_idx] / denom;
                            }
                        }
                    }

                    for &idx in &death_indices {
                        residuals.push(
                            (0..nvar)
                                .map(|col_idx| covar[(idx, col_idx)] - mean[col_idx])
                                .collect(),
                        );
                    }
                }

                if time_end == stratum_end {
                    break;
                }
                time_start = time_end + 1;
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn schoenfeld_residuals_sweep(
        &self,
        nvar: usize,
        order: &[usize],
        entry_times: Option<&Vec<f64>>,
        method: CoxMethod,
    ) -> Vec<Vec<f64>> {
        let event_count = order.iter().filter(|&&idx| self.status[idx] == 1).count();
        let mut residuals = Vec::with_capacity(event_count);
        let use_entry_times = entry_times.is_some();
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }

            let rows: Vec<CoxSweepRow> = order[stratum_start..=stratum_end]
                .iter()
                .map(|&idx| CoxSweepRow {
                    original_idx: idx,
                    stop: self.event_times[idx],
                    entry: entry_times.map_or(f64::NEG_INFINITY, |values| values[idx]),
                    risk: self.linear_predictors[idx]
                        .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                        .exp()
                        * self.weights[idx],
                    weight: self.weights[idx],
                    status: self.status[idx],
                })
                .collect();

            let mut active_cov = vec![0.0; nvar];
            if !use_entry_times {
                for row in &rows {
                    for (col_idx, value) in self.covariates[row.original_idx]
                        .iter()
                        .copied()
                        .enumerate()
                        .take(nvar)
                    {
                        active_cov[col_idx] += row.risk * value;
                    }
                }
            }
            let mut active = ActiveRiskSet::new(&rows, use_entry_times);

            let mut deaths: Vec<usize> = Vec::new();
            let mut mean = vec![0.0; nvar];
            let mut death_cov = vec![0.0; nvar];
            let mut time_start = 0usize;
            while time_start < rows.len() {
                let event_time = rows[time_start].stop;
                let mut time_end = time_start;
                while time_end + 1 < rows.len() && same_time(rows[time_end + 1].stop, event_time) {
                    time_end += 1;
                }

                active.advance_to(event_time, |row_idx, entered| {
                    let sign = if entered { 1.0 } else { -1.0 };
                    for (col_idx, value) in self.covariates[rows[row_idx].original_idx]
                        .iter()
                        .copied()
                        .enumerate()
                        .take(nvar)
                    {
                        active_cov[col_idx] += sign * rows[row_idx].risk * value;
                    }
                });

                deaths.clear();
                deaths.extend((time_start..=time_end).filter(|&idx| rows[idx].status == 1));
                if !deaths.is_empty() && active.risk_sum > 0.0 {
                    mean.fill(0.0);
                    if matches!(method, CoxMethod::Efron) && deaths.len() > 1 {
                        let mut death_risk = 0.0;
                        death_cov.fill(0.0);
                        for &row_idx in &deaths {
                            death_risk += rows[row_idx].risk;
                            for (col_idx, value) in self.covariates[rows[row_idx].original_idx]
                                .iter()
                                .copied()
                                .enumerate()
                                .take(nvar)
                            {
                                death_cov[col_idx] += rows[row_idx].risk * value;
                            }
                        }
                        let death_count = deaths.len() as f64;
                        for step in 0..deaths.len() {
                            let fraction = step as f64 / death_count;
                            let step_denom = active.risk_sum - fraction * death_risk;
                            if step_denom > 0.0 {
                                for col_idx in 0..nvar {
                                    mean[col_idx] += (active_cov[col_idx]
                                        - fraction * death_cov[col_idx])
                                        / step_denom
                                        / death_count;
                                }
                            }
                        }
                    } else {
                        for col_idx in 0..nvar {
                            mean[col_idx] = active_cov[col_idx] / active.risk_sum;
                        }
                    }

                    for &row_idx in &deaths {
                        residuals.push(
                            self.covariates[rows[row_idx].original_idx]
                                .iter()
                                .zip(mean.iter())
                                .map(|(&value, &mean)| value - mean)
                                .collect(),
                        );
                    }
                }

                time_start = time_end + 1;
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn partial_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let expected = self.expected_events_internal()?;
        let n = expected.len();
        if nvar == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        if self.covariates.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match residual length",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }

        Ok(self
            .covariates
            .iter()
            .zip(self.status.iter().zip(expected.iter()))
            .map(|(row, (&status, &expected))| {
                let residual = status as f64 - expected;
                row.iter()
                    .zip(beta.iter())
                    .map(|(&value, &coefficient)| residual + value * coefficient)
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cox_diagnostic_helpers_match_python_formulas() {
        let indices = cox_event_indices(
            vec![2.0, 1.0, 2.0, 3.0],
            vec![1, 0, 1, 1],
            Some(vec![1, 0, 0, 1]),
        )
        .expect("event indices should compute");
        assert_eq!(indices, vec![2, 0, 3]);
        assert_eq!(
            diagnostic_order(&[1, 0, 0, 1], &[2.0, 1.0, 2.0, 1.0]),
            vec![1, 2, 3, 0]
        );
        let implicit_strata = cox_event_indices(vec![2.0, 1.0, 2.0, 3.0], vec![1, 0, 1, 1], None)
            .expect("event indices should compute without strata");
        let explicit_zero_strata = cox_event_indices(
            vec![2.0, 1.0, 2.0, 3.0],
            vec![1, 0, 1, 1],
            Some(vec![0, 0, 0, 0]),
        )
        .expect("event indices should compute with explicit zero strata");
        assert_eq!(implicit_strata, explicit_zero_strata);
        assert!(cox_event_indices(vec![1.0, 2.0], vec![1, 0], Some(vec![0])).is_err());

        let scaled = scale_schoenfeld_residuals(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0.5, -0.5],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        )
        .expect("scaled Schoenfeld residuals should compute");
        assert!((scaled[0][0] - 1.9).abs() < 1e-12);
        assert!((scaled[0][1] - 1.5).abs() < 1e-12);
        assert!((scaled[1][0] - 3.5).abs() < 1e-12);
        assert!((scaled[1][1] - 3.9).abs() < 1e-12);

        let dfbeta = cox_dfbeta_from_score_residuals(
            vec![vec![1.0, 2.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            false,
        )
        .expect("dfbeta should compute");
        assert!((dfbeta[0][0] - 0.5).abs() < 1e-12);
        assert!((dfbeta[0][1] - 1.1).abs() < 1e-12);

        let grouped = cox_zph_term_matrix(
            vec![vec![1.0, 2.0, 3.0]],
            vec![vec![0, 1], vec![2]],
            vec![0.5, 2.0, 7.0],
        )
        .expect("term matrix should compute");
        assert_eq!(grouped, vec![vec![4.5, 3.0]]);

        let variance = cox_zph_group_variance(
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![vec![0, 1], vec![2]],
            vec![0.5, 2.0, 7.0],
        )
        .expect("group variance should compute");
        assert_eq!(variance, vec![vec![4.25, 0.0], vec![0.0, 1.0]]);

        let residuals = vec![
            vec![0.1, 1.2, -0.3],
            vec![0.4, 0.8, 0.2],
            vec![-0.2, 1.5, 0.7],
            vec![0.9, -0.1, 0.4],
        ];
        let event_information = vec![
            vec![
                vec![2.0, 0.2, 0.1],
                vec![0.2, 1.5, 0.1],
                vec![0.1, 0.1, 1.0],
            ],
            vec![
                vec![1.0, 0.1, 0.0],
                vec![0.1, 2.0, 0.2],
                vec![0.0, 0.2, 1.5],
            ],
            vec![
                vec![1.5, 0.0, 0.1],
                vec![0.0, 1.0, 0.1],
                vec![0.1, 0.1, 2.0],
            ],
            vec![
                vec![2.0, 0.1, 0.2],
                vec![0.1, 1.5, 0.0],
                vec![0.2, 0.0, 1.0],
            ],
        ];
        let transformed_time = vec![4.0, 1.0, 3.0, 2.0];
        let event_counts = vec![1, 1, 1, 1];
        let groups = vec![vec![0, 1], vec![2]];
        let beta = vec![0.5, -0.25, 1.5];
        let grouped_test = cox_zph_tests(
            residuals.clone(),
            event_information.clone(),
            transformed_time.clone(),
            event_counts.clone(),
            groups.clone(),
            beta.clone(),
            false,
            true,
        )
        .expect("grouped proportional-hazards tests should compute");
        assert_eq!(grouped_test.chi2_values.len(), 2);
        assert!(
            grouped_test
                .chi2_values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(grouped_test.global_chi2.is_finite() && grouped_test.global_chi2 >= 0.0);
        assert_eq!(grouped_test.global_df, 3);

        let single_df_test = cox_zph_tests(
            residuals.clone(),
            event_information,
            transformed_time.clone(),
            event_counts,
            groups.clone(),
            beta.clone(),
            true,
            true,
        )
        .expect("single-df proportional-hazards tests should compute");
        assert_eq!(single_df_test.global_df, 3);
        assert_eq!(single_df_test.chi2_values.len(), 2);
        assert!(
            single_df_test
                .chi2_values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );

        let dense_information = vec![
            vec![2.0, 0.5, -0.2],
            vec![0.4, 3.0, 0.7],
            vec![-0.1, 0.6, 4.0],
        ];
        let sparse_variance = cox_zph_group_variance(
            dense_information.clone(),
            vec![vec![0, 2], vec![1]],
            vec![0.5, 2.0, -1.5],
        )
        .expect("sparse grouped variance should compute");
        let loadings = [vec![0.5, 0.0, -1.5], vec![0.0, 1.0, 0.0]];
        let expected_variance: Vec<Vec<f64>> = loadings
            .iter()
            .map(|left| {
                loadings
                    .iter()
                    .map(|right| {
                        left.iter()
                            .enumerate()
                            .map(|(row, &left_value)| {
                                right
                                    .iter()
                                    .enumerate()
                                    .map(|(col, &right_value)| {
                                        left_value * dense_information[row][col] * right_value
                                    })
                                    .sum::<f64>()
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();
        assert_eq!(sparse_variance, expected_variance);
        assert!(
            cox_zph_tests(
                vec![vec![1.0], vec![2.0]],
                vec![vec![vec![1.0]], vec![vec![1.0]]],
                vec![1.0],
                vec![1, 1],
                vec![vec![0]],
                vec![0.5],
                false,
                true,
            )
            .is_err()
        );

        let crossprod = clustered_crossprod(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![1.0, 0.5, 2.0],
            vec![0, 0, 1],
            Some(2),
        )
        .expect("clustered cross-product should compute");
        assert_eq!(crossprod, vec![vec![106.25, 130.0], vec![130.0, 160.0]]);

        let sandwich = clustered_sandwich_variance(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![1.0, 0.5, 2.0],
            vec![0, 0, 1],
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        )
        .expect("clustered sandwich variance should compute");
        assert_eq!(sandwich, vec![vec![725.0, 478.75], vec![478.75, 316.5625]]);

        let prediction_se = prediction_se_from_variance(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        )
        .expect("prediction SEs should compute");
        assert!((prediction_se[0] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((prediction_se[1] - 46.0_f64.sqrt()).abs() < 1e-12);

        let term_prediction_se = term_prediction_se_from_variance(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
            vec![vec![0], vec![1], vec![0, 1]],
        )
        .expect("term prediction SEs should compute");
        assert!((term_prediction_se[0][0] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((term_prediction_se[0][1] - 2.0).abs() < 1e-12);
        assert!((term_prediction_se[0][2] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((term_prediction_se[1][0] - 18.0_f64.sqrt()).abs() < 1e-12);
        assert!((term_prediction_se[1][1] - 4.0).abs() < 1e-12);
        assert!((term_prediction_se[1][2] - 46.0_f64.sqrt()).abs() < 1e-12);

        let interval_se = cox_interval_cumulative_hazard_se(
            vec![vec![1.0, 2.0], vec![0.0, 0.0]],
            vec![0.25, 0.0],
            vec![0.04, 0.50],
            vec![vec![0.1, 0.2], vec![0.0, 0.0]],
            vec![1.0, 0.0],
            vec![0.25, 0.25],
            vec![vec![0.4, 0.8], vec![0.0, 0.0]],
            vec![3.0, 2.0],
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        )
        .expect("interval SEs should compute");
        assert!((interval_se[0] - 3.0 * 1.83_f64.sqrt()).abs() < 1e-12);
        assert_eq!(interval_se[1], 0.0);
    }

    #[test]
    fn native_cox_zph_path_matches_materialized_detail_blocks() {
        let time = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 0];
        let covariates = vec![
            vec![0.2, 1.0],
            vec![0.8, 0.5],
            vec![0.4, 0.9],
            vec![1.1, 0.2],
            vec![0.3, 0.8],
            vec![0.7, 0.4],
        ];
        let coefficients = vec![0.15, -0.2];
        let weights = vec![1.0, 2.0, 0.5, 1.5, 0.75, 1.0];
        let entry_times = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let transformed_events = vec![1.5, 1.5, 4.0, 5.0];
        let groups = vec![vec![0], vec![1]];

        let detail = coxph_detail(
            time.clone(),
            status.clone(),
            covariates.clone(),
            coefficients.clone(),
            Some(weights.clone()),
            Some(entry_times.clone()),
            None,
            None,
            "efron".to_string(),
            0.0,
            false,
        )
        .expect("detail blocks should compute");
        let materialized = cox_zph_tests(
            detail.rows.iter().map(|row| row.score.clone()).collect(),
            detail.rows.iter().map(|row| row.imat.clone()).collect(),
            vec![1.5, 4.0, 5.0],
            detail.rows.iter().map(|row| row.n_event).collect(),
            groups.clone(),
            coefficients.clone(),
            false,
            true,
        )
        .expect("materialized diagnostic should compute");
        let native = cox_zph_tests_from_data(
            time,
            status,
            covariates,
            coefficients,
            transformed_events,
            groups,
            false,
            true,
            Some(weights),
            Some(entry_times),
            None,
            None,
            "efron".to_string(),
        )
        .expect("native diagnostic should compute");

        assert_eq!(native.chi2_values, materialized.chi2_values);
        assert_eq!(native.p_values, materialized.p_values);
        assert_eq!(native.global_chi2, materialized.global_chi2);
        assert_eq!(native.global_p_value, materialized.global_p_value);
    }

    #[test]
    fn fit_owned_cox_zph_path_matches_selected_data_path() {
        let time = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 0];
        let covariates = vec![
            vec![0.2, 1.0],
            vec![0.8, 0.5],
            vec![0.4, 0.9],
            vec![1.1, 0.2],
            vec![0.3, 0.8],
            vec![0.7, 0.4],
        ];
        let coefficients = vec![0.15, -0.2];
        let weights = vec![1.0, 2.0, 0.5, 1.5, 0.75, 1.0];
        let entry_times = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let transformed_events = vec![1.5, 1.5, 4.0, 5.0];
        let variance = vec![vec![0.4, 0.1], vec![0.1, 0.3]];
        let linear_predictors = covariates
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&coefficients)
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum()
            })
            .collect::<Vec<f64>>();
        let fit = CoxPHFit {
            coefficients: vec![coefficients.clone()],
            means: vec![0.0; coefficients.len()],
            score_vector: vec![0.0; coefficients.len()],
            information_matrix: variance.clone(),
            degrees_of_freedom: 0.0,
            log_likelihood: vec![0.0, 0.0],
            score_test: 0.0,
            convergence_flag: 0,
            initial_information_rank: 0,
            iterations: 0,
            risk_scores: linear_predictors.iter().map(|value| value.exp()).collect(),
            event_times: time.clone(),
            status: status.clone(),
            linear_predictors: linear_predictors.clone(),
            entry_times: Some(entry_times.clone()),
            weights: weights.clone(),
            covariates: covariates.clone(),
            strata: vec![0; time.len()],
            method: "efron".to_string(),
            nocenter: Vec::new(),
            counting_roundoff_compatibility: false,
        };

        let (grouped, test) = fit
            .cox_zph_diagnostics_internal(
                transformed_events.clone(),
                vec![1],
                vec![vec![0]],
                variance.clone(),
                false,
                true,
                None,
            )
            .expect("fit-owned Cox zph diagnostic should compute");
        let expected_scaled = fit
            .scaled_schoenfeld_residuals_with_variance_internal(&variance)
            .expect("scaled residuals should compute");
        for (actual, expected) in grouped.iter().zip(&expected_scaled) {
            assert_eq!(actual, &vec![expected[1]]);
        }
        let (grouped_term, _) = fit
            .cox_zph_diagnostics_internal(
                transformed_events.clone(),
                vec![0, 1],
                vec![vec![0, 1]],
                variance.clone(),
                false,
                true,
                None,
            )
            .expect("grouped fit-owned Cox zph diagnostic should compute");
        for (actual, expected) in grouped_term.iter().zip(&expected_scaled) {
            assert_eq!(
                actual,
                &vec![expected[0] * coefficients[0] + expected[1] * coefficients[1]]
            );
        }

        let selected_rows = covariates
            .iter()
            .map(|row| vec![row[1]])
            .collect::<Vec<_>>();
        let selected_beta = vec![coefficients[1]];
        let offset = selected_rows
            .iter()
            .zip(&linear_predictors)
            .map(|(row, &linear_predictor)| linear_predictor - row[0] * selected_beta[0])
            .collect();
        let expected_test = cox_zph_tests_from_data(
            time,
            status,
            selected_rows,
            selected_beta,
            transformed_events,
            vec![vec![0]],
            false,
            true,
            Some(weights),
            Some(entry_times),
            Some(vec![0; covariates.len()]),
            Some(offset),
            "efron".to_string(),
        )
        .expect("selected data diagnostic should compute");

        assert_eq!(test.chi2_values, expected_test.chi2_values);
        assert_eq!(test.p_values, expected_test.p_values);
        assert_eq!(test.global_chi2, expected_test.global_chi2);
        assert_eq!(test.global_p_value, expected_test.global_p_value);
    }

    #[test]
    fn fit_owned_cox_zph_surface_scales_each_stratum_by_active_terms() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 1, 0, 1, 1, 1, 0];
        let covariates = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 2.0],
            vec![0.0, 3.0],
        ];
        let coefficients = vec![0.1, -0.2];
        let linear_predictors = covariates
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&coefficients)
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum()
            })
            .collect::<Vec<f64>>();
        let fit = CoxPHFit {
            coefficients: vec![coefficients],
            means: vec![0.0, 0.0],
            score_vector: vec![0.0, 0.0],
            information_matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            degrees_of_freedom: 0.0,
            log_likelihood: vec![0.0, 0.0],
            score_test: 0.0,
            convergence_flag: 0,
            initial_information_rank: 0,
            iterations: 0,
            risk_scores: linear_predictors.iter().map(|value| value.exp()).collect(),
            event_times,
            status,
            linear_predictors,
            entry_times: None,
            weights: vec![1.0; 8],
            covariates,
            strata: vec![0, 0, 0, 0, 1, 1, 1, 1],
            method: "efron".to_string(),
            nocenter: Vec::new(),
            counting_roundoff_compatibility: false,
        };

        let (surface, variance, test) = fit
            .cox_zph_diagnostics_with_surface_internal(
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                vec![0, 1],
                vec![vec![0], vec![1]],
                false,
                true,
            )
            .expect("stratified Cox zph surface should compute");

        assert_eq!(surface.len(), 6);
        for row in &surface[..3] {
            assert!(row[0].is_finite());
            assert!(row[1].is_nan());
        }
        for row in &surface[3..] {
            assert!(row[0].is_nan());
            assert!(row[1].is_finite());
        }
        assert_eq!(variance.len(), 2);
        assert!(variance[0][0].is_finite() && variance[0][0] > 0.0);
        assert!(variance[1][1].is_finite() && variance[1][1] > 0.0);
        assert!(variance[0][1].abs() < 1e-12);
        assert!(variance[1][0].abs() < 1e-12);
        assert_eq!(test.chi2_values.len(), 2);
        assert!(test.global_chi2.is_finite());
    }

    #[test]
    fn partial_residuals_reuse_expected_events() {
        let fit = CoxPHFit {
            coefficients: vec![vec![0.5, -0.25]],
            means: vec![0.0, 0.0],
            score_vector: vec![],
            information_matrix: vec![],
            degrees_of_freedom: 0.0,
            log_likelihood: vec![],
            score_test: 0.0,
            convergence_flag: 0,
            initial_information_rank: 0,
            iterations: 0,
            risk_scores: vec![],
            event_times: vec![1.0, 2.0, 3.0],
            status: vec![1, 0, 1],
            linear_predictors: vec![0.2, -0.1, 0.3],
            entry_times: None,
            weights: vec![1.0, 1.0, 1.0],
            covariates: vec![vec![1.0, 0.5], vec![0.0, 1.5], vec![2.0, -1.0]],
            strata: vec![0, 0, 0],
            method: "breslow".to_string(),
            nocenter: Vec::new(),
            counting_roundoff_compatibility: false,
        };
        let beta = fit.coefficients.first().expect("test coefficients exist");
        let martingale = fit
            .martingale_residuals()
            .expect("martingale residuals should compute");
        let expected: Vec<Vec<f64>> = fit
            .covariates
            .iter()
            .zip(martingale.iter())
            .map(|(row, &residual)| {
                row.iter()
                    .zip(beta.iter())
                    .map(|(&value, &coefficient)| residual + value * coefficient)
                    .collect()
            })
            .collect();

        let actual = fit
            .partial_residuals_internal()
            .expect("partial residuals should compute");

        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual, &expected) in actual_row.iter().zip(expected_row.iter()) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scaled_schoenfeld_residuals_accept_custom_variance() {
        let fit = CoxPHFit {
            coefficients: vec![vec![0.25]],
            means: vec![0.0],
            score_vector: vec![],
            information_matrix: vec![vec![0.5]],
            degrees_of_freedom: 1.0,
            log_likelihood: vec![],
            score_test: 0.0,
            convergence_flag: 0,
            initial_information_rank: 0,
            iterations: 0,
            risk_scores: vec![],
            event_times: vec![1.0, 2.0, 3.0],
            status: vec![1, 0, 1],
            linear_predictors: vec![0.1, -0.2, 0.3],
            entry_times: None,
            weights: vec![1.0, 1.0, 1.0],
            covariates: vec![vec![1.0], vec![0.0], vec![2.0]],
            strata: vec![0, 0, 0],
            method: "breslow".to_string(),
            nocenter: Vec::new(),
            counting_roundoff_compatibility: false,
        };
        let raw = fit
            .schoenfeld_residuals_internal()
            .expect("Schoenfeld residuals should compute");
        let custom_variance = vec![vec![0.75]];
        let expected = scale_schoenfeld_residuals_impl(
            raw,
            fit.coefficients.first().expect("test coefficient exists"),
            &custom_variance,
        )
        .expect("custom scaling should compute");

        let actual = fit
            .scaled_schoenfeld_residuals_with_variance_internal(&custom_variance)
            .expect("fused custom scaling should compute");

        assert_eq!(actual, expected);
    }

    #[test]
    fn exact_inclusion_weights_match_pairwise_tie_probabilities() {
        let log_risk = [2.0_f64.ln(), 3.0_f64.ln(), 5.0_f64.ln()];
        let inclusion = exact_inclusion_probabilities(&[0, 1, 2], 2, &log_risk)
            .expect("two deaths among three risk scores should compute");

        assert_eq!(inclusion.len(), 3);
        assert!((inclusion[0].1 - 16.0 / 31.0).abs() < 1e-12);
        assert!((inclusion[1].1 - 21.0 / 31.0).abs() < 1e-12);
        assert!((inclusion[2].1 - 25.0 / 31.0).abs() < 1e-12);
        assert!((inclusion.iter().map(|(_, value)| value).sum::<f64>() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cox_zph_score_test_uses_augmented_information_blocks() {
        let result = cox_zph_tests(
            vec![vec![1.0], vec![-0.5], vec![0.25]],
            vec![vec![vec![2.0]], vec![vec![1.0]], vec![vec![3.0]]],
            vec![1.0, 2.0, 4.0],
            vec![1, 1, 1],
            vec![vec![0]],
            vec![0.75],
            false,
            true,
        )
        .expect("augmented Cox zph score test should compute");

        let expected = 27.0 / 544.0;
        assert!((result.chi2_values[0] - expected).abs() < 1e-12);
        assert!((result.global_chi2 - expected).abs() < 1e-12);
        assert_eq!(result.global_df, 1);

        let single_df = cox_zph_tests(
            vec![vec![1.0], vec![-0.5], vec![0.25]],
            vec![vec![vec![2.0]], vec![vec![1.0]], vec![vec![3.0]]],
            vec![1.0, 2.0, 4.0],
            vec![1, 1, 1],
            vec![vec![0]],
            vec![0.75],
            true,
            true,
        )
        .expect("single-column terms should retain their ordinary score test");
        assert!((single_df.chi2_values[0] - expected).abs() < 1e-12);

        let tied = cox_zph_tests(
            vec![vec![0.5], vec![-0.25]],
            vec![vec![vec![2.0]], vec![vec![3.0]]],
            vec![1.0, 4.0],
            vec![2, 1],
            vec![vec![0]],
            vec![1.0],
            false,
            true,
        )
        .expect("event counts should weight transformed-time centering");
        assert!(tied.global_chi2.is_finite());
    }

    #[test]
    fn exact_diagnostics_are_invariant_to_common_large_log_risk_shifts() {
        let fit = CoxPHFit {
            coefficients: vec![vec![0.0]],
            means: vec![0.0],
            score_vector: vec![0.0],
            information_matrix: vec![vec![0.0]],
            degrees_of_freedom: 0.0,
            log_likelihood: vec![0.0, 0.0],
            score_test: 0.0,
            convergence_flag: 0,
            initial_information_rank: 0,
            iterations: 0,
            risk_scores: vec![0.0; 3],
            event_times: vec![1.0, 1.0, 1.0],
            status: vec![1, 1, 0],
            linear_predictors: vec![1_000.0, 900.0, 800.0],
            entry_times: None,
            weights: vec![1.0; 3],
            covariates: vec![vec![0.0], vec![1.0], vec![2.0]],
            strata: vec![0; 3],
            method: "exact".to_string(),
            nocenter: Vec::new(),
            counting_roundoff_compatibility: false,
        };
        let mut shifted = fit.clone();
        for value in &mut shifted.linear_predictors {
            *value -= 1_000.0;
        }

        let schoenfeld = fit
            .schoenfeld_residuals_internal()
            .expect("large exact Schoenfeld residuals should compute");
        let shifted_schoenfeld = shifted
            .schoenfeld_residuals_internal()
            .expect("shifted exact Schoenfeld residuals should compute");
        let score = fit
            .score_residuals_internal()
            .expect("large exact score residuals should compute");
        let shifted_score = shifted
            .score_residuals_internal()
            .expect("shifted exact score residuals should compute");
        let mut counting = fit.clone();
        counting.entry_times = Some(vec![0.0; 3]);
        let mut shifted_counting = shifted.clone();
        shifted_counting.entry_times = Some(vec![0.0; 3]);
        let counting_score = counting
            .score_residuals_internal()
            .expect("large counting-process exact score residuals should compute");
        let shifted_counting_score = shifted_counting
            .score_residuals_internal()
            .expect("shifted counting-process exact score residuals should compute");

        assert!((schoenfeld[0][0] + 0.5).abs() < 1e-12);
        assert!((schoenfeld[1][0] - 0.5).abs() < 1e-12);
        for (actual, expected) in schoenfeld.iter().zip(&shifted_schoenfeld) {
            assert!((actual[0] - expected[0]).abs() < 1e-12);
        }
        for (actual, expected) in score.iter().zip(&shifted_score) {
            assert!((actual[0] - expected[0]).abs() < 1e-12);
        }
        for (actual, expected) in counting_score.iter().zip(&shifted_counting_score) {
            assert!((actual[0] - expected[0]).abs() < 1e-12);
        }
    }
}
