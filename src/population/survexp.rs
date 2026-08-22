use super::ratetable::RateTable;
use crate::constants::{PARALLEL_THRESHOLD_XLARGE, same_time};
use crate::internal::validation::{validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const DAYS_PER_YEAR: f64 = 365.25;
const SUBJECT_BATCH_SIZE: usize = 2048;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvExpResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub cumhaz: Vec<f64>,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub n: usize,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_survexp_inputs(time: &[f64], age: &[f64], year: &[f64]) -> PyResult<()> {
    if age.len() != time.len() || year.len() != time.len() {
        return Err(value_error("time, age, and year must have same length"));
    }
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_finite(age, "age")?;
    validate_non_negative(age, "age")?;
    validate_finite(year, "year")?;
    Ok(())
}

fn validate_sex(sex: &[i32], n: usize) -> PyResult<()> {
    if sex.len() != n {
        return Err(value_error("sex must have same length as time"));
    }
    for (index, &value) in sex.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "sex values must be non-negative; got {value} at index {index}"
            )));
        }
    }
    Ok(())
}

fn validate_optional_sex(sex: Option<&[i32]>, n: usize) -> PyResult<()> {
    if let Some(values) = sex {
        validate_sex(values, n)?;
    }
    Ok(())
}

#[inline]
fn sex_at(sex: Option<&[i32]>, index: usize) -> i32 {
    sex.map_or(0, |values| values[index])
}

fn validate_eval_times(eval_times: &[f64]) -> PyResult<()> {
    validate_finite(eval_times, "times")?;
    validate_non_negative(eval_times, "times")?;
    for (index, pair) in eval_times.windows(2).enumerate() {
        if pair[1] < pair[0] && !same_time(pair[0], pair[1]) {
            return Err(value_error(format!(
                "times must be sorted in nondecreasing order; index {} is less than index {}",
                index + 1,
                index
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (time, age, year, ratetable, sex=None, times=None, method=None))]
pub fn survexp(
    time: Vec<f64>,
    age: Vec<f64>,
    year: Vec<f64>,
    ratetable: &RateTable,
    sex: Option<Vec<i32>>,
    times: Option<Vec<f64>>,
    method: Option<&str>,
) -> PyResult<SurvExpResult> {
    let n = time.len();
    validate_survexp_inputs(&time, &age, &year)?;

    let sex = sex.as_deref();
    validate_optional_sex(sex, n)?;

    let calc_method = method.unwrap_or("hakulinen");
    if !["ederer", "hakulinen", "conditional", "individual"].contains(&calc_method) {
        return Err(value_error(
            "method must be 'ederer', 'hakulinen', 'conditional', or 'individual'",
        ));
    }

    if n == 0 {
        return Ok(SurvExpResult {
            time: vec![],
            surv: vec![],
            n_risk: vec![],
            cumhaz: vec![],
            method: calc_method.to_string(),
            n: 0,
        });
    }

    let eval_times = match times {
        Some(t) => t,
        None => {
            let mut unique_times: Vec<f64> = time.clone();
            unique_times.sort_by(|a, b| a.total_cmp(b));
            unique_times.dedup_by(|left, right| same_time(*left, *right));
            unique_times
        }
    };
    validate_eval_times(&eval_times)?;

    match calc_method {
        "ederer" => compute_ederer(&time, &age, &year, sex, ratetable, &eval_times),
        "hakulinen" => compute_hakulinen(&time, &age, &year, sex, ratetable, &eval_times),
        "conditional" => compute_conditional(&time, &age, &year, sex, ratetable, &eval_times),
        "individual" => compute_individual(&time, &age, &year, sex, ratetable, &eval_times),
        _ => compute_hakulinen(&time, &age, &year, sex, ratetable, &eval_times),
    }
}

fn coordinate_rows(columns: &[Vec<f64>], row_count: usize) -> Vec<f64> {
    let mut rows = Vec::with_capacity(row_count.saturating_mul(columns.len()));
    for row in 0..row_count {
        rows.extend(columns.iter().map(|column| column[row]));
    }
    rows
}

fn subject_coordinate_cumulative_hazards(
    base_coordinates: &[f64],
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<Vec<f64>> {
    let mut result = Vec::with_capacity(eval_times.len());
    let mut previous_time = 0.0;
    let mut cumulative_hazard = 0.0;
    for &eval_time in eval_times {
        if eval_time < previous_time {
            cumulative_hazard =
                ratetable.cumulative_hazard_from_values(base_coordinates, eval_time)?;
        } else {
            cumulative_hazard += ratetable.cumulative_hazard_interval_from_values(
                base_coordinates,
                previous_time,
                eval_time,
            )?;
        }
        result.push(cumulative_hazard);
        previous_time = eval_time;
    }
    Ok(result)
}

fn compute_coordinate_curve_averages(
    time: &[f64],
    coordinates: &[f64],
    dimension_count: usize,
    ratetable: &RateTable,
    eval_times: &[f64],
    observed_risk_set: bool,
) -> PyResult<CurveAverages> {
    let n_times = eval_times.len();
    let mut survival_totals = vec![0.0; n_times];
    let mut hazard_totals = vec![0.0; n_times];
    let mut n_risk = vec![0.0; n_times];

    for batch_start in (0..time.len()).step_by(SUBJECT_BATCH_SIZE) {
        let batch_end = (batch_start + SUBJECT_BATCH_SIZE).min(time.len());
        let batch_hazards = (batch_start..batch_end)
            .into_par_iter()
            .map(|row| {
                let start = row * dimension_count;
                subject_coordinate_cumulative_hazards(
                    &coordinates[start..start + dimension_count],
                    ratetable,
                    eval_times,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;

        for (offset, hazards) in batch_hazards.into_iter().enumerate() {
            let subject = batch_start + offset;
            for (time_index, (&eval_time, cumulative_hazard)) in
                eval_times.iter().zip(hazards).enumerate()
            {
                if observed_risk_set && time[subject] < eval_time {
                    continue;
                }
                survival_totals[time_index] += (-cumulative_hazard).exp();
                hazard_totals[time_index] += cumulative_hazard;
                n_risk[time_index] += 1.0;
            }
        }
    }

    let mut survival = Vec::with_capacity(n_times);
    let mut mean_cumhaz = Vec::with_capacity(n_times);
    for ((survival_total, hazard_total), count) in
        survival_totals.into_iter().zip(hazard_totals).zip(&n_risk)
    {
        if *count > 0.0 {
            survival.push(survival_total / count);
            mean_cumhaz.push(hazard_total / count);
        } else {
            survival.push(0.0);
            mean_cumhaz.push(0.0);
        }
    }
    Ok(CurveAverages {
        survival,
        mean_cumhaz,
        n_risk,
    })
}

fn compute_coordinate_conditional(
    time: &[f64],
    coordinates: &[f64],
    dimension_count: usize,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    let mut surv = vec![1.0; eval_times.len()];
    let mut cumhaz = vec![0.0; eval_times.len()];
    let mut n_risk = vec![time.len() as f64; eval_times.len()];
    let mut previous_time = 0.0;
    let mut previous_survival: f64 = 1.0;

    for (time_index, &eval_time) in eval_times.iter().enumerate() {
        let mut at_risk = 0usize;
        let mut total_hazard = 0.0;
        for (row, &follow_up) in time.iter().enumerate() {
            if follow_up < eval_time {
                continue;
            }
            at_risk += 1;
            let start = row * dimension_count;
            total_hazard += ratetable.cumulative_hazard_interval_from_values(
                &coordinates[start..start + dimension_count],
                previous_time,
                eval_time,
            )?;
        }
        n_risk[time_index] = at_risk as f64;
        if at_risk == 0 {
            surv[time_index] = previous_survival;
            cumhaz[time_index] = if previous_survival > 0.0 {
                -previous_survival.ln()
            } else {
                f64::INFINITY
            };
            continue;
        }
        let interval_survival = (-(total_hazard / at_risk as f64)).exp();
        previous_survival *= interval_survival;
        surv[time_index] = previous_survival;
        cumhaz[time_index] = if previous_survival > 0.0 {
            -previous_survival.ln()
        } else {
            f64::INFINITY
        };
        previous_time = eval_time;
    }

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv,
        n_risk,
        cumhaz,
        method: "conditional".to_string(),
        n: time.len(),
    })
}

#[pyfunction]
#[pyo3(signature = (time, ratetable, coordinates, times=None, method=None))]
pub fn survexp_from_coords(
    time: Vec<f64>,
    ratetable: &RateTable,
    coordinates: HashMap<String, Vec<f64>>,
    times: Option<Vec<f64>>,
    method: Option<&str>,
) -> PyResult<SurvExpResult> {
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    let coordinate_columns = ratetable.aligned_coordinate_columns(&coordinates, time.len())?;
    let dimension_count = coordinate_columns.len();
    let coordinate_rows = coordinate_rows(&coordinate_columns, time.len());
    let calculation = method.unwrap_or("hakulinen");
    if !["ederer", "hakulinen", "conditional", "individual"].contains(&calculation) {
        return Err(value_error(
            "method must be 'ederer', 'hakulinen', 'conditional', or 'individual'",
        ));
    }
    if time.is_empty() {
        return Ok(SurvExpResult {
            time: vec![],
            surv: vec![],
            n_risk: vec![],
            cumhaz: vec![],
            method: calculation.to_string(),
            n: 0,
        });
    }
    let eval_times = match times {
        Some(values) => values,
        None => {
            let mut values = time.clone();
            values.sort_by(f64::total_cmp);
            values.dedup_by(|left, right| same_time(*left, *right));
            values
        }
    };
    validate_eval_times(&eval_times)?;
    if calculation == "conditional" {
        return compute_coordinate_conditional(
            &time,
            &coordinate_rows,
            dimension_count,
            ratetable,
            &eval_times,
        );
    }
    let observed_risk_set = calculation != "ederer";
    let averages = compute_coordinate_curve_averages(
        &time,
        &coordinate_rows,
        dimension_count,
        ratetable,
        &eval_times,
        observed_risk_set,
    )?;
    let cumhaz = if calculation == "individual" {
        averages
            .survival
            .iter()
            .map(|&value| {
                if value > 0.0 {
                    -value.ln()
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    } else {
        averages.mean_cumhaz
    };
    Ok(SurvExpResult {
        time: eval_times,
        surv: averages.survival,
        n_risk: averages.n_risk,
        cumhaz,
        method: calculation.to_string(),
        n: time.len(),
    })
}

#[pyfunction]
pub fn survexp_individual_from_coords(
    time: Vec<f64>,
    ratetable: &RateTable,
    coordinates: HashMap<String, Vec<f64>>,
) -> PyResult<Vec<f64>> {
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    let coordinate_columns = ratetable.aligned_coordinate_columns(&coordinates, time.len())?;
    let dimension_count = coordinate_columns.len();
    let coordinate_rows = coordinate_rows(&coordinate_columns, time.len());
    let expected_for_subject = |row: usize| {
        let start = row * dimension_count;
        ratetable
            .cumulative_hazard_from_values(
                &coordinate_rows[start..start + dimension_count],
                time[row],
            )
            .map(|hazard| (-hazard).exp())
    };
    if time.len() > PARALLEL_THRESHOLD_XLARGE {
        (0..time.len())
            .into_par_iter()
            .map(expected_for_subject)
            .collect()
    } else {
        (0..time.len()).map(expected_for_subject).collect()
    }
}

#[derive(Clone, Copy)]
enum CohortMethod {
    Ederer,
    Hakulinen,
}

impl CohortMethod {
    fn uses_observed_risk_set(self) -> bool {
        matches!(self, Self::Hakulinen)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ederer => "ederer",
            Self::Hakulinen => "hakulinen",
        }
    }
}

struct CurveAverages {
    survival: Vec<f64>,
    mean_cumhaz: Vec<f64>,
    n_risk: Vec<f64>,
}

fn subject_cumulative_hazards(
    age: f64,
    year: f64,
    sex: i32,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> Vec<f64> {
    let mut result = Vec::with_capacity(eval_times.len());
    let mut previous_time = 0.0;
    let mut cumulative_hazard = 0.0;

    for &eval_time in eval_times {
        if eval_time < previous_time {
            cumulative_hazard = ratetable
                .cumulative_hazard(age, age + eval_time, year, Some(sex))
                .unwrap_or(0.0);
        } else {
            cumulative_hazard += ratetable
                .cumulative_hazard(
                    age + previous_time,
                    age + eval_time,
                    year + previous_time / DAYS_PER_YEAR,
                    Some(sex),
                )
                .unwrap_or(0.0);
        }
        result.push(cumulative_hazard);
        previous_time = eval_time;
    }
    result
}

fn compute_curve_averages(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
    observed_risk_set: bool,
) -> CurveAverages {
    let n_times = eval_times.len();
    let mut survival_totals = vec![0.0; n_times];
    let mut hazard_totals = vec![0.0; n_times];
    let mut n_risk = vec![0.0; n_times];

    for batch_start in (0..time.len()).step_by(SUBJECT_BATCH_SIZE) {
        let batch_end = (batch_start + SUBJECT_BATCH_SIZE).min(time.len());
        let batch_hazards = (batch_start..batch_end)
            .into_par_iter()
            .map(|index| {
                subject_cumulative_hazards(
                    age[index],
                    year[index],
                    sex_at(sex, index),
                    ratetable,
                    eval_times,
                )
            })
            .collect::<Vec<_>>();

        for (offset, hazards) in batch_hazards.into_iter().enumerate() {
            let subject = batch_start + offset;
            for (time_index, (&eval_time, cumulative_hazard)) in
                eval_times.iter().zip(hazards).enumerate()
            {
                if observed_risk_set && time[subject] < eval_time {
                    continue;
                }
                survival_totals[time_index] += (-cumulative_hazard).exp();
                hazard_totals[time_index] += cumulative_hazard;
                n_risk[time_index] += 1.0;
            }
        }
    }

    let mut survival = Vec::with_capacity(n_times);
    let mut mean_cumhaz = Vec::with_capacity(n_times);
    for ((survival_total, hazard_total), count) in
        survival_totals.into_iter().zip(hazard_totals).zip(&n_risk)
    {
        if *count > 0.0 {
            survival.push(survival_total / count);
            mean_cumhaz.push(hazard_total / count);
        } else {
            survival.push(0.0);
            mean_cumhaz.push(0.0);
        }
    }

    CurveAverages {
        survival,
        mean_cumhaz,
        n_risk,
    }
}

fn compute_hakulinen(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    compute_cohort_average(
        time,
        age,
        year,
        sex,
        ratetable,
        eval_times,
        CohortMethod::Hakulinen,
    )
}

fn compute_ederer(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    compute_cohort_average(
        time,
        age,
        year,
        sex,
        ratetable,
        eval_times,
        CohortMethod::Ederer,
    )
}

fn compute_cohort_average(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
    method: CohortMethod,
) -> PyResult<SurvExpResult> {
    let n = time.len();
    let averages = compute_curve_averages(
        time,
        age,
        year,
        sex,
        ratetable,
        eval_times,
        method.uses_observed_risk_set(),
    );

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv: averages.survival,
        n_risk: averages.n_risk,
        cumhaz: averages.mean_cumhaz,
        method: method.as_str().to_string(),
        n,
    })
}

fn compute_conditional(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    let n = time.len();
    let n_times = eval_times.len();

    let mut surv = vec![1.0; n_times];
    let mut cumhaz = vec![0.0; n_times];
    let mut n_risk = vec![n as f64; n_times];

    let mut prev_time: f64 = 0.0;
    let mut prev_surv: f64 = 1.0;

    for (t_idx, &eval_t) in eval_times.iter().enumerate() {
        let mut at_risk_count = 0usize;
        let mut total_hazard = 0.0;

        for i in 0..n {
            if time[i] < eval_t {
                continue;
            }
            at_risk_count += 1;
            let age_start = age[i] + prev_time;
            let age_end = age[i] + eval_t;
            let year_start = year[i] + prev_time / DAYS_PER_YEAR;

            let interval_hazard = ratetable
                .cumulative_hazard(age_start, age_end, year_start, Some(sex_at(sex, i)))
                .unwrap_or(0.0);
            total_hazard += interval_hazard;
        }

        n_risk[t_idx] = at_risk_count as f64;

        if at_risk_count == 0 {
            surv[t_idx] = prev_surv;
            cumhaz[t_idx] = if prev_surv > 0.0 {
                -prev_surv.ln()
            } else {
                f64::INFINITY
            };
            continue;
        }

        let avg_hazard = total_hazard / at_risk_count as f64;
        let interval_surv = (-avg_hazard).exp();

        surv[t_idx] = prev_surv * interval_surv;
        cumhaz[t_idx] = if surv[t_idx] > 0.0 {
            -surv[t_idx].ln()
        } else {
            f64::INFINITY
        };

        prev_time = eval_t;
        prev_surv = surv[t_idx];
    }

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv,
        n_risk,
        cumhaz,
        method: "conditional".to_string(),
        n,
    })
}

fn compute_individual(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: Option<&[i32]>,
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    let n = time.len();
    let averages = compute_curve_averages(time, age, year, sex, ratetable, eval_times, true);
    let cumhaz = averages
        .survival
        .iter()
        .map(|&survival| {
            if survival > 0.0 {
                -survival.ln()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv: averages.survival,
        n_risk: averages.n_risk,
        cumhaz,
        method: "individual".to_string(),
        n,
    })
}

#[pyfunction]
#[pyo3(signature = (time, age, year, ratetable, sex=None))]
pub fn survexp_individual(
    time: Vec<f64>,
    age: Vec<f64>,
    year: Vec<f64>,
    ratetable: &RateTable,
    sex: Option<Vec<i32>>,
) -> PyResult<Vec<f64>> {
    let n = time.len();
    validate_survexp_inputs(&time, &age, &year)?;

    let sex = sex.as_deref();
    validate_optional_sex(sex, n)?;

    let expected_for_subject = |i: usize| {
        let age_end = age[i] + time[i];
        ratetable
            .expected_survival(age[i], age_end, year[i], Some(sex_at(sex, i)))
            .unwrap_or(1.0)
    };
    let expected = if n > PARALLEL_THRESHOLD_XLARGE {
        (0..n).into_par_iter().map(expected_for_subject).collect()
    } else {
        (0..n).map(expected_for_subject).collect()
    };

    Ok(expected)
}

#[derive(Clone, Copy)]
enum CoxRateMethod {
    Ederer,
    Hakulinen,
    Conditional,
}

impl CoxRateMethod {
    fn parse(method: &str) -> PyResult<Self> {
        match method {
            "ederer" => Ok(Self::Ederer),
            "hakulinen" => Ok(Self::Hakulinen),
            "conditional" => Ok(Self::Conditional),
            _ => Err(value_error(
                "method must be 'ederer', 'hakulinen', or 'conditional'",
            )),
        }
    }
}

type CoxAggregateOutput = (Vec<Vec<f64>>, Option<Vec<usize>>);

fn validate_cox_curves(time: &[f64], surv: &[Vec<f64>], cumhaz: &[Vec<f64>]) -> PyResult<()> {
    validate_eval_times(time)?;
    if surv.len() != cumhaz.len() {
        return Err(value_error(
            "surv and cumhaz must contain the same number of curves",
        ));
    }
    for (row, (survival_curve, hazard_curve)) in surv.iter().zip(cumhaz).enumerate() {
        if survival_curve.len() != time.len() || hazard_curve.len() != time.len() {
            return Err(value_error(format!(
                "surv[{row}] and cumhaz[{row}] must have the same length as time"
            )));
        }
        for (index, &value) in survival_curve.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(value_error(format!(
                    "surv[{row}] must contain finite probabilities; got {value} at index {index}"
                )));
            }
        }
        for (index, &value) in hazard_curve.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(value_error(format!(
                    "cumhaz[{row}] must contain finite non-negative values; got {value} at index {index}"
                )));
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn survexp_cox_aggregate(
    time: Vec<f64>,
    surv: Vec<Vec<f64>>,
    cumhaz: Vec<Vec<f64>>,
    followup: Vec<f64>,
    group: Vec<usize>,
    weights: Vec<f64>,
    method: &str,
) -> PyResult<CoxAggregateOutput> {
    let calculation = CoxRateMethod::parse(method)?;
    validate_cox_curves(&time, &surv, &cumhaz)?;

    let row_count = followup.len();
    if surv.len() != row_count || group.len() != row_count || weights.len() != row_count {
        return Err(value_error(
            "surv, cumhaz, followup, group, and weights must have the same row count",
        ));
    }
    for (index, &value) in followup.iter().enumerate() {
        if value.is_nan() || value < 0.0 {
            return Err(value_error(format!(
                "followup values must be non-negative and not NaN; got {value} at index {index}"
            )));
        }
    }
    for (index, &value) in group.iter().enumerate() {
        if value == 0 {
            return Err(value_error(format!(
                "group values must be positive integers; got 0 at index {index}"
            )));
        }
    }
    validate_finite(&weights, "weights")?;

    let group_count = group.iter().copied().max().unwrap_or(0);
    let mut group_rows = vec![Vec::new(); group_count];
    let mut group_weight_totals = vec![0.0; group_count];
    for (row, (&group_value, &weight)) in group.iter().zip(&weights).enumerate() {
        let group_index = group_value - 1;
        group_rows[group_index].push(row);
        group_weight_totals[group_index] += weight;
    }
    let aggregate_group = |group_index: usize| {
        let rows = &group_rows[group_index];
        if rows.is_empty() {
            return if matches!(calculation, CoxRateMethod::Ederer) {
                vec![0.0; time.len()]
            } else {
                vec![f64::NAN; time.len()]
            };
        }
        let total_weight = group_weight_totals[group_index];

        if matches!(calculation, CoxRateMethod::Ederer) {
            let mut curve = vec![0.0; time.len()];
            for &row in rows {
                let normalized_weight = weights[row] / total_weight;
                for (aggregate, &subject_survival) in curve.iter_mut().zip(&surv[row]) {
                    *aggregate += normalized_weight * subject_survival;
                }
            }
            return curve;
        }

        let mut curve = Vec::with_capacity(time.len());
        let mut cumulative_hazard = 0.0;
        for (time_index, &eval_time) in time.iter().enumerate() {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for &row in rows {
                if followup[row] < eval_time {
                    continue;
                }
                let previous_survival =
                    if matches!(calculation, CoxRateMethod::Conditional) || time_index == 0 {
                        1.0
                    } else {
                        surv[row][time_index - 1]
                    };
                let contribution = previous_survival * weights[row] / total_weight;
                let previous_hazard = if time_index == 0 {
                    0.0
                } else {
                    cumhaz[row][time_index - 1]
                };
                numerator += (cumhaz[row][time_index] - previous_hazard) * contribution;
                denominator += contribution;
            }
            let increment = numerator / denominator;
            cumulative_hazard += increment;
            curve.push((-cumulative_hazard).exp());
        }
        curve
    };

    let survival_by_group = if group_count > PARALLEL_THRESHOLD_XLARGE {
        (0..group_count)
            .into_par_iter()
            .map(aggregate_group)
            .collect()
    } else {
        (0..group_count).map(aggregate_group).collect()
    };
    let group_sizes = matches!(calculation, CoxRateMethod::Ederer)
        .then(|| group_rows.iter().map(Vec::len).collect());
    Ok((survival_by_group, group_sizes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::ratetable::{DimType, RateDimension, create_simple_ratetable};

    fn create_test_ratetable() -> RateTable {
        let age_breaks = vec![0.0, 36500.0, 73000.0];
        let year_breaks = vec![1990.0, 2020.0];

        let rates_male = vec![0.00001, 0.00005];
        let rates_female = vec![0.000008, 0.00004];

        create_simple_ratetable(age_breaks, year_breaks, rates_male, rates_female).unwrap()
    }

    #[test]
    fn test_survexp_basic() {
        let rt = create_test_ratetable();

        let time = vec![365.0, 730.0, 1095.0];
        let age = vec![18250.0, 21900.0, 25550.0];
        let year = vec![2000.0, 2000.0, 2000.0];
        let sex = vec![0, 1, 0];

        let result = survexp(time, age, year, &rt, Some(sex), None, Some("hakulinen")).unwrap();

        assert_eq!(result.n, 3);
        assert!(!result.time.is_empty());
        for s in &result.surv {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_survexp_empty() {
        let rt = create_test_ratetable();

        let result = survexp(vec![], vec![], vec![], &rt, None, None, None).unwrap();

        assert_eq!(result.n, 0);
        assert!(result.time.is_empty());
    }

    #[test]
    fn cox_aggregate_computes_weighted_ederer_curves_and_group_sizes() {
        let result = survexp_cox_aggregate(
            vec![1.0, 2.0],
            vec![vec![0.9, 0.8], vec![0.8, 0.6], vec![0.7, 0.5]],
            vec![vec![0.1, 0.2], vec![0.2, 0.4], vec![0.3, 0.6]],
            vec![2.0, 2.0, 2.0],
            vec![1, 1, 3],
            vec![1.0, 3.0, 2.0],
            "ederer",
        )
        .unwrap();

        assert_eq!(result.1, Some(vec![2, 0, 1]));
        assert!((result.0[0][0] - 0.825).abs() < 1e-12);
        assert!((result.0[0][1] - 0.65).abs() < 1e-12);
        assert_eq!(result.0[1], vec![0.0, 0.0]);
        assert_eq!(result.0[2], vec![0.7, 0.5]);
    }

    #[test]
    fn cox_aggregate_computes_hakulinen_and_conditional_hazards() {
        let time = vec![1.0, 2.0];
        let cumhaz: Vec<Vec<f64>> = vec![vec![0.1, 0.3], vec![0.2, 0.5]];
        let surv = cumhaz
            .iter()
            .map(|curve| curve.iter().map(|&value| (-value).exp()).collect())
            .collect::<Vec<Vec<f64>>>();
        let call = |method| {
            survexp_cox_aggregate(
                time.clone(),
                surv.clone(),
                cumhaz.clone(),
                vec![2.0, 2.0],
                vec![1, 1],
                vec![1.0, 1.0],
                method,
            )
            .unwrap()
            .0
        };

        let conditional = call("conditional");
        assert!((conditional[0][0] - (-0.15_f64).exp()).abs() < 1e-12);
        assert!((conditional[0][1] - (-0.4_f64).exp()).abs() < 1e-12);

        let weighted_increment = (0.2 * (-0.1_f64).exp() + 0.3 * (-0.2_f64).exp())
            / ((-0.1_f64).exp() + (-0.2_f64).exp());
        let hakulinen = call("hakulinen");
        assert!((hakulinen[0][0] - (-0.15_f64).exp()).abs() < 1e-12);
        assert!((hakulinen[0][1] - (-0.15 - weighted_increment).exp()).abs() < 1e-12);

        let signed = survexp_cox_aggregate(
            time,
            surv,
            cumhaz,
            vec![2.0, 1.0],
            vec![1, 1],
            vec![1.0, -2.0],
            "conditional",
        )
        .unwrap()
        .0;
        assert!((signed[0][0] - (-0.3_f64).exp()).abs() < 1e-12);
        assert!((signed[0][1] - (-0.5_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn cox_aggregate_validates_groups_finite_weights_and_curve_shapes() {
        Python::initialize();
        let call = |group, weights, surv| {
            survexp_cox_aggregate(
                vec![1.0],
                surv,
                vec![vec![0.1]],
                vec![1.0],
                group,
                weights,
                "ederer",
            )
        };

        assert!(call(vec![0], vec![1.0], vec![vec![0.9]]).is_err());
        assert!(call(vec![1], vec![f64::NAN], vec![vec![0.9]]).is_err());
        assert!(call(vec![1], vec![1.0], vec![vec![0.9, 0.8]]).is_err());

        let zero_weight = call(vec![1], vec![0.0], vec![vec![0.9]]).unwrap();
        assert!(zero_weight.0[0][0].is_nan());
    }

    #[test]
    fn coordinate_survexp_matches_specialized_age_year_sex_path() {
        let rt = create_test_ratetable();
        let time = vec![180.0, 730.5, 1095.75];
        let age = vec![14610.0, 18262.5, 25567.5];
        let year = vec![1999.0, 2000.0, 2001.0];
        let sex = vec![0, 1, 0];
        let times = vec![90.0, 365.25, 730.5];
        let coordinates = HashMap::from([
            ("age".to_string(), age.clone()),
            ("year".to_string(), year.clone()),
            (
                "sex".to_string(),
                sex.iter().map(|&value| f64::from(value)).collect(),
            ),
        ]);

        for method in ["ederer", "hakulinen", "conditional", "individual"] {
            let specialized = survexp(
                time.clone(),
                age.clone(),
                year.clone(),
                &rt,
                Some(sex.clone()),
                Some(times.clone()),
                Some(method),
            )
            .unwrap();
            let generic = survexp_from_coords(
                time.clone(),
                &rt,
                coordinates.clone(),
                Some(times.clone()),
                Some(method),
            )
            .unwrap();
            assert_eq!(generic.time, specialized.time);
            assert_eq!(generic.n_risk, specialized.n_risk);
            for (actual, expected) in generic.surv.iter().zip(&specialized.surv) {
                assert!((actual - expected).abs() < 1e-12);
            }
            for (actual, expected) in generic.cumhaz.iter().zip(&specialized.cumhaz) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }

        let specialized = survexp_individual(time.clone(), age, year, &rt, Some(sex)).unwrap();
        let generic = survexp_individual_from_coords(time, &rt, coordinates).unwrap();
        for (actual, expected) in generic.iter().zip(&specialized) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn coordinate_survexp_uses_custom_factor_and_continuous_dimensions() {
        let ratetable = RateTable::new(
            vec![
                RateDimension::new("age".to_string(), DimType::Age, vec![0.0, 100.0], None),
                RateDimension::new(
                    "year".to_string(),
                    DimType::Year,
                    vec![2000.0, 2010.0],
                    None,
                ),
                RateDimension::new(
                    "region".to_string(),
                    DimType::Factor,
                    vec![],
                    Some(vec!["urban".to_string(), "rural".to_string()]),
                ),
                RateDimension::new(
                    "exposure".to_string(),
                    DimType::Continuous,
                    vec![0.0, 1.0, 2.0],
                    None,
                ),
            ],
            vec![0.001, 0.002, 0.003, 0.004],
            None,
        )
        .unwrap();
        let coordinates = HashMap::from([
            ("age".to_string(), vec![20.0]),
            ("year".to_string(), vec![2001.0]),
            ("region".to_string(), vec![1.0]),
            ("exposure".to_string(), vec![1.5]),
        ]);

        let actual = survexp_individual_from_coords(vec![10.0], &ratetable, coordinates).unwrap();
        assert!((actual[0] - (-0.04_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn coordinate_survexp_validates_named_column_contract() {
        Python::initialize();
        let ratetable = create_test_ratetable();
        let valid = HashMap::from([
            ("age".to_string(), vec![20.0]),
            ("year".to_string(), vec![2001.0]),
            ("sex".to_string(), vec![0.0]),
        ]);

        let mut missing = valid.clone();
        missing.remove("year");
        assert!(survexp_individual_from_coords(vec![10.0], &ratetable, missing).is_err());

        let mut ragged = valid.clone();
        ragged.insert("age".to_string(), vec![20.0, 30.0]);
        assert!(survexp_individual_from_coords(vec![10.0], &ratetable, ragged).is_err());

        let mut unknown = valid.clone();
        unknown.insert("region".to_string(), vec![0.0]);
        assert!(survexp_individual_from_coords(vec![10.0], &ratetable, unknown).is_err());

        let mut fractional_factor = valid;
        fractional_factor.insert("sex".to_string(), vec![0.5]);
        assert!(survexp_individual_from_coords(vec![10.0], &ratetable, fractional_factor).is_err());
    }

    #[test]
    fn ederer_keeps_the_full_reference_cohort() {
        let rt = create_test_ratetable();
        let time = vec![180.0, 1095.0];
        let age = vec![14610.0, 25567.5];
        let year = vec![2000.0, 2000.0];
        let sex = vec![0, 1];
        let times = vec![365.25, 730.5];

        let ederer = survexp(
            time.clone(),
            age.clone(),
            year.clone(),
            &rt,
            Some(sex.clone()),
            Some(times.clone()),
            Some("ederer"),
        )
        .unwrap();
        let hakulinen = survexp(
            time,
            age.clone(),
            year.clone(),
            &rt,
            Some(sex.clone()),
            Some(times.clone()),
            Some("hakulinen"),
        )
        .unwrap();

        assert_eq!(ederer.method, "ederer");
        assert_eq!(ederer.n_risk, vec![2.0, 2.0]);
        assert_eq!(hakulinen.n_risk, vec![1.0, 1.0]);
        for (index, eval_time) in times.into_iter().enumerate() {
            let individual = survexp_individual(
                vec![eval_time, eval_time],
                age.clone(),
                year.clone(),
                &rt,
                Some(sex.clone()),
            )
            .unwrap();
            let expected = individual.iter().sum::<f64>() / individual.len() as f64;
            assert!((ederer.surv[index] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn survexp_default_sex_matches_explicit_zero_sex() {
        let rt = create_test_ratetable();

        let time = vec![365.0, 730.0, 1095.0];
        let age = vec![18250.0, 21900.0, 25550.0];
        let year = vec![2000.0, 2000.0, 2000.0];
        let times = vec![365.0, 730.0, 1095.0];

        for method in ["ederer", "hakulinen", "conditional", "individual"] {
            let default_result = survexp(
                time.clone(),
                age.clone(),
                year.clone(),
                &rt,
                None,
                Some(times.clone()),
                Some(method),
            )
            .unwrap();
            let explicit_zero_result = survexp(
                time.clone(),
                age.clone(),
                year.clone(),
                &rt,
                Some(vec![0; time.len()]),
                Some(times.clone()),
                Some(method),
            )
            .unwrap();

            assert_eq!(default_result.time, explicit_zero_result.time);
            assert_eq!(default_result.surv, explicit_zero_result.surv);
            assert_eq!(default_result.n_risk, explicit_zero_result.n_risk);
            assert_eq!(default_result.cumhaz, explicit_zero_result.cumhaz);
            assert_eq!(default_result.method, explicit_zero_result.method);
            assert_eq!(default_result.n, explicit_zero_result.n);
        }

        let default_individual =
            survexp_individual(time.clone(), age.clone(), year.clone(), &rt, None).unwrap();
        let explicit_zero_individual =
            survexp_individual(time, age, year, &rt, Some(vec![0; 3])).unwrap();
        assert_eq!(default_individual, explicit_zero_individual);
    }

    #[test]
    fn parallel_individual_survival_matches_scalar_boundary() {
        let rt = create_test_ratetable();
        let n = PARALLEL_THRESHOLD_XLARGE + 7;
        let time = (0..n)
            .map(|index| 100.0 + (index % 1200) as f64)
            .collect::<Vec<_>>();
        let age = (0..n)
            .map(|index| 1000.0 + (index % 90) as f64 * 365.25)
            .collect::<Vec<_>>();
        let year = (0..n)
            .map(|index| 1990.0 + (index % 30) as f64)
            .collect::<Vec<_>>();
        let sex = (0..n).map(|index| (index % 2) as i32).collect::<Vec<_>>();

        let actual = survexp_individual(
            time.clone(),
            age.clone(),
            year.clone(),
            &rt,
            Some(sex.clone()),
        )
        .unwrap();
        let expected = (0..n)
            .map(|index| {
                rt.expected_survival(
                    age[index],
                    age[index] + time[index],
                    year[index],
                    Some(sex[index]),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn incremental_curve_averages_match_direct_integration_across_batches() {
        let rt = create_test_ratetable();
        let n = SUBJECT_BATCH_SIZE + 3;
        let time = (0..n)
            .map(|index| 300.0 + (index % 9) as f64 * 100.0)
            .collect::<Vec<_>>();
        let age = (0..n)
            .map(|index| 15000.0 + (index % 13) as f64 * 2500.0)
            .collect::<Vec<_>>();
        let year = (0..n)
            .map(|index| 1995.0 + (index % 17) as f64)
            .collect::<Vec<_>>();
        let sex = (0..n).map(|index| (index % 2) as i32).collect::<Vec<_>>();
        let eval_times = vec![180.0, 540.0, 900.0];

        for observed_risk_set in [false, true] {
            let averages = compute_curve_averages(
                &time,
                &age,
                &year,
                Some(&sex),
                &rt,
                &eval_times,
                observed_risk_set,
            );
            for (time_index, &eval_time) in eval_times.iter().enumerate() {
                let mut expected_survival = 0.0;
                let mut expected_hazard = 0.0;
                let mut expected_count = 0.0;
                for subject in 0..n {
                    if observed_risk_set && time[subject] < eval_time {
                        continue;
                    }
                    let hazard = rt
                        .cumulative_hazard(
                            age[subject],
                            age[subject] + eval_time,
                            year[subject],
                            Some(sex[subject]),
                        )
                        .unwrap();
                    expected_survival += (-hazard).exp();
                    expected_hazard += hazard;
                    expected_count += 1.0;
                }
                assert_eq!(averages.n_risk[time_index], expected_count);
                if expected_count > 0.0 {
                    assert!(
                        (averages.survival[time_index] - expected_survival / expected_count).abs()
                            < 1e-12
                    );
                    assert!(
                        (averages.mean_cumhaz[time_index] - expected_hazard / expected_count).abs()
                            < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn survexp_validates_public_inputs() {
        let rt = create_test_ratetable();

        assert!(
            survexp(
                vec![f64::NAN],
                vec![18250.0],
                vec![2000.0],
                &rt,
                None,
                None,
                None,
            )
            .expect_err("non-finite time should fail")
            .to_string()
            .contains("time contains non-finite")
        );
        assert!(
            survexp(vec![365.0], vec![-1.0], vec![2000.0], &rt, None, None, None,)
                .expect_err("negative age should fail")
                .to_string()
                .contains("age contains negative")
        );
        assert!(
            survexp(
                vec![365.0],
                vec![18250.0],
                vec![2000.0],
                &rt,
                Some(vec![-1]),
                None,
                None,
            )
            .expect_err("negative sex should fail")
            .to_string()
            .contains("sex values must be non-negative")
        );
        assert!(
            survexp(
                vec![365.0, 730.0],
                vec![18250.0, 21900.0],
                vec![2000.0, 2000.0],
                &rt,
                None,
                Some(vec![730.0, 365.0]),
                Some("conditional"),
            )
            .expect_err("unsorted eval times should fail")
            .to_string()
            .contains("times must be sorted")
        );
        assert!(
            survexp_individual(
                vec![365.0],
                vec![18250.0],
                vec![2000.0],
                &rt,
                Some(vec![0, 1]),
            )
            .expect_err("sex length mismatch should fail")
            .to_string()
            .contains("sex must have same length")
        );
    }
}
