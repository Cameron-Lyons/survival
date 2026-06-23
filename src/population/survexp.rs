use super::ratetable::RateTable;
use crate::constants::same_time;
use crate::internal::validation::{validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Result of expected survival calculation
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvExpResult {
    /// Time points
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Expected survival probabilities
    #[pyo3(get)]
    pub surv: Vec<f64>,
    /// Number at risk at each time
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    /// Cumulative expected hazard
    #[pyo3(get)]
    pub cumhaz: Vec<f64>,
    /// Method used for calculation
    #[pyo3(get)]
    pub method: String,
    /// Number of subjects
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

/// Compute expected survival based on population mortality.
///
/// This function computes expected survival for a cohort based on external
/// population mortality rates (e.g., national mortality tables).
///
/// # Arguments
/// * `time` - Follow-up times for each subject
/// * `age` - Age at baseline for each subject (in days or years depending on ratetable)
/// * `year` - Calendar year at baseline for each subject
/// * `ratetable` - Population mortality rate table
/// * `sex` - Optional sex indicator for each subject
/// * `times` - Optional times at which to compute expected survival
/// * `method` - Method: "hakulinen", "conditional", or "individual" (default: "hakulinen")
///
/// # Returns
/// * `SurvExpResult` with expected survival estimates
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

    let sex_vec = sex.unwrap_or_else(|| vec![0; n]);
    validate_sex(&sex_vec, n)?;

    let calc_method = method.unwrap_or("hakulinen");
    if !["hakulinen", "conditional", "individual"].contains(&calc_method) {
        return Err(value_error(
            "method must be 'hakulinen', 'conditional', or 'individual'",
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
        "hakulinen" => compute_hakulinen(&time, &age, &year, &sex_vec, ratetable, &eval_times),
        "conditional" => compute_conditional(&time, &age, &year, &sex_vec, ratetable, &eval_times),
        "individual" => compute_individual(&time, &age, &year, &sex_vec, ratetable, &eval_times),
        _ => compute_hakulinen(&time, &age, &year, &sex_vec, ratetable, &eval_times),
    }
}

/// Hakulinen method: expected survival for a cohort
fn compute_hakulinen(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: &[i32],
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    let n = time.len();

    let results: Vec<(f64, f64, f64)> = eval_times
        .par_iter()
        .map(|&eval_t| {
            let (total_surv, total_cumhaz, count) = (0..n)
                .filter(|&i| time[i] >= eval_t)
                .map(|i| {
                    let age_at_eval = age[i] + eval_t;
                    let exp_surv = ratetable
                        .expected_survival(age[i], age_at_eval, year[i], Some(sex[i]))
                        .unwrap_or(1.0);
                    let exp_cumhaz = ratetable
                        .cumulative_hazard(age[i], age_at_eval, year[i], Some(sex[i]))
                        .unwrap_or(0.0);
                    (exp_surv, exp_cumhaz, 1.0)
                })
                .fold((0.0, 0.0, 0.0), |acc, x| {
                    (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
                });

            let surv = if count > 0.0 { total_surv / count } else { 0.0 };
            let cumhaz = if count > 0.0 {
                total_cumhaz / count
            } else {
                0.0
            };
            (surv, cumhaz, count)
        })
        .collect();

    let surv: Vec<f64> = results.iter().map(|r| r.0).collect();
    let cumhaz: Vec<f64> = results.iter().map(|r| r.1).collect();
    let n_risk: Vec<f64> = results.iter().map(|r| r.2).collect();

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv,
        n_risk,
        cumhaz,
        method: "hakulinen".to_string(),
        n,
    })
}

/// Conditional method: expected survival conditional on observed survival
fn compute_conditional(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: &[i32],
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
            let year_start = year[i] + prev_time / 365.25;

            let interval_hazard = ratetable
                .cumulative_hazard(age_start, age_end, year_start, Some(sex[i]))
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

/// Individual method: individual expected survival for each subject
fn compute_individual(
    time: &[f64],
    age: &[f64],
    year: &[f64],
    sex: &[i32],
    ratetable: &RateTable,
    eval_times: &[f64],
) -> PyResult<SurvExpResult> {
    let n = time.len();
    let n_times = eval_times.len();

    let individual_surv: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            eval_times
                .iter()
                .map(|&eval_t| {
                    let age_at_eval = age[i] + eval_t;
                    ratetable
                        .expected_survival(age[i], age_at_eval, year[i], Some(sex[i]))
                        .unwrap_or(1.0)
                })
                .collect()
        })
        .collect();

    let mut surv = vec![0.0; n_times];
    let mut cumhaz = vec![0.0; n_times];
    let mut n_risk = vec![0.0; n_times];

    for t_idx in 0..n_times {
        let eval_t = eval_times[t_idx];
        let mut total = 0.0;
        let mut count = 0.0;

        for i in 0..n {
            if time[i] >= eval_t {
                total += individual_surv[i][t_idx];
                count += 1.0;
            }
        }

        n_risk[t_idx] = count;
        surv[t_idx] = if count > 0.0 { total / count } else { 0.0 };
        cumhaz[t_idx] = if surv[t_idx] > 0.0 {
            -surv[t_idx].ln()
        } else {
            f64::INFINITY
        };
    }

    Ok(SurvExpResult {
        time: eval_times.to_vec(),
        surv,
        n_risk,
        cumhaz,
        method: "individual".to_string(),
        n,
    })
}

/// Compute individual expected survival for each subject
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

    let sex_vec = sex.unwrap_or_else(|| vec![0; n]);
    validate_sex(&sex_vec, n)?;

    let mut expected = Vec::with_capacity(n);

    for i in 0..n {
        let age_end = age[i] + time[i];
        let exp_s = ratetable
            .expected_survival(age[i], age_end, year[i], Some(sex_vec[i]))
            .unwrap_or(1.0);
        expected.push(exp_s);
    }

    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::ratetable::create_simple_ratetable;

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
