use super::ratetable::RateTable;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Result of expected survival calculation
#[derive(Debug, Clone)]
#[pyclass]
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

    if age.len() != n || year.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, age, and year must have same length",
        ));
    }

    let sex_vec = sex.unwrap_or_else(|| vec![0; n]);
    if sex_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sex must have same length as time",
        ));
    }

    let calc_method = method.unwrap_or("hakulinen");
    if !["hakulinen", "conditional", "individual"].contains(&calc_method) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
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
            unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_times.dedup();
            unique_times
        }
    };

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
        let at_risk: Vec<usize> = (0..n).filter(|&i| time[i] >= eval_t).collect();
        n_risk[t_idx] = at_risk.len() as f64;

        if at_risk.is_empty() {
            surv[t_idx] = prev_surv;
            cumhaz[t_idx] = if prev_surv > 0.0 {
                -prev_surv.ln()
            } else {
                f64::INFINITY
            };
            continue;
        }

        let mut total_hazard = 0.0;
        for &i in &at_risk {
            let age_start = age[i] + prev_time;
            let age_end = age[i] + eval_t;
            let year_start = year[i] + prev_time / 365.25;

            let interval_hazard = ratetable
                .cumulative_hazard(age_start, age_end, year_start, Some(sex[i]))
                .unwrap_or(0.0);
            total_hazard += interval_hazard;
        }

        let avg_hazard = total_hazard / at_risk.len() as f64;
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

    if age.len() != n || year.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, age, and year must have same length",
        ));
    }

    let sex_vec = sex.unwrap_or_else(|| vec![0; n]);

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
    use crate::specialized::ratetable::create_simple_ratetable;

    fn create_test_ratetable() -> RateTable {
        // Simple rate table for testing
        let age_breaks = vec![0.0, 36500.0, 73000.0]; // 0-100, 100-200 years in days
        let year_breaks = vec![1990.0, 2020.0];

        // Rates per day (very simplified)
        let rates_male = vec![0.00001, 0.00005];
        let rates_female = vec![0.000008, 0.00004];

        create_simple_ratetable(age_breaks, year_breaks, rates_male, rates_female).unwrap()
    }

    #[test]
    fn test_survexp_basic() {
        let rt = create_test_ratetable();

        let time = vec![365.0, 730.0, 1095.0];
        let age = vec![18250.0, 21900.0, 25550.0]; // ~50, 60, 70 years in days
        let year = vec![2000.0, 2000.0, 2000.0];
        let sex = vec![0, 1, 0];

        let result = survexp(time, age, year, &rt, Some(sex), None, Some("hakulinen")).unwrap();

        assert_eq!(result.n, 3);
        assert!(!result.time.is_empty());
        // Expected survival should be between 0 and 1
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
}
