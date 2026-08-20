use super::ratetable::RateTable;
use crate::constants::{PARALLEL_THRESHOLD_XLARGE, same_time};
use crate::internal::validation::{validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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
