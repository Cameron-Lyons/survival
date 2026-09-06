use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum ResidualType {
    Survival,
    Cumhaz,
    Rmst,
}

impl ResidualType {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "survival" => Ok(Self::Survival),
            "cumhaz" => Ok(Self::Cumhaz),
            "rmst" => Ok(Self::Rmst),
            _ => Err(PyValueError::new_err(
                "type_ must be 'survival', 'cumhaz', or 'rmst'",
            )),
        }
    }
}

/// Observation-level infinitesimal jackknife derivatives at requested times.
///
/// The supplied tables describe one fitted ordinary survival curve. In particular,
/// risk sets may contain case weights; this function does not refit the curve or
/// multiply the resulting derivatives by observation weights. Counting-process
/// observations contribute on the open-left interval `(start, time]`.
///
/// This follows `survival:::rsurvpart1`, including its approximate derivatives for
/// tied-event hazard corrections: hazard increments come from the fitted cumhaz
/// table, rather than being reconstructed from event counts. Event-only prefix
/// sums require O(e) workspace. After reading the fitted table, evaluation takes
/// O((n + m) log(e) + n*m + e) time and O(e + m) auxiliary space, including
/// restricted mean survival time derivatives.
#[pyfunction]
#[pyo3(signature = (time, status, curve_time, n_risk, n_event, survival, cumhaz, eval_times, type_, start=None, stype=1))]
#[allow(clippy::too_many_arguments)]
pub fn survfit_residuals_at_times(
    time: Vec<f64>,
    status: Vec<i32>,
    curve_time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    survival: Vec<f64>,
    cumhaz: Vec<f64>,
    eval_times: Vec<f64>,
    type_: &str,
    start: Option<Vec<f64>>,
    stype: i32,
) -> PyResult<Vec<Vec<f64>>> {
    let kind = ResidualType::parse(type_)?;
    if stype != 1 && stype != 2 {
        return Err(PyValueError::new_err("stype must be 1 or 2"));
    }
    validate_length(time.len(), status.len(), "status")?;
    validate_binary_i32(&status, "status")?;
    validate_finite(&time, "time")?;
    validate_finite(&eval_times, "eval_times")?;
    validate_finite(&curve_time, "curve_time")?;
    for (values, name) in [
        (&n_risk, "n_risk"),
        (&n_event, "n_event"),
        (&survival, "survival"),
        (&cumhaz, "cumhaz"),
    ] {
        validate_length(curve_time.len(), values.len(), name)?;
        validate_finite(values, name)?;
    }
    if curve_time.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(PyValueError::new_err(
            "curve_time must be strictly increasing",
        ));
    }
    if n_risk.iter().any(|&value| value < 0.0) || n_event.iter().any(|&value| value < 0.0) {
        return Err(PyValueError::new_err(
            "n_risk and n_event must be nonnegative",
        ));
    }
    if survival.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(PyValueError::new_err("survival must be between 0 and 1"));
    }
    if cumhaz.iter().any(|&value| value < 0.0) || cumhaz.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(PyValueError::new_err(
            "cumhaz must be nonnegative and nondecreasing",
        ));
    }
    if let Some(start) = &start {
        validate_length(time.len(), start.len(), "start")?;
        validate_finite(start, "start")?;
        if start.iter().zip(&time).any(|(start, stop)| start >= stop) {
            return Err(PyValueError::new_err("start must be less than time"));
        }
    }

    let event_count = n_event.iter().filter(|&&count| count > 0.0).count();
    let last_positive_survival =
        (0..curve_time.len()).rfind(|&i| n_event[i] > 0.0 && survival[i] > 0.0);
    let mut event_time = Vec::with_capacity(event_count);
    // Index zero represents the curve before its first event. All remaining
    // vectors use the same one-based event indices as R's findInterval results.
    let mut surv = Vec::with_capacity(event_count + 1);
    let mut dd = Vec::with_capacity(event_count + 1);
    let mut hsum = Vec::with_capacity(event_count + 1);
    let mut area = Vec::with_capacity(event_count + 1);
    let mut area_hsum = Vec::with_capacity(event_count + 1);
    surv.push(1.0);
    dd.push(0.0);
    hsum.push(0.0);
    area.push(0.0);
    area_hsum.push(0.0);
    let mut previous_cumhaz = 0.0;
    for i in 0..curve_time.len() {
        if n_event[i] == 0.0 {
            continue;
        }
        if n_risk[i] == 0.0 {
            return Err(PyValueError::new_err(
                "n_risk must be positive at event times",
            ));
        }
        let hazard = cumhaz[i] - previous_cumhaz;
        previous_cumhaz = cumhaz[i];
        let denominator = if stype == 2 || kind == ResidualType::Cumhaz {
            n_risk[i]
        } else if kind == ResidualType::Survival && hazard == 1.0 {
            // R substitutes one for 1-h at a terminal event so multiplying by
            // the zero fitted survival gives a finite zero derivative.
            n_risk[i]
        } else {
            n_risk[i] * (1.0 - hazard)
        };
        let mut derivative = 1.0 / denominator;
        if kind == ResidualType::Rmst
            && (!derivative.is_finite() || last_positive_survival.is_none_or(|last| i > last))
        {
            // After survival reaches its final zero, all RMST area differences
            // are zero. Omit these terms even if subtraction in the fitted
            // cumulative hazard rounded a terminal unit hazard slightly below
            // one: otherwise enormous finite dd values destroy the prefix
            // differences although R's direct zero-area products remain zero.
            derivative = 0.0;
        }
        let previous = event_time.len();
        let event_area = if previous == 0 {
            0.0
        } else {
            area[previous] + surv[previous] * (curve_time[i] - event_time[previous - 1])
        };
        // Only differences of integrated survival enter RMST derivatives. Use
        // the first event as the integration origin to avoid cancellation of a
        // potentially large, constant area preceding that event.
        area.push(event_area);
        area_hsum.push(area_hsum[previous] + event_area * hazard * derivative);
        hsum.push(hsum[previous] + hazard * derivative);
        dd.push(derivative);
        surv.push(survival[i]);
        event_time.push(curve_time[i]);
    }

    let query_indices: Vec<usize> = eval_times
        .iter()
        .map(|query| event_time.partition_point(|event| event <= query))
        .collect();
    let query_area: Vec<f64> = eval_times
        .iter()
        .zip(&query_indices)
        .map(|(&query, &index)| {
            if index == 0 {
                0.0
            } else {
                area[index] + surv[index] * (query - event_time[index - 1])
            }
        })
        .collect();
    let mut result = Vec::with_capacity(time.len());
    for (row, &stop) in time.iter().enumerate() {
        let stop_index = event_time.partition_point(|&event| event <= stop);
        let start_index = start.as_ref().map_or(0, |start| {
            event_time.partition_point(|&event| event <= start[row])
        });
        let mut values = Vec::with_capacity(eval_times.len());
        for (column, &query_index) in query_indices.iter().enumerate() {
            let end = stop_index.min(query_index);
            let begin = start_index.min(query_index);
            let death = if status[row] == 1 && stop_index <= query_index {
                stop_index
            } else {
                0
            };
            let accumulated = hsum[end] - hsum[begin];
            let value = match kind {
                ResidualType::Cumhaz => dd[death] - accumulated,
                ResidualType::Survival => surv[query_index] * (accumulated - dd[death]),
                ResidualType::Rmst => {
                    query_area[column] * accumulated
                        - (area_hsum[end] - area_hsum[begin])
                        - (query_area[column] - area[death]) * dd[death]
                }
            };
            values.push(value);
        }
        result.push(values);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[Vec<f64>], expected: &[&[f64]]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(*expected) {
                assert!((actual - expected).abs() < 2e-14, "{actual} != {expected}");
            }
        }
    }

    // R: survfit(Surv(c(1,2,2,4,5,6), c(1,1,0,1,0,1)) ~ 1,
    //            weights=c(.5,2,1.5,1,3,2)), then survival:::rsurvpart1.
    fn weighted_residuals(type_: &str, stype: i32) -> Vec<Vec<f64>> {
        let cumhaz = vec![
            0.05,
            0.05 + 2.0 / 9.5,
            0.05 + 2.0 / 9.5 + 1.0 / 6.0,
            0.05 + 2.0 / 9.5 + 1.0 / 6.0,
            0.05 + 2.0 / 9.5 + 1.0 / 6.0 + 1.0,
        ];
        let survival = if stype == 1 {
            vec![0.95, 0.75, 0.625, 0.625, 0.0]
        } else {
            cumhaz.iter().map(|hazard| f64::exp(-hazard)).collect()
        };
        survfit_residuals_at_times(
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            vec![1, 1, 0, 1, 0, 1],
            vec![1.0, 2.0, 4.0, 5.0, 6.0],
            vec![10.0, 9.5, 6.0, 5.0, 2.0],
            vec![0.5, 2.0, 1.0, 0.0, 2.0],
            survival,
            cumhaz,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            type_,
            None,
            stype,
        )
        .unwrap()
    }

    #[test]
    fn weighted_fitted_survival_matches_r_and_terminal_event_is_finite() {
        let actual = weighted_residuals("survival", 1);
        assert_close(
            &actual,
            &[
                &[0.0, -0.095, -0.075, -0.075, -0.0625, -0.0625, 0.0, 0.0],
                &[0.0, 0.005, -0.075, -0.075, -0.0625, -0.0625, 0.0, 0.0],
                &[0.0, 0.005, 0.025, 0.025, 1.0 / 48.0, 1.0 / 48.0, 0.0, 0.0],
                &[0.0, 0.005, 0.025, 0.025, -1.0 / 12.0, -1.0 / 12.0, 0.0, 0.0],
                &[0.0, 0.005, 0.025, 0.025, 1.0 / 24.0, 1.0 / 24.0, 0.0, 0.0],
                &[0.0, 0.005, 0.025, 0.025, 1.0 / 24.0, 1.0 / 24.0, 0.0, 0.0],
            ],
        );
        let cumhaz = weighted_residuals("cumhaz", 1);
        assert!((cumhaz[3][4] - 0.111728224068944).abs() < 1e-14);
        assert!((cumhaz[5][6] + 0.0549384425977223).abs() < 1e-14);
    }

    #[test]
    fn weighted_rmst_matches_r_between_events_and_beyond_last_event() {
        let actual = weighted_residuals("rmst", 1);
        assert_close(
            &actual,
            &[
                &[0.0, 0.0, -0.095, -0.17, -0.245, -0.3075, -0.37, -0.37],
                &[0.0, 0.0, 0.005, -0.07, -0.145, -0.2075, -0.27, -0.27],
                &[
                    0.0,
                    0.0,
                    0.005,
                    0.03,
                    0.055,
                    0.0758333333333333,
                    0.0966666666666667,
                    0.0966666666666667,
                ],
                &[
                    0.0,
                    0.0,
                    0.005,
                    0.03,
                    0.055,
                    -0.0283333333333333,
                    -0.111666666666667,
                    -0.111666666666667,
                ],
                &[
                    0.0,
                    0.0,
                    0.005,
                    0.03,
                    0.055,
                    0.0966666666666667,
                    0.138333333333333,
                    0.138333333333333,
                ],
                &[
                    0.0,
                    0.0,
                    0.005,
                    0.03,
                    0.055,
                    0.0966666666666667,
                    0.138333333333333,
                    0.138333333333333,
                ],
            ],
        );
        let exponential = weighted_residuals("rmst", 2);
        assert!((exponential[0][7] + 0.383531918302462).abs() < 1e-14);
        assert!((exponential[3][7] + 0.125963117674264).abs() < 1e-14);
        let survival = weighted_residuals("survival", 2);
        assert!((survival[3][7] + 0.0268127191245322).abs() < 1e-14);
    }

    #[test]
    fn counting_start_is_open_and_rmst_matches_r() {
        // R: survfit(Surv(c(0,0,1,2,3),c(2,3,4,5,6),c(1,0,1,0,1))~1,
        //            weights=c(1,2,.5,1.5,1)). Includes entry exactly at an event.
        let actual = survfit_residuals_at_times(
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 0, 1, 0, 1],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.5, 4.0, 3.0, 2.5, 1.0],
            vec![1.0, 0.0, 0.5, 0.0, 1.0],
            vec![5.0 / 7.0, 5.0 / 7.0, 25.0 / 42.0, 25.0 / 42.0, 0.0],
            vec![
                2.0 / 7.0,
                2.0 / 7.0,
                2.0 / 7.0 + 1.0 / 6.0,
                2.0 / 7.0 + 1.0 / 6.0,
                2.0 / 7.0 + 1.0 / 6.0 + 1.0,
            ],
            vec![2.0, 3.0, 4.0, 6.0, 7.0],
            "rmst",
            Some(vec![0.0, 0.0, 1.0, 2.0, 3.0]),
            1,
        )
        .unwrap();
        assert_close(
            &actual,
            &[
                &[
                    0.0,
                    -0.204081632653061,
                    -0.408163265306122,
                    -0.748299319727891,
                    -0.748299319727891,
                ],
                &[
                    0.0,
                    0.0816326530612245,
                    0.163265306122449,
                    0.299319727891157,
                    0.299319727891157,
                ],
                &[
                    0.0,
                    0.0816326530612245,
                    0.163265306122449,
                    -0.0975056689342404,
                    -0.0975056689342404,
                ],
                &[0.0, 0.0, 0.0, 0.0793650793650794, 0.0793650793650794],
                &[0.0, 0.0, 0.0, 0.0793650793650794, 0.0793650793650794],
            ],
        );
    }

    #[test]
    fn rmst_terminal_hazard_roundoff_does_not_erase_earlier_residuals() {
        let terminal_cumhaz = 1.4999999999999998;
        assert_ne!(terminal_cumhaz - 0.5, 1.0);
        let result = survfit_residuals_at_times(
            vec![1.0, 2.0],
            vec![1, 1],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.0],
            vec![0.5, terminal_cumhaz],
            vec![1.0, 2.0, 3.0],
            "rmst",
            None,
            1,
        )
        .unwrap();
        assert_close(&result, &[&[0.0, -0.25, -0.25], &[0.0, 0.25, 0.25]]);
    }

    #[test]
    fn no_events_and_empty_query_preserve_output_shape() {
        for kind in ["survival", "cumhaz", "rmst"] {
            for query in [vec![], vec![0.0, 3.0]] {
                let result = survfit_residuals_at_times(
                    vec![1.0, 2.0],
                    vec![0, 0],
                    vec![1.0, 2.0],
                    vec![2.0, 1.0],
                    vec![0.0, 0.0],
                    vec![1.0, 1.0],
                    vec![0.0, 0.0],
                    query.clone(),
                    kind,
                    None,
                    1,
                )
                .unwrap();
                assert_eq!(result, vec![vec![0.0; query.len()]; 2]);
            }
        }
    }

    #[test]
    fn rejects_invalid_observations_and_tables() {
        let run = |status, table_time, risk, query, kind, start, stype| {
            survfit_residuals_at_times(
                vec![2.0],
                status,
                table_time,
                risk,
                vec![1.0],
                vec![0.0],
                vec![1.0],
                query,
                kind,
                start,
                stype,
            )
        };
        assert!(run(vec![], vec![2.0], vec![1.0], vec![2.0], "survival", None, 1).is_err());
        assert!(
            run(
                vec![2],
                vec![2.0],
                vec![1.0],
                vec![2.0],
                "survival",
                None,
                1
            )
            .is_err()
        );
        assert!(
            run(
                vec![1],
                vec![f64::NAN],
                vec![1.0],
                vec![2.0],
                "survival",
                None,
                1
            )
            .is_err()
        );
        assert!(
            run(
                vec![1],
                vec![2.0],
                vec![0.0],
                vec![2.0],
                "survival",
                None,
                1
            )
            .is_err()
        );
        assert!(
            run(
                vec![1],
                vec![2.0],
                vec![1.0],
                vec![f64::INFINITY],
                "survival",
                None,
                1
            )
            .is_err()
        );
        assert!(run(vec![1], vec![2.0], vec![1.0], vec![2.0], "unknown", None, 1).is_err());
        assert!(
            run(
                vec![1],
                vec![2.0],
                vec![1.0],
                vec![2.0],
                "survival",
                None,
                3
            )
            .is_err()
        );
        assert!(
            run(
                vec![1],
                vec![2.0],
                vec![1.0],
                vec![2.0],
                "survival",
                Some(vec![2.0]),
                1
            )
            .is_err()
        );
        assert!(
            run(
                vec![1],
                vec![2.0],
                vec![1.0],
                vec![2.0],
                "survival",
                Some(vec![f64::NAN]),
                1
            )
            .is_err()
        );
    }
}
