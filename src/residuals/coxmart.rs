use crate::core::strata_order::{last_of_run, order_within_strata};
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::{CoxMartInput, SurvivalData, Weights};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::TieMethod;

pub fn coxmart(input: &CoxMartInput, method: TieMethod) -> SurvivalResult<Vec<f64>> {
    let n = input.survival.len();
    SurvivalData::validate_parts(&input.survival.time, &input.survival.status)?;
    validate_binary_i32(&input.survival.status, "status")?;
    validate_length(n, input.score.len(), "score")?;
    validate_finite(&input.score, "score")?;
    if let Some(weights) = &input.weights {
        validate_length(n, weights.values.len(), "weights")?;
        Weights::validate_values(&weights.values)?;
    }
    if let Some(strata) = &input.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    let weights = input.weights_or_unit_cow();
    let strata = input.strata_or_default_cow();
    let time = &input.survival.time;
    let order = order_within_strata(&strata, |a, b| time[a].total_cmp(&time[b]));
    Ok(coxmart_rows(
        &order,
        time,
        &input.survival.status,
        &input.score,
        &weights,
        &strata,
        method,
    ))
}

pub(crate) fn coxmart_rows(
    order: &[usize],
    time: &[f64],
    status: &[i32],
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Vec<f64> {
    let sorted_time: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = order.iter().map(|&i| status[i]).collect();
    let sorted_score: Vec<f64> = order.iter().map(|&i| score[i]).collect();
    let sorted_weights: Vec<f64> = order.iter().map(|&i| weights[i]).collect();
    let sorted_strata: Vec<i32> = order.iter().map(|&i| strata[i]).collect();

    let sorted = coxmart_sorted(
        &sorted_time,
        &sorted_status,
        &sorted_score,
        &sorted_weights,
        &sorted_strata,
        method,
    );
    let mut resid = vec![0.0; order.len()];
    for (sorted_index, &original) in order.iter().enumerate() {
        resid[original] = sorted[sorted_index];
    }
    resid
}

pub(crate) fn coxmart_sorted(
    time: &[f64],
    status: &[i32],
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Vec<f64> {
    let n = time.len();
    let mut expect = vec![0.0; n];
    if n == 0 {
        return expect;
    }
    let last = last_of_run(strata);

    let mut denom = 0.0;
    for i in (0..n).rev() {
        if last[i] {
            denom = 0.0;
        }
        denom += score[i] * weights[i];
        expect[i] = if i == 0 || last[i - 1] || time[i - 1] != time[i] {
            denom
        } else {
            0.0
        };
    }

    let mut deaths = 0.0;
    let mut wtsum = 0.0;
    let mut e_denom = 0.0;
    let mut hazard = 0.0;
    let mut lastone = 0;
    for i in 0..n {
        if expect[i] != 0.0 {
            denom = expect[i];
        }
        let status_i = f64::from(status[i]);
        expect[i] = status_i;
        deaths += status_i;
        wtsum += status_i * weights[i];
        e_denom += score[i] * status_i * weights[i];
        if last[i] || time[i + 1] != time[i] {
            if deaths < 2.0 || !method.is_efron() {
                hazard += wtsum / denom;
                for j in lastone..=i {
                    expect[j] -= score[j] * hazard;
                }
            } else {
                let mut temp = hazard;
                wtsum /= deaths;
                for j in 0..deaths as usize {
                    let downwt = j as f64 / deaths;
                    hazard += wtsum / (denom - e_denom * downwt);
                    temp += wtsum * (1.0 - downwt) / (denom - e_denom * downwt);
                }
                for j in lastone..=i {
                    if status[j] == 0 {
                        expect[j] = -score[j] * hazard;
                    } else {
                        expect[j] -= score[j] * temp;
                    }
                }
            }
            lastone = i + 1;
            deaths = 0.0;
            wtsum = 0.0;
            e_denom = 0.0;
        }
        if last[i] {
            hazard = 0.0;
        }
    }
    for j in lastone..n {
        expect[j] -= score[j] * hazard;
    }
    expect
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::typed_inputs::{SurvivalData, Weights};

    fn input(
        time: Vec<f64>,
        status: Vec<i32>,
        score: Vec<f64>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
    ) -> CoxMartInput {
        CoxMartInput::try_new(
            SurvivalData::try_new(time, status).unwrap(),
            score,
            weights.map(|w| Weights::try_new(w).unwrap()),
            strata,
        )
        .unwrap()
    }

    #[test]
    fn null_model_matches_nelson_aalen() {
        let resid = coxmart(
            &input(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1, 1, 0, 1],
                vec![1.0; 4],
                None,
                None,
            ),
            TieMethod::Breslow,
        )
        .unwrap();
        let cumhaz = [1.0 / 4.0, 1.0 / 4.0 + 1.0 / 3.0, 1.0 / 4.0 + 1.0 / 3.0];
        let expected = [
            1.0 - cumhaz[0],
            1.0 - cumhaz[1],
            -cumhaz[2],
            1.0 - (cumhaz[2] + 1.0),
        ];
        for (actual, expected) in resid.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn residuals_are_returned_in_input_order() {
        let sorted = coxmart(
            &input(
                vec![1.0, 2.0, 3.0],
                vec![1, 0, 1],
                vec![1.0, 2.0, 0.5],
                None,
                None,
            ),
            TieMethod::Efron,
        )
        .unwrap();
        let shuffled = coxmart(
            &input(
                vec![3.0, 1.0, 2.0],
                vec![1, 1, 0],
                vec![0.5, 1.0, 2.0],
                None,
                None,
            ),
            TieMethod::Efron,
        )
        .unwrap();
        assert_eq!(shuffled, vec![sorted[2], sorted[0], sorted[1]]);
    }

    #[test]
    fn strata_are_independent_risk_sets() {
        let two = coxmart(
            &input(
                vec![1.0, 2.0, 1.0, 2.0],
                vec![1, 0, 1, 1],
                vec![1.0, 3.0, 2.0, 1.0],
                None,
                Some(vec![1, 1, 2, 2]),
            ),
            TieMethod::Breslow,
        )
        .unwrap();
        let first = coxmart(
            &input(vec![1.0, 2.0], vec![1, 0], vec![1.0, 3.0], None, None),
            TieMethod::Breslow,
        )
        .unwrap();
        let second = coxmart(
            &input(vec![1.0, 2.0], vec![1, 1], vec![2.0, 1.0], None, None),
            TieMethod::Breslow,
        )
        .unwrap();
        assert_eq!(two, [first, second].concat());
    }

    #[test]
    fn tied_deaths_efron_differs_from_breslow() {
        let data = input(
            vec![1.0, 1.0, 1.0, 2.0],
            vec![1, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 0.5],
            Some(vec![1.0, 2.0, 1.0, 1.0]),
            None,
        );
        let breslow = coxmart(&data, TieMethod::Breslow).unwrap();
        let efron = coxmart(&data, TieMethod::Efron).unwrap();
        assert!(
            breslow
                .iter()
                .zip(&efron)
                .any(|(a, b)| (a - b).abs() > 1e-9)
        );
        let weights = [1.0, 2.0, 1.0, 1.0];
        for resid in [&breslow, &efron] {
            let total: f64 = resid.iter().zip(weights).map(|(r, w)| r * w).sum();
            assert!(total.abs() < 1e-12, "weighted sum {total}");
        }
    }

    #[test]
    fn rejects_multi_state_status() {
        let err = coxmart(
            &input(vec![1.0, 2.0], vec![1, 2], vec![1.0, 1.0], None, None),
            TieMethod::Breslow,
        )
        .unwrap_err();
        assert!(err.to_string().contains("status"));
    }
}
