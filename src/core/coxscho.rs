use crate::core::risk_sweep::StratumSweep;
use crate::core::strata_order::{SurvResponse, order_within_strata};
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use crate::internal::validation::validate_binary_i32;
use crate::regression::TieMethod;
use crate::scoring::validate_score_inputs;
use ndarray::ArrayView2;
use pyo3::prelude::*;

#[doc = " Schoenfeld residuals, one row per event in order of (stratum, time)."]
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxschoResiduals {
    #[doc = " Event time of each row."]
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[doc = " Row index into the input data of each event."]
    #[pyo3(get)]
    pub index: Vec<usize>,
    #[doc = " Stratum label of each event."]
    #[pyo3(get)]
    pub strata: Vec<i32>,
    #[doc = " `n_events x p` residual matrix `x - xbar(t)` (unweighted)."]
    #[pyo3(get)]
    pub residuals: Vec<Vec<f64>>,
}

pub fn schoenfeld_residuals(
    response: SurvResponse<'_>,
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    method: TieMethod,
) -> SurvivalResult<CoxschoResiduals> {
    match response {
        SurvResponse::Right(data) => SurvivalData::validate_parts(&data.time, &data.status)?,
        SurvResponse::Counting(data) => {
            CountingProcessData::validate_parts(&data.start, &data.stop, &data.event)?;
        }
    }
    let start = response.start();
    let stop = response.stop();
    let event = response.status();
    let n = stop.len();
    validate_binary_i32(event, "event")?;
    validate_score_inputs(n, covariates, score, weights, strata)?;
    method.reject_exact("schoenfeld")?;
    let unit = vec![1.0; n];
    let zero = vec![0; n];
    Ok(coxscho(
        start,
        stop,
        event,
        covariates,
        score,
        weights.unwrap_or(&unit),
        strata.unwrap_or(&zero),
        method,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn coxscho(
    start: Option<&[f64]>,
    stop: &[f64],
    event: &[i32],
    covariates: ArrayView2<'_, f64>,
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> CoxschoResiduals {
    let nvar = covariates.ncols();
    let steps_of = |ndead: usize| if method.is_efron() { ndead } else { 1 };
    let order = order_within_strata(strata, |a, b| stop[a].total_cmp(&stop[b]));
    let mut time = Vec::new();
    let mut index = Vec::new();
    let mut strata_out = Vec::new();
    let mut residuals = Vec::new();
    let mut mean = vec![0.0; nvar];
    let mut first = 0;
    while first < order.len() {
        let label = strata[order[first]];
        let last = first + order[first..].partition_point(|&row| strata[row] == label);
        let sweep = StratumSweep {
            stop,
            entry: start,
            status: event,
            x: covariates,
            weights,
            risk: score,
            rows: &order[first..last],
            second_moments: false,
        };
        let mut per_stratum: Vec<(usize, Vec<f64>)> = Vec::new();
        sweep.for_each_death_time(|death| {
            let steps = steps_of(death.ndead());
            mean.fill(0.0);
            for j in 0..steps {
                let denom = death.efron_denom(j) * steps as f64;
                for (i, value) in mean.iter_mut().enumerate() {
                    *value += death.efron_a(j, i) / denom;
                }
            }
            for &row in death.deaths.iter().rev() {
                per_stratum.push((
                    row,
                    (0..nvar).map(|i| covariates[(row, i)] - mean[i]).collect(),
                ));
            }
        });
        per_stratum.reverse();
        for (row, values) in per_stratum {
            time.push(stop[row]);
            index.push(row);
            strata_out.push(strata[row]);
            residuals.push(values);
        }
        first = last;
    }
    CoxschoResiduals {
        time,
        index,
        strata: strata_out,
        residuals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
    use ndarray::array;

    #[test]
    fn residual_is_covariate_minus_risk_weighted_mean() {
        let stop = [3.0, 1.0, 2.0, 2.0];
        let event = [1, 1, 0, 1];
        let covar = array![[1.0], [2.0], [3.0], [4.0]];
        let score = [1.0, 2.0, 0.5, 1.5];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        assert_eq!(out.time, vec![1.0, 2.0, 3.0]);
        assert_eq!(out.index, vec![1, 3, 0]);
        let xbar1 = (1.0 + 4.0 + 1.5 + 6.0) / 5.0;
        let xbar2 = (1.0 + 1.5 + 6.0) / 3.0;
        let expected = [2.0 - xbar1, 4.0 - xbar2, 1.0 - 1.0];
        for (row, expected) in expected.iter().enumerate() {
            assert!((out.residuals[row][0] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn efron_averages_the_tied_pseudo_risk_sets() {
        let stop = [1.0, 1.0, 2.0];
        let event = [1, 1, 0];
        let covar = array![[0.0], [1.0], [2.0]];
        let score = [1.0, 1.0, 1.0];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            None,
            None,
            TieMethod::Efron,
        )
        .unwrap();
        let mean = (1.0 + 1.25) / 2.0;
        assert!((out.residuals[0][0] - (0.0 - mean)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (1.0 - mean)).abs() < 1e-12);
    }

    #[test]
    fn strata_and_weights_enter_the_means() {
        let stop = [1.0, 2.0, 1.0, 2.0];
        let event = [1, 0, 1, 0];
        let covar = array![[1.0], [3.0], [10.0], [20.0]];
        let score = [1.0; 4];
        let weights = [1.0, 3.0, 1.0, 1.0];
        let data = SurvivalData::try_new(stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Right(&data),
            covar.view(),
            &score,
            Some(&weights),
            Some(&[1, 1, 2, 2]),
            TieMethod::Breslow,
        )
        .unwrap();
        assert_eq!(out.strata, vec![1, 2]);
        assert!((out.residuals[0][0] - (1.0 - (1.0 + 9.0) / 4.0)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (10.0 - 15.0)).abs() < 1e-12);
    }

    #[test]
    fn counting_process_start_times_limit_the_risk_set() {
        let start = [0.0, 1.5, 0.0];
        let stop = [1.0, 3.0, 3.0];
        let event = [1, 1, 0];
        let covar = array![[0.0], [2.0], [4.0]];
        let data =
            CountingProcessData::try_new(start.to_vec(), stop.to_vec(), event.to_vec()).unwrap();
        let out = schoenfeld_residuals(
            SurvResponse::Counting(&data),
            covar.view(),
            &[1.0; 3],
            None,
            None,
            TieMethod::Breslow,
        )
        .unwrap();
        assert!((out.residuals[0][0] - (0.0 - 2.0)).abs() < 1e-12);
        assert!((out.residuals[1][0] - (2.0 - 3.0)).abs() < 1e-12);
    }
}
