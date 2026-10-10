//! External Rust fitted-model callers may mutate public fields after fitting.
//! Predictions must reject inconsistent shapes even after baseline caching.

use ndarray::{Array2, array};
use survival::regression::{
    CoxNewData, CoxPHFit, CoxphData, CoxphOptions, PredictReference, SurvfitOptions, TieMethod,
};

fn ordinary_model() -> CoxPHFit {
    let data = CoxphData::try_new(
        vec![1., 1., 2., 3., 3., 4., 5., 5.],
        None,
        vec![1, 1, 0, 1, 1, 0, 1, 0],
        array![
            [0.2, 1.0],
            [0.8, 0.2],
            [0.4, 0.7],
            [1.1, 1.3],
            [0.7, 0.4],
            [0.3, 1.1],
            [1.3, 0.5],
            [0.5, 0.9]
        ],
        None,
        None,
        None,
    )
    .unwrap();
    CoxPHFit::fit(data, CoxphOptions::default()).unwrap()
}

fn check_prediction_outcomes(fit: &CoxPHFit, expect_errors: bool) {
    let check = |error| assert_eq!(error, expect_errors);
    let newdata = CoxNewData::try_new(
        array![[0.7, 0.8]],
        None,
        None,
        Some(vec![5.0]),
        Some(vec![0.0]),
    )
    .unwrap();
    let options = SurvfitOptions::default();
    check(fit.basehaz(true).is_err());
    check(fit.survfit(None, options).is_err());
    check(fit.survfit(Some(&newdata), options).is_err());
    check(
        fit.survfit(
            None,
            SurvfitOptions {
                ctype: Some(1),
                start_time: Some(0.5),
                ..options
            },
        )
        .is_err(),
    );
    check(fit.predict_survival_at(&[0.5, 3.0], None).is_err());
    check(fit.predict_survival_at(&[], None).is_err());
    check(fit.predict_survival_at(&[3.0], Some(&newdata)).is_err());
    check(
        fit.expected_survival(&newdata, &[0], &[1.0], None, Some(&[5.0]), "ederer")
            .is_err(),
    );
    check(fit.survfit_individual(&newdata, &[1], options).is_err());
    check(
        fit.predict_lp(None, true, PredictReference::Sample)
            .is_err(),
    );
    check(
        fit.predict_lp(None, false, PredictReference::Sample)
            .is_err(),
    );
    check(
        fit.predict_lp(Some(&newdata), true, PredictReference::Zero)
            .is_err(),
    );
    check(
        fit.predict_risk(Some(&newdata), false, PredictReference::Sample)
            .is_err(),
    );
    check(
        fit.predict_terms(None, true, PredictReference::Sample, &[vec![0, 1]])
            .is_err(),
    );
    check(
        fit.predict_terms_grouped(
            Some(&newdata),
            true,
            PredictReference::Sample,
            &[vec![0, 1]],
            &[7],
        )
        .is_err(),
    );
    check(fit.predict_expected(None, false).is_err());
    check(fit.predict_expected(None, true).is_err());
    check(fit.predict_expected(Some(&newdata), true).is_err());
    check(fit.predict_survival(None, true).is_err());
}

#[test]
fn inconsistent_fitted_dimensions_fail_before_predictions_with_or_without_cache() {
    for cached in [false, true] {
        let original = ordinary_model();
        if cached {
            original.basehaz(true).unwrap();
        }
        // Validate all control arguments on a separate clone, so this does
        // not warm the baselines used by the uncached mutation cases.
        check_prediction_outcomes(&original.clone(), false);
        for field in [
            "time",
            "status",
            "weights",
            "offset",
            "linear_predictors",
            "residuals",
            "entry",
            "strata",
            "design_rows",
            "n",
            "coefficients",
            "means",
            "design_columns",
            "variance_rows",
            "variance_columns",
        ] {
            let expected = if [
                "coefficients",
                "means",
                "design_columns",
                "variance_rows",
                "variance_columns",
            ]
            .contains(&field)
            {
                original.nvar()
            } else {
                original.n
            };
            for length in [0, expected - 1, expected + 1] {
                let mut fit = original.clone();
                match field {
                    "time" => fit.time.resize(length, 1.0),
                    "status" => fit.status.resize(length, 0),
                    "weights" => fit.weights.resize(length, 1.0),
                    "offset" => fit.offset.resize(length, 0.0),
                    "linear_predictors" => fit.linear_predictors.resize(length, 0.0),
                    "residuals" => fit.residuals.resize(length, 0.0),
                    "entry" => fit.entry = Some(vec![0.0; length]),
                    "strata" => fit.strata = Some(vec![0; length]),
                    "design_rows" => fit.x = Array2::zeros((length, original.nvar())),
                    "n" => fit.n = length,
                    "coefficients" => fit.coefficients.resize(length, 0.0),
                    "means" => fit.means.resize(length, 0.0),
                    "design_columns" => fit.x = Array2::zeros((original.n, length)),
                    "variance_rows" => fit.var = Array2::zeros((length, original.nvar())),
                    "variance_columns" => fit.var = Array2::zeros((original.nvar(), length)),
                    _ => unreachable!(),
                }
                check_prediction_outcomes(&fit, true);
            }
        }
    }
    // A caller can resize all public covariate fields together. The private
    // cached baseline still has its original width and must not be indexed
    // using the newly enlarged fitted design.
    let mut fit = ordinary_model();
    fit.basehaz(true).unwrap();
    fit.coefficients.push(0.0);
    fit.means.push(0.0);
    fit.x = Array2::zeros((fit.n, fit.nvar()));
    fit.var = Array2::zeros((fit.nvar(), fit.nvar()));
    check_prediction_outcomes(&fit, true);
}

#[test]
fn modified_training_strata_return_errors_and_missing_multistratum_rows_are_rejected() {
    for cached in [false, true] {
        let mut fit = ordinary_model();
        if cached {
            fit.basehaz(true).unwrap();
        }
        fit.strata = Some(vec![99; fit.n]);
        assert!(fit.predict_survival_at(&[3.0], None).is_err());

        let mut fit = weighted_counting_model(TieMethod::Efron);
        if cached {
            fit.basehaz(true).unwrap();
        }
        fit.strata = None;
        assert!(fit.basehaz(true).is_err());
        assert!(fit.predict_survival_at(&[3.0], None).is_err());
    }
}

const START: [f64; 6] = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
const STOP: [f64; 6] = [4.0, 4.0, 5.0, 5.0, 2.0, 6.0];
const EVENT: [i32; 6] = [1, 1, 0, 1, 1, 0];
const WEIGHT: [f64; 6] = [0.5, 1.25, 2.0, 0.75, 1.5, 0.4];
const STRATA: [i32; 6] = [11, 11, 11, -4, -4, -4];

fn weighted_counting_model(method: TieMethod) -> CoxPHFit {
    let data = CoxphData::try_new(
        STOP.to_vec(),
        Some(START.to_vec()),
        EVENT.to_vec(),
        Array2::zeros((6, 0)),
        Some(WEIGHT.to_vec()),
        Some(STRATA.to_vec()),
        None,
    )
    .unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method,
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

fn direct_integrals(stratum: i32, start: f64, stop: f64, method: TieMethod) -> (f64, f64) {
    let mut times = STOP.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut hazard = 0.0;
    let mut variance = 0.0;
    for time in times {
        if time <= start || time > stop {
            continue;
        }
        let deaths: Vec<_> = (0..6)
            .filter(|&row| STRATA[row] == stratum && STOP[row] == time && EVENT[row] == 1)
            .collect();
        if deaths.is_empty() {
            continue;
        }
        let denominator: f64 = (0..6)
            .filter(|&row| STRATA[row] == stratum && START[row] < time && STOP[row] >= time)
            .map(|row| WEIGHT[row])
            .sum();
        let death_weight: f64 = deaths.iter().map(|&row| WEIGHT[row]).sum();
        let steps = if method == TieMethod::Efron {
            deaths.len()
        } else {
            1
        };
        for step in 0..steps {
            let adjusted = denominator - step as f64 / steps as f64 * death_weight;
            hazard += death_weight / steps as f64 / adjusted;
            variance += death_weight / steps as f64 / adjusted.powi(2);
        }
    }
    (hazard, variance)
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn weighted_tied_predictions_match_direct_stratified_counting_risk_sums() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for cached in [false, true] {
            let fit = weighted_counting_model(method);
            if cached {
                fit.basehaz(true).unwrap();
            }
            let risk: [f64; 3] = [0.7, 2.0, 1.3];
            let strata = [11, -4, 11];
            let entry = [0.5, 0.5, 1.5];
            let stop = [4.5, 5.5, 5.0];
            let newdata = CoxNewData::try_new(
                Array2::zeros((3, 0)),
                Some(strata.to_vec()),
                Some(risk.iter().map(|value| value.ln()).collect()),
                Some(stop.to_vec()),
                Some(entry.to_vec()),
            )
            .unwrap();
            let expected = fit.predict_expected(Some(&newdata), true).unwrap();
            let survival = fit.predict_survival(Some(&newdata), true).unwrap();
            assert_eq!(expected.fit.len(), 3);
            assert_eq!(expected.se_fit.as_ref().unwrap().len(), 3);
            for row in 0..3 {
                let (hazard, variance) =
                    direct_integrals(strata[row], entry[row], stop[row], method);
                close(expected.fit[row], hazard * risk[row]);
                close(
                    expected.se_fit.as_ref().unwrap()[row],
                    variance.sqrt() * risk[row],
                );
                close(survival.fit[row], (-hazard * risk[row]).exp());
                close(
                    survival.se_fit.as_ref().unwrap()[row],
                    variance.sqrt() * risk[row] * survival.fit[row],
                );
            }
            let at = [0.5, 2.0, 4.0, 7.0, 2.0];
            let probabilities = fit.predict_survival_at(&at, Some(&newdata)).unwrap();
            for (time_row, &time) in at.iter().enumerate() {
                for row in 0..3 {
                    let (hazard, _) = direct_integrals(strata[row], 0.0, time, method);
                    close(
                        probabilities[[time_row, row]],
                        (-hazard).exp().powf(risk[row]),
                    );
                }
            }
        }
    }
}

#[test]
fn aliases_null_models_and_generated_extreme_risks_keep_their_numeric_behavior() {
    for columns in [0, 1] {
        let data = CoxphData::try_new(
            vec![1.0, 2.0, 2.0, 4.0],
            None,
            vec![1, 1, 0, 1],
            Array2::zeros((4, columns)),
            None,
            None,
            None,
        )
        .unwrap();
        let fit = CoxPHFit::fit(
            data,
            CoxphOptions {
                method: TieMethod::Breslow,
                ..CoxphOptions::default()
            },
        )
        .unwrap();
        if columns == 1 {
            assert!(fit.coefficients[0].is_nan());
        }
        let newdata = CoxNewData::try_new(
            Array2::zeros((3, columns)),
            None,
            Some(vec![-1000.0, 0.0, 1000.0]),
            None,
            None,
        )
        .unwrap();
        let probabilities = fit
            .predict_survival_at(&[0.0, 2.0], Some(&newdata))
            .unwrap();
        for row in 0..3 {
            close(probabilities[[0, row]], 1.0);
        }
        close(probabilities[[1, 0]], 1.0);
        close(probabilities[[1, 1]], (-0.25_f64 - 1.0 / 3.0).exp());
        close(probabilities[[1, 2]], 0.0);
        let curves = fit
            .survfit(Some(&newdata), SurvfitOptions::default())
            .unwrap();
        let curve = &curves[0];
        close(curve.cumhaz[0][0], 0.0);
        close(curve.cumhaz[0][1], 0.25);
        assert_eq!(curve.cumhaz[0][2], f64::INFINITY);
        close(curve.surv[0][0], 1.0);
        close(curve.surv[0][1], (-0.25_f64).exp());
        close(curve.surv[0][2], 0.0);
        let data = CoxphData::try_new(
            vec![1.0, 2.0],
            None,
            vec![0, 0],
            Array2::zeros((2, columns)),
            None,
            None,
            None,
        )
        .unwrap();
        let no_events = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        let prediction = no_events.predict_expected(None, true).unwrap();
        assert_eq!(prediction.fit, [0.0, 0.0]);
        assert_eq!(prediction.se_fit.unwrap(), [0.0, 0.0]);
    }
}
