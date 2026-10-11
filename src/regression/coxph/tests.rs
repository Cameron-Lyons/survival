//! Regression coverage for the Cox model façade.

use super::fitting::crossprod;
use super::*;
use crate::constants::COX_RANK_TOLERANCE;
use crate::internal::step::find_interval;
use crate::regression::cox_optimizer::TieMethod;
use ndarray::Array2;

fn lung_like_data() -> CoxphData {
    let time = vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0];
    let status = vec![1, 1, 0, 1, 1, 0, 1, 0];
    let x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5];
    let x2 = [1.0, 0.2, 0.7, 1.3, 0.4, 1.1, 0.5, 0.9];
    let x = Array2::from_shape_vec(
        (8, 2),
        x1.iter().zip(&x2).flat_map(|(&a, &b)| [a, b]).collect(),
    )
    .unwrap();
    CoxphData::try_new(time, None, status, x, None, None, None).unwrap()
}

#[test]
fn grouped_vector_predictions_preserve_addition_order_and_error_quadrature() {
    let prediction = CoxPrediction {
        fit: vec![1e16, -1e16, 1.0, 2.0],
        se_fit: Some(vec![3.0, 4.0, 0.0, 2.0]),
    };
    assert_eq!(
        prediction.clone().collapse(&[-3, -3, -3, 9]).unwrap(),
        CoxPrediction {
            fit: vec![1.0, 2.0],
            se_fit: Some(vec![5.0, 2.0]),
        }
    );
    assert!(prediction.collapse(&[1, 2]).is_err());
    assert_eq!(
        CoxPrediction {
            fit: vec![],
            se_fit: Some(vec![])
        }
        .collapse(&[])
        .unwrap(),
        CoxPrediction {
            fit: vec![],
            se_fit: Some(vec![])
        }
    );
}

#[test]
fn grouped_terms_retain_selected_empty_and_repeated_columns_and_reference() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let group = [
        i32::MAX,
        i32::MIN,
        -11,
        i32::MAX,
        i32::MIN,
        -11,
        i32::MAX,
        -11,
    ];
    let assign = vec![vec![0], vec![1], vec![0, 1], vec![], vec![1]];
    for reference in [
        PredictReference::Sample,
        PredictReference::Strata,
        PredictReference::Zero,
    ] {
        for errors in [false, true] {
            let ungrouped = fit.predict_terms(None, errors, reference, &assign).unwrap();
            let grouped = fit
                .predict_terms_grouped(None, errors, reference, &assign, &group)
                .unwrap();
            assert_eq!(grouped.constant, ungrouped.constant);
            assert_eq!(grouped.fit.len(), 3);
            for (i, label) in [i32::MIN, -11, i32::MAX].iter().enumerate() {
                for t in 0..assign.len() {
                    let rows: Vec<usize> =
                        (0..group.len()).filter(|&r| group[r] == *label).collect();
                    let expected: f64 = rows.iter().map(|&r| ungrouped.fit[r][t]).sum();
                    assert_eq!(grouped.fit[i][t], expected);
                    if let Some(se) = ungrouped.se_fit.as_ref() {
                        let expected: f64 = rows.iter().map(|&r| se[r][t] * se[r][t]).sum();
                        assert_eq!(grouped.se_fit.as_ref().unwrap()[i][t], expected.sqrt());
                    } else {
                        assert!(grouped.se_fit.is_none());
                    }
                }
            }
        }
    }
    let empty = fit
        .predict_terms_grouped(None, true, PredictReference::Sample, &[], &group)
        .unwrap();
    assert_eq!(empty.fit, vec![Vec::<f64>::new(); 3]);
    assert_eq!(empty.se_fit, Some(empty.fit));
    assert!(
        fit.predict_terms_grouped(None, false, PredictReference::Sample, &assign, &[1, 2])
            .is_err()
    );
}

#[test]
fn grouped_terms_propagate_missingness_per_term_and_preserve_independent_columns() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let newdata = CoxNewData::try_new_prediction(
        ndarray::array![[f64::NAN, 2.0], [1.0, 3.0], [2.0, 4.0]],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let ungrouped = fit
        .predict_terms(
            Some(&newdata),
            true,
            PredictReference::Zero,
            &default_assign(2),
        )
        .unwrap();
    let grouped = fit
        .predict_terms_grouped(
            Some(&newdata),
            true,
            PredictReference::Zero,
            &default_assign(2),
            &[9, 2, 9],
        )
        .unwrap();
    assert_eq!(grouped.fit[0], ungrouped.fit[1]);
    assert!(grouped.fit[1][0].is_nan());
    assert!(grouped.se_fit.as_ref().unwrap()[1][0].is_nan());
    assert_eq!(grouped.fit[1][1], ungrouped.fit[0][1] + ungrouped.fit[2][1]);
    assert!(grouped.se_fit.unwrap()[1][1].is_finite());
}

#[test]
fn missing_prediction_values_preserve_independent_terms_and_errors() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let x = ndarray::array![[f64::NAN, 2.0], [1.0, 3.0]];
    assert!(CoxNewData::try_new(x.clone(), None, None, None, None).is_err());
    let newdata =
        CoxNewData::try_new_prediction(x, None, Some(vec![0.0, f64::NAN]), None, None).unwrap();
    let lp = fit
        .predict_lp(Some(&newdata), true, PredictReference::Zero)
        .unwrap();
    assert!(lp.fit.iter().all(|v| v.is_nan()));
    let se = lp.se_fit.unwrap();
    assert!(se[0].is_nan());
    assert!(se[1].is_finite());
    let terms = fit
        .predict_terms(
            Some(&newdata),
            true,
            PredictReference::Zero,
            &[vec![0], vec![1]],
        )
        .unwrap();
    assert!(terms.fit[0][0].is_nan());
    assert_eq!(terms.fit[0][1], 2.0 * fit.coefficients[1]);
    assert!(terms.fit[1].iter().all(|v| v.is_finite()));
    assert!(terms.se_fit.unwrap()[0][1].is_finite());
    assert!(
        CoxNewData::try_new_prediction(
            ndarray::array![[f64::INFINITY, 1.0]],
            None,
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(
        CoxNewData::try_new_prediction(
            ndarray::array![[1.0, 1.0]],
            None,
            None,
            Some(vec![f64::NAN]),
            None
        )
        .is_err()
    );
}

#[test]
fn default_controls_match_reference_efron_fit() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let expected = [0.103_056_235_224_469_12, -1.021_973_929_290_916];
    for (actual, expected) in fit.coefficients.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!((fit.loglik[0] - -7.714_231_144_849_085_5).abs() < 1e-12);
    assert!((fit.loglik[1] - -7.430_873_243_936_032).abs() < 1e-12);
    assert_eq!(fit.flag, 2);
    assert_eq!(fit.iter, 4);
    assert_eq!(fit.n, 8);
    assert_eq!(fit.nevent, 5);
    assert_eq!(fit.method, TieMethod::Efron);
    // Linear predictors are centred at the means.
    let mean_lp: f64 = fit.linear_predictors.iter().sum::<f64>() / 8.0;
    assert!(mean_lp.abs() < 1e-12);
    assert_eq!(fit.residuals.len(), 8);
    let total: f64 = fit.residuals.iter().sum();
    assert!(total.abs() < 1e-10, "martingale residuals sum to zero");
}

#[test]
fn default_rank_tolerance_preserves_near_collinear_columns() {
    let n = 20;
    let time: Vec<f64> = (1..=n).map(|value| value as f64).collect();
    let status: Vec<i32> = (0..n).map(|idx| i32::from(idx % 3 != 0)).collect();
    let rows: Vec<f64> = (0..n)
        .flat_map(|idx| {
            let first = (idx % 7) as f64 * 0.3 + (idx / 7) as f64 * 0.11;
            let direction = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let perturbation = direction * (0.2 + (idx % 5) as f64 * 0.13);
            [first, first + 1e-5 * perturbation]
        })
        .collect();
    let x = Array2::from_shape_vec((n, 2), rows).unwrap();
    let fit_with = |toler: Option<f64>| {
        let data = CoxphData::try_new(
            time.clone(),
            None,
            status.clone(),
            x.clone(),
            None,
            None,
            None,
        )
        .unwrap();
        let options = CoxphOptions {
            method: TieMethod::Breslow,
            iter_max: 0,
            toler_chol: toler.unwrap_or(COX_RANK_TOLERANCE),
            ..CoxphOptions::default()
        };
        CoxPHFit::fit(data, options).unwrap()
    };
    assert_eq!(fit_with(None).flag, 2);
    assert_eq!(fit_with(Some(1e-9)).flag, 1);
}

#[test]
fn robust_variance_is_the_dfbeta_crossproduct() {
    let data = lung_like_data();
    let options = CoxphOptions {
        cluster: Some(vec![0, 0, 1, 1, 2, 2, 3, 3]),
        ..CoxphOptions::default()
    };
    let fit = CoxPHFit::fit(data, options).unwrap();
    let naive = fit.naive_var.as_ref().expect("naive variance is kept");
    let dfbeta = fit
        .dfbeta_matrix(&fit.linear_predictors, true, fit.cluster.as_deref())
        .unwrap();
    let expected = crossprod(&dfbeta);
    for i in 0..2 {
        for j in 0..2 {
            assert!((fit.var[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            assert!(fit.var[(i, j)] != naive[(i, j)] || fit.var[(i, j)] == 0.0);
        }
    }
    assert!(fit.rscore.is_some());
}

#[test]
fn basehaz_uncentred_removes_the_mean_offset() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let centred = fit.basehaz(true).unwrap();
    let uncentred = fit.basehaz(false).unwrap();
    let center: f64 = fit
        .means
        .iter()
        .zip(&fit.coefficients)
        .map(|(m, b)| m * b)
        .sum();
    assert_eq!(centred.time, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    for (c, u) in centred.hazard.iter().zip(&uncentred.hazard) {
        assert!((u - c * (-center).exp()).abs() < 1e-12);
    }
    assert!(centred.strata.is_none());
}

#[test]
fn survfit_at_the_means_matches_the_baseline_hazard() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
    assert_eq!(curves.len(), 1);
    let basehaz = fit.basehaz(true).unwrap();
    for (g, row) in curves[0].cumhaz.iter().enumerate() {
        assert!((row[0] - basehaz.hazard[g]).abs() < 1e-12);
        assert!((curves[0].surv[g][0] - (-row[0]).exp()).abs() < 1e-12);
    }
    assert!(curves[0].std_err.is_some());
    let expected = fit.predict_expected(None, true).unwrap();
    assert_eq!(expected.fit.len(), 8);
    for (e, (&s, r)) in expected
        .fit
        .iter()
        .zip(fit.status.iter().zip(&fit.residuals))
    {
        assert!((e - (f64::from(s) - r)).abs() < 1e-12);
    }
}

#[test]
fn survival_at_requested_times_matches_full_curves() {
    let times = [8.0, 2.0, -1.0, 2.0, 3.5];
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for stratified in [false, true] {
            let mut data = lung_like_data();
            data.offset = Some(vec![0.1, 0.3, -0.2, 0.0, 0.4, 0.2, -0.1, 0.5]);
            if stratified {
                data.strata = Some(vec![17, -3, 17, -3, 17, -3, 17, -3]);
            }
            let fit = CoxPHFit::fit(
                data,
                CoxphOptions {
                    method,
                    ..Default::default()
                },
            )
            .unwrap();
            let newdata = CoxNewData::try_new(
                fit.x.clone(),
                fit.strata.clone(),
                Some(fit.offset.clone()),
                None,
                None,
            )
            .unwrap();
            let full = fit
                .survfit(
                    Some(&newdata),
                    SurvfitOptions {
                        se_fit: false,
                        ..Default::default()
                    },
                )
                .unwrap();
            let actual = fit.predict_survival_at(&times, None).unwrap();
            assert_eq!(
                actual,
                fit.predict_survival_at(&times, Some(&newdata)).unwrap()
            );
            assert_eq!(actual.dim(), (times.len(), fit.n));
            for (i, &time) in times.iter().enumerate() {
                for row in 0..fit.n {
                    let (curve, column) = if stratified {
                        (&full[row], 0)
                    } else {
                        (&full[0], row)
                    };
                    let index = find_interval(&curve.time, time, false);
                    let expected = if index == 0 {
                        1.0
                    } else {
                        curve.surv[index - 1][column]
                    };
                    assert_eq!(actual[(i, row)], expected);
                }
            }
            assert_eq!(
                fit.predict_survival_at(&[], None).unwrap().dim(),
                (0, fit.n)
            );
            assert!(fit.predict_survival_at(&[f64::NAN], None).is_err());
            if stratified {
                let missing_strata = CoxNewData {
                    strata: None,
                    ..newdata
                };
                assert!(
                    fit.predict_survival_at(&times, Some(&missing_strata))
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn predictions_follow_the_reference_argument() {
    let fit = CoxPHFit::fit(lung_like_data(), CoxphOptions::default()).unwrap();
    let sample = fit
        .predict_lp(None, false, PredictReference::Sample)
        .unwrap();
    assert_eq!(sample.fit, fit.linear_predictors);
    let zero = fit.predict_lp(None, true, PredictReference::Zero).unwrap();
    let center: f64 = fit
        .means
        .iter()
        .zip(&fit.coefficients)
        .map(|(m, b)| m * b)
        .sum();
    for (z, lp) in zero.fit.iter().zip(&fit.linear_predictors) {
        assert!((z - (lp + center)).abs() < 1e-12);
    }
    assert!(zero.se_fit.is_some());
    let newdata = CoxNewData::try_new(
        Array2::from_shape_vec((1, 2), fit.means.clone()).unwrap(),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let at_means = fit
        .predict_risk(Some(&newdata), true, PredictReference::Strata)
        .unwrap();
    assert!((at_means.fit[0] - 1.0).abs() < 1e-12);
    let terms = fit
        .predict_terms(None, true, PredictReference::Sample, &default_assign(2))
        .unwrap();
    assert_eq!(terms.fit.len(), 8);
    assert!((terms.constant - center).abs() < 1e-12);
}

#[test]
fn newdata_offset_is_centred_only_in_the_use_x_branch() {
    let offset = vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let data = CoxphData {
        offset: Some(offset.clone()),
        ..lung_like_data()
    };
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    let newdata =
        CoxNewData::try_new(fit.x.clone(), None, Some(offset.clone()), None, None).unwrap();
    // predict.coxph without se.fit: newx %*% beta + newoffset, so the
    // training rows give back the linear predictors
    let plain = fit
        .predict_lp(Some(&newdata), false, PredictReference::Sample)
        .unwrap();
    for (p, lp) in plain.fit.iter().zip(&fit.linear_predictors) {
        assert!((p - lp).abs() < 1e-12);
    }
    // with se.fit the offset is centred at mean(offset)
    let with_se = fit
        .predict_lp(Some(&newdata), true, PredictReference::Sample)
        .unwrap();
    let offset_mean = offset.iter().sum::<f64>() / offset.len() as f64;
    for (p, lp) in with_se.fit.iter().zip(&fit.linear_predictors) {
        assert!((p - (lp - offset_mean)).abs() < 1e-12);
    }
}

#[test]
fn strata_and_counting_process_fits_keep_row_order() {
    let time = vec![5.0, 1.0, 4.0, 2.0, 3.0, 6.0, 8.0, 7.0];
    let entry = vec![0.0, 0.0, 1.0, 0.0, 0.5, 2.0, 0.0, 1.0];
    let status = vec![1, 1, 0, 0, 1, 0, 0, 1];
    let x = Array2::from_shape_vec((8, 1), vec![0.6, 0.5, 0.8, 1.0, 0.3, 0.4, 0.2, 0.9]).unwrap();
    let strata = vec![1, 0, 1, 0, 1, 0, 1, 0];
    let data = CoxphData::try_new(
        time.clone(),
        Some(entry.clone()),
        status.clone(),
        x.clone(),
        None,
        Some(strata.clone()),
        None,
    )
    .unwrap();
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    assert_eq!(fit.sorted.codes, vec![0, 1]);
    assert_eq!(fit.time, time);
    assert_eq!(fit.strata.as_deref(), Some(strata.as_slice()));
    let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
    assert_eq!(curves.len(), 2);
    assert_eq!(curves[0].stratum, 0);
    let basehaz = fit.basehaz(true).unwrap();
    assert_eq!(basehaz.strata.as_ref().unwrap().len(), basehaz.time.len());
    let total: f64 = fit.residuals.iter().sum();
    assert!(total.abs() < 1e-10);
}

/// `survfit(coxph(Surv(time, status) ~ x + strata(g), d), start.time =
/// 15)` in R 3.8-12 (`d` from `set.seed(1)`, `x = rnorm(20)`): the rows
/// before 15 leave, stratum 1 keeps an empty curve with `n = 0`, and the
/// risk scores stay those of the whole fit.
#[test]
fn start_time_drops_the_earlier_rows_and_keeps_an_emptied_stratum() {
    let time: Vec<f64> = (1..=10).chain(21..=30).map(f64::from).collect();
    let status = [1, 0, 1, 1, 0].repeat(4);
    let strata: Vec<i32> = [1; 10].into_iter().chain([2; 10]).collect();
    let x = [
        -0.626_453_810_742_332_4,
        0.183_643_324_222_082_24,
        -0.835_628_612_410_047_2,
        1.595_280_802_137_791_6,
        0.329_507_771_815_360_5,
        -0.820_468_384_118_015_3,
        0.487_429_052_428_485_3,
        0.738_324_705_129_217_3,
        0.575_781_351_653_492_3,
        -0.305_388_387_156_356,
        1.511_781_168_450_848,
        0.389_843_236_411_431_1,
        -0.621_240_580_541_803_8,
        -2.214_699_887_177_5,
        1.124_930_918_143_108_2,
        -0.044_933_609_015_230_85,
        -0.016_190_263_098_946_087,
        0.943_836_210_685_299_2,
        0.821_221_195_098_088_6,
        0.593_901_321_217_508_8,
    ];
    let data = CoxphData::try_new(
        time,
        None,
        status,
        Array2::from_shape_vec((20, 1), x.to_vec()).unwrap(),
        None,
        Some(strata),
        None,
    )
    .unwrap();
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    let after = |start_time| SurvfitOptions {
        start_time: Some(start_time),
        ..SurvfitOptions::default()
    };
    let curves = fit.survfit(None, after(15.0)).unwrap();
    assert_eq!(curves.len(), 2);
    assert_eq!((curves[0].stratum, curves[0].n), (1, 0));
    assert!(curves[0].time.is_empty() && curves[0].surv.is_empty());
    assert_eq!(curves[1].n, 10);
    let times: Vec<f64> = (21..=30).map(f64::from).collect();
    assert_eq!(curves[1].time, times);
    let surv = [
        0.911_703_185_270_467,
        0.911_703_185_270_467,
        0.818_916_920_480_482,
        0.721_769_214_450_701,
        0.721_769_214_450_701,
        0.579_122_152_428_905,
        0.579_122_152_428_905,
        0.378_441_860_225_577,
        0.203_946_105_189_285,
        0.203_946_105_189_285,
    ];
    let std_err = [
        0.093_842_646_361_357,
        0.093_842_646_361_357,
        0.147_738_856_126_131,
        0.202_627_188_554_77,
        0.202_627_188_554_77,
        0.295_853_546_336_666,
        0.295_853_546_336_666,
        0.516_581_455_410_216,
        0.819_221_646_496_032,
        0.819_221_646_496_032,
    ];
    let se = curves[1].std_err.as_ref().unwrap();
    for g in 0..10 {
        assert!((curves[1].surv[g][0] - surv[g]).abs() < 1e-12 * surv[g]);
        assert!((se[g][0] - std_err[g]).abs() < 1e-12 * std_err[g]);
    }
    // the cached curves of the whole fit are left alone
    assert_eq!(
        fit.survfit(None, SurvfitOptions::default()).unwrap()[0].n,
        10
    );

    let kp = fit
        .survfit(
            None,
            SurvfitOptions {
                stype: 1,
                ..after(15.0)
            },
        )
        .unwrap();
    assert!((kp[1].surv[9][0] - 0.139_802_086_569_338).abs() < 1e-12);

    // the last row, at 30, is censored
    let error = fit.survfit(None, after(30.0)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("start.time argument has removed all endpoints"),
        "{error}"
    );
}

/// `coxph()` fits at the centred offset and adds the mean back, so
/// offsets near the limits of `exp()` give R's fit
/// (`coxph(Surv(time, status) ~ x1 + offset(708 + x2))`, and `-740`).
#[test]
fn offsets_are_centred_before_fitting() {
    let base = lung_like_data();
    let lp_without_shift = [
        0.689_312_979_161_027_5,
        0.292_366_411_600_775_64,
        0.523_664_123_307_610_3,
        1.593_893_127_820_649_6,
        0.425_190_839_527_484_2,
        0.856_488_551_234_319,
        0.928_244_271_967_232_4,
        0.790_839_695_380_901_6,
    ];
    let residuals = [
        0.833_716_415_337_278_3,
        0.888_195_914_321_604_3,
        -0.190_854_674_879_980_42,
        -0.158_633_253_137_838_05,
        0.639_931_579_226_752_9,
        -0.668_367_507_721_713_4,
        -0.252_386_478_362_365_3,
        -1.091_601_994_783_737_2,
    ];
    for shift in [708.0, -740.0] {
        let offset = base.x.column(1).iter().map(|x2| shift + x2).collect();
        let data = CoxphData::try_new(
            base.time.clone(),
            None,
            base.status.clone(),
            base.x.slice(ndarray::s![.., ..1]).to_owned(),
            None,
            None,
            Some(offset),
        )
        .unwrap();
        let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
        assert!((fit.coefficients[0] - 0.671_755_720_732_935_5).abs() < 1e-9);
        assert!((fit.loglik[0] - -8.483_319_079_073_608).abs() < 1e-9);
        assert!((fit.loglik[1] - -8.313_974_981_255_075).abs() < 1e-9);
        for (lp, expected) in fit.linear_predictors.iter().zip(lp_without_shift) {
            assert!((lp - (shift + expected)).abs() < 1e-9);
        }
        for (actual, expected) in fit.residuals.iter().zip(residuals) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!((fit.concordance.concordance[0] - 6.0 / 19.0).abs() < 1e-12);
        // The offset is kept as given.
        assert!((fit.offset[0] - (shift + 1.0)).abs() < 1e-12);
    }
}

/// Without events `coxph()` returns before any fitter runs: R's
/// `coxph(Surv(start, stop, status) ~ x + offset(off), weights = w)` on
/// four censored rows.
#[test]
fn data_without_events_gets_coxph_skeleton_fit() {
    let data = CoxphData::try_new(
        vec![2.0, 3.0, 4.0, 5.0],
        Some(vec![0.0, 0.0, 1.0, 1.0]),
        vec![0; 4],
        Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Some(vec![1.0, 2.0, 3.0, 4.0]),
        None,
        Some(vec![0.1, 0.2, 0.3, 0.4]),
    )
    .unwrap();
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    assert!(fit.coefficients[0].is_nan());
    assert_eq!(fit.var[(0, 0)], 0.0);
    assert_eq!(fit.loglik, [0.0, 0.0]);
    assert_eq!((fit.score, fit.wald_test, fit.iter), (0.0, 0.0, 0));
    // Unweighted column means; the linear predictors are the centred offset.
    assert_eq!(fit.means, vec![2.5]);
    for (lp, expected) in fit.linear_predictors.iter().zip([-0.15, -0.05, 0.05, 0.15]) {
        assert!((lp - expected).abs() < 1e-12);
    }
    assert_eq!(fit.residuals, vec![0.0; 4]);
    assert_eq!(fit.nevent, 0);
    assert!(fit.concordance.concordance[0].is_nan());
    assert_eq!(fit.concordance.count[0].concordant, 0.0);
    assert!(fit.concordance.var.is_none());
}

#[test]
fn predictors_and_weights_are_checked_only_for_data_with_events() {
    let data = |status: Vec<i32>| {
        CoxphData::try_new(
            vec![2.0, 3.0, 4.0, 5.0],
            None,
            status,
            Array2::from_shape_vec((4, 1), vec![1.0, f64::INFINITY, 3.0, 4.0]).unwrap(),
            Some(vec![0.0, 2.0, 3.0, 4.0]),
            None,
            None,
        )
        .unwrap()
    };
    let fit = CoxPHFit::fit(data(vec![0; 4]), CoxphOptions::default()).unwrap();
    assert!(fit.coefficients[0].is_nan());
    assert_eq!(fit.means, vec![f64::INFINITY]);
    let err = CoxPHFit::fit(data(vec![1, 0, 0, 0]), CoxphOptions::default()).unwrap_err();
    assert!(err.to_string().contains("x contains non-finite value inf"));
    let mut finite = data(vec![1, 0, 0, 0]);
    finite.x[(1, 0)] = 2.0;
    let err = CoxPHFit::fit(finite, CoxphOptions::default()).unwrap_err();
    assert!(err.to_string().contains("Invalid weights, must be >0"));
}

#[test]
fn null_model_reports_the_log_likelihood_and_residuals() {
    let data = CoxphData::try_new(
        vec![1.0, 2.0, 3.0],
        None,
        vec![1, 1, 0],
        Array2::zeros((3, 0)),
        None,
        None,
        None,
    )
    .unwrap();
    let fit = CoxPHFit::fit(data, CoxphOptions::default()).unwrap();
    assert!(fit.coefficients.is_empty());
    assert_eq!(fit.linear_predictors, vec![0.0; 3]);
    assert!((fit.residuals[0] - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
    assert_eq!(fit.wald_test, 0.0);
}

#[test]
fn invalid_inputs_are_rejected() {
    assert!(
        CoxphData::try_new(
            vec![],
            None,
            vec![],
            Array2::zeros((0, 1)),
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(
        CoxphData::try_new(
            vec![1.0, 2.0],
            None,
            vec![1, 2],
            Array2::zeros((2, 1)),
            None,
            None,
            None
        )
        .is_err()
    );
    assert!(
        CoxphData::try_new(
            vec![1.0, 2.0],
            Some(vec![0.0, 2.0]),
            vec![1, 0],
            Array2::zeros((2, 1)),
            None,
            None,
            None
        )
        .is_err()
    );
    let data = lung_like_data();
    let options = CoxphOptions {
        method: TieMethod::Exact,
        robust: Some(true),
        ..CoxphOptions::default()
    };
    assert!(CoxPHFit::fit(data, options).is_err());
}
