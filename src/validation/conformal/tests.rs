use super::*;

#[test]
fn test_weighted_quantile_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let q50 = weighted_quantile(&values, &weights, 0.5);
    assert!((2.0..=3.5).contains(&q50));

    let q90 = weighted_quantile(&values, &weights, 0.9);
    assert!((4.0..=5.0).contains(&q90));
}

#[test]
fn test_weighted_quantile_unequal_weights() {
    let values = vec![1.0, 2.0, 3.0];
    let weights = vec![1.0, 2.0, 1.0];

    let q50 = weighted_quantile(&values, &weights, 0.5);
    assert!((1.5..=2.5).contains(&q50));
}

#[test]
fn test_conformity_scores_uncensored() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.1];

    let (scores, weights) = compute_conformity_scores(&time, &status, &predicted, false, 0.01);

    assert_eq!(scores.len(), 5);
    assert_eq!(weights.len(), 5);
    assert!((scores[0] - (-0.1)).abs() < 1e-10);
    assert!((scores[1] - 0.1).abs() < 1e-10);
    assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-10));
}

#[test]
fn test_conformal_calibrate_no_censoring() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = conformal_calibrate(time, status, predicted, Some(0.9), Some(false)).unwrap();

    assert_eq!(result.n_calibration, 5);
    assert!((result.coverage_level - 0.9).abs() < 1e-10);
    assert!(result.ipcw_weights.is_none());
}

#[test]
fn test_conformal_with_ipcw() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 0, 1, 1];
    let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8];

    let result = conformal_calibrate(time, status, predicted, Some(0.9), Some(true)).unwrap();

    assert_eq!(result.n_calibration, 4);
    assert!(result.ipcw_weights.is_some());
    let weights = result.ipcw_weights.unwrap();
    assert_eq!(weights.len(), 4);
    assert!(weights.iter().all(|&w| w >= 1.0));
}

#[test]
fn test_conformal_coverage_guarantee() {
    let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status_calib = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result = conformal_survival_from_predictions(
        time_calib,
        status_calib,
        predicted_calib,
        predicted_new.clone(),
        Some(0.9),
        Some(false),
    )
    .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 3);
    assert_eq!(result.predicted_time.len(), 3);
    for (lower, pred) in result
        .lower_predictive_bound
        .iter()
        .zip(predicted_new.iter())
    {
        assert!(lower <= pred);
    }
}

#[test]
fn test_conformal_empty_input() {
    let result = conformal_calibrate(vec![], vec![], vec![], None, None);
    assert!(result.is_err());
}

#[test]
fn test_conformal_all_censored() {
    let time = vec![1.0, 2.0, 3.0];
    let status = vec![0, 0, 0];
    let predicted = vec![1.0, 2.0, 3.0];

    let result = conformal_calibrate(time, status, predicted, None, None);
    assert!(result.is_err());
}

#[test]
fn test_conformal_coverage_test() {
    let time_test = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status_test = vec![1, 1, 1, 1, 1];
    let lpb = vec![0.5, 1.5, 2.5, 3.5, 4.5];

    let result = conformal_coverage_test(time_test, status_test, lpb, Some(0.9)).unwrap();

    assert!((result.empirical_coverage - 1.0).abs() < 1e-10);
    assert!((result.expected_coverage - 0.9).abs() < 1e-10);
}

#[test]
fn test_conformal_predict_basic() {
    let predicted = vec![5.0, 10.0, 15.0];
    let quantile_threshold = 2.0;

    let result = conformal_predict(quantile_threshold, predicted, Some(0.9)).unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 3);
    assert!((result.lower_predictive_bound[0] - 3.0).abs() < 1e-10);
    assert!((result.lower_predictive_bound[1] - 8.0).abs() < 1e-10);
    assert!((result.lower_predictive_bound[2] - 13.0).abs() < 1e-10);
}

#[test]
fn test_conformal_predict_clamps_to_zero() {
    let predicted = vec![1.0];
    let quantile_threshold = 5.0;

    let result = conformal_predict(quantile_threshold, predicted, None).unwrap();

    assert!((result.lower_predictive_bound[0] - 0.0).abs() < 1e-10);
}

#[test]
fn test_censoring_model_fit() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 0, 1, 0, 1];

    let model = CensoringModel::fit(&time, &status);

    assert!(!model.unique_times.is_empty());
    assert_eq!(model.unique_times.len(), model.survival_probs.len());
    assert!(
        model
            .survival_probs
            .iter()
            .all(|&s| (0.0..=1.0).contains(&s))
    );
}

#[test]
fn test_censoring_model_survival_at() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 0, 1, 0, 1];

    let model = CensoringModel::fit(&time, &status);

    let surv_0 = model.survival_at(0.0);
    assert!((surv_0 - 1.0).abs() < 1e-10);

    let surv_10 = model.survival_at(10.0);
    assert!(surv_10 <= 1.0);
    assert!(surv_10 >= 0.0);
}

#[test]
fn test_impute_censoring_times() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 0, 1, 0, 1];

    let model = CensoringModel::fit(&time, &status);
    let imputed = impute_censoring_times(&time, &status, &model, 42);

    assert_eq!(imputed.len(), 5);
    assert!(imputed[1] > time[1]);
    assert!(imputed[3] > time[3]);
}

#[test]
fn test_doubly_robust_conformal_calibrate() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
    let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];

    let result =
        doubly_robust_conformal_calibrate(time, status, predicted, Some(0.9), None, Some(42), None)
            .unwrap();

    assert!((result.coverage_level - 0.9).abs() < 1e-10);
    assert_eq!(result.imputed_censoring_times.len(), 10);
    assert_eq!(result.censoring_probs.len(), 10);
    assert!(result.n_imputed > 0);
    assert!(result.n_effective > 0.0);
}

#[test]
fn test_doubly_robust_conformal_survival() {
    let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status_calib = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
    let predicted_calib = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result = doubly_robust_conformal_survival(
        time_calib,
        status_calib,
        predicted_calib,
        predicted_new.clone(),
        Some(0.9),
        None,
        Some(42),
        None,
    )
    .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 3);
    assert_eq!(result.predicted_time.len(), 3);
    for (lower, pred) in result
        .lower_predictive_bound
        .iter()
        .zip(predicted_new.iter())
    {
        assert!(lower <= pred);
        assert!(*lower >= 0.0);
    }
}

#[test]
fn test_doubly_robust_empty_input() {
    let result = doubly_robust_conformal_calibrate(vec![], vec![], vec![], None, None, None, None);
    assert!(result.is_err());
}

#[test]
fn test_doubly_robust_all_uncensored() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result =
        doubly_robust_conformal_calibrate(time, status, predicted, Some(0.9), None, Some(42), None)
            .unwrap();

    assert_eq!(result.n_imputed, 0);
}

#[test]
fn test_doubly_robust_deterministic_with_seed() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 0, 1, 0, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result1 = doubly_robust_conformal_calibrate(
        time.clone(),
        status.clone(),
        predicted.clone(),
        Some(0.9),
        None,
        Some(123),
        None,
    )
    .unwrap();

    let result2 = doubly_robust_conformal_calibrate(
        time,
        status,
        predicted,
        Some(0.9),
        None,
        Some(123),
        None,
    )
    .unwrap();

    assert!((result1.quantile_threshold - result2.quantile_threshold).abs() < 1e-10);
}

#[test]
fn test_two_sided_conformal_calibrate() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
    let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];

    let result = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

    assert!((result.coverage_level - 0.9).abs() < 1e-10);
    assert_eq!(result.n_uncensored, 7);
    assert_eq!(result.n_censored, 3);
    assert!(result.lower_quantile.is_finite());
    assert!(result.upper_quantile.is_finite());
}

#[test]
fn test_two_sided_conformal_calibrate_all_uncensored() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

    assert_eq!(result.n_uncensored, 5);
    assert_eq!(result.n_censored, 0);
    assert!(result.censoring_score_threshold.is_infinite());
}

#[test]
fn test_two_sided_conformal_predict() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let calibration = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

    let predicted_new = vec![3.0, 5.0, 7.0];
    let result = two_sided_conformal_predict(&calibration, predicted_new.clone(), None).unwrap();

    assert_eq!(result.lower_bound.len(), 3);
    assert_eq!(result.upper_bound.len(), 3);
    assert_eq!(result.is_two_sided.len(), 3);

    for (i, pred) in predicted_new.iter().enumerate() {
        assert!(result.lower_bound[i] <= *pred);
        assert!(result.lower_bound[i] >= 0.0);
        if result.is_two_sided[i] {
            assert!(result.upper_bound[i] >= *pred);
            assert!(result.upper_bound[i].is_finite());
        }
    }
}

#[test]
fn test_two_sided_conformal_survival() {
    let time_calib = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status_calib = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
    let predicted_calib = vec![1.1, 1.9, 2.8, 4.2, 4.8, 6.1, 6.9, 7.8, 9.2, 9.8];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result = two_sided_conformal_survival(
        time_calib,
        status_calib,
        predicted_calib,
        predicted_new.clone(),
        Some(0.9),
    )
    .unwrap();

    assert_eq!(result.lower_bound.len(), 3);
    assert_eq!(result.upper_bound.len(), 3);
    assert_eq!(result.n_two_sided + result.n_one_sided, 3);

    for (lower, pred) in result.lower_bound.iter().zip(predicted_new.iter()) {
        assert!(lower <= pred);
        assert!(*lower >= 0.0);
    }
}

#[test]
fn test_two_sided_conformal_empty_input() {
    let result = two_sided_conformal_calibrate(vec![], vec![], vec![], None);
    assert!(result.is_err());
}

#[test]
fn test_two_sided_conformal_all_censored() {
    let time = vec![1.0, 2.0, 3.0];
    let status = vec![0, 0, 0];
    let predicted = vec![1.0, 2.0, 3.0];

    let result = two_sided_conformal_calibrate(time, status, predicted, None);
    assert!(result.is_err());
}

#[test]
fn test_two_sided_bounds_ordering() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let calibration = two_sided_conformal_calibrate(time, status, predicted, Some(0.9)).unwrap();

    let predicted_new = vec![5.0];
    let result = two_sided_conformal_predict(&calibration, predicted_new, None).unwrap();

    assert!(result.lower_bound[0] <= result.upper_bound[0]);
}

#[test]
fn test_compute_two_sided_scores() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 0, 1, 1];
    let predicted = vec![1.1, 1.9, 2.8, 4.2, 4.8];

    let (lower_scores, upper_scores) = compute_two_sided_scores(&time, &status, &predicted);

    assert_eq!(lower_scores.len(), 4);
    assert_eq!(upper_scores.len(), 4);
}

#[test]
fn test_conformalized_survival_distribution() {
    let time_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let survival_probs_calib = vec![
        vec![0.9, 0.8, 0.7, 0.6, 0.5],
        vec![0.95, 0.85, 0.75, 0.65, 0.55],
        vec![0.85, 0.75, 0.65, 0.55, 0.45],
    ];
    let time_calib = vec![3.0, 4.0, 2.0];
    let status_calib = vec![1, 1, 1];
    let survival_probs_new = vec![vec![0.9, 0.8, 0.7, 0.6, 0.5]];

    let result = conformalized_survival_distribution(
        time_points.clone(),
        survival_probs_calib,
        time_calib,
        status_calib,
        survival_probs_new,
        Some(0.9),
    )
    .unwrap();

    assert_eq!(result.time_points.len(), 5);
    assert_eq!(result.n_subjects, 1);
    assert_eq!(result.survival_lower.len(), 1);
    assert_eq!(result.survival_upper.len(), 1);
}

#[test]
fn test_bootstrap_conformal_survival() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result = bootstrap_conformal_survival(
        time,
        status,
        predicted,
        predicted_new.clone(),
        Some(0.9),
        Some(50),
        Some(42),
    )
    .unwrap();

    assert_eq!(result.lower_bound.len(), 3);
    assert_eq!(result.upper_bound.len(), 3);
    assert_eq!(result.n_bootstrap, 50);

    for i in 0..3 {
        assert!(result.lower_bound[i] <= result.upper_bound[i]);
    }
}

#[test]
fn test_cqr_conformal_survival() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result =
        cqr_conformal_survival(time, status, predicted, predicted_new, Some(0.9), None).unwrap();

    assert_eq!(result.lower_bound.len(), 3);
    assert_eq!(result.upper_bound.len(), 3);

    for i in 0..3 {
        assert!(result.lower_bound[i] <= result.upper_bound[i]);
    }
}

#[test]
fn test_conformal_width_analysis() {
    let lower_bounds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let upper_bounds = vec![3.0, 4.0, 5.0, 6.0, 7.0];
    let predicted = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let result = conformal_width_analysis(lower_bounds, upper_bounds, predicted).unwrap();

    assert!((result.mean_width - 2.0).abs() < 1e-10);
    assert!((result.median_width - 2.0).abs() < 1e-10);
    assert!((result.std_width - 0.0).abs() < 1e-10);
    assert!((result.min_width - 2.0).abs() < 1e-10);
    assert!((result.max_width - 2.0).abs() < 1e-10);
}

#[test]
fn test_conformal_coverage_cv() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = conformal_coverage_cv(
        time,
        status,
        predicted,
        Some(5),
        Some(vec![0.8, 0.9, 0.95]),
        Some(42),
    )
    .unwrap();

    assert_eq!(result.coverage_candidates.len(), 3);
    assert_eq!(result.mean_widths.len(), 3);
    assert_eq!(result.empirical_coverages.len(), 3);
    assert!(result.optimal_coverage >= 0.8 && result.optimal_coverage <= 0.95);
}

#[test]
fn test_conformal_survival_parallel() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let predicted_new = vec![3.0, 5.0, 7.0];

    let result =
        conformal_survival_parallel(time, status, predicted, predicted_new.clone(), Some(0.9))
            .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 3);
    for (lower, pred) in result
        .lower_predictive_bound
        .iter()
        .zip(predicted_new.iter())
    {
        assert!(lower <= pred);
        assert!(*lower >= 0.0);
    }
}

#[test]
fn test_bootstrap_sample_indices() {
    let indices = bootstrap_sample_indices(10, 42);
    assert_eq!(indices.len(), 10);
    assert!(indices.iter().all(|&i| i < 10));
}

#[test]
fn test_covariate_shift_conformal_survival() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let importance_weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let predicted_new = vec![2.5, 4.0];

    let result = covariate_shift_conformal_survival(
        time,
        status,
        predicted,
        importance_weights,
        predicted_new.clone(),
        Some(0.9),
        Some(false),
        None,
    )
    .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 2);
    assert_eq!(result.predicted_time.len(), 2);
    assert_eq!(result.n_calibration, 5);
    for lower in result.lower_predictive_bound.iter() {
        assert!(*lower >= 0.0);
        assert!(lower.is_finite());
    }
}

#[test]
fn test_covariate_shift_with_different_weights() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let importance_weights = vec![0.5, 1.0, 1.5, 1.0, 0.5];
    let predicted_new = vec![3.0];

    let result = covariate_shift_conformal_survival(
        time,
        status,
        predicted,
        importance_weights,
        predicted_new,
        Some(0.9),
        Some(true),
        None,
    )
    .unwrap();

    assert_eq!(result.combined_weights.len(), 5);
    assert!(result.weight_diagnostics.effective_sample_size > 0.0);
    assert!(result.weight_diagnostics.min_weight > 0.0);
}

#[test]
fn test_covariate_shift_empty_input() {
    let result = covariate_shift_conformal_survival(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![1.0],
        None,
        None,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_cvplus_conformal_calibrate() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted_loo = vec![1.2, 1.8, 3.1, 4.1, 4.9];

    let result = cvplus_conformal_calibrate(time, status, predicted_loo, Some(0.9)).unwrap();

    assert_eq!(result.n_calibration, 5);
    assert!((result.coverage_level - 0.9).abs() < 1e-10);
    assert!(result.quantile_threshold.is_finite());
    assert!(result.adjustment_factor > 1.0);
}

#[test]
fn test_cvplus_conformal_survival() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted_loo = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predicted_new = vec![2.5, 4.0];

    let result = cvplus_conformal_survival(
        time,
        status,
        predicted_loo,
        predicted_new.clone(),
        Some(0.9),
    )
    .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 2);
    assert_eq!(result.n_calibration, 5);
    for lower in result.lower_predictive_bound.iter() {
        assert!(*lower >= 0.0);
        assert!(lower.is_finite());
    }
}

#[test]
fn test_cvplus_empty_input() {
    let result = cvplus_conformal_calibrate(vec![], vec![], vec![], None);
    assert!(result.is_err());
}

#[test]
fn test_cvplus_all_censored() {
    let time = vec![1.0, 2.0, 3.0];
    let status = vec![0, 0, 0];
    let predicted_loo = vec![1.0, 2.0, 3.0];

    let result = cvplus_conformal_calibrate(time, status, predicted_loo, None);
    assert!(result.is_err());
}

#[test]
fn test_mondrian_conformal_calibrate() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let status = vec![1, 1, 1, 1, 1, 1];
    let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.2];
    let group_labels = vec![0, 0, 0, 1, 1, 1];

    let result =
        mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(2))
            .unwrap();

    assert_eq!(result.group_thresholds.len(), 2);
    assert_eq!(result.group_sizes.len(), 2);
    assert!(result.global_threshold.is_finite());
}

#[test]
fn test_mondrian_conformal_survival() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let status = vec![1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let group_labels = vec![0, 0, 0, 1, 1, 1];
    let predicted_new = vec![2.5, 5.0];
    let group_labels_new = vec![0, 1];

    let result = mondrian_conformal_survival(
        time,
        status,
        predicted,
        group_labels,
        predicted_new.clone(),
        group_labels_new,
        Some(0.9),
        Some(2),
    )
    .unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 2);
    assert_eq!(result.applied_thresholds.len(), 2);
    for lower in result.lower_predictive_bound.iter() {
        assert!(*lower >= 0.0);
        assert!(lower.is_finite());
    }
}

#[test]
fn test_mondrian_conformal_predict() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let group_labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let calibration =
        mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(3))
            .unwrap();

    let predicted_new = vec![3.0, 7.0];
    let group_labels_new = vec![0, 1];

    let result = mondrian_conformal_predict(&calibration, predicted_new, group_labels_new).unwrap();

    assert_eq!(result.lower_predictive_bound.len(), 2);
    assert_eq!(result.used_global_fallback.len(), 2);
}

#[test]
fn test_mondrian_small_group_fallback() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let group_labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

    let calibration =
        mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(5))
            .unwrap();

    assert!(calibration.diagnostics.n_small_groups > 0);

    let result = mondrian_conformal_predict(&calibration, vec![9.5], vec![1]).unwrap();
    assert!(result.used_global_fallback[0]);
}

#[test]
fn test_mondrian_empty_input() {
    let result = mondrian_conformal_calibrate(vec![], vec![], vec![], vec![], None, None);
    assert!(result.is_err());
}

#[test]
fn test_mondrian_new_group_fallback() {
    let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let status = vec![1, 1, 1, 1, 1];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let group_labels = vec![0, 0, 0, 0, 0];

    let calibration =
        mondrian_conformal_calibrate(time, status, predicted, group_labels, Some(0.9), Some(3))
            .unwrap();

    let result = mondrian_conformal_predict(&calibration, vec![3.0], vec![99]).unwrap();
    assert!(result.used_global_fallback[0]);
}
