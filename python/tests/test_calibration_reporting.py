import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_calibration_prediction_and_risk_public_apis():
    calibration = survival.calibration(
        predicted_risk=[0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95],
        observed_event=[0, 0, 0, 1, 0, 1, 1, 1],
        n_groups=4,
    )
    prediction = survival.predict_cox(
        coef=[0.5, -0.25],
        x=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        baseline_hazard=[0.1, 0.2, 0.15],
        baseline_times=[1.0, 2.0, 3.0],
        pred_times=[1.5, 2.5],
    )
    stratification = survival.risk_stratification(
        risk_scores=[0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95],
        events=[0, 0, 0, 1, 0, 1, 1, 1],
        n_groups=3,
    )
    td_auc = survival.td_auc(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
        eval_times=[1.5, 2.5, 3.5],
    )

    assert calibration.n_per_group == [2, 2, 2, 2]
    assert calibration.predicted == pytest.approx([0.15, 0.35, 0.75, 0.925])
    assert calibration.observed == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert 0.0 <= calibration.hosmer_lemeshow_pvalue <= 1.0

    default_groups = survival.calibration([0.2, 0.8], [0, 1])
    assert default_groups.n_per_group == [1, 1]

    advanced_calibration = survival.advanced_calibration_metrics([0.2, 0.8], [0, 1], 1)
    assert advanced_calibration.ici >= 0.0

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.calibration([], [], 1)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.calibration([0.2], [], 1)
    with pytest.raises(ValueError, match="predicted_risk.*probabilities between 0 and 1"):
        survival.calibration([float("nan")], [1], 1)
    with pytest.raises(ValueError, match="predicted_risk.*probabilities between 0 and 1"):
        survival.calibration([1.2], [1], 1)
    with pytest.raises(ValueError, match="observed_event.*0/1"):
        survival.calibration([0.2], [2], 1)
    with pytest.raises(ValueError, match="n_groups must be positive"):
        survival.calibration([0.2], [1], 0)

    with pytest.raises(ValueError, match="predicted_risk and observed_outcome"):
        survival.advanced_calibration_metrics([], [], None)
    with pytest.raises(ValueError, match="predicted_risk and observed_outcome"):
        survival.advanced_calibration_metrics([0.2], [], None)
    with pytest.raises(ValueError, match="predicted_risk.*probabilities between 0 and 1"):
        survival.advanced_calibration_metrics([float("nan")], [1], None)
    with pytest.raises(ValueError, match="predicted_risk.*probabilities between 0 and 1"):
        survival.advanced_calibration_metrics([1.2], [1], None)
    with pytest.raises(ValueError, match="observed_outcome.*0/1"):
        survival.advanced_calibration_metrics([0.2], [2], None)
    with pytest.raises(ValueError, match="n_spline_knots must be positive"):
        survival.advanced_calibration_metrics([0.2], [1], 0)

    assert prediction.linear_predictor == pytest.approx([0.5, -0.25, 0.25])
    assert prediction.risk_score == pytest.approx(
        [1.6487212707001282, 0.7788007830714049, 1.2840254166877414]
    )
    assert prediction.times == pytest.approx([1.5, 2.5])
    assert len(prediction.survival_prob) == 3
    assert all(len(row) == 2 for row in prediction.survival_prob)

    with pytest.raises(ValueError, match="coef must be non-empty"):
        survival.predict_cox([], [[1.0]], [0.1], [1.0], [1.0])
    with pytest.raises(ValueError, match="x must contain at least one observation"):
        survival.predict_cox([0.5], [], [0.1], [1.0], [1.0])
    with pytest.raises(ValueError, match="x row 0 has 1 columns, expected 2"):
        survival.predict_cox([0.5, -0.25], [[1.0]], [0.1], [1.0], [1.0])
    with pytest.raises(ValueError, match="x.*finite"):
        survival.predict_cox([0.5], [[float("nan")]], [0.1], [1.0], [1.0])
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.predict_cox([0.5], [[1.0]], [], [], [1.0])
    with pytest.raises(ValueError, match="baseline_hazard must be non-negative"):
        survival.predict_cox([0.5], [[1.0]], [-0.1], [1.0], [1.0])
    with pytest.raises(ValueError, match="baseline_times must be strictly increasing"):
        survival.predict_cox([0.5], [[1.0]], [0.1, 0.2], [1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="pred_times must be non-empty"):
        survival.predict_cox([0.5], [[1.0]], [0.1], [1.0], [])

    assert stratification.cutpoints == pytest.approx([0.3, 0.8])
    assert stratification.group_sizes == [2, 3, 3]
    assert stratification.group_event_rates == pytest.approx([0.0, 1.0 / 3.0, 1.0])
    assert stratification.risk_groups == [0, 0, 1, 1, 1, 2, 2, 2]

    default_stratification = survival.risk_stratification([0.2, 0.8], [0, 1])
    assert default_stratification.group_sizes == [1, 1]

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.risk_stratification([], [], 1)
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.risk_stratification([0.2], [], 1)
    with pytest.raises(ValueError, match="risk_scores.*finite"):
        survival.risk_stratification([float("nan")], [1], 1)
    with pytest.raises(ValueError, match="events.*0/1"):
        survival.risk_stratification([0.2], [2], 1)
    with pytest.raises(ValueError, match="n_groups must be positive"):
        survival.risk_stratification([0.2], [1], 0)

    assert td_auc.times == pytest.approx([1.5, 2.5, 3.5])
    assert td_auc.auc == pytest.approx([1.0, 1.0, 1.0])
    assert td_auc.integrated_auc == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.td_auc([], [], [], [1.0])
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.td_auc([1.0], [], [0.5], [1.0])
    with pytest.raises(ValueError, match="time.*finite"):
        survival.td_auc([float("nan")], [1], [0.5], [1.0])
    with pytest.raises(ValueError, match="time must be non-negative"):
        survival.td_auc([-1.0], [1], [0.5], [1.0])
    with pytest.raises(ValueError, match="status.*0/1"):
        survival.td_auc([1.0], [2], [0.5], [1.0])
    with pytest.raises(ValueError, match="risk_score.*finite"):
        survival.td_auc([1.0], [1], [float("inf")], [1.0])
    with pytest.raises(ValueError, match="eval_times must be non-empty"):
        survival.td_auc([1.0], [1], [0.5], [])
    with pytest.raises(ValueError, match="eval_times.*finite"):
        survival.td_auc([1.0], [1], [0.5], [float("nan")])
    with pytest.raises(ValueError, match="eval_times must be non-negative"):
        survival.td_auc([1.0], [1], [0.5], [-1.0])
    with pytest.raises(ValueError, match="eval_times must be sorted"):
        survival.td_auc([1.0, 2.0], [1, 0], [0.8, 0.2], [2.0, 1.0])

    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    boundary_status = [1, 1, 0, 0]
    boundary_risk = [0.7, 0.9, 0.8, 0.1]

    exact_auc = survival.td_auc(exact_time, boundary_status, boundary_risk, [1.0, 2.0])
    near_auc = survival.td_auc(near_time, boundary_status, boundary_risk, [1.0, 2.0])
    assert near_auc.times == pytest.approx(exact_auc.times)
    assert near_auc.auc == pytest.approx(exact_auc.auc)
    assert near_auc.integrated_auc == pytest.approx(exact_auc.integrated_auc)

    boundary_survival = [[0.7, 0.6], [0.4, 0.3], [0.8, 0.7], [0.9, 0.8]]
    exact_calibration = survival.time_dependent_calibration(
        exact_time, boundary_status, boundary_survival, [1.0, 2.0]
    )
    near_calibration = survival.time_dependent_calibration(
        near_time, boundary_status, boundary_survival, [1.0, 2.0]
    )
    assert near_calibration.time_points == pytest.approx(exact_calibration.time_points)
    assert near_calibration.ici == pytest.approx(exact_calibration.ici)
    assert near_calibration.e50 == pytest.approx(exact_calibration.e50)
    assert near_calibration.e90 == pytest.approx(exact_calibration.e90)
    assert near_calibration.calibration_slope == pytest.approx(exact_calibration.calibration_slope)
    assert near_calibration.calibration_intercept == pytest.approx(
        exact_calibration.calibration_intercept
    )

    with pytest.raises(ValueError, match="same non-zero length"):
        survival.time_dependent_calibration([], [], [], [1.0])
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.time_dependent_calibration([1.0], [], [[0.9]], [1.0])
    with pytest.raises(ValueError, match="same non-zero length"):
        survival.time_dependent_calibration([1.0], [1], [], [1.0])
    with pytest.raises(ValueError, match="time.*finite"):
        survival.time_dependent_calibration([float("nan")], [1], [[0.9]], [1.0])
    with pytest.raises(ValueError, match="time must be non-negative"):
        survival.time_dependent_calibration([-1.0], [1], [[0.9]], [1.0])
    with pytest.raises(ValueError, match="event.*0/1"):
        survival.time_dependent_calibration([1.0], [2], [[0.9]], [1.0])
    with pytest.raises(ValueError, match="eval_times must be non-empty"):
        survival.time_dependent_calibration([1.0], [1], [[0.9]], [])
    with pytest.raises(ValueError, match="eval_times.*finite"):
        survival.time_dependent_calibration([1.0], [1], [[0.9]], [float("nan")])
    with pytest.raises(ValueError, match="eval_times must be non-negative"):
        survival.time_dependent_calibration([1.0], [1], [[0.9]], [-1.0])
    with pytest.raises(ValueError, match="eval_times must be sorted"):
        survival.time_dependent_calibration([1.0], [1], [[0.9, 0.8]], [2.0, 1.0])
    with pytest.raises(ValueError, match="predicted_survival row 0 has 1 elements, expected 2"):
        survival.time_dependent_calibration([1.0], [1], [[0.9]], [1.0, 2.0])
    with pytest.raises(ValueError, match="predicted_survival.*probabilities"):
        survival.time_dependent_calibration([1.0], [1], [[1.2]], [1.0])

    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        survival.d_calibration([0.1], [1], 1)


def test_conformal_coverage_cv_public_api_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 1]
    predicted = [1.1, 2.1, 3.1, 4.1]

    result = survival.conformal_coverage_cv(
        time,
        status,
        predicted,
        n_folds=2,
        coverage_candidates=[0.8, 0.9],
        seed=42,
    )

    assert result.coverage_candidates == pytest.approx([0.8, 0.9])
    assert len(result.mean_widths) == 2
    assert len(result.empirical_coverages) == 2
    assert result.optimal_coverage in result.coverage_candidates

    default_result = survival.conformal_coverage_cv(time[:3], status[:3], predicted[:3])
    assert default_result.coverage_candidates == pytest.approx([0.8, 0.85, 0.9, 0.95, 0.99])

    with pytest.raises(ValueError, match="At least two observations"):
        survival.conformal_coverage_cv([1.0], [1], [1.0])

    with pytest.raises(ValueError, match="same length"):
        survival.conformal_coverage_cv([1.0, 2.0], [1], [1.0, 2.0])

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.conformal_coverage_cv([float("nan"), 2.0], [1, 1], [1.0, 2.0])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.conformal_coverage_cv([1.0, 2.0], [1, 2], [1.0, 2.0])

    with pytest.raises(ValueError, match="predicted contains non-finite"):
        survival.conformal_coverage_cv([1.0, 2.0], [1, 1], [1.0, float("inf")])

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        survival.conformal_coverage_cv(time, status, predicted, n_folds=1)

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        survival.conformal_coverage_cv(time, status, predicted, n_folds=5)

    with pytest.raises(ValueError, match="coverage_candidates cannot be empty"):
        survival.conformal_coverage_cv(time, status, predicted, coverage_candidates=[])

    with pytest.raises(ValueError, match="coverage_candidates must contain finite"):
        survival.conformal_coverage_cv(time, status, predicted, coverage_candidates=[0.9, 1.0])

    with pytest.raises(ValueError, match="At least one uncensored"):
        survival.conformal_coverage_cv([1.0, 2.0], [0, 0], [1.0, 2.0])


def test_timepoint_calibration_public_apis_and_validation():
    time = list(range(1, 21))
    status = [1] * 20
    predicted = [0.99 - i * 0.03 for i in range(20)]

    d_cal = survival.d_calibration(
        survival_probs=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] * 2,
        status=[1] * 20,
        n_bins=5,
    )
    one = survival.one_calibration(time, status, predicted, 10.0, 4)
    plot = survival.calibration_plot(time, status, predicted, 10.0, 4)
    brier = survival.brier_calibration(time, status, predicted, 10.0, 4)
    multi = survival.multi_time_calibration(
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        status=[1, 1, 0, 1, 0, 1],
        survival_predictions=[
            [0.95, 0.9],
            [0.9, 0.8],
            [0.98, 0.95],
            [0.85, 0.7],
            [0.99, 0.97],
            [0.8, 0.6],
        ],
        prediction_times=[2.0, 4.0],
        n_groups=2,
    )
    smooth = survival.smoothed_calibration(
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        status=[1, 1, 0, 1, 0, 1],
        predicted_survival_at_t=[0.95, 0.9, 0.98, 0.85, 0.99, 0.8],
        time_point=4.0,
        n_grid_points=10,
        bandwidth=0.1,
    )

    assert d_cal.n_bins == 5
    assert d_cal.n_events == 20
    assert d_cal.observed_counts == [4, 4, 4, 4, 4]
    assert d_cal.p_value == pytest.approx(1.0)
    assert d_cal.is_calibrated is True

    assert one.time_point == pytest.approx(10.0)
    assert one.n_groups == 4
    assert one.n_per_group == [5, 5, 5, 5]
    assert one.n_events_per_group == [0, 0, 5, 5]
    assert one.is_calibrated is False

    assert plot.predicted == pytest.approx([0.4800000000000001, 0.63, 0.78, 0.93])
    assert plot.observed == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert plot.ici == pytest.approx(0.65)
    assert plot.emax >= plot.e90 >= plot.e50

    assert brier.time_point == pytest.approx(10.0)
    assert brier.brier_score == pytest.approx(0.47195)
    assert brier.calibration_slope == pytest.approx(-2.666666666666667)
    assert brier.ici == pytest.approx(plot.ici)

    assert multi.time_points == pytest.approx([2.0, 4.0])
    assert len(multi.brier_scores) == 2
    assert len(multi.calibration_slopes) == 2
    assert multi.integrated_brier > 0.0
    assert multi.mean_ici > 0.0

    assert len(smooth.predicted_grid) == 10
    assert len(smooth.smoothed_observed) == 10
    assert smooth.bandwidth == pytest.approx(0.1)
    assert all(0.0 <= value <= 1.0 for value in smooth.ci_lower)
    assert all(0.0 <= value <= 1.0 for value in smooth.ci_upper)

    exact_time = [1.0, 2.0, 2.0, 2.0, 3.0, 1.5, 2.0, 2.5, 3.5, 4.0]
    near_time = [1.0, 2.0 + 5e-10, 2.0 + 5e-10, 2.0, 3.0, 1.5, 2.0, 2.5, 3.5, 4.0]
    boundary_status = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    boundary_predicted = [0.92, 0.84, 0.78, 0.7, 0.62, 0.56, 0.48, 0.4, 0.32, 0.24]

    exact_one = survival.one_calibration(exact_time, boundary_status, boundary_predicted, 2.0, 2)
    near_one = survival.one_calibration(near_time, boundary_status, boundary_predicted, 2.0, 2)
    assert near_one.n_events_per_group == exact_one.n_events_per_group
    assert near_one.observed_survival == pytest.approx(exact_one.observed_survival)
    assert near_one.statistic == pytest.approx(exact_one.statistic)

    exact_plot = survival.calibration_plot(exact_time, boundary_status, boundary_predicted, 2.0, 2)
    near_plot = survival.calibration_plot(near_time, boundary_status, boundary_predicted, 2.0, 2)
    assert near_plot.observed == pytest.approx(exact_plot.observed)
    assert near_plot.ici == pytest.approx(exact_plot.ici)
    assert near_plot.e50 == pytest.approx(exact_plot.e50)
    assert near_plot.e90 == pytest.approx(exact_plot.e90)
    assert near_plot.emax == pytest.approx(exact_plot.emax)

    exact_brier = survival.brier_calibration(
        exact_time, boundary_status, boundary_predicted, 2.0, 2
    )
    near_brier = survival.brier_calibration(near_time, boundary_status, boundary_predicted, 2.0, 2)
    assert near_brier.brier_score == pytest.approx(exact_brier.brier_score)
    assert near_brier.observed == pytest.approx(exact_brier.observed)

    exact_smooth = survival.smoothed_calibration(
        exact_time, boundary_status, boundary_predicted, 2.0, 10, 0.2
    )
    near_smooth = survival.smoothed_calibration(
        near_time, boundary_status, boundary_predicted, 2.0, 10, 0.2
    )
    assert near_smooth.predicted_grid == pytest.approx(exact_smooth.predicted_grid)
    assert near_smooth.smoothed_observed == pytest.approx(exact_smooth.smoothed_observed)

    boundary_survival_predictions = [[value, max(value - 0.1, 0.0)] for value in boundary_predicted]
    exact_multi = survival.multi_time_calibration(
        exact_time, boundary_status, boundary_survival_predictions, [2.0, 3.0], 2
    )
    near_multi = survival.multi_time_calibration(
        near_time, boundary_status, boundary_survival_predictions, [2.0, 3.0], 2
    )
    assert near_multi.brier_scores == pytest.approx(exact_multi.brier_scores)
    assert near_multi.ici_values == pytest.approx(exact_multi.ici_values)
    assert near_multi.integrated_brier == pytest.approx(exact_multi.integrated_brier)

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.one_calibration([1.0], [1, 0], [0.9], 1.0, 2)

    with pytest.raises(ValueError, match="survival_predictions row 0 has 2 elements, expected 1"):
        survival.multi_time_calibration([1.0], [1], [[0.9, 0.8]], [1.0], 2)

    with pytest.raises(ValueError, match="n_grid_points must be at least 10"):
        survival.smoothed_calibration([1.0], [1], [0.9], 1.0, 5, None)

    with pytest.raises(ValueError, match="probabilities between 0 and 1"):
        survival.d_calibration([1.2], [1])

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.d_calibration([0.8], [2])

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.one_calibration([float("nan")], [1], [0.9], 1.0, 2)

    with pytest.raises(ValueError, match="predicted_survival_at_t.*probabilities"):
        survival.brier_calibration([1.0], [1], [1.1], 1.0, 2)

    with pytest.raises(ValueError, match="prediction_times must be sorted"):
        survival.multi_time_calibration([1.0], [1], [[0.9, 0.8]], [2.0, 1.0], 2)

    with pytest.raises(ValueError, match="bandwidth must be positive"):
        survival.smoothed_calibration([1.0], [1], [0.9], 1.0, 10, 0.0)


def test_reporting_public_apis_and_validation():
    km = survival.km_plot_data([1.0, 2.0, 3.0, 4.0, 5.0], [1, 0, 1, 0, 1], 0.95, "Test")
    forest = survival.forest_plot_data(["age", "trt"], [0.2, -0.3], [0.1, 0.15], 0.95)
    calibration_curve = survival.calibration_plot_data(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0, 0, 0, 1, 0, 1, 1, 1],
        4,
    )
    report = survival.generate_survival_report(
        "Test Report",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1, 0, 1, 0, 1],
        [2.0, 4.0],
    )
    roc = survival.roc_plot_data([0.9, 0.8, 0.7, 0.6, 0.3, 0.2], [1, 1, 1, 0, 0, 0])

    assert km.group_name == "Test"
    assert km.time_points == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert km.at_risk == [5, 5, 4, 3, 2, 1]
    assert km.survival_prob[0] == pytest.approx(1.0)

    assert forest.hazard_ratios == pytest.approx([1.2214027581601699, 0.7408182206817179])
    assert forest.significant_at(0.2) == [True, True]
    assert all(value > 0.0 for value in forest.lower_ci)
    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        forest.significant_at(float("nan"))
    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        forest.significant_at(0.0)
    with pytest.raises(ValueError, match="alpha must be finite and between 0 and 1"):
        forest.significant_at(1.0)

    assert calibration_curve.predicted_prob == pytest.approx([0.15, 0.35, 0.55, 0.75])
    assert calibration_curve.observed_prob == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert calibration_curve.n_per_bin == [2, 2, 2, 2]
    assert len(calibration_curve.bin_boundaries) == 5

    assert report.title == "Test Report"
    assert report.n_subjects == 5
    assert report.n_events == 3
    assert report.median_survival == pytest.approx(5.0)
    assert report.rmst == pytest.approx(3.666666666666667)
    assert "# Test Report" in report.to_markdown()
    assert "\\section{Test Report}" in report.to_latex()

    assert roc.auc == pytest.approx(1.0)
    assert roc.fpr[0] == pytest.approx(0.0)
    assert roc.tpr[-1] == pytest.approx(1.0)
    assert roc.optimal_threshold("youden") == pytest.approx(0.7)
    assert roc.optimal_threshold("closest_topleft") == pytest.approx(0.7)
    assert roc.optimal_threshold("closest-top-left") == pytest.approx(0.7)
    with pytest.raises(ValueError, match="method must be 'youden' or 'closest_topleft'"):
        roc.optimal_threshold("typo")

    reversed_roc = survival.roc_plot_data([0.1, 0.9], [1, 0])
    assert reversed_roc.optimal_threshold("youden") == pytest.approx(0.1)
    assert reversed_roc.optimal_threshold("closest_topleft") == pytest.approx(0.1)

    with pytest.raises(ValueError, match="time and event must have the same non-zero length"):
        survival.km_plot_data([], [], 0.95, None)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.km_plot_data([float("nan")], [1], 0.95, None)

    with pytest.raises(ValueError, match="event.*0/1"):
        survival.km_plot_data([1.0], [2], 0.95, None)

    with pytest.raises(ValueError, match="confidence_level"):
        survival.km_plot_data([1.0], [1], 1.0, None)

    with pytest.raises(ValueError, match="variable_names must be non-empty"):
        survival.forest_plot_data([], [], [], 0.95)

    with pytest.raises(ValueError, match="variable_names must not contain empty names"):
        survival.forest_plot_data([" "], [0.1], [0.1], 0.95)

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.forest_plot_data(["x"], [0.1, 0.2], [0.1], 0.95)

    with pytest.raises(ValueError, match="coefficients contains non-finite"):
        survival.forest_plot_data(["x"], [float("nan")], [0.1], 0.95)

    with pytest.raises(ValueError, match="standard_errors must contain positive values"):
        survival.forest_plot_data(["x"], [0.1], [0.0], 0.95)

    with pytest.raises(ValueError, match="hazard ratios and confidence intervals must be finite"):
        survival.forest_plot_data(["x"], [1000.0], [0.1], 0.95)

    with pytest.raises(ValueError, match="confidence_level"):
        survival.forest_plot_data(["x"], [0.1], [0.1], float("nan"))

    with pytest.raises(
        ValueError,
        match="predicted and observed must have the same non-zero length",
    ):
        survival.calibration_plot_data([0.1], [0, 1], 2)

    with pytest.raises(ValueError, match="n_bins must be between 1 and the number of observations"):
        survival.calibration_plot_data([0.1], [0], 0)

    with pytest.raises(ValueError, match="n_bins must be between 1 and the number of observations"):
        survival.calibration_plot_data([0.1], [0], 2)

    with pytest.raises(ValueError, match="predicted.*probabilities between 0 and 1"):
        survival.calibration_plot_data([1.1], [1], 1)

    with pytest.raises(ValueError, match="observed.*0/1"):
        survival.calibration_plot_data([0.1], [2], 1)

    with pytest.raises(ValueError, match="title must be non-empty"):
        survival.generate_survival_report(" ", [1.0], [1], None)

    with pytest.raises(ValueError, match="landmark_times contains non-finite"):
        survival.generate_survival_report("bad", [1.0], [1], [float("inf")])

    with pytest.raises(ValueError, match="Both positive and negative labels required"):
        survival.roc_plot_data([0.1, 0.2], [1, 1])

    with pytest.raises(ValueError, match="scores contains non-finite"):
        survival.roc_plot_data([float("nan"), 0.2], [1, 0])

    with pytest.raises(ValueError, match="labels.*0/1"):
        survival.roc_plot_data([0.1, 0.2], [1, 2])

    tied_roc = survival.roc_plot_data([0.5, 0.5, 0.2, 0.8], [1, 0, 0, 1])
    assert tied_roc.thresholds == pytest.approx([float("inf"), 0.8, 0.5, 0.2])
    assert tied_roc.auc == pytest.approx(0.875)


def test_decision_curve_public_apis_and_validation():
    result = survival.decision_curve_analysis(
        predicted_risk=[0.1, 0.3, 0.5, 0.7, 0.9],
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        event=[0, 1, 0, 1, 1],
        time_horizon=3.0,
        thresholds=[0.25, 0.5, 0.75],
    )
    comparison = survival.compare_decision_curves(
        model_predictions=[
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.2, 0.2, 0.4, 0.6, 0.8],
        ],
        model_names=["m1", "m2"],
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        event=[0, 1, 0, 1, 1],
        time_horizon=3.0,
        thresholds=[0.25, 0.5, 0.75],
    )

    assert result.thresholds == pytest.approx([0.25, 0.5, 0.75])
    assert result.net_benefit_none == pytest.approx([0.0, 0.0, 0.0])
    assert result.optimal_threshold() == pytest.approx(0.25)
    assert result.area_under_curve() == pytest.approx(0.0)

    assert comparison.model_names == ["m1", "m2"]
    assert comparison.best_model_per_threshold == ["m1", "m2", "m2"]
    assert len(comparison.net_benefit_difference) == 2
    assert len(comparison.net_benefit_difference[0]) == 2

    exact_time = [1.0, 2.0, 2.0, 3.0, 4.0]
    near_time = [1.0, 2.0 + 5e-10, 2.0 + 5e-10, 3.0, 4.0]
    boundary_event = [0, 1, 0, 1, 0]
    boundary_risk = [0.1, 0.8, 0.6, 0.7, 0.2]
    challenger_risk = [0.2, 0.7, 0.4, 0.8, 0.3]
    boundary_thresholds = [0.25, 0.5, 0.75]

    exact_curve = survival.decision_curve_analysis(
        boundary_risk, exact_time, boundary_event, 2.0, boundary_thresholds
    )
    near_curve = survival.decision_curve_analysis(
        boundary_risk, near_time, boundary_event, 2.0, boundary_thresholds
    )
    assert near_curve.net_benefit == pytest.approx(exact_curve.net_benefit)
    assert near_curve.net_benefit_all == pytest.approx(exact_curve.net_benefit_all)
    assert near_curve.interventions_avoided == pytest.approx(exact_curve.interventions_avoided)

    exact_clinical = survival.clinical_utility_at_threshold(
        boundary_risk, exact_time, boundary_event, 2.0, 0.5
    )
    near_clinical = survival.clinical_utility_at_threshold(
        boundary_risk, near_time, boundary_event, 2.0, 0.5
    )
    assert near_clinical.sensitivity == pytest.approx(exact_clinical.sensitivity)
    assert near_clinical.specificity == pytest.approx(exact_clinical.specificity)
    assert near_clinical.ppv == pytest.approx(exact_clinical.ppv)
    assert near_clinical.npv == pytest.approx(exact_clinical.npv)
    assert near_clinical.net_benefit == pytest.approx(exact_clinical.net_benefit)

    exact_curve_comparison = survival.compare_decision_curves(
        [boundary_risk, challenger_risk],
        ["m1", "m2"],
        exact_time,
        boundary_event,
        2.0,
        boundary_thresholds,
    )
    near_curve_comparison = survival.compare_decision_curves(
        [boundary_risk, challenger_risk],
        ["m1", "m2"],
        near_time,
        boundary_event,
        2.0,
        boundary_thresholds,
    )
    for actual_row, expected_row in zip(
        near_curve_comparison.net_benefit_difference,
        exact_curve_comparison.net_benefit_difference,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert (
        near_curve_comparison.best_model_per_threshold
        == exact_curve_comparison.best_model_per_threshold
    )

    with pytest.raises(ValueError, match="All inputs must have the same non-zero length"):
        survival.decision_curve_analysis([], [], [], 1.0, None)

    with pytest.raises(ValueError, match="predicted_risk.*probabilities between 0 and 1"):
        survival.decision_curve_analysis([1.2], [1.0], [1], 1.0, [0.5])

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.decision_curve_analysis([0.2], [float("nan")], [1], 1.0, [0.5])

    with pytest.raises(ValueError, match="event.*0/1"):
        survival.decision_curve_analysis([0.2], [1.0], [2], 1.0, [0.5])

    with pytest.raises(ValueError, match="time_horizon"):
        survival.decision_curve_analysis([0.2], [1.0], [1], float("inf"), [0.5])

    with pytest.raises(ValueError, match="thresholds cannot be empty"):
        survival.decision_curve_analysis([0.2], [1.0], [1], 1.0, [])

    with pytest.raises(ValueError, match="thresholds.*between 0 and 1 exclusive"):
        survival.decision_curve_analysis([0.2], [1.0], [1], 1.0, [1.0])

    with pytest.raises(ValueError, match="threshold.*between 0 and 1 exclusive"):
        survival.clinical_utility_at_threshold([0.2], [1.0], [1], 1.0, 0.0)

    with pytest.raises(
        ValueError,
        match="model_predictions and model_names must have the same non-zero length",
    ):
        survival.compare_decision_curves([[0.1]], [], [1.0], [1], 1.0, None)

    with pytest.raises(ValueError, match="model_predictions row 1 length mismatch"):
        survival.compare_decision_curves([[0.1], [0.1, 0.2]], ["m1", "m2"], [1.0], [1], 1.0, [0.5])

    with pytest.raises(ValueError, match="model_predictions.*probabilities between 0 and 1"):
        survival.compare_decision_curves([[1.2]], ["m1"], [1.0], [1], 1.0, [0.5])


def test_fairness_public_api_groups_near_tied_threshold_events():
    risk = [0.1, 0.9, 0.7, 0.6, 0.95, 0.2]
    exact_time = [1.0, 2.0, 2.0, 3.0, 2.0, 4.0]
    near_time = [1.0, 2.0 + 5e-10, 2.0, 3.0, 2.0, 4.0]
    event = [0, 1, 0, 1, 1, 0]
    protected = [0, 0, 0, 1, 1, 1]

    exact = survival.compute_fairness_metrics(risk, exact_time, event, protected, 2.0)
    near = survival.compute_fairness_metrics(risk, near_time, event, protected, 2.0)

    assert near.group_sizes == exact.group_sizes
    assert near.demographic_parity == pytest.approx(exact.demographic_parity)
    assert near.equalized_odds == pytest.approx(exact.equalized_odds)

    with pytest.raises(ValueError, match="risk_scores contains non-finite"):
        survival.compute_fairness_metrics([float("nan")], [1.0], [1], [0], None)

    with pytest.raises(ValueError, match="time contains negative"):
        survival.compute_fairness_metrics([0.5], [-1.0], [1], [0], None)

    with pytest.raises(ValueError, match="event values must be 0 or 1"):
        survival.compute_fairness_metrics([0.5], [1.0], [2], [0], None)

    with pytest.raises(ValueError, match="protected_attribute"):
        survival.compute_fairness_metrics([0.5], [1.0], [1], [], None)

    with pytest.raises(ValueError, match="threshold_time must be finite and non-negative"):
        survival.compute_fairness_metrics([0.5], [1.0], [1], [0], float("nan"))

    with pytest.raises(ValueError, match="noise_levels must not be empty"):
        survival.assess_model_robustness(risk, exact_time, event, [], 10, 42)

    with pytest.raises(ValueError, match="noise_levels must contain finite non-negative"):
        survival.assess_model_robustness(risk, exact_time, event, [float("inf")], 10, 42)

    with pytest.raises(ValueError, match="n_perturbations must be positive"):
        survival.assess_model_robustness(risk, exact_time, event, None, 0, 42)

    with pytest.raises(ValueError, match="subgroup_labels must not contain empty labels"):
        survival.subgroup_analysis(risk, exact_time, event, ["a", " ", "a", "b", "b", "b"])
