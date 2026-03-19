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

    assert prediction.linear_predictor == pytest.approx([0.5, -0.25, 0.25])
    assert prediction.risk_score == pytest.approx(
        [1.6487212707001282, 0.7788007830714049, 1.2840254166877414]
    )
    assert prediction.times == pytest.approx([1.5, 2.5])
    assert len(prediction.survival_prob) == 3
    assert all(len(row) == 2 for row in prediction.survival_prob)

    assert stratification.cutpoints == pytest.approx([0.3, 0.8])
    assert stratification.group_sizes == [2, 3, 3]
    assert stratification.group_event_rates == pytest.approx([0.0, 1.0 / 3.0, 1.0])
    assert stratification.risk_groups == [0, 0, 1, 1, 1, 2, 2, 2]

    assert td_auc.times == pytest.approx([1.5, 2.5, 3.5])
    assert td_auc.auc == pytest.approx([1.0, 1.0, 1.0])
    assert td_auc.integrated_auc == pytest.approx(1.0)

    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        survival.d_calibration([0.1], [1], 1)


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

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.one_calibration([1.0], [1, 0], [0.9], 1.0, 2)

    with pytest.raises(ValueError, match="survival_predictions row 0 has 2 elements, expected 1"):
        survival.multi_time_calibration([1.0], [1], [[0.9, 0.8]], [1.0], 2)

    with pytest.raises(ValueError, match="n_grid_points must be at least 10"):
        survival.smoothed_calibration([1.0], [1], [0.9], 1.0, 5, None)


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

    with pytest.raises(ValueError, match="time and event must have the same non-zero length"):
        survival.km_plot_data([], [], 0.95, None)

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.forest_plot_data(["x"], [0.1, 0.2], [0.1], 0.95)

    with pytest.raises(
        ValueError,
        match="predicted and observed must have the same non-zero length",
    ):
        survival.calibration_plot_data([0.1], [0, 1], 2)

    with pytest.raises(ValueError, match="Both positive and negative labels required"):
        survival.roc_plot_data([0.1, 0.2], [1, 1])


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

    with pytest.raises(ValueError, match="All inputs must have the same non-zero length"):
        survival.decision_curve_analysis([], [], [], 1.0, None)

    with pytest.raises(
        ValueError,
        match="model_predictions and model_names must have the same non-zero length",
    ):
        survival.compare_decision_curves([[0.1]], [], [1.0], [1], 1.0, None)
