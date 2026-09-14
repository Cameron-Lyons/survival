import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_brier_validates_prediction_shape():
    time, status = [1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1]
    result = survival.validation.brier(time, status, [2.5], [[0.5, 0.5, 0.2, 0.9]])
    assert result.times == pytest.approx([2.5])
    assert len(result.brier) == 1
    assert 0.0 <= result.brier[0] <= 1.0

    with pytest.raises(ValueError, match="phat"):
        survival.validation.brier(time, status, [2.5], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="phat"):
        survival.validation.brier(time, status, [2.5, 3.5], [[0.5, 0.5, 0.2, 0.9]])


def test_statistical_test_helpers():
    lrt = survival.validation.lrt_test(-10.0, -12.0, 1)
    assert lrt.statistic == pytest.approx(4.0)
    assert lrt.df == 1
    assert lrt.p_value == pytest.approx(0.0455, rel=1e-3)
    assert lrt.test_name == "LikelihoodRatioTest"

    # coxph.wtest(var, coef): coef' V^-1 coef with V diagonal (1, 2)
    wald = survival.validation.wald_test([1.0, 2.0], [[1.0, 0.0], [0.0, 2.0]])
    assert wald.statistic == pytest.approx(3.0)
    assert wald.df == 2
    assert wald.p_value == pytest.approx(0.22313016014842982)
    assert wald.test_name == "WaldTest"
    assert survival.validation.wald_test(
        [1.0, 2.0], [[1.0, 0.0], [0.0, 2.0]], init=[1.0, 0.0]
    ).statistic == pytest.approx(2.0)

    score = survival.validation.score_test([1.0], [[2.0]])
    assert score.statistic == pytest.approx(0.5)
    assert score.df == 1
    assert score.p_value == pytest.approx(0.4795, rel=1e-3)
    assert score.test_name == "ScoreTest"

    with pytest.raises(ValueError, match="log-likelihoods must be finite"):
        survival.validation.lrt_test(float("nan"), -12.0, 1)
    with pytest.raises(ValueError, match="df must be positive"):
        survival.validation.lrt_test(-10.0, -12.0, 0)
    with pytest.raises(ValueError, match="coef must not be empty"):
        survival.validation.wald_test([], [])
    with pytest.raises(ValueError, match="coef contains non-finite"):
        survival.validation.wald_test([float("inf")], [[1.0]])
    with pytest.raises(ValueError, match="var columns length mismatch"):
        survival.validation.wald_test([1.0], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="var rows length mismatch"):
        survival.validation.score_test([1.0, 2.0], [[1.0]])
    with pytest.raises(ValueError, match="score must not be empty"):
        survival.validation.score_test([], [])
    with pytest.raises(ValueError, match="var contains non-finite"):
        survival.validation.score_test([1.0], [[float("nan")]])


def test_meta_analysis_public_apis_and_validation():
    effects = [0.5, 0.7, 0.4, 0.6]
    std_errors = [0.1, 0.15, 0.12, 0.11]
    config = survival.validation.MetaAnalysisConfig("fixed", 0.95, "dl")

    result = survival.validation.survival_meta_analysis(effects, std_errors, config)
    assert result.pooled_effect > 0.0
    assert result.pooled_se > 0.0
    assert sum(result.study_weights) == pytest.approx(1.0)
    assert 0.0 <= result.i_squared <= 100.0

    forest = survival.validation.generate_forest_plot_data(
        ["A", "B", "C", "D"], effects, std_errors, None
    )
    assert forest.study_names == ["A", "B", "C", "D"]
    assert len(forest.weights) == 4
    assert forest.pooled_effect == pytest.approx(
        survival.validation.survival_meta_analysis(effects, std_errors, None).pooled_effect
    )

    bias = survival.validation.publication_bias_tests(effects, std_errors)
    assert 0.0 <= bias.egger_p <= 1.0
    assert 0.0 <= bias.begg_p <= 1.0

    with pytest.raises(ValueError, match="method must be"):
        survival.validation.MetaAnalysisConfig("bad", 0.95, "dl")
    with pytest.raises(ValueError, match="confidence_level"):
        survival.validation.MetaAnalysisConfig("random", 1.0, "dl")
    with pytest.raises(ValueError, match="tau_method must be"):
        survival.validation.MetaAnalysisConfig("random", 0.95, "bad")
    with pytest.raises(ValueError, match="effects contains non-finite"):
        survival.validation.survival_meta_analysis([0.5, float("nan")], [0.1, 0.2], None)
    with pytest.raises(ValueError, match="std_errors must contain positive values"):
        survival.validation.survival_meta_analysis([0.5, 0.7], [0.1, 0.0], None)
    with pytest.raises(ValueError, match="Need at least 3 studies"):
        survival.validation.publication_bias_tests([0.5, 0.7], [0.1, 0.2])


def test_uncertainty_interval_helpers_validate_inputs():
    predictions = [
        [[0.9, 0.8], [0.7, 0.6]],
        [[0.85, 0.75], [0.65, 0.55]],
        [[0.95, 0.82], [0.75, 0.62]],
    ]

    ensemble = survival.validation.ensemble_uncertainty(predictions, 0.95)
    quantiles = survival.validation.quantile_regression_intervals(predictions, [0.1, 0.5, 0.9])
    calibration = survival.validation.calibrate_prediction_intervals(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [0.5, 1.5, 2.5],
        [1.5, 2.5, 3.5],
        0.9,
    )

    assert len(ensemble.mean_prediction) == 2
    assert len(ensemble.prediction_intervals[0]) == 2
    assert quantiles.quantiles == [0.1, 0.5, 0.9]
    assert quantiles.prediction_interval_width()[0][0] >= 0.0
    assert calibration.observed_coverage == pytest.approx(1.0)

    with pytest.raises(ValueError, match="model_predictions must be rectangular"):
        survival.validation.ensemble_uncertainty([[[0.9]], []], 0.95)
    with pytest.raises(ValueError, match="confidence_level"):
        survival.validation.ensemble_uncertainty([[[0.9]], [[0.8]]], 1.0)
    with pytest.raises(ValueError, match="quantiles must contain exactly three"):
        survival.validation.quantile_regression_intervals(predictions, [0.1, 0.9])
    with pytest.raises(ValueError, match="quantiles must be nondecreasing"):
        survival.validation.quantile_regression_intervals(predictions, [0.5, 0.1, 0.9])
    with pytest.raises(ValueError, match="bootstrap_predictions contains non-finite"):
        survival.validation.quantile_regression_intervals([[[float("nan")]]], None)
    with pytest.raises(ValueError, match="true_events values"):
        survival.validation.calibrate_prediction_intervals([1.0], [2], [0.0], [2.0], 0.9)
    with pytest.raises(ValueError, match="lower_bounds must be less than or equal"):
        survival.validation.calibrate_prediction_intervals([1.0], [1], [2.0], [1.0], 0.9)


def test_bayesian_bootstrap_survival_groups_near_tied_event_times():
    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    event = [1, 1, 0, 0]
    eval_times = [1.0, 2.0, 3.0]
    config = survival.validation.BayesianBootstrapConfig(25, 0.8, 7)

    exact = survival.validation.bayesian_bootstrap_survival(exact_time, event, eval_times, config)
    near = survival.validation.bayesian_bootstrap_survival(near_time, event, eval_times, config)

    assert near.mean_survival == pytest.approx(exact.mean_survival)
    assert near.lower_ci == pytest.approx(exact.lower_ci)
    assert near.upper_ci == pytest.approx(exact.upper_ci)
    for near_row, exact_row in zip(near.posterior_samples, exact.posterior_samples, strict=True):
        assert near_row == pytest.approx(exact_row)

    with pytest.raises(ValueError, match="event length mismatch"):
        survival.validation.bayesian_bootstrap_survival([1.0], [1, 0], [1.0], config)

    with pytest.raises(ValueError, match="event.*0/1"):
        survival.validation.bayesian_bootstrap_survival([1.0], [2], [1.0], config)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.validation.bayesian_bootstrap_survival([float("nan")], [1], [1.0], config)

    bad_config = survival.validation.BayesianBootstrapConfig(0, 0.95, 1)
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        survival.validation.bayesian_bootstrap_survival([1.0], [1], [1.0], bad_config)


def test_conformal_and_jackknife_uncertainty_validate_inputs():
    conformal_config = survival.validation.ConformalSurvivalConfig(0.1, "cqr", 100, 7)
    conformal = survival.validation.conformal_survival(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [1.1, 2.2, 2.8],
        [1.5, 2.5],
        conformal_config,
    )
    assert len(conformal.lower_bounds) == 2
    assert len(conformal.calibration_scores) == 3

    with pytest.raises(ValueError, match="cal_event length mismatch"):
        survival.validation.conformal_survival([1.0], [1, 0], [1.0], [1.0], conformal_config)
    with pytest.raises(ValueError, match="cal_predictions contains non-finite"):
        survival.validation.conformal_survival([1.0], [1], [float("nan")], [1.0], conformal_config)
    with pytest.raises(ValueError, match="cal_event.*0/1"):
        survival.validation.conformal_survival([1.0], [2], [1.0], [1.0], conformal_config)
    with pytest.raises(ValueError, match="test_predictions contains non-finite"):
        survival.validation.conformal_survival([1.0], [1], [1.0], [float("inf")], conformal_config)
    with pytest.raises(ValueError, match="method must be"):
        survival.validation.conformal_survival(
            [1.0],
            [1],
            [1.0],
            [1.0],
            survival.validation.ConformalSurvivalConfig(0.1, "unknown", 100, None),
        )
    with pytest.raises(ValueError, match="positive value"):
        survival.validation.conformal_survival(
            [0.0],
            [0],
            [0.0],
            [1.0],
            survival.validation.ConformalSurvivalConfig(0.1, "censoring_adjusted", 100, None),
        )

    jackknife_config = survival.validation.JackknifePlusConfig(0.1, True, 5)
    jackknife = survival.validation.jackknife_plus_survival(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [[], [], []],
        jackknife_config,
    )
    assert len(jackknife.point_predictions) == 3

    with pytest.raises(ValueError, match="Need at least 2 observations"):
        survival.validation.jackknife_plus_survival([1.0], [1], [[0.0]], jackknife_config)
    with pytest.raises(ValueError, match="event length mismatch"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0], [1], [[0.0], [1.0]], jackknife_config
        )
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.validation.jackknife_plus_survival(
            [1.0, float("inf")], [1, 0], [[0.0], [1.0]], jackknife_config
        )
    with pytest.raises(ValueError, match="event.*0/1"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0], [1, 2], [[0.0], [1.0]], jackknife_config
        )
    with pytest.raises(ValueError, match="one row per observation"):
        survival.validation.jackknife_plus_survival([1.0, 2.0], [1, 0], [[0.0]], jackknife_config)
    with pytest.raises(ValueError, match="covariates must be rectangular"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0], [1, 0], [[0.0], [1.0, 2.0]], jackknife_config
        )
    with pytest.raises(ValueError, match="covariates contains non-finite"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0], [1, 0], [[0.0], [float("nan")]], jackknife_config
        )
    with pytest.raises(ValueError, match="alpha"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0],
            [1, 0],
            [[0.0], [1.0]],
            survival.validation.JackknifePlusConfig(1.0, True, 5),
        )
    with pytest.raises(ValueError, match="cv_folds must be positive"):
        survival.validation.jackknife_plus_survival(
            [1.0, 2.0],
            [1, 0],
            [[0.0], [1.0]],
            survival.validation.JackknifePlusConfig(0.1, True, 0),
        )


def test_fast_cox_numpy_uses_shifted_risk_scores_for_large_offsets():
    config = survival.regression.FastCoxConfig(
        0.0,
        1.0,
        1,
        1e-7,
        survival.regression.ScreeningRule("none"),
        None,
        10,
        False,
        True,
    )
    result = survival.regression.fast_cox_numpy(
        np.zeros((3, 1), dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.array([1, 0, 1], dtype=np.int32),
        config,
        offset=np.array([710.0, 709.0, 708.0], dtype=float),
    )
    expected = 2.0 * np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))

    assert np.isfinite(result.deviance)
    assert result.deviance == pytest.approx(expected)


def test_elastic_net_cox_uses_shifted_risk_scores_for_large_offsets():
    covariates = survival.core.CovariateMatrix([0.0, 0.0, 0.0], 3, 1)
    survival_data = survival.core.SurvivalData([1.0, 2.0, 3.0], [1, 0, 1])
    input_data = survival.core.CoxRegressionInput(
        covariates,
        survival_data,
        None,
        [710.0, 709.0, 708.0],
    )
    config = survival.regression.ElasticNetConfig(0.0, 0.0, 1, 1e-7, False, False)
    result = survival.regression.elastic_net_cox(input_data, config)
    expected = 2.0 * np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))

    assert np.isfinite(result.deviance)
    assert result.deviance == pytest.approx(expected)


def test_bootstrap_ci_helpers_smoke():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    status_i32 = [1, 1, 0, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    cox_covariates = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    survreg_covariates = [[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5], [1.0, 0.6]]

    cox = survival.validation.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=123,
    )
    assert len(cox.coefficients) == 1
    assert len(cox.std_errors) == 1
    assert len(cox.ci_lower) == 1
    assert len(cox.ci_upper) == 1
    assert len(cox.bootstrap_samples) > 0
    assert np.isfinite(cox.coefficients[0])

    survreg = survival.validation.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=123,
    )
    assert len(survreg.coefficients) == 3
    assert len(survreg.std_errors) == 3
    assert len(survreg.ci_lower) == 3
    assert len(survreg.ci_upper) == 3
    assert len(survreg.bootstrap_samples) > 0
    assert np.all(np.isfinite(survreg.coefficients))


def test_bootstrap_ci_helpers_are_deterministic_with_seed():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    status_i32 = [1, 1, 0, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    cox_covariates = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    survreg_covariates = [[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5], [1.0, 0.6]]

    first_cox = survival.validation.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    second_cox = survival.validation.bootstrap_cox_ci(
        time,
        status_i32,
        cox_covariates,
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    assert first_cox.coefficients == pytest.approx(second_cox.coefficients)
    assert first_cox.std_errors == pytest.approx(second_cox.std_errors)
    assert first_cox.ci_lower == pytest.approx(second_cox.ci_lower)
    assert first_cox.ci_upper == pytest.approx(second_cox.ci_upper)
    assert first_cox.bootstrap_samples == second_cox.bootstrap_samples

    first_survreg = survival.validation.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    second_survreg = survival.validation.bootstrap_survreg_ci(
        time,
        status_f64,
        survreg_covariates,
        distribution="weibull",
        n_bootstrap=8,
        confidence_level=0.9,
        seed=321,
    )
    assert first_survreg.coefficients == pytest.approx(second_survreg.coefficients)
    assert first_survreg.std_errors == pytest.approx(second_survreg.std_errors)
    assert first_survreg.ci_lower == pytest.approx(second_survreg.ci_lower)
    assert first_survreg.ci_upper == pytest.approx(second_survreg.ci_upper)
    assert first_survreg.bootstrap_samples == second_survreg.bootstrap_samples


def test_bootstrap_ci_helpers_validate_inputs():
    time = [1.0, 2.0, 3.0, 4.0]
    status_i32 = [1, 1, 0, 1]
    status_f64 = [float(value) for value in status_i32]
    covariates = [[0.1], [0.2], [0.3], [0.4]]

    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        survival.validation.bootstrap_cox_ci(time, status_i32, covariates, n_bootstrap=1)

    with pytest.raises(ValueError, match="status length mismatch"):
        survival.validation.bootstrap_cox_ci(time, status_i32[:-1], covariates, n_bootstrap=8)

    with pytest.raises(ValueError, match="weights length mismatch"):
        survival.validation.bootstrap_cox_ci(
            time,
            status_i32,
            covariates,
            weights=[1.0, 1.0],
            n_bootstrap=8,
        )

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        survival.validation.bootstrap_survreg_ci(
            time,
            status_f64,
            covariates,
            distribution="weibull",
            n_bootstrap=8,
            confidence_level=1.2,
        )

    with pytest.raises(ValueError, match="time\\[0\\] must be positive"):
        survival.validation.bootstrap_survreg_ci(
            [0.0, 2.0, 3.0, 4.0],
            status_f64,
            covariates,
            distribution="weibull",
            n_bootstrap=8,
        )

    with pytest.raises(ValueError, match="covariates row count mismatch"):
        survival.validation.bootstrap_survreg_ci(
            time,
            status_f64,
            covariates[:-1],
            distribution="weibull",
            n_bootstrap=8,
        )

    bad_covariates = [[0.1], [0.2], [float("inf")], [0.4]]
    with pytest.raises(ValueError, match="covariates contains non-finite"):
        survival.validation.bootstrap_survreg_ci(
            time,
            status_f64,
            bad_covariates,
            distribution="weibull",
            n_bootstrap=8,
        )

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.validation.bootstrap_survreg_ci(
            time,
            status_f64,
            covariates,
            distribution="mystery",
            n_bootstrap=8,
        )


def test_cox_callback_roundtrip():
    def callback(coef, *, which):
        assert isinstance(coef, np.ndarray)
        return {
            "coef": [value + which for value in coef],
            "first": [1.0, 2.0],
            "second": [3.0, 4.0],
            "penalty": 5.0,
            "flag": [True, False],
        }

    result = survival.pybridge.cox_callback(2, [1.0, 2.0], callback)

    assert isinstance(result, survival.pybridge.CoxPenaltyTerms)
    assert result.coef == pytest.approx([3.0, 4.0])
    assert result.first == pytest.approx([1.0, 2.0])
    assert result.second == pytest.approx([3.0, 4.0])
    assert result.penalty == pytest.approx(5.0)
    assert result.flag == [True, False]

    def numpy_callback(coef, *, which):
        return {
            "coef": coef * which,
            "first": np.zeros(2),
            "second": np.ones(2),
            "penalty": np.array([1.5]),
            "flag": np.array([False, True]),
        }

    arrays = survival.pybridge.cox_callback(3, np.array([1.0, 2.0]), numpy_callback)
    assert arrays.coef == pytest.approx([3.0, 6.0])
    assert arrays.penalty == pytest.approx(1.5)
    assert arrays.flag == [False, True]

    with pytest.raises(KeyError, match="no 'penalty' entry"):
        survival.pybridge.cox_callback(1, [1.0], lambda coef, which: {"coef": coef})
