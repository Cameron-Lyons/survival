import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_survreg():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    covariates = [
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5],
    ]

    result = survival.survreg(
        time=time,
        status=status,
        covariates=covariates,
        weights=None,
        offsets=None,
        initial_beta=None,
        strata=None,
        distribution="extreme_value",
        max_iter=20,
        eps=1e-5,
        tol_chol=1e-9,
    )
    assert hasattr(result, "coefficients")
    assert hasattr(result, "log_likelihood")
    assert hasattr(result, "iterations")
    assert isinstance(result.coefficients, list)


def test_survival_fit_predict_quantile_rejects_nonfinite_quantiles():
    fit = survival.survreg(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1.0, 1.0, 0.0, 1.0],
        covariates=[[1.0], [2.0], [3.0], [4.0]],
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
        tol_chol=1e-9,
    )

    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        fit.predict_quantile(quantiles=[float("nan")])


def test_spline_config_validates_public_inputs():
    config = survival.SplineConfig(3, 3, " Uniform ", (1.0, 10.0))
    assert config.knot_placement == "equal"

    with pytest.raises(ValueError, match="n_knots"):
        survival.SplineConfig(0, 3, "quantile", None)
    with pytest.raises(ValueError, match="degree"):
        survival.SplineConfig(3, 0, "quantile", None)
    with pytest.raises(ValueError, match="knot_placement"):
        survival.SplineConfig(3, 3, "unknown", None)
    with pytest.raises(ValueError, match="boundary_knots"):
        survival.SplineConfig(3, 3, "quantile", (0.0, 10.0))
    with pytest.raises(ValueError, match="boundary_knots"):
        survival.SplineConfig(3, 3, "quantile", (10.0, 10.0))


def test_flexible_parametric_model_revalidates_mutated_spline_config():
    config = survival.SplineConfig(3, 3, "quantile", None)
    config.knot_placement = "unknown"
    time = [float(value) for value in range(1, 21)]
    event = [1 if idx % 3 == 0 else 0 for idx in range(20)]
    covariates = [[idx * 0.1] for idx in range(20)]

    with pytest.raises(ValueError, match="knot_placement"):
        survival.flexible_parametric_model(time, event, covariates, config)


def test_restricted_cubic_spline_validates_public_inputs():
    x = [float(value) for value in range(1, 6)]

    with pytest.raises(ValueError, match="x must contain only finite"):
        survival.restricted_cubic_spline([1.0, 2.0, float("nan"), 4.0, 5.0], 4, None)
    with pytest.raises(ValueError, match="n_knots"):
        survival.restricted_cubic_spline(x, 2, None)
    with pytest.raises(ValueError, match="knots must contain only finite"):
        survival.restricted_cubic_spline(x, None, [1.0, 2.0, float("inf")])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.restricted_cubic_spline(x, None, [1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.restricted_cubic_spline([1.0] * 5, 4, None)


def test_predict_hazard_spline_validates_public_inputs():
    time = [float(value) for value in range(1, 21)]
    event = [1 if idx % 3 == 0 else 0 for idx in range(20)]
    covariates = [[idx * 0.1] for idx in range(20)]
    config = survival.SplineConfig(3, 3, "quantile", None)
    model = survival.flexible_parametric_model(time, event, covariates, config)

    with pytest.raises(ValueError, match="eval_times"):
        survival.predict_hazard_spline(model, [], [0.5])
    with pytest.raises(ValueError, match="eval_times must contain only finite"):
        survival.predict_hazard_spline(model, [1.0, float("nan")], [0.5])
    with pytest.raises(ValueError, match="non-negative"):
        survival.predict_hazard_spline(model, [-1.0, 2.0], [0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.predict_hazard_spline(model, [1.0, 1.0], [0.5])
    with pytest.raises(ValueError, match="covariate_values length"):
        survival.predict_hazard_spline(model, [1.0, 2.0], [0.5, 1.0])
    with pytest.raises(ValueError, match="covariate_values must contain only finite"):
        survival.predict_hazard_spline(model, [1.0, 2.0], [float("nan")])

    bad_coefficients = survival.FlexibleParametricResult(
        [float("nan")],
        model.spline_coefficients,
        model.std_errors,
        model.knots,
        model.log_likelihood,
        model.aic,
        model.bic,
        model.n_iterations,
        model.converged,
    )
    with pytest.raises(ValueError, match="coefficients must contain only finite"):
        survival.predict_hazard_spline(bad_coefficients, [1.0, 2.0], [0.5])

    bad_spline_coefficients = survival.FlexibleParametricResult(
        model.coefficients,
        model.spline_coefficients[:-1],
        model.std_errors,
        model.knots,
        model.log_likelihood,
        model.aic,
        model.bic,
        model.n_iterations,
        model.converged,
    )
    with pytest.raises(ValueError, match="spline_coefficients length"):
        survival.predict_hazard_spline(bad_spline_coefficients, [1.0, 2.0], [0.5])


def test_coxmart():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    survival_data = survival.SurvivalData(time, status)
    input_data = survival.CoxMartInput(survival_data, score)

    result = survival.coxmart(input_data, method=0)
    assert isinstance(result, list)
    assert len(result) == len(time)


def test_coxph_model():
    covariates = [
        [0.5, 1.2],
        [1.8, 0.3],
        [0.2, 2.1],
        [2.5, 0.8],
        [0.8, 1.5],
        [1.5, 0.5],
        [0.3, 1.8],
        [2.2, 1.1],
        [1.0, 0.9],
        [0.7, 1.7],
        [2.0, 0.4],
        [1.2, 1.3],
        [0.9, 2.0],
        [1.6, 0.7],
        [0.4, 1.4],
        [2.1, 1.0],
    ]
    event_times = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    censoring = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]

    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)
    model.fit(n_iters=20)

    assert hasattr(model, "baseline_hazard")
    assert hasattr(model, "risk_scores")

    coefficients = model.coefficients
    assert coefficients is not None

    new_covariates = [[1.0, 2.0], [2.0, 3.0]]
    predictions = model.predict(new_covariates)
    assert isinstance(predictions, list)

    brier = model.brier_score()
    assert isinstance(brier, float)


def _survival_quantile_cox_model(censoring=None):
    event_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    if censoring is None:
        censoring = [1, 0, 1, 1, 0, 1, 0, 1]
    covariates = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)
    model.fit(n_iters=50)
    return model


def test_coxph_model_survival_time_quantiles_match_step_reference():
    model = _survival_quantile_cox_model()
    rows = [[0.0], [1.0]]

    assert model.predicted_survival_time(rows, 0.0) == [0.0, 0.0]
    assert model.predicted_survival_time(rows, 0.25) == [3.0, 4.0]
    assert model.predicted_survival_time(rows) == [6.0, 6.0]
    assert model.predicted_survival_time(rows, 0.75) == [8.0, 8.0]
    assert model.predicted_survival_time(rows, 1.0) == [None, None]

    _, curves = model.survival_curve([[0.0]], None)
    plateau_probability = 1.0 - curves[0][3]
    tolerance = math.sqrt(math.ulp(1.0))
    assert model.predicted_survival_time([[0.0]], plateau_probability) == [5.0]
    assert model.predicted_survival_time([[0.0]], plateau_probability + tolerance / 2.0) == [5.0]
    assert model.predicted_survival_time([[0.0]], plateau_probability + 2.0 * tolerance) == [6.0]

    terminal_model = _survival_quantile_cox_model([1, 0, 1, 1, 0, 1, 0, 0])
    _, terminal_curves = terminal_model.survival_curve([[0.0]], None)
    terminal_probability = 1.0 - terminal_curves[0][5]
    assert terminal_model.predicted_survival_time([[0.0]], terminal_probability) == [7.0]


def test_coxph_model_survival_time_quantiles_validate_inputs():
    model = _survival_quantile_cox_model()

    for percentile in [-0.01, 1.01, math.nan, -math.inf, math.inf]:
        with pytest.raises(ValueError, match="percentile must be a finite value"):
            model.predicted_survival_time([[0.0]], percentile)

    with pytest.raises(ValueError, match="has 2 columns but expected 1"):
        model.predicted_survival_time([[0.0, 1.0]])
    with pytest.raises(ValueError, match="contains NaN"):
        model.predicted_survival_time([[math.nan]])
    with pytest.raises(ValueError, match="contains non-finite"):
        model.predicted_survival_time([[math.inf]])

    unfitted = survival.CoxPHModel.new_with_data([[0.0]], [1.0], [1])
    with pytest.raises(ValueError, match="model must be fit before prediction"):
        unfitted.predicted_survival_time([[0.0]])

    assert model.predicted_survival_time([]) == []


def test_subject():
    subject = survival.Subject(
        id=1,
        covariates=[1.0, 2.0],
        is_case=True,
        is_subcohort=True,
        stratum=0,
    )
    assert subject.id == 1
    assert subject.is_case is True


def test_aareg_public_api():
    options = survival.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        variable_names=["time", "x1"],
        max_iter=20,
    )

    result = survival.aareg(options)

    assert len(result.coefficients) == 2
    assert len(result.standard_errors) == 2
    assert len(result.confidence_intervals) == 2
    assert len(result.p_values) == 2
    assert result.fit_details is not None
    assert result.fit_details.iterations <= 20
    assert result.fit_details.converged is True
    assert len(result.residuals) == 4
    assert math.isfinite(result.goodness_of_fit)

    weighted_subset = survival.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        variable_names=["time", "x1"],
        max_iter=20,
    )
    weighted_subset.subset = [0, 1, 2]
    weighted_subset.weights = [1.0, 2.0, 1.0, 99.0]
    weighted_result = survival.aareg(weighted_subset)

    assert len(weighted_result.residuals) == 3


def test_aareg_rejects_invalid_formula():
    options = survival.AaregOptions(
        formula="time",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
        max_iter=5,
    )

    with pytest.raises(ValueError, match="Formula Error"):
        survival.aareg(options)


def test_aareg_validates_public_inputs():
    with pytest.raises(ValueError, match="data cannot be empty"):
        survival.aareg(
            survival.AaregOptions(
                formula="time ~ x1",
                data=[],
                variable_names=["time", "x1"],
            )
        )

    with pytest.raises(ValueError, match="data row 1 has 1 columns"):
        survival.aareg(
            survival.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, 2.0], [2.0]],
                variable_names=["time", "x1"],
            )
        )

    with pytest.raises(ValueError, match="variable_names length"):
        survival.aareg(
            survival.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, 2.0], [2.0, 3.0]],
                variable_names=["time"],
            )
        )

    with pytest.raises(ValueError, match="data contains non-finite"):
        survival.aareg(
            survival.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, float("inf")], [2.0, 3.0]],
                variable_names=["time", "x1"],
            )
        )

    missing = survival.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, float("nan")], [2.0, 3.0], [3.0, 4.0]],
        variable_names=["time", "x1"],
    )
    with pytest.raises(ValueError, match="missing values in data"):
        survival.aareg(missing)

    missing.na_action = "Exclude"
    excluded = survival.aareg(missing)
    assert len(excluded.residuals) == 2

    bad_weights = survival.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
    )
    bad_weights.weights = [1.0, float("inf")]
    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.aareg(bad_weights)

    bad_iter = survival.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
        max_iter=0,
    )
    with pytest.raises(ValueError, match="max_iter must be positive"):
        survival.aareg(bad_iter)


def test_coxph_detail_public_api():
    detail = survival.coxph_detail(
        time=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        covariates=[[1.0], [2.0], [3.0]],
        coefficients=[0.5],
    )
    expected_risk = math.exp(0.5) + math.exp(1.0) + math.exp(1.5)
    expected_mean = (math.exp(0.5) + 2.0 * math.exp(1.0) + 3.0 * math.exp(1.5)) / expected_risk

    assert detail.n_events == 2
    assert detail.n_observations == 3
    assert detail.n_covariates == 1
    assert detail.times() == [1.0, 3.0]
    assert len(detail.hazards()) == 2
    assert detail.cumulative_hazards()[0] <= detail.cumulative_hazards()[1]
    assert detail.n_risk_at_times() == [3, 1]
    assert len(detail.schoenfeld_residuals()) == 2
    assert detail.rows[0].wtrisk == pytest.approx(expected_risk)
    assert detail.rows[0].means == pytest.approx([expected_mean])
    assert detail.rows[0].score == pytest.approx([1.0 - expected_mean])
    assert detail.rows[0].imat[0][0] > 0.0
    assert detail.rows[0].varhaz > 0.0
    assert detail.scores()[0] == pytest.approx(detail.rows[0].score)
    assert detail.information_matrices()[0][0][0] == pytest.approx(detail.rows[0].imat[0][0])


def test_coxph_detail_uses_shifted_risk_scores_for_large_linear_predictors():
    detail = survival.coxph_detail(
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        covariates=[[1.0], [709.0 / 710.0], [708.0 / 710.0]],
        coefficients=[710.0],
    )
    expected_first = math.exp(-710.0) / (1.0 + math.exp(-1.0) + math.exp(-2.0))

    assert detail.times() == pytest.approx([1.0, 2.0, 3.0])
    assert detail.hazards()[0] == pytest.approx(expected_first, rel=1e-12, abs=0.0)
    assert (
        0.0 < detail.hazards()[0] < detail.cumulative_hazards()[1] < detail.cumulative_hazards()[2]
    )


def test_coxph_detail_low_level_supports_entry_strata_and_efron():
    breslow = survival.regression.coxph_detail(
        time=[2.0, 2.0, 4.0, 5.0, 5.0, 6.0],
        status=[1, 1, 1, 0, 1, 0],
        covariates=[[0.2], [0.8], [0.4], [1.1], [0.7], [0.3]],
        coefficients=[0.0],
        entry_times=[0.0, 0.0, 1.5, 2.5, 0.0, 3.0],
        strata=[0, 0, 0, 0, 1, 1],
    )
    efron = survival.regression.coxph_detail(
        time=[2.0, 2.0, 4.0, 5.0, 5.0, 6.0],
        status=[1, 1, 1, 0, 1, 0],
        covariates=[[0.2], [0.8], [0.4], [1.1], [0.7], [0.3]],
        coefficients=[0.0],
        entry_times=[0.0, 0.0, 1.5, 2.5, 0.0, 3.0],
        strata=[0, 0, 0, 0, 1, 1],
        method="efron",
    )

    assert [row.stratum for row in breslow.rows] == [0, 0, 1]
    assert breslow.times() == pytest.approx([2.0, 4.0, 5.0])
    assert breslow.n_risk_at_times() == [3, 2, 2]
    assert breslow.hazards()[0] == pytest.approx(2.0 / 3.0)
    assert efron.hazards()[0] == pytest.approx((1.0 / 3.0) + (1.0 / 2.0))
    assert efron.hazards()[0] > breslow.hazards()[0]


def test_coxph_detail_validates_input_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        survival.coxph_detail(
            time=[1.0, 2.0],
            status=[1],
            covariates=[[1.0], [2.0]],
            coefficients=[0.5],
        )


def test_predict_survreg_linear_predictor_and_standard_errors():
    prediction = survival.predict_survreg(
        covariates=[[1.0, 2.0], [2.0, 3.0]],
        coefficients=[0.1, 0.2],
        scale=1.0,
        distribution="weibull",
        predict_type="lp",
        var_matrix=[[1.0, 0.0], [0.0, 1.0]],
        se_fit=True,
    )

    assert prediction.predictions == pytest.approx([0.5, 0.8])
    assert prediction.se == pytest.approx([5**0.5, 13**0.5])
    assert prediction.prediction_type == "lp"
    assert prediction.n == 2


def test_predict_survreg_quantiles_and_validation():
    quantiles = survival.predict_survreg_quantile(
        covariates=[[1.0, 2.0], [2.0, 3.0]],
        coefficients=[0.1, 0.2],
        scale=1.0,
        distribution="weibull",
        quantiles=[0.25, 0.5],
    )

    assert quantiles.quantiles == [0.25, 0.5]
    assert len(quantiles.predictions) == 2
    assert all(len(row) == 2 for row in quantiles.predictions)

    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        survival.predict_survreg_quantile(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            quantiles=[0.0],
        )


def test_predict_survreg_rejects_invalid_scale():
    with pytest.raises(ValueError, match="scale must be a finite positive value"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=0.0,
            distribution="weibull",
        )


def test_predict_survreg_validates_numeric_inputs():
    with pytest.raises(ValueError, match="coefficients contains non-finite"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, float("nan")],
            scale=1.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match=r"covariates\[0\]\[1\] contains non-finite"):
        survival.predict_survreg(
            covariates=[[1.0, float("inf")]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="covariates row 0 has 1 columns"):
        survival.predict_survreg(
            covariates=[[1.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="offset has 1 values"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0], [2.0, 3.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            offset=[0.0],
        )

    with pytest.raises(ValueError, match="offset contains non-finite"):
        survival.predict_survreg_quantile(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            quantiles=[0.5],
            offset=[float("nan")],
        )

    with pytest.raises(ValueError, match="var_matrix must have at least 2 rows"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            var_matrix=[[1.0, 0.0]],
            se_fit=True,
        )

    with pytest.raises(ValueError, match=r"var_matrix\[1\]\[1\] contains non-finite"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            var_matrix=[[1.0, 0.0], [0.0, float("nan")]],
            se_fit=True,
        )

    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        survival.predict_survreg_quantile(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="weibull",
            quantiles=[float("nan")],
        )

    with pytest.raises(ValueError, match="scale must be a finite positive value"):
        survival.predict_survreg_quantile(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=float("inf"),
            distribution="weibull",
            quantiles=[0.5],
        )

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.predict_survreg(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="mystery",
        )

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.predict_survreg_quantile(
            covariates=[[1.0, 2.0]],
            coefficients=[0.1, 0.2],
            scale=1.0,
            distribution="mystery",
            quantiles=[0.5],
        )


def test_survfit_and_survreg_residual_public_apis():
    survfit_residuals = survival.residuals_survfit(
        time=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        surv_time=[1.0, 2.0, 3.0],
        surv=[0.9, 0.8, 0.7],
        residual_type="deviance",
    )
    survreg_residuals = survival.residuals_survreg(
        time=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        linear_pred=[0.0, 0.5, 1.0],
        scale=1.0,
        distribution="weibull",
        residual_type="working",
    )

    assert len(survfit_residuals.residuals) == 3
    assert survfit_residuals.time == [1.0, 2.0, 3.0]
    assert survfit_residuals.residual_type == "deviance"
    assert len(survreg_residuals.residuals) == 3
    assert survreg_residuals.residual_type == "working"
    assert survreg_residuals.n == 3


def test_survfit_residuals_use_last_duplicate_survival_time():
    result = survival.residuals_survfit(
        time=[2.0],
        status=[1],
        surv_time=[1.0, 2.0, 2.0, 3.0],
        surv=[0.9, 0.8, 0.7, 0.6],
    )

    assert result.residuals == pytest.approx([1.0 + math.log(0.7)])


def test_survreg_residual_matrix_public_api_returns_derivative_columns():
    matrix = survival.survreg_residual_matrix(
        time=[1.5],
        status=[1],
        linear_pred=[1.0],
        scale=1.0,
        distribution="gaussian",
    )
    z = 0.5
    expected_loglik = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)

    assert len(matrix) == 1
    assert matrix[0] == pytest.approx(
        [
            expected_loglik,
            z,
            -1.0,
            z * z - 1.0,
            -2.0 * z * z,
            -2.0 * z,
        ],
        abs=1e-5,
    )


def test_survreg_influence_residual_public_api_matches_quadratic_forms():
    derivative_matrix = [[0.0, 2.0, 3.0, 5.0, 7.0, 11.0]]
    covariates = [[1.0, 4.0]]
    scales = [1.5]
    strata = [0]
    var_matrix = [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]

    assert survival.survreg_influence_residuals(
        derivative_matrix,
        covariates,
        scales,
        strata,
        var_matrix,
        "ldcase",
        True,
    ) == pytest.approx([238.2])
    assert survival.survreg_influence_residuals(
        derivative_matrix,
        covariates,
        scales,
        strata,
        var_matrix,
        "ldresp",
        True,
    ) == pytest.approx([1709.1])
    assert survival.survreg_influence_residuals(
        derivative_matrix,
        covariates,
        scales,
        strata,
        var_matrix,
        "ldshape",
        True,
    ) == pytest.approx([4452.4])


def test_survreg_dfbeta_residual_public_api_matches_score_times_variance():
    derivative_matrix = [[0.0, 2.0, 3.0, 5.0, 7.0, 11.0]]
    covariates = [[1.0, 4.0]]
    scales = [1.5]
    strata = [0]
    var_matrix = [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]

    dfbeta = survival.survreg_dfbeta_residuals(
        derivative_matrix,
        covariates,
        scales,
        strata,
        var_matrix,
        True,
        False,
    )
    dfbetas = survival.survreg_dfbeta_residuals(
        derivative_matrix,
        covariates,
        scales,
        strata,
        var_matrix,
        True,
        True,
    )

    assert len(dfbeta) == 1
    assert len(dfbetas) == 1
    assert dfbeta[0] == pytest.approx([3.8, 17.7, 17.8])
    assert dfbetas[0] == pytest.approx([3.8, 17.7 / math.sqrt(2.0), 17.8 / math.sqrt(3.0)])


def test_residual_apis_validate_type_and_lengths():
    with pytest.raises(ValueError, match="Unknown residual type"):
        survival.residuals_survfit(
            time=[1.0],
            status=[1],
            surv_time=[1.0],
            surv=[0.9],
            residual_type="unknown",
        )

    with pytest.raises(ValueError, match="All inputs must have the same length"):
        survival.dfbeta_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            covariates=[[1.0]],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            var_matrix=[[1.0]],
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="matrix residuals are matrix-valued"):
        survival.residuals_survreg(
            time=[1.0],
            status=[1],
            linear_pred=[0.0],
            scale=1.0,
            distribution="weibull",
            residual_type="matrix",
        )


def test_survfit_residual_api_validates_numeric_inputs():
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.residuals_survfit(
            time=[1.0],
            status=[2],
            surv_time=[1.0],
            surv=[0.9],
        )

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.residuals_survfit(
            time=[math.inf],
            status=[1],
            surv_time=[1.0],
            surv=[0.9],
        )

    with pytest.raises(ValueError, match="surv_time contains non-finite"):
        survival.residuals_survfit(
            time=[1.0],
            status=[1],
            surv_time=[math.nan],
            surv=[0.9],
        )

    with pytest.raises(ValueError, match="probabilities between 0 and 1"):
        survival.residuals_survfit(
            time=[1.0],
            status=[1],
            surv_time=[1.0],
            surv=[1.2],
        )

    with pytest.raises(ValueError, match="surv_time must be sorted"):
        survival.residuals_survfit(
            time=[1.0],
            status=[1],
            surv_time=[2.0, 1.0],
            surv=[0.9, 0.8],
        )


def test_survreg_residual_apis_validate_numeric_inputs():
    with pytest.raises(ValueError, match="status must contain only 0/1/2/3"):
        survival.residuals_survreg(
            time=[1.0],
            status=[4],
            linear_pred=[0.0],
            scale=1.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="linear_pred contains non-finite"):
        survival.residuals_survreg(
            time=[1.0],
            status=[1],
            linear_pred=[float("inf")],
            scale=1.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="scale must be a finite positive value"):
        survival.residuals_survreg(
            time=[1.0],
            status=[1],
            linear_pred=[0.0],
            scale=0.0,
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.residuals_survreg(
            time=[1.0],
            status=[1],
            linear_pred=[0.0],
            scale=1.0,
            distribution="mystery",
        )

    with pytest.raises(ValueError, match="use dfbeta_survreg"):
        survival.residuals_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            distribution="weibull",
            residual_type="dfbeta",
        )

    with pytest.raises(ValueError, match="use dfbeta_survreg"):
        survival.residuals_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            distribution="weibull",
            residual_type="dfbetas",
        )

    with pytest.raises(ValueError, match="greater than time"):
        survival.residuals_survreg(
            time=[2.0],
            status=[3],
            linear_pred=[0.0],
            scale=1.0,
            distribution="weibull",
            residual_type="ldcase",
            time2=[1.5],
        )

    with pytest.raises(ValueError, match="covariates row 1"):
        survival.dfbeta_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            covariates=[[1.0, 0.5], [1.0]],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            var_matrix=[[1.0, 0.0], [0.0, 1.0]],
            distribution="weibull",
        )

    with pytest.raises(ValueError, match=r"var_matrix\[1\]\[1\] contains non-finite"):
        survival.dfbeta_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            covariates=[[1.0, 0.5], [1.0, 0.2]],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            var_matrix=[[1.0, 0.0], [0.0, float("nan")]],
            distribution="weibull",
        )

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.dfbeta_survreg(
            time=[1.0, 2.0],
            status=[1, 0],
            covariates=[[1.0, 0.5], [1.0, 0.2]],
            linear_pred=[0.0, 0.5],
            scale=1.0,
            var_matrix=[[1.0, 0.0], [0.0, 1.0]],
            distribution="mystery",
        )


def test_recurrent_event_regression_validates_public_inputs():
    with pytest.raises(ValueError, match="x length"):
        survival.gap_time_model([0, 1], [0.0, 0.0], [1.0, 1.0], [1, 0], [0.5], 2, 1, 10, 1e-6)
    with pytest.raises(ValueError, match="stop_time"):
        survival.gap_time_model([0], [1.0], [1.0], [1], [0.5], 1, 1, 10, 1e-6)
    with pytest.raises(ValueError, match="event_status"):
        survival.pwp_gap_time([0], [1.0], [2], [0.5], 1, 1, False)
    with pytest.raises(ValueError, match="max_iter"):
        survival.gap_time_model([0], [0.0], [1.0], [1], [0.5], 1, 1, 0, 1e-6)

    gap = survival.gap_time_model(
        [10, 10, 42],
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 0, 1],
        [1.0, 0.5, 0.0],
        3,
        1,
        20,
        1e-4,
    )
    assert gap.n_subjects == 2

    method = survival.MarginalMethod("andersen_gill")
    with pytest.raises(ValueError, match="x length"):
        survival.marginal_recurrent_model(
            [0, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            2,
            1,
            method,
            10,
            1e-6,
        )
    with pytest.raises(ValueError, match="event_status"):
        survival.wei_lin_weissfeld([0], [1.0], [2], [0.5], 1, 1)

    marginal = survival.marginal_recurrent_model(
        [10, 10, 42],
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 0, 1],
        [1.0, 0.5, 0.0],
        3,
        1,
        method,
        20,
        1e-4,
    )
    assert marginal.n_subjects == 2

    with pytest.raises(ValueError, match="covariates length"):
        survival.anderson_gill_model(
            [1, 2],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            10,
            1e-6,
        )

    pwp_config = survival.PWPConfig(survival.PWPTimescale("gap"), 10, 1e-6, True, True)
    with pytest.raises(ValueError, match="stop must be greater than start"):
        survival.pwp_model([1], [0.0], [0.0], [1], [1], [], pwp_config)
    with pytest.raises(ValueError, match="event_number"):
        survival.pwp_model([1], [0.0], [1.0], [1], [0], [], pwp_config)

    wlw_config = survival.WLWConfig(10, 1e-6, True, False)
    with pytest.raises(ValueError, match="event must contain only 0/1"):
        survival.wlw_model([1], [1.0], [2], [1], [], wlw_config)

    bad_wlw_config = survival.WLWConfig(0, 1e-6, True, False)
    with pytest.raises(ValueError, match="max_iter"):
        survival.wlw_model([1], [1.0], [1], [1], [], bad_wlw_config)

    nb_config = survival.NegativeBinomialFrailtyConfig(10, 1e-6, 10)
    with pytest.raises(ValueError, match="same length"):
        survival.negative_binomial_frailty([1, 2], [1.0, 1.0], [1, 0], [], [0.0], nb_config)
    with pytest.raises(ValueError, match="event counts"):
        survival.negative_binomial_frailty([1], [1.0], [-1], [], None, nb_config)

    bad_nb_config = survival.NegativeBinomialFrailtyConfig(10, 1e-6, 0)
    with pytest.raises(ValueError, match="em_max_iter"):
        survival.negative_binomial_frailty([1], [1.0], [1], [], None, bad_nb_config)

    with pytest.raises(ValueError, match="x length"):
        survival.joint_frailty_model(
            [0, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            2,
            1,
            [1.0, 1.0],
            [1, 0],
            [0.5, 0.2],
            2,
            1,
        )
    with pytest.raises(ValueError, match="subject_id values"):
        survival.joint_frailty_model(
            [2],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0, 1.0],
            [1, 0],
            [0.5, 0.2],
            2,
            1,
        )
    with pytest.raises(ValueError, match="term_status"):
        survival.joint_frailty_model(
            [0],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0],
            [2],
            [0.5],
            1,
            1,
        )
    with pytest.raises(ValueError, match="max_iter"):
        survival.joint_frailty_model(
            [0],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0],
            [1],
            [0.5],
            1,
            1,
            survival.FrailtyDistribution("gamma"),
            0,
            1e-4,
        )
