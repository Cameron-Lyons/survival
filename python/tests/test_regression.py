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


def test_coxmart():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    result = survival.coxmart(
        time=time,
        status=status,
        score=score,
        weights=None,
        strata=None,
        method=0,
    )
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


def test_aareg_rejects_invalid_formula():
    options = survival.AaregOptions(
        formula="time",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
        max_iter=5,
    )

    with pytest.raises(RuntimeError, match="Formula Error"):
        survival.aareg(options)


def test_coxph_detail_public_api():
    detail = survival.coxph_detail(
        time=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        covariates=[[1.0], [2.0], [3.0]],
        coefficients=[0.5],
    )

    assert detail.n_events == 2
    assert detail.n_observations == 3
    assert detail.n_covariates == 1
    assert detail.times() == [1.0, 3.0]
    assert len(detail.hazards()) == 2
    assert detail.cumulative_hazards()[0] <= detail.cumulative_hazards()[1]
    assert detail.n_risk_at_times() == [3, 1]
    assert len(detail.schoenfeld_residuals()) == 2


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
