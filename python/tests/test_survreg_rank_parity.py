"""Static R references for singular AFT fitting and its public reporting."""

import json
import math
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "survreg_rank_r_reference.json").read_text()
)


def _vector(value):
    return value if isinstance(value, list) else [value]


def _assert_vector(actual, expected, *, rel=3e-7, atol=3e-8):
    expected = _vector(expected)
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert math.isnan(value)
        elif reference in ("Inf", "-Inf"):
            assert value == (math.inf if reference == "Inf" else -math.inf)
        else:
            assert value == pytest.approx(reference, rel=rel, abs=atol)


def _assert_matrix(actual, expected, **kwargs):
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected, strict=True):
        _assert_vector(row, reference, **kwargs)


@pytest.fixture(params=REFERENCE["cases"], ids=lambda case: case["id"])
def rank_fit(request):
    case = request.param
    fit = survival.survreg(
        case["formula"],
        data=case["data"],
        distribution=case["distribution"],
        scale=case["fixed_scale"],
        init=None if case["initial"] is None else _vector(case["initial"]),
        weights=case["weights"],
        score=True,
        max_iter=case["maxiter"],
        eps=case["eps"],
        tol_chol=case["tolerance"],
    )
    return case, fit


def test_aft_rank_covariance_and_score_match_r(rank_fit):
    case, fit = rank_fit
    covariance_tolerance = 5e-8 if "near_dependent_retained" in case["id"] else 3e-7
    _assert_matrix(fit.variance_matrix, case["variance"], rel=covariance_tolerance)
    _assert_vector(fit.score_vector, case["score"], atol=2e-6)
    _assert_vector(fit.linear_predictors, case["linear_predictors"])
    _assert_vector(fit.scales, case["scales"])
    assert survival.loglik(fit) == pytest.approx(case["loglik"], rel=3e-10, abs=3e-10)
    assert survival.degrees_freedom(fit) == case["df"]
    if case["id"] == "weibull_singular_interaction_convergence":
        assert fit.convergence_flag == 0
    if case["maxiter"] <= 1:
        assert fit.iterations == case["iterations"]
    for index, coefficient in enumerate(case["coefficients"]):
        if coefficient is None:
            # R's generalized covariance zeros each redundant direction;
            # information-matrix or regularized-inverse fallbacks fail here.
            assert fit.variance_matrix[index] == [0.0] * len(fit.variance_matrix)
            assert all(row[index] == 0.0 for row in fit.variance_matrix)


def test_aft_rank_reporting_and_confidence_match_r(rank_fit):
    case, fit = rank_fit
    _assert_vector(survival.coef(fit), case["coefficients"])
    _assert_vector(fit.coefficients[: fit.n_covariates], case["coefficients"])
    _assert_matrix(survival.vcov(fit), case["variance"])
    _assert_matrix(survival.vcov(fit, complete=False), case["reduced_variance"])
    intervals = survival.confint(fit)
    _assert_matrix([[row["lower"], row["upper"]] for row in intervals], case["confidence"])
    # R's zero-rank summary returns the original fit without a table;
    # coefficient/covariance behavior is still checked above.
    if case["summary"] is not None:
        summary = survival.model_summary(fit)
        actual = [[row["coef"], row["se"], row["z"], row["p"]] for row in summary["coefficients"]]
        _assert_matrix(actual, case["summary"], rel=2e-6)


def test_aft_alias_predictions_preserve_stored_predictors_and_r_missing_values(rank_fit):
    case, fit = rank_fit
    _assert_vector(survival.predict(fit, type="lp"), case["training_prediction"])
    _assert_vector(survival.predict(fit, case["data"], type="lp"), case["newdata_prediction"])
    _assert_vector(survival.r_api.residuals(fit, type="response"), case["response_residuals"])
    if case["terms"] is not None:
        terms = survival.predict(fit, type="terms", se_fit=True)
        _assert_matrix(terms.fit, case["terms"])
        _assert_matrix(terms.se_fit, case["terms_se"])
    aliases = [
        index for index, coefficient in enumerate(case["coefficients"]) if coefficient is None
    ]
    if aliases:
        dfbeta = survival.r_api.residuals(fit, type="dfbeta")
        dfbetas = survival.r_api.residuals(fit, type="dfbetas")
        for index in aliases:
            _assert_vector([row[index] for row in dfbeta], [row[index] for row in case["dfbeta"]])
            _assert_vector([row[index] for row in dfbetas], [row[index] for row in case["dfbetas"]])
