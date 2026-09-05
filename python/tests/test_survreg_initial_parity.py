"""R default AFT starts, final fits, and explicit-start controls."""

import json
import math
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "survreg_initial_r_reference.json").read_text()
)


def vector(value):
    return value if isinstance(value, list) else [value]


def fit_case(case):
    data = REFERENCE["datasets"][case["dataset"]]
    arguments = {
        "data": data,
        "dist": case["distribution"],
        "weights": data["weight"],
        "max_iter": case["maxiter"],
        "eps": case["eps"],
        "tol_chol": case["tolerance"],
        "score": True,
    }
    if case["distribution"] not in ("exponential", "rayleigh"):
        arguments["scale"] = case["scale"]
    if case["initial"] is not None:
        arguments["init"] = case["initial"]
    if case["distribution"] == "t":
        arguments["parms"] = [5.0]
    return survival.survreg(case["formula"], **arguments)


@pytest.fixture(params=REFERENCE["cases"], ids=lambda case: case["id"])
def reference_fit(request):
    case = request.param
    return case, fit_case(case)


def gaussian_censoring(case):
    return case["distribution"] in ("gaussian", "lognormal") and any(
        value != 1 for value in REFERENCE["datasets"][case["dataset"]]["status"]
    )


def test_default_initialization_and_converged_parameters_match_r(reference_fit):
    case, fit = reference_fit
    # Existing Gaussian CDF approximations affect mixed-censoring references.
    # At explicit R parameter values, measured LL/score errors are at most
    # 4.3e-6/8.3e-6 on these cases; all-event Gaussian identities remain tight.
    absolute_tolerance = 4e-6 if gaussian_censoring(case) else 5e-8
    relative_tolerance = 3e-6 if gaussian_censoring(case) else 3e-7
    expected = vector(case["coefficients"])
    for actual, reference in zip(survival.coef(fit), expected, strict=True):
        if reference is None:
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(
                reference, rel=relative_tolerance, abs=absolute_tolerance
            )
    expected = [0.0 if value is None else value for value in expected]
    if case["scale"] == 0:
        expected = expected + [math.log(value) for value in vector(case["scales"])]
    assert fit.fit.coefficients == pytest.approx(
        expected, rel=relative_tolerance, abs=absolute_tolerance
    )
    assert fit.scales == pytest.approx(
        vector(case["scales"]), rel=relative_tolerance, abs=absolute_tolerance
    )
    assert fit.linear_predictors == pytest.approx(
        case["linear_predictors"], rel=relative_tolerance, abs=absolute_tolerance
    )
    if case["maxiter"] == 0:
        assert fit.iterations == 0
    else:
        assert fit.convergence_flag == 0
    if case["initial"] is not None:
        assert fit.fit.coefficients == case["initial"]
    else:
        for index, coefficient in enumerate(vector(case["coefficients"])):
            if coefficient is None:
                assert fit.fit.coefficients[index] == 0.0


def test_default_initialization_likelihood_covariance_and_working_score_match_r(reference_fit):
    case, fit = reference_fit
    assert survival.loglik(fit) == pytest.approx(
        case["loglik"], rel=3e-10, abs=1e-5 if gaussian_censoring(case) else 3e-9
    )
    # Like R, omitted-init rescaling retains the score in working coordinates.
    # The existing approximate Gaussian CDF can flatten likelihood changes near
    # the optimum while leaving a working score around 2.2e-5 on these data.
    assert fit.score_vector == pytest.approx(
        vector(case["score"]), rel=3e-6, abs=3e-5 if gaussian_censoring(case) else 2e-6
    )
    for actual, expected in zip(fit.variance_matrix, case["variance"], strict=True):
        assert actual == pytest.approx(
            expected,
            rel=1e-5 if gaussian_censoring(case) else 3e-6,
            abs=1e-5 if gaussian_censoring(case) else 5e-8,
        )


def selected_case(identifier):
    return next(case for case in REFERENCE["cases"] if case["id"] == identifier)


@pytest.mark.parametrize("suffix", ["", "no_intercept_"])
def test_gaussian_all_event_fixed_scale_start_is_weighted_least_squares(suffix):
    fit = fit_case(selected_case(f"gaussian_wls_{suffix}0"))
    key = "no_intercept_coefficients" if suffix else "coefficients"
    assert survival.coef(fit) == pytest.approx(REFERENCE["gaussian_wls"][key], rel=3e-12, abs=3e-12)
    assert all(abs(value) < 2e-12 for value in fit.score_vector)


def test_default_rescaling_preserves_covariate_units_and_stored_predictions():
    baseline = fit_case(selected_case("gaussian_wls_0"))
    changed = fit_case(selected_case("gaussian_covariate_units_0"))
    intercept, x, z = survival.coef(baseline)
    expected = [intercept - x * 1e10 / 1e8 + z * 0.02 / 1e-4, x / 1e8, z / 1e-4]
    assert survival.coef(changed) == pytest.approx(expected, rel=3e-11, abs=3e-11)
    assert changed.linear_predictors == pytest.approx(baseline.linear_predictors, abs=3e-12)
    assert survival.loglik(changed) == pytest.approx(survival.loglik(baseline), abs=3e-11)


def test_gaussian_initialization_preserves_response_translation_and_scale():
    baseline = fit_case(selected_case("gaussian_wls_0"))
    changed = fit_case(selected_case("gaussian_response_units_0"))
    expected = [7 * value for value in survival.coef(baseline)]
    expected[0] += 10
    assert survival.coef(changed) == pytest.approx(expected, rel=3e-12, abs=3e-12)
    assert changed.linear_predictors == pytest.approx(
        [7 * value + 10 for value in baseline.linear_predictors], rel=3e-12, abs=3e-12
    )
    for actual, original in zip(changed.variance_matrix, baseline.variance_matrix, strict=True):
        assert actual == pytest.approx([49 * value for value in original], rel=3e-12, abs=3e-12)
    total_weight = sum(REFERENCE["datasets"]["events"]["weight"])
    assert survival.loglik(changed) == pytest.approx(
        survival.loglik(baseline) - total_weight * math.log(7), abs=3e-11
    )


def test_explicit_initial_values_bypass_nonbinary_constant_rescaling():
    data = {"time": [1.0, 2.0, 3.0, 4.0], "status": [1] * 4, "x": [2.0] * 4}
    arguments = {"data": data, "dist": "gaussian", "scale": 1.0, "max_iter": 0}
    with pytest.raises(RuntimeError, match="cannot rescale a constant"):
        survival.survreg("Surv(time, status) ~ x", **arguments)
    fit = survival.survreg("Surv(time, status) ~ x", **arguments, init=[0.0, 0.0])
    assert fit.fit.coefficients == [0.0, 0.0]
    assert fit.iterations == 0


def test_explicit_initial_values_bypass_unrepresentable_interval_start_derivative():
    time = [1e-100, 2.0, 3.0]
    data = {
        "time": time,
        "upper": [math.nextafter(value, math.inf) for value in time],
        "status": [3] * 3,
    }
    formula = "Surv(time, upper, status, type='interval') ~ 1"
    arguments = {"data": data, "dist": "logistic", "scale": 1.0, "max_iter": 0}
    with pytest.raises(RuntimeError, match="interval probability is not finite and positive"):
        survival.survreg(formula, **arguments)
    fit = survival.survreg(formula, **arguments, init=[0.0])
    assert fit.fit.coefficients == [0.0]
    assert fit.iterations == 0
    assert math.isfinite(survival.loglik(fit))


def test_zero_weight_interval_matches_row_deletion_during_native_initialization():
    arguments = {
        "time": [1.0, 3.0],
        "status": [1.0, 1.0],
        "covariates": [[1.0], [1.0]],
        "weights": [1.0, 1.0],
        "distribution": "logistic",
        "fixed_scale": 1.0,
        "max_iter": 0,
    }
    baseline = survival.regression.survreg(**arguments)
    augmented = survival.regression.survreg(
        **{
            **arguments,
            "time": [1.0, 3.0, 1e-100],
            "time2": [1.0, 3.0, math.nextafter(1e-100, math.inf)],
            "status": [1.0, 1.0, 3.0],
            "covariates": [[1.0], [1.0], [1.0]],
            "weights": [1.0, 1.0, 0.0],
        }
    )
    assert augmented.coefficients == baseline.coefficients
    assert augmented.variance_matrix == baseline.variance_matrix
    assert augmented.log_likelihood == baseline.log_likelihood
    assert augmented.score_vector == baseline.score_vector


@pytest.mark.parametrize("fixed_scale", [1.0, None], ids=["fixed", "estimated"])
def test_zero_weight_extreme_covariate_does_not_change_native_rescaling(fixed_scale):
    arguments = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1.0] * 4,
        "covariates": [[1.0, value] for value in [1.0, 2.0, 3.0, 4.0]],
        "weights": [1.0] * 4,
        "distribution": "gaussian",
        "fixed_scale": fixed_scale,
        "max_iter": 0,
    }
    baseline = survival.regression.survreg(**arguments)
    augmented = survival.regression.survreg(
        **{
            **arguments,
            "time": arguments["time"] + [2.0],
            "status": arguments["status"] + [1.0],
            "covariates": arguments["covariates"] + [[1.0, 1e20]],
            "weights": arguments["weights"] + [0.0],
        }
    )
    assert augmented.coefficients == baseline.coefficients
    assert augmented.scales == baseline.scales
    assert augmented.variance_matrix == baseline.variance_matrix
    assert augmented.linear_predictors[:4] == baseline.linear_predictors
    assert augmented.log_likelihood == baseline.log_likelihood
    assert augmented.score_vector == baseline.score_vector
