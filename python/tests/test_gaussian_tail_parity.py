"""R references for Gaussian tails through the public survival interfaces."""

import json
import math
from pathlib import Path

import pytest
import survival

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "gaussian_tail_r_reference.json").read_text()
)


def _assert_probabilities(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        # A fixed absolute tolerance would silently accept a zero for tiny tails.
        assert value == pytest.approx(reference, rel=3e-12, abs=8 * math.ulp(reference))


@pytest.mark.parametrize("case", REFERENCE["distributions"], ids=lambda case: case["id"])
def test_gaussian_cdf_preserves_r_tail_probabilities(case):
    actual = survival.psurvreg(
        case["x"], mean=case["mean"], scale=case["scale"], distribution=case["distribution"]
    )
    _assert_probabilities(actual, case["cdf"])
    assert all(left <= right for left, right in zip(actual[:-1], actual[1:], strict=True))


@pytest.mark.parametrize("case", REFERENCE["distributions"], ids=lambda case: case["id"])
def test_gaussian_density_matches_r_across_tail_range(case):
    actual = survival.dsurvreg(
        case["x"], mean=case["mean"], scale=case["scale"], distribution=case["distribution"]
    )
    _assert_probabilities(actual, case["density"])


@pytest.mark.parametrize("case", REFERENCE["quantiles"], ids=lambda case: case["id"])
def test_gaussian_quantiles_match_r_to_representable_probability_limits(case):
    actual = survival.qsurvreg(
        case["probabilities"],
        mean=case["mean"],
        scale=case["scale"],
        distribution=case["distribution"],
    )
    _assert_probabilities(actual, case["values"])
    assert all(left <= right for left, right in zip(actual[:-1], actual[1:], strict=True))


@pytest.mark.parametrize("case", REFERENCE["predictions"], ids=lambda case: case["id"])
def test_native_gaussian_quantile_predictions_match_r(case):
    actual = survival.regression.predict_survreg_quantile(
        case["covariates"],
        case["coefficients"],
        case["scale"],
        case["distribution"],
        case["probabilities"],
        offset=case["offsets"],
    )
    for row, expected in zip(actual.predictions, case["predictions"], strict=True):
        _assert_probabilities(row, expected)


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda case: case["id"])
def test_gaussian_aft_tail_likelihood_score_and_information_match_r(case):
    initial = case["initial"]
    if not isinstance(initial, list):
        initial = [initial]
    fit = survival.survreg(
        time=case["time"],
        time2=case["time2"],
        status=case["status"],
        covariates=[[1.0] for _ in case["time"]],
        distribution=case["distribution"],
        weights=case["weights"],
        offsets=case["offsets"],
        init=initial,
        scale=case["fixed_scale"],
        max_iter=0,
        score=True,
    )
    coefficients = case["coefficients"]
    if not isinstance(coefficients, list):
        coefficients = [coefficients]
    score = case["score"]
    if not isinstance(score, list):
        score = [score]
    assert fit.location_coefficients == pytest.approx(coefficients, abs=1e-14)
    assert fit.scale == pytest.approx(case["scale"], rel=1e-14)
    assert fit.iterations == 0
    assert survival.loglik(fit) == pytest.approx(case["loglik"], rel=2e-13, abs=2e-13)
    assert fit.score_vector == pytest.approx(score, rel=2e-11, abs=2e-12)
    for row, expected in zip(fit.variance_matrix, case["variance"], strict=True):
        assert row == pytest.approx(expected, rel=5e-10, abs=2e-12)


def test_model_summary_retains_small_two_sided_normal_p_value():
    fit = survival.survreg(
        time=[8.0, 9.0, 10.0],
        status=[1, 1, 1],
        covariates=[[1.0], [1.0], [1.0]],
        distribution="gaussian",
        init=[9.0],
        scale=1.0,
        max_iter=0,
    )
    reference = REFERENCE["confidence"]
    row = survival.model_summary(fit)["coefficients"][0]
    _assert_probabilities([row["coef"], row["se"], row["z"], row["p"]], reference["summary"])
    bounds = survival.confint(fit, level=reference["level"])[0]
    _assert_probabilities([bounds["lower"], bounds["upper"]], reference["coefficient_bounds"])


def test_probability_confidence_interval_handles_largest_finite_confidence_level():
    reference = REFERENCE["confidence"]
    assert reference["level"] == math.nextafter(1.0, 0.0)
    bounds = survival.survfit_confint(
        reference["probability"],
        reference["standard_error"],
        logse=False,
        conf_type="plain",
        conf_int=reference["level"],
    )
    _assert_probabilities(bounds.lower, reference["probability_lower"])
    _assert_probabilities(bounds.upper, reference["probability_upper"])


def test_cox_curve_confidence_interval_handles_largest_finite_confidence_level():
    reference = REFERENCE["confidence"]
    case = reference["cox"]
    fit = survival.coxph("Surv(time, status) ~ x", data=case["data"], init=[0.0], max_iter=0)
    curve = survival.survfit(
        fit,
        newdata={"x": [case["newdata"]["x"]]},
        conf_level=reference["level"],
        conf_type="log",
    )
    assert curve.time == case["time"]
    _assert_probabilities(curve.surv[0], case["survival"])
    _assert_probabilities(curve.conf_lower[0], case["lower"])
    _assert_probabilities(curve.conf_upper[0], case["upper"])
