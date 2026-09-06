"""Static R OLS and Student-t references for public Egger inference."""

import json
import math
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "egger_r_reference.json").read_text())
FIELDS = ("egger_intercept", "egger_se", "egger_t", "egger_p")


@pytest.fixture(params=REFERENCE["cases"], ids=lambda case: case["id"])
def case(request):
    return request.param


@pytest.fixture(params=["root", "validation"])
def publication_bias(request):
    namespace = survival if request.param == "root" else survival.validation
    return namespace.publication_bias_tests


def assert_reference(result, case, *, sign=1, response_scale=1.0):
    for field in FIELDS:
        expected = case[field]
        if field in ("egger_intercept", "egger_t"):
            expected *= sign
        # Relative-only probability comparison retains the df100 tiny tail.
        absolute_tolerance = 0.0 if field == "egger_p" else 2e-12
        actual = getattr(result, field)
        if field in ("egger_intercept", "egger_se"):
            actual /= response_scale
        assert actual == pytest.approx(expected, rel=2e-10, abs=absolute_tolerance)
    assert 0.0 <= result.egger_p <= 1.0


def test_egger_intercept_inference_matches_r(case, publication_bias):
    result = publication_bias(case["effects"], case["std_errors"])
    assert_reference(result, case)
    if case["id"] == "constructed_df100":
        assert result.egger_t == pytest.approx(10.0, abs=2e-12)
        assert 0.0 < result.egger_p < 1e-16


def test_egger_sign_reversal_preserves_standard_error_and_probability(case, publication_bias):
    result = publication_bias([-value for value in case["effects"]], case["std_errors"])
    assert_reference(result, case, sign=-1)


def test_egger_joint_effect_and_standard_error_scaling_preserves_inference(case, publication_bias):
    for factor in (1e-8, 1.0, 1e8):
        result = publication_bias(
            [factor * value for value in case["effects"]],
            [factor * value for value in case["std_errors"]],
        )
        assert_reference(result, case)


def test_egger_large_representable_responses_preserve_inference(case, publication_bias):
    factor = 1e160
    result = publication_bias([factor * value for value in case["effects"]], case["std_errors"])
    assert_reference(result, case, response_scale=factor)


@pytest.mark.parametrize("std_errors", [[1.0, 1.0, 1.0], [1.0, 1.0 + 1e-10, 1.0 - 1e-10]])
def test_egger_unidentifiable_precision_returns_missing_inference_in_any_units(
    publication_bias, std_errors
):
    for factor in (1e-8, 1.0, 1e8):
        result = publication_bias(
            [factor * value for value in [1.0, 2.0, 4.0]],
            [factor * value for value in std_errors],
        )
        assert all(math.isnan(getattr(result, field)) for field in FIELDS)


def test_egger_exact_zero_response_has_undefined_statistic(publication_bias):
    result = publication_bias([0.0, 0.0, 0.0], [1.0, 0.5, 0.25])
    assert result.egger_intercept == 0.0
    assert result.egger_se == 0.0
    assert math.isnan(result.egger_t)
    assert math.isnan(result.egger_p)


@pytest.mark.parametrize("sign", [-1, 1])
def test_egger_exact_nonzero_standardized_response_has_infinite_statistic(publication_bias, sign):
    std_errors = [1.0, 0.5, 0.25]
    result = publication_bias([sign * 2.0 * value for value in std_errors], std_errors)
    assert result.egger_intercept == sign * 2.0
    assert result.egger_se == 0.0
    assert result.egger_t == sign * math.inf
    assert result.egger_p == 0.0


def test_egger_unrepresentable_standardized_response_returns_missing_inference(publication_bias):
    result = publication_bias([1e308, -1e308, 1e308], [0.5, 0.25, 0.125])
    assert all(math.isnan(getattr(result, field)) for field in FIELDS)
