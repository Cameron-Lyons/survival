"""Tight R references around Student-t normal-approximation boundaries."""

import json
import math
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "student_t_normal_reference.json").read_text()
)
CASES = REFERENCE["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"df={case['df']}")
def test_student_t_normal_approximation_matches_direct_r_lower_tails(case):
    actual = survival.psurvreg(case["x"], 0.0, distribution="t", parms=case["df"])
    assert actual == sorted(actual)
    for x, value, expected, log_expected in zip(
        case["x"], actual, case["cdf"], case["log_cdf"], strict=True
    ):
        label = f"pt({x}, df={case['df']})"
        assert value > 0.0, label
        assert value == pytest.approx(expected, rel=1e-13, abs=0.0), label
        assert abs(math.log(value) - log_expected) <= 1e-13, label


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"df={case['df']}")
def test_student_t_normal_approximation_quantiles_match_r(case):
    actual = survival.qsurvreg(case["p"], 0.0, distribution="t", parms=case["df"])
    assert actual == sorted(actual)
    for p, value, expected in zip(case["p"], actual, case["quantile"], strict=True):
        assert value == pytest.approx(expected, rel=1e-13, abs=0.0), f"qt({p}, df={case['df']})"
