"""R references for AFT fits whose starting Newton step needs backtracking."""

import json
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "survreg_line_search_r_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["distribution"])
def test_backtracking_preserves_converged_aft_state(case):
    arguments = {
        "data": REFERENCE["data"],
        "dist": case["distribution"],
        "weights": REFERENCE["data"]["weight"],
        "scale": 0.9,
        "init": case["initial"],
        "score": True,
        "eps": 1e-12,
        "tol_chol": 1e-10,
    }
    if case["distribution"] == "t":
        arguments["parms"] = [5.0]
    initial = survival.survreg(REFERENCE["formula"], **arguments, max_iter=0)
    delta = [
        sum(value * score for value, score in zip(row, initial.score_vector, strict=True))
        for row in initial.variance_matrix
    ]
    full_step = survival.survreg(
        REFERENCE["formula"],
        **{
            **arguments,
            "init": [value + change for value, change in zip(case["initial"], delta, strict=True)],
        },
        max_iter=0,
    )
    assert survival.loglik(full_step) < survival.loglik(initial)

    fit = survival.survreg(REFERENCE["formula"], **arguments, max_iter=150)
    assert fit.convergence_flag == 0
    assert fit.fit.coefficients == pytest.approx(case["coefficients"], rel=2e-7, abs=2e-8)
    assert fit.linear_predictors == pytest.approx(case["linear_predictors"], rel=2e-7, abs=2e-8)
    assert fit.scales == [case["scales"]]
    assert survival.loglik(fit) == pytest.approx(case["loglik"], rel=3e-10, abs=3e-10)
    assert fit.score_vector == pytest.approx(case["score"], abs=2e-6)
    for actual, expected in zip(fit.variance_matrix, case["variance"], strict=True):
        assert actual == pytest.approx(expected, rel=2e-7, abs=2e-8)
