"""Joint-variable marginal means and tests against R survival 3.8-12."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/yates_joint_reference.json").read_text())


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-8, atol=2e-10
    )


def model(case):
    if case["cox"]:
        return r.coxph(case["formula"], case["data"])
    return r.YatesModel(
        case["formula"],
        case["data"],
        [math.nan if value is None else value for value in case["beta"]],
        case["variance"],
        sigma2=case["sigma2"],
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_joint_yates_matches_r(case):
    result = r.yates(
        model(case),
        case["term"],
        levels=case["levels"],
        population=case["population"],
        test=case["test"],
        method=case["method"],
        predict=case["predict"],
        nsim=100,
        options={"seed": 123} if case["predict"] == "risk" else None,
    )
    assert list(result.estimate) == list(case["estimate"])
    for name, expected in case["estimate"].items():
        if name in ("pmm", "std"):
            assert_close(result.estimate[name], expected)
        else:
            assert result.estimate[name] == expected
    for name in ("cmat", "mvar", "sas"):
        expected = case[name]
        if expected is not None:
            assert_close(getattr(result, name), expected)
    assert [row.name for row in result.test] == case["test_names"]
    tests = [[row.chisq, row.df] for row in result.test]
    if not case["cox"]:
        tests = [values + [row.ss] for values, row in zip(tests, result.test, strict=True)]
    assert_close(tests, case["tests"])


def test_joint_yates_keeps_requested_factor_order():
    # R 3.8-12 assigns the formula's factor levels to the request's columns
    # positionally and fails here with "infinite or missing values in 'x'".
    fit = model(REFERENCE["cases"][0])
    forward = r.yates(fit, "a + b")
    reverse = r.yates(fit, "b + a")
    assert reverse.estimate["b"] == ["x", "y", "x", "y", "x", "y"]
    assert reverse.estimate["a"] == ["a", "a", "b", "b", "c", "c"]
    permutation = [0, 3, 1, 4, 2, 5]
    assert_close(reverse.estimate["pmm"], np.asarray(forward.estimate["pmm"])[permutation])
    assert_close(reverse.mvar, np.asarray(forward.mvar)[np.ix_(permutation, permutation)])
    assert reverse.test[0].chisq == pytest.approx(forward.test[0].chisq)
    assert reverse.test[0].df == forward.test[0].df
    sgtt = r.yates(fit, "b + a", method="sgtt")
    assert [row.name for row in sgtt.test] == ["b", "a"]


def test_joint_risk_global_test_has_one_name():
    # R 3.8-12 gives its one-row test two row names and errors. The Rust
    # contrast kernel already tests all combinations jointly.
    fit = model(next(case for case in REFERENCE["cases"] if case["name"] == "cox_risk"))
    result = r.yates(fit, "a+b", predict="risk", nsim=100, options={"seed": 123})
    difference = np.eye(6)[:5]
    difference[:, -1] = -1
    estimate = difference @ result.estimate["pmm"]
    variance = difference @ result.mvar @ difference.T
    assert result.test[0].name == "global"
    assert result.test[0].df == 5
    assert result.test[0].chisq == pytest.approx(estimate @ np.linalg.solve(variance, estimate))


@pytest.mark.parametrize(
    ("term", "levels", "message"),
    [
        ("a + absent", None, "variable absent not found"),
        ("a + z", None, "continuous variables require"),
        ("a + z", {"a": ["a"]}, "levels information not found for: z"),
        ("a + b", ["a", "b"], "levels should be a data frame or mapping"),
        ("a + b", {"a": ["a", "a"]}, "has duplicates"),
        ("a + b", {"b": ["absent"]}, "invalid level for term b"),
        ("a + b", {"a": []}, "must not be empty"),
        ("1", None, "must select variables"),
        ("strata(a)", None, "must select variables"),
    ],
)
def test_joint_yates_rejects_invalid_selection(term, levels, message):
    fit = model(next(case for case in REFERENCE["cases"] if case["name"] == "sas"))
    with pytest.raises(ValueError, match=message):
        r.yates(fit, term, levels=levels)
