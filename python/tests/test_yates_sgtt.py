"""SAS type III Yates tests against R survival 3.8-12."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/yates_sgtt_reference.json").read_text())


def numbers(values):
    return [math.nan if value is None else value for value in values]


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-9, atol=2e-11
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_sgtt_matches_r_for_linear_models(case):
    fit = r.YatesModel(
        case["formula"],
        case["data"],
        numbers(case["beta"]),
        case["variance"],
        sigma2=case["sigma2"],
    )
    for expected in case["results"]:
        result = r.yates(fit, expected["term"], method="sgtt")
        assert result.estimate[expected["term"]] == expected["estimate"][expected["term"]]
        for field in ("pmm", "std"):
            assert_close(result.estimate[field], expected["estimate"][field])
        for field in ("cmat", "mvar", "sas"):
            assert_close(getattr(result, field), expected[field])
        assert result.sas_names == expected["sas_names"]
        assert result.sas_row_names == expected["sas_row_names"]
        assert [row.name for row in result.test] == expected["test_names"]
        assert_close([[row.chisq, row.df, row.ss] for row in result.test], expected["tests"])
        # SGTT replaces the hypothesis tests; its marginal estimates remain
        # those of the SAS population, including non-estimable levels.
        direct = r.yates(fit, expected["term"], population="sas")
        assert_close(result.estimate["pmm"], direct.estimate["pmm"])
        assert direct.sas is None


def test_sgtt_cox_model_removes_the_baseline_intercept():
    fit = r.coxph("Surv(time, status) ~ factor(sex) + age", survival.datasets.load_lung())
    result = r.yates(fit, "sex", method="SGTT")
    direct = r.yates(fit, "sex", population="sas")
    assert_close(result.estimate["pmm"], direct.estimate["pmm"])
    assert_close(result.sas, np.eye(2))
    assert result.sas_names == ["factor(sex)2", "age"]
    assert result.test[0].name == "factor(sex)"
    assert result.test[0].df == 1
    assert result.test[0].chisq == pytest.approx(fit.coefficients[0] ** 2 / fit.var[0][0])


@pytest.mark.parametrize("population", ["data", "factorial", {"age": [50, 60]}])
def test_sgtt_requires_sas_population(population):
    fit = r.coxph("Surv(time, status) ~ factor(sex) + age", survival.datasets.load_lung())
    with pytest.raises(ValueError, match="population = sas"):
        r.yates(fit, "sex", population=population, method="sgtt")


def test_sgtt_requires_linear_prediction():
    fit = r.coxph("Surv(time, status) ~ factor(sex) + age", survival.datasets.load_lung())
    with pytest.raises(ValueError, match="predict = linear"):
        r.yates(fit, "sex", predict="risk", method="sgtt")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x", [], "rows and columns"),
        ("x", [[1]], "SAS design columns"),
        ("x", [[1, math.nan]], "SAS design"),
        ("assign", [0, 2], "term assignment"),
        ("adjustment_terms", [[1]], "term assignment"),
        ("test_terms", [(0, "invalid")], "term assignment"),
        ("coefficient_assign", [0], "coefficient assignments"),
        ("vmat", [[1, 0]], "variance rows"),
        ("vmat", [[1], [0]], "variance columns"),
        ("beta", [1, math.inf], "beta"),
        ("sigma2", -1, "sigma2"),
        ("include_intercept", False, "estimable SAS columns"),
    ],
)
def test_native_sgtt_validates_design_and_hypothesis_shapes(field, value, message):
    arguments = {
        "x": [[1, 0], [1, 1]],
        "assign": [0, 1],
        "adjustment_terms": [[]],
        "beta": [1, 2],
        "vmat": [[1, 0], [0, 1]],
        "coefficient_assign": [0, 1],
        "test_terms": [(1, "a")],
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        survival.validation.yates_sgtt(**arguments)


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
def test_sgtt_numpy_layouts_match_lists(layout, dtype):
    x = np.array(
        [
            [1, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 0],
        ],
        dtype=dtype,
    )
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "strided":
        x = np.repeat(x, 2, axis=1)[:, ::2]
    beta = np.arange(2, 6, dtype=dtype)[::-1]
    variance = np.eye(4, dtype=dtype)
    arguments = {
        "assign": [0, 1, 1, 2, 2, 3, 3, 3, 3],
        "adjustment_terms": [[3], [3], []],
        "coefficient_assign": [0, 1, 2, 3],
        "test_terms": [(1, "a"), (2, "b")],
    }
    actual = survival.validation.yates_sgtt(x=x, beta=beta, vmat=variance, **arguments)
    expected = survival.validation.yates_sgtt(
        x=x.tolist(), beta=beta.tolist(), vmat=variance.tolist(), **arguments
    )
    assert actual.sas == expected.sas
    assert actual.columns == expected.columns
    assert [(row.chisq, row.df) for row in actual.test] == [
        (row.chisq, row.df) for row in expected.test
    ]
