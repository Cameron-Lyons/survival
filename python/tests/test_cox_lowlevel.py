"""Bare Cox fitting: R numerical components, output shapes and safe ownership."""

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "cox_lowlevel_reference.json").read_text()
)


def response(case):
    if case["start"] is None:
        return r.Surv(case["time"], case["event"])
    return r.Surv(case["start"], case["time"], case["event"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_bare_fitters_against_r(case):
    function = getattr(r, case["fitter"] + "_fit")
    kwargs = {**case["arguments"], "column_names": case["column_names"]}
    kwargs["control"] = kwargs["control"] or None
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            function(case["x"], response(case), **kwargs)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(case["x"], response(case), **kwargs)
    assert [" ".join(str(w.message).split()) for w in caught] == [
        " ".join(message.split()) for message in case["warnings"]
    ]
    for name in (
        "coefficients",
        "var",
        "loglik",
        "score",
        "iter",
        "linear_predictors",
        "residuals",
        "means",
        "first",
    ):
        actual, value = getattr(result, name), expected[name]
        if value is None:
            assert actual is None, name
        elif name == "var" and case["name"] == "coxph_nonconvergence":
            # R omits Cholesky factorization before inversion after exhaustion.
            # The shared optimizer deliberately fixes that bug; verify the
            # information inverse independently from likelihood differences.
            beta = np.asarray(result.coefficients)
            step = 1e-4

            def likelihood(value):
                return function(
                    case["x"], response(case), init=value, control={"iter.max": 0}
                ).loglik[0]

            information = np.empty((len(beta), len(beta)))
            for i in range(len(beta)):
                for j in range(len(beta)):
                    di = np.eye(len(beta))[i] * step
                    dj = np.eye(len(beta))[j] * step
                    information[i, j] = -(
                        likelihood(beta + di + dj)
                        - likelihood(beta + di - dj)
                        - likelihood(beta - di + dj)
                        + likelihood(beta - di - dj)
                    ) / (4 * step**2)
            np.testing.assert_allclose(actual, np.linalg.inv(information), rtol=2e-6)
        else:
            numeric = np.asarray(value, dtype=float)
            np.testing.assert_allclose(actual, numeric, rtol=2e-8, atol=2e-9, err_msg=name)
    for name in ("method", "info"):
        assert getattr(result, name) == expected[name], name
    for name in ("classes", "coefficient_names", "row_names"):
        value = expected[name]
        assert getattr(result, name) == (None if value is None else tuple(value)), name


@pytest.mark.parametrize("order", ["C", "F", "strided", "mapping", "frame"])
def test_design_inputs_and_owned_results(order):
    case = REFERENCE["cases"][0]
    matrix = np.array(case["x"], order="F" if order == "F" else "C")
    if order == "strided":
        backing = np.empty((len(matrix), 4))
        backing[:, ::2] = matrix
        matrix = backing[:, ::2]
    x = matrix
    if order in {"mapping", "frame"}:
        x = dict(zip(case["column_names"], matrix.T, strict=True))
        if order == "frame":
            x = pytest.importorskip("pandas").DataFrame(x)
    result = r.coxph_fit(x, response(case), rownames=[str(i) for i in range(len(matrix))])
    np.testing.assert_allclose(result.coefficients, case["expected"]["coefficients"])
    if order in {"mapping", "frame"}:
        assert result.coefficient_names == ("age", "group")
    restored = pickle.loads(pickle.dumps(result))  # noqa: S301 - own test data
    assert restored.coefficients == result.coefficients
    assert restored.var == result.var
    assert restored.row_names == result.row_names
    original = result.linear_predictors
    matrix[:] = 0
    result.coefficients[0] = 99
    result.var[0][0] = 99
    result.residuals[0] = 99
    assert result.linear_predictors == original
    assert result.coefficients == restored.coefficients
    assert result.var == restored.var
    assert result.residuals == restored.residuals
    assert not hasattr(result._fit, "x")


def test_vector_design_and_explicit_metadata():
    y = r.Surv([1, 2, 3, 4], [1, 1, 0, 1])
    result = r.coxph_fit([0, 1, 0, 1], y, column_names=["group"], nocenter=[0, 1])
    assert result.coefficient_names == ("group",)
    assert result.means == [0]
    assert r.coxph_fit([], y, resid=False).residuals is None
    with pytest.raises(ValueError, match="Invalid formula"):
        r.agexact_fit([0, 1, 0, 1], y)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset": [0]}, "different numbers"),
        ({"weights": [1]}, "different numbers"),
        ({"weights": [0] * 4}, "weights"),
        ({"strata": [1]}, "different numbers"),
        ({"strata": [None] * 4}, "missing values"),
        ({"rownames": ["a"]}, "rownames"),
        ({"column_names": ["a", "b"]}, "column_names"),
        ({"method": "unknown"}, "method"),
        ({"resid": "no"}, "resid"),
    ],
)
def test_validation(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        r.coxph_fit([[0], [1], [0], [1]], r.Surv([1, 2, 3, 4], [1, 1, 0, 1]), **kwargs)


def test_response_and_matrix_validation():
    y = r.Surv([1, 2, 3, 4], [1, 1, 0, 1])
    with pytest.raises(TypeError, match="Surv"):
        r.coxph_fit([[1]], [1])
    with pytest.raises(ValueError, match="counting"):
        r.agreg_fit([[0], [1], [0], [1]], y)
    with pytest.raises(ValueError, match="different numbers"):
        r.coxph_fit([[1]], y)
    with pytest.raises(TypeError, match="numeric"):
        r.coxph_fit([["a"]] * 4, y)
    with pytest.raises(ValueError, match="finite"):
        r.coxph_fit([[0], [np.nan], [0], [1]], y)


def test_raw_offsets_and_full_model_agree_on_fitted_quantities():
    case = REFERENCE["cases"][0]
    offsets = np.linspace(1, 2, len(case["time"]))
    raw = r.coxph_fit(case["x"], response(case), offset=offsets)
    full = survival.regression.coxph_fit(
        case["time"], case["event"], case["x"], offset=offsets, nocenter=[]
    )
    np.testing.assert_allclose(raw.coefficients, full.coefficients, atol=1e-10)
    np.testing.assert_allclose(raw.var, full.var, atol=1e-10)
    np.testing.assert_allclose(raw.loglik, full.loglik, atol=1e-10)
    np.testing.assert_allclose(raw.linear_predictors, full.linear_predictors, atol=1e-10)
    np.testing.assert_allclose(raw.residuals, full.residuals, atol=1e-10)
