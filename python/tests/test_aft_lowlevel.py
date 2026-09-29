"""Prepared AFT fits against R, with native ownership and callback checks."""

import json
import math
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
    (Path(__file__).parent / "fixtures" / "aft_lowlevel_reference.json").read_text()
)


def gaussian_density(z):
    z = np.asarray(z)
    return np.column_stack(
        (
            [math.erfc(-value / math.sqrt(2)) / 2 for value in z],
            [math.erfc(value / math.sqrt(2)) / 2 for value in z],
            np.exp(-z * z / 2) / math.sqrt(2 * math.pi),
            -z,
            z * z - 1,
        )
    )


def gaussian_init(y, weights):
    mean = np.average(y, weights=weights)
    return [mean, np.average((y - mean) ** 2, weights=weights)]


def custom_distribution(initial=True):
    value = {
        "name": "Custom Gaussian",
        "density": gaussian_density,
        "trans": lambda x: pytest.fail("bare fitter called response transform"),
        "scale": 100,
    }
    if initial:
        value["init"] = gaussian_init
    return value


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survreg_fit_against_r(case):
    kwargs = {**case["arguments"], "column_names": case["column_names"]}
    kwargs["controlvals"] = kwargs["controlvals"] or None
    if kwargs["dist"].startswith("custom"):
        kwargs["dist"] = custom_distribution(kwargs["dist"] != "custom_no_init")
    expected = case["expected"]
    if "error" in expected:
        message = (
            "Student-t distribution requires an explicit parms value"
            if case["name"] == "t_missing_parameters"
            else expected["error"]
        )
        with pytest.raises((ValueError, RuntimeError), match=re.escape(message)):
            r.survreg_fit(case["x"], case["y"], **kwargs)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.survreg_fit(case["x"], case["y"], **kwargs)
    assert [str(w.message) for w in caught] == case["warnings"]
    for name in ("coefficients", "icoef", "var", "loglik", "linear_predictors", "score"):
        np.testing.assert_allclose(
            getattr(fit, name),
            np.asarray(expected[name], dtype=float),
            rtol=2e-7,
            atol=2e-8,
            err_msg=name,
        )
    for name in ("iter", "df"):
        assert getattr(fit, name) == expected[name], name
    for name in ("coefficient_names", "icoef_names", "variance_names"):
        value = expected[name]
        assert getattr(fit, name) == (None if value is None else tuple(value)), name


@pytest.mark.parametrize("layout", ["C", "F", "strided", "mapping", "frame"])
def test_matrix_inputs_and_output_ownership(layout):
    case = REFERENCE["cases"][0]
    x = np.array(case["x"], order="F" if layout == "F" else "C")
    if layout == "strided":
        backing = np.zeros((len(x), 2 * x.shape[1]))
        backing[:, ::2] = x
        x = backing[:, ::2]
    design = x
    if layout in {"mapping", "frame"}:
        design = dict(zip(case["column_names"], x.T, strict=True))
        if layout == "frame":
            design = pytest.importorskip("pandas").DataFrame(design)
    result = r.survreg_fit(design, case["y"])
    restored = pickle.loads(pickle.dumps(result))  # noqa: S301 - own test data
    assert restored.coefficients == result.coefficients
    assert restored.var == result.var
    if layout in {"mapping", "frame"}:
        assert result.coefficient_names == ("Intercept", "age", "group", "Log(scale)")
    x[:] = 100
    result.coefficients[0] = 100
    result.var[0][0] = 100
    result.score[0] = 100
    assert result.coefficients == restored.coefficients
    assert result.var == restored.var
    assert result.score == restored.score
    assert not hasattr(result._fit, "covariates")
    assert not hasattr(result._fit, "distribution")


def test_minimal_callbacks_are_not_retained_or_probed():
    case = REFERENCE["cases"][0]
    result = r.survreg_fit(case["x"], case["y"], dist=custom_distribution())
    # The distribution has an unpicklable lambda; only numerical output survives.
    restored = pickle.loads(pickle.dumps(result))  # noqa: S301 - own test data
    assert restored.loglik == result.loglik
    expected = r.survreg_fit(case["x"], case["y"], dist="gaussian")
    np.testing.assert_allclose(result.coefficients, expected.coefficients, atol=1e-9)


@pytest.mark.parametrize("name", ["custom_gaussian", "weibull", "t"])
def test_registered_custom_density(name, monkeypatch):
    monkeypatch.setitem(r.survreg_distributions, name, custom_distribution())
    case = REFERENCE["cases"][0]
    fit = r.survreg_fit(case["x"], case["y"], dist=name)
    expected = r.survreg_fit(case["x"], case["y"], dist="gaussian")
    np.testing.assert_allclose(fit.coefficients, expected.coefficients, atol=1e-9)


def test_surv_inputs_keep_prepared_status_codes():
    case = REFERENCE["cases"][0]
    y = np.asarray(case["y"])
    expected = r.survreg_fit(case["x"], y)
    for kind in ("right", "left"):
        # R's bare function reads the numeric columns, ignoring the type attribute.
        result = r.survreg_fit(case["x"], r.Surv(y[:, 0], y[:, 1], type=kind))
        assert result.coefficients == expected.coefficients
    intervals = np.column_stack((y[:, 0], y[:, 0] + 0.5, np.tile([0, 1, 2, 3], 3)))
    result = r.survreg_fit(
        case["x"], r.Surv(intervals[:, 0], intervals[:, 1], intervals[:, 2], type="interval")
    )
    expected = r.survreg_fit(case["x"], intervals)
    np.testing.assert_allclose(result.coefficients, expected.coefficients)


def test_raw_and_full_model_alias_handling():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "gaussian_alias")
    y = np.asarray(case["y"])
    data = survival.regression.SurvregData(y[:, 0], y[:, 1].astype(np.int32), case["x"])
    distribution = survival.regression.SurvregDistribution("gaussian")
    raw = survival.regression.survreg_fit_raw(data, distribution)
    full = survival.regression.survreg_fit(data, distribution)
    assert raw.coefficients[3] == 0
    assert math.isnan(full.coefficients[3])
    assert raw.var == full.variance_matrix
    assert raw.loglik == [full.intercept_only_log_likelihood, full.log_likelihood]
    assert raw.linear_predictors == full.linear_predictors


def test_native_distribution_settings_and_cluster_guard():
    case = REFERENCE["cases"][0]
    base = survival.regression.SurvregDistribution("gaussian")

    def unused(values):
        pytest.fail("bare fitting invoked a transform callback")

    distribution = base.derived(
        "Fixed transformed Gaussian", survival.regression.SurvregTransform.Identity, 100
    ).with_transform(unused, unused, unused)
    result = r.survreg_fit(case["x"], case["y"], dist=distribution)
    expected = r.survreg_fit(case["x"], case["y"], dist=base)
    assert result.coefficients == expected.coefficients
    y = np.asarray(case["y"])
    data = survival.regression.SurvregData(
        y[:, 0], y[:, 1].astype(np.int32), case["x"], cluster=[0] * len(y)
    )
    with pytest.raises(ValueError, match="bare AFT fits do not compute robust variance"):
        survival.regression.survreg_fit_raw(data, base)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nstrat": 0}, "positive"),
        ({"nstrat": 2, "strata": [1, 2.5, 1, 2]}, "Invalid strata"),
        ({"nstrat": 2, "strata": [1, 3, 1, 2]}, "Invalid strata"),
        ({"nstrat": 2, "strata": [1, float("nan"), 1, 2]}, "Invalid strata"),
        ({"column_names": ["a", "b"]}, "column_names"),
        ({"offset": [0]}, "offset"),
        ({"dist": {"name": "Missing density"}}, "Missing density"),
    ],
)
def test_validation(kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        r.survreg_fit([[1]] * 4, [[1, 1], [2, 0], [3, 1], [4, 1]], **kwargs)


@pytest.mark.parametrize("y", [[[1]], [[1, 4]], [[1, 0.5]], [[1, None]], [1, 2]])
def test_invalid_response(y):
    with pytest.raises((ValueError, TypeError), match="response|status|numeric"):
        r.survreg_fit([[1]], y)
