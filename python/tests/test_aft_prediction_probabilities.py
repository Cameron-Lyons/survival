"""AFT quantile columns and ignored arguments match independent stock R."""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aft_prediction_probability_reference.json").read_text()
)


def number(value):
    if value in ("NA", "NaN"):
        return float("nan")
    return float(value)


@cache
def fitted(name):
    spec = REFERENCE["fits"][name]
    return r.survreg(
        spec["formula"],
        REFERENCE["data"],
        dist=spec["dist"],
        scale=spec["scale"],
        parms=spec["parms"],
    )


def assert_reference(actual, reference):
    expected = np.asarray([number(value) for value in reference["values"]])
    shape = reference["dim"] or [len(expected)]
    expected = expected.reshape(shape, order="F")
    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=4e-8, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("as_arrays", [False, True], ids=["lists", "arrays"])
def test_aft_prediction_probabilities_match_stock_r(case, as_arrays):
    probabilities = case["p"]
    if isinstance(probabilities, list):
        probabilities = [number(value) for value in probabilities]
    actual = r.predict(
        fitted(case["fit"]),
        REFERENCE["newdata"],
        type=case["type"],
        p=probabilities,
        se_fit=case["se_fit"],
        _as_arrays=as_arrays,
    )
    if case["se_fit"]:
        assert_reference(actual.fit, case["expected"]["fit"])
        assert_reference(actual.se_fit, case["expected"]["se.fit"])
    else:
        assert_reference(actual, case["expected"])


@pytest.mark.parametrize("kind", ["response", "link", "lp", "linear", "terms"])
@pytest.mark.parametrize("errors", [False, True])
def test_nonquantile_predictions_do_not_coerce_unused_probability_objects(kind, errors):
    class Unused:
        def __float__(self):
            raise AssertionError("unused probability was converted")

        def __iter__(self):
            raise AssertionError("unused probability was iterated")

    fit = fitted("weibull/estimated")
    actual = r.predict(fit, REFERENCE["newdata"], type=kind, p=Unused(), se_fit=errors)
    expected = r.predict(fit, REFERENCE["newdata"], type=kind, se_fit=errors)
    if errors:
        np.testing.assert_array_equal(actual.fit, expected.fit)
        np.testing.assert_array_equal(actual.se_fit, expected.se_fit)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["quantile", "uquantile"])
def test_quantile_predictions_still_reject_nonnumeric_probability_strings(kind):
    with pytest.raises(TypeError, match="p must be array-like"):
        r.predict(fitted("weibull/estimated"), type=kind, p="unused")


def test_custom_quantile_callback_receives_missing_and_invalid_probabilities_in_one_batch():
    from .test_survreg_callbacks import REFERENCE as DENSITY_REFERENCE
    from .test_survreg_callbacks import definition, fit_case

    calls = []

    def constant_quantile(probabilities, parms=None):
        calls.append(np.asarray(probabilities).copy())
        return np.zeros(len(probabilities))

    distribution = definition()
    distribution["quantile"] = constant_quantile
    model = fit_case(DENSITY_REFERENCE["cases"][0], dist=distribution)
    calls.clear()
    probabilities = [0.2, np.nan, -0.1, np.inf, 0.8]
    actual = r.predict(model, type="quantile", p=probabilities, se_fit=True)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], probabilities)
    # The user's distribution determines the missing-probability behavior;
    # a constant standardized quantile gives the location and its standard error.
    location = r.predict(model, type="lp", se_fit=True)
    np.testing.assert_array_equal(
        actual.fit, np.repeat(np.asarray(location.fit)[:, None], 5, axis=1)
    )
    np.testing.assert_array_equal(
        actual.se_fit, np.repeat(np.asarray(location.se_fit)[:, None], 5, axis=1)
    )


MISSING_CASES = [
    case
    for case in REFERENCE["cases"]
    if case["fit"].startswith("weibull/") and case["p"] == [0.2, "NA", 0.8]
]


def nullable_values(container):
    if container == "list":
        return [0.2, None, 0.8]
    if container == "tuple":
        return (0.2, pd.NA, 0.8)
    if container == "object_array":
        return np.array([0.2, None, 0.8], dtype=object)
    if container == "pandas":
        return pd.Series([0.2, pd.NA, 0.8], dtype="Float64")
    return pl.Series([0.2, None, 0.8])


@pytest.mark.parametrize("case", MISSING_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("container", ["list", "tuple", "object_array", "pandas", "polars"])
def test_nullable_probability_containers_match_stock_r(case, container):
    actual = r.predict(
        fitted(case["fit"]),
        REFERENCE["newdata"],
        type=case["type"],
        p=nullable_values(container),
        se_fit=case["se_fit"],
    )
    if case["se_fit"]:
        assert_reference(actual.fit, case["expected"]["fit"])
        assert_reference(actual.se_fit, case["expected"]["se.fit"])
    else:
        assert_reference(actual, case["expected"])


@pytest.mark.parametrize("case", REFERENCE["queries"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", ["list", "tuple", "object_array", "pandas", "polars"])
def test_nullable_distribution_query_values_match_stock_r(case, container):
    actual = getattr(r, case["routine"])(
        nullable_values(container),
        case["mean"],
        scale=case["scale"],
        distribution=case["distribution"],
        parms=case["parms"],
    )
    assert_reference(actual, case["expected"])
