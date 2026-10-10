"""Confidence-band transforms and array boundaries match independent stock R."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/confidence_band_reference.json").read_text()
)


def numbers(values):
    return [float("nan") if value in ("NA", "NaN") else float(value) for value in values]


def assert_bands(actual, expected):
    for field in ("lower", "upper"):
        values = np.asarray(getattr(actual, field))
        wanted = np.asarray(numbers(expected[field]))
        assert values.shape == wanted.shape
        np.testing.assert_allclose(values, wanted, rtol=2e-13, atol=2e-14, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("arrays", [False, True], ids=["lists", "arrays"])
@pytest.mark.parametrize("facade", [False, True], ids=["native", "r_facade"])
def test_confidence_bands_match_stock_r(case, arrays, facade):
    inputs = {
        field: None if case[field] is None else numbers(case[field])
        for field in ("p", "se", "selow")
    }
    if arrays:
        inputs = {
            field: None if values is None else np.asarray(values)
            for field, values in inputs.items()
        }
    function = survival.r_api.survfit_confint if facade else survival.surv_analysis.survfit_confint
    actual = function(
        **inputs,
        **{field: case[field] for field in ("logse", "conf_type", "conf_int", "ulimit")},
    )
    assert_bands(actual, case["expected"])


@pytest.mark.parametrize("conf_type", ["plain", "log", "log-log", "logit", "arcsin"])
def test_scalar_estimates_and_standard_errors_match_stock_r(conf_type):
    case = next(
        item
        for item in REFERENCE["cases"]
        if item["name"] == f"scalar/{conf_type}/TRUE/0.95/absent/TRUE"
    )
    assert_bands(survival.r_api.survfit_confint(0.5, 0.1, conf_type=conf_type), case["expected"])


@pytest.mark.parametrize("field", ["p", "se", "selow"])
@pytest.mark.parametrize("container", ["list", "object_array", "pandas", "polars", "masked"])
def test_nullable_confidence_band_inputs_preserve_missing_values(field, container):
    values = [0.8, None, 0.3]
    if container == "object_array":
        values = np.asarray(values, dtype=object)
    elif container == "pandas":
        values = pd.Series(values, dtype="Float64")
    elif container == "polars":
        values = pl.Series(values)
    elif container == "masked":
        values = np.ma.array([0.8, 99.0, 0.3], mask=[False, True, False])
    arguments = {"p": [0.8, 0.7, 0.3], "se": [0.1, 0.1, 0.1], "selow": [0.15, 0.15, 0.15]}
    arguments[field] = values
    actual = survival.r_api.survfit_confint(**arguments, conf_type="log-log")
    arguments[field] = [0.8, np.nan, 0.3]
    expected = survival.surv_analysis.survfit_confint(**arguments, conf_type="log-log")
    for name in ("lower", "upper"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("layout", ["strided", "negative", "unaligned", "readonly"])
@pytest.mark.parametrize("facade", [False, True], ids=["native", "r_facade"])
def test_confidence_bands_own_numeric_array_inputs(layout, facade):
    arrays = [np.asarray(values) for values in ([0.8, 0.7, 0.3], [0.1] * 3, [0.15] * 3)]
    for index, source in enumerate(arrays):
        if layout == "strided":
            storage = np.empty(6)
            storage[::2] = source
            arrays[index] = storage[::2]
        elif layout == "negative":
            arrays[index] = source[::-1].copy()[::-1]
        elif layout == "unaligned":
            view = np.ndarray(source.shape, dtype=np.float64, buffer=bytearray(25), offset=1)
            view[:] = source
            arrays[index] = view
        else:
            source.flags.writeable = False
    function = survival.r_api.survfit_confint if facade else survival.surv_analysis.survfit_confint
    actual = function(*arrays[:2], selow=arrays[2], conf_type="log")
    expected = survival.surv_analysis.survfit_confint(
        [0.8, 0.7, 0.3], [0.1] * 3, selow=[0.15] * 3, conf_type="log"
    )
    for array in arrays:
        if array.flags.writeable:
            array[:] = 0
    for field in ("lower", "upper"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("empty", [False, True], ids=["one_value", "empty"])
def test_invalid_confidence_type_rejected_for_any_input_width(empty):
    values = [] if empty else [0.5]
    with pytest.raises(ValueError, match="invalid conf.int type"):
        survival.surv_analysis.survfit_confint(values, values, conf_type="none")


@pytest.mark.parametrize("field", ["p", "se", "selow"])
def test_native_confidence_band_arrays_must_be_one_dimensional(field):
    arguments = {"p": [0.5], "se": [0.1], "selow": [0.15]}
    arguments[field] = np.ones((1, 1))
    with pytest.raises(TypeError, match="1-dimensional array"):
        survival.surv_analysis.survfit_confint(**arguments)
