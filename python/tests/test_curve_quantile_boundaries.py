"""Curve quantile/median boundaries and numeric layouts against independent stock R."""

import json
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from survival.r._coerce import _r_factor

from .helpers import setup_survival_import
from .test_gil_release import _assert_detaches

survival = setup_survival_import()
r = survival.r_api
core = survival._survival
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/curve_quantile_boundary_reference.json").read_text()
)


def number(value):
    return float("nan") if value in ("NA", "NaN") else float(value)


@cache
def fitted(name):
    spec = REFERENCE["fits"][name]
    data = dict(REFERENCE["data"])
    data["event"] = pd.Categorical(data["event"], categories=REFERENCE["event_levels"])
    if spec["kind"] == "cox":
        model = r.coxph(spec["formula"], data, weights=data["weight"])
        newdata = {
            key: [values[row - 1] for row in spec["nd_rows"]]
            for key, values in REFERENCE["newdata"].items()
        }
        options = {"start_time": spec["start_time"]} if "start_time" in spec else {}
        return r.survfit(model, newdata=newdata, **options)
    options = {"conf_type": spec["conf_type"]} if "conf_type" in spec else {}
    if spec["kind"] == "aj":
        options["id"] = data["id"]
    return r.survfit(spec["formula"], data, weights=data["weight"], **options)


def probabilities(spec, arrays=False):
    kind, values = spec["kind"], spec["values"]
    if kind == "null":
        return None
    if kind == "factor":
        return pd.Categorical(values)
    if kind == "numeric":
        values = [number(value) for value in values]
        return np.asarray(values) if arrays else values
    dtype = bool if kind == "logical" else str
    # Empty logical/character vectors retain their type, as they do in R.
    return np.asarray(values, dtype=dtype) if arrays or not values else values


def assert_quantiles(actual, expected):
    for field in ("quantile", "lower", "upper"):
        wanted = expected.get(field)
        values = getattr(actual, field)
        if wanted is None:
            assert values is None
            continue
        wanted = np.asarray(wanted, dtype=float)
        values = np.asarray(values, dtype=float)
        assert values.shape == wanted.shape
        np.testing.assert_allclose(values, wanted, rtol=2e-9, atol=2e-10, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("arrays", [False, True], ids=["lists", "arrays"])
def test_curve_quantiles_and_medians_match_stock_r(case, arrays):
    tolerance = None if case["tolerance"] is None else number(case["tolerance"])
    options = {"scale": case["scale"], "tolerance": tolerance}

    def call():
        if case["method"] == "median":
            return r.median(fitted(case["fit"]), **options)
        return r.quantile(
            fitted(case["fit"]),
            probabilities(case["probs"], arrays),
            conf_int=case["conf_int"],
            **options,
        )

    expected = case["expected"]
    if "error" in expected:
        if case["fit"] == "aj":
            message = "multi-state"
        elif "invalid_tolerance" in case["name"]:
            message = "tolerance"
        else:
            message = expected["error"]
        with pytest.raises(ValueError, match=message):
            call()
    else:
        actual = call()
        assert actual.probs == [number(value) for value in case["probs"]["values"]]
        assert_quantiles(actual, expected)


@pytest.mark.parametrize("response", [False, True], ids=["fit", "Surv"])
@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(False),
        "0.5",
        b"0.5",
        None,
        [0.2, None, 0.8],
        (0.2, pd.NA, 0.8),
        np.array([0.2, None, 0.8], dtype=object),
        pd.Series([0.2, pd.NA, 0.8], dtype="Float64"),
        pl.Series([0.2, None, 0.8]),
        pd.Series([], dtype="boolean"),
        pd.Series([], dtype="string"),
        pl.Series([], dtype=pl.Boolean),
        pl.Series([], dtype=pl.String),
        pd.Categorical([], categories=[0.2, 0.8]),
        pl.Series([], dtype=pl.Categorical),
        pl.Series([], dtype=pl.Enum(["0.2", "0.8"])),
        _r_factor([], ["0.2", "0.8"]),
        _r_factor([0.2, 0.8], [0.2, 0.8]),
        np.array([], dtype="datetime64[D]"),
        np.array([], dtype="timedelta64[D]"),
        np.array([], dtype=np.complex128),
        np.array([], dtype=[("p", float)]),
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series([], dtype="period[M]"),
        pl.Series([], dtype=pl.Date),
        pl.Series([], dtype=pl.Datetime),
        pl.Series([], dtype=pl.Duration),
        pl.Series([], dtype=pl.Binary),
        pl.Series([], dtype=pl.Null),
    ],
)
def test_scalar_and_nullable_invalid_probabilities_follow_stock_refusal(invalid, response):
    value = r.Surv([1, 2, 3, 4]) if response else fitted("km_right")
    with pytest.raises(ValueError, match="invalid probability"):
        r.quantile(value, invalid)


@pytest.mark.parametrize(
    "numeric",
    [
        [],
        np.array([], dtype=float),
        np.array([], dtype=np.int64),
        np.array([], dtype=object),
        pd.Series([], dtype="Float64"),
        pd.Series([], dtype="Int64"),
        pd.Series([], dtype=object),
        pl.Series([], dtype=pl.Float64),
        pl.Series([], dtype=pl.Int64),
        pl.Series([], dtype=pl.Decimal),
    ],
)
def test_empty_numeric_probability_dtypes_preserve_stock_numeric_zero_shape(numeric):
    result = r.quantile(fitted("km_right"), numeric)
    assert result.probs == []
    assert result.quantile == [[]]
    assert result.lower == [[]]
    assert result.upper == [[]]


DTYPE_CASES = [
    ("numpy_b", "logical"),
    ("numpy_i", "integer"),
    ("numpy_u", "integer"),
    ("numpy_f", "numeric"),
    ("numpy_c", "complex"),
    ("numpy_m", "duration"),
    ("numpy_M", "date"),
    ("numpy_O", "numeric"),
    ("numpy_S", "character"),
    ("numpy_U", "character"),
    ("numpy_V", "raw"),
    ("pandas_float", "numeric"),
    ("pandas_integer", "integer"),
    ("pandas_nullable_float", "numeric"),
    ("pandas_nullable_integer", "integer"),
    ("pandas_nullable_missing", "missing"),
    ("pandas_object", "numeric"),
    ("pandas_categorical", "factor"),
    ("pandas_datetime", "date"),
    ("pandas_timedelta", "duration"),
    ("pandas_string", "character"),
    ("pandas_boolean", "logical"),
    ("polars_float", "numeric"),
    ("polars_integer", "integer"),
    ("polars_unsigned", "integer"),
    ("polars_decimal", "numeric"),
    ("polars_boolean", "logical"),
    ("polars_string", "character"),
    ("polars_categorical", "factor"),
    ("polars_enum", "factor"),
    ("polars_date", "date"),
    ("polars_datetime", "date"),
    ("polars_duration", "duration"),
    ("polars_binary", "raw"),
    ("polars_null", "missing"),
]


def dtype_values(name, empty):
    values = [] if empty else [0, 1]
    if name.startswith("numpy_"):
        kind = name[len("numpy_") :]
        dtypes = {
            "b": bool,
            "i": np.int64,
            "u": np.uint64,
            "f": float,
            "c": complex,
            "m": "timedelta64[s]",
            "M": "datetime64[D]",
            "O": object,
            "S": "S1",
            "U": "U1",
        }
        if kind == "V":
            return np.asarray([(value,) for value in values], dtype=[("p", float)])
        return np.asarray(values, dtype=dtypes[kind])
    kind = name.split("_", 1)[1]
    if name.startswith("pandas_"):
        if kind == "categorical":
            return pd.Categorical(values, categories=[0, 1])
        if kind in {"datetime", "timedelta"}:
            unit = "datetime64[ns]" if kind == "datetime" else "timedelta64[ns]"
            return pd.Series(np.asarray(values, dtype=unit))
        dtypes = {
            "float": float,
            "integer": np.int64,
            "nullable_float": "Float64",
            "nullable_integer": "Int64",
            "nullable_missing": "Float64",
            "object": object,
            "string": "string",
            "boolean": "boolean",
        }
        if kind == "nullable_missing":
            values = [] if empty else [pd.NA, pd.NA]
        elif kind == "string":
            values = [str(value) for value in values]
        return pd.Series(values, dtype=dtypes[kind])
    dtypes = {
        "float": pl.Float64,
        "integer": pl.Int64,
        "unsigned": pl.UInt64,
        "decimal": pl.Decimal,
        "boolean": pl.Boolean,
        "string": pl.String,
        "categorical": pl.Categorical,
        "enum": pl.Enum(["0", "1"]),
        "date": pl.Date,
        "datetime": pl.Datetime,
        "duration": pl.Duration,
        "binary": pl.Binary,
        "null": pl.Null,
    }
    if kind in {"string", "categorical", "enum"}:
        values = [str(value) for value in values]
    elif kind == "boolean":
        values = [bool(value) for value in values]
    elif kind == "date":
        values = [date(1970, 1, value + 1) for value in values]
    elif kind == "datetime":
        values = [datetime(1970, 1, value + 1) for value in values]
    elif kind == "duration":
        values = [timedelta(seconds=value) for value in values]
    elif kind == "binary":
        values = [bytes([value]) for value in values]
    elif kind == "null":
        values = [] if empty else [None, None]
    return pl.Series(values, dtype=dtypes[kind])


@pytest.mark.parametrize("case", DTYPE_CASES, ids=lambda case: case[0])
@pytest.mark.parametrize("empty", [True, False], ids=["empty", "nonempty"])
def test_probability_dtype_table_matches_stock_r(case, empty):
    name, r_kind = case
    # A nullable numeric column with no rows still represents numeric(0).
    if name == "pandas_nullable_missing" and empty:
        r_kind = "numeric"
    expected = REFERENCE["dtype_cases"][f"{r_kind}/{'empty' if empty else 'nonempty'}"]["expected"]
    values = dtype_values(name, empty)
    if "error" in expected:
        with pytest.raises(ValueError, match=expected["error"]):
            r.quantile(fitted("km_right"), values)
    else:
        assert_quantiles(r.quantile(fitted("km_right"), values), expected)


def array_layout(values, layout):
    array = np.asarray(values, dtype=float)
    if layout == "strided":
        storage = np.empty(2 * len(array))
        storage[::2] = array
        return storage[::2]
    if layout == "negative":
        return array[::-1].copy()[::-1]
    if layout == "unaligned":
        view = np.ndarray(
            array.shape, dtype=np.float64, buffer=bytearray(array.nbytes + 1), offset=1
        )
        view[:] = array
        return view
    if layout == "readonly":
        array.flags.writeable = False
    return array


NATIVE_CASES = [
    case
    for case in REFERENCE["cases"]
    if case["fit"] in {"km_right", "km_groups", "km_counting", "km_censor", "turnbull"}
    and case["method"] == "quantile"
    and case["probs"]["kind"] == "numeric"
    and "error" not in case["expected"]
    and case["scale"] == 1
    and case["conf_int"]
    and case["tolerance"] in (None, "Inf", "-Inf")
]


@pytest.mark.parametrize("case", NATIVE_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "negative", "unaligned", "readonly"])
@pytest.mark.parametrize("stacked", [False, True], ids=["prepared", "stacked"])
def test_native_curve_quantile_array_layouts_match_stock_r(case, layout, stacked):
    engine = fitted(case["fit"]).engine
    probs = array_layout(probabilities(case["probs"]), layout)
    tolerance = None if case["tolerance"] is None else number(case["tolerance"])
    options = {"probs": probs, "tolerance": tolerance}
    if stacked:
        curves = {
            field: None
            if getattr(engine, field) is None
            else array_layout(getattr(engine, field), layout)
            for field in ("time", "surv", "lower", "upper")
        }
        actual = core.quantile_survfit_curves(**curves, strata=engine.strata, **options)
    else:
        actual = core.quantile_survfit(engine, **options)
    assert_quantiles(actual, case["expected"])
    # Returned values remain owned when mutable source arrays are reused.
    if probs.flags.writeable:
        probs[:] = 0
    if stacked:
        for values in curves.values():
            if values is not None and values.flags.writeable:
                values[:] = 0
        assert_quantiles(actual, case["expected"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_stacked_numeric_dtypes_preserve_half_step_quantiles(dtype):
    # A fully observed four-subject empirical curve has R's midpoint quantiles.
    times = np.asarray([1, 2, 3, 4], dtype=dtype)
    probabilities_ = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    actual = core.quantile_survfit_curves(
        times, np.asarray([0.75, 0.5, 0.25, 0], dtype=np.float32), probs=probabilities_
    )
    assert actual.quantile == [[1.5, 2.5, 3.5]]


@pytest.mark.parametrize("field", ["time", "surv", "lower", "upper", "probs"])
def test_stacked_curve_inputs_require_one_dimension(field):
    arguments = {"time": [1], "surv": [0.5], "lower": [0.2], "upper": [0.8], "probs": [0.5]}
    arguments[field] = np.ones((1, 1))
    with pytest.raises(TypeError, match="1-dimensional array"):
        core.quantile_survfit_curves(**arguments)


def test_stacked_curve_quantiles_release_the_gil():
    n = 1_000_000
    times = np.arange(n, dtype=float)
    estimates = 1 - (times + 1) / n
    lower = np.maximum(estimates - 0.1, 0)
    upper = np.minimum(estimates + 0.1, 1)
    probabilities_ = np.linspace(0, 1, 1001)
    _assert_detaches(
        lambda: core.quantile_survfit_curves(
            times, estimates, lower=lower, upper=upper, probs=probabilities_
        )
    )
