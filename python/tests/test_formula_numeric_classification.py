"""Numeric formula classification preserves scalar conversion and factor semantics."""

import importlib
import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
fit = importlib.import_module("survival.r._fit")
coerce = importlib.import_module("survival.r._coerce")
formula = importlib.import_module("survival.r._formula")
types = importlib.import_module("survival.r._types")


def _frame(source, *, rhs="x", **options):
    n = len(source)
    data = {"time": np.arange(1, n + 1, dtype=float), "status": np.ones(n), "x": source}
    return fit._model_frame("Surv(time, status) ~ " + rhs, data, **options)


def _assert_frame_equal(actual, expected):
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(np.signbit(actual.x), np.signbit(expected.x))
    assert actual.design == expected.design
    assert actual.names == expected.names
    assert actual.assign == expected.assign
    assert actual.na_action == expected.na_action
    assert actual.row_names == expected.row_names
    assert actual.y.time == expected.y.time
    assert actual.y.event == expected.y.event


@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint64, np.float16, np.float32, float, ">f8", np.longdouble]
)
@pytest.mark.parametrize("stride", [1, 2, -1])
def test_numeric_array_frames_match_scalar_lists(dtype, stride):
    values = np.array([0, 1, 3, 5, 9, 13], dtype=dtype)[::stride]
    values.flags.writeable = False
    _assert_frame_equal(_frame(values), _frame(values.tolist()))


@pytest.mark.parametrize("action", ["na.omit", "na.exclude", "na.pass"])
@pytest.mark.parametrize("subset", [None, [5, 3, 1, 0, 3, 2]])
def test_missing_and_special_float_values_keep_row_and_sign_semantics(action, subset):
    values = np.array([-0.0, 0.0, math.nan, math.inf, -math.inf, 1e-300])
    _assert_frame_equal(
        _frame(values, na_action=action, subset=subset),
        _frame(values.tolist(), na_action=action, subset=subset),
    )


def test_unsigned_values_above_signed_range_use_the_same_float_conversion():
    values = np.array([0, 2**63, 2**64 - 1], dtype=np.uint64)
    _assert_frame_equal(_frame(values), _frame(values.tolist()))


def test_extended_float_values_keep_scalar_conversion():
    values = np.array([np.finfo(np.longdouble).max, np.finfo(np.longdouble).tiny, -0.0])
    _assert_frame_equal(_frame(values), _frame(values.tolist()))


@pytest.mark.parametrize("rhs", ["x", "factor(x)", "log(x)", "I(x + 1)"])
def test_logical_explicit_factor_and_transformed_arrays_keep_their_design(rhs):
    values = np.array([False, True, False, True]) if rhs == "x" else np.array([1, 2, 1, 2])
    _assert_frame_equal(_frame(values, rhs=rhs), _frame(values.tolist(), rhs=rhs))


@pytest.mark.parametrize(
    "values", [np.array([1, None, 3], dtype=object), np.array(["1", "2", "3"])]
)
def test_object_and_string_columns_keep_scalar_classification(values):
    _assert_frame_equal(_frame(values), _frame(values.tolist()))


def test_masked_columns_keep_their_missing_rows():
    values = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    actual = _frame(values, na_action="na.exclude")
    _assert_frame_equal(actual, _frame([1.0, None, 3.0], na_action="na.exclude"))
    assert actual.na_action.rows == (2,)


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "Float64", "boolean"])
def test_pandas_numeric_and_nullable_columns_keep_scalar_classification(dtype):
    pd = pytest.importorskip("pandas")
    values = (
        pd.Series([True, None, False], dtype=dtype)
        if dtype == "boolean"
        else pd.Series([1, None, 3] if dtype[0].isupper() else [1, 2, 3], dtype=dtype)
    )
    _assert_frame_equal(
        _frame(values, na_action="na.exclude"), _frame(values.tolist(), na_action="na.exclude")
    )


def test_declared_numeric_factor_levels_keep_their_order_and_unused_levels():
    pd = pytest.importorskip("pandas")
    values = pd.Series(pd.Categorical([2, 1, 2, 1], categories=[2, 1, 3]))
    expected = coerce._RFactorVector([2, 1, 2, 1], [2, 1, 3])
    actual = _frame(values)
    _assert_frame_equal(actual, _frame(expected))
    assert actual.names == ["x1", "x3"]


def test_polars_numeric_columns_match_scalar_lists():
    pl = pytest.importorskip("polars")
    values = pl.Series([1.0, None, -0.0, 3.0])
    _assert_frame_equal(
        _frame(values, na_action="na.exclude"), _frame(values.to_list(), na_action="na.exclude")
    )


def test_numeric_dtype_classification_checks_response_length():
    term = types._CovariateTerm("x")
    data = {"x": np.arange(2)}
    with pytest.raises(ValueError, match="formula columns must have the same length"):
        formula._fit_single_design_term(data, term, 3, data)
