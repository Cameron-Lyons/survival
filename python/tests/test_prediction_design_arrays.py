"""Prediction designs preserve formula values, ownership and list-based methods."""

import importlib
import math
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import
from .test_prediction_row_labels import REFERENCE, fitted

survival = setup_survival_import()
formula = importlib.import_module("survival.r._formula")
types = importlib.import_module("survival.r._types")


def numeric_design(*names, intercept=False):
    return types._FormulaDesign(
        response=None,
        covariates=tuple(types._NumericDesignTerm(types._CovariateTerm(name)) for name in names),
        offsets=(),
        intercept=intercept,
    )


@pytest.mark.parametrize(
    "dtype", [bool, np.int32, np.int64, np.uint64, np.float32, float, ">f8", np.longdouble]
)
@pytest.mark.parametrize("stride", [1, 2, -1])
def test_numeric_arrays_preserve_float_conversion_and_own_storage(dtype, stride):
    source = np.array([0, 1, 3, 5, 9, 13], dtype=dtype)[::stride]
    source.flags.writeable = False
    expected = [[1.0, float(value)] for value in source]
    matrix = formula._design_array_from_spec(
        {"x": source}, numeric_design("x", intercept=True), len(source)
    )
    assert matrix.dtype == np.dtype("float64")
    assert matrix.flags.c_contiguous
    assert matrix.flags.owndata
    assert matrix.flags.writeable
    assert not np.shares_memory(matrix, source)
    np.testing.assert_array_equal(matrix, expected)
    original = source.copy()
    matrix[:, 1] = -99
    np.testing.assert_array_equal(source, original)


class UnboxedColumn(np.ndarray):
    def tolist(self):
        raise AssertionError("numeric prediction columns must not become scalar lists")

    def __iter__(self):
        raise AssertionError("numeric prediction columns must not be iterated as scalars")


def test_numeric_columns_remain_arrays_until_copied():
    source = np.array([1.5, 2.5, 3.5]).view(UnboxedColumn)
    actual = formula._design_array_from_spec({"x": source}, numeric_design("x"), 3)
    np.testing.assert_array_equal(actual, [[1.5], [2.5], [3.5]])


@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("names", [(), ("x",), ("x", "y")])
@pytest.mark.parametrize("intercept", [False, True])
def test_designs_keep_empty_dimensions(rows, names, intercept):
    data = {name: np.arange(rows) for name in names}
    actual = formula._design_array_from_spec(
        data, numeric_design(*names, intercept=intercept), rows
    )
    assert actual.shape == (rows, len(names) + int(intercept))
    assert actual.flags.c_contiguous
    assert actual.flags.owndata
    np.testing.assert_array_equal(
        actual,
        np.asarray(
            formula._design_rows_from_spec(data, numeric_design(*names, intercept=intercept), rows)
        ).reshape(actual.shape),
    )


@pytest.mark.parametrize("rows", [0, 2])
def test_evaluated_matrix_terms_keep_their_width(rows):
    term = types._CovariateTerm("x", transform="tt")
    design = types._FormulaDesign(
        response=None,
        covariates=(types._MatrixDesignTerm(term, ("first", "second")),),
        offsets=(),
    )
    values = np.arange(rows * 2, dtype=float).reshape(rows, 2)
    actual = formula._design_array_from_spec({}, design, rows, evaluated={term: values})
    assert actual.shape == (rows, 2)
    np.testing.assert_array_equal(actual, values)
    assert not np.shares_memory(actual, values)


def test_cached_numeric_variables_take_precedence_over_source_columns():
    design = numeric_design("x")
    term = design.covariates[0].term
    actual = formula._design_array_from_spec(
        {"x": np.array([100, 200])}, design, 2, evaluated={term: np.array([1.5, 2.5])}
    )
    np.testing.assert_array_equal(actual, [[1.5], [2.5]])


@pytest.mark.parametrize(
    "source", [np.array([1, None, 3], dtype=object), np.array(["1", "2", "3"])]
)
def test_non_numeric_array_dtypes_keep_scalar_coercion(source):
    design = numeric_design("x")
    np.testing.assert_array_equal(
        formula._design_array_from_spec({"x": source}, design, 3),
        formula._design_rows_from_spec({"x": source}, design, 3),
    )


@pytest.mark.parametrize("nullable", [False, True])
def test_pandas_columns_keep_missing_value_conversion(nullable):
    pd = pytest.importorskip("pandas")
    source = pd.Series([1, None, 3], dtype="Int64" if nullable else "float64")
    actual = formula._design_array_from_spec({"x": source}, numeric_design("x"), 3)
    np.testing.assert_array_equal(actual, [[1], [math.nan], [3]])


def test_interactions_preserve_order_rounding_and_special_float_values():
    data = {
        "x": np.array([-0.0, 1e308, 1e-308, math.inf, math.nan, 0.7]),
        "y": np.array([2.0, 2.0, 1e-308, 0.0, 1.0, 0.3]),
        "z": np.array([-1.0, 0.0, 1e308, 1.0, 0.0, 0.9]),
    }
    design = types._FormulaDesign(
        response=None,
        covariates=(types._InteractionDesignTerm(numeric_design("x", "y", "z").covariates),),
        offsets=(),
    )
    expected = np.asarray(formula._design_rows_from_spec(data, design, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = formula._design_array_from_spec(data, design, 6)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_numeric_column_length_errors_are_unchanged():
    with pytest.raises(ValueError, match="formula columns must have the same length"):
        formula._design_array_from_spec({"x": np.arange(2)}, numeric_design("x"), 3)


def test_extended_float_conversion_keeps_scalar_warning_behavior():
    source = np.array([np.finfo(np.longdouble).max, np.finfo(np.longdouble).tiny])
    design = numeric_design("x")
    expected = formula._design_rows_from_spec({"x": source}, design, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = formula._design_array_from_spec({"x": source}, design, 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model", ["cox", "cox_ridge", "aft", "aft_ridge"])
def test_public_model_matrices_and_predictions_keep_lists(model):
    fit = fitted(model, "na.omit")
    newdata = {name: np.asarray(values) for name, values in REFERENCE["data"].items()}
    matrix = survival.r.model_matrix(fit, newdata)
    assert isinstance(matrix["data"], list)
    assert all(isinstance(row, list) for row in matrix["data"])
    result = survival.r.predict(fit, newdata, type="lp", se_fit=True)
    assert isinstance(result.fit, list)
    assert isinstance(result.se_fit, list)
