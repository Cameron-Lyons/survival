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


@pytest.mark.parametrize("container", ["polars", "pandas"])
def test_numeric_extension_designs_vectorize_when_the_null_mask_is_empty(container, monkeypatch):
    if container == "polars":
        pl = pytest.importorskip("polars")
        complete = pl.Series([1.0, np.nan])
        incomplete = pl.Series([None, np.nan])
    else:
        pd = pytest.importorskip("pandas")
        values = np.array([1.0, np.nan])
        complete = pd.arrays.FloatingArray(values, np.array([False, False]))
        incomplete = pd.arrays.FloatingArray(values, np.array([True, False]))

    design = numeric_design("x")
    matrix = formula._design_array_from_spec({"x": incomplete}, design, 2)
    assert matrix[:, 0].view(np.uint64).tolist() == [
        0x7FF80000000007A2,
        np.asarray(np.nan).view(np.uint64).item(),
    ]

    def unexpected_scalar_conversion(*args):
        raise AssertionError("a null-free numeric column should use the array path")

    monkeypatch.setattr(formula, "_numeric_variable", unexpected_scalar_conversion)
    matrix = formula._design_array_from_spec({"x": complete}, design, 2)
    np.testing.assert_array_equal(matrix[:, 0], [1.0, np.nan])


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


@pytest.mark.parametrize("as_array", [False, True])
def test_interaction_reuses_only_matching_factor_codings(monkeypatch, as_array):
    from dataclasses import replace

    group = types._CovariateTerm("g")
    x = types._NumericDesignTerm(types._CovariateTerm("x"))
    treatment = types._CategoricalDesignTerm(
        group, ("a", "b"), contrasts=((0.0,), (1.0,)), contrast_metadata={"data": np.array([0, 1])}
    )
    # Restored specs may carry distinct metadata arrays for the same coding.
    restored = replace(treatment, contrast_metadata={"data": np.array([0, 1])})
    full = replace(treatment, full=True)
    reverse = replace(treatment, levels=("b", "a"))
    contrast = replace(treatment, contrasts=((-0.0,), (1.0,)))
    design = types._FormulaDesign(
        response=None,
        covariates=(treatment, x)
        + tuple(types._InteractionDesignTerm((g, x)) for g in (restored, full, reverse, contrast)),
        offsets=(),
    )
    calls = []
    level_codes = formula._level_codes

    def count_codes(values, levels, column):
        calls.append(column)
        return level_codes(values, levels, column)

    monkeypatch.setattr(formula, "_level_codes", count_codes)
    build = formula._design_array_from_spec if as_array else formula._design_rows_from_spec
    actual = build({"g": ["a", "b"], "x": [2, 3]}, design, 2)
    np.testing.assert_array_equal(actual, [[0, 2, 0, 2, 0, 2, -0.0], [1, 3, 3, 0, 3, 0, 3]])
    assert np.signbit(actual[0][-1])
    assert calls == ["g"] * 4


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


def test_aft_matrix_input_preserves_fitted_and_new_row_names():
    pd = pytest.importorskip("pandas")
    time = np.arange(1, 19)
    names = [f"d{i}" for i in time]
    x = pd.DataFrame({"one": np.ones(18), "x": np.sin(time)}, index=names)
    r = survival.r
    fit = r.survreg(r.Surv(time, np.ones(18)), x=x)
    assert r.model_matrix(fit, _with_metadata=True)["row_names"] == names
    assert r.predict(fit, type="terms", _with_row_names=True)["fit_names"] == names

    new = pd.DataFrame({"x": [1, np.nan, 3], "one": [1, 1, 1]}, index=list("abc"))
    for data, labels in [(new, ["a", "c"]), ([[1, 1], [1, None], [1, 3]], ["1", "3"])]:
        matrix = r.model_matrix(fit, data, _with_metadata=True)
        assert matrix["row_names"] == labels
        assert matrix["data"] == [[1.0, 1.0], [1.0, 3.0]]
        assert (
            r.predict(fit, data, type="terms", na_action="omit", _with_row_names=True)["fit_names"]
            == labels
        )

    # survival 3.8-12: predict(survreg(Surv(time) ~ 0 + one + x), new,
    #                          type="terms", terms=c("x","one"), se.fit=TRUE).
    actual = r.predict(fit, new, type="terms", terms=["x", "one"], se_fit=True)
    np.testing.assert_allclose(
        actual.fit,
        [
            [-0.15882939434507, 2.35648790302136],
            [np.nan, 2.35648790302136],
            [-0.476488183035209, 2.35648790302136],
        ],
    )
    np.testing.assert_allclose(
        actual.se_fit,
        [
            [0.0274974363698456, 0.309379896523035],
            [np.nan, 0.309379896523035],
            [0.0824923091095368, 0.309379896523035],
        ],
    )


def test_aft_matrix_prediction_accepts_numeric_views_without_scalar_conversion():
    time = np.arange(1, 19)
    design = np.column_stack((np.ones(18), np.sin(time)))
    r = survival.r
    fit = r.survreg(r.Surv(time, np.ones(18)), x=design)
    new = design[::2].view(UnboxedColumn)
    new.flags.writeable = False
    expected = r.predict(fit, design[::2].tolist(), type="terms")
    np.testing.assert_array_equal(r.predict(fit, new, type="terms"), expected)


@pytest.mark.parametrize("formula_fit", [False, True])
def test_aft_keeps_nullable_and_masked_na_distinct_from_numeric_nan(formula_fit):
    pd = pytest.importorskip("pandas")
    r = survival.r
    time = np.arange(1, 19)
    x = {"one": np.ones(18), "x": np.sin(time)}
    fit = (
        r.survreg("Surv(time) ~ 0 + one + x", {"time": time, **x})
        if formula_fit
        else r.survreg(r.Surv(time, np.ones(18)), x=x)
    )
    for column in (pd.Series([1, pd.NA], dtype="Int64"), np.ma.array([1, 0], mask=[False, True])):
        new = {"one": [1, float("nan")], "x": column}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            outputs = [
                r.model_matrix(fit, new, na_action="pass")["data"],
                r.predict(fit, new, type="terms"),
            ]
        for rows in outputs:
            ordinary_nan, missing = np.array(rows[1]).view(np.uint64)
            assert ordinary_nan == np.asarray(float("nan")).view(np.uint64)
            assert missing == 0x7FF80000000007A2
