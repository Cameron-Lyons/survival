"""Numeric matrix preparation preserves concordance counts and influence outputs."""

import importlib
from dataclasses import asdict

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
concordance_module = importlib.import_module("survival.r._concordance")


def _matrix(dtype, layout):
    values = np.array(
        [[3, 0, 2], [2, 1, 4], [4, 2, 0], [1, 4, 3], [0, 3, 1], [4, 1, 2], [2, 0, 4], [1, 2, 3]],
        dtype=dtype,
    )
    if layout == "fortran":
        return np.asfortranarray(values)
    if layout == "strided":
        return np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)[::2, ::2]
    if layout == "reversed":
        return values[::-1, ::-1]
    if layout == "readonly":
        values.flags.writeable = False
    return values


def _response(counting):
    stop = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]
    status = [1, 1, 0, 1, 1, 1, 0, 1]
    return (
        r.Surv([0.0, 0.0, 0.5, 1.0, 0.0, 1.5, 2.0, 1.0], stop, status)
        if counting
        else r.Surv(stop, status)
    )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.longdouble, np.int64, np.uint64, bool]
)
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "reversed", "readonly"])
@pytest.mark.parametrize("counting", [False, True])
def test_numeric_matrices_match_row_lists(dtype, layout, counting):
    values = _matrix(dtype, layout)
    original = values.copy()
    response = _response(counting)
    options = {
        "weights": [0.5, 1.0, 1.5, 2.0, 1.0, 0.5, 1.0, 1.5],
        "strata": ["b", "a", "b", "a", "b", "a", "b", "a"],
        "cluster": [1, 2, 3, 4, 1, 2, 3, 4],
        "names": ["first", "second", "third"],
        "timewt": "S",
        "influence": 3,
        "ranks": True,
    }
    actual = r.concordancefit(response, values, **options)
    expected = r.concordancefit(response, values.tolist(), **options)
    np.testing.assert_equal(asdict(actual), asdict(expected))
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("columns", [1, 3])
@pytest.mark.parametrize("std_err", [False, True])
def test_matrix_result_shapes_match_row_lists(columns, std_err):
    values = _matrix(np.float64, "strided")[:, :columns]
    options = {"std_err": std_err, "reverse": True}
    actual = r.concordancefit(_response(False), values, **options)
    expected = r.concordancefit(_response(False), values.tolist(), **options)
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize("dtype", [object, str])
def test_convertible_nonnumeric_matrices_preserve_scalar_coercion(dtype):
    values = _matrix(dtype, "contiguous")
    actual = r.concordancefit(_response(False), values)
    expected = r.concordancefit(_response(False), values.tolist())
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.empty((0, 3)), "x must not be empty"),
        (np.empty((8, 0)), "x must have at least one column"),
        (np.ones((7, 3)), "x and y are not the same length"),
        (np.full((8, 3), np.nan), "values contains non-finite value"),
        (np.full((8, 3), np.inf), "values contains non-finite value"),
    ],
)
def test_invalid_numeric_matrices_keep_validation_errors(values, message):
    with pytest.raises(ValueError, match=message):
        r.concordancefit(_response(False), values)


def test_masked_matrix_keeps_missing_value_validation():
    values = np.ma.array(_matrix(float, "contiguous"), mask=False)
    values.mask[2, 1] = True
    with pytest.raises(TypeError, match="float"):
        r.concordancefit(_response(False), values)


def test_numpy_matrix_subclass_keeps_row_major_predictor_order():
    with pytest.warns(PendingDeprecationWarning):
        values = np.matrix(_matrix(float, "contiguous"))
    actual = r.concordancefit(_response(False), values, influence=3, ranks=True)
    expected = r.concordancefit(_response(False), values.tolist(), influence=3, ranks=True)
    np.testing.assert_equal(asdict(actual), asdict(expected))


def test_extended_precision_coercion_ignores_numpy_underflow_setting():
    values = _matrix(np.longdouble, "contiguous")
    values[0, 0] = np.finfo(np.longdouble).smallest_subnormal
    expected = r.concordancefit(_response(False), values.tolist())
    with np.errstate(all="raise"):
        actual = r.concordancefit(_response(False), values)
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "readonly"])
def test_native_covariate_matrix_owns_flat_array_values(layout):
    values = np.arange(24.0)
    if layout == "strided":
        values = np.repeat(values, 2)[::2]
    elif layout == "reversed":
        values = values[::-1]
    elif layout == "readonly":
        values.flags.writeable = False
    expected = values.tolist()
    matrix = survival.core.CovariateMatrix(values, 8, 3)
    assert matrix.shape() == (8, 3)
    assert matrix.values == expected
    if values.flags.writeable:
        values[:] = -1.0
        assert matrix.values == expected


def test_native_covariate_matrix_preserves_masked_values():
    values = np.ma.array([1.0, 2.0], mask=[False, True])
    with (
        pytest.warns(UserWarning, match="converting a masked element to nan"),
        pytest.raises(ValueError, match="values contains non-finite value"),
    ):
        survival.core.CovariateMatrix(values, 1, 2)


@pytest.mark.parametrize("values", [np.array(["1", "2"]), np.array([1, None], dtype=object)])
def test_native_covariate_matrix_preserves_nonnumeric_validation(values):
    with pytest.raises(TypeError, match="must be real number"):
        survival.core.CovariateMatrix(values, 1, 2)


@pytest.mark.parametrize("as_arrays", [False, True])
@pytest.mark.parametrize(
    "formula",
    [
        "Surv(time, status) ~ x + z",
        "Surv(start, time, status) ~ x * z + strata(group)",
        "score ~ x + z",
        "Surv(time, status) ~ factor(group) + x",
    ],
)
def test_formula_arrays_preserve_complete_row_list_results(monkeypatch, formula, as_arrays):
    data = {
        "time": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 1, 1, 0, 1],
        "start": [0.0, 0.0, 0.5, 1.0, 0.0, 1.5, 2.0, 1.0],
        "x": [3.0, 2.0, np.nan, 1.0, 0.0, 4.0, 2.0, 1.0],
        "z": [0.0, 1.0, 2.0, 4.0, np.nan, 1.0, 0.0, 2.0],
        "score": [2, 1, 5, 4, 2, 5, 4, 3],
        "group": ["b", "a", "b", "a", "b", "a", "b", "a"],
        "weight": [0.5, 1.0, 1.5, 2.0, 1.0, 0.5, 1.0, 1.5],
        "subject": [1, 2, 3, 4, 1, 2, 3, 4],
    }
    if as_arrays:
        data = {name: np.asarray(values) for name, values in data.items()}
    options = {
        "weights": "weight",
        "cluster": "subject",
        "subset": [7, 6, 5, 4, 3, 2, 1, 0, 1],
        "na_action": "na.exclude",
        "influence": 3,
        "ranks": True,
    }
    actual = r.concordance(formula, data=data, **options)
    original_model_frame = concordance_module._model_frame

    def row_list_model_frame(*args, **kwargs):
        kwargs["as_array"] = False
        return original_model_frame(*args, **kwargs)

    monkeypatch.setattr(concordance_module, "_model_frame", row_list_model_frame)
    expected = r.concordance(formula, data=data, **options)
    np.testing.assert_equal(asdict(actual), asdict(expected))
