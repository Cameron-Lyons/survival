"""Native array readers preserve NumPy masks before kernels validate missingness."""

import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

kernels = setup_survival_import().kernels
DTYPES = [np.float64, np.float32, np.int32, np.bool_, ">f8", object]
LAYOUTS = ["C", "F", "reverse", "strided", "readonly", "unaligned"]


def masked_array(values, dtype, layout):
    storage = np.asarray(values, dtype=dtype)
    mask = np.zeros(storage.shape, dtype=bool)
    mask.flat[1] = True
    if layout == "unaligned" and storage.dtype.kind != "O":
        array = np.ndarray(
            storage.shape, dtype=dtype, buffer=bytearray(storage.nbytes + 1), offset=1
        )
        array[:] = storage
        storage = array
    if layout == "F":
        storage = np.asfortranarray(storage)
        mask = np.asfortranarray(mask)
    array = np.ma.array(storage, mask=mask, copy=False)
    if layout == "reverse":
        array = array[::-1]
    elif layout == "strided":
        array = np.ma.repeat(array, 2, axis=0)[::2]
    elif layout == "readonly":
        array.flags.writeable = False
    return array


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_masked_times_reach_native_missing_value_validation(dtype, layout):
    time = masked_array([1, 2, 3, 4], dtype, layout)
    before_data, before_mask = time.data.copy(), time.mask.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="time contains NaN"):
            kernels.survfitkm(time, [1, 1, 0, 1])
    assert not caught
    np.testing.assert_array_equal(time.data, before_data)
    np.testing.assert_array_equal(time.mask, before_mask)


@pytest.mark.parametrize("dtype", [*DTYPES, np.int64])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_integer_status_and_boolean_flags_cannot_discard_masks(dtype, layout):
    status = masked_array([1, 0, 1, 0], dtype, layout)
    with pytest.raises(TypeError, match="not an int32"):
        kernels.survfitkm([1, 2, 3, 4], status)
    with pytest.raises(TypeError, match="not an int32"):
        kernels.finegray(
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 0.8, 0.6, 0.4],
            status,
            [True] * 4,
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_native_design_matrices_and_rows_cannot_discard_masks(dtype, layout):
    design = masked_array([[1, 0], [1, 1], [1, 0], [1, 1]], dtype, layout)
    with pytest.raises(ValueError, match="x contains non-finite value NaN"):
        kernels.coxph_fit([1, 2, 3, 4], [1, 0, 1, 1], design)
    with pytest.raises(ValueError, match="cmat contains non-finite value NaN"):
        kernels.yates(design, [1, 2], [[1, 0], [0, 1]])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("field", ["surv", "pstate"])
def test_aggregation_propagates_missing_matrix_and_cube_values(dtype, layout, field):
    values = [[1, 0], [0, 1], [1, 1]]
    if field == "pstate":
        values = [[[value, 1 - value] for value in row] for row in values]
    array = masked_array(values, dtype, layout)
    before_data, before_mask = array.data.copy(), array.mask.copy()
    reference = array.astype(float).filled(np.nan)
    actual = getattr(kernels.aggregate_survfit(**{field: array}), field)
    expected = getattr(kernels.aggregate_survfit(**{field: reference}), field)
    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(actual).any()
    np.testing.assert_array_equal(array.data, before_data)
    np.testing.assert_array_equal(array.mask, before_mask)


@pytest.mark.parametrize("dtype", [object, "U7"])
def test_masked_backing_objects_are_not_converted_to_numbers(dtype):
    array = np.ma.array([["1", "ignored"], ["0", "1"]], dtype=dtype, mask=[[0, 1], [0, 0]])
    actual = kernels.aggregate_survfit(surv=array).surv
    expected = kernels.aggregate_survfit(surv=[[1.0, np.nan], [0.0, 1.0]]).surv
    np.testing.assert_array_equal(actual, expected)
    assert array.data[0, 1] == "ignored"


@pytest.mark.parametrize("sequence", [list, tuple])
def test_nested_sequences_of_masked_rows_preserve_missing_values(sequence):
    matrix = np.ma.array([[1.0, 42.0], [0.0, 1.0]], mask=[[0, 1], [0, 0]])
    rows = sequence(matrix)
    cube = sequence([rows])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")
        with pytest.raises(ValueError, match="x contains non-finite value NaN"):
            kernels.coxph_fit([1, 2], [1, 1], rows)
        with pytest.raises(ValueError, match="cmat contains non-finite value NaN"):
            kernels.yates(rows, [1, 2], [[1, 0], [0, 1]])
        for field, values, reference in [
            ("surv", rows, matrix.filled(np.nan)),
            ("pstate", cube, matrix.filled(np.nan)[None]),
        ]:
            actual = getattr(kernels.aggregate_survfit(**{field: values}), field)
            expected = getattr(kernels.aggregate_survfit(**{field: reference}), field)
            np.testing.assert_array_equal(actual, expected)
            assert np.isnan(actual).any()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask", [np.ma.nomask, False])
def test_complete_masked_arrays_match_plain_arrays(dtype, mask):
    time = np.ma.array([1, 2, 3, 4], dtype=dtype, mask=mask)
    status = np.ma.array([1, 0, 1, 0], dtype=dtype, mask=mask)
    actual = kernels.survfitkm(time, status)
    expected = kernels.survfitkm(time.data, status.data)
    np.testing.assert_array_equal(actual.surv, expected.surv)
    np.testing.assert_array_equal(actual.time, expected.time)
    matrix = np.ma.array([[1, 0], [0, 1]], dtype=dtype, mask=mask)
    for field, values in [("surv", matrix), ("pstate", matrix.reshape(1, 2, 2))]:
        actual = getattr(kernels.aggregate_survfit(**{field: values}), field)
        expected = getattr(kernels.aggregate_survfit(**{field: values.data}), field)
        np.testing.assert_array_equal(actual, expected)
