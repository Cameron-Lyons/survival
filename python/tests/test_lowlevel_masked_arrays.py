"""Prepared numeric interfaces must never fit data hidden by a NumPy mask."""

import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r


def masked_input(values, layout, position=(0, 0)):
    array = np.ma.array(values, mask=False)
    array.mask[position] = True
    if layout == "mapping":
        prepared = {f"x{i}": array[:, i] for i in range(array.shape[1])}
    elif layout == "rows":
        prepared = list(array)
    elif layout == "readonly":
        array.flags.writeable = False
        prepared = array
    else:
        prepared = array
    return array, prepared


@pytest.mark.parametrize("layout", ["array", "mapping", "rows", "readonly"])
@pytest.mark.parametrize("dtype", [float, int, bool])
@pytest.mark.parametrize("fitter", ["coxph_fit", "agreg_fit", "agexact_fit", "survreg_fit"])
def test_fitting_rejects_masked_design_without_modifying_input(layout, dtype, fitter):
    original, design = masked_input(np.asarray([[0], [1], [0], [1]], dtype=dtype), layout)
    before_data, before_mask = original.data.copy(), original.mask.copy()
    if fitter == "survreg_fit":
        response = [[1, 1], [2, 0], [3, 1], [4, 1]]
    elif fitter == "agreg_fit":
        response = r.Surv([0, 0, 0, 0], [1, 2, 3, 4], [1, 0, 1, 1])
    else:
        response = r.Surv([1, 2, 3, 4], [1, 0, 1, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite"):
            getattr(r, fitter)(design, response)
    assert not caught
    np.testing.assert_array_equal(original.data, before_data)
    np.testing.assert_array_equal(original.mask, before_mask)


@pytest.mark.parametrize("layout", ["array", "rows", "readonly"])
@pytest.mark.parametrize("column", [0, 1])
def test_aft_response_preserves_masked_time_and_event(layout, column):
    _, response = masked_input([[1, 1], [2, 0], [3, 1], [4, 1]], layout, (0, column))
    with pytest.raises(ValueError, match="finite|status"):
        r.survreg_fit([[1]] * 4, response)


def cox_curve_inputs():
    return {
        "y": [[1, 1], [2, 1], [3, 0]],
        "x": [[0.0], [1.0], [2.0]],
        "risk": [1.0] * 3,
        "wt": [1.0] * 3,
        "x2": [[0.5], [1.5]],
        "risk2": [1.0, 2.0],
        "varmat": [[0.2]],
    }


@pytest.mark.parametrize("name", ["y", "x", "risk", "wt", "x2", "risk2", "varmat"])
def test_cox_curves_reject_masked_numeric_inputs(name):
    arguments = cox_curve_inputs()
    array = np.ma.array(arguments[name], mask=False)
    array.mask.flat[0] = True
    arguments[name] = array
    with pytest.raises(ValueError, match="finite"):
        r.coxsurv_fit(**arguments)


@pytest.mark.parametrize("dtype", [object, "U4"])
def test_masked_covariance_preserves_missingness_when_converted_to_float(dtype):
    arguments = cox_curve_inputs()
    covariance = np.ma.array([[0.2]], dtype=dtype, mask=False)
    expected = r.coxsurv_fit(**arguments)
    actual = r.coxsurv_fit(**{**arguments, "varmat": covariance})
    np.testing.assert_array_equal(actual.std_err, expected.std_err)
    covariance.mask[0, 0] = True
    before_data, before_mask = covariance.data.copy(), covariance.mask.copy()
    for values in (covariance, [[np.nan]]):
        with pytest.raises(ValueError, match="varmat must be finite"):
            r.coxsurv_fit(**{**arguments, "varmat": values})
    np.testing.assert_array_equal(covariance.data, before_data)
    np.testing.assert_array_equal(covariance.mask, before_mask)


@pytest.mark.parametrize("layout", ["mapping", "rows", "readonly"])
@pytest.mark.parametrize("name", ["x", "x2"])
def test_cox_curve_design_layouts_preserve_masks(name, layout):
    arguments = cox_curve_inputs()
    _, arguments[name] = masked_input(arguments[name], layout)
    with pytest.raises(ValueError, match="finite"):
        r.coxsurv_fit(**arguments)


@pytest.mark.parametrize("column", [0, 1])
def test_cox_curve_individual_response_preserves_masks(column):
    arguments = cox_curve_inputs()
    _, arguments["y2"] = masked_input([[0.0, 1.5], [1.5, 3.0]], "array", (0, column))
    arguments["id2"] = [1, 1]
    with pytest.raises(ValueError, match="finite"):
        r.coxsurv_fit(**arguments)


def test_unmasked_arrays_keep_curve_values_and_input_storage():
    arguments = cox_curve_inputs()
    expected = r.coxsurv_fit(**arguments)
    for mask in (np.ma.nomask, False):
        arrays = {name: np.ma.array(values, mask=mask) for name, values in arguments.items()}
        actual = r.coxsurv_fit(**arrays)
        for name in ("surv", "cumhaz", "std_err", "n_risk", "n_event"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        for name, array in arrays.items():
            np.testing.assert_array_equal(array.data, arguments[name])
            assert not np.ma.getmaskarray(array).any()


@pytest.mark.parametrize("mask", [None, np.ma.nomask, False])
def test_complete_design_arrays_keep_shared_storage(mask):
    storage = np.arange(12, dtype=float).reshape(4, 3)
    values = storage if mask is None else np.ma.array(storage, mask=mask)
    values.flags.writeable = False
    prepared = r._coerce._numeric_design_matrix(values, 4)
    assert np.shares_memory(prepared, storage)
    assert not prepared.flags.writeable


@pytest.mark.parametrize("layout", ["array", "mapping", "rows"])
def test_masked_strings_do_not_become_numeric_designs(layout):
    _, design = masked_input([["0"], ["1"], ["0"], ["1"]], layout)
    with pytest.raises(TypeError, match="numeric"):
        r.coxph_fit(design, r.Surv([1, 2, 3, 4], [1, 0, 1, 1]))
