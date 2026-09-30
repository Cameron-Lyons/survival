"""Native predictions provide owned numeric buffers with explicit matrix widths."""

import gc
import json
from pathlib import Path

import numpy as np
import pytest

from .r_fixture_support import RFactor
from .test_prediction_row_labels import REFERENCE, _r_data_frame, fitted, r

EMPTY_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/empty_aft_prediction_reference.json").read_text()
)


def check_snapshots(make_result, shape=None):
    result = make_result()
    expected = np.asarray(result.fit, dtype=float)
    if hasattr(result, "n_columns"):
        expected = expected.reshape(len(result.fit), result.n_columns)
    first, second = result.to_arrays(), result.to_arrays()
    if shape is not None:
        assert first["fit"].shape == shape
        assert result.n_columns == shape[1]
        if hasattr(result, "predict_type"):
            assert f"columns={shape[1]}" in repr(result)
    assert isinstance(result.fit, list)
    for name in ("fit", "se_fit"):
        native = getattr(result, name)
        if native is None:
            assert first[name] is second[name] is None
            continue
        values = expected if name == "fit" else np.asarray(native).reshape(expected.shape)
        assert first[name].dtype == np.dtype("float64")
        assert first[name].flags.writeable
        np.testing.assert_array_equal(first[name], values)
        assert not np.shares_memory(first[name], second[name])
        if values.size:
            first[name].flat[0] = 12345
        np.testing.assert_array_equal(second[name], values)
        np.testing.assert_array_equal(
            np.asarray(getattr(result, name)).reshape(values.shape), values
        )
    if hasattr(result, "constant"):
        assert first["constant"] == result.constant
    del result
    gc.collect()
    np.testing.assert_array_equal(second["fit"], expected)


@pytest.mark.parametrize("model", ["cox", "cox_ridge"])
@pytest.mark.parametrize("kind", ["lp", "risk", "expected", "survival", "terms"])
@pytest.mark.parametrize("errors", [False, True])
def test_native_cox_snapshots_own_values_and_preserve_list_properties(model, kind, errors):
    fit = fitted(model, "na.omit").fit
    check_snapshots(
        lambda: (
            fit.predict_terms(se_fit=errors, assign=[[1], [0], [1]])
            if kind == "terms"
            else fit.predict(kind, se_fit=errors)
        )
    )


@pytest.mark.parametrize("model", ["aft", "aft_ridge", "aft_strata", "aft_fixed"])
@pytest.mark.parametrize("kind", ["response", "link", "terms", "quantile", "uquantile"])
@pytest.mark.parametrize("errors", [False, True])
def test_native_aft_snapshots_own_values_and_preserve_list_properties(model, kind, errors):
    model = fitted(model, "na.omit")
    check_snapshots(
        lambda: model.fit.predict(
            predict_type=kind, se_fit=errors, p=[0.1, 0.5, 0.9], assign=list(model.assign)
        )
    )


@pytest.mark.parametrize("model", ["cox", "cox_ridge", "aft", "aft_ridge"])
@pytest.mark.parametrize("errors", [False, True])
def test_prediction_snapshots_preserve_zero_columns(model, errors):
    cox = model.startswith("cox")
    model = fitted(model, "na.omit")
    check_snapshots(
        lambda: (
            model.fit.predict_terms(assign=[], se_fit=errors)
            if cox
            else model.fit.predict(
                predict_type="terms", assign=list(model.assign), terms=[], se_fit=errors
            )
        ),
        shape=(model.fit.n, 0),
    )


@pytest.mark.parametrize(("kind", "width"), [("response", 1), ("terms", 2), ("quantile", 3)])
@pytest.mark.parametrize("errors", [False, True])
def test_native_aft_empty_rows_retain_their_column_count(kind, width, errors):
    model = fitted("aft", "na.omit")
    check_snapshots(
        lambda: model.fit.predict(
            np.empty((0, len(model.coefficients))),
            predict_type=kind,
            assign=list(model.assign),
            p=[0.1, 0.5, 0.9],
            se_fit=errors,
        ),
        shape=(0, width),
    )


@pytest.mark.parametrize("case", EMPTY_REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("as_arrays", [False, True], ids=["lists", "arrays"])
def test_entirely_omitted_aft_predictions_match_independent_r(case, as_arrays):
    data = {**REFERENCE["newdata"], "age": [None] * 4}
    data["cl"] = RFactor(data["cl"], REFERENCE["levels"])
    result = r.predict(
        fitted(case["model"], "na.omit"),
        _r_data_frame(data, 4, REFERENCE["new_names"]),
        type=case["type"],
        p=case["p"],
        se_fit=case["se_fit"],
        na_action=case["na_action"],
        _with_row_names=True,
        _as_arrays=as_arrays,
    )
    value = result["values"]
    pairs = (
        [(value.fit, case["result"]["fit"]), (value.se_fit, case["result"]["se.fit"])]
        if case["se_fit"]
        else [(value, case["result"])]
    )
    for actual, reference in pairs:
        shape = reference["dim"] or [len(reference["values"])]
        expected = np.asarray(reference["values"], dtype=float).reshape(shape)
        actual = np.asarray(actual)
        if actual.size == 0 and not as_arrays:
            actual = actual.reshape(shape)
        np.testing.assert_array_equal(actual, expected)
    assert result["fit_names"] is result["se_names"] is None
    reference = case["result"]["fit"] if case["se_fit"] else case["result"]
    assert result["null_dimnames"] == reference["has_dimnames"]
