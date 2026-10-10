"""Masked response rows obey independently recorded stock model-frame missingness."""

import importlib
import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
formula = importlib.import_module("survival.r._formula")
coerce = importlib.import_module("survival.r._coerce")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/masked_matrix_formula_reference.json").read_text()
)


class CountedRows(Iterator):
    def __init__(self, values):
        self.values = list(values)
        self.seen = []

    def __next__(self):
        if len(self.seen) == len(self.values):
            raise StopIteration
        value = self.values[len(self.seen)]
        self.seen.append(value)
        return value


class UnusedRows(Iterator):
    def __next__(self):
        raise AssertionError("unused response source was read")


def assert_values(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for name, value in expected.items():
            assert_values(actual[name], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_values(value, reference)
    elif expected is None:
        assert actual is None or coerce._is_missing_value(actual)
    elif isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def record(value):
    return None if value is None else {"rows": list(value.rows), "kind": value.kind}


def response(case, layout):
    rows = REFERENCE["inputs"][case["input"]]["Y"]
    mask = np.asarray([[cell is None for cell in row] for row in rows])
    storage = np.asarray([[987654.0 if cell is None else cell for cell in row] for row in rows])
    masked = np.ma.array(storage, mask=mask)
    counted = None
    if layout == "nullable_lists":
        source = [list(row) for row in rows]
    elif layout == "numeric_array":
        source = np.asarray(rows, dtype=float)
    elif layout == "masked":
        source = masked
    elif layout == "counted_rows":
        source = counted = CountedRows(masked)
    elif layout == "generator_rows":
        source = (row for row in masked)
    elif layout == "mixed_rows":
        source = [list(row) if i % 2 == 0 else masked[i] for i, row in enumerate(rows)]
    else:
        source = [
            [np.ma.array(987654.0, mask=True) if cell is None else cell for cell in row]
            for row in rows
        ]
    data = {"unused": UnusedRows(), "Y": source}
    data.update(
        {
            name: list(values)
            for name, values in REFERENCE["inputs"][case["input"]].items()
            if name != "Y"
        }
    )
    return data, counted, masked


def kwargs(case):
    return {"weights": "wt", "subset": case["subset"], "na_action": case["na_action"]}


LAYOUTS = [
    "nullable_lists",
    "numeric_array",
    "masked",
    "counted_rows",
    "generator_rows",
    "mixed_rows",
    "mixed_scalar_cells",
]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("method", ["frame", "pyears"])
def test_masked_response_rows_match_independent_stock(case, layout, method):
    data, counted, masked = response(case, layout)
    before_data, before_mask = masked.data.copy(), np.ma.getmaskarray(masked).copy()
    expected = case[method]
    assert expected["warnings"] == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected["value"]:
            # Stock pyears na.pass reaches its kernel with missing responses;
            # the port rejects that invalid kernel input explicitly.
            pattern = "missing" if case["na_action"] == "fail" else "non-finite|finite|NaN"
            if method == "frame":
                with pytest.raises(ValueError, match=pattern):
                    r.model_frame(case["formula"], data, **kwargs(case))
            else:
                with pytest.raises(ValueError, match=pattern):
                    r.pyears(case["formula"], data, scale=1, model=True, **kwargs(case))
        elif method == "frame":
            actual = r.model_frame(case["formula"], data, **kwargs(case))
            assert_values(actual, expected["value"]["columns"])
            # A fresh source lets the internal frame expose the omitted row
            # action and labels independently of the public raw-column schema.
            prepared, _, _ = response(case, layout)
            frame = formula.model_frame(case["formula"], prepared, **kwargs(case))
            assert record(frame.na_action) == expected["value"]["na_action"]
            if case["na_action"] == "pass":
                expected_mask = np.asarray(
                    [[value is None for value in row] for row in expected["value"]["columns"]["Y"]]
                )
                payload = frame.y.view(np.uint64)
                na_payload = np.asarray(coerce._NA_REAL).view(np.uint64)
                np.testing.assert_array_equal(np.isnan(frame.y), expected_mask)
                # Plain arrays supply genuine numerical NaN; masks and nullable
                # cells supply R NA. Both have the same stock omission rows.
                np.testing.assert_array_equal(
                    payload == na_payload,
                    expected_mask if layout != "numeric_array" else np.zeros_like(expected_mask),
                )
            labels = formula._data_row_labels(frame.data, frame.n)
            actual_labels = (
                list(labels) if labels is not None else [str(i + 1) for i in range(frame.n)]
            )
            assert actual_labels == expected["value"]["row_names"]
        else:
            fit = r.pyears(
                case["formula"], data, scale=1, x="group" in case["formula"], y=True, **kwargs(case)
            )
            value = expected["value"]
            for name in ("pyears", "event", "n"):
                assert_values(np.asarray(getattr(fit, name)).ravel(order="F").tolist(), value[name])
            for name in ("offtable", "observations"):
                assert_values(getattr(fit, name), value[name])
            assert list(fit.dim) == (value["dim"] or [])
            assert fit.dimnames == (value["dimnames"] or {})
            assert_values(fit.y, value["y"])
            assert_values(fit.x, value["x"])
            assert record(fit.na_action) == value["na_action"]
            retained, _, _ = response(case, layout)
            stored = r.pyears(case["formula"], retained, scale=1, model=True, **kwargs(case))
            assert_values(stored.model, value["model"]["columns"])
            assert_values(r.model_frame(stored), value["model"]["columns"])
            # Both public frame and retained model are independent writable
            # output snapshots, including after masking or row selection.
            stored.model["Y"][0][0] = -12345
            fit.y[0][0] = -12345
    assert caught == []
    if counted is not None:
        assert len(counted.seen) == len(counted.values)
    np.testing.assert_array_equal(masked.data, before_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(masked), before_mask)


@pytest.mark.parametrize("case", REFERENCE["vector_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["masked", "iterator", "generator", "mixed_list", "mixed_0d"])
def test_masked_scalar_responses_follow_stock_na_vector(case, layout):
    missing = case["input"] == "missing"
    values = np.ma.array([2.0, 5.0, 3.0], mask=[False, missing, False])
    if layout == "masked":
        source = values
    elif layout == "iterator":
        source = CountedRows(values)
    elif layout == "generator":
        source = (item for item in values)
    elif layout == "mixed_list":
        source = [2.0, np.ma.masked if missing else 5.0, 3.0]
    else:
        source = [2.0, np.ma.array(5.0, mask=missing), 3.0]
    expected = case["frame"]
    assert expected["warnings"] == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected["value"]:
            with pytest.raises(ValueError, match="missing"):
                r.model_frame(
                    "Y~1", {"Y": source}, weights=[1, 0.5, 2], na_action=case["na_action"]
                )
        else:
            frame = r.model_frame(
                "Y~1", {"Y": source}, weights=[1, 0.5, 2], na_action=case["na_action"]
            )
            assert_values(frame, expected["value"]["columns"])
    assert caught == []
    np.testing.assert_array_equal(values.data, [2, 5, 3])
    np.testing.assert_array_equal(np.ma.getmaskarray(values), [False, missing, False])


def test_specific_mask_fallback_preserves_unrelated_float_warnings():
    class WarnsFloat:
        def __float__(self):
            warnings.warn("independent numeric conversion warning", UserWarning, stacklevel=2)
            return 2.0

    with pytest.warns(UserWarning, match="independent numeric conversion warning"):
        assert coerce._floats_or_nan([WarnsFloat()]) == [2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="independent numeric conversion warning"):
            coerce._floats_or_nan([WarnsFloat()])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_zero_dimensional_masked_scalars_keep_missingness_and_numeric_value(masked, dtype):
    value = np.ma.array(1, dtype=dtype, mask=masked)
    assert coerce._is_missing_value(value) is masked
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = coerce._floats_or_nan([2.0, value, 3.0])
    assert caught == []
    assert actual[0] == 2.0
    assert actual[2] == 3.0
    if masked:
        assert np.asarray(actual[1]).view(np.uint64) == np.asarray(coerce._NA_REAL).view(np.uint64)
    else:
        assert actual[1] == 1.0


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shape", [(2,), (1, 2)])
@pytest.mark.parametrize("action", ["pass", "fail", "omit", "exclude"])
def test_nonscalar_masked_cells_follow_equivalent_nullable_shape(masked, shape, action):
    data = np.full(shape, 5.0)
    mask = np.zeros(shape, dtype=bool)
    mask.flat[0] = masked
    value = np.ma.array(data, mask=mask)
    for cell in (value, value.tolist()):
        source = {"y": [2.0, cell, 3.0]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if masked and action in {"omit", "exclude"}:
                assert r.model_frame("y~1", source, na_action=action) == {"y": [2.0, 3.0]}
            else:
                match = "missing" if masked and action == "fail" else "must be numeric"
                with pytest.raises(ValueError, match=match):
                    r.model_frame("y~1", source, na_action=action)
        assert caught == []


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("action", ["pass", "fail", "omit", "exclude"])
@pytest.mark.parametrize("iterator", [False, True])
def test_single_element_masked_arrays_retain_numpy_scalar_conversion(
    masked, shape, action, iterator
):
    value = np.ma.array(np.full(shape, 5.0), mask=masked)
    rows = [2.0, value, 3.0]
    source = iter(rows) if iterator else rows
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if masked and action == "fail":
            with pytest.raises(ValueError, match="missing"):
                r.model_frame("y~1", {"y": source}, na_action=action)
        else:
            frame = r.model_frame("y~1", {"y": source}, na_action=action)
            expected = (
                [2.0, 3.0]
                if masked and action in {"omit", "exclude"}
                else [
                    2.0,
                    None if masked else 5.0,
                    3.0,
                ]
            )
            for actual, reference in zip(frame["y"], expected, strict=True):
                if reference is None:
                    assert coerce._is_missing_value(actual)
                else:
                    assert float(actual) == reference
    assert caught == []
