"""Full frailty matrices against independent stock R, including iterator ownership."""

import copy
import json
import pickle
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/model_matrix_iterator_reference.json").read_text()
)


class ColumnIterator:
    def __init__(self, values, spec, *, generator=False):
        self.source = (value for value in values) if generator else iter(values)
        self.consumed = []
        if spec["levels"] is not None:
            self.categories = tuple(spec["levels"])
            self.ordered = spec["ordered"]

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.source)
        self.consumed.append(value)
        return value


class UnusedIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("unused model-matrix column was consumed")


class FactorArray(np.ndarray):
    def __new__(cls, values, spec):
        result = np.asarray(values, dtype=object).view(cls)
        result.categories = tuple(spec["levels"])
        result.ordered = spec["ordered"]
        return result

    def __array_finalize__(self, source):
        if source is not None:
            self.categories = getattr(source, "categories", ())
            self.ordered = getattr(source, "ordered", False)


def columns(specs, layout, *, shared=False):
    result = {"unused": UnusedIterator()}
    for name, spec in specs.items():
        values = copy.deepcopy(spec["values"])
        if layout in {"iterator", "generator"}:
            result[name] = ColumnIterator(values, spec, generator=layout == "generator")
        elif layout == "array":
            result[name] = (
                FactorArray(values, spec)
                if spec["levels"] is not None
                else np.asarray(values, dtype=float)
            )
        elif layout == "pandas":
            result[name] = (
                pd.Categorical(values, categories=spec["levels"], ordered=spec["ordered"])
                if spec["levels"] is not None
                else pd.Series(values, dtype="Float64")
            )
        else:
            result[name] = RFactor(values, spec["levels"]) if spec["levels"] is not None else values
    if shared:
        result["age"] = result["group"]
    return result


@cache
def fitted(name, retained):
    spec = REFERENCE["fits"][name]
    return r.coxph(
        spec["formula"],
        columns(REFERENCE["inputs"][spec["source"]], "list"),
        model=retained,
        x=False,
    )


def compact(value):
    return value.replace(" ", "")


def check_matrix(actual, expected):
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["data"], expected["data"], rtol=1e-12, atol=1e-14)
    assert [compact(name) for name in actual["columns"]] == [
        compact(name) for name in expected["columns"]
    ]
    for name in ("assign", "row_names", "strata"):
        assert actual[name] == expected[name]
    if expected["contrasts"] is None:
        assert actual["contrasts"] is None
    else:
        actual_contrasts = {compact(name): value for name, value in actual["contrasts"].items()}
        expected_contrasts = {compact(name): value for name, value in expected["contrasts"].items()}
        assert actual_contrasts == expected_contrasts


CASES = [case for case in REFERENCE["cases"] if case["row_names"] is None]
NAMED_CASES = [case for case in REFERENCE["cases"] if case["row_names"] is not None]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array", "pandas", "iterator", "generator"])
@pytest.mark.parametrize("retained", [False, True])
def test_frailty_newdata_matrix_and_metadata_match_stock(case, layout, retained):
    fit = fitted(case["fit"], retained)
    stored = r.model_matrix(fit, _with_metadata=True)
    check_matrix(stored, REFERENCE["fits"][case["fit"]]["stored"]["value"])
    data = columns(case["newdata"], layout, shared=case["shared_age_group"])
    snapshots = {
        name: copy.deepcopy(value)
        for name, value in data.items()
        if name != "unused" and not isinstance(value, ColumnIterator)
    }
    expected = case["expected"]["value"]
    if "error" in expected:
        with pytest.raises(ValueError, match=expected["error"]):
            r.model_matrix(fit, data, _with_metadata=True)
    else:
        actual = r.model_matrix(fit, data, _with_metadata=True)
        check_matrix(actual, expected)
    for name, value in data.items():
        if isinstance(value, ColumnIterator):
            assert value.consumed == case["newdata"][name]["values"] or (
                "error" in expected and not value.consumed
            )
        elif name != "unused":
            if isinstance(value, pd.Categorical):
                pd.testing.assert_extension_array_equal(value, snapshots[name])
            elif isinstance(value, pd.Series):
                pd.testing.assert_series_equal(value, snapshots[name])
            else:
                np.testing.assert_array_equal(value, snapshots[name])
            if case["newdata"][name]["levels"] is not None:
                assert tuple(value.categories) == tuple(case["newdata"][name]["levels"])
    # Sparse matrix insertion and local recoding must not accumulate across calls.
    for _ in range(2):
        assert r.model_matrix(fit, _with_metadata=True) == stored
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own fit
    assert r.model_matrix(fit, _with_metadata=True) == r.model_matrix(restored, _with_metadata=True)


@pytest.mark.parametrize("case", NAMED_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("retained", [False, True])
def test_explicit_prediction_frame_labels_survive_missing_row_omission(case, retained):
    data = columns(case["newdata"], "pandas")
    data.pop("unused")
    frame = pd.DataFrame(data)
    frame.index = case["row_names"]
    original = frame.copy(deep=True)
    check_matrix(
        r.model_matrix(fitted(case["fit"], retained), frame, _with_metadata=True),
        case["expected"]["value"],
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("name", REFERENCE["fits"])
@pytest.mark.parametrize("layout", ["iterator", "generator"])
def test_unretained_iterator_fit_matrix_rebuild_and_pickle_match_stock(name, layout):
    spec = REFERENCE["fits"][name]
    data = columns(REFERENCE["inputs"][spec["source"]], layout)
    fit = r.coxph(spec["formula"], data, model=False, x=False)
    for candidate in (fit, pickle.loads(pickle.dumps(fit))):  # noqa: S301 - own fit
        check_matrix(r.model_matrix(candidate, _with_metadata=True), spec["stored"]["value"])
    for column, value in data.items():
        if isinstance(value, ColumnIterator):
            assert value.consumed == (
                [] if column == "id" else REFERENCE["inputs"][spec["source"]][column]["values"]
            )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case["fit"].startswith("numeric_")],
    ids=lambda case: case["name"],
)
def test_plain_generator_columns_and_shared_alias_are_read_once(case):
    consumed = {"age": [], "group": []}

    def values(name):
        for value in case["newdata"][name]["values"]:
            consumed[name].append(value)
            yield value

    data = {"unused": UnusedIterator(), "age": values("age"), "group": values("group")}
    if case["shared_age_group"]:
        data["age"] = data["group"]
    check_matrix(
        r.model_matrix(fitted(case["fit"], False), data, _with_metadata=True),
        case["expected"]["value"],
    )
    assert consumed["group"] == case["newdata"]["group"]["values"]
    assert consumed["age"] == ([] if case["shared_age_group"] else case["newdata"]["age"]["values"])
