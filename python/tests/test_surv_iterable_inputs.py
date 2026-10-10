"""Surv/Surv2 one-shot inputs preserve stock-R rows, codes and factor metadata."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/surv_iterable_reference.json").read_text()
)


class CountedIterator:
    def __init__(self, values, levels=None):
        self.values = iter(values)
        self.consumed = []
        if levels is not None:
            self.categories = tuple(levels)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.consumed.append(value)
        return value


def numbers(spec):
    return [
        float(value)
        if spec["kind"] not in {"character", "factor"} and isinstance(value, str)
        else value
        for value in spec["values"]
    ]


def input_column(spec, container):
    values = numbers(spec)
    levels = spec["levels"]
    if container == "iterator":
        return CountedIterator(values, levels)
    if container == "generator":
        generator = (value for value in values)
        return generator if levels is None else CountedIterator(generator, levels)
    if levels is not None:
        return (
            pd.Categorical(values, categories=levels)
            if container == "pandas"
            else RFactor(values, levels)
        )
    if container == "array":
        dtype = (
            bool
            if spec["kind"] == "logical" and None not in values
            else object
            if spec["kind"] in {"logical", "character"}
            else float
        )
        return np.asarray(values, dtype=dtype)
    if container == "masked":
        if spec["kind"] == "character":
            return list(values)
        mask = [value is None for value in values]
        return np.ma.array(
            [0 if absent else value for value, absent in zip(values, mask, strict=True)],
            mask=mask,
            dtype=bool if spec["kind"] == "logical" else float,
        )
    if container == "pandas":
        dtype = (
            "boolean"
            if spec["kind"] == "logical"
            else "string"
            if spec["kind"] == "character"
            else "Float64"
        )
        return pd.Series(values, dtype=dtype)
    return list(values)


def assert_response(actual, expected):
    assert getattr(actual, "type", None) == expected["type"]
    assert list(actual.states) == expected["states"]
    assert actual.clabel == expected["clabel"]
    assert getattr(actual, "repeated", None) == expected["repeated"]
    assert len(actual.status) == len(actual)
    ncol = expected["ncol"]
    np.testing.assert_equal(
        np.asarray(actual.as_matrix(), dtype=float).reshape(-1, ncol),
        np.asarray(expected["matrix"], dtype=float).reshape(-1, ncol),
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize(
    "container", ["list", "iterator", "generator", "array", "masked", "pandas"]
)
def test_survival_inputs_match_independent_stock_r(case, container):
    inputs = [input_column(spec, container) for spec in case["arguments"]]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if "error" in case:
            with pytest.raises(ValueError, match=case["error"]):
                getattr(r, case["constructor"])(*inputs, **(case["options"] or {}))
        else:
            actual = getattr(r, case["constructor"])(*inputs, **(case["options"] or {}))
            assert_response(actual, case["response"])
    assert [str(warning.message) for warning in captured] == case["warnings"]
    for index, (spec, values) in enumerate(zip(case["arguments"], inputs, strict=True)):
        if isinstance(values, CountedIterator):
            assert len(values.consumed) == (0 if index in case["unused"] else len(spec["values"]))


@pytest.mark.parametrize("name", ["numeric_mright", "numeric_mcounting", "numeric_timeline"])
def test_declared_numeric_factor_array_retains_level_order_and_unused_states(name):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    event = case["arguments"][-1]

    class FactorArray(np.ndarray):
        pass

    events = np.asarray(
        [np.nan if value is None else float(value) for value in event["values"]]
    ).view(FactorArray)
    events.categories = [float(level) for level in event["levels"]]
    actual = getattr(r, case["constructor"])(
        *(numbers(spec) for spec in case["arguments"][:-1]),
        events,
        **(case["options"] or {}),
    )
    assert_response(actual, case["response"])
    assert events.categories == [30.0, 10.0, 2.0, 99.0]


def test_constructor_inputs_are_unmodified_and_outputs_own_normalized_values():
    times = np.array([1.0, 2.0, 3.0])
    events = RFactor(["event", "censor", None], ["censor", "unused", "event"])
    actual = r.Surv(times, events)
    np.testing.assert_array_equal(times, [1, 2, 3])
    assert events == ["event", "censor", None]
    times[0] = 99
    events[0] = "censor"
    events.categories = ("censor", "event")
    assert actual.time == (1.0, 2.0, 3.0)
    assert actual.event == (2, 0, None)
    assert actual.states == ("unused", "event")


@pytest.mark.parametrize("name", ["mright", "timeline_factor"])
def test_missing_declared_factor_levels_are_excluded_before_status_coding(name):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    event = case["arguments"][1]
    inputs = RFactor(numbers(event), [None, *event["levels"], np.nan])
    actual = getattr(r, case["constructor"])(
        numbers(case["arguments"][0]), inputs, **(case["options"] or {})
    )
    assert_response(actual, case["response"])
    assert len(inputs.categories) == len(event["levels"]) + 2


@pytest.mark.parametrize("container", ["iterator", "generator"])
def test_one_shot_response_remains_valid_in_a_stock_r_kaplan_meier_fit(container):
    reference = REFERENCE["kaplan_meier"]
    if container == "iterator":
        times = CountedIterator(reference["time"])
        events = CountedIterator(reference["event"])
    else:
        times = (value for value in reference["time"])
        events = (value for value in reference["event"])
    fit = r.survfit(r.Surv(times, events), conf_type="none")
    for field, expected in reference["result"].items():
        np.testing.assert_allclose(getattr(fit, field), expected, rtol=1e-14, atol=0)
    if container == "iterator":
        assert times.consumed == reference["time"]
        assert events.consumed == reference["event"]
