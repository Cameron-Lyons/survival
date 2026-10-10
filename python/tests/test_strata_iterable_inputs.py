"""One-shot strata inputs and reusable factor results against independent stock R."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r_api
factor_codes = importlib.import_module("survival.r._coerce")._factor
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/strata_iterable_reference.json").read_text()
)


class FactorIterator:
    def __init__(self, values, levels):
        self._values = iter(values)
        self.categories = tuple(levels)
        self.observed = []

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self._values)
        self.observed.append(value)
        return value


def column(spec, container):
    values = spec["values"]
    levels = spec["levels"]
    if container == "list":
        return list(values) if levels is None else RFactor(values, levels)
    if levels is not None:
        return FactorIterator(values, levels)
    if container == "iterator":
        return iter(values)
    return (value for value in values)


def grouping(case, container):
    columns = {name: column(spec, container) for name, spec in case["columns"].items()}
    options = case["options"]
    if case["named"]:
        return r.strata(columns, **options)
    if container == "nested":
        return r.strata(list(columns.values()), labels=list(columns), **options)
    if container == "outer_iterator":
        return r.strata(iter(columns.values()), labels=list(columns), **options)
    return r.strata(*columns.values(), labels=list(columns), **options)


def assert_factor(actual, expected):
    for field in ("codes", "levels", "labels", "counts"):
        assert getattr(actual, field) == expected[field]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", ["list", "iterator", "generator", "nested", "outer_iterator"])
def test_strata_iterable_columns_match_stock_r(case, container):
    actual = grouping(case, container)
    assert_factor(actual, case["expected"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_strata_factor_roundtrip_retains_factor_metadata(case):
    actual = grouping(case, "iterator")
    assert_factor(r.strata(actual), case["roundtrip"])
    for counting in (False, True):
        times = list(range(1, len(actual) + 1))
        response = r.Surv([0] * len(actual), times, actual) if counting else r.Surv(times, actual)
        expected = case["counting_response" if counting else "right_response"]
        assert response.type == expected["type"]
        assert list(response.states) == expected["states"]
        assert response.clabel == expected["clabel"]
        np.testing.assert_equal(
            np.asarray(response.as_matrix(), dtype=float),
            np.asarray(expected["matrix"], dtype=float),
        )


@pytest.mark.parametrize("container", ["list", "iterator", "generator"])
def test_three_group_factor_fit_matches_stock_r(container):
    case = REFERENCE["fit"]
    data = dict(case["data"])
    values = data.pop("group")
    if container == "iterator":
        values = iter(values)
    elif container == "generator":
        values = (x for x in values)
    data["g"] = r.strata(values, shortlabel=True)
    fit = r.survreg(case["formula"], data, dist=case["dist"], model=True, x=True)
    assert r.coef_names(fit) == case["beta_names"]
    design = r.model_matrix(fit)
    assert design["columns"] == case["design_names"]
    np.testing.assert_allclose(design["data"], case["design"], rtol=0, atol=0)
    for actual, expected in (
        (r.coef(fit), case["beta"]),
        (r.vcov(fit), case["variance"]),
        (r.predict(fit, type="lp"), case["lp"]),
        (fit.scale, case["scale"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)
    new = dict(case["newdata"])
    new["g"] = r.strata(iter(new.pop("group")), shortlabel=True)
    np.testing.assert_allclose(
        r.predict(fit, new, type="response"), case["prediction"], rtol=2e-7, atol=2e-9
    )


@pytest.mark.parametrize("container", ["list", "iterator", "generator"])
def test_iterator_grouping_produces_stock_r_curves(container):
    case = REFERENCE["fit"]
    data = dict(case["data"])
    values = data.pop("group")
    if container == "iterator":
        values = iter(values)
    elif container == "generator":
        values = (value for value in values)
    data["g"] = r.strata(values, shortlabel=True)
    actual = r.survfit("Surv(time,status) ~ g", data)
    expected = case["grouped_curves"]
    assert actual.strata == expected["strata"]
    for field in ("n", "time", "n_risk", "n_event", "n_censor"):
        np.testing.assert_array_equal(getattr(actual, field), expected[field])
    np.testing.assert_allclose(actual.surv, expected["surv"], rtol=2e-14, atol=2e-14)


def test_generators_are_consumed_once_and_factor_levels_remain_owned():
    source = FactorIterator(["b", "a", None, "b"], ["b", "unused", "a"])
    actual = r.strata(source)
    assert source.observed == ["b", "a", None, "b"]
    assert actual.levels == ["b", "a"]
    assert actual.codes == [0, 1, None, 0]
    assert list(source) == []
    category_view = actual.categories
    with pytest.raises(AttributeError):
        category_view.append("extra")
    assert actual.levels == ["b", "a"]


def test_mutable_columns_are_reusable_and_results_do_not_alias_inputs():
    values = [1, 2, 1, None]
    labels = RFactor(["b", "a", "b", None], ["b", "unused long label", "a"])
    columns = {"number": values, "label": labels}
    first = r.strata(columns, na_group=True)
    assert_factor(r.strata(columns, na_group=True), vars(first))
    assert values == [1, 2, 1, None]
    assert labels == ["b", "a", "b", None]
    values[0] = 99
    labels[0] = "a"
    labels.categories = ("a", "b")
    assert first.codes == [0, 1, 0, 2]
    assert first.labels[0].startswith("number=1")


def test_unequal_iterator_columns_raise_after_materialization():
    with pytest.raises(ValueError, match="all arguments must be the same length"):
        r.strata(iter([1, 2]), iter(["a"]))


def test_literal_na_state_label_and_missing_factor_value_match_stock_r():
    case = REFERENCE["state_label"]
    events = RFactor(case["values"], case["levels"])
    actual = r.Surv(list(range(1, len(events) + 1)), events)
    expected = case["expected"]
    assert actual.type == expected["type"] == "mright"
    assert list(actual.states) == expected["states"] == ["NA", "unused state"]
    assert actual.event == (0, 1, None, 1, 0)
    assert actual.clabel == expected["clabel"] == "censor"
    np.testing.assert_equal(
        np.asarray(actual.as_matrix(), dtype=float), np.asarray(expected["matrix"], dtype=float)
    )


class NumericFactorArray(np.ndarray):
    """An ndarray participating in the public declared-category input protocol."""


def numeric_label(value):
    return None if value is None else int(value) if value in {"2", "10", "30"} else value


def numeric_factor_column(spec, container):
    values = [numeric_label(value) for value in spec["values"]]
    levels = [numeric_label(level) for level in spec["levels"]]
    if container == "list":
        return RFactor(values, levels)
    if container == "iterator":
        return FactorIterator(values, levels)
    result = np.asarray(values, dtype=container).view(NumericFactorArray)
    result.categories = tuple(levels)
    return result


NUMERIC_FACTOR_CASES = [
    case
    for case in REFERENCE["cases"]
    if case["name"].split("/")[0] in {"numeric_factor", "numeric_factor_character"}
]


@pytest.mark.parametrize("case", NUMERIC_FACTOR_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("container", ["float32", "float64", "list", "iterator"])
def test_numeric_declared_factors_and_nested_reuse_match_stock_r(case, container):
    columns = {
        name: numeric_factor_column(spec, container)
        if spec["levels"] is not None
        else list(spec["values"])
        for name, spec in case["columns"].items()
    }
    if case["named"]:
        actual = r.strata(columns, **case["options"])
    else:
        actual = r.strata(*columns.values(), labels=list(columns), **case["options"])
    assert_factor(actual, case["expected"])
    assert_factor(r.strata(actual), case["roundtrip"])
    for name, values in columns.items():
        spec = case["columns"][name]
        if isinstance(values, FactorIterator):
            assert values.observed == [numeric_label(value) for value in spec["values"]]
        if isinstance(values, NumericFactorArray):
            assert values.categories == tuple(numeric_label(level) for level in spec["levels"])


ORDINARY_NUMERIC_CASES = [
    case for case in REFERENCE["cases"] if case["name"].split("/")[0] in {"integers", "doubles"}
]


@pytest.mark.parametrize("case", ORDINARY_NUMERIC_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ordinary_numeric_arrays_keep_stock_r_grouping(case, dtype):
    columns = {
        name: np.asarray(spec["values"], dtype=dtype) for name, spec in case["columns"].items()
    }
    if case["named"]:
        actual = r.strata(columns, **case["options"])
    else:
        actual = r.strata(*columns.values(), labels=list(columns), **case["options"])
    assert_factor(actual, case["expected"])


@pytest.mark.parametrize("container", ["float32", "float64", "list", "iterator"])
@pytest.mark.parametrize("explicit", [False, True])
def test_factor_coding_reads_one_shot_metadata_once_and_honors_explicit_levels(container, explicit):
    case = next(
        case
        for case in REFERENCE["cases"]
        if case["name"] == "numeric_factor/positional/FALSE/TRUE"
    )
    spec = case["columns"]["v1"]
    values = numeric_factor_column(spec, container)
    declared = [numeric_label(level) for level in spec["levels"]]
    values.categories = iter([999]) if explicit else iter(declared)
    codes, labels = factor_codes(values, levels=iter(declared) if explicit else None)
    assert codes == [2, 0, 2, None, 1, 0, None]
    assert labels == spec["levels"]
    assert list(values.categories) == ([999] if explicit else [])
    if isinstance(values, FactorIterator):
        assert values.observed == [10, 2, 10, None, 30, 2, None]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_declared_numeric_array_factor_preserves_categorical_model_design(dtype):
    case = REFERENCE["fit"]
    data = dict(case["data"])
    values = np.asarray(data.pop("group"), dtype=dtype).view(NumericFactorArray)
    values.categories = (2, 30, 10, "unused")
    data["g"] = r.strata(values, shortlabel=True)
    fit = r.survreg(case["formula"], data, dist=case["dist"], model=True, x=True)
    names = ["(Intercept)", "g30", "g10", "z"]
    order = [case["beta_names"].index(name) for name in names]
    variance_order = [*order, *range(len(order), len(case["variance"]))]
    assert r.coef_names(fit) == names
    design = r.model_matrix(fit)
    assert design["columns"] == names
    np.testing.assert_equal(design["data"], np.asarray(case["design"])[:, order])
    np.testing.assert_allclose(r.coef(fit), np.asarray(case["beta"])[order], rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(
        r.vcov(fit),
        np.asarray(case["variance"])[np.ix_(variance_order, variance_order)],
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(r.predict(fit, type="lp"), case["lp"], rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(fit.scale, case["scale"], rtol=2e-7, atol=2e-9)
    new = dict(case["newdata"])
    groups = np.asarray(new.pop("group"), dtype=dtype).view(NumericFactorArray)
    groups.categories = values.categories
    new["g"] = r.strata(groups, shortlabel=True)
    np.testing.assert_allclose(
        r.predict(fit, new, type="response"), case["prediction"], rtol=2e-7, atol=2e-9
    )
