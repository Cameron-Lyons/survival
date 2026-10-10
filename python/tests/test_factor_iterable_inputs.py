"""Public factor/grouping iterator boundaries against independent stock R."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r_api
factor_codes = importlib.import_module("survival.r._coerce")._factor
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/factor_iterable_reference.json").read_text()
)
CONTAINERS = ["list", "iterator", "generator", "array", "pandas"]


class CountedIterator:
    def __init__(self, values, levels=None):
        self.source = iter(values)
        self.observed = []
        if levels is not None:
            self.categories = tuple(levels)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.source)
        self.observed.append(value)
        return value


def values_of(spec):
    return [
        float(value) if isinstance(value, str) and spec["kind"] == "double" else value
        for value in spec["values"]
    ]


def column(spec, container):
    values = values_of(spec)
    levels = spec["levels"]
    if container == "iterator":
        return CountedIterator(values, levels)
    if container == "generator":
        return CountedIterator((value for value in values), levels)
    if levels is not None:
        return (
            pd.Categorical(values, categories=levels)
            if container == "pandas"
            else RFactor(values, levels)
        )
    if container == "array":
        return np.asarray(
            values, dtype=object if spec["kind"] in {"character", "logical"} else float
        )
    if container == "pandas":
        dtype = (
            "string"
            if spec["kind"] == "character"
            else "boolean"
            if spec["kind"] == "logical"
            else "Float64"
        )
        return pd.Series(values, dtype=dtype)
    return list(values)


def assert_consumed(source, spec):
    if isinstance(source, CountedIterator):
        np.testing.assert_equal(source.observed, values_of(spec))


@pytest.mark.parametrize("case", REFERENCE["factors"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", CONTAINERS)
def test_inferred_and_declared_factor_inputs_match_stock_r(case, container):
    source = column(case["input"], container)
    codes, labels = factor_codes(source)
    assert codes == case["codes"]
    assert labels == case["levels"]
    assert_consumed(source, case["input"])


@pytest.mark.parametrize("case", REFERENCE["aggregate"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", CONTAINERS)
def test_public_curve_aggregation_matches_stock_r_with_one_shot_groups(case, container):
    if case["named"]:
        inputs = {name: column(spec, container) for name, spec in case["by"].items()}
        by = inputs
        pairs = [(source, case["by"][name]) for name, source in inputs.items()]
    else:
        spec = case["by"][0]
        by = column(spec, container)
        pairs = [(by, spec)]
    curves = SimpleNamespace(**{case["margin"]: REFERENCE["curves"][case["margin"]]})
    actual = r.aggregate_survfit(curves, by=by, FUN=case["fun"])
    np.testing.assert_allclose(
        getattr(actual, case["margin"]), case["result"]["values"], rtol=1e-14, atol=0
    )
    groups = actual.newdata
    actual_groups = {
        name: [labels[index] for labels in groups.labels] for index, name in enumerate(groups.names)
    }
    expected = case["result"]["newdata"]
    assert list(actual_groups) == list(expected)
    for name, labels in expected.items():
        assert actual_groups[name] == [str(label) for label in labels]
    for source, spec in pairs:
        assert_consumed(source, spec)


def fit_concordance(case, clusters):
    data = REFERENCE["data"]
    return r.concordancefit(
        r.Surv(data["time"], data["event"]),
        data["x"],
        weights=data["weights"] if case["weighted"] else None,
        cluster=clusters,
        influence=3,
        ranks=True,
    )


def assert_concordance(actual, expected):
    assert actual.n == expected["n"]
    assert actual.names is None
    assert actual.formula is None
    assert actual.na_action is None
    assert actual.count == pytest.approx(expected["count"], rel=1e-13, abs=1e-14)
    for name in ("concordance", "var", "cvar", "dfbeta", "influence"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=2e-13, atol=1e-14)
    assert set(actual.ranks) == set(expected["ranks"])
    for name, values in expected["ranks"].items():
        np.testing.assert_allclose(actual.ranks[name], values, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("case", REFERENCE["concordance"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", CONTAINERS)
def test_public_clustered_concordance_matches_whole_stock_r_result(case, container):
    source = column(case["cluster"], container)
    assert_concordance(fit_concordance(case, source), case["result"])
    assert_consumed(source, case["cluster"])


@pytest.mark.parametrize("case", REFERENCE["legacy"], ids=lambda case: case["name"])
@pytest.mark.parametrize("container", CONTAINERS)
def test_public_legacy_concordance_retains_one_shot_strata(case, container):
    data = REFERENCE["data"]
    source = column(case["strata"], container)
    with pytest.warns(DeprecationWarning, match="survConcordance.fit is deprecated"):
        actual = r.survConcordance_fit(
            r.Surv(data["time"], data["event"]), data["x"], strata=source
        )
    assert list(actual) == case["names"]
    for name, expected in zip(case["names"], case["result"], strict=True):
        assert actual[name] == pytest.approx(expected, rel=2e-13, abs=1e-14)
    assert_consumed(source, case["strata"])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numeric_factor_array_cluster_keeps_declared_order_and_omits_unused_groups(weighted, dtype):
    case = next(
        case
        for case in REFERENCE["concordance"]
        if case["name"] == f"numeric_declared/{str(weighted).upper()}"
    )

    class FactorArray(np.ndarray):
        pass

    source = np.asarray([float(value) for value in case["cluster"]["values"]], dtype=dtype).view(
        FactorArray
    )
    source.categories = tuple(float(level) for level in case["cluster"]["levels"])
    assert_concordance(fit_concordance(case, source), case["result"])
    assert source.categories == (30.0, 99.0, 20.0, 10.0)


def test_missing_and_wrong_length_cluster_iterators_still_raise():
    data = REFERENCE["data"]
    response = r.Surv(data["time"], data["event"])
    with pytest.raises(ValueError, match="y and cluster are not the same length"):
        r.concordancefit(response, data["x"], cluster=iter(["b", "a"]))
    with pytest.raises(ValueError, match="cluster contains missing values"):
        r.concordancefit(
            response, data["x"], cluster=iter(["b", "a", None, "b", "a", "c", "b", "a"])
        )


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE["concordance"] if case["cluster"]["levels"] is not None],
    ids=lambda case: case["name"],
)
def test_cluster_values_and_declared_level_iterators_are_consumed_once(case):
    spec = case["cluster"]
    source = CountedIterator(values_of(spec))
    levels = CountedIterator(spec["levels"])
    source.categories = levels
    assert_concordance(fit_concordance(case, source), case["result"])
    assert_consumed(source, spec)
    assert levels.observed == spec["levels"]


def test_aggregate_inputs_are_reusable_and_outputs_do_not_alias_inputs():
    case = next(case for case in REFERENCE["aggregate"] if case["name"] == "surv/declared/mean")
    curves = SimpleNamespace(surv=[list(row) for row in REFERENCE["curves"]["surv"]])
    groups = RFactor(case["by"][0]["values"], case["by"][0]["levels"])
    actual = r.aggregate_survfit(curves, by=groups)
    np.testing.assert_equal(curves.surv, REFERENCE["curves"]["surv"])
    assert groups == case["by"][0]["values"]
    curves.surv[0][0] = 99
    groups[0] = "a"
    groups.categories = ("a", "b")
    np.testing.assert_allclose(actual.surv, case["result"]["values"], rtol=1e-14, atol=0)
    assert actual.newdata.labels == [["b"], ["a"]]
