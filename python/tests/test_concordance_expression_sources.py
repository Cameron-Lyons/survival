"""Coerced concordance responses preserve source columns and stock R dot terms."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/concordance_expression_reference.json").read_text()
)
COUNT_NAMES = ("concordant", "discordant", "tied.x", "tied.y", "tied.xy")


def assert_reference(fit, expected):
    p = len(expected["concordance"])
    counts = [fit.count] if isinstance(fit.count, dict) else fit.count
    np.testing.assert_allclose(np.atleast_1d(fit.concordance), expected["concordance"], atol=1e-14)
    np.testing.assert_allclose(
        [[row[name] for name in COUNT_NAMES] for row in counts], expected["count"], atol=1e-14
    )
    np.testing.assert_allclose(np.asarray(fit.var).reshape(p, p), expected["var"], atol=1e-14)
    np.testing.assert_allclose(np.atleast_1d(fit.cvar), expected["cvar"], atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(fit.dfbeta).reshape(-1, p), expected["dfbeta"], atol=1e-14
    )
    ranks = [fit.ranks] if isinstance(fit.ranks, dict) else fit.ranks
    for actual, reference in zip(ranks, expected["ranks"], strict=True):
        for name, values in reference.items():
            np.testing.assert_allclose(actual[name], values, atol=1e-14)
    assert fit.n == expected["n"]
    if expected["names"] is not None:
        # Formula names in Python omit R's deparsed backtick quoting.
        assert fit.names == [name.replace("`", "") for name in expected["names"]]
    else:
        # R drops names for a scalar concordance; Python keeps its predictor label.
        assert fit.names is not None
        assert len(fit.names) == 1
    assert ([] if fit.na_action is None else list(fit.na_action.rows)) == expected["omitted"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["lists", "numpy", "pandas", "iterators"])
def test_response_sources_match_stock_r(case, layout):
    data = {name: list(values) for name, values in case["data"].items()}
    if layout == "numpy":
        data = {name: np.asarray(values) for name, values in data.items()}
        for values in data.values():
            values.flags.writeable = False
    elif layout == "pandas":
        data = pytest.importorskip("pandas").DataFrame(data)
    elif layout == "iterators":
        data = {name: iter(values) for name, values in data.items()}
    arguments = {
        name: case[name]
        for name in ("weights", "cluster", "subset", "na_action")
        if case[name] is not None
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.concordance(case["formula"], data, influence=1, ranks=True, **arguments)
    if case["name"] == "warning_once":
        assert len(caught) == 1
        assert "NaNs produced" in str(caught[0].message)
    else:
        assert not caught
    assert_reference(fit, case["expected"])
    assert fit.formula == case["formula"]


def test_one_shot_response_source_is_shared_with_weights_and_cluster():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "shared_weights_cluster")
    source = iter(case["data"]["x"])
    fit = r.concordance(
        case["formula"],
        {"x": source, "z": iter(case["data"]["z"])},
        weights=source,
        cluster=source,
        influence=1,
        ranks=True,
    )
    assert_reference(fit, case["expected"])


def test_factor_response_keeps_original_predictor_levels_and_values():
    source = RFactor(["b", "a", "b", "a", "b"], ["b", "a"])
    data = {"g": source, "x": [3, 1, 2, 0, 4]}
    actual = r.concordance("I(g) ~ g + x", data, influence=1, ranks=True)
    expected = r.concordancefit(
        r.Surv([1, 2, 1, 2, 1]),
        np.column_stack(([0, 1, 0, 1, 0], data["x"])),
        names=["ga", "x"],
        influence=1,
        ranks=True,
    )
    for name in ("concordance", "count", "names", "var", "cvar", "dfbeta", "ranks"):
        assert getattr(actual, name) == getattr(expected, name)
    assert data["g"] is source
    assert source == ["b", "a", "b", "a", "b"]
    assert source.categories == ("b", "a")
