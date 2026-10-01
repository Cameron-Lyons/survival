"""Prepared term helpers compared with unmodified R survival exports."""

import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures/term_helpers_reference.json").read_text())
QUERIES = [(case, query) for case in REFERENCE["special_cases"] for query in case["queries"]]


@pytest.mark.parametrize("case", REFERENCE["assign_cases"], ids=lambda case: case["name"])
def test_column_assignments_match_r(case):
    matrix = {"assign": case["assign"]}
    expected = case["expected"]
    if expected == []:  # R's unnamed empty list
        expected = {}
    found = r.attrassign(matrix, r.TermMetadata(case["term_labels"]))
    assert found == expected
    assert list(found) == list(expected)
    assert r.attrassign(matrix, {"term.labels": case["term_labels"]}) == expected


@pytest.mark.parametrize(
    ("case", "query"),
    QUERIES,
    ids=[f"{case['formula']}/{query['special']}/{query['order']}" for case, query in QUERIES],
)
def test_special_variable_and_term_positions_match_r(case, query):
    tt = r.TermMetadata(**case["metadata"])
    if query["error"] is not None:
        assert query["error"] == "incorrect number of dimensions"
        with pytest.raises(ValueError, match="factors and order"):
            r.untangle_specials(tt, query["special"], query["order"])
        return
    expected = query["expected"]
    found = r.untangle_specials(tt, query["special"], query["order"])
    assert found == expected
    assert list(found) == list(expected)
    assert r.untangle_specials(case["metadata"], query["special"], query["order"]) == expected


def test_prepared_metadata_is_immutable_and_pickles():
    labels, variables = ["strata(g)"], ["y", "strata(g)"]
    factors, order, specials = [[0], [1]], [1], {"strata": [2]}
    tt = r.TermMetadata(labels, variables, factors, order, 1, specials)
    labels[0] = variables[0] = "changed"
    factors[1][0] = 0
    order[0] = 2
    specials["strata"].clear()
    expected = {"vars": ["strata(g)"], "tvar": [1], "terms": [1]}
    assert r.untangle_specials(tt, "strata") == expected
    restored = pickle.loads(pickle.dumps(tt))  # noqa: S301 - own round-trip data
    assert r.untangle_specials(restored, "strata") == expected
    with pytest.raises(FrozenInstanceError):
        tt.response = 0
    with pytest.raises(TypeError):
        tt.specials["strata"] = ()


def test_helpers_do_not_touch_design_data_and_use_first_column_order():
    class Matrix:
        assign = np.array([2, 0, 1, 2, 0])

        @property
        def data(self):
            raise AssertionError("matrix values should not be read")

    assert r.attrassign(Matrix(), r.TermMetadata(["a", "b"])) == {
        "b": [1, 4],
        "(Intercept)": [2, 5],
        "a": [3],
    }
    assert r.attrassign(Matrix(), {"term_labels": ["same", "same"]}) == {
        "same": [1, 3, 4],
        "(Intercept)": [2, 5],
    }
    from survival.r._survpenal import assign_list

    assert assign_list(Matrix.assign, ["a", "b"], 0) == (
        ["b", "(Intercept)", "a"],
        [[0, 3], [1, 4], [2]],
    )
    assert assign_list(Matrix.assign, ["a", "b"], 2) == (["(Intercept)", "a"], [[1, 4], [2]])


@pytest.mark.parametrize(
    "case", REFERENCE["fit_cases"], ids=lambda case: f"{case['kind']}/{case['formula']}"
)
def test_fitted_models_supply_matching_labels(case):
    data = survival.datasets.load_lung()
    fit = getattr(r, case["kind"])(case["formula"], data)
    matrix = r.model_matrix(fit)
    assert r.model_term_names(fit) == case["labels"]
    assert r.model_term_names(fit, [2]) == case["labels"][1:2]
    assert matrix["assign"] == case["assign"]
    assert r.attrassign(matrix, fit) == case["expected"]
    assert survival.r_api.attrassign is r.attrassign
    assert survival.r_api.untangle_specials is r.untangle_specials


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"term_labels": [1]}, "strings"),
        ({"factors": [[0, 1], [1, 0]]}, "one row"),
        ({"factors": [[0], [3]]}, "only 0"),
        ({"order": []}, "positive degree"),
        ({"order": [0]}, "positive degree"),
        ({"response": 2}, "0 or 1"),
        ({"specials": {"strata": [0]}}, "one-based"),
        ({"specials": {"strata": [3]}}, "one-based"),
        ({"specials": {"strata": [1.5]}}, "integer"),
    ],
)
def test_malformed_term_metadata_is_rejected(changes, message):
    args = {
        "term_labels": ["strata(g)"],
        "variables": ["y", "strata(g)"],
        "factors": [[0], [1]],
        "order": [1],
        "response": 1,
        "specials": {"strata": [2]},
    }
    with pytest.raises((ValueError, TypeError), match=message):
        r.TermMetadata(**{**args, **changes})


@pytest.mark.parametrize("assign", [[-1], [2], [0.5], [np.nan], [True], [[1]]])
def test_invalid_assign_codes(assign):
    with pytest.raises((ValueError, TypeError)):
        r.attrassign({"assign": assign}, r.TermMetadata(["x"]))


def test_missing_metadata_and_absent_specials():
    with pytest.raises(TypeError, match="model matrix"):
        r.attrassign(np.ones((2, 1)), r.TermMetadata(["x"]))
    with pytest.raises(TypeError, match="term metadata"):
        r.attrassign({"assign": [1]}, SimpleNamespace())
    with pytest.raises(TypeError, match="TermMetadata"):
        r.untangle_specials("~ strata(g)", "strata")
    assert r.untangle_specials(r.TermMetadata(["x"]), "strata", object()) == {
        "vars": [],
        "terms": [],
    }
    partial = r.TermMetadata(["strata(g)"], ["strata(g)"], specials={"strata": [1]})
    with pytest.raises(ValueError, match="factors and order"):
        r.untangle_specials(partial, "strata")
