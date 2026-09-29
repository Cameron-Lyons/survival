"""Population model components against R survival 3.8-12."""

import json
import math
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
from survival.r._coerce import _r_factor  # noqa: E402

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/population_retention_reference.json").read_text()
)


def data_columns():
    result = {}
    for name, column in REFERENCE["data"].items():
        values = column["values"]
        if column["kind"] == "factor":
            result[name] = _r_factor(values, column["levels"])
        elif column["kind"] == "tcut":
            result[name] = r.tcut(values, column["cutpoints"], column["levels"])
        elif column["kind"] == "date":
            result[name] = [date.fromisoformat(value) for value in values]
        else:
            result[name] = [math.nan if value is None else value for value in values]
    return result


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=2e-10, atol=2e-12)


def assert_component(actual, expected):
    if expected is None:
        assert actual is None
        return
    kind = expected["kind"]
    if kind == "surv":
        assert isinstance(actual, r.Surv)
        assert actual.type == expected["surv_type"]
        assert_close(actual.as_matrix(), expected["values"])
    elif kind == "factor":
        if isinstance(actual, r.StrataFactor):
            assert actual.levels == expected["levels"]
            assert actual.labels == expected["values"]
            assert [code + 1 for code in actual.codes] == expected["codes"]
            assert actual.counts == [expected["values"].count(level) for level in actual.levels]
        else:
            assert list(actual.categories) == expected["levels"]
            assert list(actual) == expected["values"]
    elif kind == "tcut":
        assert_close(actual.values, expected["values"])
        assert_close(actual.cutpoints, expected["cutpoints"])
        assert actual.labels == expected["levels"]
    elif kind == "date":
        assert [value.isoformat() for value in actual] == expected["values"]
    else:
        assert_close(actual, expected["values"])


@pytest.fixture(scope="module")
def cox():
    return r.coxph("Surv(time, status) ~ z", REFERENCE["training"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_population_components_match_r(case, cox):
    options = dict(case["options"])
    options["na_action"] = case["na_action"]
    if case["subset"] is not None:
        options["subset"] = case["subset"]
    if case["weights"]:
        options["weights"] = [1, 2, 1, 3, 1, 2, 1, 1]
    if case["table"] == "population":
        options.update(
            ratetable=r.survexp_us(), rmap={"age": "agey*365.25", "sex": "sex", "year": "entry"}
        )
    elif case["table"] == "cox":
        options.update(ratetable=cox, rmap={"z": "score+1"})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="weights ignored")
        fit = getattr(r, case["function_name"])(case["formula"], data_columns(), **options)
    expected = case["expected"]
    if "individual" in expected:
        assert isinstance(fit, list)
        assert_close(fit, expected["individual"])
        return
    assert r.model_formula(fit) == case["formula"]
    assert r.model_term_names(fit) == case["term_labels"]
    if case["term_labels"]:
        assert r.model_term_names(fit, terms=[1]) == case["term_labels"][:1]
    assert_component(fit.x, expected["x"])
    assert_component(fit.y, expected["y"])
    if expected["model"] is None:
        assert fit.model is None
        with pytest.raises(TypeError, match="model=TRUE"):
            r.model_frame(fit)
    else:
        assert list(fit.model) == list(expected["model"])
        plain = {}
        for name, column in expected["model"].items():
            assert_component(fit.model[name], column)
            if column["kind"] == "surv":
                names = (
                    ["time", "status"]
                    if column["surv_type"] == "right"
                    else ["start", "stop", "status"]
                )
                plain.update(
                    zip(names, map(list, zip(*column["values"], strict=True)), strict=True)
                )
            elif column["kind"] == "date":
                plain[name] = [date.fromisoformat(value) for value in column["values"]]
            else:
                plain[name] = column["values"]
        assert r.model_frame(fit) == plain
        assert survival.model_frame(fit) == plain
    values = fit.pyears if case["function_name"] == "pyears" else fit.surv
    assert_close(np.asarray(values).ravel(order="F"), np.asarray(expected["values"]).ravel())


def test_direct_vector_calls_keep_requested_components():
    py = r.pyears(time=[100, 200], group=["b", "a"], x=True, y=True)
    assert py.x == [[2.0], [1.0]]
    assert py.y == [[100.0], [200.0]]
    ex = r.survexp(
        time=[100, 200],
        age=[14610, 18262.5],
        sex=[1, 2],
        year=[date(2000, 1, 1)] * 2,
        model=True,
        x=True,
        y=True,
    )
    assert ex.x is None
    assert ex.y is None
    assert ex.model == {
        "time": [100.0, 200.0],
        "age": [14610, 18262.5],
        "sex": [1, 2],
        "year": [date(2000, 1, 1)] * 2,
    }


@pytest.mark.parametrize("function", [r.pyears, r.survexp])
@pytest.mark.parametrize("flag", ["model", "x", "y"])
def test_retention_flags_validate_scalar_logicals(function, flag):
    with pytest.raises(TypeError, match=flag):
        function("time ~ 1", data_columns(), **{flag: [True, False]})


def test_retained_model_is_independent_of_source_columns():
    data = data_columns()
    fit = r.pyears("time ~ grp", data, model=True)
    before = r.model_frame(fit)
    data["time"][0] = -99
    data["grp"] = ["changed"] * 8
    assert r.model_frame(fit) == before


def test_model_precedence_skips_unused_flags_and_keeps_table_layout():
    fit = r.pyears(
        "time ~ grp", data_columns(), data_frame=True, model=True, x="unused", y="unused"
    )
    assert fit.model is not None
    assert fit.x is None
    assert fit.y is None
    assert fit.data is not None
    assert r.as_data_frame(fit) == fit.data
    ex = r.survexp(
        "time ~ 1",
        data_columns(),
        model=True,
        x="unused",
        y="unused",
        rmap={"age": "agey*365.25", "sex": "sex", "year": "entry"},
    )
    assert ex.model is not None
    assert ex.x is None
    assert ex.y is None


def test_cox_individual_exclude_restores_subset_rows(cox):
    # R errors when rows are removed in this branch; the supported Python
    # extension uses the same subset-relative exclusion convention as rate tables.
    data = {"time": [2, math.nan, 4], "status": [1, 0, 1], "z": [1, 2, 3]}
    omitted = r.survexp(
        "time ~ 1",
        data,
        ratetable=cox,
        method="individual.h",
        subset=[2, 1, 0, 1],
        na_action="omit",
    )
    excluded = r.survexp(
        "time ~ 1",
        data,
        ratetable=cox,
        method="individual.h",
        subset=[2, 1, 0, 1],
        na_action="exclude",
    )
    assert_close(excluded, [omitted[0], math.nan, omitted[1], math.nan])
