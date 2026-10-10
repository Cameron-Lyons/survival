"""Population preprocessing owns iterators before expressions and row selection."""

import json
import math
import warnings
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/population_iterable_reference.json").read_text()
)


class CountedColumn(Iterator):
    def __init__(self, values, *, categories=None, ordered=False):
        self.values = list(values)
        self.source = iter(self.values)
        self.seen = []
        if categories is not None:
            self.categories = tuple(categories)
            self.ordered = ordered

    def __next__(self):
        value = next(self.source)
        self.seen.append(value)
        return value


def data_columns(case, layout):
    data, counted = {}, {}
    for name, column in case["data"].items():
        values = column["values"]
        if column["kind"] == "date":
            values = [None if value is None else date.fromisoformat(value) for value in values]
        if column["kind"] not in {"factor", "vector"} or name != "unused":
            values = [math.nan if value is None else value for value in values]
        if column["kind"] == "factor":
            values = column["values"]
            if layout in {"iterator", "generator"}:
                source = CountedColumn(
                    values, categories=column["levels"], ordered=column["ordered"]
                )
                counted[name] = source
            else:
                source = RFactor(values, column["levels"])
        elif layout == "iterator":
            source = CountedColumn(values)
            counted[name] = source
        elif layout == "generator":
            source = (value for value in values)
        elif layout == "array" and name != "unused":
            source = np.asarray(values, dtype=object if column["kind"] == "date" else float)
        else:
            source = [list(value) if isinstance(value, list) else value for value in values]
        data[name] = source
    return data, counted


def table():
    return r.RateTable(
        [3, 2],
        ["age", "sex"],
        [["0", "5", "10"], ["male", "female"]],
        [[0, 5, 10], None],
        [2, 1],
        [0.01, 0.02, 0.03, 0.005, 0.01, 0.015],
    )


@pytest.fixture(scope="module")
def cox():
    return r.coxph("Surv(time, status) ~ z", REFERENCE["training"])


def options(case, cox):
    result = dict(case["options"] or {})
    result.update(subset=case["subset"], na_action=case["na_action"])
    if case["table"] == "population":
        result.update(ratetable=table(), rmap={"age": "agebase + 1", "sex": "sex"})
        if case["rmap_kind"] == "response":
            result["rmap"]["age"] = "time"
        elif case["rmap_kind"] == "vectors":
            result["rmap"] = {
                name: RFactor(column["values"], column["levels"])
                if column["kind"] == "factor"
                else column["values"]
                for name, column in case["rate_vectors"].items()
            }
    elif case["table"] == "cox":
        result.update(ratetable=cox, rmap={"z": "score + 1"})
    elif case["table"] == "cox_factor":
        training = {
            **REFERENCE["training"],
            "zc": RFactor(REFERENCE["training"]["zc"], ["a", "b", "c"]),
        }
        result.update(ratetable=r.coxph("Surv(time, status) ~ zc", training), rmap={"zc": "zc"})
    elif case["table"] == "us":
        result.update(
            ratetable=r.survexp_us(), rmap={"age": "age_days", "sex": "sex", "year": "entry"}
        )
    if case["weights"]:
        result["weights"] = "time" if case["weights"] == "time" else "weight"
    return result


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=2e-10,
        atol=2e-12,
        equal_nan=True,
    )


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
            assert [None if code is None else code + 1 for code in actual.codes] == expected[
                "codes"
            ]
        else:
            assert list(actual.categories) == expected["levels"]
            assert list(actual) == expected["values"]
            assert bool(getattr(actual, "ordered", False)) == expected["ordered"]
    elif kind == "tcut":
        assert_close(actual.values, expected["values"])
        assert_close(actual.cutpoints, expected["cutpoints"])
        assert actual.labels == expected["levels"]
    elif kind == "date":
        assert [value.isoformat() for value in actual] == expected["values"]
    else:
        assert_close(actual, expected["values"])


def assert_result(fit, case, retention):
    expected = case["expected"]
    if "individual" in expected:
        assert isinstance(fit, list)
        assert_close(fit, expected["individual"])
        return
    if case["function_name"] == "pyears":
        for field in ("pyears", "n", "event", "expected"):
            actual = getattr(fit, field)
            if expected[field] is None:
                assert actual is None
            else:
                assert_close(np.asarray(actual).ravel(order="F"), expected[field])
        assert fit.offtable == pytest.approx(expected["offtable"])
        assert fit.observations == expected["observations"]
        assert fit.tcut == expected["tcut"]
        assert fit.dim == expected["dim"]
        assert fit.dimnames == expected["dimnames"]
    else:
        assert_close(fit.time, expected["time"])
        assert_component(fit.surv, expected["surv"])
        if expected["n_risk"] is not None:
            assert_component(fit.n_risk, expected["n_risk"])
        assert fit.strata == expected["strata"]
        assert fit.method == expected["method"]
        assert fit.n == expected["n"]
    if case["function_name"] == "pyears":
        assert (list(fit.na_action.rows) if fit.na_action else []) == expected["na_action"]
    if retention == "model":
        model = expected["model"]
        if case["rmap_kind"] == "vectors":
            # The Python direct vectors do not carry R expression symbol names.
            model = {
                name: value for name, value in model.items() if name not in {"rate_age", "rate_sex"}
            }
        assert list(fit.model) == list(model)
        for name, component in model.items():
            assert_component(fit.model[name], component)
        assert fit.x is None
        assert fit.y is None
    else:
        assert fit.model is None
        assert_component(fit.x, expected["x"])
        assert_component(fit.y, expected["y"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array", "iterator", "generator"])
@pytest.mark.parametrize("retention", ["model", "xy"])
def test_population_complete_outputs_match_stock_r(case, layout, retention, cox):
    data, counted = data_columns(case, layout)
    arguments = options(case, cox)
    if layout in {"iterator", "generator"}:
        arguments["subset"] = iter(arguments["subset"])
        if case["rmap_kind"] == "vectors":
            arguments["rmap"] = {
                name: CountedColumn(values, categories=getattr(values, "categories", None))
                for name, values in arguments["rmap"].items()
            }
    arguments.update({"model": True} if retention == "model" else {"x": True, "y": True})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(r, case["function_name"])(case["formula"], data, **arguments)
    assert sorted({str(item.message) for item in caught}) == sorted(case["warnings"])
    assert_result(fit, case, retention)
    for source in counted.values():
        assert len(source.seen) in {0, len(source.values)}
    if "unused" in counted:
        assert counted["unused"].seen == []
    if case["formula"].startswith("~") and "time" in counted:
        assert counted["time"].seen == []


@pytest.mark.parametrize("function_name", ["pyears", "survexp"])
def test_direct_and_mapped_aliases_consume_each_source_once(function_name, cox):
    case = next(
        case
        for case in REFERENCE["cases"]
        if case["name"]
        == ("pyears_factor_complete" if function_name == "pyears" else "survexp_ederer_complete")
    )
    data, counted = data_columns(case, "iterator")
    arguments = options(case, cox)
    arguments["rmap"] = {
        "age": CountedColumn([value + 1 for value in counted["agebase"].values]),
        "sex": data["sex"],
    }
    arguments["model"] = True
    fit = getattr(r, function_name)(case["formula"], data, **arguments)
    # The raw sex column appears both in data and in a direct rmap vector.
    assert counted["sex"].seen == counted["sex"].values
    # Literal rmap vectors do not become retained source columns in stock R.
    expected_model = dict(case["expected"]["model"])
    expected_model.pop("agebase", None)
    expected_model.pop("sex", None)
    expected_case = {**case, "expected": {**case["expected"], "model": expected_model}}
    assert_result(fit, expected_case, "model")


@pytest.mark.parametrize("layout", ["list", "iterator"])
def test_direct_pyears_named_response_reuses_original_rate_and_weight_sources(layout, cox):
    case = next(
        case for case in REFERENCE["cases"] if case["name"] == "pyears_matrix_entry_complete"
    )
    data, counted = data_columns(case, layout)
    arguments = options(case, cox)
    arguments.update(start="start", stop="time", group="grp", x=True, y=True)
    fit = r.pyears(data=data, **arguments)
    direct_case = {
        **case,
        "expected": {
            **case["expected"],
            "dimnames": {"group": case["expected"]["dimnames"]["grp"]},
        },
    }
    assert_result(fit, direct_case, "xy")
    if counted:
        assert counted["time"].seen == counted["time"].values
        assert counted["unused"].seen == []


def test_cox_prediction_mapping_keeps_unused_iterators_unread(cox):
    data = {
        "time": [2, 3, 4],
        "grp": ["a", "b", "a"],
        "score": [0, 1, 2],
        "unused": CountedColumn(["never", "read"]),
    }
    output = r.survexp("time ~ grp", data, ratetable=cox, rmap={"z": "score + 1"}, times=[2, 5])
    assert output.n == 3
    assert data["unused"].seen == []


@pytest.mark.parametrize(
    "case_name", ["survexp_response_free_complete", "survexp_cox_response_free_complete"]
)
def test_response_free_population_calls_ignore_first_unused_column(case_name, cox):
    case = next(case for case in REFERENCE["cases"] if case["name"] == case_name)
    data, counted = data_columns(case, "iterator")
    data = {
        "unused": data["unused"],
        **{name: value for name, value in data.items() if name != "unused"},
    }
    fit = r.survexp(case["formula"], data, model=True, **options(case, cox))
    assert_result(fit, case, "model")
    assert counted["unused"].seen == []
    assert counted["time"].seen == []


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("layout", ["list", "array", "iterator", "generator"])
@pytest.mark.parametrize("retention", ["model", "xy"])
def test_direct_survexp_vectors_match_complete_stock_formula_outputs(
    missing, layout, retention, cox
):
    case = next(
        case
        for case in REFERENCE["cases"]
        if case["name"] == "survexp_direct_vectors_" + ("missing" if missing else "complete")
    )
    data, counted = data_columns(case, layout)
    arguments = options(case, cox)
    arguments.update(time=data["time"], age=data["age_days"], sex=data["sex"], year=data["entry"])
    arguments.update({"model": True} if retention == "model" else {"x": True, "y": True})
    fit = r.survexp(data=data, **arguments)
    expected_model = case["expected"]["model"]
    # Direct vector names are canonical; the R reference formula names its sources.
    direct_model = {
        "time": expected_model["time"],
        "age": expected_model["age_days"],
        "year": expected_model["entry"],
        "sex": expected_model["sex"],
    }
    direct_case = {**case, "expected": {**case["expected"], "model": direct_model}}
    assert_result(fit, direct_case, retention)
    for name in ("time", "age_days", "sex", "entry"):
        if name in counted:
            assert len(counted[name].seen) == len(counted[name].values)
    if "unused" in counted:
        assert counted["unused"].seen == []


@pytest.mark.parametrize("missing", [False, True])
def test_one_iterator_aliased_across_direct_response_weight_and_rate_mapping(missing, cox):
    case = next(
        case
        for case in REFERENCE["cases"]
        if case["name"] == "pyears_response_alias_" + ("missing" if missing else "complete")
    )
    data, counted = data_columns(case, "iterator")
    arguments = options(case, cox)
    arguments.update(time=data["time"], group=data["grp"], weights=data["time"], x=True, y=True)
    arguments["rmap"]["age"] = data["time"]
    fit = r.pyears(data=data, **arguments)
    direct_case = {
        **case,
        "expected": {
            **case["expected"],
            "dimnames": {"group": case["expected"]["dimnames"]["grp"]},
        },
    }
    assert_result(fit, direct_case, "xy")
    assert len(counted["time"].seen) == len(counted["time"].values)
    assert counted["unused"].seen == []


def test_retained_population_outputs_do_not_own_reusable_input_storage(cox):
    case = next(
        case for case in REFERENCE["cases"] if case["name"] == "pyears_matrix_entry_complete"
    )
    data, _ = data_columns(case, "array")
    before = data["Y_interval"].copy()
    categories = data["grp"].categories
    fit = r.pyears(case["formula"], data, model=True, **options(case, cox))
    fit.model["Y_interval"][0][0] = -999
    fit.model["grp"] = ["changed"]
    np.testing.assert_array_equal(data["Y_interval"], before)
    assert data["grp"].categories is categories
    assert data["grp"][0] == "b"


def test_scalar_only_rate_map_does_not_get_rows_from_unused_mapping_data():
    unused = CountedColumn(range(8))
    with pytest.raises(ValueError, match=REFERENCE["scalar_mapping_error"]):
        r.survexp(
            "~1",
            {"unused": unused},
            ratetable=table(),
            rmap={"age": 2, "sex": "male"},
            times=[0, 2],
        )
    assert unused.seen == []


MATRIX_CASES = [case for case in REFERENCE["cases"] if case["formula"].startswith("Y_")]


@pytest.mark.parametrize("case", MATRIX_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, object])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("retention", ["model", "xy"])
def test_numpy_matrix_row_iterators_match_stock_complete_output(
    case, dtype, direct, retention, cox
):
    data, _ = data_columns(case, "list")
    response_name = case["formula"].split("~", 1)[0].strip()
    matrix = np.asarray(data[response_name], dtype=dtype)
    before = matrix.copy()
    data[response_name] = iter(matrix)
    arguments = options(case, cox)
    arguments.update({"model": True} if retention == "model" else {"x": True, "y": True})
    expected_case = case
    if direct:
        arguments["group"] = "grp"
        fit = r.pyears(data[response_name], data, **arguments)
        model = case["expected"]["model"]
        direct_model = {
            "Y": model[response_name],
            "group": model["grp"],
            **{name: value for name, value in model.items() if name not in {response_name, "grp"}},
        }
        expected_case = {
            **case,
            "expected": {
                **case["expected"],
                "model": direct_model,
                "dimnames": {"group": case["expected"]["dimnames"]["grp"]},
            },
        }
    else:
        fit = r.pyears(case["formula"], data, **arguments)
    assert_result(fit, expected_case, retention)
    if retention == "model":
        fit.model["Y" if direct else response_name][0][0] = -999
    else:
        fit.y[0][0] = -999
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize(
    "case",
    REFERENCE["scalar_weights"],
    ids=lambda case: f"frame{case['data_frame']}-n{len(case['weights'])}",
)
@pytest.mark.parametrize("layout", ["list", "iterator"])
def test_scalar_rate_maps_expand_to_stock_weight_rows(case, layout):
    unused = CountedColumn(range(4))
    data = (
        pytest.importorskip("pandas").DataFrame({"unused": range(4)})
        if case["data_frame"]
        else {"unused": unused}
    )
    weights = CountedColumn(case["weights"]) if layout == "iterator" else case["weights"]
    fit = r.survexp(
        "~1",
        data,
        weights=weights,
        times=[100, 200],
        x=True,
        y=True,
        rmap={"age": 14610, "sex": 1, "year": date(2000, 1, 1)},
    )
    expected = case["expected"]
    assert_close(fit.time, expected["time"])
    for name in ("surv", "n_risk", "x", "y"):
        assert_component(getattr(fit, name), expected[name])
    assert fit.method == expected["method"]
    assert fit.n == len(case["weights"])
    assert unused.seen == []
    if layout == "iterator":
        assert weights.seen == case["weights"]
