"""Formula one-shot columns retain independent stock-R fitting and row selection."""

import copy
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r_api
from survival.r._coerce import _mstate_categories, _r_factor  # noqa: E402
from survival.r._formula import model_frame as formula_frame  # noqa: E402

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/formula_iterator_reference.json").read_text()
)


class ColumnIterator:
    def __init__(self, values, spec=None):
        self.values = iter(values)
        self.consumed = []
        if spec is not None and spec["levels"] is not None:
            self.categories = tuple(spec["levels"])
            self.ordered = spec["ordered"]

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.consumed.append(value)
        return value


class UnusedIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("an unused formula column was consumed")


def layout_columns(specs, layout):
    # Row naming and dimension queries must use evaluated formula columns;
    # an unused first column must remain unread through every entrance.
    columns = {"unused": UnusedIterator()}
    for key, spec in specs.items():
        values = copy.deepcopy(spec["values"])
        if layout in {"iterator", "generator"}:
            source = (value for value in values) if layout == "generator" else values
            columns[key] = ColumnIterator(source, spec)
        elif spec["levels"] is not None:
            columns[key] = (
                pd.Categorical(values, categories=spec["levels"], ordered=spec["ordered"])
                if layout == "pandas"
                else _r_factor(values, spec["levels"], ordered=spec["ordered"])
            )
        elif layout == "array":
            columns[key] = np.asarray(values, dtype=float)
        elif layout == "pandas":
            columns[key] = pd.Series(values, dtype="Float64")
        else:
            columns[key] = values
    return columns


def data_columns(name, layout):
    return layout_columns(REFERENCE["inputs"][name], layout)


def assert_values(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_values(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_values(value, reference)
    elif isinstance(expected, str | bool):
        if expected in {"Inf", "-Inf"} and not isinstance(actual, str):
            assert actual == float(expected)
        else:
            assert actual == expected
    elif expected is None:
        assert actual is None or np.isnan(actual)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)


def action_record(value):
    return None if value is None else {"rows": list(value.rows), "kind": value.kind}


def fit_case(case, data, **extra):
    options = {
        "weights": "wt",
        "subset": case["subset"],
        "na_action": case["na_action"],
        **extra,
    }
    routine = case["routine"]
    if routine == "frame":
        return r.model_frame(case["formula"], data, **options)
    if routine == "survfit":
        options.update(model=True, timefix=False)
    elif routine == "aareg":
        options.update(model=True, nmin=3)
    elif routine != "concordance":
        options.update(model=True, x=True)
    return getattr(r, routine)(case["formula"], data, **options)


def check_matrix(actual, expected):
    # Python labels use compact expressions; R deparse adds spaces.
    assert [name.replace(" ", "") for name in actual["columns"]] == [
        name.replace(" ", "") for name in expected["columns"]
    ]
    assert actual["assign"] == expected["assign"]
    assert actual["row_names"] == expected["row_names"]
    assert_values(actual["data"], expected["data"])


def flattened_fit_frame(expected):
    # The public Python frame exposes source columns and the total offset;
    # stock stores its evaluated offset expression as one model-frame column.
    frame = copy.deepcopy(expected)
    if "offset(off)" in frame:
        values = frame.pop("offset(off)")
        frame["off"] = values
        frame["(offset)"] = list(values)
    return frame


def check_result(case, fit, *, with_frame=True):
    expected = case["result"]
    routine = case["routine"]
    if routine == "frame":
        assert_values(fit, expected["frame"])
        return
    assert action_record(fit.na_action) == expected["na_action"]
    if routine == "survfit":
        assert_values(r.as_data_frame(fit), expected["frame"])
        assert_values(r.model_frame(fit), expected["model"])
    elif routine == "concordance":
        assert_values(
            {key: getattr(fit, key) for key in ["concordance", "count", "var", "n"]},
            {key: expected[key] for key in ["concordance", "count", "var", "n"]},
        )
    elif routine == "aareg":
        for key in [
            "n",
            "times",
            "n_risk",
            "coefficient",
            "test_statistic",
            "test_variance",
            "time_weights",
        ]:
            assert_values(getattr(fit, key), expected[key])
        if with_frame:
            assert_values(r.model_frame(fit), flattened_fit_frame(expected["frame"]))
    else:
        for key in ["coefficients", "var", "loglik", "linear_predictors"]:
            assert_values(getattr(fit, key), expected[key])
        assert_values(r.predict(fit), expected["predict"])
        if routine == "coxph":
            assert_values(fit.means, expected["means"])
            assert_values(fit.residuals, expected["residuals"])
            assert_values(r.predict(fit, type="expected"), expected["expected"])
        else:
            assert_values(fit.scale, expected["scale"])
            assert_values(r.predict(fit, type="quantile", p=[0.25, 0.75]), expected["quantile"])
        check_matrix(r.model_matrix(fit, _with_metadata=True), expected["matrix"])
        if with_frame:
            assert_values(r.model_frame(fit), flattened_fit_frame(expected["frame"]))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array", "pandas", "iterator", "generator"])
def test_formula_outputs_match_independent_stock_r(case, layout):
    data = data_columns(case["input"], layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_case(case, data)
        check_result(case, fit)
    for name, column in data.items():
        if isinstance(column, ColumnIterator):
            assert len(column.consumed) in {
                0,
                len(REFERENCE["inputs"][case["input"]][name]["values"]),
            }


FRAME_CASES = [case for case in REFERENCE["cases"] if case["routine"] == "frame"]


@pytest.mark.parametrize("case", FRAME_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "pandas", "iterator"])
def test_formula_factor_metadata_and_relative_omission_match_stock(case, layout):
    frame = formula_frame(
        case["formula"],
        data_columns(case["input"], layout),
        weights="wt",
        subset=case["subset"],
        na_action=case["na_action"],
    )
    expected = case["result"]
    assert action_record(frame.na_action) == expected["na_action"]
    source = frame.data["g"]
    assert list(_mstate_categories(source)) == expected["factor"]["levels"]
    assert list(source) == expected["factor"]["values"]
    assert (
        bool(getattr(source, "ordered", getattr(getattr(source, "dtype", None), "ordered", False)))
        == expected["factor"]["ordered"]
    )


@pytest.mark.parametrize("routine", ["frame", "coxph", "survreg", "survfit"])
def test_shared_column_and_explicit_weights_iterators_have_one_owned_row_sequence(routine):
    case = next(
        case for case in REFERENCE["cases"] if case["name"] == f"{routine}/aliased/all/omit"
    )
    data = data_columns("aliased", "list")
    # R sees equal response times and case weights. Python may express both
    # columns and the explicit argument with the same one-shot source.
    shared = ColumnIterator(REFERENCE["inputs"]["complete"]["time"]["values"])
    data["time"] = shared
    data["wt"] = shared
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_case(case, data, weights=shared)
    check_result(case, fit)
    assert shared.consumed == REFERENCE["inputs"]["complete"]["time"]["values"]


@pytest.mark.parametrize("routine", ["coxph", "survreg", "survfit", "aareg"])
def test_retained_models_snapshot_source_rows_and_return_owned_public_frames(routine):
    case = next(
        case for case in REFERENCE["cases"] if case["name"] == f"{routine}/complete/all/omit"
    )
    data = data_columns("complete", "list")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_case(case, data)
    expected = r.model_frame(fit)
    data["time"][0] = 99
    data["x"][0] = 88
    assert r.model_frame(fit) == expected
    for values in expected.values():
        if values:
            values[0] = "changed"
    check_result(case, fit)


@pytest.mark.parametrize("kind", ["list", "frame"])
@pytest.mark.parametrize("unused_first", [False, True])
def test_response_free_formula_reads_used_columns_and_leaves_other_lengths_unused(
    kind, unused_first
):
    # Independent stats::model.frame ignores an unused scalar in list data.
    data = {"unused": [8], "x": [4, 5, 6]} if unused_first else {"x": [4, 5, 6], "unused": [8]}
    if kind == "frame":
        data = pd.DataFrame(
            {key: values * 3 if len(values) == 1 else values for key, values in data.items()}
        )
    assert r.model_frame("~x", data) == {"x": [4, 5, 6]}


@pytest.mark.parametrize("case", REFERENCE["extra_row_count_rules"])
def test_variable_free_frames_and_row_aligned_extras_follow_stock_row_counts(case):
    data = {"unused": [1, 2, 3, 4], "x": [5, 6, 7, 8]}
    if case["kind"] == "frame":
        data = pd.DataFrame(data)
    options = {} if case["weights"] is None else {"weights": case["weights"]}
    if "error" in case:
        with pytest.raises(ValueError, match="weights|length"):
            r.model_frame(case["formula"], data, **options)
    else:
        assert_values(r.model_frame(case["formula"], data, **options), case["columns"])


@pytest.fixture(scope="module")
def prediction_fits():
    return {
        routine: fit_case(
            next(
                case
                for case in REFERENCE["cases"]
                if case["name"] == f"{routine}/complete/all/omit"
            ),
            data_columns("complete", "list"),
        )
        for routine in ["coxph", "survreg"]
    }


@pytest.mark.parametrize("case", REFERENCE["prediction_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array", "pandas", "iterator", "generator"])
def test_prediction_newdata_columns_and_shared_collapse_match_stock(case, layout, prediction_fits):
    specs = REFERENCE["prediction_inputs"][case["input"]]
    data = layout_columns(specs, layout)
    options = {"type": case["type"], "se_fit": case["se_fit"], "na_action": case["na_action"]}
    if case["routine"] == "survreg":
        options["p"] = case["p"]
    if "collapse" in case:
        # Reusing a one-shot predictor as an explicit row-aligned argument must
        # produce the same groups and predictors as the two equal R vectors.
        options["collapse"] = data[case["collapse"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = r.predict(prediction_fits[case["routine"]], data, **options)
    actual = {"fit": result.fit, "se_fit": result.se_fit} if case["se_fit"] else result
    assert_values(actual, case["result"])
    for name, column in data.items():
        if isinstance(column, ColumnIterator):
            assert len(column.consumed) in {0, len(specs[name]["values"])}


@pytest.mark.parametrize("routine", ["coxph", "survreg"])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("copy_method", ["pickle", "deepcopy", "copy"])
def test_formula_iterator_fits_serialize_without_reading_unused_columns(
    routine, retained, copy_method
):
    case = next(
        case for case in REFERENCE["cases"] if case["name"] == f"{routine}/complete/all/omit"
    )
    data = data_columns("complete", "iterator")
    fit = getattr(r, routine)(case["formula"], data, weights="wt", model=retained, x=True)
    if copy_method == "pickle":
        restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own generated object
    else:
        restored = getattr(copy, copy_method)(fit)
    # Public AFT model-frame extraction requires model=True; its numerical and
    # matrix methods remain available without that optional stored frame.
    check_result(case, restored, with_frame=retained or routine == "coxph")
    prediction = next(
        case
        for case in REFERENCE["prediction_cases"]
        if case["name"] == f"{routine}/complete/pass/lp/FALSE"
    )
    newdata = layout_columns(REFERENCE["prediction_inputs"]["complete"], "iterator")
    assert_values(r.predict(restored, newdata, type="lp"), prediction["result"])
    for name, column in data.items():
        if isinstance(column, ColumnIterator):
            assert len(column.consumed) in {
                0,
                len(REFERENCE["inputs"]["complete"][name]["values"]),
            }


@pytest.mark.parametrize("case", REFERENCE["special_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("copy_method", ["pickle", "deepcopy", "copy"])
def test_typed_and_formally_removed_iterator_columns_survive_subset_and_serialization(
    case, retained, copy_method
):
    specs = REFERENCE["special_inputs"][case["input"]]
    data = layout_columns(specs, "iterator")
    if case["input"] == "typed_double":
        # R x is double while each JSON number/iterator item is an integer.
        # Multiplication by 2L must preserve double arithmetic beyond INT_MAX.
        data["x"].dtype = np.dtype("float64")
    fit = r.coxph(
        case["formula"],
        data,
        weights="wt",
        subset=case["subset"],
        na_action="na.pass",
        model=retained,
        x=True,
    )
    if copy_method == "pickle":
        restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own generated object
    else:
        restored = getattr(copy, copy_method)(fit)
    check_result(case, restored, with_frame=case["input"] != "typed_double")
    if case["input"] == "typed_double":
        # The Python public frame retains raw source columns. Compare those
        # with an independent stock raw-source frame, then check the full
        # transformed design against the original fitted stock matrix above.
        assert_values(r.model_frame(restored), flattened_fit_frame(case["source_frame"]))
        matrix = r.model_matrix(restored, _with_metadata=True)["data"]
        assert all(np.isfinite(row[0]) for row in matrix)
        assert any(row[0] > np.iinfo(np.int32).max for row in matrix)
    if case["input"] == "cancelled":
        assert data["z"].consumed == specs["z"]["values"]


COX_INDIVIDUAL_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_individual_reference.json").read_text()
)


@pytest.fixture(scope="module")
def individual_curve_fit():
    return r.coxph(
        "Surv(start, stop, event) ~ x + z + strata(g) + offset(offset)",
        COX_INDIVIDUAL_REFERENCE["data"],
        weights="weight",
        ties="efron",
        robust=False,
    )


@pytest.mark.parametrize("case", COX_INDIVIDUAL_REFERENCE["cases"])
@pytest.mark.parametrize("layout", ["list", "iterator", "generator"])
@pytest.mark.parametrize("id_source", ["column", "shared", "separate"])
def test_individual_cox_curves_own_newdata_and_id_rows_against_stock(
    case, layout, id_source, individual_curve_fit
):
    source = COX_INDIVIDUAL_REFERENCE["newdata"]
    specs = {
        key: {"values": values, "levels": None, "ordered": False} for key, values in source.items()
    }
    data = layout_columns(specs, layout)
    ids = (
        "id"
        if id_source == "column"
        else data["id"]
        if id_source == "shared"
        else ColumnIterator(source["id"])
    )
    result = r.survfit(
        individual_curve_fit,
        data,
        id=ids,
        stype=case["stype"],
        ctype=case["ctype"],
        se_fit=case["se_fit"],
    )
    for name, expected in case["expected"].items():
        assert_values(getattr(result, name.replace(".", "_")), expected)
    if not case["se_fit"]:
        assert result.std_err is result.lower is result.upper is None
    for name, column in data.items():
        if isinstance(column, ColumnIterator):
            assert len(column.consumed) in {0, len(source[name])}
    if id_source == "separate":
        assert ids.consumed == source["id"]


@pytest.mark.parametrize("case", REFERENCE["direct_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "iterator", "generator"])
def test_direct_surv_row_aligned_iterators_match_stock_after_missingness_and_subset(case, layout):
    specs = REFERENCE["direct_inputs"][case["routine"]]
    data = layout_columns(specs, layout)
    expected = case["result"]
    if case["routine"] == "survcheck":
        result = r.survcheck(
            r.Surv(data["start"], data["stop"], data["status"]),
            id=data["id"],
            subset=case["subset"],
        )
        for name in ["states", "istate", "n", "id", "na_action"]:
            assert_values(getattr(result, name), expected[name])
        for name in expected["flag"]:
            assert getattr(result.flag, name) == expected["flag"][name]
        for table in ["transitions", "events"]:
            assert_values(
                {name: getattr(getattr(result, table), name) for name in expected[table]},
                expected[table],
            )
        for name in ["gap", "overlap", "jump", "teleport"]:
            problem = getattr(result, name)
            assert_values(
                None if problem is None else {"row": problem.row, "id": problem.id}, expected[name]
            )
    else:
        result = r.survdiff(
            r.Surv(data["time"], data["status"]),
            group=data["g"],
            subset=case["subset"],
        )
        actual = {name: getattr(result, name) for name in expected}
        actual["na_action"] = action_record(result.na_action)
        assert_values(actual, expected)
    for name, column in data.items():
        if isinstance(column, ColumnIterator):
            assert column.consumed == specs[name]["values"]


@pytest.mark.parametrize("case", REFERENCE["subset_alias_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "iterator", "generator"])
def test_one_selector_source_is_shared_with_formula_response_and_row_arguments(case, layout):
    data = layout_columns(case["input"], layout)
    selector = data[case["source"]]
    options = {"subset": selector}
    if "weights" in case["name"]:
        options["weights"] = selector
    if case["routine"] == "frame":
        actual = r.model_frame(case["formula"], data, **options)
    elif case["routine"] == "coxph":
        fit = r.coxph(case["formula"], data, **options)
        actual = {name: getattr(fit, name) for name in ["n", "nevent", "coefficients", "loglik"]}
        actual["predict"] = {name: r.predict(fit, type=name) for name in case["result"]["predict"]}
    elif case["routine"] == "survfit":
        fit = (
            r.survfit(r.Surv(data["time"], data["status"]), group=selector, **options)
            if case["name"] == "direct-km-group"
            else r.survfit(case["formula"], data, **options)
        )
        actual = {name: getattr(fit, name) for name in case["result"]}
    else:
        fit = r.survcheck(r.Surv(data["time"], data["status"]), id=selector, **options)
        actual = {}
        for name, expected in case["result"].items():
            value = getattr(fit, name)
            actual[name] = (
                {key: getattr(value, key) for key in expected}
                if isinstance(expected, dict) and not isinstance(value, dict)
                else value
            )
    assert_values(actual, case["result"])
    if isinstance(selector, ColumnIterator):
        assert selector.consumed == case["input"][case["source"]]["values"]
