"""Infinite KM endpoints and near ties preserve independently fitted stock rows."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
core = survival._survival
REFERENCE = json.loads((Path(__file__).parent / "fixtures/km_infinite_reference.json").read_text())
CASES = REFERENCE["cases"]
SUMMARY_CASES = [(case, query) for case in CASES for query in case["summaries"]]
QUANTILE_CASES = [(case, query) for case in CASES for query in case["quantiles"]]
TABLE_CASES = [(case, query) for case in CASES for query in case["tables"]]


def numbers(values):
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def assert_numeric(actual, expected, field, *, signed_zero=False):
    if expected is None:
        assert actual is None, field
        return
    expected = numbers(expected)
    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape, field
    np.testing.assert_allclose(
        actual, expected, rtol=2e-11, atol=2e-13, equal_nan=True, err_msg=field
    )
    if signed_zero:
        zeros = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


def data_for(case):
    data = {
        key: numbers(value).tolist() for key, value in REFERENCE["inputs"][case["input"]].items()
    }
    if case["group"] is not None:
        data["group"] = list(case["group"])
    return data


def response_for(data):
    return (
        r.Surv(data["time"], data["status"])
        if "start" not in data
        else r.Surv(data["start"], data["time"], data["status"])
    )


def options_for(case, rows):
    options = {
        "weights": None if case["weights"] is None else numbers(case["weights"]).tolist(),
        "stype": case["stype"],
        "ctype": case["ctype"],
        "id": list(range(1, rows + 1)),
        "entry": "start" in REFERENCE["inputs"][case["input"]],
    }
    if case["weights"] is not None:
        options["influence"] = 3
    return options


def make_fit(case, interface):
    data = data_for(case)
    options = options_for(case, len(data["time"]))
    if interface == "native":
        strata = None if case["group"] is None else [int(value == "b") for value in case["group"]]
        return core.survfitkm(
            np.asarray(data["time"]),
            np.asarray(data["status"], dtype=np.int32),
            start=None if "start" not in data else np.asarray(data["start"]),
            strata=strata,
            timefix=case["timefix"],
            **options,
        )
    if interface == "formula":
        response = "Surv(time,status)" if "start" not in data else "Surv(start,time,status)"
        group = "group" if case["group"] is not None else "1"
        return r.survfit(f"{response} ~ {group}", data, timefix=case["timefix"], **options)
    response = response_for(data)
    if interface == "lowlevel":
        if case["timefix"]:
            response = r.aeqSurv(response)
        groups = case["group"] or ["1"] * len(data["time"])
        return r.survfitKM(r.strata(groups, shortlabel=True), response, **options)
    return r.survfit(response, group=case["group"], timefix=case["timefix"], **options)


def engine_of(fit):
    if isinstance(fit, core.SurvfitKMResult):
        return fit
    return fit._fit if hasattr(fit, "_fit") else fit.engine


def assert_fit(fit, expected):
    for field in (
        "n",
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "n_enter",
        "n_id",
        "surv",
        "cumhaz",
        "std_err",
        "std_chaz",
        "lower",
        "upper",
        "strata",
        "t0",
    ):
        actual = getattr(fit, field)
        if field == "t0":
            actual = [actual]
        if isinstance(actual, dict):
            actual = list(actual.values())
        assert_numeric(actual, expected[field], field)
    if expected.get("counts") is not None:
        assert fit.counts is not None
        for field, values in expected["counts"].items():
            assert_numeric(getattr(fit.counts, field), values, f"counts.{field}")
    for field in ("influence_surv", "influence_chaz"):
        expected_values = expected[field]
        actual = getattr(fit, field)
        if expected_values is None:
            assert actual is None
        else:
            expected_curves = (
                expected_values if expected["strata"] is not None else [expected_values]
            )
            assert len(actual) == len(expected_curves)
            for curve, reference in zip(actual, expected_curves, strict=True):
                assert_numeric(curve.values, reference, field)


@pytest.fixture(scope="module")
def fits():
    return {case["name"]: make_fit(case, "facade") for case in CASES}


@pytest.mark.parametrize("interface", ["facade", "formula", "lowlevel", "native"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_complete_fits_preserve_stock_infinite_endpoints(case, interface):
    assert_fit(make_fit(case, interface), case["fit"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_zero_rows_preserve_infinite_origins(case, fits):
    actual = r.survfit0(fits[case["name"]])
    assert_fit(actual, case["zero"])


def summary_id(item):
    case, query = item
    return f"{case['name']}-{case['summaries'].index(query)}"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("item", SUMMARY_CASES, ids=summary_id)
def test_summary_queries_accept_infinite_times(item, native, fits):
    case, query = item
    fit = fits[case["name"]]
    options = {"times": numbers(query["times"]), "extend": query["extend"]}
    if query["dosum"] is not None:
        options["dosum"] = query["dosum"]
    call = (
        (lambda: core.summary_survfit(fit.engine, **options))
        if native
        else (lambda: r.summary_survfit(fit, rmean="none", **options))
    )
    expected = query["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            call()
        return
    actual = call()
    for field in (
        "n",
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "n_enter",
        "surv",
        "cumhaz",
        "std_err",
        "std_chaz",
        "lower",
        "upper",
    ):
        assert_numeric(getattr(actual, field), expected["value"][field], field)
    if native:
        assert_numeric(
            actual.strata,
            None
            if expected["value"]["strata"] is None
            else [expected["value"]["strata"].count(f"group={level}") for level in ("a", "b")],
            "strata",
        )
    elif actual.strata is not None:
        assert actual.strata == [
            name.removeprefix("group=") for name in expected["value"]["strata"]
        ]


def quantile_id(item):
    case, query = item
    return f"{case['name']}-{case['quantiles'].index(query)}"


def stacked_engine(fit, arrays=False):
    arguments = {
        field: getattr(fit, field)
        for field in (
            "time",
            "n_risk",
            "n_event",
            "surv",
            "n",
            "n_censor",
            "cumhaz",
            "std_err",
            "std_chaz",
            "lower",
            "upper",
        )
    }
    if arrays:
        arguments = {
            key: None if value is None else np.asarray(value) for key, value in arguments.items()
        }
    return core.SurvfitKMResult.from_stacked(
        **arguments,
        strata=fit.engine.strata,
        n_id=fit.n_id,
        t0=fit.t0,
        logse=fit.logse,
        conf_int=fit.conf_int,
        conf_type=fit.conf_type,
    )


@pytest.mark.parametrize("interface", ["facade", "prepared", "stacked"])
@pytest.mark.parametrize("item", QUANTILE_CASES, ids=quantile_id)
def test_curve_quantiles_preserve_infinite_times_and_scaled_signs(item, interface, fits):
    case, query = item
    fit = fits[case["name"]]
    options = {"conf_int": query["confidence"], "scale": float(query["scale"][0])}
    probs = np.asarray([0, 0.25, 0.5, 0.75, 1])
    if interface == "facade":
        actual = r.quantile(fit, probs, **options)
    else:
        engine = fit.engine if interface == "prepared" else stacked_engine(fit, arrays=True)
        actual = core.quantile_survfit(engine, probs, **options)
    for field in ("quantile", "lower", "upper"):
        values = query["expected"]["value"].get(field)
        if values is not None and case["fit"]["strata"] is None:
            values = [values]
        assert_numeric(getattr(actual, field), values, field, signed_zero=True)


@pytest.mark.parametrize(
    "item", TABLE_CASES, ids=lambda item: f"{item[0]['name']}-{item[1]['rmean']}"
)
def test_complete_summary_tables_match_stock_with_infinite_areas(item, fits):
    case, query = item

    def call():
        return r.summary_survfit(fits[case["name"]], rmean=query["rmean"])

    expected = query["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=re.escape(expected["error"])):
            call()
        return
    actual = call()
    values = expected["value"]["table"]
    if case["fit"]["strata"] is None:
        values = [values]
    assert_numeric(actual.table.values, values, "table")
    assert_numeric(actual.rmean_endtime, expected["value"]["endtime"], "endtime")


@pytest.mark.parametrize("case", REFERENCE["aeq_cases"], ids=lambda case: case["name"])
def test_near_tie_normalization_preserves_endpoints_and_status_rows(case):
    data = {
        key: numbers(value).tolist() for key, value in REFERENCE["inputs"][case["name"]].items()
    }
    response = response_for(data)
    actual = r.aeqSurv(response)
    assert_numeric(actual.as_matrix(), case["expected"], "aeqSurv")
    assert actual.status == response.status
    assert len(actual) == len(response)
    if "start" not in data:
        timeline = r.Surv2(data["time"], data["status"])
        assert_numeric(r.aeqSurv(timeline).as_matrix(), case["expected"], "aeqSurv2")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "case", REFERENCE["start_cases"], ids=lambda case: f"{case['input']}-{case['start']}"
)
def test_infinite_start_time_matches_stock_selection(case, native):
    data = {
        key: numbers(value).tolist() for key, value in REFERENCE["inputs"][case["input"]].items()
    }
    start = float("nan") if case["start"][0] is None else float(case["start"][0])
    call = (
        (lambda: core.survfitkm(data["time"], data["status"], start_time=start))
        if native
        else (lambda: r.survfit(response_for(data), start_time=start))
    )
    expected = case["expected"]
    if "error" in expected:
        message = "start.time" if math.isnan(start) else re.escape(expected["error"])
        with pytest.raises(ValueError, match=message):
            call()
    else:
        assert_fit(call(), expected["value"])


@pytest.mark.parametrize("field", ["time", "start", "weights"])
def test_native_missing_values_remain_invalid(field):
    arguments = {
        "time": [1.0, 2.0, float("inf")],
        "status": [1, 0, 1],
        "start": [0.0, 0.0, 0.0],
        "weights": [1.0, 1.0, 1.0],
    }
    arguments[field][1] = float("nan")
    with pytest.raises(ValueError, match=field):
        core.survfitkm(**arguments)


def test_summary_queries_still_refuse_missing_times(fits):
    fit = fits[CASES[0]["name"]]
    for query in ([0.0, float("nan"), float("inf")], [None]):
        with pytest.raises((ValueError, TypeError)):
            r.summary_survfit(fit, times=query)
        with pytest.raises(ValueError, match="times"):
            core.summary_survfit(fit.engine, times=np.asarray(query, dtype=float))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    ("origin", "expected_control"),
    [(2.0, [1.0, 1.0, 0.0]), (3.0, [0.5, 0.5, 0.0]), (float("inf"), [0.0, 0.0, 0.0])],
)
def test_requested_summary_refuses_origin_inserted_after_earlier_entry_rows(
    origin, expected_control, native
):
    # Direct stock summary.survfit rejects these survfit0 grids via findInterval.
    response = r.Surv([-float("inf"), 0, 1, 2], [1, 2, 3, float("inf")], [1, 0, 1, 1])
    options = {"id": [1, 2, 3, 4], "start_time": origin, "timefix": False}
    fit = r.survfit(response, entry=True, **options)
    zero = core.survfit0(fit.engine)
    assert zero.time[0] == origin
    assert zero.time[1] < origin
    query = np.asarray([0.0, 2.0, float("inf")])
    call = (
        (lambda: core.summary_survfit(fit.engine, times=query, extend=True))
        if native
        else (lambda: r.summary_survfit(fit, times=query, extend=True, rmean="none"))
    )
    with pytest.raises(ValueError, match="sorted non-decreasingly"):
        call()
    # The conditional origin remains usable when earlier entries are not emitted.
    control = r.survfit(response, entry=False, **options)
    summary = core.summary_survfit(control.engine, times=query, extend=True)
    assert_numeric(summary.surv, expected_control, "surv")


def test_stacked_curves_still_refuse_nan_rows_and_origins(fits):
    fit = fits[CASES[0]["name"]]
    for field in ("time", "t0"):
        values = list(fit.time)
        values[0] = float("nan")
        with pytest.raises(ValueError, match="time"):
            core.SurvfitKMResult.from_stacked(
                values if field == "time" else fit.time,
                fit.n_risk,
                fit.n_event,
                fit.surv,
                fit.n,
                t0=float("nan") if field == "t0" else fit.t0,
            )
