"""Expected-survival, rate-table and tmerge summaries against R survival 3.8-12."""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/population_summary_reference.json").read_text()
)


def numbers(values):
    return [math.nan if value is None else float(value) for value in values]


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("case", REFERENCE["native_cases"], ids=lambda case: case["name"])
def test_expected_survival_time_selection_matches_r(case):
    ncols = len(case["surv"][0])
    surv = [numbers(row) for row in case["surv"]]
    n_risk = [numbers(row) for row in case["n_risk"]]
    labels = [f"curve={i}" for i in range(ncols)] if ncols > 1 else None
    fit = r.SurvExpResult(
        time=case["time"],
        surv=surv if ncols > 1 else [row[0] for row in surv],
        n_risk=n_risk if ncols > 1 else [row[0] for row in n_risk],
        method="cohort",
        n=6,
        strata=labels,
    )
    kwargs = {"scale": numbers([case["scale"]])[0]}
    if not case["omitted"]:
        kwargs["times"] = numbers(case["times"])
    for function in (r.summary_survexp, r.model_summary, survival.model_summary):
        summary = function(fit, **kwargs)
        assert isinstance(summary, r.SurvExpSummary)
        assert summary.method == "cohort"
        assert summary.strata == labels
        assert_close(summary.time, numbers(case["expected"]["time"]))
        for name in ("surv", "n_risk"):
            actual = np.asarray(getattr(summary, name)).reshape(-1, ncols)
            expected = np.asarray(case["expected"][name], dtype=float).reshape(-1, ncols)
            assert_close(actual, expected)
        frame = r.as_data_frame(summary)
        assert len(frame["time"]) == len(summary.time) * ncols
        if labels is not None:
            assert frame["strata"] == [label for label in labels for _ in summary.time]
        if summary.time:
            assert_close(frame["surv"], np.asarray(summary.surv).reshape(-1, ncols).T.ravel())


@pytest.mark.parametrize("case", REFERENCE["fit_cases"], ids=lambda case: case["method"])
def test_population_curve_summary_matches_r(case):
    data = dict(REFERENCE["data"])
    data["year"] = [date.fromisoformat(value) for value in data["year"]]
    fit = r.survexp(
        "Surv(time, status) ~ grp",
        data,
        method=case["method"],
        rmap={"age": "age", "sex": "sex", "year": "year"},
        times=[50, 200, 400, 800],
        na_action="omit",
    )
    summary = r.summary_survexp(fit, times=[0, 100, 100, 200, 750, 900], scale=10)
    assert_close(summary.time, case["time"])
    assert_close(summary.surv, case["surv"])
    assert_close(summary.n_risk, case["n_risk"])
    assert summary.strata == case["labels"]
    assert r.as_data_frame(fit)["strata"] == [label for label in fit.strata for _ in fit.time]


@pytest.mark.parametrize("name", ["us", "usr", "mn"])
def test_rate_table_summaries_match_r(name):
    table = getattr(r, "survexp_" + name)()
    expected = REFERENCE["tables"][name]
    for function in (r.summary_ratetable, r.model_summary, survival.model_summary):
        summary = function(table)
        assert isinstance(summary, r.RateTableSummary)
        assert str(summary) == expected["text"]
        assert summary.attributes["dim"] == expected["dims"]
        assert summary.attributes["dimid"] == expected["dimid"]
        assert summary.attributes["type"] == expected["types"]
        assert summary.attributes["dimnames"] == expected["dimnames"]
        frame = r.as_data_frame(summary)
        assert frame == r.as_data_frame(table)
        assert frame["dimension"] == expected["dimid"]
        assert frame["categories"] == expected["dims"]
        for index, kind in enumerate(expected["types"]):
            if kind == 1:
                assert frame["levels"][index] == expected["dimnames"][index]
                assert frame["lower"][index] is None
                assert frame["upper"][index] is None
            elif kind > 2:
                date.fromisoformat(frame["lower"][index])
                date.fromisoformat(frame["upper"][index])


def test_tmerge_summary_matches_r_and_copies_counts():
    case = REFERENCE["tmerge"]
    merged = r.tmerge(
        case["base"], case["base"], id="id", tstop="stop", death=r.event("stop", "status")
    )
    merged = r.tmerge(merged, case["updates"], id="id", dose=r.tdc("time", "value"))
    expected = {"term": case["terms"]}
    expected.update(zip(case["columns"], map(list, zip(*case["counts"], strict=True)), strict=True))
    for function in (r.summary_tmerge, r.model_summary, survival.model_summary):
        summary = function(merged)
        assert summary == expected
        assert r.as_data_frame(summary) == expected
        summary["early"][0] = -1
        assert merged.tcount[case["terms"][0]]["early"] != -1


def test_summary_methods_validate_types_shapes_and_scalar_options():
    for function in (r.summary_survexp, r.summary_ratetable, r.summary_tmerge):
        with pytest.raises(TypeError):
            function([1, 2, 3])
    fit = r.SurvExpResult([1, 3], [0.9, 0.7], [4, 2], "cohort", 4)
    assert r.summary_survexp(fit, times=2).surv == [0.9]
    assert r.summary_survexp(fit, times=[]).surv == []
    with pytest.raises(ValueError, match="length 1"):
        r.summary_survexp(fit, scale=[1, 2])
    bad = r.SurvExpResult([1, 3], [0.9], [4, 2], "cohort", 4)
    with pytest.raises(ValueError, match="one row per time"):
        r.summary_survexp(bad)
    empty = r.SurvExpResult([], [], [], "cohort", 0)
    assert r.summary_survexp(empty, times=[0, 1]).time == []


def test_curve_tables_preserve_unnamed_columns_and_single_column_summaries():
    fit = r.SurvExpResult([1, 3], [[0.9, 0.8], [0.7, 0.6]], [[4, 5], [2, 3]], "cohort", 9)
    frame = r.as_data_frame(fit)
    assert frame["curve"] == [1, 1, 2, 2]
    assert frame["surv"] == [0.9, 0.7, 0.8, 0.6]
    assert r.as_data_frame(r.summary_survexp(fit)) == frame
    one = r.SurvExpResult([1, 3], [[0.9], [0.7]], [[4], [2]], "cohort", 4, ["group=a"])
    summary = r.summary_survexp(one)
    assert summary.surv == [0.9, 0.7]
    assert summary.n_risk == [4, 2]
    assert r.as_data_frame(summary)["strata"] == ["group=a", "group=a"]
