"""Detailed reports: exact R text, numeric tables, and native-summary integration."""

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .test_survfit_print import fits as fits

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/detailed_report_reference.json").read_text()
)


def decoded(field):
    if field is None:
        return None
    values = np.asarray(field["values"], dtype=float)
    if field["dim"] is not None:
        values = values.reshape(field["dim"], order="F")
    return values.tolist()


def input_object(case):
    source = case["input"]
    values = {name: decoded(field) for name, field in source["fields"].items()}
    method = case["method"]
    if method in {"survexp", "summary.survexp"}:
        common = {name: values[name] for name in ("time", "surv", "n_risk")}
        common.update(method=source["method"], strata=source["names"])
        return r.SurvExpResult(**common, n=0) if method == "survexp" else r.SurvExpSummary(**common)
    for name in ("time", "n_risk", "n_event", "n_censor"):
        values[name] = values[name] or []
    common = {name: values[name] for name in ("time", "n_risk", "n_event", "n_censor")}
    common.update(
        strata=source["strata"],
        strata_levels=source["strata_levels"],
        table=r.NamedMatrix(None, [], []),
        states=source["states"],
        start_time=source["start_time"],
    )
    if source["coxms"]:
        return r.SummarySurvfitCoxmsResult(
            **common,
            pstate=np.asarray(values["pstate"]),
            cumhaz=None,
            n_transition=None,
            rmean_endtime=None,
            newdata=None,
        )
    if method == "summary.survfitms" and values["pstate"] is not None:
        probability = np.asarray(values["pstate"])
        values["pstate"] = (
            probability[:, None].tolist() if probability.ndim == 1 else probability.tolist()
        )
    common.update(
        {name: values[name] for name in ("surv", "pstate", "std_err", "lower", "upper", "n_enter")}
    )
    return r.SummarySurvfitResult(
        **common, n=[], cumhaz=[], type=source["type"], conf_int=source["conf_int"]
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_detailed_report_matches_r(case, capsys):
    function = getattr(r, "print_" + case["method"].replace(".", "_"))
    value = input_object(case)
    if case["expected"]["error"] is not None:
        message = "no events" if case["name"] == "censored" else "one row"
        with pytest.raises(ValueError, match=message):
            function(value)
        return
    result = function(value, width=case["width"], **dict(case["arguments"]))
    assert len(result.tables) == len(case["expected"]["tables"])
    for actual, expected in zip(result.tables, case["expected"]["tables"], strict=True):
        assert actual.colnames == expected["columns"]
        np.testing.assert_allclose(
            actual.values, np.asarray(expected["values"], dtype=float), rtol=1e-14, atol=1e-15
        )
    assert result.lines == case["expected"]["lines"]
    assert str(result) == "\n".join(result.lines) + "\n"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "name",
    [
        "km",
        "groups",
        "weighted",
        "no_se",
        "cox",
        "cox_groups",
        "aj",
        "aj_groups",
        "coxms",
        "interval",
    ],
)
def test_native_summary_report_matches_r(name, fits):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    summary = r.summary_survfit(fits[name])
    actual = r.print_summary_survfit(summary)
    assert actual.lines == case["expected"]["lines"]


def test_scaled_conditional_summary_retains_valid_rows(fits):
    summary = r.summary_survfit(fits["cox_conditional"], times=[0, 5.5, 10, 20], scale=10)
    assert summary.start_time == 0.55
    report = r.print_summary_survfit(summary)
    assert r.as_data_frame(report)["time"] == [0.55, 1, 2]
    multistate = r.summary_survfit(fits["aj_conditional"], times=[0, 1, 2, 6], scale=2)
    assert multistate.start_time == 1
    assert r.as_data_frame(r.print_summary_survfitms(multistate))["time"] == [1, 3]


def test_empty_group_levels_and_single_row_layout_survive_native_summary():
    fit = r.survfit(
        "Surv(time,status)~group",
        {"time": [1, 2, 3, 5], "status": [1, 0, 0, 0], "group": ["a", "a", "b", "b"]},
    )
    summary = r.summary_survfit(fit)
    assert summary.strata_levels == ["group=a", "group=b"]
    report = r.print_summary_survfit(summary)
    assert report.groups == ["group=a", "group=b"]
    assert report.tables[1].values == []
    assert report.lines[1].startswith(" " * 8 + "time")


def test_counting_summary_retains_censor_column(fits):
    data = fits["delayed"].model
    response = data["Surv(start, time, status)"]
    fit = r.survfit(response, entry=True, id=list(range(len(response))))
    summary = r.summary_survfit(fit, censored=True)
    assert summary.type == "counting"
    report = r.print_summary_survfit(summary)
    assert report.tables[0].colnames[:4] == ["time", "n.risk", "n.event", "censored"]
    assert r.as_data_frame(report)["censored"] == summary.n_censor


def test_reports_preserve_input_and_return_independent_full_precision_columns(fits):
    summary = r.summary_survfit(fits["groups"])
    original = pickle.dumps(summary)
    report = r.print_summary_survfit(summary)
    frame = r.as_data_frame(report)
    assert frame["strata"] == summary.strata
    np.testing.assert_array_equal(frame["survival"], summary.surv)
    frame["survival"][0] = -1
    assert report.tables[0].values[0][3] >= 0
    report.tables[0].values[0][3] = -1
    assert pickle.dumps(summary) == original
    assert pickle.loads(pickle.dumps(report)) == report  # noqa: S301 - this test's own bytes


def test_native_expected_summaries_keep_group_risk_columns():
    lung = survival.datasets.load_lung()
    fit = r.survexp(
        "~sex", lung, ratetable=r.coxph("Surv(time,status)~age+sex", lung), times=[100, 300, 600]
    )
    report = r.print_survexp(fit)
    assert report.tables[0].colnames == ["time", "nrisk1", "nrisk2", "sex=1", "sex=2"]
    summary = r.summary_survexp(fit, times=[0, 200, 500])
    frame = r.as_data_frame(r.print_summary_survexp(summary))
    assert frame["nrisk1"] == [138, 138, 138]
    assert frame["nrisk2"] == [90, 90, 90]
    assert frame["time"] == [0, 200, 500]


def test_legacy_summary_without_new_metadata_still_formats(fits):
    summary = r.summary_survfit(fits["groups"])
    legacy = dataclasses.replace(summary, type=None, start_time=None, strata_levels=None)
    assert r.print_summary_survfit(legacy).lines == r.print_summary_survfit(summary).lines


@pytest.mark.parametrize(
    "name",
    ["print_summary_survfit", "print_summary_survfitms", "print_survexp", "print_summary_survexp"],
)
def test_wrong_report_objects(name):
    with pytest.raises(TypeError, match="requires"):
        getattr(r, name)(None)


def test_malformed_summary_shapes_fail_clearly(fits):
    summary = r.summary_survfit(fits["km"])
    with pytest.raises(ValueError, match="one row"):
        r.print_summary_survfit(dataclasses.replace(summary, n_risk=[1]))
    with pytest.raises(ValueError, match="one label"):
        r.print_summary_survfit(dataclasses.replace(summary, strata=["bad"]))
    with pytest.raises(ValueError, match="confidence"):
        r.print_summary_survfit(dataclasses.replace(summary, upper=None))
