"""Diagnostic reports checked against native fits and independent R snapshots."""

import copy
import dataclasses
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/diagnostic_report_reference.json").read_text()
)


def options(arguments):
    return {key.replace(".", "_"): value for key, value in dict(arguments).items()}


def omitted(source):
    return r.NaAction(tuple(source["omit"]), "omit") if source.get("omit") else None


def count_input(source):
    columns = source["columns"]
    if "rows" in source:
        return [dict(zip(columns, row, strict=True)) for row in source["values"]], source[
            "rows"
        ] or None
    return dict(zip(columns, source["values"], strict=True)), None


@pytest.fixture(scope="module")
def objects():
    @lru_cache(None)
    def fit(name):
        spec = REFERENCE["objects"][name]
        data = pd.DataFrame(
            {
                key: [np.nan if value == "NA" else value for value in values]
                for key, values in REFERENCE["datasets"][spec["data"]].items()
            }
        )
        if spec["kind"] == "cox_zph":
            formula = (
                "Surv(time,status)~pspline(age, df = 4)+sex"
                if name == "zph_spline"
                else "Surv(time,status)~age+sex"
            )
            kwargs = (
                {"transform": "identity", "global_test": False} if name == "zph_identity" else {}
            )
            return r.cox_zph(r.coxph(formula, data), **kwargs)
        if spec["kind"] == "concordancefit":
            if name == "multiple_none_comparable":
                return r.concordancefit(
                    r.Surv([1, 2, 3], [0, 0, 0]),
                    [[1, 3], [2, 2], [3, 1]],
                    names=["first", "second"],
                )
            if name == "none_comparable":
                return r.concordancefit(r.Surv([1, 2, 3], [0, 0, 0]), [1, 2, 3])
            y = r.Surv(data.time, data.status)
            if name in {"multiple", "multiple_no_variance"}:
                return r.concordancefit(
                    y, data[["age", "sex"]].values, names=["age", "sex"], std_err=name == "multiple"
                )
            return r.concordancefit(y, data.age, std_err=False)
        if spec["kind"] == "survConcordance":
            with pytest.warns(DeprecationWarning, match="survConcordance is deprecated"):
                return r.survConcordance(spec["formula"], data, **options(spec["arguments"]))
        return getattr(r, spec["kind"])(spec["formula"], data, **options(spec["arguments"]))

    return fit


def function(case):
    return getattr(
        r, "print_" + ("clogit" if case["method"] == "coxph" else case["method"].replace(".", "_"))
    )


def snapshot(case):
    source = case["input"]
    method = case["method"]
    if method == "cox.zph":
        table = source["table"]
        return r.CoxZPHResult(
            [
                dict(name=name, **dict(zip(["chisq", "df", "p"], values, strict=True)))
                for name, values in zip(table["rows"], table["values"], strict=True)
            ],
            [],
            [],
            [],
            [],
            source["transform"],
            table["rows"],
        )
    if method == "summary.cch":
        table = source["table"]
        result = dict(source, model_type="cch")
        names = table["rows"] or ["age"]
        result["coefficients"] = [
            dict(name=name, **dict(zip(["coef", "se", "z", "p"], values, strict=True)))
            for name, values in zip(names, table["values"], strict=True)
        ]
        return result
    if method == "concordance":
        count, names = count_input(source["count"])
        estimates = np.asarray(source["concordance"], dtype=float).tolist()
        variance = source["variance"]
        return r.ConcordanceResult(
            estimates if len(estimates) > 1 else estimates[0],
            count,
            source["n"],
            names=names,
            var=np.asarray(variance, dtype=float).tolist() if variance != {} else None,
            na_action=omitted(source),
        )
    if method == "survConcordance":
        count, names = count_input(source["stats"])
        stats = dict(zip(names, count, strict=True)) if names else count
        return r.SurvConcordanceResult(
            source["concordance"], stats, source["n"], source["std_err"], na_action=omitted(source)
        )
    if method == "survdiff":
        return r.SurvDiffResult(
            source["n"],
            np.atleast_1d(source["obs"]).tolist(),
            np.atleast_1d(source["exp"]).tolist(),
            np.atleast_2d(source["variance"]).tolist(),
            source["chisq"],
            source["pvalue"],
            1,
            source["groups"] or [],
            na_action=omitted(source),
        )
    raise AssertionError(method)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_native_report_matches_r(case, objects, capsys):
    value = objects(case["object"])
    if case["method"] == "summary.cch":
        value = r.model_summary(value)
    report = function(case)(value, width=case["width"], **options(case["arguments"]))
    assert report.lines == case["lines"]
    assert str(report) == "\n".join(report.lines) + "\n"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "case", [case for case in REFERENCE["cases"] if case["input"]], ids=lambda case: case["name"]
)
def test_independent_snapshot_matches_r(case):
    value = snapshot(case)
    report = function(case)(value, width=case["width"], **options(case["arguments"]))
    assert report.lines == case["lines"]
    if case["method"] == "cox.zph":
        np.testing.assert_array_equal(
            report.tables["tests"].values, case["input"]["table"]["values"]
        )
    if case["method"] == "summary.cch":
        np.testing.assert_array_equal(
            np.asarray(report.tables["coefficients"].values)[:, 0],
            np.asarray(case["input"]["table"]["values"])[:, 0],
        )


@pytest.mark.parametrize(
    ("name", "printer"),
    [
        ("zph", "print_cox_zph"),
        ("multiple", "print_concordance"),
        ("logrank", "print_survdiff"),
        ("Prentice", "print_cch"),
        ("legacy_strata", "print_survConcordance"),
    ],
)
def test_tables_are_independent_and_convert_to_frames(name, printer, objects):
    value = objects(name)
    before = value.coefficients if name == "Prentice" else copy.deepcopy(value)
    report = getattr(r, printer)(value)
    restored = pickle.loads(pickle.dumps(report))  # noqa: S301 - this test's own bytes
    assert restored == report
    table = report.tables[report.primary_table]
    frame = r.as_data_frame(report)
    if table.rownames:
        assert frame[report.row_label] == table.rownames
    column = table.colnames[0]
    original = table.values[0][0]
    frame[column][0] = 999
    assert table.values[0][0] == original
    table.values[0][0] = 999
    if name == "Prentice":
        assert value.coefficients == before
    else:
        assert value == before


@pytest.mark.parametrize(
    "printer",
    [
        "print_cch",
        "print_summary_cch",
        "print_clogit",
        "print_cox_zph",
        "print_concordance",
        "print_survConcordance",
        "print_survdiff",
    ],
)
def test_report_rejects_wrong_object(printer):
    with pytest.raises(TypeError):
        getattr(r, printer)(None)


def test_omission_records_preserve_rows_and_action(objects):
    for name in ["omitted", "legacy_omitted", "logrank_missing"]:
        assert objects(name).na_action is not None
    data = survival.datasets.load_lung()
    for function in [r.concordance, r.survConcordance, r.survdiff]:
        if function is r.survConcordance:
            with pytest.warns(DeprecationWarning, match="survConcordance is deprecated"):
                result = function("Surv(time,status)~ph.ecog", data, na_action="exclude")
        else:
            result = function("Surv(time,status)~ph.ecog", data, na_action="exclude")
        assert result.na_action == r.NaAction((14,), "exclude")
    raw = r.survdiff(r.Surv([1, 2, 3, 4], [1, None, 1, 0]), group=["a", "a", "b", "b"])
    assert raw.na_action == r.NaAction((2,), "omit")


def test_explicit_stratum_labels_survive_case_cohort_reports(objects):
    fit = objects("I.Borgan")
    renamed = dataclasses.replace(fit, stratum_names=("first", "second"))
    assert r.print_cch(renamed).tables["sizes"].colnames == ["first", "second"]


@pytest.mark.parametrize(
    "case", [case for case in REFERENCE["cases"] if case["input"]], ids=lambda case: case["name"]
)
def test_numeric_results_retain_r_precision(case, objects):
    value = objects(case["object"])
    if case["method"] == "summary.cch":
        value = r.model_summary(value)
    report = function(case)(value, width=case["width"], **options(case["arguments"]))
    reference = function(case)(snapshot(case), width=case["width"], **options(case["arguments"]))
    for name, expected in reference.tables.items():
        assert report.tables[name].colnames == expected.colnames
        np.testing.assert_allclose(
            report.tables[name].values, expected.values, rtol=2e-8, atol=1e-10
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"digits": 0}, {"digits": 23}, {"digits": True}, {"width": 9}, {"width": float("inf")}],
)
def test_numeric_report_options_validate(objects, kwargs):
    with pytest.raises((TypeError, ValueError), match="digits|width"):
        r.print_cox_zph(objects("zph"), **kwargs)


def test_omission_records_follow_subset_positions_and_missing_predictors():
    data = {"time": [1, 2, 3, 4, 5, 6], "status": [1, 1, 1, 0, 1, 0], "x": [0, 1, None, 1, 0, 1]}
    for function in [r.survdiff, r.concordance]:
        value = function("Surv(time,status)~x", data, subset=[1, 2, 3, 4, 5], na_action="exclude")
        assert value.na_action == r.NaAction((2,), "exclude")
        printer = r.print_survdiff if function is r.survdiff else r.print_concordance
        assert "1 observation deleted due to missingness" in printer(value).lines[0]


def test_fitted_concordance_does_not_invent_formula_omission_metadata():
    data = survival.datasets.load_lung()
    fit = r.coxph("Surv(time,status)~ph.ecog", data)
    assert fit.na_action is not None
    assert r.concordance(fit).na_action is None


def test_case_cohort_large_hazard_ratios_remain_infinite(objects):
    summary = r.model_summary(objects("Prentice_single"))
    summary["coefficients"][0]["coef"] = 1000
    report = r.print_summary_cch(summary)
    assert report.tables["coefficients"].values[0][1:4] == [float("inf")] * 3
    assert "Inf" in report.lines[-1]
