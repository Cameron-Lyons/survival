"""Cox model reports against R, independently checking layout and native fits."""

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
from survival.r._print import coefficient_matrix_lines, numeric_matrix_lines  # noqa: E402

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cox_report_reference.json").read_text())


def matrix(source):
    return r.NamedMatrix(
        source["rows"], source["columns"], np.asarray(source["values"], dtype=float).tolist()
    )


def summary_input(case):
    source = case["input"]
    result = {key: value for key, value in source.items() if value != {}}
    result.pop("omit", None)
    result["na_action"] = source["omit"] or None
    table = matrix(source["coefficients"])
    penal = source["model_type"] == "coxph.penal"
    robust = "robust se" in table.colnames
    keys = (
        ["coef", "se", "se2", "chisq", "df", "p"]
        if penal
        else ["coef", "exp_coef", "naive_se", "robust_se", "z", "p"]
        if robust
        else ["coef", "exp_coef", "se", "z", "p"]
    )
    result["coefficient_columns"] = table.colnames
    result["coefficients"] = [
        dict(name=name, **dict(zip(keys, row, strict=True)))
        for name, row in zip(table.rownames, table.values, strict=True)
    ]
    if source["conf_int"]:
        table = matrix(source["conf_int"])
        result["conf_int"] = [
            dict(
                name=name,
                **dict(zip(["exp(coef)", "exp(-coef)", "lower", "upper"], row, strict=True)),
            )
            for name, row in zip(table.rownames, table.values, strict=True)
        ]
    for name in ("logtest", "waldtest", "sctest", "robscore"):
        if source[name]:
            result[name] = dict(zip(["test", "df", "pvalue"], source[name], strict=True))
    if source["concordance"]:
        result["concordance"] = dict(zip(["C", "se(C)"], source["concordance"], strict=True))
    if source["cmap"]:
        result["cmap"] = matrix(source["cmap"])
    return result


@pytest.fixture(scope="module")
def fits():
    @lru_cache(None)
    def fit(name):
        spec = REFERENCE["models"][name]
        data = pd.DataFrame(REFERENCE["datasets"][spec["data"]])
        if spec["endpoint_levels"]:
            data["endpoint"] = pd.Categorical(data["endpoint"], categories=spec["endpoint_levels"])
        formula = spec["formula"][0] if len(spec["formula"]) == 1 else spec["formula"]
        return r.coxph(formula, data, **dict(spec["arguments"]))

    return fit


def options(arguments):
    return {name.replace(".", "_"): value for name, value in dict(arguments).items()}


@pytest.mark.parametrize("case", REFERENCE["matrices"])
def test_shared_coefficient_format_matches_r(case):
    actual = coefficient_matrix_lines(
        matrix(case["table"]),
        case["digits"],
        case["width"],
        signif_stars=case["stars"],
        row_title=case["title"] if case["title"] else None,
    )
    assert actual == case["lines"]


@pytest.mark.parametrize(
    "case", [case for case in REFERENCE["cases"] if case["input"]], ids=lambda case: case["name"]
)
def test_summary_text_from_r_snapshot(case, capsys):
    x = summary_input(case)
    report = (
        r.print_summary_coxph_penal(x, width=case["width"], **options(case["arguments"]))
        if x["model_type"] == "coxph.penal"
        else r.print_summary_coxph(x, width=case["width"], **options(case["arguments"]))
    )
    assert report.lines == case["lines"]
    assert str(report) == "\n".join(report.lines) + "\n"
    assert capsys.readouterr().out == ""
    expected = matrix(case["input"]["coefficients"])
    np.testing.assert_array_equal(report.tables["coefficients"].values, expected.values)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_native_cox_reports_match_r(case, fits):
    fit = fits(case["fit"])
    kwargs = dict(width=case["width"], **options(case["arguments"]))
    if case["method"] == "summary":
        summary = r.model_summary(fit, **options(case["summary_arguments"]))
        function = (
            r.print_summary_coxph_penal if fit.penalized is not None else r.print_summary_coxph
        )
        report = function(summary, **kwargs)
    else:
        function = r.print_coxph_penal if fit.penalized is not None else r.print_coxph
        report = function(fit, **kwargs)
    assert report.lines == case["lines"]


def test_report_owns_full_precision_tables_and_statistics(fits):
    summary = r.model_summary(fits("ordinary"), conf_int=0.9)
    before = pickle.dumps(summary)
    report = r.print_summary_coxph(summary, digits=2)
    frame = r.as_data_frame(report)
    assert frame["term"] == ["age", "sex"]
    assert frame["coef"] == [row["coef"] for row in summary["coefficients"]]
    assert report.tables["conf_int"].colnames[-2:] == ["lower .90", "upper .90"]
    frame["coef"][0] = 999
    assert report.tables["coefficients"].values[0][0] != 999
    report.tables["coefficients"].values[0][0] = 999
    report.statistics["logtest"]["test"] = 999
    assert pickle.dumps(summary) == before
    assert pickle.loads(pickle.dumps(report)) == report  # noqa: S301 - this test's own bytes
    assert r.as_data_frame(r.print_coxph(fits("null"))) == {}


@pytest.mark.parametrize(
    "kwargs",
    [{"digits": 0}, {"digits": 23}, {"width": 9}, {"width": True}, {"signif_stars": "yes"}],
)
def test_invalid_print_options_fail(fits, kwargs):
    with pytest.raises((TypeError, ValueError)):
        r.print_coxph(fits("ordinary"), **kwargs)


def test_dispatch_and_invalid_input(fits):
    assert r.print_coxph(fits("spline")).lines == r.print_coxph_penal(fits("spline")).lines
    summary = r.model_summary(fits("spline"))
    assert r.print_summary_coxph(summary).lines == r.print_summary_coxph_penal(summary).lines
    with pytest.raises(TypeError, match="summary"):
        r.print_summary_coxph(fits("ordinary"))
    with pytest.raises(TypeError, match="penalized"):
        r.print_coxph_penal(fits("ordinary"))
    with pytest.raises(TypeError, match="null"):
        r.print_coxph_null(fits("ordinary"))
    with pytest.raises(ValueError, match="maxlabel"):
        r.print_coxph_penal(fits("spline"), maxlabel=0)


@pytest.mark.parametrize("case", REFERENCE["numeric_matrices"])
def test_numeric_transition_table_wrapping_matches_r(case):
    assert (
        numeric_matrix_lines(
            matrix(case["table"]), case["digits"], case["width"], row_title=case["title"]
        )
        == case["lines"]
    )
