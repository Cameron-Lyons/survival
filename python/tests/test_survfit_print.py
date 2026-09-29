"""Compact numeric reports and text, checked against R's print methods."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/survfit_print_reference.json").read_text()
)


def options(encoded):
    return {key.replace(".", "_"): value for key, value in dict(encoded).items()}


@pytest.fixture(scope="module")
def fits():
    from survival.r._coerce import _r_factor

    data = {
        name: {
            key: _r_factor(value["values"], value["levels"]) if isinstance(value, dict) else value
            for key, value in columns.items()
        }
        for name, columns in REFERENCE["data"].items()
    }
    result = {}
    for key, spec in REFERENCE["fits"].items():
        arguments = options(spec["arguments"])
        if spec["kind"] == "cox":
            model = r.coxph(spec["formula"], data[spec["data"]], **arguments)
            fit = r.survfit(model, **options(spec["curve"]))
        else:
            fit = r.survfit(spec["formula"], data[spec["data"]], **arguments)
        result[key] = r.survfit0(fit) if spec["origin"] else fit
    return result


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_curve_report_matches_r(case, fits, capsys):
    result = r.print_survfit(fits[case["fit"]], width=case["width"], **options(case["arguments"]))
    expected = case["expected"]
    assert result.table.colnames == expected["table"]["columns"]
    assert result.table.rownames == expected["table"]["rows"]
    np.testing.assert_allclose(
        result.table.values,
        np.asarray(expected["table"]["values"], dtype=float),
        rtol=2e-9,
        atol=1e-10,
    )
    assert result.lines == expected["lines"]
    assert str(result) == "\n".join(expected["lines"]) + "\n"
    assert capsys.readouterr().out == ""


def test_compact_report_does_not_expand_event_time_summaries(fits, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("compact reports must use native mean tables directly")

    monkeypatch.setattr(survival._survival, "summary_survfit", forbidden)
    assert r.print_survfit(fits["km"]).table.colnames[0] == "n"
    assert len(r.print_survfit(fits["cox"]).table.values) == 2


def test_reports_have_independent_data_and_default_precision(fits):
    fit = fits["groups"]
    original = pickle.dumps(fit)
    result = r.print_survfit(fit, rmean="common")
    result.table.values[0][0] = -1
    result.table.rownames[0] = "changed"
    result.rmean_endtime[0] = -1
    assert pickle.dumps(fit) == original
    assert r.print_survfit(fit).digits == 3
    assert r.print_survfit(fits["aj"]).digits == 7
    assert r.print_survfitms(fits["aj"], digits=3) == r.print_survfit(fits["aj"], digits=3)


def test_report_table_conversion_keeps_full_precision_and_copies_columns(fits):
    report = r.print_survfit(fits["groups"], rmean="common", digits=1)
    frame = r.as_data_frame(report)
    assert list(frame) == ["curve", *report.table.colnames]
    assert frame["curve"] == report.table.rownames
    assert frame["se(rmean)"] == [row[3] for row in report.table.values]
    assert any(value != round(value, 1) for value in frame["se(rmean)"])
    frame["curve"][0] = "edited"
    frame["se(rmean)"][0] = -1
    assert report.table.rownames[0] != "edited"
    assert report.table.values[0][3] > 0
    assert "curve" not in r.as_data_frame(r.print_survfit(fits["km"]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"digits": 0}, "digits"),
        ({"digits": 23}, "digits"),
        ({"width": 9}, "width"),
        ({"width": 10001}, "width"),
        ({"scale": 0}, "scale"),
        ({"scale": -1}, "scale"),
        ({"rmean": "unknown"}, "rmean"),
        ({"rmean": True}, "rmean"),
        ({"rmean": 0}, "Truncation"),
        ({"print_rmean": "yes"}, "print_rmean"),
    ],
)
def test_invalid_report_options(fits, kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        r.print_survfit(fits["km"], **kwargs)


def test_invalid_report_objects(fits):
    with pytest.raises(TypeError, match="fitted survival curve"):
        r.print_survfit(None)
    with pytest.raises(TypeError, match="multistate"):
        r.print_survfitms(fits["km"])
    with pytest.raises(ValueError, match="Truncation"):
        r.print_survfitms(fits["aj_conditional"], rmean=1.5)
    with pytest.raises(ValueError, match="rmean"):
        r.print_survfitms(fits["aj"], rmean=True)
