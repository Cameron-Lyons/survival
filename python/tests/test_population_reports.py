"""Population and consistency reports against R text and independent snapshots."""

import copy
import dataclasses
import datetime
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
DIRECTORY = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((DIRECTORY / "population_report_reference.json").read_text())
JOINT = json.loads((DIRECTORY / "yates_joint_reference.json").read_text())


def options(arguments):
    return {key.replace(".", "_"): value for key, value in dict(arguments).items()}


@pytest.fixture(scope="module")
def objects():
    @lru_cache(None)
    def fit(name):
        spec = REFERENCE["objects"][name]
        if spec["kind"] == "yates":
            source = next(
                case for case in JOINT["cases"] if case["name"] == spec["arguments"]["reference"]
            )
            model = (
                r.coxph(source["formula"], source["data"])
                if source["cox"]
                else r.YatesModel(
                    source["formula"],
                    source["data"],
                    [math.nan if x is None else x for x in source["beta"]],
                    source["variance"],
                    sigma2=source["sigma2"],
                )
            )
            arguments = {
                key: source[key] for key in ("term", "levels", "population", "test", "method")
            }
            arguments.update(
                {key: value for key, value in spec["arguments"].items() if key != "reference"}
            )
            return r.yates(model, **arguments)
        data = copy.deepcopy(REFERENCE["datasets"][spec["data"]])
        if "year" in data:
            data["year"] = [datetime.date.fromisoformat(value) for value in data["year"]]
        kwargs = options(spec["arguments"])
        if "ratetable" in kwargs:
            kwargs["ratetable"] = getattr(r, kwargs["ratetable"].replace(".", "_"))()
        return getattr(r, spec["kind"])(spec["formula"], data, **kwargs)

    return fit


def snapshot(case):
    source = case["input"]
    if case["kind"] == "pyears":
        return r.PyearsResult(
            source["pyears"],
            None,
            source["offtable"],
            source["observations"],
            False,
            [],
            {},
            event=source["event"],
            data=source["data"],
            summary=source["summary"],
            na_action=r.NaAction(tuple(source["omit"]), "omit") if source["omit"] else None,
        )
    if case["kind"] == "survcheck":
        transitions = source["transitions"]
        events = source["events"]

        def problem(name):
            value = source["problems"].get(name)
            return (
                None
                if value is None
                else r.SurvCheckProblem(
                    np.atleast_1d(value["row"]).tolist(), np.atleast_1d(value["id"]).tolist()
                )
            )

        return r.SurvCheckResult(
            [],
            SimpleNamespace(
                from_states=transitions["rows"],
                to_states=transitions["columns"],
                counts=transitions["values"],
            ),
            None
            if events is None
            else SimpleNamespace(
                states=events["rows"], count=events["columns"], subjects=events["values"]
            ),
            SimpleNamespace(**source["flag"]),
            [],
            *source["n"],
            problem("overlap"),
            problem("gap"),
            problem("jump"),
            problem("teleport"),
            None,
            [],
            source["omit"],
        )
    tests = source["test"]
    rows = []
    for name, values in zip(tests["rows"], tests["values"], strict=True):
        fields = dict(zip(tests["columns"], values, strict=True))
        rows.append(
            SimpleNamespace(
                name=name,
                chisq=math.nan if fields["chisq"] is None else fields["chisq"],
                df=fields["df"],
                ss=(math.nan if fields["ss"] is None else fields["ss"]) if "ss" in fields else None,
            )
        )
    estimate = {
        name: [math.nan if value is None else value for value in values]
        for name, values in source["estimate"].items()
    }
    return r.YatesResult(estimate, rows, [], [], [])


def report(case, value):
    kwargs = options(case["arguments"])
    if case["kind"] != "pyears":
        kwargs["width"] = case["width"]
    return getattr(r, "print_" + case["kind"])(value, **kwargs)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_native_reports_match_r(case, objects, capsys):
    result = report(case, objects(case["object"]))
    assert result.lines == case["lines"]
    assert str(result) == "\n".join(case["lines"]) + "\n"
    assert capsys.readouterr().out == ""
    reference = report(case, snapshot(case))
    for name, table in reference.tables.items():
        np.testing.assert_allclose(result.tables[name].values, table.values, rtol=3e-8, atol=1e-10)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_snapshot_reports_match_r(case):
    assert report(case, snapshot(case)).lines == case["lines"]


def test_report_ownership_and_dataframe_levels(objects):
    value = objects("yates_pairwise")
    before = copy.deepcopy(value.estimate)
    result = r.print_yates(value)
    assert len(result.estimates["pmm"]) < len(result.tables["tests"].values)
    frame = r.as_data_frame(result)
    assert frame == before
    frame["a"][0] = "changed"
    result.tables["estimates"].values[0][0] = 999
    result.tables["tests"].values[0][0] = 999
    assert value.estimate == before
    assert result.estimates == before
    assert pickle.loads(pickle.dumps(result)) == result  # noqa: S301 - own test data


@pytest.mark.parametrize("function", ["print_yates", "print_pyears", "print_survcheck"])
def test_invalid_report_type(function):
    with pytest.raises(TypeError):
        getattr(r, function)(None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": -1},
        {"eps": float("inf")},
        {"dig_tst": 0},
        {"dig_tst": 23},
        {"width": 9},
        {"digits": 0},
    ],
)
def test_yates_options_validate(objects, kwargs):
    with pytest.raises(ValueError, match="eps|dig_tst|width|digits"):
        r.print_yates(objects("yates_cox"), **kwargs)


def test_pyears_summary_is_preserved_for_array_and_frame_layouts(objects):
    full = objects("survexp.us")
    omitted = objects("population_missing")
    assert "age ranges from 60 to 80" in full.summary
    assert "male: 2  female: 1" in omitted.summary
    assert omitted.na_action == r.NaAction((2,), "omit")
    assert r.print_pyears(omitted).tables["totals"].values[0][-1] == 3
    assert r.survexp_us().source == "survexp.us"


def test_pyears_custom_summary_does_not_require_a_final_newline(objects):
    value = dataclasses.replace(objects("plain"), summary="custom")
    assert "   customObservations in the data set: 4" in r.print_pyears(value).lines


def test_factor_response_wrappers_preserve_multistate_labels():
    import pandas as pd

    data = {
        "time": [1, 2, 3, 4],
        "event": pd.Categorical(
            ["well", "ill", "well", "ill"], categories=["censor", "well", "ill", "unused"]
        ),
    }
    converted = r.survcheck("Surv(time,as.factor(event))~1", data, id=[1, 2, 3, 4])
    assert converted.y.states == ("well", "ill", "unused")
    assert converted.y.clabel == "censor"
    dropped = r.survcheck("Surv(time,factor(event))~1", data, id=[1, 2, 3, 4])
    assert dropped.y.states == ("ill",)
    assert dropped.y.clabel == "well"
    nested = r.survcheck("Surv(time,I(as.factor(event)))~1", data, id=[1, 2, 3, 4])
    assert nested.y == converted.y


def test_built_in_match_summaries_do_not_appear_on_custom_tables():
    table = r.RateTable([1], ["age"], [["0"]], [[0.0]], [2], [0.01])
    result = r.pyears("time~1", {"time": [1, 2], "age": [20, 30]}, ratetable=table, scale=1)
    assert table.source is None
    assert result.summary is None
    assert r.print_pyears(result).statistics["pyears"] == 3


def test_report_transition_labels_follow_the_response_censor_level():
    import pandas as pd

    data = {
        "time": [1, 2, 3],
        "state": pd.Categorical(["not yet", "event", "not yet"], categories=["not yet", "event"]),
    }
    result = r.survcheck("Surv(time,state)~1", data, id=[1, 2, 3])
    assert result.transitions.to_states[-1] == "(not yet)"
    assert r.print_survcheck(result).tables["transitions"].colnames[-1] == "(not yet)"


def test_count_and_transition_report_copies_are_independent(objects):
    source = objects("multistate")
    result = r.print_survcheck(source)
    expected = source.transitions.counts
    frame = r.as_data_frame(result)
    frame["from"][0] = "changed"
    result.tables["transitions"].values[0][0] = 999
    assert source.transitions.counts == expected
    assert r.as_data_frame(r.print_pyears(objects("plain")))["observations"] == [4]
