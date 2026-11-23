"""A response that is already a ``Surv`` or ``Surv2`` object: R's ``y ~ x`` idiom.

R's model frame reads a bare-name left-hand side like any variable, so a ``Surv`` column
works wherever ``Surv(...)`` does; each fit here must equal the same fit to the
``Surv(...)`` call on the same data (survival 3.8-12 gives identical results).
"""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
Factor = survival._factor.Factor


@pytest.fixture(scope="module")
def lung():
    data = {name: list(values) for name, values in datasets.load_lung().items()}
    data["y"] = r.Surv(data["time"], data["status"])
    data["my y"] = data["y"]
    return data


def test_fits_equal_the_surv_call(lung):
    reference = r.coxph("Surv(time, status) ~ age + sex", lung)
    for formula in ("y ~ age + sex", "`my y` ~ age + sex", "(y) ~ age + sex"):
        assert r.coxph(formula, lung).coefficients == reference.coefficients
    assert r.coxph("y ~ age + sex", lung).loglik == reference.loglik
    assert list(r.survfit("y ~ sex", lung).surv) == list(
        r.survfit("Surv(time, status) ~ sex", lung).surv
    )
    assert list(r.survreg("y ~ age", lung).coefficients) == list(
        r.survreg("Surv(time, status) ~ age", lung).coefficients
    )
    assert r.survdiff("y ~ sex", lung).chisq == r.survdiff("Surv(time, status) ~ sex", lung).chisq
    assert (
        r.concordance("y ~ age", lung).concordance
        == r.concordance("Surv(time, status) ~ age", lung).concordance
    )
    assert list(r.pyears("y ~ sex", lung).pyears) == list(
        r.pyears("Surv(time, status) ~ sex", lung).pyears
    )
    assert (
        r.cox_zph(r.coxph("y ~ age + sex", lung)).table
        == r.cox_zph(r.coxph("Surv(time, status) ~ age + sex", lung)).table
    )


def test_subset_and_missing_values_follow_the_response(lung):
    rows = [index < 100 for index in range(len(lung["time"]))]
    assert (
        r.coxph("y ~ age", lung, subset=rows).coefficients
        == r.coxph("Surv(time, status) ~ age", lung, subset=rows).coefficients
    )
    time = [math.nan if index == 2 else value for index, value in enumerate(lung["time"])]
    data = {**lung, "time": time, "y": r.Surv(time, lung["status"])}
    fit = r.coxph("y ~ age + ph.ecog", data, na_action="na.exclude")
    reference = r.coxph("Surv(time, status) ~ age + ph.ecog", data, na_action="na.exclude")
    assert fit.na_action == reference.na_action
    assert fit.coefficients == reference.coefficients


def test_model_frames_name_the_response(lung):
    fit = r.coxph("y ~ age", lung, model=True)
    assert list(fit.model) == ["y", "age"]


def test_a_numeric_bare_name_is_not_a_survival_response(lung):
    for fitter in (r.coxph, r.survreg, r.survfit):
        with pytest.raises(ValueError, match="response must be a survival object"):
            fitter("age ~ sex", lung)
    # A numeric bare name keeps its meaning where one is allowed.
    assert r.pyears("time ~ sex", lung).pyears == r.pyears("time ~ sex", lung).pyears
    assert r.concordance("age ~ ph.karno", lung).concordance == pytest.approx(0.4329182566780478)


def test_multistate_counting_and_timeline_responses():
    mgus2 = {name: list(values) for name, values in datasets.load_mgus2().items()}
    etime = [
        p if s == 1 else f
        for p, f, s in zip(mgus2["ptime"], mgus2["futime"], mgus2["pstat"], strict=True)
    ]
    labels = [
        "pcm" if s == 1 else ("death" if d == 1 else "censor")
        for s, d in zip(mgus2["pstat"], mgus2["death"], strict=True)
    ]
    event = Factor(labels, ["censor", "pcm", "death"])
    data = {**mgus2, "etime": etime, "event": event, "ms": r.Surv(etime, event)}
    ids = list(range(len(etime)))
    assert (
        r.coxph("ms ~ age", data, id=ids).coefficients
        == r.coxph("Surv(etime, event) ~ age", data, id=ids).coefficients
    )

    heart = {name: list(values) for name, values in datasets.load_heart().items()}
    heart["yc"] = r.Surv(heart["start"], heart["stop"], heart["event"])
    assert (
        r.coxph("yc ~ age", heart).coefficients
        == r.coxph("Surv(start, stop, event) ~ age", heart).coefficients
    )

    timeline = {"id": [1, 1, 1, 2, 2], "tt": [0.0, 5, 10, 0, 8], "ee": [0, 0, 1, 0, 1]}
    timeline["x"] = [1.0, 2, 2, 3, 3]
    timeline["y2"] = r.Surv2(timeline["tt"], timeline["ee"])
    fit = r.coxph("y2 ~ x", timeline, id="id", model=True)
    assert fit.coefficients == r.coxph("Surv2(tt, ee) ~ x", timeline, id="id").coefficients
    assert list(fit.model) == ["y2", "x", "(id)"]


def test_data_preparation_stores_the_response_under_its_name(lung):
    # survSplit(y ~ age, data, cut = 100, id = "newid"): age newid y
    split = r.survSplit("y ~ age", lung, cut=[100], id="newid")
    assert list(split) == ["age", "newid", "y"]
    assert split["y"].type == "counting"
    reference = r.survSplit("Surv(time, status) ~ age", lung, cut=[100])
    assert list(split["y"].start) == reference["tstart"]
    assert list(split["y"].time) == reference["time"]
    with pytest.raises(ValueError, match="left hand side not recognized"):
        r.survSplit("(y) ~ age", lung, cut=[100])

    heart = {name: list(values) for name, values in datasets.load_heart().items()}
    heart["yc"] = r.Surv(heart["start"], heart["stop"], heart["event"])
    condensed = r.survcondense("yc ~ age", heart, id="id")
    assert list(condensed) == ["age", "id", "yc"]
    reference = r.survcondense("Surv(start, stop, event) ~ age", heart, id="id")
    assert list(condensed["yc"].start) == reference["start"]
    assert list(condensed["yc"].time) == reference["stop"]

    timeline = {"id": [1, 1, 1, 2, 2], "x": [1.0, 2, 2, 3, 3]}
    timeline["y2"] = r.Surv2([0.0, 5, 10, 0, 8], [0, 0, 1, 0, 1])
    assert list(r.survSplit("y2 ~ x", timeline, id="id", cut=[3])) == ["x", "(id)", "y2"]
    named = r.fromtimeline("y2 ~ x", timeline, id="id", yname=["a", "b", "c"])
    assert list(named) == ["x", "a", "b", "c"]
    with pytest.raises(ValueError, match="yname is required"):
        r.fromtimeline("y2 ~ x", timeline, id="id")


def test_specs_stay_textual_and_hashable():
    from survival.r._formula import _formula_response_spec, _response_spec

    assert _response_spec("y ~ age").surv is False
    stored = _formula_response_spec("y ~ age")
    assert stored.stored
    assert stored.name == "y"
    assert stored != _formula_response_spec("Surv(y) ~ age")
    assert hash(stored) == hash(_formula_response_spec("y ~ age"))
    assert _formula_response_spec("(y) ~ age").name == "(y)"
    assert _formula_response_spec("`my y` ~ age").name == "my y"


def test_individual_expected_survival_reads_a_stored_rate_table_response(lung):
    for method in ("individual.s", "individual.h"):
        for formula in ("time ~ 1", "y ~ 1", "Surv(time, status) ~ 1"):
            assert r.survexp(
                formula, lung, ratetable=r.coxph("y ~ age", lung), method=method
            ) == r.survexp(
                formula, lung, ratetable=r.coxph("Surv(time, status) ~ age", lung), method=method
            )


def test_survival_object_columns_are_carried_by_row(lung):
    # tmerge(b, b, id = id, death = event(time, status)) keeps y as a Surv, as R does.
    data = {
        "id": list(range(1, len(lung["time"]) + 1)),
        "age": lung["age"],
        "time": lung["time"],
        "status": lung["status"],
        "y": lung["y"],
    }
    merged = r.tmerge(data, data, "id", death=r.event("time", "status"))
    assert list(merged) == ["id", "age", "time", "status", "y", "tstart", "tstop", "death"]
    assert merged["y"].time == lung["y"].time
    again = r.tmerge(
        merged, data, "id", x=r.tdc([value / 2 for value in lung["time"]], lung["age"])
    )
    assert len(again["y"]) == len(again["id"]) == 2 * len(lung["time"])
    assert list(again["y"].time) == [value for value in lung["time"] for _ in range(2)]
    # A Surv column that is not the response is split with its rows.
    frame = {name: lung[name] for name in ("time", "status", "age")}
    frame["other"] = lung["y"]
    split = r.survSplit("Surv(time, status) ~ .", frame, cut=[100, 300])
    assert len(split["other"]) == len(split["tstart"])
    # as_data_frame keeps a Surv column whole.
    table = r.as_data_frame(r.survSplit("y ~ age", lung, cut=[100]))
    assert list(table) == ["age", "y"]
    assert table["y"].type == "counting"


def test_a_covariate_named_like_a_timeline_column_stays_in_the_model_frame(lung):
    data = {"time": lung["time"], "status": lung["status"], "(tstop)": lung["age"]}
    fit = r.coxph("Surv(time, status) ~ `(tstop)`", data)
    assert list(r.model_frame(fit)) == ["time", "status", "(tstop)"]


def test_an_unused_survival_object_column_of_newdata_is_carried_whole(lung):
    # R keeps a Surv column of newdata as one column of curve summaries.
    fit = r.coxph("Surv(time, status) ~ age", lung)
    rows = [0, 1, 2]
    newdata = {name: [lung[name][row] for row in rows] for name in ("time", "status", "age")}
    plain = dict(newdata)
    newdata["y"] = r.Surv(newdata["time"], newdata["status"])
    curves = r.survfit(fit, newdata=newdata)
    frame = r.summary_survfit(curves, times=[200], data_frame=True)
    reference = r.summary_survfit(r.survfit(fit, newdata=plain), times=[200], data_frame=True)
    # newdata's time collides with a summary column, so R suffixes every newdata name.
    assert list(frame) == [*reference, "y_"]
    assert frame["surv"] == reference["surv"]
    assert list(frame["y_"].time) == newdata["time"]
    multistate_data = {
        "age": [80.0, 70.0, 60.0, 50.0, 75.0, 65.0],
        "time": [5.0, 8.0, 3.0, 9.0, 4.0, 7.0],
        "id": [1, 2, 3, 4, 5, 6],
        "event": Factor(
            ["censor", "pcm", "death", "pcm", "censor", "death"], levels=["censor", "pcm", "death"]
        ),
    }
    multistate = r.coxph("Surv(time, event) ~ age", multistate_data, id="id")
    new = {"age": [60.0, 70.0]}
    with_surv = {**new, "y": r.Surv([1.0, 2.0], [1, 0])}
    curves = r.survfit(multistate, newdata=with_surv)
    np.testing.assert_array_equal(curves.pstate, r.survfit(multistate, newdata=new).pstate)
    frame = r.summary_survfit(curves, times=[5], data_frame=True)
    assert isinstance(frame["y"], survival.r.Surv)
    assert len(frame["y"]) == len(frame["pstate"])
    assert isinstance(curves.subset(data=[1]).newdata["y"], survival.r.Surv)
