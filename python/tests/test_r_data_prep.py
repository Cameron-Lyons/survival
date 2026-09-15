"""Data-preparation helpers against R 4.5 / survival 3.8.11 reference values."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api


def _base():
    return {"id": [1, 2, 3], "futime": [10, 20, 15], "death": [1, 0, 1]}


def _long():
    return {
        "id": [1, 1, 2, 2, 3, 9],
        "time": [2, 5, 3, 25, 4, 1],
        "lab": [1.1, None, 0.9, 1.2, 2.0, 3],
        "inf": [1, 0, 1, 1, 1, 0],
    }


# --- aeqSurv ------------------------------------------------------------------


def test_aeqsurv_snaps_near_ties_and_keeps_the_type():
    # R's aeqSurv(Surv(c(1, 1 + 1e-14, 2, 2 + 1e-9, 3), c(1, 1, 0, 1, 1)))
    right = r.aeqSurv(r.Surv([1, 1 + 1e-14, 2, 2 + 1e-9, 3], [1, 1, 0, 1, 1]))
    assert right.time == (1.0, 1.0, 2.0, 2.0, 3.0)
    assert right.event == (1, 1, 0, 1, 1)
    # R's aeqSurv(Surv(c(0, 1e-14, 1, 1.5), c(1, 1 + 1e-14, 2, 2.5), c(1, 1, 0, 1)))
    counting = r.aeqSurv(r.Surv([0, 1e-14, 1, 1.5], [1, 1 + 1e-14, 2, 2.5], [1, 1, 0, 1]))
    assert counting.start == (0.0, 0.0, 1.0, 1.5)
    assert counting.time == (1.0, 1.0, 2.0, 2.5)
    assert counting.type == "counting"
    untouched = r.Surv([1, 2, 3], [1, 0, 1])
    assert r.aeqSurv(untouched).time == untouched.time
    assert r.aeqSurv(untouched, tolerance=0) is untouched
    with pytest.raises(ValueError, match="an interval has effective length 0"):
        r.aeqSurv(r.Surv([0, 1], [1, 1 + 1e-14], [1, 1]))
    with pytest.raises(ValueError, match="invalid value for tolerance"):
        r.aeqSurv(untouched, tolerance="big")
    with pytest.raises(TypeError, match="argument is not a Surv object"):
        r.aeqSurv([1, 2, 3])


# --- survSplit ----------------------------------------------------------------


def test_survsplit_dot_formula_keeps_the_data_columns_and_names():
    # survSplit(Surv(futime, death) ~ ., base, cut = c(5, 12), episode = "ep", added = "added")
    frame = r.survSplit(
        "Surv(futime, death) ~ .", _base(), cut=[5, 12], episode="ep", added="added"
    )
    assert list(frame) == ["id", "futime", "death", "tstart", "ep", "added"]
    assert frame["id"] == [1, 1, 2, 2, 2, 3, 3, 3]
    assert frame["futime"] == [5.0, 10.0, 5.0, 12.0, 20.0, 5.0, 12.0, 15.0]
    assert frame["death"] == [0, 1, 0, 0, 0, 0, 0, 1]
    assert frame["tstart"] == [0.0, 5.0, 0.0, 5.0, 12.0, 0.0, 5.0, 12.0]
    assert frame["ep"] == [1, 2, 1, 2, 3, 1, 2, 3]
    assert frame["added"] == [True, False, True, True, False, True, True, False]


def test_survsplit_names_counting_columns_after_the_surv_arguments():
    data = {"id": [1, 1, 2], "t1": [0, 2, 0], "t2": [2, 5, 3], "s": [0, 1, 0], "x": [1, 1, 3]}
    frame = r.survSplit("Surv(t1, t2, s) ~ x", data, cut=[1, 4], episode="epi")
    assert list(frame) == ["x", "t1", "t2", "s", "epi"]
    assert frame["t1"] == [0.0, 1.0, 2.0, 4.0, 0.0, 1.0]
    assert frame["t2"] == [1.0, 2.0, 4.0, 5.0, 1.0, 3.0]
    assert frame["s"] == [0, 0, 0, 1, 0, 0]
    renamed = r.survSplit(
        "Surv(t1, t2, s) ~ x", data, cut=[4], start="a", end="b", event="c", id="subject"
    )
    assert list(renamed) == ["x", "a", "b", "c"]
    with pytest.raises(ValueError, match="cut must be a vector of finite numbers"):
        r.survSplit("Surv(t1, t2, s) ~ x", data, cut=[math.inf])


def test_survsplit_multistate_response_labels_states_and_skips_the_id():
    data = {
        "id": [1, 1, 2],
        "t": [1, 4, 2],
        "s": survival.r._coerce._RFactorVector(["a", "b", "b"], ["censor", "a", "b"]),
    }
    # survSplit(Surv(t, s) ~ id, d, cut = 3, episode = "e", zero = -1)
    frame = r.survSplit("Surv(t, s) ~ id", data, cut=3, episode="e", zero=-1)
    assert list(frame) == ["id", "tstart", "t", "s", "e"]
    assert frame["tstart"] == [-1.0, -1.0, 3.0, -1.0]
    assert frame["t"] == [1.0, 3.0, 4.0, 2.0]
    assert frame["s"] == ["a", "censor", "b", "b"]
    assert frame["e"] == [1, 1, 2, 1]
    # R invents the id column for right-censored (time, status) data only
    with_id = r.survSplit("Surv(t, s) ~ id", data, cut=[3], id="row")
    assert "row" not in with_id
    right = r.survSplit("Surv(t, e) ~ id", {**data, "e": [1, 1, 0]}, cut=[3], id="row")
    assert right["row"] == [1, 2, 2, 3]
    with pytest.raises(ValueError, match="'zero' parameter must be less than any observed times"):
        r.survSplit("Surv(t, s) ~ id", data, cut=[3], zero=2)


def test_survsplit_accepts_a_surv_object_with_covariates():
    surv = r.Surv([0, 2, 0], [2, 5, 3], [0, 1, 0])
    frame = r.survSplit(surv, {"x": [1, 1, 3]}, cut=[4], episode="epi")
    assert frame["x"] == [1, 1, 1, 3]
    assert frame["tstart"] == [0.0, 2.0, 4.0, 0.0]
    assert frame["tstop"] == [2.0, 4.0, 5.0, 3.0]
    assert frame["event"] == [0, 0, 1, 0]
    assert frame["epi"] == [1, 1, 2, 1]
    same = r.survSplit(response=surv, data={"x": [1, 1, 3]}, cut=[4])
    assert same["tstop"] == frame["tstop"]
    with pytest.raises(ValueError, match="not valid for left censored survival data"):
        r.survSplit(r.Surv([1, 2], [1, 0], type="left"), cut=[1.5])


# --- survcondense -------------------------------------------------------------


def test_survcondense_merges_adjacent_rows_with_equal_covariates():
    data = {
        "id": [1, 1, 1, 2, 2],
        "t1": [0, 2, 5, 0, 3],
        "t2": [2, 5, 8, 3, 6],
        "s": [0, 0, 1, 0, 0],
        "x": [1, 1, 2, 3, 3],
    }
    # survcondense(Surv(t1, t2, s) ~ x, d, id = id)
    frame = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id")
    assert list(frame) == ["x", "id", "t1", "t2", "s"]
    assert frame["x"] == [1, 2, 3]
    assert frame["id"] == [1, 1, 2]
    assert frame["t1"] == [0.0, 5.0, 0.0]
    assert frame["t2"] == [5.0, 8.0, 6.0]
    assert frame["s"] == [0, 1, 0]
    renamed = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id", start="a", end="b", event="e")
    assert list(renamed) == ["x", "id", "a", "b", "e"]
    weighted = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id", weights=[1, 2, 1, 1, 1])
    assert weighted["t1"] == [0.0, 2.0, 5.0, 0.0]
    assert list(weighted)[:3] == ["x", "(weights)", "id"]
    with pytest.raises(ValueError, match="id is required"):
        r.survcondense("Surv(t1, t2, s) ~ x", data, id=None)
    with pytest.raises(ValueError, match="invalid survival type"):
        r.survcondense("Surv(t2, s) ~ x", data, id="id")
    with pytest.raises(ValueError, match="does not handle cluster"):
        r.survcondense("Surv(t1, t2, s) ~ x + cluster(id)", data, id="id")


# --- rttright -----------------------------------------------------------------


def test_rttright_matches_r_weights_and_time_matrix():
    aml = {k: v for k, v in survival.datasets.load_aml().items() if not k.startswith("_")}
    first_six = {name: list(values[:6]) for name, values in aml.items()}
    # rttright(Surv(time, status) ~ 1, aml[1:6, ])
    weights = r.rttright("Surv(time, status) ~ 1", first_six)
    assert weights == pytest.approx([1 / 6, 1 / 6, 0, 2 / 9, 2 / 9, 0])
    # rttright(Surv(time, status) ~ x, aml, times = c(10, 30))
    matrix = r.rttright("Surv(time, status) ~ x", aml, times=[10, 30])
    assert len(matrix) == 23
    assert len(matrix[0]) == 2
    assert matrix[0] == pytest.approx([0.09090909, 0.09090909], rel=1e-6)
    assert matrix[2] == pytest.approx([0.09090909, 0.0], rel=1e-6)
    assert matrix[3] == pytest.approx([0.09090909, 0.10227273], rel=1e-6)
    single = r.rttright("Surv(time, status) ~ x", aml, times=10)
    assert single == pytest.approx([row[0] for row in matrix])
    with pytest.raises(ValueError, match="Interaction terms are not valid"):
        r.rttright("Surv(time, status) ~ x:time", aml)
    with pytest.raises(ValueError, match="response must be right censored"):
        r.rttright("Surv(time, status, type='left') ~ 1", first_six)
    with pytest.raises(ValueError, match="weights must be non-negative"):
        r.rttright("Surv(time, status) ~ 1", first_six, weights=[-1, 1, 1, 1, 1, 1])


# --- lvcf, nostutter, neardate, tcut ---------------------------------------------


def test_lvcf_carries_values_forward_like_r():
    # R's lvcf(c(1,1,1,2,2,2), c(NA, 3, NA, NA, NA, 5))
    assert r.lvcf([1, 1, 1, 2, 2, 2], [None, 3, None, None, None, 5]) == [None, 3, 3, None, None, 5]
    # logical and 0/1 variables get FALSE/0 on a missing first observation
    assert r.lvcf([1, 1, 1, 2, 2, 2], [None, True, None, None, None, False]) == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert r.lvcf([1, 1, 2], [None, 1, None]) == [0, 1, 0]
    assert r.lvcf([1, 1, 2], [None, 1, None], first=False) == [None, 1, None]
    # R's lvcf(c(2,2,1,1), c(NA, 4, 7, NA), time = c(2,1,2,1))
    assert r.lvcf([2, 2, 1, 1], [None, 4, 7, None], time=[2, 1, 2, 1]) == [4, 4, 7, None]
    assert r.lvcf(["b", "b", "a"], ["x", None, None]) == ["x", "x", None]
    with pytest.raises(ValueError, match="x must have the same length as id"):
        r.lvcf([1, 2], [1])


def test_nostutter_censors_repeated_states_like_r():
    # R's nostutter(c(1,1,1,1,2,2), c(0,"a","a","b","b","b"))
    assert r.nostutter([1, 1, 1, 1, 2, 2], [0, "a", "a", "b", "b", "b"]) == [0, "a", 0, "b", "b", 0]
    assert r.nostutter([1, 1, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]) == [0, 1, 0, 2, 2, 0]
    assert r.nostutter([1, 1, 2, 2], ["a", "a", "a", "b"], censor="none") == ["a", "none", "a", "b"]
    with pytest.raises(ValueError, match="wrong length for x or id"):
        r.nostutter([1, 2], [1])


def test_neardate_returns_zero_based_rows_like_r_minus_one():
    id1 = [1, 1, 2, 2, 2, 3, 4, 4, 5]
    y1 = [10, 20, 5, 15, 25, 30, 4, 12, 7]
    id2 = [1, 1, 1, 2, 2, 3, 4, 4, 4, 6]
    y2 = [8, 12, 22, 4, 26, 30, 3, 11, 13, 1]
    # neardate(id1, id2, y1, y2): 2 3 5 5 5 6 8 9 NA (one based)
    assert r.neardate(id1, id2, y1, y2) == [1, 2, 4, 4, 4, 5, 7, 8, None]
    assert r.neardate(id1, id2, y1, y2, best="prior") == [0, 1, 3, 3, 3, 5, 6, 7, None]
    assert r.neardate(id1, id2, y1, y2, nomatch=-1)[-1] == -1
    # neardate(c(1,2,2), c(2,2,1), c(5, 3, 9), c(4, 8, 6), best = "prior"): NA NA 2
    assert r.neardate([1, 2, 2], [2, 2, 1], [5, 3, 9], [4, 8, 6], best="prior") == [None, None, 1]
    assert r.neardate([1, None], [1], [1, 2], [1])[1] is None
    with pytest.raises(ValueError, match="id1 and y1 have different lengths"):
        r.neardate([1, 2], [1], [1], [1])


def test_tcut_scales_values_and_cutpoints_like_r():
    # R's tcut(c(10, 25, 40), c(0, 20, 50), scale = 2)
    cut = r.tcut([10, 25, 40], [0, 20, 50], scale=2)
    assert list(cut.values) == [20.0, 50.0, 80.0]
    assert list(cut.cutpoints) == [0.0, 40.0, 100.0]
    assert list(cut.labels) == [" 0+ thru 20", "20+ thru 50"]
    assert list(r.tcut([1, 2], [0, 5], labels=["young"]).labels) == ["young"]
    assert len(r.tcut([1, 2, 3], 3).cutpoints) == 4
    with pytest.raises(ValueError, match="breaks must be given in ascending order"):
        r.tcut([1, 2], [5, 0])
    with pytest.raises(ValueError, match="Number of labels must be 1 less"):
        r.tcut([1, 2], [0, 5], labels=["a", "b"])


# --- tmerge -------------------------------------------------------------------


def test_tmerge_arguments_match_r_values_and_tcount():
    base, long = _base(), _long()
    d1 = r.tmerge(base, base, id="id", death=r.event("futime", "death"))
    assert list(d1) == ["id", "futime", "death", "tstart", "tstop"]
    assert d1["death"] == [1, 0, 1]
    d2 = r.tmerge(
        d1,
        long,
        id="id",
        lab=r.tdc("time", "lab"),
        n=r.cumtdc("time"),
        inf=r.event("time", "inf"),
        ninf=r.cumevent("time", "inf"),
    )
    assert d2["id"] == [1, 1, 1, 2, 2, 3, 3]
    assert d2["tstart"] == [0.0, 2.0, 5.0, 0.0, 3.0, 0.0, 4.0]
    assert d2["tstop"] == [2.0, 5.0, 10.0, 3.0, 20.0, 4.0, 15.0]
    assert d2["death"] == [0, 0, 1, 0, 0, 0, 1]
    assert [None if math.isnan(v) else v for v in d2["lab"]] == [
        None,
        1.1,
        1.1,
        None,
        0.9,
        None,
        2.0,
    ]
    assert d2["n"] == [0, 1, 2, 0, 1, 0, 1]
    assert d2["inf"] == [1, 0, 0, 1, 0, 1, 0]
    assert d2["ninf"] == [1, 0, 0, 1, 0, 1, 0]
    assert d2.tevent == {"death": 0, "inf": 0, "ninf": 0}
    assert d2.tdcvar == ("lab", "n")
    counts = d2.tcount
    assert counts["death"]["trailing"] == 3
    assert (counts["lab"]["within"], counts["lab"]["late"], counts["lab"]["missid"]) == (3, 1, 1)
    assert (counts["n"]["boundary"], counts["n"]["within"]) == (3, 1)
    assert counts["inf"]["boundary"] == 4


def test_tmerge_options_delay_na_rm_and_names():
    base, long = _base(), _long()
    d1 = r.tmerge(base, base, id="id", death=r.event("futime", "death"))
    # options = list(delay = 1, na.rm = FALSE): change times move by one day, NA values stay
    d3 = r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"), options={"delay": 1, "na.rm": False})
    assert d3["tstop"] == [3.0, 6.0, 10.0, 4.0, 20.0, 5.0, 15.0]
    assert [None if math.isnan(v) else v for v in d3["lab"]] == [
        None,
        1.1,
        None,
        None,
        0.9,
        None,
        2.0,
    ]
    # tstart = 2 with renamed interval columns
    d4 = r.tmerge(
        base, base, id="id", tstop="futime", tstart=2, options={"tstartname": "a", "tstopname": "b"}
    )
    assert list(d4) == ["id", "futime", "death", "a", "b"]
    assert d4["a"] == [2.0, 2.0, 2.0]
    assert d4.tname == {"idname": "id", "tstartname": "a", "tstopname": "b"}
    init = r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab", init=0.5))["lab"][0]
    assert init == 0.5
    with_start = r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"), options={"tdcstart": -1})
    assert with_start["lab"][0] == -1


def test_tmerge_reports_r_errors():
    base, long = _base(), _long()
    with pytest.raises(ValueError, match="data2 has id values not in data1"):
        r.tmerge(base, long, id="id", death=r.event("time"))
    d1 = r.tmerge(base, base, id="id", death=r.event("futime", "death"))
    with pytest.raises(ValueError, match="attempt to turn event variable death into a tdc"):
        r.tmerge(d1, long, id="id", death=r.tdc("time"))
    with pytest.raises(
        ValueError, match="attempt to turn time-dependent covariate lab into an event"
    ):
        r.tmerge(
            r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab")),
            long,
            id="id",
            lab=r.event("time"),
        )
    with pytest.raises(ValueError, match="tstart and tstop arguments only apply to the first call"):
        r.tmerge(d1, long, id="id", tstop="time", lab=r.tdc("time"))
    with pytest.raises(ValueError, match="not a recognized type"):
        r.tmerge(base, base, id="id", death=[1, 0, 1])
    with pytest.raises(ValueError, match="unrecognized option"):
        r.tmerge(base, base, id="id", tstop="futime", options={"bogus": 1})
    duplicated = {"id": [1, 1], "futime": [5, 6]}
    with pytest.raises(ValueError, match="must have no duplicate identifiers"):
        r.tmerge(duplicated, duplicated, id="id", tstop="futime")
    with pytest.raises(ValueError, match="tstart must be < tstop"):
        r.tmerge(base, base, id="id", tstop="futime", tstart=10)


def test_tmerge_accepts_bridge_style_operations_and_metadata():
    base, long = _base(), _long()
    d1 = r.tmerge(base, base, id="id", tstop="futime")
    operations = {"lab": {"kind": "tdc", "time": long["time"], "value": long["lab"]}}
    metadata = {"tname": d1.tname, "tevent": {}, "tdcvar": []}
    plain = r.tmerge(
        dict(d1.columns), long, id=long["id"], operations=operations, metadata=metadata
    )
    assert plain["lab"][1] == 1.1
    assert plain.tdcvar == ("lab",)
