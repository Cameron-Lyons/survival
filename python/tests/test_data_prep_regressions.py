"""Data-preparation regressions against R 4.5.3 / survival 3.8-12 (values hard-coded)."""

import importlib
import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_coerce = importlib.import_module("survival.r._coerce")
r_data_prep = importlib.import_module("survival.r._data_prep")


def _na(values):
    """``NaN`` (the package's numeric NA) as ``None``, for comparing with R's ``NA``."""

    return [None if isinstance(v, float) and math.isnan(v) else v for v in values]


# --- survcondense -------------------------------------------------------------


def test_survcondense_compares_covariates_with_r_equality():
    # survcondense(Surv(t1, t2, s) ~ x, d, id = id) with x = c(0.3, 0.1 + 0.2, 0.1 + 0.2):
    # 0.3 != 0.1 + 0.2, so R keeps (0, 1] and (1, 3]
    data = {"id": [1, 1, 1], "t1": [0, 1, 2], "t2": [1, 2, 3], "s": [0, 0, 1]}
    data["x"] = [0.3, 0.1 + 0.2, 0.1 + 0.2]
    frame = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id")
    assert frame["x"] == [0.3, 0.1 + 0.2]
    assert frame["t1"] == [0.0, 1.0]
    assert frame["t2"] == [1.0, 3.0]
    assert frame["s"] == [0, 1]


def test_survcondense_reads_the_kernel_start_times_once(monkeypatch):
    # the start column is gathered from one read of the kernel's result, not one per row
    reads = []
    kernel = r_data_prep._core.survcondense

    class Counting:
        def __init__(self, result):
            self.keep = result.keep
            self._start = result.start

        @property
        def start(self):
            reads.append(1)
            return self._start

    monkeypatch.setattr(
        r_data_prep._core,
        "survcondense",
        lambda *args: Counting(kernel(*args)),
        raising=False,
    )
    data = {
        "id": [1, 1, 1, 2, 2, 3],
        "t1": [0, 2, 5, 0, 3, 0],
        "t2": [2, 5, 8, 3, 6, 4],
        "s": [0, 0, 1, 0, 0, 1],
        "x": [1, 2, 3, 4, 5, 6],
    }
    frame = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id")
    assert frame["t1"] == [0.0, 2.0, 5.0, 0.0, 3.0, 0.0]
    assert len(reads) == 1


# --- survSplit ----------------------------------------------------------------


def _old_style_data():
    return {"time": [5, 8], "status": [1, 0], "x": [10, 20]}


def test_survsplit_old_style_call_builds_the_formula_like_r():
    # R's survSplit(d, cut = 3, end = "time", event = "status")
    frame = r.survSplit(_old_style_data(), cut=3, end="time", event="status")
    assert list(frame) == ["time", "status", "x", "tstart"]
    assert frame["time"] == [3.0, 5.0, 3.0, 8.0]
    assert frame["status"] == [0, 1, 0, 0]
    assert frame["x"] == [10, 10, 20, 20]
    assert frame["tstart"] == [0.0, 3.0, 0.0, 3.0]
    # R's survSplit(data = d, cut = 3, end = "time", event = "status", episode = "ep",
    #              id = "subj")
    frame = r.survSplit(
        data=_old_style_data(), cut=3, end="time", event="status", episode="ep", id="subj"
    )
    assert list(frame) == ["time", "status", "x", "subj", "tstart", "ep"]
    assert frame["subj"] == [1, 1, 2, 2]
    assert frame["ep"] == [1, 2, 1, 2]
    # a start column in the data makes it Surv(tstart, time, status)
    counting = {"tstart": [0, 1], "time": [5, 8], "status": [1, 0]}
    frame = r.survSplit(counting, cut=3, end="time", event="status")
    assert list(frame) == ["tstart", "time", "status"]
    assert frame["tstart"] == [0.0, 3.0, 1.0, 3.0]
    assert frame["time"] == [3.0, 5.0, 3.0, 8.0]
    renamed = {"a": [0, 1], "time": [5, 8], "status": [1, 0]}
    frame = r.survSplit(renamed, cut=3, end="time", event="status", start="a")
    assert frame["a"] == [0.0, 3.0, 1.0, 3.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"end": "time"}, "either a formula or the end and event arguments are required"),
        ({"end": "time", "event": "stat"}, "'event' must be a variable name in the data set"),
        ({"end": "tim", "event": "status"}, "'end' must be a variable name in the data set"),
        ({"end": "time", "event": "status", "start": 1}, "'start' must be a variable name"),
    ],
)
def test_survsplit_old_style_call_checks_its_arguments_like_r(kwargs, message):
    with pytest.raises(ValueError, match=message):
        r.survSplit(_old_style_data(), cut=3, **kwargs)


def test_survsplit_old_style_call_requires_data():
    with pytest.raises(ValueError, match="a data frame is required"):
        r.survSplit(cut=3, end="time", event="status")


def test_survsplit_puts_a_missing_row_in_the_second_episode():
    # survSplit(Surv(time, status) ~ ., d, cut = 3, episode = "ep"): survsplit.c gives the
    # NA row interval 1 (episode 2); its status is uninitialised in R and kept here
    data = {"time": [5, None, 8], "status": [1, 1, 0]}
    frame = r.survSplit("Surv(time, status) ~ .", data, cut=3, episode="ep")
    assert _na(frame["time"]) == [3.0, 5.0, None, 3.0, 8.0]
    assert frame["tstart"] == [0.0, 3.0, 0.0, 0.0, 3.0]
    assert frame["ep"] == [1, 2, 2, 1, 2]
    assert frame["status"] == [0, 1, 1, 0, 0]


# --- aeqSurv ------------------------------------------------------------------


def test_aeqsurv_snaps_a_surv2_time_column():
    # aeqSurv(Surv2(c(1, 1 + 1e-12, 3), c(0, 1, 1))): times 1 1 3, status kept
    fixed = r.aeqSurv(r.Surv2([1, 1 + 1e-12, 3], [0, 1, 1]))
    assert isinstance(fixed, r.Surv2)
    assert fixed.time == (1.0, 1.0, 3.0)
    assert fixed.status == (0, 1, 1)
    assert fixed.repeated is False
    states = r_coerce._RFactorVector(["a", "b", "a"], ["a", "b"])
    multi = r.Surv2([1, 1 + 1e-12, 3], states, repeated=True)
    fixed = r.aeqSurv(multi)
    assert fixed.time == (1.0, 1.0, 3.0)
    assert (fixed.status, fixed.states, fixed.repeated) == (
        multi.status,
        multi.states,
        multi.repeated,
    )


# --- tmerge -------------------------------------------------------------------


def _tmerge_base():
    base = {"id": [1, 2, 3], "futime": [10.0, 20.0, 15.0]}
    return r.tmerge(base, base, id="id", tstop="futime")


def _tmerge_updates():
    return {
        "id": [1, 1, 2, 3, 3],
        "t": [2.0, 5.0, 3.0, 4.0, 30.0],
        "status": [1, 0, 1, 0, 1],
        "flag": [True, False, True, False, True],
        "ilab": [1, 2, None, 3, 4],
        "dlab": [1.5, 2, None, 3, 4],
        "clab": ["a", "b", "a", "b", "c"],
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        # tmerge(d1, u, id = id, ev = event(t, stauts)): object 'stauts' not found
        ({"ev": ("t", "stauts")}, "object 'stauts' not found"),
        ({"ev": ("tt", None)}, "object 'tt' not found"),
        # event(t, 1) and event(5): argument ev is not the same length as id
        ({"ev": ("t", 1)}, "argument ev is not the same length as id"),
        ({"ev": (5.0, None)}, "argument ev is not the same length as id"),
        ({"ev": ("t", [1])}, "argument ev is not the same length as id"),
    ],
)
def test_tmerge_evaluates_arguments_in_data2_like_r(arguments, message):
    events = {name: r.event(time, value) for name, (time, value) in arguments.items()}
    with pytest.raises(ValueError, match=message):
        r.tmerge(_tmerge_base(), _tmerge_updates(), id="id", **events)


def test_tmerge_recycles_only_tstart():
    base = {"id": [1, 2, 3], "futime": [10.0, 20.0, 15.0]}
    # tmerge(d, d, id = id, tstop = 10): tstop and id must be the same length
    with pytest.raises(ValueError, match="tstop and id must be the same length"):
        r.tmerge(base, base, id="id", tstop=10.0)
    with pytest.raises(ValueError, match="object 'futim' not found"):
        r.tmerge(base, base, id="id", tstop="futim")
    with pytest.raises(ValueError, match="tstart and id must be the same length"):
        r.tmerge(base, base, id="id", tstop="futime", tstart=[1.0, 2.0])
    # R's tmerge(d, d, id = id, tstop = futime, tstart = 2)
    frame = r.tmerge(base, base, id="id", tstop="futime", tstart=2)
    assert frame["tstart"] == [2.0, 2.0, 2.0]
    assert frame["tstop"] == [10.0, 20.0, 15.0]


def test_tmerge_keeps_the_type_of_the_values_like_r():
    d1, updates = _tmerge_base(), _tmerge_updates()
    # event(t, flag): logi TRUE FALSE FALSE TRUE FALSE FALSE FALSE, censor FALSE
    frame = r.tmerge(d1, updates, id="id", ev=r.event("t", "flag"))
    assert frame["tstart"] == [0.0, 2.0, 5.0, 0.0, 3.0, 0.0, 4.0]
    assert frame["ev"] == [True, False, False, True, False, False, False]
    assert all(type(value) is bool for value in frame["ev"])
    assert frame.tevent == {"ev": False}
    # tdc(t, ilab): int NA 1 2 NA NA 3
    frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "ilab"))
    assert _na(frame["lab"]) == [None, 1, 2, None, None, 3]
    assert all(type(value) is int for value in _na(frame["lab"]) if value is not None)
    # tdc(t, clab) with tdcstart = -1, and with init = 0: chr "-1" "a" ... / "0" "a" ...
    frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "clab"), options={"tdcstart": -1})
    assert frame["lab"] == ["-1", "a", "b", "-1", "a", "-1", "b"]
    frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "clab", init=0))
    assert frame["lab"] == ["0", "a", "b", "0", "a", "0", "b"]
    # event(t, clab): chr "a" "b" "" "a" "" "b" "", censor ""
    frame = r.tmerge(d1, updates, id="id", ev=r.event("t", "clab"))
    assert frame["ev"] == ["a", "b", "", "a", "", "b", ""]
    assert frame.tevent == {"ev": ""}
    # event(t): int 1 1 0 1 0 1 0, censor 0L
    frame = r.tmerge(d1, updates, id="id", ev=r.event("t"))
    assert frame["ev"] == [1, 1, 0, 1, 0, 1, 0]
    assert all(type(value) is int for value in frame["ev"])
    # cumevent(t): int 1 2 0 1 0 1 0; cumevent(t, flag): num 1 0 0 1 0 0 0, censor 0
    frame = r.tmerge(d1, updates, id="id", n=r.cumevent("t"))
    assert frame["n"] == [1, 2, 0, 1, 0, 1, 0]
    assert all(type(value) is int for value in frame["n"])
    assert frame.tevent == {"n": 0}
    frame = r.tmerge(d1, updates, id="id", n=r.cumevent("t", "flag"))
    assert frame["n"] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert all(type(value) is float for value in frame["n"])
    assert frame.tevent == {"n": 0.0}
    assert type(frame.tevent["n"]) is float


@pytest.mark.parametrize(
    ("column", "init", "expected"),
    [
        # tdc(t, ilab, 0L) and tdc(t, ilab, TRUE): num 0 1 2 0 0 3 / num 1 1 2 1 1 3
        ("ilab", 0, [0.0, 1.0, 2.0, 0.0, 0.0, 3.0]),
        ("ilab", True, [1.0, 1.0, 2.0, 1.0, 1.0, 3.0]),
        # tdc(t, dlab, 0L): num 0 1.5 2 0 0 3
        ("dlab", 0, [0.0, 1.5, 2.0, 0.0, 0.0, 3.0]),
        # tdc(t, flag, FALSE / 0L / 0 / "no"): logi, int, num and chr
        ("flag", False, [False, True, False, False, True, False, False]),
        ("flag", 0, [0, 1, 0, 0, 1, 0, 0]),
        ("flag", 0.0, [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ("flag", "no", ["no", "TRUE", "FALSE", "no", "TRUE", "no", "FALSE"]),
        # tdc(t, clab, TRUE): chr "TRUE" "a" "b" "TRUE" "a" "TRUE" "b"
        ("clab", True, ["TRUE", "a", "b", "TRUE", "a", "TRUE", "b"]),
    ],
)
def test_tmerge_tdc_default_converts_the_variable_like_r(column, init, expected):
    frame = r.tmerge(_tmerge_base(), _tmerge_updates(), id="id", lab=r.tdc("t", column, init))
    assert frame["lab"] == expected
    assert [type(value) for value in frame["lab"]] == [type(value) for value in expected]


def test_tmerge_numeric_tdc_takes_the_default_as_numeric_like_r():
    d1, updates = _tmerge_base(), _tmerge_updates()
    # tdc(t, ilab) with options(tdcstart = -1L): num -1 1 2 -1 -1 3
    frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "ilab"), options={"tdcstart": -1})
    assert frame["lab"] == [-1.0, 1.0, 2.0, -1.0, -1.0, 3.0]
    assert all(type(value) is float for value in frame["lab"])
    # tdc(t, ilab, "x"): num NA 1 2 NA NA 3, warning "NAs introduced by coercion"
    with pytest.warns(UserWarning, match="NAs introduced by coercion"):
        frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "ilab", "x"))
    assert _na(frame["lab"]) == [None, 1.0, 2.0, None, None, 3.0]
    # every interval starts at or after an update, so no row takes the default: int 4 5 6
    updates = {"id": [1, 2, 3], "t": [0.0, 0.0, 0.0], "ilab": [4, 5, 6]}
    frame = r.tmerge(d1, updates, id="id", lab=r.tdc("t", "ilab", 0.0))
    assert frame["lab"] == [4, 5, 6]
    assert all(type(value) is int for value in frame["lab"])


def test_tmerge_cumevent_with_a_missing_increment_like_r():
    d1 = _tmerge_base()
    options = {"na.rm": False}
    # an NA increment at an event time: R stops (NAs are not allowed in subscripted assignments)
    updates = {"id": [1, 1, 2, 3, 3], "t": [2.0, 5.0, 3.0, 4.0, 30.0], "ilab": [1, None, 2, 3, 4]}
    with pytest.raises(ValueError, match="argument n has a missing cumevent increment"):
        r.tmerge(d1, updates, id="id", n=r.cumevent("t", "ilab"), options=options)
    # an NA increment before follow-up is not an event, but the later counts are NA:
    # tstart 0 2 5 0 3 0, n int NA NA 0 3 0 0
    updates = {"id": [1, 1, 1, 2], "t": [-1.0, 2.0, 5.0, 3.0], "ilab": [None, 1, 2, 3]}
    frame = r.tmerge(d1, updates, id="id", n=r.cumevent("t", "ilab"), options=options)
    assert frame["tstart"] == [0.0, 2.0, 5.0, 0.0, 3.0, 0.0]
    assert _na(frame["n"]) == [None, None, 0, 3, 0, 0]
    assert all(type(value) is int for value in _na(frame["n"]) if value is not None)


# --- subject ids --------------------------------------------------------------


def test_ids_convert_from_python_and_numpy_scalars():
    ids = [np.int64(3), 3.0, np.float32(1.5), 1.5, "a", np.str_("a"), True, 1, 2**70]
    result = survival.data_prep.cluster(ids)
    assert list(result.codes) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    assert list(result.sizes) == [2, 2, 2, 2, 1]
    with pytest.raises(TypeError, match="an id must be an int, float or str, not NoneType"):
        survival.data_prep.cluster([1, None])


# --- rttright -----------------------------------------------------------------


def _rtt_right():
    return {
        "time": [1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8, 2, 4, 4, 9],
        "status": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "a", "b", "b"],
        "w": [1, 2, 1, 0.5, 1, 3, 2, 1, 1, 1, 2, 1, 0.25, 1, 2, 1],
    }


def test_rttright_right_censored_ties_weights_and_strata_match_r():
    data = _rtt_right()
    # rttright(Surv(time, status) ~ 1, d)
    assert r.rttright("Surv(time, status) ~ 1", data) == pytest.approx(
        [
            0.0625,
            0,
            0.0625,
            0,
            0.0677083333333333,
            0,
            0.0827546296296296,
            0.0965470679012346,
            0,
            0,
            0.160911779835391,
            0,
            0.0625,
            0,
            0.0827546296296296,
            0,
        ],
        rel=1e-13,
    )
    # rttright(Surv(time, status) ~ g, d, weights = w)
    a, b = 0.108108108108108, 0.216216216216216
    assert r.rttright("Surv(time, status) ~ g", data, weights="w") == pytest.approx(
        [a, 0, a, 0, a, 0, b, 1 / 6, 0, 0, 2 * b, 0, a / 4, 0, 1 / 3, 0], rel=1e-13
    )
    # ... renorm = FALSE
    assert r.rttright("Surv(time, status) ~ g", data, weights="w", renorm=False) == pytest.approx(
        [1, 0, 1, 0, 1, 0, 2, 1.91666666666667, 0, 0, 4, 0, 0.25, 0, 3.83333333333333, 0],
        rel=1e-13,
    )
    # ... times = c(2, 3, 4.5, 10), one column per time
    matrix = r.rttright("Surv(time, status) ~ g", data, weights="w", times=[2, 3, 4.5, 10])
    c, e, f = 0.173913043478261, 0.0869565217391304, 0.105263157894737
    expected = [
        [a, c, a, 0.0434782608695652, a, 0.260869565217391, b, e, a, e, b, e, a / 4, a, c, e],
        [a, 0, a, 0.0526315789473684, a, 0.315789473684211, b, f, a, f, b, f, a / 4, a, 2 * f, f],
        [a, 0, a, 0, a, 0, b, 1 / 6, 0.144144144144144, 1 / 6, 0.288288288288288, 1 / 6]
        + [a / 4, 0, 1 / 3, 1 / 6],
        [a, 0, a, 0, a, 0, b, 1 / 6, 0, 0, 2 * b, 0, a / 4, 0, 1 / 3, 0],
    ]
    for column, values in enumerate(expected):
        assert [row[column] for row in matrix] == pytest.approx(values, rel=1e-13)


def test_rttright_counting_process_data_match_r():
    data = {
        "id": [1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7],
        "t1": [0, 2, 0, 0, 1, 4, 0, 0, 3, 0, 0, 5],
        "t2": [2, 5, 3, 1, 4, 6, 4, 3, 7, 5, 5, 8],
        "s": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        "x": [1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2],
        "z": [1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2],
        "w": [1, 1, 2, 1, 1, 1, 0.5, 1, 1, 3, 1, 1],
    }
    # rttright(Surv(t1, t2, s) ~ 1, dc, id = id)
    expected = [0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0.4]
    assert r.rttright("Surv(t1, t2, s) ~ 1", data, id="id") == pytest.approx(expected, rel=1e-13)
    # ... ~ x, weights = w
    assert r.rttright("Surv(t1, t2, s) ~ x", data, id="id", weights="w") == pytest.approx(
        [0, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0.6, 0, 1], rel=1e-13
    )
    # ... ~ z, where z changes within a subject: R counts a subject once per stratum
    assert r.rttright("Surv(t1, t2, s) ~ z", data, id="id") == pytest.approx(expected, rel=1e-13)


def test_rttright_multistate_data_match_r():
    data = {
        "time": [1, 2, 2, 3, 4, 4, 5, 6],
        "st": r_coerce._RFactorVector(["c", "a", "c", "b", "a", "c", "b", "c"], ["c", "a", "b"]),
    }
    # rttright(Surv(time, st) ~ 1, dm)
    p, q, u = 1 / 7, 0.171428571428571, 0.257142857142857
    assert r.rttright("Surv(time, st) ~ 1", data) == pytest.approx(
        [0, p, 0, q, q, 0, u, 0], rel=1e-13
    )
    # ... times = c(2, 4, 7)
    matrix = r.rttright("Surv(time, st) ~ 1", data, times=[2, 4, 7])
    expected = [[0, p, p, p, p, p, p, p], [0, p, 0, q, q, q, q, q], [0, p, 0, q, q, 0, u, 0]]
    for column, values in enumerate(expected):
        assert [row[column] for row in matrix] == pytest.approx(values, rel=1e-13)
