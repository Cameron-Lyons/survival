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


# --- subject ids --------------------------------------------------------------


def test_ids_convert_from_python_and_numpy_scalars():
    ids = [np.int64(3), 3.0, np.float32(1.5), 1.5, "a", np.str_("a"), True, 1, 2**70]
    result = survival.data_prep.cluster(ids)
    assert list(result.codes) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    assert list(result.sizes) == [2, 2, 2, 2, 1]
    with pytest.raises(TypeError, match="an id must be an int, float or str, not NoneType"):
        survival.data_prep.cluster([1, None])
