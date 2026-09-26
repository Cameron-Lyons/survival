"""Data-preparation regressions against R 4.5.3 / survival 3.8-12 (values hard-coded)."""

import importlib
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
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
