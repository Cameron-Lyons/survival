"""``finegray`` against R 4.5 / survival 3.8.11 reference values."""

import importlib
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_coerce = importlib.import_module("survival.r._coerce")


def _data():
    return {
        "time": [1, 2, 2, 3, 4, 5, 6, 7],
        "ev": r_coerce._RFactorVector(
            ["a", "b", "censor", "a", "b", "censor", "a", "b"], ["censor", "a", "b"]
        ),
        "x": [0.5, 1.2, 0.7, 0.9, 1.5, 0.3, 1.1, 0.8],
        "g": ["p", "q", "p", "q", "p", "q", "p", "q"],
    }


def test_finegray_expands_right_censored_competing_risks_like_r():
    # finegray(Surv(time, ev) ~ x, d)
    frame = r.finegray("Surv(time, ev) ~ x", _data())
    assert list(frame) == ["x", "fgstart", "fgstop", "fgstatus", "fgwt"]
    assert frame.event == "a"
    assert frame["x"] == [0.5, 1.2, 1.2, 1.2, 0.7, 0.9, 1.5, 1.5, 0.3, 1.1, 0.8]
    assert frame["fgstart"] == [0, 0, 2, 5, 0, 0, 0, 5, 0, 0, 0]
    assert frame["fgstop"] == [1, 2, 5, 7, 2, 3, 5, 7, 5, 6, 7]
    assert frame["fgstatus"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    assert frame["fgwt"] == pytest.approx(
        [1, 1, 0.8333333, 0.5555556, 1, 1, 1, 0.6666667, 1, 1, 1], rel=1e-6
    )
    assert r.as_data_frame(frame)["fgwt"] == frame["fgwt"]


def test_finegray_strata_weights_count_prefix_and_etype_like_r():
    # finegray(Surv(time, ev) ~ x + strata(g), d, etype = "b", prefix = "cr", count = "n",
    #          weights = c(1, 2, 1, 2, 1, 2, 1, 2))
    frame = r.finegray(
        "Surv(time, ev) ~ x + strata(g)",
        _data(),
        etype="b",
        prefix="cr",
        count="n",
        weights=[1, 2, 1, 2, 1, 2, 1, 2],
    )
    assert list(frame) == ["x", "(weights)", "crstart", "crstop", "crstatus", "crwt", "n"]
    assert frame.event == "b"
    assert frame["x"] == [0.5, 0.5, 0.7, 1.5, 1.1, 1.2, 0.9, 0.9, 0.3, 0.8]
    assert frame["(weights)"] == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert frame["crstart"] == [0, 2, 0, 0, 0, 0, 0, 5, 0, 0]
    assert frame["crstop"] == [2, 6, 2, 4, 6, 2, 5, 7, 5, 7]
    assert frame["crstatus"] == [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    # R multiplies by user.weights[split$row], the row number within the stratum
    assert frame["crwt"] == pytest.approx([1, 2 / 3, 2, 1, 2, 1, 2, 1, 1, 2], rel=1e-6)
    assert frame["n"] == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    with pytest.warns(UserWarning, match="only the first endpoint was used"):
        both = r.finegray("Surv(time, ev) ~ x", _data(), etype=["b", "a"])
    assert both.event == "b"


def test_finegray_counting_process_data_with_delayed_entry():
    data = {
        "id": [1, 1, 2, 2, 3],
        "t1": [0, 2, 1, 3, 0],
        "t2": [2, 5, 3, 6, 4],
        "ev": r_coerce._RFactorVector(["censor", "a", "censor", "b", "a"], ["censor", "a", "b"]),
        "x": [1, 2, 3, 4, 5],
    }
    # finegray(Surv(t1, t2, ev) ~ x, dd, id = id)
    frame = r.finegray("Surv(t1, t2, ev) ~ x", data, id="id")
    assert list(frame) == ["x", "fgstart", "fgstop", "fgstatus", "fgwt"]
    assert frame["fgstart"] == [0, 2, 1, 3, 0]
    assert frame["fgstop"] == [2, 5, 3, 6, 4]
    assert frame["fgstatus"] == [0, 1, 0, 0, 1]
    assert frame["fgwt"] == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match=r"\(start, stop\] data requires a subject id"):
        r.finegray("Surv(t1, t2, ev) ~ x", data)
    gappy = {**data, "t1": [0, 3, 1, 3, 0]}
    with pytest.raises(ValueError, match="a subject has gaps in time"):
        r.finegray("Surv(t1, t2, ev) ~ x", gappy, id="id")
    early = {
        **data,
        "ev": r_coerce._RFactorVector(["a", "a", "censor", "b", "a"], ["censor", "a", "b"]),
    }
    with pytest.raises(ValueError, match="a subject has a transition before their last time point"):
        r.finegray("Surv(t1, t2, ev) ~ x", early, id="id")


def test_finegray_argument_checks_follow_r():
    data = _data()
    with pytest.raises(ValueError, match="etype argument has a state that is not in the data"):
        r.finegray("Surv(time, ev) ~ x", data, etype="zzz")
    binary = {**data, "x": [1, 0, 1, 0, 1, 0, 1, 0]}
    with (
        pytest.raises(ValueError, match="Fine-Gray model requires a multi-state survival"),
        warnings.catch_warnings(action="ignore"),
    ):
        r.finegray("Surv(time, x) ~ g", binary)
    with pytest.raises(ValueError, match="survival time has only a single state"):
        r.finegray(
            "Surv(time, ev) ~ x",
            {**data, "ev": r_coerce._RFactorVector(["a"] * 8, ["censor", "a"])},
        )
    with pytest.raises(ValueError, match=r"a cluster\(\) term is not valid"):
        r.finegray("Surv(time, ev) ~ x + cluster(g)", data)
    with pytest.raises(ValueError, match="No \\(non-missing\\) observations"):
        r.finegray("Surv(time, ev) ~ x", {**data, "x": [None] * 8}, na_action="na.omit")
    subset = r.finegray(
        "Surv(time, ev) ~ x", data, subset=[True, True, True, True, False, False, False, False]
    )
    assert max(subset["fgstop"]) == 3
