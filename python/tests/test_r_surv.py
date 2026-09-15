"""``Surv``, ``Surv2``, ``strata`` and the timeline conversions against R's ``Surv.R``.

Reference values come from R 4.5 / survival 3.8.11 (``Rscript`` calls quoted next
to the assertions).
"""

import importlib
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_coerce = importlib.import_module("survival.r._coerce")


def _factor(values, levels):
    return r_coerce._RFactorVector(values, levels)


# --- Surv ------------------------------------------------------------------


def test_surv_right_censoring_codes_follow_r():
    # Surv(1:4, c(1,0,1,1)); Surv(1:4, c(2,1,2,2)); Surv(1:4, c(TRUE,FALSE,TRUE,TRUE))
    for status in ([1, 0, 1, 1], [2, 1, 2, 2], [True, False, True, True]):
        surv = r.Surv([1, 2, 3, 4], status)
        assert surv.type == "right"
        assert surv.as_matrix() == [[1.0, 1], [2.0, 0], [3.0, 1], [4.0, 1]]
    assert r.Surv([3, 4]).event == (1, 1)
    assert r.Surv([3, 4, 5], [1, 0, 1], origin=2).time == (1.0, 2.0, 3.0)
    assert r.Surv([1, 2], [1, 0], type="left").type == "left"
    assert r.Surv(time=[1, 2], event=[1, 0]).as_matrix() == [[1.0, 1], [2.0, 0]]


def test_surv_invalid_status_becomes_na_with_r_warning():
    # Surv(c(1, 2, 3), c(1, 0, 5)): "Invalid status value, converted to NA"
    with pytest.warns(UserWarning, match="Invalid status value, converted to NA"):
        surv = r.Surv([1, 2, 3], [1, 0, 5])
    assert surv.event == (1, 0, None)
    assert r.is_na_surv(surv) == [False, False, True]
    # is.na(Surv(c(1, NA, 3), c(1, 0, NA)))
    assert r.is_na_surv(r.Surv([1, None, 3], [1, 0, None])) == [False, True, True]
    with pytest.raises(ValueError, match="Invalid status value, must be logical or numeric"):
        r.Surv([1, 2], ["a", "b"])
    with pytest.raises(ValueError, match="Time and status are different lengths"):
        r.Surv([1, 2, 3], [1, 0])
    with pytest.raises(ValueError, match="Time variable is not numeric"):
        r.Surv(["a", "b"], [1, 0])


def test_surv_counting_and_argument_count_rules():
    surv = r.Surv([0, 1, 2, 0], [3, 4, 5, 2], [1, 0, 1, 1])
    assert surv.type == "counting"
    assert surv.as_matrix() == [[0.0, 3.0, 1], [1.0, 4.0, 0], [2.0, 5.0, 1], [0.0, 2.0, 1]]
    assert surv.ncol == 3
    with pytest.warns(UserWarning, match="Stop time must be > start time, NA created"):
        backwards = r.Surv([0, 5], [3, 4], [1, 0])
    assert math.isnan(backwards.start[1])
    with pytest.raises(ValueError, match="Wrong number of args"):
        r.Surv([1, 2], [1, 0], type="counting")
    with pytest.raises(ValueError, match="Wrong number of args"):
        r.Surv([0, 1], [2, 3], [1, 0], type="right")
    with pytest.raises(ValueError, match="Start and stop are different lengths"):
        r.Surv([0, 1], [2], [1, 0])


def test_surv_interval_and_interval2_match_r_columns():
    # Surv(1:5, c(2, NA, 6, NA, 7), c(3, 0, 3, 1, 2), type = "interval")
    interval = r.Surv([1, 2, 3, 4, 5], [2, None, 6, None, 7], [3, 0, 3, 1, 2], type="interval")
    assert interval.type == "interval"
    assert interval.as_matrix() == [
        [1.0, 2.0, 3],
        [2.0, 1.0, 0],
        [3.0, 6.0, 3],
        [4.0, 1.0, 1],
        [5.0, 1.0, 2],
    ]
    # R's Surv(c(1, NA, 3, 4, 5), c(2, 3, NA, 4, 8), type = "interval2")
    interval2 = r.Surv([1, None, 3, 4, 5], [2, 3, None, 4, 8], type="interval2")
    assert interval2.type == "interval"
    assert interval2.as_matrix() == [
        [1.0, 2.0, 3],
        [3.0, 1.0, 2],
        [3.0, 1.0, 0],
        [4.0, 1.0, 1],
        [5.0, 8.0, 3],
    ]
    assert r.Surv([-math.inf, 2], [1, math.inf], type="interval2").event == (2, 0)
    with pytest.warns(UserWarning, match="Status must be 0, 1, 2 or 3"):
        bad = r.Surv([1, 2], [2, 3], [3, 7], type="interval")
    assert bad.event == (3, None)
    with pytest.warns(UserWarning, match="Invalid interval: start > stop"):
        assert r.Surv([5, 1], [2, 3], type="interval2").event == (None, 3)


def test_surv_multistate_uses_factor_levels_and_states():
    event = _factor(["a", "censor", "b", "a"], ["censor", "a", "b"])
    surv = r.Surv([1, 2, 3, 4], event)
    assert surv.type == "mright"
    assert surv.states == ("a", "b")
    assert surv.event == (1, 0, 2, 1)
    counting = r.Surv([0, 1, 2, 0], [3, 4, 5, 2], event)
    assert counting.type == "mcounting"
    assert counting.as_matrix()[2] == [2.0, 5.0, 2]
    # type = "mstate" sorts the levels of a plain vector: c("b", "a", "c") -> a, b, c
    sorted_levels = r.Surv([1, 2, 3], ["b", "a", "c"], type="mstate")
    assert sorted_levels.states == ("b", "c")
    assert sorted_levels.event == (1, 0, 2)
    with pytest.raises(ValueError, match="each state must have a non-blank name"):
        r.Surv([1, 2], _factor(["", "x"], ["x", ""]))


def test_surv_subset_and_replace_times_keep_metadata():
    surv = r.Surv([0, 1, 2], [3, 4, 5], _factor(["a", "censor", "b"], ["censor", "a", "b"]))
    part = surv.subset([2, 0])
    assert part.type == "mcounting"
    assert part.states == ("a", "b")
    assert part.as_matrix() == [[2.0, 5.0, 2], [0.0, 3.0, 1]]
    shifted = surv.replace_times(start=[0, 0, 0])
    assert shifted.start == (0.0, 0.0, 0.0)
    assert shifted.time == surv.time
    assert r.is_surv(surv)
    assert not r.is_surv([1.0])


def test_format_surv_matches_r_as_character():
    # R's format(Surv(c(1, 2, 3), c(1, 0, 1))); format(Surv(c(0, 1), c(3, 4), c(1, 0)))
    assert r.format_surv(r.Surv([1, 2, 3], [1, 0, 1])) == ["1 ", "2+", "3 "]
    assert r.format_surv(r.Surv([0, 1], [3, 4], [1, 0])) == ["(0,3] ", "(1,4+]"]
    # R's format(Surv(c(1, NA, 3), c(2, 3, NA), type = "interval2"))
    assert r.format_surv(r.Surv([1, None, 3], [2, 3, None], type="interval2")) == [
        "[1, 2]",
        "3-    ",
        "3+    ",
    ]
    # R's format(Surv(c(1, 2), factor(c("a", "censor"), levels = c("censor", "a"))))
    assert r.format_surv(r.Surv([1, 2], _factor(["a", "censor"], ["censor", "a"]))) == [
        "1:a",
        "2+ ",
    ]
    # R's format(Surv(c(1.5, 2.25, 10), c(1, 0, 1), type = "left"))
    assert r.format_surv(r.Surv([1.5, 2.25, 10], [1, 0, 1], type="left")) == [
        " 1.50 ",
        " 2.25-",
        "10.00 ",
    ]
    assert r.format_surv(r.Surv([1, 2], [1, None])) == ["1 ", "2?"]


# --- strata ----------------------------------------------------------------


def test_strata_matches_r_level_order_labels_and_padding():
    # R's strata(sex = c(1, 2, 1), grp = c("a", "b", "b"))
    named = r.strata({"sex": [1, 2, 1], "grp": ["a", "b", "b"]})
    assert named.levels == ["sex=1, grp=a", "sex=1, grp=b", "sex=2, grp=b"]
    assert named.codes == [0, 2, 1]
    assert named.counts == [1, 1, 1]
    # strata(c(1, 2, 1, 2, NA), c("a", "a", "b", "b", "a")): NA rows are NA
    plain = r.strata([1, 2, 1, 2, None], ["a", "a", "b", "b", "a"], labels=["x", "y"])
    assert plain.codes == [0, 2, 1, 3, None]
    assert plain.labels[4] is None
    # ... na.group = TRUE adds an "NA" level
    grouped = r.strata(
        [1, 2, 1, 2, None], ["a", "a", "b", "b", "a"], na_group=True, labels=["x", "y"]
    )
    assert grouped.levels[-1] == "x=NA, y=a"
    assert grouped.codes == [0, 2, 1, 3, 4]
    # strata(c("x", "y", "x")): character arguments get short labels
    assert r.strata(["x", "y", "x"]).levels == ["x", "y"]
    # strata(c(10, 2, 10), c("b", "a", "b"), sep = "/"): numeric levels sort numerically
    assert r.strata([10, 2, 10], ["b", "a", "b"], sep="/", labels=["n", "s"]).levels == [
        "n=2/s=a",
        "n=10/s=b",
    ]
    # the second variable's labels are padded to a common width, as R's format() does
    padded = r.strata([1, 1], ["a", "bb"], labels=["v", "w"])
    assert padded.levels == ["v=1, w=a ", "v=1, w=bb"]
    assert list(padded) == padded.labels
    with pytest.raises(ValueError, match="all arguments must be the same length"):
        r.strata([1, 2], ["a"])


# --- Surv2 and the timeline conversions --------------------------------------


def test_surv2_codes_states_and_repeated_option():
    event = _factor(["a", "censor", "b"], ["censor", "a", "b"])
    surv2 = r.Surv2([1, 2, 3], event, repeated="first")
    assert surv2.states == ("a", "b")
    assert surv2.status == (1, 0, 2)
    assert surv2.repeated == "first"
    assert r.format_surv(surv2) == ["1:a", "2+ ", "3:b"]
    plain = r.Surv2([1, 2], [True, False])
    assert plain.status == (1, 0)
    assert plain.states == ()
    with pytest.raises(ValueError, match="invalid value for repeated option"):
        r.Surv2([1, 2], [1, 0], repeated="sometimes")
    with pytest.raises(ValueError, match="Time and event are different lengths"):
        r.Surv2([1, 2, 3], [1, 0])


def test_surv2data_builds_counting_rows_and_initial_states():
    # a two-subject multi-state timeline, rows deliberately out of order
    result = r.Surv2data(
        time=[0, 5, 9, 0, 3],
        status=[1, 2, 0, 1, 2],
        states=["well", "ill"],
        id=[1, 1, 1, 2, 2],
    )
    assert result.type == "mcounting"
    assert result.row == [0, 1, 3]
    assert result.start == [0.0, 5.0, 0.0]
    assert result.stop == [5.0, 9.0, 3.0]
    assert result.status == [2, 0, 2]
    assert result.istate == [1, 2, 1]
    assert result.states == ["well", "ill"]
    single = r.Surv2data(time=[0, 5, 0, 3], status=[0, 1, 0, 0], id=[1, 1, 2, 2])
    assert single.type == "right"
    assert single.istate is None
    with pytest.raises(ValueError, match="id and time cannot be missing"):
        r.Surv2data(time=[0, None], status=[0, 1], id=[1, 1])
    assert r.fromtimeline([0, 5], [0, 1], id=[1, 1]).stop == [5.0]


def test_totimeline_expands_counting_rows_like_r_draft():
    result = r.totimeline(
        [0, 3, 0],
        [3, 6, 4],
        [1, 0, 2],
        states=["ill", "dead"],
        id=[1, 1, 2],
    )
    assert result.time == [0.0, 3.0, 6.0, 0.0, 4.0]
    assert result.state_levels == ["censor", "(s0)", "ill", "dead"]
    assert result.status == [1, 2, 0, 1, 3]
    assert result.data_row == [0, 1, 1, 2, 2]
    with_istate = r.totimeline(
        [0, 3], [3, 6], [1, 0], states=["ill"], id=[1, 1], istate=["well", "well"]
    )
    assert with_istate.state_levels == ["censor", "well", "ill"]
    assert with_istate.status == [1, 2, 0]
    with pytest.raises(ValueError, match="is not a recognized state"):
        r.totimeline([0], [3], [1], states=["ill"], id=[1], istate=["x"], istate_levels=["y"])
