import importlib
import math
import random

import pytest

from .helpers import setup_survival_import
from .r_api_support import _reference_lvcf_with_time, _toy_data

survival = setup_survival_import()
r_coerce = importlib.import_module("survival.r._coerce")


def test_aeqSurv_adjusts_surv_response_like_r():
    right = survival.Surv([1.0, 1.0 + 1e-8, 2.0], [1, 0, 1])
    adjusted_right = survival.aeqSurv(right, tolerance=1e-7)

    assert adjusted_right.type == "right"
    assert adjusted_right.time == pytest.approx((1.0, 1.0, 2.0))
    assert adjusted_right.event == (1, 0, 1)

    counting = survival.Surv([0.0, 0.0, 1.0], [1.0, 1.0 + 1e-8, 2.0], [1, 0, 1])
    adjusted_counting = survival.aeqSurv(counting, tolerance=1e-7)

    assert adjusted_counting.type == "counting"
    assert adjusted_counting.start == pytest.approx((0.0, 0.0, 1.0))
    assert adjusted_counting.time == pytest.approx((1.0, 1.0, 2.0))
    assert adjusted_counting.event == (1, 0, 1)

    interval = survival.Surv(
        [1.0, 1.0 + 1e-8, 2.0],
        [1.0, 2.0 + 1e-8, 3.0],
        [1, 3, 3],
        type="interval",
    )
    adjusted_interval = survival.aeqSurv(interval, tolerance=1e-7)

    assert adjusted_interval.type == "interval"
    assert adjusted_interval.time == pytest.approx((1.0, 1.0, 2.0))
    assert adjusted_interval.time2 == pytest.approx((1.0, 2.0, 3.0))
    assert adjusted_interval.event == (1, 3, 3)

    interval2 = survival.Surv(
        [float("-inf"), 1.0, 2.0],
        [1.0, 2.0 + 1e-8, float("inf")],
        type="interval2",
    )
    adjusted_interval2 = survival.aeqSurv(interval2, tolerance=1e-7)

    assert adjusted_interval2.type == "interval2"
    assert math.isinf(adjusted_interval2.time[0])
    assert adjusted_interval2.time[0] < 0.0
    assert adjusted_interval2.time[1] == pytest.approx(1.0)
    assert adjusted_interval2.time2[1] == pytest.approx(2.0)
    assert math.isinf(adjusted_interval2.time2[2])
    assert adjusted_interval2.time2[2] > 0.0
    assert adjusted_interval2.event == interval2.event

    no_adjust = survival.aeqSurv(right, tolerance=0.0)
    assert no_adjust is right
    with pytest.raises(TypeError, match="Surv object"):
        survival.aeqSurv([1.0, 2.0])
    with pytest.raises(ValueError, match="effective length 0"):
        survival.aeqSurv(
            survival.Surv([1.0], [1.0 + 1e-8], [1]),
            tolerance=1e-7,
        )


def test_survSplit_splits_right_and_counting_surv_responses_like_r():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    right = survival.Surv([5.0, 8.0], [1, 0])
    right_split = survival.survSplit(
        right,
        {"group": ["a", "b"], "x": [10, 20]},
        cut=[3.0, 6.0],
        episode="episode",
        id="rowid",
        end="time",
        event="status",
    )

    assert right_split == {
        "group": ["a", "a", "b", "b", "b"],
        "x": [10, 10, 20, 20, 20],
        "rowid": [1, 1, 2, 2, 2],
        "tstart": [0.0, 3.0, 0.0, 3.0, 6.0],
        "time": [3.0, 5.0, 3.0, 6.0, 8.0],
        "status": [0, 1, 0, 0, 0],
        "episode": [1, 2, 1, 2, 3],
    }

    counting = survival.Surv([0.0, 2.0], [5.0, 8.0], [1, 0])
    counting_split = survival.survSplit(
        counting,
        {"group": ["a", "b"]},
        cut=[3.0, 6.0],
        start="start",
        end="stop",
        event="status",
        episode="episode",
        id="rowid",
    )

    assert counting_split == {
        "group": ["a", "a", "b", "b", "b"],
        "rowid": [1, 1, 2, 2, 2],
        "start": [0.0, 3.0, 2.0, 3.0, 6.0],
        "stop": [3.0, 5.0, 3.0, 6.0, 8.0],
        "status": [0, 1, 0, 0, 0],
        "episode": [1, 2, 1, 2, 3],
    }

    multistate_right = survival.Surv(
        [1.0, 3.0, 4.0],
        Factor(["a", "censor", "b"], ["censor", "a", "b"]),
        type="mstate",
    )
    multistate_right_split = survival.survSplit(
        multistate_right,
        {"x": [11, 12, 13]},
        cut=[2.0, 3.5],
        episode="episode",
        id="subject",
        end="time",
        event="state",
    )
    assert multistate_right_split == {
        "x": [11, 12, 12, 13, 13, 13],
        "subject": [1, 2, 2, 3, 3, 3],
        "tstart": [0.0, 0.0, 2.0, 0.0, 2.0, 3.5],
        "time": [1.0, 2.0, 3.0, 2.0, 3.5, 4.0],
        "state": [1, 0, 0, 0, 0, 2],
        "episode": [1, 1, 2, 1, 2, 3],
    }

    multistate_counting = survival.Surv(
        [0.0, 1.0],
        [3.0, 4.0],
        Factor(["a", "b"], ["censor", "a", "b"]),
        type="mstate",
    )
    multistate_counting_split = survival.survSplit(
        multistate_counting,
        {"x": [1, 2]},
        cut=[2.0],
        start="start",
        end="stop",
        event="state",
        episode="episode",
        id="subject",
    )
    assert multistate_counting_split == {
        "x": [1, 1, 2, 2],
        "subject": [1, 1, 2, 2],
        "start": [0.0, 2.0, 1.0, 2.0],
        "stop": [2.0, 3.0, 2.0, 4.0],
        "state": [0, 1, 0, 2],
        "episode": [1, 2, 1, 2],
    }

    with pytest.raises(ValueError, match="not valid for interval2"):
        survival.survSplit(survival.Surv([1.0], [2.0], type="interval2"), cut=[1.5])
    with pytest.raises(ValueError, match="suggested id name"):
        survival.survSplit(right, {"rowid": [1, 2]}, cut=[3.0], id="rowid")
    with pytest.raises(ValueError, match="'zero' parameter"):
        survival.survSplit(right, cut=[3.0], zero=5.0)
    with pytest.raises(ValueError, match="finite numbers"):
        survival.survSplit(right, cut=[math.inf])


def test_rttright_formula_wrapper_preserves_legacy_direct_api():
    legacy = survival.rttright([3.0, 1.0, 2.0], [1, 0, 1])
    formula = survival.rttright(
        "Surv(time, status) ~ 1",
        data={"time": [3.0, 1.0, 2.0], "status": [1, 0, 1]},
    )
    formula_raw = survival.rttright(
        "Surv(time, status) ~ 1",
        data={"time": [3.0, 1.0, 2.0], "status": [1, 0, 1]},
        renorm=False,
    )
    weighted = survival.rttright(
        "Surv(time, status) ~ 1",
        data={
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": [1, 0, 1, 0],
            "wt": [1.0, 2.0, 1.0, 1.0],
        },
        weights="wt",
    )
    with_id = survival.rttright(
        "Surv(time, status) ~ 1",
        data={"time": [3.0, 1.0, 2.0], "status": [1, 0, 1], "id": ["c", "a", "b"]},
        id="id",
    )
    timed = survival.rttright(
        "Surv(time, status) ~ 1",
        data={"time": [3.0, 1.0, 2.0], "status": [1, 0, 1]},
        times=[1.0, 2.0, 3.0],
    )
    timed_single = survival.rttright(
        "Surv(time, status) ~ 1",
        data={"time": [3.0, 1.0, 2.0], "status": [1, 0, 1]},
        times=[2.0],
    )
    direct_timed = survival.rttright(
        [3.0, 1.0, 2.0],
        [1, 0, 1],
        times=[1.0, 2.0, 3.0],
    )
    grouped = survival.rttright(
        "Surv(time, status) ~ group",
        data={
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": [0, 1, 0, 1],
            "group": ["A", "A", "B", "B"],
        },
    )
    offset_grouped_data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "group": ["treated", "treated", "control", "control"],
        "off": [1.0, 2.0, 3.0, 4.0],
    }
    with pytest.warns(RuntimeWarning, match="Offset term ignored"):
        offset_grouped = survival.rttright(
            "Surv(time, status) ~ group + offset(off)",
            data=offset_grouped_data,
        )
    offset_grouped_reference = survival.rttright(
        "Surv(time, status) ~ group",
        data=offset_grouped_data,
    )
    grouped_timed = survival.rttright(
        "Surv(time, status) ~ group",
        data={
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": [0, 1, 0, 1],
            "group": ["A", "A", "B", "B"],
        },
        times=[1.0, 2.0, 3.0, 4.0],
    )
    counting_data = {
        "id": ["a", "a", "b", "b"],
        "start": [0.0, 1.0, 0.0, 2.0],
        "stop": [1.0, 3.0, 2.0, 4.0],
        "status": [0, 1, 0, 1],
    }
    counting = survival.rttright(
        "Surv(start, stop, status) ~ 1",
        data=counting_data,
        id="id",
    )
    counting_raw = survival.rttright(
        "Surv(start, stop, status) ~ 1",
        data=counting_data,
        id="id",
        renorm=False,
    )
    counting_timed = survival.rttright(
        "Surv(start, stop, status) ~ 1",
        data=counting_data,
        id="id",
        times=[1.0, 2.0, 3.0, 4.0],
    )
    counting_grouped_data = {
        "id": ["a", "a", "b", "b", "c"],
        "start": [0.0, 1.0, 0.0, 2.0, 0.0],
        "stop": [1.0, 3.0, 2.0, 4.0, 2.5],
        "status": [0, 1, 0, 1, 1],
        "group": ["x", "x", "y", "y", "x"],
    }
    counting_grouped = survival.rttright(
        "Surv(start, stop, status) ~ group",
        data=counting_grouped_data,
        id="id",
    )
    counting_grouped_timed = survival.rttright(
        "Surv(start, stop, status) ~ group",
        data=counting_grouped_data,
        id="id",
        times=[1.0, 2.0, 3.0],
    )
    weighted_counting_grouped_data = {
        "id": ["a", "a", "b", "b", "c", "c"],
        "start": [0.0, 1.0, 0.0, 2.0, 0.0, 1.5],
        "stop": [1.0, 3.0, 2.0, 4.0, 1.5, 2.5],
        "status": [0, 1, 0, 1, 0, 0],
        "weights": [2.0, 2.0, 1.0, 1.0, 3.0, 3.0],
        "group": ["x", "x", "y", "y", "x", "x"],
    }
    weighted_counting_grouped_timed = survival.rttright(
        "Surv(start, stop, status) ~ group",
        data=weighted_counting_grouped_data,
        id="id",
        weights="weights",
        times=[1.0, 2.0, 3.0, 4.0],
    )

    assert legacy.time == pytest.approx([1.0, 2.0, 3.0])
    assert legacy.weights == pytest.approx([0.0, 0.5, 0.5])
    assert formula == pytest.approx([0.5, 0.0, 0.5])
    assert formula_raw == pytest.approx([1.5, 0.0, 1.5])
    assert weighted == pytest.approx([0.2, 0.0, 0.4, 0.0])
    assert with_id == pytest.approx(formula)
    for actual_row, expected_row in zip(
        timed,
        [
            [1.0 / 3.0, 0.5, 0.5],
            [1.0 / 3.0, 0.0, 0.0],
            [1.0 / 3.0, 0.5, 0.5],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert timed_single == pytest.approx([0.5, 0.0, 0.5])
    for actual_row, expected_row in zip(direct_timed, timed, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert grouped == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert offset_grouped == pytest.approx(offset_grouped_reference)
    for actual_row, expected_row in zip(
        grouped_timed,
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5, 1.0],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert counting == pytest.approx([0.0, 0.5, 0.0, 0.5])
    assert counting_raw == pytest.approx([0.0, 1.0, 0.0, 1.0])
    for actual_row, expected_row in zip(
        counting_timed,
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert counting_grouped == pytest.approx([0.0, 0.5, 0.0, 1.0, 0.5])
    for actual_row, expected_row in zip(
        counting_grouped_timed,
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    for actual_row, expected_row in zip(
        weighted_counting_grouped_timed,
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.4, 0.4, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6, 0.6],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)

    with pytest.raises(ValueError, match="id is required"):
        survival.rttright(survival.Surv([0.0], [1.0], [1]))
    with pytest.raises(ValueError, match="survcheck"):
        survival.rttright([1.0, 2.0, 3.0], [0, 0, 1], id=["a", "a", "b"])
    with pytest.raises(ValueError, match="id must have"):
        survival.rttright([1.0, 2.0], [1, 0], id=["a"])
    with pytest.raises(ValueError, match="multiple weights"):
        survival.rttright(
            survival.Surv([0.0, 1.0, 0.0, 2.0], [1.0, 3.0, 2.0, 4.0], [0, 1, 0, 1]),
            id=["a", "a", "b", "b"],
            weights=[1.0, 2.0, 1.0, 1.0],
        )
    with pytest.raises(NotImplementedError, match="delayed entry"):
        survival.rttright(
            survival.Surv([0.0, 1.0], [2.0, 3.0], [1, 1]),
            id=["a", "b"],
        )


def test_rttright_supports_simple_multistate_right_censoring():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "state": Factor(
            ["a", "censor", "b", "a"],
            ["censor", "a", "b"],
        ),
        "id": ["a", "b", "c", "d"],
    }

    result = survival.rttright("Surv(time, state) ~ 1", data=data, id="id")
    timed = survival.rttright(
        "Surv(time, state) ~ 1",
        data=data,
        times=[1.0, 2.0, 3.0, 4.0],
    )

    assert result == pytest.approx([0.25, 0.0, 0.375, 0.375])
    for actual_row, expected_row in zip(
        timed,
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.0, 0.0],
            [0.25, 0.25, 0.375, 0.375],
            [0.25, 0.25, 0.375, 0.375],
        ],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)

    with pytest.raises(NotImplementedError, match="delayed entry or multistate"):
        survival.rttright(
            survival.Surv(
                [0.0, 1.0],
                [1.0, 2.0],
                Factor(["a", "censor"], ["censor", "a"]),
                type="mstate",
            ),
            id=["a", "a"],
        )


def test_lvcf_and_nostutter_match_r_data_prep_helpers():
    carried = survival.lvcf([1, 1, 1, 2, 2], [10.0, None, 12.0, None, 20.0])
    carried_by_time = survival.lvcf([1, 1, 1], [None, 10.0, None], time=[2.0, 1.0, 3.0])
    carried_with_missing_time = survival.lvcf([1, 1, 1], [10.0, None, 20.0], time=[1.0, None, 2.0])
    carried_with_infinite_time = survival.lvcf(
        [1, 1, 1], [10.0, None, 20.0], time=[1.0, math.inf, 2.0]
    )
    carried_with_negative_infinity = survival.lvcf(
        [1, 1, 1], [10.0, None, 20.0], time=[1.0, -math.inf, 2.0]
    )
    carried_by_character_time = survival.lvcf([1, 1, 1], ["x", None, None], time=["b", "a", "c"])
    factor_id = r_coerce._r_factor(["b", "a", "b", "a"], ["b", "a"])
    carried_by_factor_id = survival.lvcf(factor_id, [None, 1, 2, None])
    factor_time = r_coerce._r_factor(["b", "a", "c"], ["c", "b", "a"])
    carried_by_factor_time = survival.lvcf([1, 1, 1], ["x", None, None], factor_time)
    carried_scalar = survival.lvcf(1, 10, time=2)
    carried_missing_scalar = survival.lvcf("a", None)
    stuttered = survival.nostutter([1, 1, 1, 2, 2], [0, 1, 1, 1, 1])
    stuttered_with_missing = survival.nostutter([1, 1, 1], [None, 1, 1])
    stuttered_single = survival.nostutter(
        [1, 1, 1, 1, 2, 2, 2],
        [1, 2, 1, 3, 1, 1, 2],
        single=True,
    )
    stuttered_single_with_censor_gap = survival.nostutter([1, 1, 1], [1, 0, 1], single=True)
    stuttered_single_with_missing = survival.nostutter([1, 1, 1, 1], [None, 1, 2, 1], single=True)
    stuttered_character = survival.nostutter(
        [1, 1, 1, 2, 2],
        ["censor", "a", "a", "b", "b"],
        censor="censor",
    )
    stuttered_character_ids = survival.nostutter(
        ["a", "a", "a", "b", "b"],
        [1, 2, 2, 1, 1],
    )
    stuttered_character_ids_and_states = survival.nostutter(
        ["a", "a", "a", "b", "b"],
        ["censor", "x", "x", "y", "y"],
        censor="censor",
    )
    stuttered_large_integers = survival.nostutter(
        [2**53, 2**53, 2**53, 2**53 + 1],
        [2**53, 2**53 + 1, 2**53 + 1, 2**53 + 1],
    )
    stuttered_large_integer_ids_character = survival.nostutter(
        [2**53, 2**53 + 1],
        ["x", "x"],
        censor="censor",
    )
    stuttered_mixed_fallback = survival.nostutter([1, 1, 1], [1, "x", "x"])

    assert carried == [10.0, 10.0, 12.0, None, 20.0]
    assert carried_by_time == [10.0, 10.0, 10.0]
    assert carried_with_missing_time == [10.0, 20.0, 20.0]
    assert carried_with_infinite_time == [10.0, 20.0, 20.0]
    assert carried_with_negative_infinity == [10.0, None, 20.0]
    assert carried_by_character_time == ["x", None, "x"]
    assert carried_by_factor_id == [None, 1, 2, 1]
    assert carried_by_factor_time == ["x", "x", None]
    assert carried_scalar == [10]
    assert carried_missing_scalar == [None]
    assert stuttered == [0, 1, 0, 1, 0]
    assert stuttered_with_missing == [None, 1, 0]
    assert stuttered_single == [1, 2, 0, 3, 1, 0, 2]
    assert stuttered_single_with_censor_gap == [1, 0, 0]
    assert stuttered_single_with_missing == [None, 1, 2, 0]
    assert stuttered_character == ["censor", "a", "censor", "b", "censor"]
    assert stuttered_character_ids == [1, 2, 0, 1, 0]
    assert stuttered_character_ids_and_states == ["censor", "x", "censor", "y", "censor"]
    assert stuttered_large_integers == [2**53, 2**53 + 1, 0, 2**53 + 1]
    assert stuttered_large_integer_ids_character == ["x", "x"]
    assert stuttered_mixed_fallback == [1, "x", 0]

    with pytest.raises(ValueError, match="same length"):
        survival.lvcf([1], [1, None])
    with pytest.raises(ValueError, match="missing"):
        survival.nostutter([1, None], [0, 1])


@pytest.mark.parametrize(
    ("character_id", "character_time"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_lvcf_time_scan_matches_reference_randomized(character_id, character_time):
    rng = random.Random(20260801 + 2 * int(character_id) + int(character_time))  # noqa: S311
    for _case in range(200):
        n = rng.randrange(0, 80)
        ids = [rng.choice("abcde") if character_id else rng.randrange(5) for _ in range(n)]
        values = [None if rng.random() < 0.45 else rng.randrange(-20, 21) for _ in range(n)]
        if character_time:
            times = [None if rng.random() < 0.1 else rng.choice("abcdef") for _ in range(n)]
        else:
            times = [None if rng.random() < 0.1 else rng.randrange(-5, 11) for _ in range(n)]

        assert survival.lvcf(ids, values, time=times) == _reference_lvcf_with_time(
            ids,
            values,
            times,
        )


def test_lvcf_preserves_large_integer_id_groups():
    large = 1 << 53
    assert survival.lvcf(
        [large, large + 1, large, large + 1],
        [10, 20, None, None],
        time=[1, 1, 2, 2],
    ) == [10, 20, 10, 20]


def test_survcondense_accepts_formula_and_preserves_direct_api():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    data = {
        "id": [2, 1, 1, 2],
        "tstart": [0.0, 0.0, 5.0, 3.0],
        "tstop": [3.0, 5.0, 8.0, 5.0],
        "event": [0, 0, 0, 1],
        "x": ["a", "b", "b", "a"],
        "wt": [1, 2, 2, 1],
    }

    result = survival.survcondense("Surv(tstart, tstop, event) ~ x", data=data, id="id")
    weighted = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data=data,
        weights="wt",
        id="id",
    )
    direct = survival.survcondense(
        [2, 1, 1, 2],
        [0.0, 0.0, 5.0, 3.0],
        [3.0, 5.0, 8.0, 5.0],
        [0, 0, 0, 1],
    )
    custom_names = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data=data,
        id="id",
        start="begin",
        end="finish",
        event="status",
    )
    subset = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data=data,
        subset=[False, True, True, False],
        id="id",
    )
    subset_unsorted = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data=data,
        subset=[True, True, True, False],
        id="id",
    )
    no_drops = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data={
            "id": [1, 1, 1],
            "tstart": [0.0, 1.0, 2.0],
            "tstop": [1.0, 2.0, 3.0],
            "event": [0, 0, 0],
            "x": ["a", "a", "a"],
            "wt": [1, 2, 1],
        },
        weights="wt",
        id="id",
    )
    missing_values = survival.survcondense(
        "Surv(tstart, tstop, event) ~ x",
        data={
            "id": ["b", "a", "b", "a"],
            "tstart": [0.0, 0.0, 1.0, 1.0],
            "tstop": [1.0, 1.0, 2.0, 2.0],
            "event": [0, 0, 0, 0],
            "x": [None, "x", None, "x"],
        },
        id="id",
        na_action="pass",
    )
    special_data = {
        **data,
        "site": ["south", "north", "north", "south"],
        "phase": ["late", "early", "early", "late"],
        "off": [1.0, 2.0, 2.0, 1.0],
    }
    strata = survival.survcondense(
        "Surv(tstart, tstop, event) ~ strata(site)",
        data=special_data,
        id="id",
    )
    multi_strata = survival.survcondense(
        "Surv(tstart, tstop, event) ~ strata(site, phase)",
        data=special_data,
        id="id",
    )
    offset_first = survival.survcondense(
        "Surv(tstart, tstop, event) ~ offset(off) + x",
        data=special_data,
        id="id",
    )
    multistate_data = {
        "id": [1, 1, 2, 2, 3, 3],
        "tstart": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "tstop": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "state": Factor(
            ["a", "censor", "b", "a", "a", "b"],
            ["censor", "a", "b"],
        ),
        "x": [1, 1, 2, 2, 3, 3],
    }
    multistate = survival.survcondense(
        "Surv(tstart, tstop, state) ~ x",
        data=multistate_data,
        id="id",
        event="state",
    )

    assert result == {
        "x": ["b", "a"],
        "id": [1, 2],
        "tstart": [0.0, 0.0],
        "tstop": [8.0, 5.0],
        "event": [0, 1],
    }
    assert weighted == {
        "x": ["b", "a"],
        "wt": [2, 1],
        "id": [1, 2],
        "tstart": [0.0, 0.0],
        "tstop": [8.0, 5.0],
        "event": [0, 1],
    }
    assert direct.id == [1, 2]
    assert direct.time2 == pytest.approx([8.0, 5.0])
    assert custom_names["begin"] == pytest.approx([0.0, 0.0])
    assert custom_names["finish"] == pytest.approx([8.0, 5.0])
    assert custom_names["status"] == [0, 1]
    assert subset == {"x": ["b"], "id": [1], "tstart": [0.0], "tstop": [8.0], "event": [0]}
    assert subset_unsorted == {
        "x": ["a", "b"],
        "id": [2, 1],
        "tstart": [0.0, 5.0],
        "tstop": [3.0, 8.0],
        "event": [0, 0],
    }
    assert no_drops == {"x": [], "wt": [], "id": [], "tstart": [], "tstop": [], "event": []}
    assert missing_values == {
        "x": [None, "x"],
        "id": ["b", "a"],
        "tstart": [0.0, 0.0],
        "tstop": [2.0, 2.0],
        "event": [0, 0],
    }
    assert strata == {
        "strata(site)": ["north", "south"],
        "id": [1, 2],
        "tstart": [0.0, 0.0],
        "tstop": [8.0, 5.0],
        "event": [0, 1],
    }
    assert multi_strata["strata(site, phase)"] == ["north, early", "south, late"]
    assert list(offset_first)[:2] == ["offset(off)", "x"]
    assert offset_first["offset(off)"] == pytest.approx([2.0, 1.0])
    assert offset_first["x"] == ["b", "a"]
    assert multistate == {
        "x": [1, 2, 3],
        "id": [1, 2, 3],
        "tstart": [0.0, 0.0, 0.0],
        "tstop": [2.0, 2.0, 2.0],
        "state": ["censor", "a", "b"],
    }

    with pytest.raises(ValueError, match="counting-process"):
        survival.survcondense(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            id=list(range(len(_toy_data()["time"]))),
        )
    with pytest.raises(ValueError, match="requires an id"):
        survival.survcondense("Surv(tstart, tstop, event) ~ x", data=data)


def test_r_style_neardate_and_tcut_use_native_data_prep_helpers():
    assert survival.neardate(
        [1, 1, 2],
        [1, 1, 2],
        [4.0, 12.0, 7.0],
        [5.0, 10.0, 9.0],
    ) == [1, None, 3]
    assert survival.neardate(
        [1, 1, 2],
        [1, 1, 2],
        [4.0, 12.0, 7.0],
        [5.0, 10.0, 9.0],
        best="prior",
    ) == [None, 2, None]
    assert survival.neardate(
        ["a", "b"],
        ["a", "b"],
        [4.0, 12.0],
        [5.0, 10.0],
        nomatch=0,
    ) == [1, 0]
    missing_and_infinite = [None, 2.0, float("inf"), float("-inf")]
    reference_dates = [1.0, None, float("inf"), float("-inf")]
    assert survival.neardate(
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        missing_and_infinite,
        reference_dates,
        nomatch=0,
    ) == [0, 3, 3, 4]
    assert survival.neardate(
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        missing_and_infinite,
        reference_dates,
        best="prior",
        nomatch=0,
    ) == [0, 1, 3, 4]
    assert survival.neardate([None, 1], [None, 1], [1.0, 2.0], [1.0, 2.0], nomatch=0) == [None, 2]
    assert survival.neardate([None, 1], [None, 2], [1.0, 2.0], [1.0, 2.0], nomatch=0) == [None, 0]
    with pytest.raises(ValueError, match="No valid entries"):
        survival.neardate([1], [1], [1.0], [None])
    with pytest.raises(ValueError, match="No valid entries"):
        survival.neardate([1], [2], [1.0], [1.0])

    cut = survival.tcut([5.0, 15.0, 30.0], [0.0, 10.0, 20.0, 30.0])
    assert isinstance(cut, survival.TcutResult)
    assert cut.codes == [0, 1, 2]
    assert cut.levels == ["0+ thru 10", "10+ thru 20", "20+ thru 30"]
    assert cut.breaks == pytest.approx([0.0, 10.0, 20.0, 30.0])
    assert cut.counts == [1, 1, 1]

    special = survival.tcut(
        [5.0, None, float("inf"), float("-inf")],
        [float("-inf"), 10.0, 20.0, float("inf")],
    )
    assert special.values[0] == pytest.approx(5.0)
    assert math.isnan(special.values[1])
    assert special.values[2:] == [float("inf"), float("-inf")]
    assert special.codes == [0, -1, 2, 0]
    assert special.counts == [2, 0, 1]

    repeated = survival.tcut([0.0, 1.0, 1.5, 2.0], [0.0, 1.0, 1.0, 2.0])
    assert repeated.codes == [0, 2, 2, 2]
    assert repeated.counts == [1, 0, 3]

    generated = survival.tcut([1.0, None, 3.0], 2)
    assert generated.codes == [0, -1, 1]
    assert generated.levels == ["Range 1", "Range 2"]
    assert generated.breaks == pytest.approx([0.98, 2.0, 3.02])
    assert generated.counts == [1, 1]

    scaled = survival.tcut(
        [5.0, 15.0, 30.0],
        [0.0, 10.0, 20.0, 30.0],
        labels=["a", "b", "c"],
        scale=365.25,
    )
    assert scaled.codes == [0, 1, 2]
    assert scaled.levels == ["a", "b", "c"]
    assert scaled.breaks == pytest.approx([0.0, 3652.5, 7305.0, 10957.5])
    assert survival.tcut is survival.r_api.tcut

    with pytest.raises(ValueError, match="breaks must have at least 1"):
        survival.tcut([1.0], [])
    with pytest.raises(ValueError, match="contain no NA"):
        survival.tcut([1.0], [0.0, None, 2.0])


def test_r_style_tmerge_matches_mixed_operation_fixture():
    base = {"id": [1, 2], "group": ["a", "b"]}
    span = {"id": [1, 2], "stop": [10, 8]}
    initial = survival.tmerge(base, span, "id", tstop="stop")
    updates = {
        "id": [1, 1, 1, 1, 1, 1, 2, 2, 3],
        "time": [-1, 0, 2, 5, 10, 11, 0, 4, 3],
        "value": [10, 20, 30, 40, 50, 60, 1, 2, 99],
    }

    result = survival.tmerge(
        initial,
        updates,
        "id",
        x=survival.tdc("time", "value", init=-9),
        cx=survival.cumtdc("time", "value", init=100),
        ev=survival.event("time", "value"),
        cev=survival.cumevent("time", "value"),
    )

    assert isinstance(result, survival.TMergeFrame)
    assert result.columns == {
        "id": [1, 1, 1, 2, 2],
        "group": ["a", "a", "a", "b", "b"],
        "tstart": [0.0, 2.0, 5.0, 0.0, 4.0],
        "tstop": [2.0, 5.0, 10.0, 4.0, 8.0],
        "x": [20, 30, 40, 1, 2],
        "cx": [130.0, 160.0, 200.0, 101.0, 103.0],
        "ev": [30, 40, 50, 2, 0],
        "cev": [60.0, 100.0, 150.0, 3.0, 0],
    }
    assert result.tname == {
        "idname": "id",
        "tstartname": "tstart",
        "tstopname": "tstop",
    }
    assert result.tevent == {"ev": 0, "cev": 0}
    assert result.tdcvar == ("x", "cx")
    first_counts = {
        "early": 1,
        "late": 1,
        "gap": 0,
        "within": 3,
        "boundary": 0,
        "leading": 2,
        "trailing": 1,
        "tied": 0,
        "missid": 1,
    }
    later_counts = dict(first_counts, within=0, boundary=3)
    assert result.tcount == {
        "x": first_counts,
        "cx": later_counts,
        "ev": later_counts,
        "cev": later_counts,
    }


def test_r_style_tmerge_classifies_gaps_edges_and_ties():
    base = {"id": [1], "z": [2]}
    span = {"id": [1, 1], "start": [0, 7], "stop": [5, 10]}
    initial = survival.tmerge(base, span, "id", tstart="start", tstop="stop")
    updates = {
        "id": [1] * 10,
        "time": [-1, 0, 3, 5, 6, 7, 8, 8, 10, 11],
        "value": [-1, 0, 3, 5, 6, 7, 8, 70, 10, 11],
    }

    result = survival.tmerge(initial, updates, "id", x=survival.tdc("time", "value"))

    assert result.columns == {
        "id": [1, 1, 1, 1],
        "z": [2, 2, 2, 2],
        "tstart": [0.0, 3.0, 7.0, 8.0],
        "tstop": [3.0, 5.0, 8.0, 10.0],
        "x": [0, 3, 7, 70],
    }
    assert result.tcount["x"] == {
        "early": 1,
        "late": 1,
        "gap": 1,
        "within": 3,
        "boundary": 0,
        "leading": 2,
        "trailing": 2,
        "tied": 1,
        "missid": 0,
    }


def test_r_style_tmerge_delay_and_missing_update_semantics():
    base = {"id": [1], "z": [0]}
    span = {"id": [1], "stop": [10]}
    initial = survival.tmerge(base, span, "id", tstop="stop")
    updates = {"id": [1, 1, 1], "time": [2, 4, 6], "value": [1, None, 3]}

    delayed = survival.tmerge(
        initial,
        updates,
        "id",
        options={"delay": 2, "na.rm": False},
        x=survival.tdc("time", "value", init=-1),
    )
    cumulative = survival.tmerge(
        initial,
        updates,
        "id",
        options={"na.rm": False},
        total=survival.cumtdc("time", "value", init=0),
    )

    assert delayed["tstart"] == [0.0, 4.0, 6.0, 8.0]
    assert delayed["x"][:2] == [-1, 1]
    assert delayed["x"][2] is None
    assert delayed["x"][3] == 3
    assert cumulative["total"][:2] == [0.0, 1.0]
    assert math.isnan(cumulative["total"][2])
    assert math.isnan(cumulative["total"][3])
