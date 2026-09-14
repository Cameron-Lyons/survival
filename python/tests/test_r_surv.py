import importlib
import math
import random

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r_surv = importlib.import_module("survival.r._surv")


def test_strata_matches_r_factor_codes_and_preserves_low_level_call():
    single = survival.strata(["b", "a", "b", None])
    named = survival.strata(["b", "a", "b", None], [2, 1, 1, 1], labels=["x", "y"])
    na_group = survival.strata(
        ["b", "a", "b", None],
        [2, 1, 1, 1],
        labels=["x", "y"],
        na_group=True,
    )
    short = survival.strata(
        ["b", "a", "b", None],
        [2, 1, 1, 1],
        labels=["x", "y"],
        shortlabel=True,
    )
    separated = survival.strata(
        ["b", "a", "b", None],
        [2, 1, 1, 1],
        labels=["x", "y"],
        sep="|",
    )
    legacy = survival.strata([[2, 1, 2], [1, 2, 1]])

    assert type(single).__name__ == "StrataFactor"
    assert single.codes == [2, 1, 2, None]
    assert single.levels == ["a", "b"]
    assert single.labels == ["b", "a", "b", None]
    assert single.counts == [1, 2]

    assert named.codes == [3, 1, 2, None]
    assert named.levels == ["x=a, y=1", "x=b, y=1", "x=b, y=2"]
    assert na_group.codes == [3, 1, 2, 4]
    assert na_group.levels == ["x=a, y=1", "x=b, y=1", "x=b, y=2", "x=NA, y=1"]
    assert short.levels == ["a, 1", "b, 1", "b, 2"]
    assert separated.levels == ["x=a|y=1", "x=b|y=1", "x=b|y=2"]

    assert type(legacy).__name__ == "StrataResult"
    assert legacy.strata == [1, 0, 1]
    with pytest.raises(ValueError, match="same length"):
        survival.strata(["a", "b"], [1])


def test_strata_counts_high_cardinality_levels_once():
    values = [f"group-{idx:04d}" for idx in range(2_000)]

    result = survival.strata(values)

    assert result.codes == list(range(1, len(values) + 1))
    assert result.levels == values
    assert result.counts == [1] * len(values)


def test_strata_matches_r_multicolumn_label_padding():
    result = survival.strata(
        ["b", "a", None, "b"],
        [2, 1, 1, None],
        [True, False, None, True],
        na_group=True,
        labels=["x", "y", "z"],
    )
    short = survival.strata(
        ["b", "a", None, "b"],
        [2, 1, 1, None],
        [True, False, None, True],
        na_group=True,
        shortlabel=True,
    )

    assert result.codes == [2, 1, 4, 3]
    assert result.levels == [
        "x=a, y=1 , z=FALSE",
        "x=b, y=2 , z=TRUE ",
        "x=b, y=NA, z=TRUE ",
        "x=NA, y=1 , z=NA   ",
    ]
    assert short.levels == [
        "a, 1, FALSE",
        "b, 2, TRUE",
        "b, NA, TRUE",
        "NA, 1, NA",
    ]


def test_strata_compaction_matches_python_reference():
    rng = random.Random(20260801)  # noqa: S311
    for _ in range(300):
        n_rows = rng.randrange(0, 40)
        level_counts = [rng.randrange(0, 8) for _ in range(rng.randrange(1, 6))]
        variables = [
            [
                None if level_count == 0 or rng.randrange(5) == 0 else rng.randrange(level_count)
                for _ in range(n_rows)
            ]
            for level_count in level_counts
        ]

        raw_codes = []
        raw_to_parts = {}
        for row_idx in range(n_rows):
            parts = [variable[row_idx] for variable in variables]
            if any(part is None for part in parts):
                raw_codes.append(None)
                continue
            raw_code = 0
            numeric_parts = [int(part) for part in parts if part is not None]
            for part, level_count in zip(numeric_parts, level_counts, strict=True):
                raw_code = raw_code * level_count + part
            raw_codes.append(raw_code)
            raw_to_parts.setdefault(raw_code, numeric_parts)
        observed_raw = sorted(raw_to_parts)
        compact = {raw_code: idx + 1 for idx, raw_code in enumerate(observed_raw)}
        expected_codes = [None if raw_code is None else compact[raw_code] for raw_code in raw_codes]
        expected_parts = [raw_to_parts[raw_code] for raw_code in observed_raw]
        expected_counts = [0] * len(observed_raw)
        for code in expected_codes:
            if code is not None:
                expected_counts[code - 1] += 1

        actual = survival._survival.strata_compact(variables, level_counts)
        assert actual == (expected_codes, expected_parts, expected_counts)


def test_surv_right_censored_response():
    response = survival.Surv([1, 2, 3], [1, 0, 1])
    all_observed = survival.Surv([1, 2, 3])
    right_abbrev = survival.Surv([1, 2, 3], [1, 0, 1], type="r")
    missing_response = survival.Surv([1.0, math.nan, 3.0], [1, 0, 1])

    assert len(response) == 3
    assert response.type == "right"
    assert response.time == pytest.approx((1.0, 2.0, 3.0))
    assert response.status == (1, 0, 1)
    assert all_observed.type == "right"
    assert all_observed.status == (1, 1, 1)
    assert right_abbrev.type == "right"
    assert survival.r_api.is_surv(response) is True
    assert survival.r_api.is_surv({"time": [1, 2, 3]}) is False
    assert survival.is_na_surv(response) == [False, False, False]
    assert survival.is_na_surv(missing_response) == [False, True, False]
    assert survival.format_surv(response) == ["1 ", "2+", "3 "]
    assert survival.format_surv(missing_response) == [" 1 ", "NA+", " 3 "]

    for explicit_type in ("right", "left"):
        with pytest.raises(ValueError, match="one-argument Surv"):
            survival.Surv([1, 2, 3], type=explicit_type)
    with pytest.raises(ValueError, match="ambiguous"):
        survival.Surv([1, 2], [1, 0], type="i")
    with pytest.raises(TypeError, match="Surv object"):
        survival.is_na_surv([1.0, 2.0])
    with pytest.raises(TypeError, match="Surv object"):
        survival.format_surv([1.0, 2.0])


def test_surv_multistate_responses_match_r_shape_and_formatting():
    right = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        ["censor", "ill", "death", "ill"],
        type="mstate",
    )
    counting = survival.Surv(
        [0.0, 0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        ["ill", "censor", "death", "ill"],
        type="mstate",
    )
    missing = survival.Surv(
        [1.0, 2.0, 3.0],
        [None, "censor", "death"],
        type="m",
    )

    assert right.type == "mright"
    assert right.status == (0, 2, 1, 2)
    assert right.states == ("death", "ill")
    assert survival.format_surv(right) == ["1+     ", "2:ill  ", "3:death", "4:ill  "]

    assert counting.type == "mcounting"
    assert counting.status == (2, 0, 1, 2)
    assert counting.states == ("death", "ill")
    assert survival.format_surv(counting) == [
        "(0,1:ill]".ljust(11),
        "(0,2+]".ljust(11),
        "(1,3:death]",
        "(2,4:ill]".ljust(11),
    ]

    assert missing.status == (None, 0, 1)
    assert missing.states == ("death",)
    assert survival.is_na_surv(missing) == [True, False, False]
    assert survival.format_surv(missing) == ["1?     ", "2+     ", "3:death"]


def test_surv_multistate_helpers_preserve_state_metadata():
    response = survival.Surv(
        [1.0, 1.0 + 1e-10, 2.0],
        ["censor", "ill", "death"],
        type="mstate",
    )

    adjusted = survival.aeqSurv(response)
    subset = r_surv._subset_surv(response, [2, 0])

    assert adjusted.type == "mright"
    assert adjusted.states == response.states
    assert adjusted.status == response.status
    assert adjusted.time[0] == adjusted.time[1]
    assert subset.type == "mright"
    assert subset.states == response.states
    assert subset.status == (1, 0)
    curve = survival.survfit(response)
    assert isinstance(curve, survival.SurvfitMultiStateResult)
    assert curve.states == ("(s0)", *response.states)
    assert curve.time == pytest.approx([1.0, 2.0])


def test_surv_multistate_preserves_categorical_level_order():
    pandas = pytest.importorskip("pandas")
    events = pandas.Categorical(
        ["censor", "ill", "censor"],
        categories=["censor", "ill", "death"],
    )

    response = survival.Surv([1.0, 2.0, 3.0], events)
    explicit_right = survival.Surv([1.0, 2.0, 3.0], events, type="right")
    explicit_counting = survival.Surv(
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        events,
        type="counting",
    )

    assert response.status == (0, 1, 0)
    assert response.states == ("ill", "death")
    assert explicit_right.type == "mright"
    assert explicit_right.states == response.states
    assert explicit_counting.type == "mcounting"
    assert explicit_counting.states == response.states

    numeric = survival.Surv([1.0, 2.0, 3.0], [0, 10, 2], type="mstate")
    numeric_strings = survival.Surv(
        [1.0, 2.0, 3.0],
        ["0", "10", "2"],
        type="mstate",
    )
    assert numeric.states == ("2", "10")
    assert numeric.status == (0, 2, 1)
    assert numeric_strings.states == ("10", "2")
    assert numeric_strings.status == (0, 1, 2)


def test_surv2_response_matches_r_multistate_shape():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    response = survival.Surv2([1.0, 2.0, 3.0], ["a", "b", "c"])
    missing = survival.Surv2([1.0, math.nan, 3.0], [None, "b", "c"], repeated=True)
    repeated_first = survival.Surv2([1.0, 2.0], ["a", "a"], repeated="first")
    categorical = survival.Surv2(
        [1.0, 2.0, 3.0],
        Factor(
            ["entry", "death", "ill"],
            ["censor", "entry", "ill", "death"],
        ),
    )

    assert len(response) == 3
    assert response.time == pytest.approx((1.0, 2.0, 3.0))
    assert response.status == (0, 1, 2)
    assert response.states == ("b", "c")
    assert response.repeated is False
    assert survival.is_surv(response) is False
    assert survival.format_surv(response) == ["1+ ", "2:b", "3:c"]
    assert missing.status == (None, 0, 1)
    assert missing.states == ("c",)
    assert missing.repeated is True
    assert repeated_first.repeated == "first"
    assert survival.is_na_surv(missing) == [True, True, False]
    assert survival.format_surv(missing) == ["1? ", "NA+", "3:c"]
    assert categorical.status == (1, 3, 2)
    assert categorical.states == ("entry", "ill", "death")

    with pytest.raises(ValueError, match="different lengths"):
        survival.Surv2([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="repeated"):
        survival.Surv2([1.0], ["a"], repeated=[True])


def test_surv2data_converts_timeline_rows_to_counting_transitions():
    result = survival.Surv2data(
        [0.0, 2.0, 5.0, 0.0, 3.0],
        [1, 2, 3, 1, 0],
        states=["entry", "ill", "death"],
        id=[1, 1, 1, 2, 2],
    )

    assert result["row"] == [0, 1, 3]
    assert result["start"] == pytest.approx([0.0, 2.0, 0.0])
    assert result["stop"] == pytest.approx([2.0, 5.0, 3.0])
    assert result["status"] == [2, 3, 0]
    assert result["id"] == [1, 1, 2]
    assert result["istate"] == [1, 2, 1]
    assert result["states"] == ["entry", "ill", "death"]
    assert result["type"] == "mcounting"

    shuffled = survival.Surv2data(
        [5.0, 0.0, 3.0, 2.0, 0.0],
        [3, 1, 0, 2, 1],
        states=["entry", "ill", "death"],
        id=["a", "a", "b", "a", "b"],
    )
    assert shuffled["row"] == [1, 3, 4]
    assert shuffled["start"] == pytest.approx([0.0, 2.0, 0.0])
    assert shuffled["stop"] == pytest.approx([2.0, 5.0, 3.0])
    assert shuffled["status"] == [2, 3, 0]
    assert shuffled["id"] == ["a", "a", "b"]
    assert shuffled["istate"] == [1, 2, 1]

    repeated_result = survival.Surv2data(
        [0.0, 1.0, 2.0],
        [1, 1, 2],
        states=["entry", "death"],
        id=[1, 1, 1],
        repeated=False,
    )
    assert repeated_result["status"] == [0, 2]

    repeated_first_result = survival.Surv2data(
        [0.0, 1.0, 2.0, 3.0],
        [1, 2, 1, 2],
        states=["entry", "death"],
        id=[1, 1, 1, 1],
        repeated="first",
    )
    assert repeated_first_result["status"] == [2, 0, 0]

    ordinary_result = survival.Surv2data(
        [0.0, 0.0, 2.0, 3.0, 5.0, 6.0],
        [0, 0, 1, 1, 1, 0],
        id=[1, 2, 1, 2, 1, 2],
    )
    assert ordinary_result["row"] == [0, 1, 2, 3]
    assert ordinary_result["status"] == [1, 1, 1, 0]

    with pytest.raises(ValueError, match="duplicated time"):
        survival.Surv2data([0.0, 0.0], [1, 2], states=["a", "b"], id=[1, 1])


def test_totimeline_expands_counting_rows_to_state_timeline():
    result = survival.totimeline(
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 2, 0],
        states=["ill", "death"],
        id=[1, 1, 2],
        istate=["entry", "ill", "entry"],
        istate_levels=["entry", "ill", "death"],
    )

    assert result["time"] == pytest.approx([0.0, 2.0, 5.0, 0.0, 3.0])
    assert result["status"] == [1, 2, 3, 1, 0]
    assert result["data_row"] == [0, 1, 1, 2, 2]
    assert result["state_levels"] == ["censor", "entry", "ill", "death"]

    default_state_result = survival.totimeline(
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 2, 0],
        states=["ill", "death"],
        id=[1, 1, 2],
    )
    assert default_state_result["status"] == [1, 2, 3, 1, 0]
    assert default_state_result["state_levels"] == ["censor", "(s0)", "ill", "death"]

    with pytest.raises(ValueError, match="same length"):
        survival.totimeline([0.0], [1.0, 2.0], [1], states=["ill"], id=[1])


def test_fromtimeline_builds_intervals_and_covariate_row_maps():
    result = survival.fromtimeline(
        [0.0, 2.0, 5.0, 0.0, 3.0, 6.0],
        [1, 1, 1, 1, 1, 0],
        id=[1, 1, 1, 2, 2, 2],
        data={
            "z": ["A", "A", "A", "B", "B", "B"],
            "x": [10, 11, 12, 20, 21, 22],
            "id": [1, 1, 1, 2, 2, 2],
        },
    )

    assert result["start"] == pytest.approx([0.0, 2.0, 0.0, 3.0])
    assert result["stop"] == pytest.approx([2.0, 5.0, 3.0, 6.0])
    assert result["status"] == [1, 1, 1, 0]
    assert result["istate"] == [1, 1, 1, 1]
    assert result["static"] == [True, False, True]
    assert result["static_row"] == [0, 0, 3, 3]
    assert result["dynamic_row"] == [0, 1, 3, 4]

    multistate_result = survival.fromtimeline(
        [0.0, 2.0, 5.0, 0.0, 3.0, 6.0],
        [1, 2, 3, 1, 2, 0],
        id=[1, 1, 1, 2, 2, 2],
        states=["entry", "ill", "death"],
    )
    assert multistate_result["status"] == [2, 3, 2, 0]
    assert multistate_result["istate"] == [1, 2, 1, 2]
    assert multistate_result["state_levels"] == ["censor", "entry", "ill", "death"]
    assert multistate_result["istate_levels"] == ["entry", "ill", "death"]

    with pytest.raises(ValueError, match="censored state"):
        survival.fromtimeline([0.0, 1.0], [0, 1], id=[1, 1])


def test_surv_accepts_named_response_arguments():
    positional = survival.Surv([1, 2, 3], [1, 2, 1])
    named = survival.Surv(time=[1, 2, 3], status=[1, 2, 1])
    all_observed = survival.Surv(time1=[1, 2, 3])
    counting = survival.Surv(start=[0.0, 1.0], stop=[2.0, 3.0], event=[1, 0])
    interval = survival.Surv(
        time=[1.0, 2.0],
        time2=[1.5, 3.0],
        event=[3, 0],
        type="interval",
    )
    interval2 = survival.Surv(
        time=[float("-inf"), 2.0],
        time2=[1.0, float("inf")],
        type="interval2",
    )

    assert named.time == pytest.approx(positional.time)
    assert named.status == positional.status
    assert all_observed.status == (1, 1, 1)
    assert counting.type == "counting"
    assert counting.start == pytest.approx((0.0, 1.0))
    assert counting.time == pytest.approx((2.0, 3.0))
    assert interval.type == "interval"
    assert interval.time2 == pytest.approx((1.5, 3.0))
    assert interval.status == (3, 0)
    assert interval2.type == "interval2"
    assert interval2.status == (2, 0)


def test_surv_format_and_missingness_cover_censoring_types():
    left = survival.Surv([1.0, math.nan, 3.0], [1, 0, 1], type="left")
    counting = survival.Surv([0.0, math.nan, 1.0], [1.0, 2.0, math.nan], [1, 0, 1])
    interval = survival.Surv(
        [1.0, math.nan, 3.0],
        [1.0, 2.0, math.nan],
        [1, 3, 0],
        type="interval",
    )
    interval2 = survival.Surv(
        [float("-inf"), 1.0, 2.0],
        [1.0, 2.0, float("inf")],
        type="interval2",
    )

    assert survival.is_na_surv(left) == [False, True, False]
    assert survival.format_surv(left) == [" 1 ", "NA-", " 3 "]

    assert survival.is_na_surv(counting) == [False, True, True]
    counting_labels = survival.format_surv(counting)
    assert counting_labels[0].strip().startswith("(")
    assert counting_labels[0].strip().endswith("1]")
    assert "+]" in counting_labels[1]

    assert survival.is_na_surv(interval) == [False, True, True]
    interval_labels = survival.format_surv(interval)
    assert interval_labels[0].strip() == "1"
    assert interval_labels[1].strip() == "[NA, 2]"
    assert interval_labels[2].strip() == "3+"

    assert survival.is_na_surv(interval2) == [False, False, False]
    assert [value.strip() for value in survival.format_surv(interval2)] == [
        "1-",
        "[1, 2]",
        "2+",
    ]


def test_surv_rejects_invalid_named_response_arguments():
    with pytest.raises(ValueError, match="multiple time="):
        survival.Surv(time=[1.0], start=[0.0], event=[1])
    with pytest.raises(ValueError, match="multiple event="):
        survival.Surv(time=[1.0], event=[1], status=[1])
    with pytest.raises(TypeError, match="must not mix"):
        survival.Surv([1.0], event=[1])
    with pytest.raises(ValueError, match="requires time="):
        survival.Surv(time2=[1.0], event=[1])
    with pytest.raises(ValueError, match="event is required"):
        survival.Surv(time=[1.0], event=None)


def test_surv_accepts_r_one_two_event_coding():
    response = survival.Surv([1, 2, 3], [1, 2, 1])

    assert response.status == (0, 1, 0)

    with pytest.raises(ValueError, match="0/1 or 1/2"):
        survival.Surv([1, 2], [0, 2])
    with pytest.raises(ValueError, match="0/1 or 1/2"):
        survival.Surv([1, 2], [0.5, 1.0])


def test_surv_origin_shifts_supported_time_columns():
    all_observed = survival.Surv([11.0, 12.0], origin=10.0)
    right = survival.Surv([11.0, 12.0], [1, 0], origin=10.0)
    counting = survival.Surv([10.0, 11.0], [12.0, 14.0], [1, 0], origin=10.0)
    interval = survival.Surv([11.0, 12.0], [11.0, 15.0], [1, 3], type="interval", origin=10.0)
    interval2 = survival.Surv(
        [float("-inf"), 12.0, 13.0],
        [11.0, float("inf"), 15.0],
        type="interval2",
        origin=10.0,
    )

    assert all_observed.time == pytest.approx((1.0, 2.0))
    assert all_observed.status == (1, 1)
    assert right.time == pytest.approx((1.0, 2.0))
    assert counting.start == pytest.approx((0.0, 1.0))
    assert counting.time == pytest.approx((2.0, 4.0))
    assert interval.time == pytest.approx((1.0, 2.0))
    assert interval.time2 == pytest.approx((1.0, 5.0))
    assert interval2.time[0] == float("-inf")
    assert interval2.time[1:] == pytest.approx((2.0, 3.0))
    assert interval2.time2[:2] == pytest.approx((1.0, float("inf")))
    assert interval2.time2[2] == pytest.approx(5.0)

    with pytest.raises(TypeError, match="origin must be numeric"):
        survival.Surv([1.0, 2.0], origin="baseline")
    with pytest.raises(ValueError, match="origin must be finite"):
        survival.Surv([1.0, 2.0], origin=float("nan"))
