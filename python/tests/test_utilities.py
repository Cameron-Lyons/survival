import pytest

from .helpers import setup_survival_import

survival_package = setup_survival_import()
survival = survival_package.data_prep


def test_collapse():
    y = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0]
    x = [1, 1, 1, 1]
    istate = [0, 0, 0, 0]
    subject_id = [1, 1, 2, 2]
    wt = [1.0, 1.0, 1.0, 1.0]
    order = [0, 1, 2, 3]

    result = survival.collapse(y, x, istate, subject_id, wt, order)
    assert isinstance(result, dict)
    assert result["matrix"] == [[1, 2], [3, 3], [4, 4]]
    assert result["dimnames"] == ["start", "end"]

    with pytest.raises(ValueError, match="y must have 3 columns"):
        survival.collapse(y[:-1], x, istate, subject_id, wt, order)
    with pytest.raises(ValueError, match="x length mismatch"):
        survival.collapse(y, x[:-1], istate, subject_id, wt, order)
    with pytest.raises(ValueError, match="y values must be finite"):
        survival.collapse([float("nan"), *y[1:]], x, istate, subject_id, wt, order)
    with pytest.raises(ValueError, match="wt values must be finite"):
        survival.collapse(y, x, istate, subject_id, [float("inf"), 1.0, 1.0, 1.0], order)
    with pytest.raises(ValueError, match="order values must be non-negative"):
        survival.collapse(y, x, istate, subject_id, wt, [-1, 1, 2, 3])
    with pytest.raises(ValueError, match="order values must be less than"):
        survival.collapse(y, x, istate, subject_id, wt, [0, 1, 2, 4])
    with pytest.raises(ValueError, match="order must be a permutation"):
        survival.collapse(y, x, istate, subject_id, wt, [0, 1, 1, 3])


def test_cluster_and_strata_public_apis():
    clusters = survival.cluster([2, 2, 1, 3])
    string_clusters = survival.cluster_str(["b", "a", "b"])
    numeric_strata = survival.strata([[2, 1, 2], [1, 2, 1]])
    string_strata = survival.strata_str([["b", "a", "b"], ["x", "y", "x"]])

    assert clusters.cluster_ids == [0, 0, 1, 2]
    assert clusters.levels == ["2", "1", "3"]
    assert clusters.cluster_sizes == [2, 1, 1]

    assert string_clusters.cluster_ids == [0, 1, 0]
    assert string_clusters.levels == ["b", "a"]
    assert string_clusters.cluster_sizes == [2, 1]

    assert numeric_strata.strata == [1, 0, 1]
    assert numeric_strata.levels == ["v1=1, v2=2", "v1=2, v2=1"]
    assert numeric_strata.counts == [1, 2]
    assert numeric_strata.n_strata == 2

    assert string_strata.strata == [1, 0, 1]
    assert string_strata.levels == ["a, y", "b, x"]
    assert string_strata.counts == [1, 2]
    assert string_strata.n_strata == 2

    with pytest.raises(ValueError, match="Variable 1 has length"):
        survival.strata([[1, 2], [1]])


def test_tmerge_family_public_apis():
    merged = survival.tmerge(
        [1, 1, 2],
        [1.0, 2.0, 1.0],
        [0.0, float("nan"), 0.0],
        [1, 1, 2],
        [0.5, 1.5, 0.5],
        [2.0, 3.0, 4.0],
    )
    indices = survival.tmerge2(
        [1, 1, 2],
        [1.0, 2.0, 1.0],
        [1, 1, 2],
        [0.5, 1.5, 0.5],
    )
    carry = survival.tmerge3([1, 1, 1, 2, 2], [False, True, False, True, False])

    assert merged == pytest.approx([2.0, 5.0, 4.0])
    assert indices == [1, 2, 3]
    assert carry == [1, 1, 3, 0, 5]

    negative_ids = survival.tmerge(
        [-1, -1],
        [1.0, 2.0],
        [float("nan"), float("nan")],
        [-1],
        [0.5],
        [4.0],
    )
    assert negative_ids == pytest.approx([4.0, 4.0])

    with pytest.raises(ValueError, match="time1 must have same length as id"):
        survival.tmerge([1], [], [0.0], [], [], [])
    with pytest.raises(ValueError, match="ntime must have same length as nid"):
        survival.tmerge2([1], [1.0], [1], [])
    with pytest.raises(ValueError, match="miss must have same length as id"):
        survival.tmerge3([1], [])
    with pytest.raises(ValueError, match="time1 values must be finite"):
        survival.tmerge([1], [float("nan")], [0.0], [], [], [])
    with pytest.raises(ValueError, match="newx values may be finite or NaN"):
        survival.tmerge([1], [1.0], [float("inf")], [], [], [])
    with pytest.raises(ValueError, match="id must be sorted in non-decreasing order"):
        survival.tmerge([2, 1], [1.0, 1.0], [0.0, 0.0], [], [], [])
    with pytest.raises(ValueError, match="time1 must be non-decreasing within id"):
        survival.tmerge([1, 1], [2.0, 1.0], [0.0, 0.0], [], [], [])
    with pytest.raises(ValueError, match="ntime values must be finite"):
        survival.tmerge([1], [1.0], [0.0], [1], [float("inf")], [1.0])
    with pytest.raises(ValueError, match="x values must be finite"):
        survival.tmerge([1], [1.0], [0.0], [1], [0.5], [float("nan")])
    with pytest.raises(ValueError, match="nid must be sorted in non-decreasing order"):
        survival.tmerge2([1], [1.0], [2, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="id must be sorted in non-decreasing order"):
        survival.tmerge3([2, 1], [False, False])


def test_survsplit_public_api_and_validation():
    result = survival.survsplit(
        [0.0],
        [10.0],
        [7.0, 3.0, 3.0],
    )
    boundary_result = survival.survsplit([0.0], [10.0], [0.0, 3.0, 10.0])

    assert result.row == [1, 1, 1]
    assert result.interval == [1, 2, 3]
    assert result.start == pytest.approx([0.0, 3.0, 7.0])
    assert result.end == pytest.approx([3.0, 7.0, 10.0])
    assert result.censor == [True, True, False]

    assert boundary_result.row == [1, 1]
    assert boundary_result.interval == [1, 2]
    assert boundary_result.start == pytest.approx([0.0, 3.0])
    assert boundary_result.end == pytest.approx([3.0, 10.0])
    assert boundary_result.censor == [True, False]

    with pytest.raises(ValueError, match="tstart and tstop must have same length"):
        survival.survsplit([0.0], [], [1.0])
    with pytest.raises(ValueError, match="cut must be a vector of finite numbers"):
        survival.survsplit([0.0], [10.0], [3.0, float("nan")])
    with pytest.raises(ValueError, match="cut must be a vector of finite numbers"):
        survival.survsplit([0.0], [10.0], [float("inf")])
    with pytest.raises(ValueError, match="not infinite"):
        survival.survsplit([0.0], [float("inf")], [5.0])
    with pytest.raises(ValueError, match="not infinite"):
        survival.survsplit([float("-inf")], [10.0], [5.0])


def test_surv2data_and_survcondense_public_apis():
    surv2data = survival.surv2data(
        [2, 1, 1, 2],
        [3.0, 0.0, 2.0, 0.0],
        [5.0, 4.0, 4.0, 5.0],
        [0, 1, 1, 0],
    )
    condensed = survival.survcondense(
        [2, 1, 1, 2],
        [0.0, 0.0, 5.0, 3.0],
        [3.0, 5.0, 8.0, 5.0],
        [1, 0, 0, 0],
    )

    assert surv2data.id == [1, 1, 2, 2]
    assert surv2data.time1 == pytest.approx([0.0, 2.0, 0.0, 3.0])
    assert surv2data.time2 == pytest.approx([2.0, 4.0, 3.0, 5.0])
    assert surv2data.status == [0, 1, 0, 0]
    assert surv2data.row_index == [2, 3, 4, 1]

    assert condensed.id == [1, 2, 2]
    assert condensed.time1 == pytest.approx([0.0, 0.0, 3.0])
    assert condensed.time2 == pytest.approx([8.0, 3.0, 5.0])
    assert condensed.status == [0, 1, 0]
    assert condensed.row_map == [[2, 3], [1], [4]]

    with pytest.raises(ValueError, match="time must have same length as id"):
        survival.surv2data([1], [])
    with pytest.raises(
        ValueError,
        match="event_time and event_status must both be provided or both be None",
    ):
        survival.surv2data([1], [0.0], [1.0])
    with pytest.raises(ValueError, match="event_status must have same length as id"):
        survival.surv2data([1], [0.0], [1.0], [])
    with pytest.raises(ValueError, match="time contains NaN"):
        survival.surv2data([1], [float("nan")])
    with pytest.raises(ValueError, match="event_time contains non-finite"):
        survival.surv2data([1], [0.0], [float("inf")], [1])
    with pytest.raises(ValueError, match="event_status must contain only 0/1"):
        survival.surv2data([1], [0.0], [1.0], [2])
    with pytest.raises(ValueError, match="event_time/event_status must be constant within id"):
        survival.surv2data([1, 1], [0.0, 1.0], [3.0, 4.0], [1, 1])
    with pytest.raises(ValueError, match="event_time must be >= time"):
        survival.surv2data([1], [2.0], [1.0], [1])
    with pytest.raises(ValueError, match="duplicated time values for a single id"):
        survival.surv2data([1, 1], [2.0, 2.0])
    with pytest.raises(ValueError, match="time1 must have same length as id"):
        survival.survcondense([1], [], [1.0], [0])
    with pytest.raises(ValueError, match="time2 must have same length as id"):
        survival.survcondense([1], [0.0], [], [0])
    with pytest.raises(ValueError, match="status must have same length as id"):
        survival.survcondense([1], [0.0], [1.0], [])
    with pytest.raises(ValueError, match="time1 contains NaN"):
        survival.survcondense([1], [float("nan")], [1.0], [0])
    with pytest.raises(ValueError, match="time2 contains non-finite"):
        survival.survcondense([1], [0.0], [float("inf")], [0])
    with pytest.raises(ValueError, match="time1 must be <= time2"):
        survival.survcondense([1], [2.0], [1.0], [0])
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.survcondense([1], [0.0], [1.0], [2])
    with pytest.raises(ValueError, match="intervals must not overlap within id"):
        survival.survcondense([1, 1], [0.0, 4.0], [5.0, 6.0], [0, 0])


def test_aeq_surv_neardate_and_tcut_public_apis():
    adjusted = survival.aeq_surv([1.0, 1.0 + 1e-10, 2.0], 1e-8)
    transitive_adjusted = survival.aeq_surv([1.0, 1.0 + 9e-9, 1.0 + 18e-9], 1e-8)
    relative_adjusted = survival.aeq_surv([1e9, 1e9 + 1.0, 1e9 + 20.0], 1e-8)
    no_adjust = survival.aeq_surv([1.0, 1.0 + 1e-10], -1.0)
    nearest = survival.neardate([1, 1, 2], [4.0, 12.0, 7.0], [1, 1, 2], [5.0, 10.0, 9.0])
    nearest_tie = survival.neardate(
        [1, 1, 1],
        [15.0, 10.0, 11.0],
        [1, 1, 1, 1],
        [10.0, 20.0, 10.0, 12.0],
    )
    nearest_str = survival.neardate_str(["a"], [4.0], ["a"], [1.0], "prior")
    nearest_after_prefix = survival.neardate([1], [15.0], [1, 1], [10.0, 20.0], "a")
    nearest_prior_prefix = survival.neardate([1], [15.0], [1, 1], [10.0, 20.0], "pr")
    nearest_closest_prefix = survival.neardate([1], [18.0], [1, 1], [10.0, 20.0], "cl")
    cut = survival.tcut([5.0, 15.0, 30.0], [0.0, 10.0, 20.0, 30.0])
    cut_boundaries = survival.tcut([0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0])
    cut_generated = survival.tcut([5.0, 15.0, 25.0], 3.0)
    cut_duplicate_break = survival.tcut([5.0, 15.0, 25.0], [0.0, 10.0, 10.0, 30.0])
    expanded = survival.tcut_expand([0.0], [25.0], [0.0, 10.0, 20.0, 30.0])
    expanded_edges = survival.tcut_expand([-5.0, 35.0], [25.0, 40.0], [0.0, 10.0, 20.0])

    assert adjusted.time[0] == pytest.approx(adjusted.time[1])
    assert adjusted.adjusted_count == 1
    assert adjusted.adjusted_indices == [1]
    assert transitive_adjusted.time == pytest.approx([1.0, 1.0, 1.0])
    assert transitive_adjusted.adjusted_indices == [1, 2]
    assert relative_adjusted.time == pytest.approx([1e9, 1e9, 1e9 + 20.0])
    assert relative_adjusted.adjusted_indices == [1]
    assert no_adjust.time == pytest.approx([1.0, 1.0 + 1e-10])
    assert no_adjust.adjusted_count == 0

    assert nearest.indices == [0, 1, 2]
    assert nearest.distances == pytest.approx([1.0, 2.0, 2.0])
    assert nearest.n_matched == 3
    assert nearest_tie.indices == [3, 0, 0]
    assert nearest_tie.distances == pytest.approx([3.0, 0.0, 1.0])
    assert nearest_str.indices == [0]
    assert nearest_str.distances == pytest.approx([3.0])
    assert nearest_after_prefix.indices == [1]
    assert nearest_after_prefix.distances == pytest.approx([5.0])
    assert nearest_prior_prefix.indices == [0]
    assert nearest_prior_prefix.distances == pytest.approx([5.0])
    assert nearest_closest_prefix.indices == [1]
    assert nearest_closest_prefix.distances == pytest.approx([2.0])

    assert cut.codes == [0, 1, 2]
    assert cut.counts == [1, 1, 1]
    assert cut.breaks == pytest.approx([0.0, 10.0, 20.0, 30.0])
    assert cut_boundaries.codes == [0, 1, 2, 2]
    assert cut_boundaries.counts == [1, 1, 2]
    assert cut_generated.codes == [0, 1, 2]
    assert cut_generated.levels == ["Range 1", "Range 2", "Range 3"]
    assert cut_generated.breaks == pytest.approx([4.8, 11.6, 18.4, 25.2])
    assert cut_generated.counts == [1, 1, 1]
    assert cut_duplicate_break.codes == [0, 2, 2]
    assert cut_duplicate_break.counts == [1, 0, 2]

    start, stop, codes, original = expanded
    assert start == pytest.approx([0.0, 10.0, 20.0])
    assert stop == pytest.approx([10.0, 20.0, 25.0])
    assert codes == [0, 1, 2]
    assert original == [0, 0, 0]
    edge_start, edge_stop, edge_codes, edge_original = expanded_edges
    assert edge_start == pytest.approx([-5.0, 0.0, 10.0, 20.0, 35.0])
    assert edge_stop == pytest.approx([0.0, 10.0, 20.0, 25.0, 40.0])
    assert edge_codes == [-1, 0, 1, 2, 2]
    assert edge_original == [0, 0, 0, 0, 1]

    with pytest.raises(ValueError, match="best must be 'prior', 'after', or 'closest'"):
        survival.neardate([1], [1.0], [1], [1.0], "sideways")
    with pytest.raises(ValueError, match="best must be 'prior', 'after', or 'closest'"):
        survival.neardate([1], [1.0], [1], [1.0], "")
    with pytest.raises(ValueError, match="labels length"):
        survival.tcut([1.0], [0.0, 1.0, 2.0], ["only-one"])
    with pytest.raises(ValueError, match="tolerance must be finite"):
        survival.aeq_surv([1.0], float("inf"))
    with pytest.raises(ValueError, match="time values must be finite"):
        survival.aeq_surv([1.0, float("nan")])
    with pytest.raises(ValueError, match="date1 values must be finite"):
        survival.neardate([1], [float("nan")], [1], [1.0])
    with pytest.raises(ValueError, match="date2 values must be finite"):
        survival.neardate_str(["a"], [1.0], ["a"], [float("inf")])
    with pytest.raises(ValueError, match="value values must be finite"):
        survival.tcut([float("nan")], [0.0, 1.0])
    with pytest.raises(ValueError, match="breaks must be given in ascending order"):
        survival.tcut([0.5], [2.0, 1.0])
    with pytest.raises(ValueError, match="Must specify at least one interval"):
        survival.tcut([0.5], 0.0)
    with pytest.raises(ValueError, match="start values must be finite"):
        survival.tcut_expand([float("nan")], [1.0], [0.0])
    with pytest.raises(ValueError, match="cuts must contain unique values"):
        survival.tcut_expand([0.0], [1.0], [0.0, 0.0])


def test_timeline_public_apis_and_validation():
    timeline = survival.to_timeline(
        [1, 1, 2],
        [0.0, 5.0, 0.0],
        [5.0, 10.0, 10.0],
        [0, 1, 2],
        [0.0, 5.0, 10.0],
    )
    intervals = survival.from_timeline(timeline.id, timeline.states, timeline.time_points)
    precise = survival.to_timeline([1, 1], [0.0, 1.0004], [1.0004, 1.0008], [0, 1])

    assert timeline.id == [1, 2]
    assert timeline.time_points == pytest.approx([0.0, 5.0, 10.0])
    assert timeline.states == [[0, 1, 1], [2, 2, 2]]
    assert intervals.id == [1, 1, 2, 2]
    assert intervals.time1 == pytest.approx([0.0, 5.0, 0.0, 5.0])
    assert intervals.time2 == pytest.approx([5.0, 10.0, 5.0, 10.0])
    assert intervals.status == [0, 1, 2, 2]
    assert precise.time_points == pytest.approx([0.0, 1.0004, 1.0008])
    assert precise.states == [[0, 1, 1]]

    with pytest.raises(ValueError, match="time1 must have same length as id"):
        survival.to_timeline([1], [], [1.0], [0])
    with pytest.raises(ValueError, match="time1 contains NaN"):
        survival.to_timeline([1], [float("nan")], [1.0], [0])
    with pytest.raises(ValueError, match="time2 contains non-finite"):
        survival.to_timeline([1], [0.0], [float("inf")], [0])
    with pytest.raises(ValueError, match="time1 must be <= time2"):
        survival.to_timeline([1], [2.0], [1.0], [0])
    with pytest.raises(ValueError, match="intervals must not overlap within id"):
        survival.to_timeline([1, 1], [0.0, 4.0], [5.0, 6.0], [0, 1])
    with pytest.raises(ValueError, match="time_points must be strictly increasing"):
        survival.to_timeline([1], [0.0], [1.0], [0], [0.0, 0.0])
    with pytest.raises(ValueError, match="states must have one row per id"):
        survival.from_timeline([1, 2], [[0]], [0.0])
    with pytest.raises(ValueError, match="states row 0 has length 2 but expected 1"):
        survival.from_timeline([1], [[0, 1]], [0.0])
    with pytest.raises(ValueError, match="time_points contains NaN"):
        survival.from_timeline([1], [[0]], [float("nan")])
    with pytest.raises(ValueError, match="time_points must be strictly increasing"):
        survival.from_timeline([1], [[0, 1]], [1.0, 0.0])


def test_rttright_public_apis_and_validation():
    result = survival.rttright([3.0, 1.0, 2.0], [1, 0, 1])
    stratified = survival.rttright_stratified(
        [3.0, 1.0, 2.0, 1.5],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
    )

    assert result.time == pytest.approx([1.0, 2.0, 3.0])
    assert result.status == [0, 1, 1]
    assert result.weights == pytest.approx([0.0, 0.5, 0.5])
    assert result.order == [1, 2, 0]

    raw = survival.rttright([1.0, 2.0, 3.0], [0, 1, 1], renorm=False)
    assert raw.weights == pytest.approx([0.0, 1.5, 1.5])

    weighted = survival.rttright([1.0, 2.0, 3.0], [0, 1, 1], [2.0, 1.0, 3.0])
    weighted_raw = survival.rttright(
        [1.0, 2.0, 3.0],
        [0, 1, 1],
        [2.0, 1.0, 3.0],
        renorm=False,
    )
    assert weighted.weights == pytest.approx([0.0, 0.25, 0.75])
    assert weighted_raw.weights == pytest.approx([0.0, 1.5, 4.5])

    tied = survival.rttright([1.0, 2.0, 2.0, 3.0], [0, 1, 1, 1])
    assert tied.weights == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3])

    fixed = survival.rttright([1.0, 1.0 + 5e-10, 2.0], [0, 1, 1], renorm=False)
    exact = survival.rttright(
        [1.0, 1.0 + 5e-10, 2.0],
        [0, 1, 1],
        timefix=False,
        renorm=False,
    )
    assert fixed.weights == pytest.approx([0.0, 1.0, 2.0])
    assert exact.weights == pytest.approx([0.0, 1.5, 1.5])

    assert stratified.time == pytest.approx([3.0, 1.0, 2.0, 1.5])
    assert stratified.status == [1, 0, 1, 1]
    assert stratified.weights == pytest.approx([1.0, 0.0, 0.5, 0.5])
    assert stratified.order == [1, 0, 3, 2]

    with pytest.raises(ValueError, match="time and status must have same length"):
        survival.rttright([1.0], [1, 0])
    with pytest.raises(ValueError, match="weights must have same length as time"):
        survival.rttright_stratified([1.0], [1], [0], [1.0, 2.0])
    with pytest.raises(ValueError, match="time contains NaN"):
        survival.rttright([float("nan")], [1])
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.rttright([1.0], [2])
    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.rttright([1.0], [1], [-1.0])
    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.rttright_stratified([1.0], [1], [0], [float("inf")])
    with pytest.raises(ValueError, match="weights must have positive sum"):
        survival.rttright([1.0], [1], [0.0])


def test_agexact_public_api_and_validation():
    result = survival_package.agexact(
        0,
        1,
        1,
        [0.0],
        [1.0],
        [1],
        [1.0],
        [0.0],
        [0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0, 0.0],
        [0.0] * 5,
        [0, 0],
        1e-9,
        1e-9,
        [1],
    )

    assert sorted(result) == [
        "beta",
        "covar",
        "flag",
        "imat",
        "loglik",
        "maxiter",
        "means",
        "sctest",
        "u",
    ]
    assert result["beta"] == pytest.approx([0.0])
    assert result["loglik"] == pytest.approx([0.0, 0.0])
    assert result["flag"] == 0

    fitted = survival_package.agexact(
        20,
        6,
        1,
        [0.0] * 6,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 0, 1, 1, 0, 1],
        [0.2, 1.1, -0.4, 0.8, 1.5, -0.2],
        [0.0] * 6,
        [0] * 6,
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0, 0.0],
        [0.0] * 10,
        [0] * 12,
        1e-9,
        1e-9,
        [1],
    )
    assert fitted["beta"] == pytest.approx([-0.716230334066463], abs=1e-10)
    assert fitted["loglik"] == pytest.approx([-4.276666119016055, -3.923517065659742], abs=1e-10)
    assert fitted["imat"] == pytest.approx([0.8099422437616611], abs=1e-10)
    assert fitted["sctest"] == pytest.approx(0.6770036246476036, abs=1e-10)
    assert fitted["maxiter"] == 4
    assert fitted["flag"] == 1

    with pytest.raises(ValueError, match="work must have length at least 5"):
        survival_package.agexact(
            1,
            1,
            1,
            [0.0],
            [1.0],
            [1],
            [1.0],
            [0.0],
            [0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0, 0.0],
            [0.0],
            [0, 0],
            1e-9,
            1e-9,
            [1],
        )
