"""``survival.data_prep`` against R survival 3.8.11 (``strata``, ``neardate``, ``aeqSurv``,
``survSplit``, ``survcondense``, ``tcut``, ``tmerge``, ``rttright``, ``fromtimeline``).

Every row index returned by these bindings is zero-based.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
data_prep = survival.data_prep


def test_cluster_recodes_ids_in_order_of_appearance():
    numeric = data_prep.cluster([2, 2, 1, 3])
    assert isinstance(numeric, data_prep.ClusterResult)
    assert numeric.codes == [0, 0, 1, 2]
    assert numeric.levels == ["2", "1", "3"]
    assert numeric.sizes == [2, 1, 1]

    strings = data_prep.cluster(["b", "a", "b"])
    assert strings.codes == [0, 1, 0]
    assert strings.levels == ["b", "a"]
    assert strings.sizes == [2, 1]

    floats = data_prep.cluster([2.5, 1.0, 2.5])
    assert floats.codes == [0, 1, 0]
    assert floats.levels == ["2.5", "1"]


def test_strata_labels_match_r():
    # R: strata of two numeric vectors; levels are name=level pairs joined by a comma
    numeric = data_prep.strata(
        ["c(2, 1, 2)", "c(1, 2, 1)"], [["1", "2"], ["1", "2"]], [[1, 0, 1], [0, 1, 0]]
    )
    assert isinstance(numeric, data_prep.StrataResult)
    assert numeric.codes == [1, 0, 1]
    assert numeric.levels == ["c(2, 1, 2)=1, c(1, 2, 1)=2", "c(2, 1, 2)=2, c(1, 2, 1)=1"]
    assert numeric.counts == [1, 2]

    # R: shortlabel = TRUE drops the variable names from the levels
    short = data_prep.strata(
        ["a", "b"], [["a", "b"], ["x", "y"]], [[1, 0, 1], [0, 1, 0]], shortlabel=True
    )
    assert short.codes == [1, 0, 1]
    assert short.levels == ["a, y", "b, x"]
    assert short.counts == [1, 2]

    # R: na.group = TRUE keeps a v=NA level, without it the code is NA
    with_na = data_prep.strata(["v"], [["1", "2"]], [[0, None, 1, 0]], na_group=True)
    assert with_na.codes == [0, 2, 1, 0]
    assert with_na.levels == ["v=1", "v=2", "v=NA"]
    assert with_na.counts == [2, 1, 1]
    without_na = data_prep.strata(["v"], [["1", "2"]], [[0, None, 1, 0]])
    assert without_na.codes == [0, None, 1, 0]
    assert without_na.levels == ["v=1", "v=2"]

    with pytest.raises(ValueError, match="length"):
        data_prep.strata(["a", "b"], [["1"], ["1"]], [[0, 0], [0]])


def test_neardate_matches_r():
    id1, y1 = [1, 1, 2, 3], [10.0, 20.0, 5.0, 7.0]
    id2, y2 = [1, 1, 2, 2, 4], [8.0, 15.0, 6.0, 1.0, 3.0]

    # R neardate: first date in y2 at or after y1, as zero-based rows of y2
    assert data_prep.neardate(id1, y1, id2, y2) == [1, None, 2, None]
    # best = "prior": last date at or before
    assert data_prep.neardate(id1, y1, id2, y2, best="prior") == [0, 1, 3, None]

    with pytest.raises(ValueError, match="best"):
        data_prep.neardate(id1, y1, id2, y2, best="nearest")


def test_lvcf_and_nostutter():
    # last value carried forward within subject: row 1 is missing, row 3 starts subject 2
    assert data_prep.lvcf([1, 1, 1, 2, 2], [False, True, False, True, False]) == [0, 0, 2, 3, 4]
    assert data_prep.lvcf(
        [1, 1, 1, 2, 2], [False, True, False, True, False], time=[0.0, 1.0, 2.0, 0.0, 1.0]
    ) == [0, 0, 2, 3, 4]

    # nostutter: a repeat of the subject's current state is censored
    assert data_prep.nostutter([1, 1, 1, 2, 2], ["a", "a", "b", "a", "a"], "censor") == [
        False,
        True,
        False,
        False,
        True,
    ]
    assert data_prep.nostutter([1, 1, 1, 2, 2], [1, 1, 2, 1, 1], 0) == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_aeq_surv_folds_near_ties_like_r():
    result = data_prep.aeq_surv([1.0, 1.0 + 1e-12, 2.0, 3.0 - 1e-13])
    assert isinstance(result, data_prep.AeqSurvResult)
    # R aeqSurv: near-ties become equal, 3 - 1e-13 stays put
    assert result.time == pytest.approx([1.0, 1.0, 2.0, 2.9999999999999])
    assert result.time[0] == result.time[1]
    assert result.time2 is None
    assert result.changed == [1]

    two_column = data_prep.aeq_surv([0.0, 0.0, 1.0], [1.0, 1.0 + 1e-12, 2.0])
    assert two_column.time == pytest.approx([0.0, 0.0, 1.0])
    assert two_column.time2 == pytest.approx([1.0, 1.0, 2.0])
    assert two_column.time2[0] == two_column.time2[1]

    with pytest.raises(ValueError, match="length"):
        data_prep.aeq_surv([1.0, 2.0], [1.0])


def test_survsplit_matches_r():
    result = data_prep.survsplit([5.0, 12.0, 20.0], [1, 0, 1], [4.0, 10.0])

    # R survSplit at cut points 4 and 10
    assert isinstance(result, data_prep.SurvSplitResult)
    assert result.row == [0, 0, 1, 1, 1, 2, 2, 2]
    assert result.interval == [0, 1, 0, 1, 2, 0, 1, 2]
    assert result.start == pytest.approx([0.0, 4.0, 0.0, 4.0, 10.0, 0.0, 4.0, 10.0])
    assert result.end == pytest.approx([4.0, 5.0, 4.0, 10.0, 12.0, 4.0, 10.0, 20.0])
    assert result.status == pytest.approx([0, 1, 0, 0, 0, 0, 0, 1])
    assert result.censor == [True, False, True, True, False, True, True, False]
    assert result.cut == pytest.approx([4.0, 10.0])

    # R survSplit with zero = -1: the first interval starts at -1
    shifted = data_prep.survsplit([5.0, 12.0, 20.0], [1, 0, 1], [4.0, 10.0], zero=-1.0)
    assert shifted.start[:3] == pytest.approx([-1.0, 4.0, -1.0])

    with pytest.raises(ValueError, match="length"):
        data_prep.survsplit([5.0, 12.0], [1], [4.0])


def test_survcondense_matches_r():
    # R survcondense: consecutive rows of a subject with the same covariate value merge
    result = data_prep.survcondense(
        [1, 1, 1, 2, 2], [0.0, 5.0, 10.0, 0.0, 3.0], [5.0, 10.0, 15.0, 3.0, 8.0], [0, 0, 1, 1, 1]
    )
    assert isinstance(result, data_prep.SurvcondenseResult)
    assert result.keep == [1, 2, 4]
    assert [result.start[i] for i in result.keep] == pytest.approx([0.0, 10.0, 0.0])

    with pytest.raises(ValueError, match="length"):
        data_prep.survcondense([1, 1], [0.0], [5.0, 10.0], [0, 0])


def test_tcut_matches_r():
    result = data_prep.tcut([1.0, 5.0, 12.0, 25.0], [0.0, 4.0, 10.0, 30.0])
    assert isinstance(result, data_prep.TcutResult)
    assert result.values == pytest.approx([1.0, 5.0, 12.0, 25.0])
    assert result.cutpoints == pytest.approx([0.0, 4.0, 10.0, 30.0])
    # R tcut labels use formatReal
    assert result.labels == [" 0+ thru  4", " 4+ thru 10", "10+ thru 30"]

    labelled = data_prep.tcut(
        [1.0, 5.0, 12.0, 25.0], [0.0, 4.0, 10.0, 30.0], labels=["a", "b", "c"]
    )
    assert labelled.labels == ["a", "b", "c"]

    # R tcut with a count: three equal-width ranges over the data (with R's 1% padding)
    counted = data_prep.tcut([1.0, 5.0, 12.0, 25.0], [3.0])
    assert counted.labels == ["Range 1", "Range 2", "Range 3"]
    assert counted.cutpoints == pytest.approx([0.76, 8.92, 17.08, 25.24])

    with pytest.raises(ValueError, match="labels"):
        data_prep.tcut([1.0], [0.0, 4.0, 10.0], labels=["a"])


def test_tmerge_step_matches_r_tmerge():
    # R tmerge: death = event(tstop, c(1, 0)) on the base data
    death = data_prep.tmerge_step(
        [1, 2], [0.0, 0.0], [10.0, 8.0], [1, 2], [10.0, 8.0], "event", value=[1.0, 0.0]
    )
    assert isinstance(death, data_prep.TmergeStep)
    assert death.row == [0, 1]
    assert death.start == pytest.approx([0.0, 0.0])
    assert death.stop == pytest.approx([10.0, 8.0])
    assert death.censor_rows == []
    # tcount columns: early, late, gap, within, boundary, leading, trailing, tied, missid
    assert death.tcount == [0, 0, 0, 0, 0, 0, 2, 0, 0]
    assert death.event_row == [0, 1]
    assert death.event_source == [0, 1]
    assert death.event_value == pytest.approx([1.0, 0.0])

    # R tmerge: x = tdc(t, v) splits the intervals at the update times
    tdc = data_prep.tmerge_step(
        [1, 2], [0.0, 0.0], [10.0, 8.0], [1, 1, 2], [3.0, 7.0, 4.0], "tdc", value=[1.5, 2.5, 3.5]
    )
    assert tdc.row == [0, 0, 0, 1, 1]
    assert tdc.start == pytest.approx([0.0, 3.0, 7.0, 0.0, 4.0])
    assert tdc.stop == pytest.approx([3.0, 7.0, 10.0, 4.0, 8.0])
    assert tdc.censor_rows == [0, 1, 3]
    assert tdc.tcount == [0, 0, 0, 3, 0, 0, 0, 0, 0]
    assert tdc.source == [None, 0, 1, None, 2]
    values = [1.5, 2.5, 3.5]
    assert [values[s] if s is not None else math.nan for s in tdc.source][1:3] == [1.5, 2.5]

    # nevt = cumevent(t) on the split data: counts 1, 2 for subject 1 and 1 for subject 2
    ids = [[1, 2][r] for r in tdc.row]
    cumevent = data_prep.tmerge_step(
        ids, tdc.start, tdc.stop, [1, 1, 2], [3.0, 7.0, 4.0], "cumevent"
    )
    assert cumevent.row == [0, 1, 2, 3, 4]
    assert cumevent.tcount == [0, 0, 0, 0, 3, 0, 0, 0, 0]
    assert cumevent.event_row == [0, 1, 3]
    assert cumevent.event_value == pytest.approx([1.0, 2.0, 1.0])

    cumtdc = data_prep.tmerge_step(
        ids,
        tdc.start,
        tdc.stop,
        [1, 1, 2],
        [3.0, 7.0, 4.0],
        "cumtdc",
        value=[1.0, 1.0, 1.0],
        prior=[0.0] * 5,
        default=0.0,
    )
    assert cumtdc.cumulative == pytest.approx([0.0, 1.0, 2.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="not a recognized type"):
        data_prep.tmerge_step([1, 2], [0.0, 0.0], [10.0, 8.0], [1], [3.0], "bogus")
    with pytest.raises(ValueError, match="length"):
        data_prep.tmerge_step([1, 2], [0.0], [10.0, 8.0], [1], [3.0], "tdc", value=[1.0])


def test_rttright_matches_r():
    time, status = [1.0, 2.0, 2.0, 3.0, 4.0], [1, 0, 1, 0, 1]

    result = data_prep.rttright(time, status)
    # R rttright: redistribute-to-the-right weights
    assert isinstance(result, data_prep.RttrightResult)
    assert [row[0] for row in result.weights] == pytest.approx([0.2, 0.0, 0.2, 0.0, 0.6])
    assert result.times == []

    at_times = data_prep.rttright(time, status, times=[2.0, 3.0])
    # R rttright with reporting times 2 and 3: one column per time
    assert at_times.times == pytest.approx([2.0, 3.0])
    assert [row[0] for row in at_times.weights] == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])
    assert [row[1] for row in at_times.weights] == pytest.approx([0.2, 0.0, 0.2, 0.3, 0.3])

    stratified = data_prep.rttright(time, status, strata=[0, 0, 1, 1, 1])
    # R rttright by group: weights renormalised within stratum
    assert [row[0] for row in stratified.weights] == pytest.approx([0.5, 0.0, 1 / 3, 0.0, 2 / 3])

    raw = data_prep.rttright(time, status, strata=[0, 0, 1, 1, 1], renorm=False)
    assert [row[0] for row in raw.weights] == pytest.approx([1.0, 0.0, 1.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="length"):
        data_prep.rttright(time, status[:-1])


def test_surv2counting_and_totimeline():
    # timeline rows per subject: (id, time, state) with the state entered at each time
    counting = data_prep.surv2counting(
        [1, 1, 1, 2, 2], [0.0, 3.0, 7.0, 0.0, 5.0], [None, 1, 2, None, 1], has_states=True
    )
    assert isinstance(counting, data_prep.Surv2CountingResult)
    assert counting.row == [0, 1, 3]
    assert counting.tstart == pytest.approx([0.0, 3.0, 0.0])
    assert counting.tstop == pytest.approx([3.0, 7.0, 5.0])
    assert counting.status == [1, 2, 1]
    assert counting.counting is True

    with_missing = data_prep.surv2counting(
        [1, 1, 1, 2, 2],
        [0.0, 3.0, 7.0, 0.0, 5.0],
        [None, 1, 2, None, 1],
        has_states=True,
        missing=[[False, True, False, False, False]],
    )
    # a covariate missing on row 1 is carried forward from row 0
    assert with_missing.carry_from == [[0, 0, 3]]

    with pytest.raises(ValueError, match="repeated"):
        data_prep.surv2counting([1, 1], [0.0, 1.0], [None, 1], repeated="sometimes")

    timeline = data_prep.totimeline(
        [1, 1, 2], [0.0, 3.0, 0.0], [3.0, 7.0, 5.0], [1, 2, 1], [0, 1, 0]
    )
    assert isinstance(timeline, data_prep.TotimelineResult)
    assert timeline.time_row == [0, 0, 1, 2, 2]
    assert timeline.covariate_row == [0, 1, 1, 2, 2]
    assert timeline.time == pytest.approx([0.0, 3.0, 7.0, 0.0, 5.0])
    assert timeline.state == [0, 1, 2, 0, 1]
