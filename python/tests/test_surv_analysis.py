from math import exp, log
from statistics import NormalDist

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _plain_confidence_interval(estimate, std_err, conf_level=0.95):
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf_level) / 2.0)
    return max(estimate - z * std_err, 0.0), min(estimate + z * std_err, 1.0)


def test_compute_baseline_survival_steps():
    ndeath = [1, 1, 0, 1, 0]
    risk = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0]
    sn = 5
    denom = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = survival.compute_baseline_survival_steps(ndeath, risk, wt, sn, denom)
    assert isinstance(result, list)
    assert len(result) == sn


def test_compute_baseline_survival_steps_validates_inputs():
    with pytest.raises(ValueError, match="ndeath length must be 2"):
        survival.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[1.0],
            wt=[1.0],
            sn=2,
            denom=[2.0, 1.0],
        )

    with pytest.raises(ValueError, match="risk length must be at least 2"):
        survival.compute_baseline_survival_steps(
            ndeath=[2],
            risk=[1.0],
            wt=[1.0, 1.0],
            sn=1,
            denom=[2.0],
        )

    with pytest.raises(ValueError, match="risk must be positive"):
        survival.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[0.0],
            wt=[1.0],
            sn=1,
            denom=[2.0],
        )

    with pytest.raises(ValueError, match="denom contains non-finite"):
        survival.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[1.0],
            wt=[1.0],
            sn=1,
            denom=[float("inf")],
        )

    with pytest.raises(ValueError, match="death contribution must not exceed denom"):
        survival.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[2.0],
            wt=[1.0],
            sn=1,
            denom=[1.0],
        )


def test_agsurv4_alias_matches_validated_baseline_steps():
    ndeath = [1, 2, 0]
    risk = [1.0, 1.0, 1.0]
    wt = [0.2, 0.3, 0.4]
    denom = [5.0, 4.0, 3.0]

    direct = survival.compute_baseline_survival_steps(ndeath, risk, wt, 3, denom)
    alias = survival.agsurv4(ndeath, risk, wt, 3, denom)

    assert alias == pytest.approx(direct)

    with pytest.raises(ValueError, match="risk length must be at least 1"):
        survival.agsurv4([1], [], [1.0], 1, [2.0])


def test_compute_tied_baseline_summaries():
    n = 5
    nvar = 2
    dd = [1, 1, 2, 1, 1]
    x1 = [10.0, 9.0, 8.0, 7.0, 6.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0]
    xsum = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    xsum2 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.5, 2.0, 1.5, 1.0, 0.5]

    result = survival.compute_tied_baseline_summaries(n, nvar, dd, x1, x2, xsum, xsum2)
    assert isinstance(result, dict)
    assert "sum1" in result
    assert "sum2" in result
    assert "xbar" in result


def test_compute_tied_baseline_summaries_validates_inputs():
    with pytest.raises(ValueError, match="dd length must be 2"):
        survival.compute_tied_baseline_summaries(
            2,
            1,
            [1],
            [10.0, 9.0],
            [5.0, 4.0],
            [10.0, 9.0],
            [5.0, 4.0],
        )

    with pytest.raises(ValueError, match="xsum length must be 2"):
        survival.compute_tied_baseline_summaries(
            2,
            1,
            [1, 1],
            [10.0, 9.0],
            [5.0, 4.0],
            [10.0],
            [5.0, 4.0],
        )

    with pytest.raises(ValueError, match="positive event counts"):
        survival.compute_tied_baseline_summaries(
            1,
            1,
            [0],
            [10.0],
            [5.0],
            [10.0],
            [5.0],
        )

    with pytest.raises(ValueError, match="x1 contains non-finite"):
        survival.compute_tied_baseline_summaries(
            1,
            1,
            [1],
            [float("nan")],
            [5.0],
            [10.0],
            [5.0],
        )

    with pytest.raises(ValueError, match="tied denominator must be positive"):
        survival.compute_tied_baseline_summaries(
            1,
            1,
            [2],
            [1.0],
            [3.0],
            [1.0],
            [0.5],
        )


def test_agsurv5_alias_matches_validated_tied_baseline_summaries():
    args = (
        2,
        1,
        [1, 2],
        [10.0, 9.0],
        [0.0, 1.0],
        [10.0, 9.0],
        [0.0, 0.5],
    )

    direct = survival.compute_tied_baseline_summaries(*args)
    alias = survival.agsurv5(*args)

    assert alias == direct

    with pytest.raises(ValueError, match="positive event counts"):
        survival.agsurv5(1, 1, [0], [1.0], [0.0], [1.0], [0.0])


def test_agmart():
    n = 5
    method = 0
    start = [0.0, 0.0, 1.0, 1.0, 2.0]
    stop = [1.0, 2.0, 2.0, 3.0, 3.0]
    event = [1, 0, 1, 0, 1]
    score = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    strata = [1, 0, 0, 0, 0]

    counting = survival.CountingProcessData(start, stop, event)
    weights = survival.Weights(wt)
    input_data = survival.AndersenGillInput(counting, score, weights, strata)

    result = survival.agmart(input_data, method)
    assert isinstance(result, list)
    assert len(result) == n


def test_survfitkm():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    result = survival.survfitkm(
        time=time,
        status=status,
        weights=None,
        entry_times=None,
        position=None,
        reverse=False,
        computation_type=0,
    )
    assert hasattr(result, "time")
    assert hasattr(result, "estimate")
    assert hasattr(result, "std_err")
    assert hasattr(result, "cumhaz")
    assert hasattr(result, "std_chaz")
    assert hasattr(result, "cumulative_hazard")
    assert len(result.time) > 0
    assert len(result.estimate) == len(result.time)
    assert len(result.cumhaz) == len(result.time)
    assert len(result.std_chaz) == len(result.time)
    assert result.cumulative_hazard == pytest.approx(result.cumhaz)


def test_survfitkm_groups_near_tied_times():
    result = survival.survfitkm(
        time=[1.0 + 5e-10, 2.0, 1.0],
        status=[1.0, 0.0, 1.0],
    )

    assert result.time == pytest.approx([1.0, 2.0])
    assert result.n_risk == pytest.approx([3.0, 1.0])
    assert result.n_event == pytest.approx([2.0, 0.0])
    assert result.n_censor == pytest.approx([0.0, 1.0])
    assert result.estimate == pytest.approx([1.0 / 3.0, 1.0 / 3.0])


def test_survfitkm_rejects_non_binary_status_values():
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.survfitkm([1.0, 2.0, 3.0], [1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.survfitkm_with_options(
            [1.0, 2.0, 3.0],
            [1.0, 0.5, 0.0],
            survival.SurvfitKMOptions(),
        )


def test_survfitkm_with_options_honors_conf_level():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    options = survival.SurvfitKMOptions(conf_level=0.9)

    with_options = survival.survfitkm_with_options(time, status, options)
    direct = survival.survfitkm(time, status, conf_level=0.9)
    default = survival.survfitkm(time, status)

    assert with_options.conf_lower == pytest.approx(direct.conf_lower)
    assert with_options.conf_upper == pytest.approx(direct.conf_upper)
    assert with_options.conf_lower != pytest.approx(default.conf_lower)

    with pytest.raises(ValueError, match="conf_level"):
        survival.survfitkm_with_options(
            time,
            status,
            survival.SurvfitKMOptions().with_conf_level(1.0),
        )


def test_survfitkm_honors_conf_type():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    plain = survival.survfitkm(time, status, conf_type="plain")
    with_options = survival.survfitkm_with_options(
        time,
        status,
        survival.SurvfitKMOptions().with_conf_type("plain"),
    )
    default = survival.survfitkm(time, status)
    none = survival.survfitkm(time, status, conf_type="none")
    expected_lower, expected_upper = _plain_confidence_interval(
        plain.estimate[0],
        plain.std_err[0],
    )

    assert plain.conf_lower[0] == pytest.approx(expected_lower)
    assert plain.conf_upper[0] == pytest.approx(expected_upper)
    assert plain.conf_lower == pytest.approx(with_options.conf_lower)
    assert plain.conf_upper == pytest.approx(with_options.conf_upper)
    assert plain.conf_lower != pytest.approx(default.conf_lower)
    assert none.conf_lower == []
    assert none.conf_upper == []

    with pytest.raises(ValueError, match="conf_type"):
        survival.survfitkm(time, status, conf_type="weird")

    with pytest.raises(ValueError, match="conf_type"):
        survival.survfitkm_with_options(
            time,
            status,
            survival.SurvfitKMOptions().with_conf_type("weird"),
        )


def test_survfitkm_honors_delayed_entry():
    entry_times = [0.0, 0.0, 1.0, 2.0, 3.0]
    time = [2.0, 4.0, 3.0, 5.0, 5.0]
    status = [1.0, 0.0, 1.0, 1.0, 0.0]

    result = survival.survfitkm(time, status, entry_times=entry_times)
    with_options = survival.survfitkm_with_options(
        time,
        status,
        survival.SurvfitKMOptions(entry_times=entry_times),
    )

    assert result.time == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert result.n_risk == pytest.approx([3.0, 3.0, 3.0, 2.0])
    assert result.n_event == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert result.n_censor == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert result.estimate == pytest.approx([2.0 / 3.0, 4.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0])
    assert result.cumhaz == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 7.0 / 6.0])
    assert result.std_chaz == pytest.approx(
        [
            (1.0 / 9.0) ** 0.5,
            (2.0 / 9.0) ** 0.5,
            (2.0 / 9.0) ** 0.5,
            (2.0 / 9.0 + 1.0 / 4.0) ** 0.5,
        ]
    )
    assert with_options.n_risk == pytest.approx(result.n_risk)
    assert with_options.estimate == pytest.approx(result.estimate)
    assert with_options.cumhaz == pytest.approx(result.cumhaz)
    assert with_options.std_chaz == pytest.approx(result.std_chaz)


def test_survfitkm_reverse_estimates_censoring_distribution():
    time = [1.0, 2.0, 2.0, 3.0, 4.0]
    status = [1.0, 0.0, 1.0, 0.0, 1.0]

    result = survival.survfitkm(time, status, reverse=True)
    with_options = survival.survfitkm_with_options(
        time,
        status,
        survival.SurvfitKMOptions(reverse=True),
    )

    assert result.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result.n_risk == pytest.approx([5.0, 4.0, 2.0, 1.0])
    assert result.n_event == pytest.approx([0.0, 1.0, 1.0, 0.0])
    assert result.n_censor == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert result.estimate == pytest.approx([1.0, 0.75, 0.375, 0.375])
    assert with_options.time == pytest.approx(result.time)
    assert with_options.estimate == pytest.approx(result.estimate)


def test_nelson_aalen_estimator_groups_near_tied_times_and_validates_inputs():
    result = survival.nelson_aalen_estimator(
        time=[1.0 + 5e-10, 2.0, 1.0],
        status=[1, 0, 1],
    )

    assert result.time == pytest.approx([1.0])
    assert result.n_risk == [3]
    assert result.n_events == [2]
    assert result.cumulative_hazard == pytest.approx([2.0 / 3.0])
    assert result.variance == pytest.approx([2.0 / 9.0])
    assert result.survival() == pytest.approx([exp(-2.0 / 3.0)])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.nelson_aalen_estimator([1.0, 2.0], [1, 2])

    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.nelson_aalen_estimator([1.0, 2.0], [1, 0], weights=[1.0, float("inf")])

    with pytest.raises(ValueError, match="confidence_level"):
        survival.nelson_aalen_estimator([1.0, 2.0], [1, 0], confidence_level=1.0)


def test_stratified_kaplan_meier_validates_public_inputs():
    result = survival.stratified_kaplan_meier(
        [1.0, 2.0, 1.0 + 5e-10],
        [1, 0, 1],
        [0, 0, 0],
    )

    assert result.times[0] == pytest.approx([1.0])
    assert result.n_events[0] == [2]

    with pytest.raises(ValueError, match="strata length mismatch"):
        survival.stratified_kaplan_meier([1.0, 2.0], [1, 0], [0])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.stratified_kaplan_meier([1.0, 2.0], [1, 2], [0, 0])


def test_survfit_matrix_public_apis_validate_shapes_and_values():
    result = survival.survfit_from_hazard(
        [1.0, 2.0],
        [0.1, 0.2],
        n_risk=[10.0, 8.0],
        n_event=[1.0, 2.0],
    )

    assert result.n_states == 1
    assert result.get_cumhaz_at_state(0) == pytest.approx([0.1, 0.3])
    assert result.get_surv_at_state(0) == pytest.approx([exp(-0.1), exp(-0.3)])

    multistate = survival.survfit_multistate(
        [1.0],
        [[[0.0, 0.25], [0.10, 0.0]]],
        0,
    )
    assert multistate.n_states == 2
    assert multistate.surv[0] == pytest.approx([0.75, 0.25])
    assert sum(multistate.surv[0]) == pytest.approx(1.0)

    with pytest.raises(IndexError, match="out of range"):
        result.get_surv_at_state(1)

    with pytest.raises(ValueError, match="surv length must match time length"):
        survival.SurvfitMatrixResult([1.0], [], [[0.1]], None, [], [], 1)

    with pytest.raises(ValueError, match="surv values must be between 0 and 1"):
        survival.SurvfitMatrixResult([1.0], [[1.2]], [[0.1]], None, [], [], 1)

    with pytest.raises(ValueError, match="hazard contains non-finite"):
        survival.survfit_from_hazard([1.0], [float("nan")])

    with pytest.raises(ValueError, match="n_risk must have the same length as time"):
        survival.survfit_from_hazard([1.0, 2.0], [0.1, 0.2], n_risk=[10.0])

    with pytest.raises(ValueError, match="hazard_matrix must be non-negative"):
        survival.survfit_from_matrix([1.0], [[-0.1]])

    with pytest.raises(ValueError, match="hazard_matrix length must match time length"):
        survival.survfit_from_matrix([1.0, 2.0], [[0.1]])

    with pytest.raises(ValueError, match="hazard_matrix must have at least one column"):
        survival.survfit_from_matrix([1.0], [[]])

    with pytest.raises(ValueError, match="off-diagonal entries must be non-negative"):
        survival.survfit_multistate(
            [1.0],
            [[[0.0, -0.1], [0.0, 0.0]]],
            0,
        )

    with pytest.raises(ValueError, match="transition_hazards must have at least one state"):
        survival.survfit_multistate([1.0], [[]], 0)

    with pytest.raises(ValueError, match="outgoing row sums"):
        survival.survfit_multistate(
            [1.0],
            [[[0.0, 0.8, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            0,
        )


def test_compute_logrank_components():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    result = survival.compute_logrank_components(
        time=time,
        status=status,
        group=group,
        strata=None,
        rho=0.0,
    )
    assert hasattr(result, "observed")
    assert hasattr(result, "expected")
    assert hasattr(result, "chi_squared")
    assert len(result.observed) > 0


def test_compute_logrank_components_aggregates_stratified_variance():
    stratified = survival.compute_logrank_components(
        [1.0, 2.0, 1.0, 2.0],
        [1, 0, 0, 1],
        [1, 2, 1, 2],
        [0, 1, 0, 1],
        None,
    )
    first = survival.compute_logrank_components([1.0, 2.0], [1, 0], [1, 2], None, None)
    second = survival.compute_logrank_components([1.0, 2.0], [0, 1], [1, 2], None, None)

    assert len(stratified.variance) == 2
    assert all(len(row) == 2 for row in stratified.variance)
    for row in range(2):
        for col in range(2):
            assert stratified.variance[row][col] == pytest.approx(
                first.variance[row][col] + second.variance[row][col]
            )


def test_compute_logrank_components_multigroup_matches_r_survdiff_chisquare():
    result = survival.compute_logrank_components(
        [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [1, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 2, 1, 3, 2, 3, 2, 3],
        None,
        None,
    )

    assert result.degrees_of_freedom == 2
    assert result.observed == pytest.approx([2.0, 2.0, 2.0])
    assert result.expected == pytest.approx([1.0, 2.25, 2.75])
    assert result.variance[0][0] == pytest.approx(0.6825396825396826)
    assert result.variance[0][1] == pytest.approx(-0.3273809523809524)
    assert result.chi_squared == pytest.approx(1.5105257668985863)


def test_aggregate_survfit_public_apis():
    result = survival.aggregate_survfit(
        times=[[1.0, 2.0], [1.5, 2.5]],
        survs=[[0.9, 0.8], [0.95, 0.85]],
        std_errs=[[0.05, 0.06], [0.04, 0.05]],
        weights=[2.0, 1.0],
        conf_level=0.95,
    )
    by_group = survival.aggregate_survfit_by_group(
        times=[[1.0, 2.0], [1.0, 2.0], [1.5, 2.5]],
        survs=[[0.9, 0.8], [0.8, 0.7], [0.95, 0.85]],
        groups=[1, 1, 2],
        weights=[1.0, 2.0, 1.0],
    )
    near_times = survival.aggregate_survfit(
        times=[[1.0, 2.0], [1.0 + 5e-10, 2.0]],
        survs=[[0.9, 0.8], [0.95, 0.85]],
    )
    by_group_sorted = sorted(by_group, key=lambda item: item.n_curves)

    assert result.n_curves == 2
    assert result.time == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert result.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert len(result.surv) == len(result.time)
    assert len(result.std_err) == len(result.time)
    assert all(0.0 <= value <= 1.0 for value in result.lower)
    assert all(0.0 <= value <= 1.0 for value in result.upper)

    assert len(by_group_sorted) == 2
    assert by_group_sorted[0].n_curves == 1
    assert by_group_sorted[0].surv == pytest.approx([0.95, 0.85])
    assert by_group_sorted[1].n_curves == 2
    assert by_group_sorted[1].weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert near_times.time == pytest.approx([1.0, 2.0])
    assert near_times.surv == pytest.approx([0.925, 0.825])

    with pytest.raises(ValueError, match="times and survs must have same length"):
        survival.aggregate_survfit([[1.0]], [[0.9], [0.8]], None, None, None)

    with pytest.raises(ValueError, match=r"times\[0\] and survs\[0\]"):
        survival.aggregate_survfit([[1.0, 2.0]], [[0.9]])

    with pytest.raises(ValueError, match="between 0 and 1"):
        survival.aggregate_survfit([[1.0]], [[1.1]])

    with pytest.raises(ValueError, match="at least one positive"):
        survival.aggregate_survfit([[1.0]], [[0.9]], weights=[0.0])

    with pytest.raises(ValueError, match="conf_level"):
        survival.aggregate_survfit([[1.0]], [[0.9]], conf_level=1.0)

    with pytest.raises(ValueError, match="weights must have same length"):
        survival.aggregate_survfit_by_group([[1.0]], [[0.9]], [1], weights=[])


def test_survcheck_public_apis_and_validation():
    result = survival.survcheck(
        id=[1, 1, 2],
        time1=[0.0, 2.0, 0.0],
        time2=[2.0, 4.0, 1.0],
        status=[1, 0, 1],
        istate=[0, 1, 0],
    )
    simple = survival.survcheck_simple(
        time=[1.0, -1.0, float("nan")],
        status=[1, 0, 2],
    )

    assert result.n_subjects == 2
    assert result.n_transitions == 3
    assert result.n_problems == 0
    assert result.is_valid is True
    assert result.transitions == {"0 -> 1": 2, "1 -> 0": 1}
    assert result.flags == [0, 0, 0]

    assert simple.n_subjects == 3
    assert simple.n_problems == 2
    assert simple.is_valid is False
    assert simple.invalid_ids == [1, 2]
    assert simple.flags == [0, 4, 4]
    assert any("negative time" in message for message in simple.messages)

    with pytest.raises(ValueError, match="All input vectors must have the same length"):
        survival.survcheck([1], [0.0], [1.0, 2.0], [1], None)


def test_royston_public_apis_and_validation():
    result = survival.royston(
        linear_predictor=[0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.9, -0.2],
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        status=[1, 1, 1, 0, 1, 1, 1, 0],
    )
    from_model = survival.royston_from_model(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        coef=[0.5, 1.0],
        n_obs=3,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
    )

    assert result.d > 0.0
    assert result.se > 0.0
    assert 0.0 <= result.r_squared_d <= 1.0
    assert 0.0 <= result.r_squared_ko <= 1.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_events == 6
    assert from_model.n_events == 3

    with pytest.raises(ValueError, match="At least 2 events required"):
        survival.royston([0.1, 0.2], [1.0, 2.0], [1, 0])


def test_yates_public_apis_and_validation():
    result = survival.yates(
        predictions=[1.0, 2.0, 2.0, 4.0],
        factor=["A", "A", "B", "B"],
        weights=[1.0, 2.0, 1.0, 1.0],
        conf_level=0.95,
    )
    pairwise = survival.yates_pairwise(result)
    contrast = survival.yates_contrast(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        coef=[0.5, 1.0],
        n_obs=3,
        n_vars=2,
        factor_col=0,
        factor_levels=[0.0, 1.0],
        predict_type="risk",
    )

    assert result.levels == ["A", "B"]
    assert result.means == pytest.approx([5.0 / 3.0, 3.0])
    assert result.n == [2, 2]
    assert result.predict_type == "linear"
    assert pairwise.level1 == ["A"]
    assert pairwise.level2 == ["B"]
    assert pairwise.difference == pytest.approx([-4.0 / 3.0])
    assert contrast.levels == ["0", "1"]
    assert contrast.predict_type == "risk"
    assert contrast.means[1] > contrast.means[0]

    with pytest.raises(ValueError, match="weights must have same length as predictions"):
        survival.yates([1.0], ["A"], [1.0, 2.0], 0.95)


def test_concordance_metric_public_apis_and_validation():
    uno = survival.uno_c_index(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    comparison = survival.compare_uno_c_indices(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score_1=[0.9, 0.7, 0.4, 0.2],
        risk_score_2=[0.2, 0.4, 0.7, 0.9],
    )
    decomposition = survival.c_index_decomposition(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    gonen = survival.gonen_heller_concordance([0.9, 0.7, 0.4, 0.2])

    assert uno.c_index == pytest.approx(1.0)
    assert uno.comparable_pairs > 0.0
    assert 0.0 <= uno.ci_lower <= uno.ci_upper <= 1.0
    assert comparison.c_index_1 > comparison.c_index_2
    assert comparison.difference > 0.0
    assert 0.0 <= comparison.p_value <= 1.0
    assert decomposition.c_index == pytest.approx(1.0)
    assert 0.0 <= decomposition.alpha <= 1.0
    assert decomposition.n_event_event_pairs > 0
    assert 0.0 <= gonen.cpe <= 1.0
    assert gonen.std_error >= 0.0

    with pytest.raises(ValueError, match="time, status, and risk_score must have the same length"):
        survival.uno_c_index([1.0], [1], [0.1, 0.2])


def test_time_dependent_auc_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    marker = [0.9, 0.2, 0.7, 0.1]

    auc = survival.time_dependent_auc(time, status, marker, 2.5)
    cumulative = survival.cumulative_dynamic_auc(time, status, marker, [1.5, 2.5, 3.5])

    assert 0.0 <= auc.auc <= 1.0
    assert auc.n_cases == 1
    assert auc.n_controls == 2
    assert len(cumulative.auc) == 3
    assert 0.0 <= cumulative.mean_auc <= 1.0

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.time_dependent_auc([float("nan")], [1], [0.5], 1.0)

    with pytest.raises(ValueError, match="time contains negative value"):
        survival.time_dependent_auc([-1.0], [1], [0.5], 1.0)

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.time_dependent_auc([1.0], [2], [0.5], 1.0)

    with pytest.raises(ValueError, match="marker contains non-finite"):
        survival.time_dependent_auc([1.0], [1], [float("inf")], 1.0)

    with pytest.raises(ValueError, match="t contains non-finite"):
        survival.time_dependent_auc([1.0], [1], [0.5], float("nan"))

    with pytest.raises(ValueError, match="times must be sorted"):
        survival.cumulative_dynamic_auc(time, status, marker, [2.5, 1.5])


def test_rcll_public_apis_and_validation():
    result = survival.rcll(
        survival_predictions=[
            [0.95, 0.85, 0.70],
            [0.90, 0.75, 0.55],
            [0.98, 0.92, 0.80],
        ],
        prediction_times=[1.0, 2.0, 3.0],
        event_times=[2.5, 1.5, 3.0],
        status=[1, 1, 0],
        weights=[1.0, 2.0, 1.0],
    )
    single_time = survival.rcll_single_time(
        survival_probs=[0.8, 0.7, 0.9],
        event_times=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        prediction_time=2.0,
        weights=[1.0, 2.0, 1.0],
    )

    assert result.n_events == 2
    assert result.n_censored == 1
    assert result.mean_rcll > 0.0
    assert result.event_contribution > result.censored_contribution
    assert single_time.n_events == 1
    assert single_time.n_censored == 2
    assert single_time.rcll > 0.0

    with pytest.raises(ValueError, match="survival_predictions row 0 has 1 elements, expected 2"):
        survival.rcll([[0.9]], [1.0, 2.0], [1.0], [1], None)

    duplicate_times = survival.rcll(
        survival_predictions=[
            [0.95, 0.8, 0.7, 0.5],
            [0.95, 0.8, 0.7, 0.5],
        ],
        prediction_times=[1.0, 2.0, 2.0, 3.0],
        event_times=[2.0, 2.0],
        status=[1, 0],
    )

    assert duplicate_times.event_contribution == pytest.approx(-log(0.15))
    assert duplicate_times.censored_contribution == pytest.approx(-log(0.7))

    with pytest.raises(ValueError, match="prediction_times must be sorted"):
        survival.rcll([[0.9, 0.8]], [2.0, 1.0], [1.0], [1])

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.rcll([[0.9]], [1.0], [1.0], [2])

    with pytest.raises(ValueError, match="probabilities between 0 and 1"):
        survival.rcll([[1.2]], [1.0], [1.0], [1])

    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.rcll([[0.9]], [1.0], [1.0], [1], [-1.0])

    with pytest.raises(ValueError, match="prediction_time contains non-finite"):
        survival.rcll_single_time([0.9], [1.0], [1], float("nan"))


def test_rmst_family_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1]
    group = [0, 0, 1, 1]

    rmst = survival.rmst(time, status, 4.0)
    comparison = survival.rmst_comparison(time, status, group, 4.0)
    quantile = survival.survival_quantile(time, status, 0.5)
    incidence = survival.cumulative_incidence(time, [1, 2, 0, 1])
    nnt = survival.number_needed_to_treat(time, status, group, 4.0)
    threshold = survival.rmst_optimal_threshold(time, status)

    assert 0.0 <= rmst.rmst <= 4.0
    assert comparison.rmst_group1.rmst >= 0.0
    assert quantile.quantile == pytest.approx(0.5)
    assert incidence.event_types == [1, 2]
    assert nnt.time_horizon == pytest.approx(4.0)
    assert threshold.max_followup == pytest.approx(4.0)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.rmst([float("nan")], [1], 1.0)

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.rmst([1.0], [2], 1.0)

    with pytest.raises(ValueError, match="tau must be non-negative"):
        survival.rmst([1.0], [1], -1.0)

    with pytest.raises(ValueError, match="confidence_level must be greater than 0"):
        survival.survival_quantile([1.0], [1], 0.5, 1.0)

    with pytest.raises(ValueError, match="quantile must be greater than 0"):
        survival.survival_quantile([1.0], [1], 0.0)

    with pytest.raises(ValueError, match="non-negative event codes"):
        survival.cumulative_incidence([1.0], [-1])

    with pytest.raises(ValueError, match="time, status, and group must have the same length"):
        survival.number_needed_to_treat([1.0], [1], [0, 1], 1.0)

    with pytest.raises(ValueError, match="alpha must be greater than 0"):
        survival.rmst_optimal_threshold([1.0], [1], 0.0)

    with pytest.raises(ValueError, match="min_events_per_interval must be at least 2"):
        survival.rmst_optimal_threshold([1.0], [1], None, 1)


def test_pseudo_public_apis_and_validation():
    result = survival.pseudo(
        time=[1.0, 1.0 + 5e-10, 2.0],
        status=[0, 1, 0],
        eval_times=None,
        type_="survival",
    )
    fast = survival.pseudo_fast(
        time=[1.0, 1.0 + 5e-10, 2.0],
        status=[0, 1, 0],
        eval_times=[1.0],
        type_="cumhaz",
    )

    assert result.n == 3
    assert result.time == pytest.approx([1.0])
    assert len(result.pseudo) == 3
    assert sum(row[0] for row in result.pseudo) / 3.0 == pytest.approx(2.0 / 3.0)
    assert fast.time == pytest.approx([1.0])
    assert sum(row[0] for row in fast.pseudo) / 3.0 == pytest.approx(1.0 / 3.0)
    gee = survival.pseudo_gee_regression(
        pseudo_values=[[0.8], [0.7], [0.6]],
        covariates=[[1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
        cluster_id=None,
        config=survival.GEEConfig(),
    )

    assert gee.coefficients
    assert len(gee.coefficients) == len(gee.std_errors) == 2
    assert len(gee.confidence_intervals) == 2

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.pseudo([1.0, 2.0], [1, 2], None, "survival")

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.pseudo([1.0, float("inf")], [1, 0], None, "survival")

    with pytest.raises(ValueError, match="eval_times contains negative value"):
        survival.pseudo([1.0, 2.0], [1, 0], [-1.0], "survival")

    with pytest.raises(ValueError, match="type must be"):
        survival.pseudo([], [], None, "weird")

    with pytest.raises(ValueError, match="correlation_structure"):
        survival.GEEConfig(correlation_structure="weird")

    with pytest.raises(ValueError, match="tol must be finite"):
        survival.GEEConfig(tol=float("nan"))

    with pytest.raises(ValueError, match="pseudo_values row 1"):
        survival.pseudo_gee_regression([[0.8], [0.7, 0.6]], [[1.0], [1.0]], None, None)

    with pytest.raises(ValueError, match="covariates length"):
        survival.pseudo_gee_regression([[0.8], [0.7]], [[1.0]], None, None)

    with pytest.raises(ValueError, match="cluster_id length"):
        survival.pseudo_gee_regression([[0.8], [0.7]], [[1.0], [1.0]], [0], None)


def test_survfitaj_extended_public_apis_and_validation():
    config = survival.AalenJohansenExtendedConfig()
    result = survival.survfitaj_extended(
        from_state=[0, 0, 0],
        to_state=[1, 2, 0],
        time=[1.0, 1.0 + 5e-10, 2.0],
        config=config,
        weights=None,
    )

    assert result.n_obs == 3
    assert result.n_events == 2
    assert result.time == pytest.approx([1.0, 2.0])
    assert result.transition_matrices[0].n_transitions[0][1] == 1
    assert result.transition_matrices[0].n_transitions[0][2] == 1
    assert result.transition_matrices[0].matrix[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result.get_cif(1) == pytest.approx([1 / 3, 1 / 3])
    assert result.get_state_prob(0, 1) == pytest.approx([1 / 3, 1 / 3])
    assert result.interpolate_at(1.0)[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    with pytest.raises(ValueError, match="from_state, to_state, and time must have equal length"):
        survival.survfitaj_extended([0], [1, 2], [1.0], config, None)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.survfitaj_extended([0, 0], [1, 2], [1.0, float("inf")], config, None)

    with pytest.raises(ValueError, match="weights must have length n_obs"):
        survival.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, [1.0])

    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, [1.0, -1.0])

    config.confidence_level = float("nan")
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        survival.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, None)

    config = survival.AalenJohansenExtendedConfig()
    config.variance_estimator = survival.VarianceEstimator("bootstrap")
    config.n_bootstrap = 0
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        survival.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, None)


def test_ridge_public_apis_and_validation():
    penalty = survival.RidgePenalty(0.1, False)
    fit = survival.ridge_fit(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        n_obs=3,
        n_vars=2,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        penalty=penalty,
        weights=None,
    )
    best_theta, cv_scores = survival.ridge_cv(
        x=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        n_obs=3,
        n_vars=2,
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        theta_grid=[0.01, 0.1, 1.0],
        n_folds=2,
    )
    tied_fit = survival.ridge_fit(
        x=[0.0, 2.0, 2.0],
        n_obs=3,
        n_vars=1,
        time=[1.0, 1.0, 2.0],
        status=[1, 0, 0],
        penalty=survival.RidgePenalty(0.0, False),
        weights=None,
    )

    assert penalty.penalty_value([1.0, 2.0]) == pytest.approx(0.25)
    assert penalty.penalty_gradient([1.0, 2.0]) == pytest.approx([0.1, 0.2])
    assert penalty.apply_to_information([1.0, 2.0]) == pytest.approx([1.1, 2.1])
    assert len(fit.coefficients) == 2
    assert len(fit.std_err) == 2
    assert fit.theta == pytest.approx(0.1)
    assert fit.df > 0.0
    assert fit.scale_factors is None
    assert best_theta in [0.01, 0.1, 1.0]
    assert len(cv_scores) == 3
    assert tied_fit.coefficients == pytest.approx([-1.5])

    with pytest.raises(ValueError, match="x length must equal n_obs \\* n_vars"):
        survival.ridge_fit([1.0], 2, 1, [1.0, 2.0], [1, 1], penalty, None)

    with pytest.raises(ValueError, match="theta must be finite and non-negative"):
        survival.RidgePenalty(float("inf"), False)

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.ridge_fit([1.0, 2.0], 2, 1, [1.0, 2.0], [1, 2], penalty, None)

    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.ridge_fit(
            [1.0, 2.0],
            2,
            1,
            [1.0, 2.0],
            [1, 1],
            penalty,
            [1.0, float("inf")],
        )

    with pytest.raises(ValueError, match="theta_grid cannot be empty"):
        survival.ridge_cv([1.0, 2.0], 2, 1, [1.0, 2.0], [1, 1], [], 2)

    with pytest.raises(ValueError, match="n_folds must be between 2 and n_obs"):
        survival.ridge_cv([1.0, 2.0], 2, 1, [1.0, 2.0], [1, 1], [0.1], 0)


def test_anova_and_basehaz_public_apis():
    anova = survival.anova_coxph(
        logliks=[-10.0, -8.0, -7.5],
        dfs=[1, 2, 3],
        model_names=["null", "m1", "m2"],
        test="LRT",
    )
    single = survival.anova_coxph_single(-10.0, -8.0, 1, 2)
    times, hazard = survival.basehaz(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 0, 1, 1],
        linear_predictors=[0.0, 0.2, 0.1, -0.1],
        centered=False,
    )

    assert anova.test_type == "LRT"
    assert len(anova.rows) == 3
    assert anova.rows[1].chisq == pytest.approx(4.0)
    assert "Analysis of Deviance Table" in anova.to_table()
    assert len(single.rows) == 2
    assert single.rows[0].model_name == "Null"
    assert single.rows[1].chisq == pytest.approx(4.0)
    assert times == pytest.approx([1.0, 3.0, 4.0])
    assert len(hazard) == 3
    assert hazard[0] < hazard[1] < hazard[2]

    large_lp_times, large_lp_hazard = survival.basehaz(
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        linear_predictors=[710.0, 709.0, 708.0],
        centered=False,
    )
    expected_first = exp(-710.0) / (1.0 + exp(-1.0) + exp(-2.0))
    assert large_lp_times == pytest.approx([1.0, 2.0, 3.0])
    assert large_lp_hazard[0] == pytest.approx(expected_first, rel=1e-12, abs=0.0)
    assert 0.0 < large_lp_hazard[0] < large_lp_hazard[1] < large_lp_hazard[2]

    with pytest.raises(ValueError, match="Need at least 2 models for comparison"):
        survival.anova_coxph([-1.0], [1], None, "LRT")

    with pytest.raises(
        ValueError,
        match="time, status, and linear_predictors must have the same length",
    ):
        survival.basehaz([1.0], [1, 0], [0.1], False)
