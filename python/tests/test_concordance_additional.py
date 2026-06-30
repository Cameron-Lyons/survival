import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_concordance():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = [1, 2, 1, 2, 1]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    timewt = [1.0, 1.0, 1.0, 1.0, 1.0]
    sortstart = None
    sortstop = [0, 1, 2, 3, 4]

    result = survival.core.concordance(y, x, wt, timewt, sortstart, sortstop)
    assert isinstance(result, dict)
    assert "count" in result


def test_legacy_concordance_validates_index_inputs():
    y = [1.0, 2.0, 3.0]
    wt = [1.0, 1.0, 1.0]
    timewt = [1.0, 1.0, 1.0]

    with pytest.raises(ValueError, match="x length mismatch"):
        survival.core.concordance(y, [0, 1], wt, timewt, None, [0, 1, 2])

    with pytest.raises(ValueError, match="x contains negative value"):
        survival.core.concordance(y, [0, -1, 2], wt, timewt, None, [0, 1, 2])

    with pytest.raises(ValueError, match="x value 3 .* outside observation count"):
        survival.core.concordance(y, [0, 1, 3], wt, timewt, None, [0, 1, 2])

    with pytest.raises(ValueError, match="sortstop value 3 .* outside observation count"):
        survival.core.concordance(y, [0, 1, 2], wt, timewt, None, [0, 1, 3])

    with pytest.raises(ValueError, match="sortstart length mismatch"):
        survival.core.concordance(y, [0, 1, 2], wt, timewt, [0, 1], [0, 1, 2])

    with pytest.raises(ValueError, match="sortstart value 3 .* outside observation count"):
        survival.core.concordance(y, [0, 1, 2], wt, timewt, [0, 1, 3], [0, 1, 2])


def test_concordance_summary_counts_near_tied_risk_scores():
    summary = survival.core.concordance_summary(
        [1.0, 2.0, 3.0],
        [1, 1, 1],
        [0.5, 0.5 + 5e-11, 0.1],
    )

    assert summary["comparable"] == pytest.approx(3.0)
    assert summary["concordant"] == pytest.approx(2.5)
    assert summary["concordance"] == pytest.approx(2.5 / 3.0)
    assert survival.core.concordance_index(
        [1.0, 2.0, 3.0],
        [1, 1, 1],
        [0.5, 0.5 + 5e-11, 0.1],
    ) == pytest.approx(2.5 / 3.0)


def test_concordance_summary_groups_near_tied_event_times():
    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [1, 1, 1, 0]
    risk = [0.9, 0.1, 0.5, 0.2]

    exact = survival.core.concordance_summary(exact_time, status, risk)
    near = survival.core.concordance_summary(near_time, status, risk)
    assert near["concordant"] == pytest.approx(exact["concordant"])
    assert near["comparable"] == pytest.approx(exact["comparable"])
    assert near["concordance"] == pytest.approx(exact["concordance"])

    exact_weighted = survival.core.concordance_summary(exact_time, status, risk, timewt="S")
    near_weighted = survival.core.concordance_summary(near_time, status, risk, timewt="S")
    assert near_weighted["concordant"] == pytest.approx(exact_weighted["concordant"])
    assert near_weighted["comparable"] == pytest.approx(exact_weighted["comparable"])
    assert near_weighted["concordance"] == pytest.approx(exact_weighted["concordance"])


def test_concordance_diagnostic_rows_group_near_tied_event_times():
    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [1, 1, 1, 0]
    risk = [0.9, 0.1, 0.5, 0.2]
    weights = [2.0, 1.0, 3.0, 1.0]
    core = survival._survival

    exact_ranks = core.concordance_rank_rows(exact_time, status, risk, weights, timewt="S")
    near_ranks = core.concordance_rank_rows(near_time, status, risk, weights, timewt="S")
    assert len(near_ranks) == len(exact_ranks)
    for near_row, exact_row in zip(near_ranks, exact_ranks, strict=True):
        assert near_row[1:] == pytest.approx(exact_row[1:])

    exact_influence, exact_dfbeta, exact_variance = core.concordance_influence_rows(
        exact_time, status, risk, weights, timewt="S"
    )
    near_influence, near_dfbeta, near_variance = core.concordance_influence_rows(
        near_time, status, risk, weights, timewt="S"
    )
    assert len(near_influence) == len(exact_influence)
    for near_row, exact_row in zip(near_influence, exact_influence, strict=True):
        assert near_row == pytest.approx(exact_row)
    assert near_dfbeta == pytest.approx(exact_dfbeta)
    assert near_variance == pytest.approx(exact_variance)


def test_counting_concordance_summary_handles_duplicate_event_times():
    start = [0.0, 0.0, 0.25, 0.5, 0.0, 1.0, 1.0]
    stop = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    status = [1, 1, 1, 1, 0, 1, 0]
    risk = [0.9, 0.2, 0.7, 0.4, 0.1, 0.8, 0.3]
    weights = [1.0, 0.0, 2.0, 1.5, 0.5, 3.0, 1.0]

    summary = survival.core.counting_concordance_summary(start, stop, status, risk)
    assert summary["concordant"] == pytest.approx(6.0)
    assert summary["comparable"] == pytest.approx(8.0)
    assert summary["concordance"] == pytest.approx(0.75)

    weighted = survival.core.counting_concordance_summary(
        start, stop, status, risk, weights=weights
    )
    assert weighted["concordant"] == pytest.approx(7.5)
    assert weighted["comparable"] == pytest.approx(12.0)
    assert weighted["concordance"] == pytest.approx(0.625)


def test_concordance_public_apis_validate_values():
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.concordance_index([1.0, float("inf")], [1, 0], [0.4, 0.1])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.core.concordance_summary([1.0, 2.0], [1, 2], [0.4, 0.1])

    with pytest.raises(ValueError, match="risk_scores contains NaN"):
        survival.concordance_index([1.0, 2.0], [1, 0], [0.4, float("nan")])

    with pytest.raises(ValueError, match="start must be less than stop"):
        survival.counting_concordance_index(
            [0.0, 2.0],
            [1.0, 2.0],
            [1, 0],
            [0.4, 0.1],
        )

    with pytest.raises(ValueError, match="start contains negative"):
        survival.core.counting_concordance_summary(
            [-0.1, 0.0],
            [1.0, 2.0],
            [1, 0],
            [0.4, 0.1],
        )


def test_perform_concordance3_calculation():
    time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    indices = [0, 1, 2, 3, 4]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    time_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    sort_stop = [0, 1, 2, 3, 4]
    do_residuals = False
    result = survival.perform_concordance3_calculation(
        time_data, indices, weights, time_weights, sort_stop, do_residuals
    )
    assert isinstance(result, dict)
    assert "concordance_index" in result


def test_perform_concordance_calculation():
    time_data_v5 = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    predictor_values = [0, 1, 2, 3, 4]
    weights_v5 = [1.0, 1.0, 1.0, 1.0, 1.0]
    time_weights_v5 = [1.0, 1.0, 1.0, 1.0, 1.0]
    sort_stop_v5 = [0, 1, 2, 3, 4]
    result = survival.perform_concordance_calculation(
        time_data=time_data_v5,
        predictor_values=predictor_values,
        weights=weights_v5,
        time_weights=time_weights_v5,
        sort_stop=sort_stop_v5,
    )
    assert isinstance(result, dict)
    assert "concordance_index" in result


def test_low_level_concordance_wrappers_validate_index_inputs():
    time_data = [1.0, 2.0, 3.0, 1.0, 1.0, 0.0]
    weights = [1.0, 1.0, 1.0]

    with pytest.raises(RuntimeError, match="ntree must be positive"):
        survival.perform_concordance1_calculation(time_data, weights, [0, 1, 2], 0)

    with pytest.raises(RuntimeError, match="indices contains negative value"):
        survival.perform_concordance1_calculation(time_data, weights, [0, -1, 2], 4)

    with pytest.raises(RuntimeError, match="indices value 4 .* outside ntree"):
        survival.perform_concordance1_calculation(time_data, weights, [0, 1, 4], 4)

    extended_time_data = [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.0, 1.0]
    extended_weights = [1.0, 1.0, 1.0, 1.0]
    time_weights = [1.0, 1.0, 1.0, 1.0]

    with pytest.raises(RuntimeError, match="indices contains negative value"):
        survival.perform_concordance3_calculation(
            extended_time_data,
            [0, -1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
            False,
        )

    with pytest.raises(RuntimeError, match="sort_stop value 4 .* outside observation count"):
        survival.perform_concordance3_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 4],
            False,
        )

    with pytest.raises(RuntimeError, match="sort_stop must be a permutation"):
        survival.perform_concordance3_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 1, 3],
            False,
        )

    with pytest.raises(RuntimeError, match="predictor_values contains negative value"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, -1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
        )

    with pytest.raises(RuntimeError, match="sort_stop must be a permutation"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 1, 3],
        )

    with pytest.raises(RuntimeError, match="sort_start length"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
            [0, 1, 2],
        )

    with pytest.raises(RuntimeError, match="sort_start value 4 .* outside observation count"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
            [0, 1, 2, 4],
        )

    with pytest.raises(RuntimeError, match="sort_start must be a permutation"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, 1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
            [0, 1, 1, 3],
        )
