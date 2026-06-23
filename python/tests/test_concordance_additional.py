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

    with pytest.raises(RuntimeError, match="predictor_values contains negative value"):
        survival.perform_concordance_calculation(
            extended_time_data,
            [0, -1, 2, 3],
            extended_weights,
            time_weights,
            [0, 1, 2, 3],
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
