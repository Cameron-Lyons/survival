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
