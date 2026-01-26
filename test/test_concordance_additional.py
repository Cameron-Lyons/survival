import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()


def test_concordance():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = [1, 2, 1, 2, 1]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    timewt = [1.0, 1.0, 1.0, 1.0, 1.0]
    sortstart = None
    sortstop = [0, 1, 2, 3, 4]

    result = survival.concordance(y, x, wt, timewt, sortstart, sortstop)
    assert isinstance(result, dict)
    assert "count" in result


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
