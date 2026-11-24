"""Zero-based Python subset contract and model-frame column handling."""

import importlib

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
formula = importlib.import_module("survival.r._formula")


@pytest.mark.parametrize("subset", [[-1], [0, -2], [0.0, 1.0], [True]])
def test_python_subset_contract_remains_zero_based_and_strict(subset):
    with pytest.raises((TypeError, ValueError), match="subset"):
        r.survfit("Surv(time, status) ~ 1", {"time": [1, 2], "status": [1, 0]}, subset=subset)


def test_python_subset_keeps_censoring_when_it_selects_no_events():
    fit = r.survfit(
        "Surv(time, status) ~ 1", {"time": [1, 2, 3, 4], "status": [1, 2, 1, 2]}, subset=[0, 2]
    )
    assert list(fit.n_event) == [0, 0]
    assert list(fit.surv) == [1, 1]


def test_dataframe_column_named_response_cache_is_ordinary_data():
    pd = pytest.importorskip("pandas")
    frame = formula.model_frame(
        "Surv(time, status) ~ response_cache",
        pd.DataFrame({"time": [1, 2], "status": [1, 0], "response_cache": [3, 4]}),
    )
    assert frame.response.event == (1, 0)
