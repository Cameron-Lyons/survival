import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()

HAS_PANDAS = False
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pass

HAS_POLARS = False
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pass

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status_data = [1, 1, 0, 1, 0, 1, 1, 0]
group_data = [0, 0, 0, 0, 1, 1, 1, 1]


def test_survfitkm_list():
    result_list = survival.survfitkm(time_data, [float(s) for s in status_data])
    assert hasattr(result_list, "estimate")
    assert len(result_list.estimate) > 0


def test_survfitkm_numpy():
    time_np = np.array(time_data)
    status_np = np.array(status_data, dtype=np.float64)
    result_np = survival.survfitkm(time_np, status_np)
    assert hasattr(result_np, "estimate")
    assert len(result_np.estimate) > 0


def test_survfitkm_list_numpy_consistency():
    result_list = survival.survfitkm(time_data, [float(s) for s in status_data])
    time_np = np.array(time_data)
    status_np = np.array(status_data, dtype=np.float64)
    result_np = survival.survfitkm(time_np, status_np)
    assert len(result_list.estimate) == len(result_np.estimate)
    for v1, v2 in zip(result_list.estimate, result_np.estimate, strict=True):
        assert v1 == pytest.approx(v2, abs=1e-10)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_pandas():
    result_list = survival.survfitkm(time_data, [float(s) for s in status_data])
    df = pd.DataFrame({"time": time_data, "status": [float(s) for s in status_data]})
    result_pd = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result_pd, "estimate")
    assert len(result_pd.estimate) > 0
    for v1, v2 in zip(result_list.estimate, result_pd.estimate, strict=True):
        assert v1 == pytest.approx(v2, abs=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_survfitkm_polars():
    result_list = survival.survfitkm(time_data, [float(s) for s in status_data])
    df = pl.DataFrame({"time": time_data, "status": [float(s) for s in status_data]})
    result_pl = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result_pl, "estimate")
    assert len(result_pl.estimate) > 0
    for v1, v2 in zip(result_list.estimate, result_pl.estimate, strict=True):
        assert v1 == pytest.approx(v2, abs=1e-10)


def test_logrank_list():
    result_list = survival.logrank_test(time_data, status_data, group_data)
    assert hasattr(result_list, "statistic")
    assert hasattr(result_list, "p_value")


def test_logrank_numpy():
    time_np = np.array(time_data)
    status_np = np.array(status_data, dtype=np.int32)
    group_np = np.array(group_data, dtype=np.int32)
    result_np = survival.logrank_test(time_np, status_np, group_np)
    assert hasattr(result_np, "statistic")

    result_list = survival.logrank_test(time_data, status_data, group_data)
    assert result_list.statistic == pytest.approx(result_np.statistic, abs=1e-10)
    assert result_list.p_value == pytest.approx(result_np.p_value, abs=1e-10)


def test_logrank_numpy_int64():
    time_np = np.array(time_data)
    status_np64 = np.array(status_data, dtype=np.int64)
    group_np64 = np.array(group_data, dtype=np.int64)
    result_np64 = survival.logrank_test(time_np, status_np64, group_np64)

    result_list = survival.logrank_test(time_data, status_data, group_data)
    assert result_list.statistic == pytest.approx(result_np64.statistic, abs=1e-10)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_logrank_pandas():
    result_list = survival.logrank_test(time_data, status_data, group_data)
    df = pd.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pd = survival.logrank_test(df["time"], df["status"], df["group"])
    assert result_list.statistic == pytest.approx(result_pd.statistic, abs=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_logrank_polars():
    result_list = survival.logrank_test(time_data, status_data, group_data)
    df = pl.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pl = survival.logrank_test(df["time"], df["status"], df["group"])
    assert result_list.statistic == pytest.approx(result_pl.statistic, abs=1e-10)


def test_nelson_aalen_list():
    result = survival.nelson_aalen_estimator(time_data, status_data)
    assert hasattr(result, "cumulative_hazard")
    assert len(result.cumulative_hazard) > 0


def test_nelson_aalen_numpy():
    result = survival.nelson_aalen_estimator(time_data, status_data)
    result_np = survival.nelson_aalen_estimator(
        np.array(time_data), np.array(status_data, dtype=np.int32)
    )
    assert len(result.cumulative_hazard) == len(result_np.cumulative_hazard)


def test_rmst_list():
    result = survival.rmst(time_data, status_data, tau=6.0)
    assert hasattr(result, "rmst")
    assert result.rmst > 0


def test_rmst_comparison():
    result = survival.rmst_comparison(time_data, status_data, group_data, tau=6.0)
    assert hasattr(result, "rmst_diff")
    assert hasattr(result, "p_value")


def test_hazard_ratio():
    result = survival.hazard_ratio(time_data, status_data, group_data)
    assert hasattr(result, "hazard_ratio")
    assert result.hazard_ratio > 0
