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


def test_survfitkm_with_lists():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    status_list = [1.0, 1.0, 0.0, 1.0, 0.0]
    result = survival.survfitkm(time_list, status_list)
    assert hasattr(result, "estimate")
    assert len(result.estimate) > 0


def test_survfitkm_with_numpy():
    time_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    status_np = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    result = survival.survfitkm(time_np, status_np)
    assert hasattr(result, "estimate")
    assert len(result.estimate) > 0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_series():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    status_list = [1.0, 1.0, 0.0, 1.0, 0.0]
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result, "estimate")
    assert len(result.estimate) > 0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_values():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    status_list = [1.0, 1.0, 0.0, 1.0, 0.0]
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.survfitkm(df["time"].values, df["status"].values)
    assert hasattr(result, "estimate")


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_survfitkm_with_polars():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    status_list = [1.0, 1.0, 0.0, 1.0, 0.0]
    df = pl.DataFrame({"time": time_list, "status": status_list})
    result = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result, "estimate")
    assert len(result.estimate) > 0


def test_logrank_with_lists():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    result = survival.logrank_test(time_list, status_list, group_list)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")


def test_logrank_with_numpy():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    time_np = np.array(time_list)
    status_np = np.array(status_list, dtype=np.int32)
    group_np = np.array(group_list, dtype=np.int32)
    result = survival.logrank_test(time_np, status_np, group_np)
    assert hasattr(result, "statistic")


def test_logrank_with_numpy_int64():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    time_np = np.array(time_list)
    status_np64 = np.array(status_list, dtype=np.int64)
    group_np64 = np.array(group_list, dtype=np.int64)
    result = survival.logrank_test(time_np, status_np64, group_np64)
    assert hasattr(result, "statistic")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_logrank_with_pandas():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    df = pd.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic")


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_logrank_with_polars():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    df = pl.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic")


def test_cv_cox_concordance_with_lists():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    result = survival.cv_cox_concordance(time_list, status_list, covariates, n_folds=2)
    assert hasattr(result, "mean_score")


def test_cv_cox_concordance_with_numpy():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    time_np = np.array(time_list)
    status_np = np.array(status_list, dtype=np.int32)
    result = survival.cv_cox_concordance(time_np, status_np, covariates, n_folds=2)
    assert hasattr(result, "mean_score")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_cv_cox_concordance_with_pandas():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    covariates = [
        [0.5],
        [0.3],
        [0.8],
        [0.2],
        [0.9],
        [0.4],
        [0.6],
        [0.1],
        [0.7],
        [0.5],
    ]
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.cv_cox_concordance(df["time"], df["status"], covariates, n_folds=2)
    assert hasattr(result, "mean_score")


def test_survfitkm_with_numpy_weights():
    weights_np = np.array([1.0, 1.0, 2.0, 1.0, 1.5])
    result = survival.survfitkm(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 1.0, 0.0, 1.0, 0.0]),
        weights=weights_np,
    )
    assert hasattr(result, "estimate")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_with_pandas_weights():
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "status": [1.0, 1.0, 0.0, 1.0, 0.0],
            "weights": [1.0, 1.0, 2.0, 1.0, 1.5],
        }
    )
    result = survival.survfitkm(df["time"], df["status"], weights=df["weights"])
    assert hasattr(result, "estimate")
