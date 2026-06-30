import numpy as np
import pytest

from .helpers import setup_survival_import

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


def test_logrank_accepts_weight_type_keyword():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    result = survival.logrank_test(
        time_list,
        status_list,
        group_list,
        weight_type="wilcoxon",
    )

    assert result.weight_type == "Wilcoxon"


def test_fleming_harrington_defaults_and_keyword_weights():
    time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
    status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    logrank = survival.logrank_test(time_list, status_list, group_list)
    default = survival.fleming_harrington_test(time_list, status_list, group_list)
    weighted = survival.fleming_harrington_test(
        time_list,
        status_list,
        group_list,
        rho=0.5,
        gamma=0.0,
    )

    assert default.statistic == pytest.approx(logrank.statistic)
    assert default.p_value == pytest.approx(logrank.p_value)
    assert weighted.weight_type == "FlemingHarrington(p=0.5, q=0)"


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


def test_logrank_with_strided_numpy_arrays():
    time = np.array([1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0, 5.0, -1.0])
    status = np.array([1, -1, 1, -1, 0, -1, 1, -1, 0, -1], dtype=np.int32)
    group = np.array([0, -1, 0, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)

    result = survival.logrank_test(time[::2], status[::2], group[::2])

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


def test_crossval_wrappers_validate_inputs_and_survreg_covariates():
    time = [float(i) for i in range(1, 31)]
    status_i32 = [1 if i % 3 != 0 else 0 for i in range(30)]
    status_f64 = [float(value) for value in status_i32]
    covariates = [[i / 30.0] for i in range(30)]

    cox = survival.cv_cox_concordance(time, status_i32, covariates, n_folds=3, seed=42)
    survreg = survival.cv_survreg_loglik(time, status_f64, covariates, "weibull", 3, True, 42)

    assert len(cox.fold_scores) == 3
    assert len(survreg.fold_scores) == 3
    assert all(len(coefficients) > 0 for coefficients in survreg.fold_coefficients)

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        survival.cv_cox_concordance(time, status_i32, covariates, n_folds=1)

    with pytest.raises(ValueError, match="time and status must have the same non-zero length"):
        survival.cv_cox_concordance(time, status_i32[:-1], covariates, n_folds=3)

    bad_time = list(time)
    bad_time[1] = float("nan")
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.cv_cox_concordance(bad_time, status_i32, covariates, n_folds=3)

    bad_status = list(status_i32)
    bad_status[2] = 2
    with pytest.raises(ValueError, match="status values must be 0 or 1"):
        survival.cv_cox_concordance(time, bad_status, covariates, n_folds=3)

    with pytest.raises(ValueError, match="covariates length must match time length"):
        survival.cv_survreg_loglik(time, status_f64, covariates[:-1], "weibull", 3, True, 42)

    ragged_covariates = [row[:] for row in covariates]
    ragged_covariates[3] = [0.1, 0.2]
    with pytest.raises(ValueError, match="covariates row 3 length"):
        survival.cv_survreg_loglik(time, status_f64, ragged_covariates, "weibull", 3, True, 42)

    bad_weights = [1.0] * 29 + [float("inf")]
    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.cv_cox_concordance(time, status_i32, covariates, weights=bad_weights, n_folds=3)


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
