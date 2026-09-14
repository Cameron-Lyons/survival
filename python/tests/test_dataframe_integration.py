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

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status_data = [1, 1, 0, 1, 0, 1, 1, 0]
group_data = [0, 0, 0, 0, 1, 1, 1, 1]


# survfit(Surv(time, status) ~ 1)$surv on the data above
_KM_SURV = [0.875, 0.75, 0.75, 0.6, 0.6, 0.4, 0.2, 0.2]


def test_survfitkm_list():
    result_list = survival.surv_analysis.survfitkm(time_data, status_data)
    assert result_list.surv == pytest.approx(_KM_SURV)


def test_survfitkm_numpy():
    result_np = survival.surv_analysis.survfitkm(np.array(time_data), np.array(status_data))
    assert result_np.surv == pytest.approx(_KM_SURV)


def test_survfitkm_list_numpy_consistency():
    result_list = survival.surv_analysis.survfitkm(time_data, status_data)
    result_np = survival.surv_analysis.survfitkm(np.array(time_data), np.array(status_data))
    assert result_np.surv == pytest.approx(result_list.surv, abs=1e-10)
    assert result_np.std_err == pytest.approx(result_list.std_err, abs=1e-10)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_survfitkm_pandas():
    df = pd.DataFrame({"time": time_data, "status": status_data})
    result_pd = survival.surv_analysis.survfitkm(df["time"], df["status"])
    assert result_pd.surv == pytest.approx(_KM_SURV)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_survfitkm_polars():
    df = pl.DataFrame({"time": time_data, "status": status_data})
    result_pl = survival.surv_analysis.survfitkm(df["time"], df["status"])
    assert result_pl.surv == pytest.approx(_KM_SURV)


def test_logrank_list():
    result_list = survival.validation.logrank_test(time_data, status_data, group_data)
    assert hasattr(result_list, "statistic")
    assert hasattr(result_list, "p_value")


def test_logrank_numpy():
    time_np = np.array(time_data)
    status_np = np.array(status_data, dtype=np.int32)
    group_np = np.array(group_data, dtype=np.int32)
    result_np = survival.validation.logrank_test(time_np, status_np, group_np)
    assert hasattr(result_np, "statistic")

    result_list = survival.validation.logrank_test(time_data, status_data, group_data)
    assert result_list.statistic == pytest.approx(result_np.statistic, abs=1e-10)
    assert result_list.p_value == pytest.approx(result_np.p_value, abs=1e-10)


def test_logrank_numpy_int64():
    time_np = np.array(time_data)
    status_np64 = np.array(status_data, dtype=np.int64)
    group_np64 = np.array(group_data, dtype=np.int64)
    result_np64 = survival.validation.logrank_test(time_np, status_np64, group_np64)

    result_list = survival.validation.logrank_test(time_data, status_data, group_data)
    assert result_list.statistic == pytest.approx(result_np64.statistic, abs=1e-10)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_logrank_pandas():
    result_list = survival.validation.logrank_test(time_data, status_data, group_data)
    df = pd.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pd = survival.validation.logrank_test(df["time"], df["status"], df["group"])
    assert result_list.statistic == pytest.approx(result_pd.statistic, abs=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_logrank_polars():
    result_list = survival.validation.logrank_test(time_data, status_data, group_data)
    df = pl.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pl = survival.validation.logrank_test(df["time"], df["status"], df["group"])
    assert result_list.statistic == pytest.approx(result_pl.statistic, abs=1e-10)


def test_nelson_aalen_list():
    result = survival.surv_analysis.nelson_aalen(time_data, status_data)
    # cumulative sum of d/n at the event times 1, 2, 4, 6, 7
    assert result.time == pytest.approx([1.0, 2.0, 4.0, 6.0, 7.0])
    assert result.cumulative_hazard == pytest.approx(
        [
            1 / 8,
            1 / 8 + 1 / 7,
            1 / 8 + 1 / 7 + 1 / 5,
            1 / 8 + 1 / 7 + 1 / 5 + 1 / 3,
            1 / 8 + 1 / 7 + 1 / 5 + 1 / 3 + 1 / 2,
        ]
    )


def test_nelson_aalen_numpy():
    result = survival.surv_analysis.nelson_aalen(time_data, status_data)
    result_np = survival.surv_analysis.nelson_aalen(
        np.array(time_data), np.array(status_data, dtype=np.int32)
    )
    assert result_np.cumulative_hazard == pytest.approx(result.cumulative_hazard)


def test_survmean_restricted_mean():
    fit = survival.surv_analysis.survfitkm(time_data, status_data)
    table = survival.surv_analysis.survmean(fit, rmean="6")
    # summary(fit, rmean = 6)$table["rmean"]: area under the KM curve up to 6
    assert table.rmean == pytest.approx([1 + 0.875 + 0.75 * 2 + 0.6 * 2])
    assert table.rmean[0] > 0


def test_rmst_comparison():
    result = survival.validation.rmst_comparison(time_data, status_data, group_data, tau=6.0)
    assert result.tau == pytest.approx(6.0)
    assert [group.group for group in result.groups] == [0, 1]
    assert len(result.difference) == 1
    assert 0.0 <= result.p_value <= 1.0


def test_hazard_ratio():
    result = survival.validation.hazard_ratio(time_data, status_data, group_data)
    assert hasattr(result, "hazard_ratio")
    assert result.hazard_ratio > 0
