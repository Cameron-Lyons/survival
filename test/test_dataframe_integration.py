#!/usr/bin/env python3
"""
Pytest-based integration tests for pandas/polars/numpy array support.
Tests that all major functions correctly accept different array-like inputs.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from helpers import setup_survival_import

survival = setup_survival_import()

# Try importing optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# Test data fixtures
@pytest.fixture
def survival_data():
    """Basic survival data for testing."""
    return {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "group": [0, 0, 0, 0, 1, 1, 1, 1],
        "weights": [1.0, 1.0, 2.0, 1.0, 1.5, 1.0, 1.0, 1.0],
    }


@pytest.fixture
def covariate_data():
    """Data with covariates for regression testing."""
    return {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        "covariates": [[0.5], [0.3], [0.8], [0.2], [0.9], [0.4], [0.6], [0.1], [0.7], [0.5]],
    }


class TestSurvfitKM:
    """Test survfitkm with different input types."""

    def test_with_lists(self, survival_data):
        result = survival.survfitkm(survival_data["time"], survival_data["status"])
        assert hasattr(result, "estimate")
        assert len(result.estimate) > 0

    def test_with_numpy_arrays(self, survival_data):
        time = np.array(survival_data["time"])
        status = np.array(survival_data["status"], dtype=np.float64)
        result = survival.survfitkm(time, status)
        assert hasattr(result, "estimate")
        assert len(result.estimate) > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_with_pandas_series(self, survival_data):
        df = pd.DataFrame(survival_data)
        result = survival.survfitkm(df["time"], df["status"].astype(float))
        assert hasattr(result, "estimate")
        assert len(result.estimate) > 0

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_with_polars_series(self, survival_data):
        df = pl.DataFrame(survival_data)
        result = survival.survfitkm(df["time"], df["status"].cast(pl.Float64))
        assert hasattr(result, "estimate")
        assert len(result.estimate) > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_with_pandas_weights(self, survival_data):
        df = pd.DataFrame(survival_data)
        result = survival.survfitkm(df["time"], df["status"].astype(float), weights=df["weights"])
        assert hasattr(result, "estimate")


class TestLogRank:
    """Test logrank_test with different input types."""

    def test_with_lists(self, survival_data):
        result = survival.logrank_test(
            survival_data["time"], survival_data["status"], survival_data["group"]
        )
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")

    def test_with_numpy_arrays(self, survival_data):
        time = np.array(survival_data["time"])
        status = np.array(survival_data["status"], dtype=np.int32)
        group = np.array(survival_data["group"], dtype=np.int32)
        result = survival.logrank_test(time, status, group)
        assert hasattr(result, "statistic")

    def test_with_numpy_int64(self, survival_data):
        """Test that int64 arrays are automatically converted."""
        time = np.array(survival_data["time"])
        status = np.array(survival_data["status"], dtype=np.int64)
        group = np.array(survival_data["group"], dtype=np.int64)
        result = survival.logrank_test(time, status, group)
        assert hasattr(result, "statistic")

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_with_pandas_series(self, survival_data):
        df = pd.DataFrame(survival_data)
        result = survival.logrank_test(df["time"], df["status"], df["group"])
        assert hasattr(result, "statistic")

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_with_polars_series(self, survival_data):
        df = pl.DataFrame(survival_data)
        result = survival.logrank_test(df["time"], df["status"], df["group"])
        assert hasattr(result, "statistic")


class TestCrossValidation:
    """Test cross-validation functions with different input types."""

    def test_cv_cox_with_lists(self, covariate_data):
        result = survival.cv_cox_concordance(
            covariate_data["time"],
            covariate_data["status"],
            covariate_data["covariates"],
            n_folds=2,
        )
        assert hasattr(result, "mean_score")
        assert 0 <= result.mean_score <= 1

    def test_cv_cox_with_numpy(self, covariate_data):
        time = np.array(covariate_data["time"])
        status = np.array(covariate_data["status"], dtype=np.int32)
        result = survival.cv_cox_concordance(time, status, covariate_data["covariates"], n_folds=2)
        assert hasattr(result, "mean_score")

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_cv_cox_with_pandas(self, covariate_data):
        df = pd.DataFrame({"time": covariate_data["time"], "status": covariate_data["status"]})
        result = survival.cv_cox_concordance(
            df["time"], df["status"], covariate_data["covariates"], n_folds=2
        )
        assert hasattr(result, "mean_score")


class TestNelsonAalen:
    """Test Nelson-Aalen estimator with different input types."""

    def test_with_lists(self, survival_data):
        result = survival.nelson_aalen_estimator(survival_data["time"], survival_data["status"])
        assert hasattr(result, "cumulative_hazard")
        assert len(result.cumulative_hazard) > 0

    def test_with_numpy(self, survival_data):
        time = np.array(survival_data["time"])
        status = np.array(survival_data["status"], dtype=np.int32)
        result = survival.nelson_aalen_estimator(time, status)
        assert hasattr(result, "cumulative_hazard")

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_with_pandas(self, survival_data):
        df = pd.DataFrame(survival_data)
        result = survival.nelson_aalen_estimator(df["time"].tolist(), df["status"].tolist())
        assert hasattr(result, "cumulative_hazard")


class TestRMST:
    """Test RMST functions with different input types."""

    def test_rmst_with_lists(self, survival_data):
        result = survival.rmst(survival_data["time"], survival_data["status"], tau=6.0)
        assert hasattr(result, "rmst")
        assert result.rmst > 0

    def test_rmst_comparison_with_lists(self, survival_data):
        result = survival.rmst_comparison(
            survival_data["time"],
            survival_data["status"],
            survival_data["group"],
            tau=6.0,
        )
        assert hasattr(result, "difference")
        assert hasattr(result, "p_value")


class TestHazardRatio:
    """Test hazard ratio with different input types."""

    def test_with_lists(self, survival_data):
        result = survival.hazard_ratio(
            survival_data["time"], survival_data["status"], survival_data["group"]
        )
        assert hasattr(result, "hazard_ratio")
        assert hasattr(result, "p_value")
        assert result.hazard_ratio > 0


class TestConsistency:
    """Test that results are consistent across input types."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_survfitkm_consistency(self, survival_data):
        """Results should be identical regardless of input type."""
        # List input
        result_list = survival.survfitkm(survival_data["time"], survival_data["status"])

        # NumPy input
        time_np = np.array(survival_data["time"])
        status_np = np.array(survival_data["status"], dtype=np.float64)
        result_np = survival.survfitkm(time_np, status_np)

        # Pandas input
        df = pd.DataFrame(survival_data)
        result_pd = survival.survfitkm(df["time"], df["status"].astype(float))

        # Check consistency
        np.testing.assert_array_almost_equal(result_list.estimate, result_np.estimate)
        np.testing.assert_array_almost_equal(result_list.estimate, result_pd.estimate)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_logrank_consistency(self, survival_data):
        """Log-rank results should be identical regardless of input type."""
        # List input
        result_list = survival.logrank_test(
            survival_data["time"], survival_data["status"], survival_data["group"]
        )

        # Pandas input
        df = pd.DataFrame(survival_data)
        result_pd = survival.logrank_test(df["time"], df["status"], df["group"])

        assert abs(result_list.statistic - result_pd.statistic) < 1e-10
        assert abs(result_list.p_value - result_pd.p_value) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
