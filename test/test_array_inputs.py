#!/usr/bin/env python3
"""
Test that survival functions accept various array-like inputs:
- Python lists
- NumPy arrays
- Pandas Series
- Polars Series
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from helpers import setup_survival_import

survival = setup_survival_import()
print("Successfully imported survival module")

print("\n=== Testing survfitkm with different input types ===")

time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
status_list = [1.0, 1.0, 0.0, 1.0, 0.0]

print("\n1. Testing with Python lists...")
result = survival.survfitkm(time_list, status_list)
assert hasattr(result, "estimate"), "Should have estimate attribute"
assert len(result.estimate) > 0, "Should have estimates"
print(f"   Passed - estimates: {result.estimate}")

print("\n2. Testing with NumPy arrays...")
time_np = np.array(time_list)
status_np = np.array(status_list)
result = survival.survfitkm(time_np, status_np)
assert hasattr(result, "estimate"), "Should have estimate attribute"
assert len(result.estimate) > 0, "Should have estimates"
print(f"   Passed - estimates: {result.estimate}")

try:
    import pandas as pd

    print("\n3. Testing with Pandas Series...")
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result, "estimate"), "Should have estimate attribute"
    assert len(result.estimate) > 0, "Should have estimates"
    print(f"   Passed - estimates: {result.estimate}")

    print("\n4. Testing with Pandas DataFrame columns (via .values)...")
    result = survival.survfitkm(df["time"].values, df["status"].values)
    assert hasattr(result, "estimate"), "Should have estimate attribute"
    print("   Passed")

except ImportError:
    print("\n3-4. Skipping Pandas tests (pandas not installed)")

try:
    import polars as pl

    print("\n5. Testing with Polars Series...")
    df = pl.DataFrame({"time": time_list, "status": status_list})
    result = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result, "estimate"), "Should have estimate attribute"
    assert len(result.estimate) > 0, "Should have estimates"
    print(f"   Passed - estimates: {result.estimate}")

except ImportError:
    print("\n5. Skipping Polars tests (polars not installed)")


print("\n=== Testing logrank_test with different input types ===")

time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
status_list = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
group_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

print("\n1. Testing with Python lists...")
result = survival.logrank_test(time_list, status_list, group_list)
assert hasattr(result, "statistic"), "Should have statistic"
assert hasattr(result, "p_value"), "Should have p_value"
print(f"   Passed - statistic: {result.statistic:.4f}, p-value: {result.p_value:.4f}")

print("\n2. Testing with NumPy arrays...")
time_np = np.array(time_list)
status_np = np.array(status_list, dtype=np.int32)
group_np = np.array(group_list, dtype=np.int32)
result = survival.logrank_test(time_np, status_np, group_np)
assert hasattr(result, "statistic"), "Should have statistic"
print(f"   Passed - statistic: {result.statistic:.4f}")

print("\n3. Testing with NumPy int64 arrays (should auto-convert)...")
status_np64 = np.array(status_list, dtype=np.int64)
group_np64 = np.array(group_list, dtype=np.int64)
result = survival.logrank_test(time_np, status_np64, group_np64)
assert hasattr(result, "statistic"), "Should have statistic"
print(f"   Passed - statistic: {result.statistic:.4f}")

try:
    import pandas as pd

    print("\n4. Testing with Pandas Series...")
    df = pd.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic"), "Should have statistic"
    print(f"   Passed - statistic: {result.statistic:.4f}")

except ImportError:
    print("\n4. Skipping Pandas tests (pandas not installed)")

try:
    import polars as pl

    print("\n5. Testing with Polars Series...")
    df = pl.DataFrame({"time": time_list, "status": status_list, "group": group_list})
    result = survival.logrank_test(df["time"], df["status"], df["group"])
    assert hasattr(result, "statistic"), "Should have statistic"
    print(f"   Passed - statistic: {result.statistic:.4f}")

except ImportError:
    print("\n5. Skipping Polars tests (polars not installed)")


print("\n=== Testing cv_cox_concordance with different input types ===")

time_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
status_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
covariates = [[0.5], [0.3], [0.8], [0.2], [0.9], [0.4], [0.6], [0.1], [0.7], [0.5]]

print("\n1. Testing with Python lists...")
result = survival.cv_cox_concordance(time_list, status_list, covariates, n_folds=2)
assert hasattr(result, "mean_score"), "Should have mean_score"
print(f"   Passed - mean C-index: {result.mean_score:.4f}")

print("\n2. Testing with NumPy arrays...")
time_np = np.array(time_list)
status_np = np.array(status_list, dtype=np.int32)
result = survival.cv_cox_concordance(time_np, status_np, covariates, n_folds=2)
assert hasattr(result, "mean_score"), "Should have mean_score"
print(f"   Passed - mean C-index: {result.mean_score:.4f}")

try:
    import pandas as pd

    print("\n3. Testing with Pandas Series...")
    df = pd.DataFrame({"time": time_list, "status": status_list})
    result = survival.cv_cox_concordance(df["time"], df["status"], covariates, n_folds=2)
    assert hasattr(result, "mean_score"), "Should have mean_score"
    print(f"   Passed - mean C-index: {result.mean_score:.4f}")

except ImportError:
    print("\n3. Skipping Pandas tests (pandas not installed)")


print("\n=== Testing optional parameters with different types ===")

print("\n1. Testing survfitkm with optional weights as numpy array...")
weights_np = np.array([1.0, 1.0, 2.0, 1.0, 1.5])
result = survival.survfitkm(
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([1.0, 1.0, 0.0, 1.0, 0.0]),
    weights=weights_np,
)
assert hasattr(result, "estimate"), "Should have estimate"
print("   Passed")

try:
    import pandas as pd

    print("\n2. Testing survfitkm with optional weights as Pandas Series...")
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "status": [1.0, 1.0, 0.0, 1.0, 0.0],
            "weights": [1.0, 1.0, 2.0, 1.0, 1.5],
        }
    )
    result = survival.survfitkm(df["time"], df["status"], weights=df["weights"])
    assert hasattr(result, "estimate"), "Should have estimate"
    print("   Passed")

except ImportError:
    print("\n2. Skipping Pandas optional weights test")


print("\n" + "=" * 50)
print("All array input tests passed!")
print("=" * 50)
