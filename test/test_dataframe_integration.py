#!/usr/bin/env python3
"""
Integration tests for pandas/polars/numpy array support.
Tests that major functions correctly accept different array-like inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from helpers import setup_survival_import

survival = setup_survival_import()
print("Successfully imported survival module")

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status_data = [1, 1, 0, 1, 0, 1, 1, 0]
group_data = [0, 0, 0, 0, 1, 1, 1, 1]

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas not installed, skipping pandas tests")

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("polars not installed, skipping polars tests")


print("\n=== Testing survfitkm consistency across input types ===")

print("\n1. Testing survfitkm with Python lists...")
result_list = survival.survfitkm(time_data, [float(s) for s in status_data])
assert hasattr(result_list, "estimate"), "Should have estimate attribute"
assert len(result_list.estimate) > 0, "Should have estimates"
print(f"   Passed - {len(result_list.estimate)} estimates")

print("\n2. Testing survfitkm with NumPy arrays...")
time_np = np.array(time_data)
status_np = np.array(status_data, dtype=np.float64)
result_np = survival.survfitkm(time_np, status_np)
assert hasattr(result_np, "estimate"), "Should have estimate attribute"
assert len(result_np.estimate) > 0, "Should have estimates"
print(f"   Passed - {len(result_np.estimate)} estimates")

assert len(result_list.estimate) == len(result_np.estimate), "Results should have same length"
for i, (v1, v2) in enumerate(zip(result_list.estimate, result_np.estimate, strict=True)):
    assert abs(v1 - v2) < 1e-10, f"Results differ at index {i}: {v1} vs {v2}"
print("   Consistency check passed - list and numpy results identical")

if HAS_PANDAS:
    print("\n3. Testing survfitkm with Pandas Series...")
    df = pd.DataFrame({"time": time_data, "status": [float(s) for s in status_data]})
    result_pd = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result_pd, "estimate"), "Should have estimate attribute"
    assert len(result_pd.estimate) > 0, "Should have estimates"
    for i, (v1, v2) in enumerate(zip(result_list.estimate, result_pd.estimate, strict=True)):
        assert abs(v1 - v2) < 1e-10, f"Pandas results differ at index {i}"
    print("   Passed - pandas results identical to list results")

if HAS_POLARS:
    print("\n4. Testing survfitkm with Polars Series...")
    df = pl.DataFrame({"time": time_data, "status": [float(s) for s in status_data]})
    result_pl = survival.survfitkm(df["time"], df["status"])
    assert hasattr(result_pl, "estimate"), "Should have estimate attribute"
    assert len(result_pl.estimate) > 0, "Should have estimates"
    for i, (v1, v2) in enumerate(zip(result_list.estimate, result_pl.estimate, strict=True)):
        assert abs(v1 - v2) < 1e-10, f"Polars results differ at index {i}"
    print("   Passed - polars results identical to list results")


print("\n=== Testing logrank_test consistency ===")

print("\n1. Testing logrank_test with Python lists...")
result_list = survival.logrank_test(time_data, status_data, group_data)
assert hasattr(result_list, "statistic"), "Should have statistic"
assert hasattr(result_list, "p_value"), "Should have p_value"
print(f"   Passed - statistic: {result_list.statistic:.4f}, p-value: {result_list.p_value:.4f}")

print("\n2. Testing logrank_test with NumPy arrays...")
time_np = np.array(time_data)
status_np = np.array(status_data, dtype=np.int32)
group_np = np.array(group_data, dtype=np.int32)
result_np = survival.logrank_test(time_np, status_np, group_np)
assert hasattr(result_np, "statistic"), "Should have statistic"
assert abs(result_list.statistic - result_np.statistic) < 1e-10, "Statistics should match"
assert abs(result_list.p_value - result_np.p_value) < 1e-10, "P-values should match"
print("   Passed - results identical to list input")

print("\n3. Testing logrank_test with NumPy int64 arrays...")
status_np64 = np.array(status_data, dtype=np.int64)
group_np64 = np.array(group_data, dtype=np.int64)
result_np64 = survival.logrank_test(time_np, status_np64, group_np64)
assert abs(result_list.statistic - result_np64.statistic) < 1e-10, "int64 should auto-convert"
print("   Passed - int64 arrays auto-converted correctly")

if HAS_PANDAS:
    print("\n4. Testing logrank_test with Pandas Series...")
    df = pd.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pd = survival.logrank_test(df["time"], df["status"], df["group"])
    assert abs(result_list.statistic - result_pd.statistic) < 1e-10, "Pandas results should match"
    print("   Passed - pandas results identical")

if HAS_POLARS:
    print("\n5. Testing logrank_test with Polars Series...")
    df = pl.DataFrame({"time": time_data, "status": status_data, "group": group_data})
    result_pl = survival.logrank_test(df["time"], df["status"], df["group"])
    assert abs(result_list.statistic - result_pl.statistic) < 1e-10, "Polars results should match"
    print("   Passed - polars results identical")


print("\n=== Testing nelson_aalen_estimator ===")

print("\n1. Testing with Python lists...")
result = survival.nelson_aalen_estimator(time_data, status_data)
assert hasattr(result, "cumulative_hazard"), "Should have cumulative_hazard"
assert len(result.cumulative_hazard) > 0, "Should have values"
print(f"   Passed - {len(result.cumulative_hazard)} hazard values")

print("\n2. Testing with NumPy arrays...")
result_np = survival.nelson_aalen_estimator(
    np.array(time_data), np.array(status_data, dtype=np.int32)
)
assert len(result.cumulative_hazard) == len(result_np.cumulative_hazard)
print("   Passed - numpy results have same length")


print("\n=== Testing rmst ===")

print("\n1. Testing rmst with lists...")
result = survival.rmst(time_data, status_data, tau=6.0)
assert hasattr(result, "rmst"), "Should have rmst"
assert result.rmst > 0, "RMST should be positive"
print(f"   Passed - RMST: {result.rmst:.4f}")

print("\n2. Testing rmst_comparison...")
result = survival.rmst_comparison(time_data, status_data, group_data, tau=6.0)
assert hasattr(result, "rmst_diff"), "Should have rmst_diff"
assert hasattr(result, "p_value"), "Should have p_value"
print(f"   Passed - rmst_diff: {result.rmst_diff:.4f}, p-value: {result.p_value:.4f}")


print("\n=== Testing hazard_ratio ===")

print("\n1. Testing with lists...")
result = survival.hazard_ratio(time_data, status_data, group_data)
assert hasattr(result, "hazard_ratio"), "Should have hazard_ratio"
assert result.hazard_ratio > 0, "HR should be positive"
print(f"   Passed - HR: {result.hazard_ratio:.4f}")


print("\n" + "=" * 60)
print("All dataframe integration tests passed!")
print("=" * 60)
