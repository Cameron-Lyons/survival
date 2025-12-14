import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing cipoisson_exact ===")
    result = survival.cipoisson_exact(k=5, time=10.0, p=0.95)
    print(" cipoisson_exact executed successfully")
    print(f"   Result (lower, upper): {result}")
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return (lower, upper)"
    assert result[0] < result[1], "Lower bound should be less than upper bound"

    print("\n=== Testing cipoisson_anscombe ===")
    result = survival.cipoisson_anscombe(k=5, time=10.0, p=0.95)
    print(" cipoisson_anscombe executed successfully")
    print(f"   Result (lower, upper): {result}")
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return (lower, upper)"

    print("\n=== Testing cipoisson ===")
    result_exact = survival.cipoisson(k=5, time=10.0, p=0.95, method="exact")
    result_anscombe = survival.cipoisson(k=5, time=10.0, p=0.95, method="anscombe")
    print(" cipoisson executed successfully")
    print(f"   Exact method: {result_exact}")
    print(f"   Anscombe method: {result_anscombe}")
    assert isinstance(result_exact, tuple), "Should return a tuple"
    assert isinstance(result_anscombe, tuple), "Should return a tuple"

    print("\n=== Testing norisk ===")
    time1: list[float] = [0.0, 1.0, 2.0, 3.0, 4.0]
    time2: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
    status: list[int] = [1, 0, 1, 0, 1]
    sort1: list[int] = [0, 1, 2, 3, 4]
    sort2: list[int] = [0, 1, 2, 3, 4]
    strata: list[int] = [1, 0, 0, 0, 0]

    result = survival.norisk(time1, time2, status, sort1, sort2, strata)
    print(" norisk executed successfully")
    print(f"   Result: {result}")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == len(time1), "Should return same length as input"

    print("\n=== Testing finegray ===")
    tstart: list[float] = [0.0, 0.0, 0.0, 0.0]
    tstop: list[float] = [1.0, 2.0, 3.0, 4.0]
    ctime: list[float] = [0.5, 1.5, 2.5, 3.5]
    cprob: list[float] = [0.1, 0.2, 0.3, 0.4]
    extend: list[bool] = [True, True, False, False]
    keep: list[bool] = [True, True, True, True]

    result = survival.finegray(
        tstart=tstart,
        tstop=tstop,
        ctime=ctime,
        cprob=cprob,
        extend=extend,
        keep=keep,
    )
    print(" finegray executed successfully")
    assert hasattr(result, "row"), "Should have row attribute"
    assert hasattr(result, "start"), "Should have start attribute"
    assert hasattr(result, "end"), "Should have end attribute"
    assert hasattr(result, "wt"), "Should have wt attribute"
    assert len(result.row) > 0, "Should have rows"
    print(f"   Number of rows: {len(result.row)}")
    print(f"   First row: {result.row[0]}")

    print("\n All specialized tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in specialized tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
