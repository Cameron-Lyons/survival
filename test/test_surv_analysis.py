import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing agsurv4 ===")
    ndeath = [1, 1, 0, 1, 0]
    risk = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0]
    sn = 5
    denom = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = survival.agsurv4(ndeath, risk, wt, sn, denom)
    print(" agsurv4 executed successfully")
    print(f"   Result: {result}")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == sn, "Should return same length as sn"

    print("\n=== Testing agsurv5 ===")
    n = 5
    nvar = 2
    dd = [1, 1, 2, 1, 1]
    x1 = [10.0, 9.0, 8.0, 7.0, 6.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0]
    xsum = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    xsum2 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.5, 2.0, 1.5, 1.0, 0.5]

    result = survival.agsurv5(n, nvar, dd, x1, x2, xsum, xsum2)
    print(" agsurv5 executed successfully")
    print(f"   Result type: {type(result)}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert "sum1" in result, "Should have 'sum1' key"
    assert "sum2" in result, "Should have 'sum2' key"
    assert "xbar" in result, "Should have 'xbar' key"
    print(f"   sum1: {result['sum1']}")
    print(f"   sum2: {result['sum2']}")

    print("\n=== Testing agmart ===")
    n = 5
    method = 0
    start = [0.0, 0.0, 1.0, 1.0, 2.0]
    stop = [1.0, 2.0, 2.0, 3.0, 3.0]
    event = [1, 0, 1, 0, 1]
    score = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    strata = [1, 0, 0, 0, 0]

    result = survival.agmart(n, method, start, stop, event, score, wt, strata)
    print(" agmart executed successfully")
    print(f"   Result: {result}")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == n, "Should return same length as n"

    print("\n All survival analysis tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in survival analysis tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
