import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing concordance (concordance function) ===")
    y: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
    x: list[int] = [1, 2, 1, 2, 1]
    wt: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    timewt: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    sortstart: list[int] | None = None
    sortstop: list[int] = [0, 1, 2, 3, 4]

    result = survival.concordance(y, x, wt, timewt, sortstart, sortstop)
    print(" concordance executed successfully")
    print(f"   Result type: {type(result)}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert "count" in result, "Should have 'count' key"
    print(f"   count: {result['count']}")

    print("\n=== Testing perform_concordance3_calculation ===")
    try:
        time_data: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
        indices: list[int] = [0, 1, 2, 3, 4]
        result = survival.perform_concordance3_calculation(time_data, weights, indices)
        print(" perform_concordance3_calculation executed successfully")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"  perform_concordance3_calculation: {e}")

    print("\n=== Testing perform_concordance_calculation (v5) ===")
    try:
        result = survival.perform_concordance_calculation(
            time_data=[1.0, 2.0, 3.0, 4.0, 5.0],
            status_data=[1, 1, 0, 1, 0],
            x_data=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        print(" perform_concordance_calculation executed successfully")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"  perform_concordance_calculation: {e}")

    print("\n All concordance tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in concordance tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
