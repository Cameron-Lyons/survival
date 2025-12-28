import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing collapse ===")
    y: list[float] = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 0.0]
    x: list[int] = [1, 1, 1, 1]
    istate: list[int] = [0, 0, 0, 0]
    subject_id: list[int] = [1, 1, 2, 2]
    wt: list[float] = [1.0, 1.0, 1.0, 1.0]
    order: list[int] = [0, 1, 2, 3]

    result = survival.collapse(y, x, istate, subject_id, wt, order)
    print(" collapse executed successfully")
    print(f"   Result type: {type(result)}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert "matrix" in result, "Should have 'matrix' key"
    assert "dimnames" in result, "Should have 'dimnames' key"
    print(f"   matrix: {result['matrix']}")
    print(f"   dimnames: {result['dimnames']}")

    print("\n All utility tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in utility tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
