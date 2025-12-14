import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target", "wheels"))

try:
    import survival

    print(" Successfully imported survival module")

    time_data: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    indices: list[int] = [0, 1, 2, 3, 4]
    ntree: int = 5

    print(f"Testing concordance1 with {len(weights)} observations...")

    result = survival.perform_concordance1_calculation(time_data, weights, indices, ntree)

    print(" concordance1 function executed successfully")
    print(f"Results: {result}")

    expected_keys: list[str] = [
        "concordant",
        "discordant",
        "tied_x",
        "tied_y",
        "tied_xy",
        "concordance_index",
        "total_pairs",
        "counts",
    ]
    for key in expected_keys:
        if key in result:
            print(f" {key}: {result[key]}")
        else:
            print(f" Missing key: {key}")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
except Exception as e:
    print(f" Error testing concordance1: {e}")
    import traceback

    traceback.print_exc()
