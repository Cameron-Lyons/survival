import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing Edge Cases and Error Handling ===")

    print("\n--- survfitkm validation tests ---")

    try:
        survival.survfitkm(time=[], status=[])
        print(" FAIL: Should have raised error for empty input")
        sys.exit(1)
    except ValueError as e:
        print(f" Empty input validation: OK ({e})")

    try:
        survival.survfitkm(time=[1.0, 2.0], status=[1.0])
        print(" FAIL: Should have raised error for length mismatch")
        sys.exit(1)
    except ValueError as e:
        print(f" Length mismatch validation: OK ({e})")

    try:
        survival.survfitkm(time=[-1.0, 2.0], status=[1.0, 0.0])
        print(" FAIL: Should have raised error for negative time")
        sys.exit(1)
    except ValueError as e:
        print(f" Negative time validation: OK ({e})")

    print("\n--- agmart validation tests ---")

    try:
        survival.agmart(
            n=3,
            method=0,
            start=[0.0, 0.0],
            stop=[1.0, 2.0, 3.0],
            event=[1, 0, 1],
            score=[1.0, 1.0, 1.0],
            wt=[1.0, 1.0, 1.0],
            strata=[0, 0, 0],
        )
        print(" FAIL: Should have raised error for length mismatch")
        sys.exit(1)
    except ValueError as e:
        print(f" Length mismatch (start) validation: OK ({e})")

    print("\n--- CoxPHModel validation tests ---")

    model = survival.CoxPHModel()

    try:
        model.fit(n_iters=10)
        print(" FAIL: Should have raised error for no data")
        sys.exit(1)
    except ValueError as e:
        print(f" Empty model fit validation: OK ({e})")

    subject = survival.Subject(
        id=1,
        covariates=[1.0, 2.0, 3.0],
        is_case=True,
        is_subcohort=True,
        stratum=0,
    )

    covariates = [[1.0, 2.0], [2.0, 3.0]]
    event_times = [1.0, 2.0]
    censoring = [1, 0]
    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)

    try:
        model.add_subject(subject)
        print(" FAIL: Should have raised error for dimension mismatch")
        sys.exit(1)
    except ValueError as e:
        print(f" Covariate dimension mismatch validation: OK ({e})")

    print("\n--- survdiff2 edge cases ---")

    result = survival.survdiff2(
        time=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        status=[1, 0, 1, 0, 1, 0],
        group=[1, 1, 1, 2, 2, 2],
        strata=[0, 0, 0, 0, 0, 0],
        rho=0.0,
    )
    print(f" Two groups: chi_squared={result.chi_squared:.4f}")

    print("\n--- All censored edge case ---")

    result = survival.survfitkm(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[0.0, 0.0, 0.0, 0.0],
    )
    print(f" All censored: estimate[0]={result.estimate[0] if result.estimate else 'empty'}")
    assert all(e == 1.0 for e in result.estimate), "All censored should have estimate=1.0"
    print(" All censored survival estimate correct")

    print("\n--- Single observation ---")

    result = survival.survfitkm(
        time=[5.0],
        status=[1.0],
    )
    print(f" Single observation: time={result.time}, estimate={result.estimate}")

    print("\n All edge case tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in edge case tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
