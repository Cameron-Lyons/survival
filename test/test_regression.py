import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import setup_survival_import

    survival = setup_survival_import()
    print(" Successfully imported survival module")

    print("\n=== Testing survreg (Parametric Survival Regression) ===")
    time: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status: list[float] = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    covariates: list[list[float]] = [
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5],
    ]

    result = survival.survreg(
        time=time,
        status=status,
        covariates=covariates,
        weights=None,
        offsets=None,
        initial_beta=None,
        strata=None,
        distribution="extreme_value",
        max_iter=20,
        eps=1e-5,
        tol_chol=1e-9,
    )
    print(" survreg executed successfully")
    assert hasattr(result, "coefficients"), "Should have coefficients attribute"
    assert hasattr(result, "log_likelihood"), "Should have log_likelihood attribute"
    assert hasattr(result, "iterations"), "Should have iterations attribute"
    assert isinstance(result.coefficients, list), "Coefficients should be a list"
    print(f"   Coefficients: {result.coefficients}")
    print(f"   Log-likelihood: {result.log_likelihood:.4f}")
    print(f"   Iterations: {result.iterations}")

    print("\n=== Testing coxmart (Cox Martingale Residuals) ===")
    time: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status: list[int] = [1, 1, 0, 1, 0, 1, 1, 0]
    score: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    result = survival.coxmart(
        time=time,
        status=status,
        score=score,
        weights=None,
        strata=None,
        method=0,
    )
    print(" coxmart executed successfully")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == len(time), "Should return same length as time"
    print(f"   Residuals: {result[:3]}... (showing first 3)")

    print("\n=== Testing CoxPHModel ===")
    model = survival.CoxPHModel()
    print(" CoxPHModel created successfully")

    covariates: list[list[float]] = [
        [0.5, 1.2],
        [1.8, 0.3],
        [0.2, 2.1],
        [2.5, 0.8],
        [0.8, 1.5],
        [1.5, 0.5],
        [0.3, 1.8],
        [2.2, 1.1],
        [1.0, 0.9],
        [0.7, 1.7],
        [2.0, 0.4],
        [1.2, 1.3],
        [0.9, 2.0],
        [1.6, 0.7],
        [0.4, 1.4],
        [2.1, 1.0],
    ]
    event_times: list[float] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    censoring: list[int] = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]

    model = survival.CoxPHModel.new_with_data(covariates, event_times, censoring)
    print(" CoxPHModel.new_with_data executed successfully")

    model.fit(n_iters=20)
    print(" model.fit executed successfully")

    assert hasattr(model, "baseline_hazard"), "Should have baseline_hazard attribute"
    assert hasattr(model, "risk_scores"), "Should have risk_scores attribute"
    print(f"   Baseline hazard length: {len(model.baseline_hazard)}")
    print(f"   Risk scores: {model.risk_scores}")

    coefficients = model.coefficients
    print(f"   Coefficients: {coefficients}")

    new_covariates: list[list[float]] = [[1.0, 2.0], [2.0, 3.0]]
    predictions = model.predict(new_covariates)
    print(f"   Predictions: {predictions}")
    assert isinstance(predictions, list), "Predictions should be a list"

    brier = model.brier_score()
    print(f"   Brier score: {brier:.4f}")
    assert isinstance(brier, float), "Brier score should be a float"

    print("\n=== Testing Subject ===")
    subject = survival.Subject(
        id=1,
        covariates=[1.0, 2.0],
        is_case=True,
        is_subcohort=True,
        stratum=0,
    )
    print(" Subject created successfully")
    assert subject.id == 1, "ID should be 1"
    assert subject.is_case is True, "is_case should be True"

    print("\n All regression model tests passed!")

except ImportError as e:
    print(f" Failed to import survival module: {e}")
    print("Make sure to build the project first with: maturin build")
    sys.exit(1)
except Exception as e:
    print(f" Error in regression model tests: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
