import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from helpers import setup_survival_import

survival = setup_survival_import()
print(" Successfully imported survival module")

print("\n=== Testing DeepSurv Configuration ===")

config = survival.DeepSurvConfig()
print(f"   Default config: hidden_layers={config.hidden_layers}")
print(f"   Default dropout_rate={config.dropout_rate}")
print(f"   Default learning_rate={config.learning_rate}")
print(f"   Default batch_size={config.batch_size}")
print(f"   Default n_epochs={config.n_epochs}")

assert config.hidden_layers == [64, 32], "Default hidden layers should be [64, 32]"
assert config.dropout_rate == 0.2, "Default dropout_rate should be 0.2"
assert config.learning_rate == 0.001, "Default learning_rate should be 0.001"

custom_config = survival.DeepSurvConfig(
    hidden_layers=[32, 16, 8],
    activation=survival.Activation("relu"),
    dropout_rate=0.1,
    learning_rate=0.01,
    batch_size=64,
    n_epochs=50,
    l2_reg=0.001,
    seed=42,
    early_stopping_patience=5,
    validation_fraction=0.2,
)
print(f"   Custom config: hidden_layers={custom_config.hidden_layers}")
assert custom_config.hidden_layers == [32, 16, 8], "Custom hidden layers mismatch"
assert custom_config.dropout_rate == 0.1, "Custom dropout_rate mismatch"
assert custom_config.early_stopping_patience == 5, "Custom early_stopping_patience mismatch"
print(" DeepSurvConfig tests passed")

print("\n=== Testing Activation enum ===")
relu_act = survival.Activation("relu")
selu_act = survival.Activation("selu")
tanh_act = survival.Activation("tanh")
print("   Created activations: relu, selu, tanh")

try:
    invalid_act = survival.Activation("invalid")
    print("   ERROR: Should have raised exception for invalid activation")
    sys.exit(1)
except ValueError as e:
    print(f"   Invalid activation correctly raised error: {e}")

print(" Activation tests passed")

print("\n=== Testing DeepSurv Model Training ===")

x = [
    1.0,
    0.5,
    0.2,
    0.8,
    0.3,
    0.1,
    0.6,
    0.7,
    0.4,
    0.4,
    0.2,
    0.8,
    0.9,
    0.1,
    0.3,
    0.3,
    0.8,
    0.5,
    0.7,
    0.4,
    0.6,
    0.2,
    0.6,
    0.9,
    0.5,
    0.9,
    0.1,
    1.0,
    0.0,
    0.7,
    0.1,
    0.5,
    0.2,
    0.6,
    0.3,
    0.4,
]
n_obs = 12
n_vars = 3
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
status = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

config = survival.DeepSurvConfig(
    hidden_layers=[8, 4],
    activation=survival.Activation("relu"),
    dropout_rate=0.0,
    learning_rate=0.01,
    batch_size=12,
    n_epochs=10,
    l2_reg=0.0,
    seed=42,
    early_stopping_patience=None,
    validation_fraction=0.0,
)

print("   Training DeepSurv model...")
model = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config)

assert model.n_features == n_vars, f"n_features should be {n_vars}"
assert model.hidden_layers == [8, 4], "hidden_layers mismatch"
assert len(model.train_loss) == 10, (
    f"Should have 10 epochs of training loss, got {len(model.train_loss)}"
)
assert len(model.unique_times) > 0, "Should have unique times"
assert len(model.baseline_hazard) > 0, "Should have baseline hazard"

print(f"   Model trained: n_features={model.n_features}")
print(f"   Training loss (first 3): {model.train_loss[:3]}")
print(f"   Unique times: {model.unique_times}")
print(" DeepSurv training tests passed")

print("\n=== Testing DeepSurv Predictions ===")

risk_scores = model.predict_risk(x, n_obs)
assert len(risk_scores) == n_obs, f"Risk scores length should be {n_obs}"
print(f"   Risk scores: {[f'{r:.4f}' for r in risk_scores[:5]]}...")

survival_probs = model.predict_survival(x, n_obs)
assert len(survival_probs) == n_obs, f"Survival probs should have {n_obs} rows"
assert len(survival_probs[0]) == len(model.unique_times), (
    "Survival probs columns should match unique_times"
)
for i, surv in enumerate(survival_probs):
    assert all(0.0 <= s <= 1.0 for s in surv), f"Survival probs at row {i} should be in [0, 1]"
    assert surv == sorted(surv, reverse=True), (
        f"Survival probs at row {i} should be monotonically decreasing"
    )
print(f"   Survival probs shape: ({len(survival_probs)}, {len(survival_probs[0])})")

cumhaz = model.predict_cumulative_hazard(x, n_obs)
assert len(cumhaz) == n_obs, f"Cumulative hazard should have {n_obs} rows"
for i, ch in enumerate(cumhaz):
    assert all(c >= 0.0 for c in ch), f"Cumulative hazard at row {i} should be non-negative"
    assert ch == sorted(ch), f"Cumulative hazard at row {i} should be monotonically increasing"
print(f"   Cumulative hazard shape: ({len(cumhaz)}, {len(cumhaz[0])})")

median_times = model.predict_median_survival_time(x, n_obs)
assert len(median_times) == n_obs, f"Median times length should be {n_obs}"
print(f"   Median survival times: {median_times[:5]}...")

percentile_times = model.predict_survival_time(x, n_obs, 0.75)
assert len(percentile_times) == n_obs, f"Percentile times length should be {n_obs}"
print(f"   75th percentile survival times: {percentile_times[:5]}...")

print(" DeepSurv prediction tests passed")

print("\n=== Testing DeepSurv with Validation and Early Stopping ===")

config_with_val = survival.DeepSurvConfig(
    hidden_layers=[8, 4],
    activation=survival.Activation("selu"),
    dropout_rate=0.1,
    learning_rate=0.01,
    batch_size=6,
    n_epochs=50,
    l2_reg=0.001,
    seed=42,
    early_stopping_patience=5,
    validation_fraction=0.25,
)

model_val = survival.DeepSurv.fit(x, n_obs, n_vars, time, status, config_with_val)
print(f"   Training epochs run: {len(model_val.train_loss)}")
print(f"   Validation epochs: {len(model_val.val_loss)}")
assert len(model_val.val_loss) > 0, "Should have validation loss when validation_fraction > 0"
assert len(model_val.val_loss) == len(model_val.train_loss), (
    "Train and val loss should have same length"
)
print(" DeepSurv validation tests passed")

print("\n=== Testing DeepSurv Error Handling ===")

try:
    bad_x = [1.0, 2.0, 3.0]
    survival.DeepSurv.fit(bad_x, 10, 3, time, status, config)
    print("   ERROR: Should have raised exception for mismatched x length")
    sys.exit(1)
except ValueError:
    print("   Mismatched x length correctly raised error")

try:
    bad_time = [1.0, 2.0]
    survival.DeepSurv.fit(x, n_obs, n_vars, bad_time, status, config)
    print("   ERROR: Should have raised exception for mismatched time length")
    sys.exit(1)
except ValueError:
    print("   Mismatched time length correctly raised error")

try:
    model.predict_risk([1.0, 2.0], 5)
    print("   ERROR: Should have raised exception for mismatched predict x")
    sys.exit(1)
except ValueError:
    print("   Mismatched predict x correctly raised error")

print(" DeepSurv error handling tests passed")

print("\n=== Testing deep_surv convenience function ===")

model_func = survival.deep_surv(x, n_obs, n_vars, time, status)
assert model_func.n_features == n_vars, "deep_surv function should work"
print(" deep_surv convenience function test passed")

print("\n=== Testing DeepSurvEstimator (sklearn-compat) ===")

try:
    import numpy as np
    from survival.sklearn_compat import DeepSurvEstimator

    X = np.array(x).reshape(n_obs, n_vars)
    y = np.column_stack([time, status])

    estimator = DeepSurvEstimator(
        hidden_layers=[8, 4],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=0.01,
        batch_size=12,
        n_epochs=10,
        seed=42,
        early_stopping_patience=None,
        validation_fraction=0.0,
    )
    print("   Created DeepSurvEstimator")

    estimator.fit(X, y)
    print(f"   Fitted estimator: n_features_in_={estimator.n_features_in_}")
    assert estimator.n_features_in_ == n_vars, "n_features_in_ should match"

    risk = estimator.predict(X)
    assert risk.shape == (n_obs,), f"Risk shape should be ({n_obs},)"
    print(f"   Predicted risk scores: shape={risk.shape}")

    times, surv = estimator.predict_survival_function(X)
    assert len(times) > 0, "Should have time points"
    assert surv.shape[0] == n_obs, "Survival should have n_obs rows"
    print(f"   Predicted survival function: times={len(times)}, surv={surv.shape}")

    median = estimator.predict_median_survival_time(X)
    assert median.shape == (n_obs,), f"Median shape should be ({n_obs},)"
    print(f"   Predicted median survival time: shape={median.shape}")

    score = estimator.score(X, y)
    assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"
    print(f"   Model score (C-index): {score:.4f}")

    train_loss = estimator.train_loss
    assert len(train_loss) == 10, "Should have 10 epochs of loss"
    print(f"   Training loss accessible: {len(train_loss)} epochs")

    print(" DeepSurvEstimator tests passed")

except ImportError as e:
    print(f"   Skipping sklearn_compat tests (import error): {e}")

print("\n=== All DeepSurv Tests Passed ===")
