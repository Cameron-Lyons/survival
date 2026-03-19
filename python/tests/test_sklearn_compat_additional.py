import builtins
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

setup_survival_import()
sklearn_compat = importlib.import_module("survival.sklearn_compat")

CoxPHEstimator = sklearn_compat.CoxPHEstimator
GradientBoostSurvivalEstimator = sklearn_compat.GradientBoostSurvivalEstimator
SurvivalForestEstimator = sklearn_compat.SurvivalForestEstimator
StreamingCoxPHEstimator = sklearn_compat.StreamingCoxPHEstimator
StreamingGradientBoostSurvivalEstimator = sklearn_compat.StreamingGradientBoostSurvivalEstimator
StreamingSurvivalForestEstimator = sklearn_compat.StreamingSurvivalForestEstimator
iter_chunks = sklearn_compat.iter_chunks
predict_large_dataset = sklearn_compat.predict_large_dataset
survival_curves_to_disk = sklearn_compat.survival_curves_to_disk


def _toy_data():
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]], dtype=np.float64)
    y = np.column_stack(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 1, 0, 1, 0, 1, 1, 0],
        ]
    )
    return x, y


def test_coxph_estimator_smoke():
    x, y = _toy_data()
    estimator = CoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    risk = estimator.predict(x)
    times, survival = estimator.predict_survival_function(x)
    median = estimator.predict_median_survival_time(x)

    assert risk.shape == (x.shape[0],)
    assert times.shape == (x.shape[0],)
    assert survival.shape == (x.shape[0], x.shape[0])
    assert median.shape == (x.shape[0],)
    assert 0.0 <= estimator.score(x, y) <= 1.0


def test_coxph_estimator_custom_times_and_feature_validation():
    x, y = _toy_data()
    estimator = CoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    custom_times = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    returned_times, survival = estimator.predict_survival_function(x[:2], times=custom_times)

    assert returned_times.tolist() == pytest.approx(custom_times.tolist())
    assert survival.shape == (2, 3)

    with pytest.raises(ValueError, match="expects 1"):
        estimator.predict(np.array([[0.1, 0.2]], dtype=np.float64))


def test_tree_estimators_smoke():
    x, y = _toy_data()

    boost = GradientBoostSurvivalEstimator(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        seed=1,
    )
    boost.fit(x, y)
    assert boost.predict(x).shape == (x.shape[0],)
    assert boost.predict_survival_function(x)[1].shape == (x.shape[0], x.shape[0])
    assert boost.predict_median_survival_time(x).shape == (x.shape[0],)

    forest = SurvivalForestEstimator(
        n_trees=5,
        min_node_size=1,
        sample_fraction=0.8,
        seed=1,
        oob_error=False,
    )
    forest.fit(x, y)
    assert forest.predict(x).shape == (x.shape[0],)
    assert forest.predict_survival_function(x)[1].shape == (x.shape[0], x.shape[0])
    assert forest.predict_median_survival_time(x).shape == (x.shape[0],)


@pytest.mark.parametrize(
    "estimator_cls, kwargs",
    [
        (CoxPHEstimator, {"n_iters": 5}),
        (GradientBoostSurvivalEstimator, {"n_estimators": 5, "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "seed": 1}),
        (SurvivalForestEstimator, {"n_trees": 5, "min_node_size": 1, "sample_fraction": 0.8, "seed": 1, "oob_error": False}),
    ],
)
def test_estimators_require_fit_before_predict(estimator_cls, kwargs):
    estimator = estimator_cls(**kwargs)

    with pytest.raises(Exception):
        estimator.predict(np.array([[0.1]], dtype=np.float64))


def test_streaming_helpers_and_disk_io(tmp_path):
    x, y = _toy_data()
    estimator = StreamingCoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    chunks = list(iter_chunks(x, batch_size=3))
    assert [start for start, _ in chunks] == [0, 3, 6]
    assert [chunk.shape[0] for _, chunk in chunks] == [3, 3, 2]

    batched = np.concatenate(list(estimator.predict_batched(x, batch_size=3)))
    assert batched.shape == (x.shape[0],)

    survival_batches = list(estimator.predict_survival_batched(x, batch_size=3))
    assert len(survival_batches) == 3
    assert survival_batches[0][1].shape[1] == x.shape[0]

    out = np.empty(x.shape[0], dtype=np.float64)
    returned = estimator.predict_to_array(x, batch_size=3, out=out)
    assert returned is out
    assert np.all(np.isfinite(out))

    prediction_file = tmp_path / "predictions.dat"
    large = predict_large_dataset(estimator, x, batch_size=3, output_file=str(prediction_file))
    assert prediction_file.exists()
    assert large.shape == (x.shape[0],)

    survival_file = tmp_path / "survival_curves.dat"
    times, survival = survival_curves_to_disk(estimator, x, str(survival_file), batch_size=3)
    assert survival_file.exists()
    assert times.shape == (x.shape[0],)
    assert survival.shape == (x.shape[0], x.shape[0])


def test_streaming_tree_estimators_smoke():
    x, y = _toy_data()

    boost = StreamingGradientBoostSurvivalEstimator(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        seed=1,
    )
    boost.fit(x, y)
    assert np.concatenate(list(boost.predict_batched(x, batch_size=3))).shape == (x.shape[0],)

    forest = StreamingSurvivalForestEstimator(
        n_trees=5,
        min_node_size=1,
        sample_fraction=0.8,
        seed=1,
        oob_error=False,
    )
    forest.fit(x, y)
    assert np.concatenate(list(forest.predict_batched(x, batch_size=3))).shape == (x.shape[0],)


def test_predict_to_array_validates_output_shape():
    x, y = _toy_data()
    estimator = StreamingCoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    with pytest.raises(ValueError, match=r"expected \(8,\)"):
        estimator.predict_to_array(x, out=np.empty(4, dtype=np.float64))


def test_sklearn_compat_fallback_without_sklearn(monkeypatch):
    module_path = Path(sklearn_compat.__file__)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("scikit-learn intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec = importlib.util.spec_from_file_location("survival.sklearn_compat_no_sklearn", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._HAS_SKLEARN is False

    estimator = module.CoxPHEstimator()
    assert estimator.get_params() == {"n_iters": 20}
    estimator.set_params(n_iters=7)
    assert estimator.n_iters == 7

    with pytest.raises(ValueError, match="not fitted yet"):
        estimator.predict(np.array([[0.1]], dtype=np.float64))
