import builtins
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
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
HAS_TREE_BINDINGS = all(
    hasattr(core, name)
    for name in (
        "GradientBoostSurvivalConfig",
        "gradient_boost_survival",
        "SurvivalForestConfig",
        "survival_forest",
    )
)


def _toy_data():
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]], dtype=np.float64)
    y = np.column_stack(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 1, 0, 1, 0, 1, 1, 0],
        ]
    )
    return x, y


class _StreamingDummy(sklearn_compat.StreamingMixin):
    def __init__(self):
        self.predict_shapes = []
        self.survival_shapes = []

    def predict(self, x):
        x = np.asarray(x)
        self.predict_shapes.append(x.shape)
        return np.zeros(x.shape[0], dtype=np.float64)

    def predict_survival_function(self, x):
        x = np.asarray(x)
        self.survival_shapes.append(x.shape)
        times = np.array([1.0, 2.0], dtype=np.float64)
        survival = np.ones((x.shape[0], times.shape[0]), dtype=np.float64)
        return times, survival


class _LazyRows:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.full_materializations = 0
        self.slices = []

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, key):
        self.slices.append(key)
        return self._data[key]

    def __array__(self, dtype=None):
        self.full_materializations += 1
        return np.asarray(self._data, dtype=dtype)


def test_score_uses_rust_concordance(monkeypatch):
    common = importlib.import_module("survival._sklearn_common")
    calls = []

    def fake_concordance_index(time, status, risk_scores):
        calls.append((time, status, risk_scores))
        return 0.8125

    monkeypatch.setattr(common._surv, "concordance_index", fake_concordance_index, raising=False)

    score = common._compute_concordance_index(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1, 0], dtype=np.int32),
        np.array([0.7, 0.2], dtype=np.float64),
    )

    assert score == 0.8125
    assert calls == [([1.0, 2.0], [1, 0], [0.7, 0.2])]


def test_coxph_estimator_smoke():
    x, y = _toy_data()
    estimator = CoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    assert isinstance(estimator.model_, core.CoxPHFit)
    assert estimator.coef_.shape == (x.shape[1],)

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
    assert np.all((survival >= 0.0) & (survival <= 1.0))

    baseline_times, baseline_hazard = estimator.model_.basehaz(True)
    positions = np.searchsorted(np.asarray(baseline_times), custom_times, side="right") - 1
    expected_hazard = np.zeros_like(custom_times)
    valid = positions >= 0
    expected_hazard[valid] = np.asarray(baseline_hazard)[positions[valid]]
    linear_predictors = np.asarray(estimator.model_.predict(x[:2].tolist()))
    center = np.mean(np.asarray(estimator.model_.linear_predictors))
    expected_survival = np.exp(-np.outer(np.exp(linear_predictors - center), expected_hazard))

    assert survival == pytest.approx(expected_survival)

    with pytest.raises(ValueError, match="expects 1"):
        estimator.predict(np.array([[0.1, 0.2]], dtype=np.float64))


@pytest.mark.skipif(
    not HAS_TREE_BINDINGS,
    reason="tree survival estimators require the Rust extension to be built with the ml feature",
)
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
    ("estimator_cls", "kwargs"),
    [
        (CoxPHEstimator, {"n_iters": 5}),
        (
            GradientBoostSurvivalEstimator,
            {
                "n_estimators": 5,
                "max_depth": 2,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "seed": 1,
            },
        ),
        (
            SurvivalForestEstimator,
            {
                "n_trees": 5,
                "min_node_size": 1,
                "sample_fraction": 0.8,
                "seed": 1,
                "oob_error": False,
            },
        ),
    ],
)
def test_estimators_require_fit_before_predict(estimator_cls, kwargs):
    estimator = estimator_cls(**kwargs)

    with pytest.raises(ValueError, match="not fitted"):
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


def test_iter_chunks_materializes_only_requested_slices():
    x, _ = _toy_data()
    rows = _LazyRows(x)

    chunks = list(iter_chunks(rows, batch_size=3))

    assert rows.full_materializations == 0
    assert [(key.start, key.stop, key.step) for key in rows.slices] == [
        (0, 3, None),
        (3, 6, None),
        (6, 8, None),
    ]
    assert [start for start, _ in chunks] == [0, 3, 6]
    assert [chunk.shape[0] for _, chunk in chunks] == [3, 3, 2]
    assert chunks[0][1] == pytest.approx(x[:3])


def test_disk_helpers_materialize_only_requested_slices(tmp_path):
    x, _ = _toy_data()
    rows = _LazyRows(x)
    estimator = _StreamingDummy()

    prediction_file = tmp_path / "predictions.dat"
    predictions = predict_large_dataset(
        estimator,
        rows,
        batch_size=3,
        output_file=str(prediction_file),
    )

    assert rows.full_materializations == 0
    assert prediction_file.exists()
    assert predictions.shape == (x.shape[0],)
    assert [(key.start, key.stop, key.step) for key in rows.slices] == [
        (0, 3, None),
        (3, 6, None),
        (6, 8, None),
    ]

    rows = _LazyRows(x)
    survival_file = tmp_path / "survival_curves.dat"
    times, survival = survival_curves_to_disk(
        estimator,
        rows,
        str(survival_file),
        batch_size=3,
    )

    assert rows.full_materializations == 0
    assert survival_file.exists()
    assert times.shape == (2,)
    assert survival.shape == (x.shape[0], 2)
    assert estimator.survival_shapes == [(3, 1), (3, 1), (2, 1)]
    assert [(key.start, key.stop, key.step) for key in rows.slices] == [
        (0, 3, None),
        (3, 6, None),
        (6, 8, None),
    ]


def test_survival_curves_to_disk_rejects_empty_input(tmp_path):
    estimator = _StreamingDummy()
    survival_file = tmp_path / "survival_curves.dat"

    with pytest.raises(ValueError, match="at least one row"):
        survival_curves_to_disk(
            estimator,
            np.empty((0, 1), dtype=np.float64),
            str(survival_file),
        )

    assert estimator.survival_shapes == []
    assert not survival_file.exists()


def test_survival_curves_to_disk_verbose_batches_are_numbered(tmp_path, capsys):
    x, _ = _toy_data()
    estimator = _StreamingDummy()
    survival_file = tmp_path / "survival_curves.dat"

    survival_curves_to_disk(estimator, x, str(survival_file), batch_size=3, verbose=True)

    assert capsys.readouterr().out.splitlines() == [
        "Processed batch 1/3 (samples 0-3)",
        "Processed batch 2/3 (samples 3-6)",
        "Processed batch 3/3 (samples 6-8)",
    ]


@pytest.mark.parametrize("batch_size", [0, -1, False, 1.5])
def test_streaming_helpers_validate_batch_size(tmp_path, batch_size):
    x, _ = _toy_data()
    estimator = _StreamingDummy()
    prediction_file = tmp_path / "predictions.dat"
    survival_file = tmp_path / "survival_curves.dat"

    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        list(iter_chunks(x, batch_size=batch_size))
    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        list(estimator.predict_batched(x, batch_size=batch_size))
    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        list(estimator.predict_survival_batched(x, batch_size=batch_size))
    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        estimator.predict_to_array(x, batch_size=batch_size)
    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        predict_large_dataset(
            estimator,
            x,
            batch_size=batch_size,
            output_file=str(prediction_file),
        )
    with pytest.raises((TypeError, ValueError), match="batch_size must"):
        survival_curves_to_disk(
            estimator,
            x,
            str(survival_file),
            batch_size=batch_size,
        )

    assert not prediction_file.exists()
    assert not survival_file.exists()


@pytest.mark.skipif(
    not HAS_TREE_BINDINGS,
    reason="tree survival estimators require the Rust extension to be built with the ml feature",
)
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
    with pytest.raises(ValueError, match=r"expected \(8,\)"):
        estimator.predict_to_array(x, out=np.empty((x.shape[0], 1), dtype=np.float64))


def test_sklearn_compat_fallback_without_sklearn(monkeypatch):
    module_path = Path(sklearn_compat.__file__)
    original_import = builtins.__import__

    def fake_import(name, globalns=None, localns=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("scikit-learn intentionally unavailable")
        return original_import(name, globalns, localns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in list(sys.modules):
        if name.startswith("survival._sklearn_"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("survival.sklearn_compat_no_sklearn", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._HAS_SKLEARN is False
    common = sys.modules["survival._sklearn_common"]
    checked = common.check_array([["1.0"], ["2.0"]], dtype=np.float64, ensure_2d=True)

    assert checked.dtype == np.float64
    assert checked.shape == (2, 1)

    with pytest.raises(ValueError, match="Expected 2D array"):
        common.check_array([1.0, 2.0], dtype=np.float64, ensure_2d=True)

    estimator = module.CoxPHEstimator()
    assert estimator.get_params() == {"n_iters": 20}
    estimator.set_params(n_iters=7)
    assert estimator.n_iters == 7

    with pytest.raises(ValueError, match="not fitted yet"):
        estimator.predict(np.array([[0.1]], dtype=np.float64))
