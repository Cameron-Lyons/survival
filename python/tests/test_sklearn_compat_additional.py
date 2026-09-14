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
    """Eight rows: t = 1:8, s = c(1,1,0,1,1,1,0,1), x = c(.5,.2,.9,.1,.7,.3,.8,.4) in R."""
    x = np.array([[0.5], [0.2], [0.9], [0.1], [0.7], [0.3], [0.8], [0.4]], dtype=np.float64)
    y = np.column_stack(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 1, 0, 1, 1, 1, 0, 1],
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


def test_score_uses_rust_concordancefit(monkeypatch):
    common = importlib.import_module("survival._sklearn_common")
    calls = []
    real_concordancefit = common._surv.concordancefit

    def spy(survival_data, x, **kwargs):
        calls.append((survival_data.time, survival_data.status, x.values, kwargs))
        return real_concordancefit(survival_data, x, **kwargs)

    monkeypatch.setattr(common._surv, "concordancefit", spy)

    score = common._compute_concordance_index(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.array([1, 1, 0, 1], dtype=np.int32),
        np.array([0.7, 0.2, 0.5, 0.1], dtype=np.float64),
    )

    # concordance(Surv(time, status) ~ risk, reverse = TRUE): 4 concordant of 5 comparable pairs
    assert score == pytest.approx(0.8)
    assert len(calls) == 1
    time, status, risk, kwargs = calls[0]
    assert time == [1.0, 2.0, 3.0, 4.0]
    assert status == [1, 1, 0, 1]
    assert risk == [0.7, 0.2, 0.5, 0.1]
    assert kwargs == {"reverse": True, "std_err": False}


def test_concordance_index_counts_tied_risk_scores_as_half():
    common = importlib.import_module("survival._sklearn_common")
    score = common._compute_concordance_index(
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.int32),
        np.array([0.5, 0.5, 0.1], dtype=np.float64),
    )
    # pairs (1,2) tied in x (1/2), (1,3) concordant, (2,3) concordant
    assert score == pytest.approx(2.5 / 3.0)


def test_coxph_estimator_matches_r_coxph():
    x, y = _toy_data()
    estimator = CoxPHEstimator(n_iters=10)
    estimator.fit(x, y)

    assert isinstance(estimator.model_, core.CoxPHFit)
    # coxph(Surv(t, s) ~ x, d)
    assert estimator.coef_ == pytest.approx([-2.5113057727233947])

    risk = estimator.predict(x)
    times, survival = estimator.predict_survival_function(x)
    median = estimator.predict_median_survival_time(x)

    # predict(fit, type = "lp") on the training rows
    assert risk == pytest.approx(
        [
            -0.031391322159042501,
            0.72200040965797585,
            -1.0359136312484003,
            0.9731309869303153,
            -0.5336524767037214,
            0.47086983238563651,
            -0.78478305397606118,
            0.21973925511329684,
        ]
    )
    assert times.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert survival.shape == (x.shape[0], x.shape[0])
    assert median.shape == (x.shape[0],)
    # R concordance of the fitted model
    assert estimator.score(x, y) == pytest.approx(0.68181818181818188)

    breslow = CoxPHEstimator(ties="breslow").fit(x, y)
    assert breslow.model_.method == core.TieMethod.Breslow
    assert breslow.coef_ == pytest.approx([-2.5113057727233947])


def test_coxph_estimator_survival_curves_match_r_survfit():
    x, y = _toy_data()
    estimator = CoxPHEstimator(n_iters=10)
    estimator.fit(x, y)
    new_x = np.array([[0.25], [0.75]], dtype=np.float64)

    # R linear predictor on new data
    assert estimator.predict(new_x) == pytest.approx([0.59643512102180618, -0.65921776533989118])

    times, survival = estimator.predict_survival_function(new_x)
    # survfit(fit, newdata = data.frame(x = c(.25, .75)))$surv, one column per new row
    assert times.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert survival[0] == pytest.approx(
        [
            0.83272324379982243,
            0.67981818400153027,
            0.67981818400153027,
            0.51493304919211669,
            0.3228767511518843,
            0.18635202048747801,
            0.18635202048747801,
            0.043387721655328455,
        ]
    )
    assert survival[1] == pytest.approx(
        [
            0.94918629237163887,
            0.89588118129769456,
            0.89588118129769456,
            0.82771483162123671,
            0.7246521605255537,
            0.61962030183299699,
            0.61962030183299699,
            0.40907197828345665,
        ]
    )
    # summary(survfit(...))$table[, "median"]
    assert estimator.predict_median_survival_time(new_x).tolist() == pytest.approx([5.0, 8.0])

    custom_times = np.array([0.5, 2.0, 3.5, 100.0], dtype=np.float64)
    returned_times, at_custom = estimator.predict_survival_function(new_x, times=custom_times)
    assert returned_times.tolist() == pytest.approx(custom_times.tolist())
    assert at_custom[0] == pytest.approx(
        [1.0, 0.67981818400153027, 0.67981818400153027, 0.043387721655328455]
    )

    with pytest.raises(ValueError, match="expects 1"):
        estimator.predict(np.array([[0.1, 0.2]], dtype=np.float64))
    with pytest.raises(ValueError, match="expects 1"):
        estimator.predict_survival_function(np.array([[0.1, 0.2]], dtype=np.float64))


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
    assert estimator.get_params() == {"n_iters": 20, "ties": "efron"}
    estimator.set_params(n_iters=7)
    assert estimator.n_iters == 7

    with pytest.raises(ValueError, match="not fitted yet"):
        estimator.predict(np.array([[0.1]], dtype=np.float64))
