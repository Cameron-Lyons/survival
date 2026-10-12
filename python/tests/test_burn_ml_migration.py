"""Exercise the native neural model bindings after the Burn runtime migration."""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
ml = survival.ml
HAS_ML_BINDINGS = all(hasattr(core, name) for name in ("DeepSurv", "DeepSurvConfig", "deep_surv"))
pytestmark = pytest.mark.skipif(
    not HAS_ML_BINDINGS,
    reason="native neural model tests require the Rust extension built with the ml feature",
)

N_OBS = 4
N_FEATURES = 2
N_DURATIONS = 3
N_EPOCHS = 2
FEATURES = [0.1, 0.5, 0.3, 0.8, 0.7, 0.2, 0.9, 0.4]
TIME = [1.0, 2.0, 3.0, 4.0]
STATUS = [1, 0, 1, 1]
CAUSES = [1, 2, 0, 1]
MAX_SEQ_LEN = 2
HISTORY = [value / 16.0 for value in range(16)]
SEQ_LENGTHS = [2, 1, 2, 1]


def _training_options():
    return {
        "learning_rate": 0.001,
        "batch_size": N_OBS,
        "n_epochs": N_EPOCHS,
        "early_stopping_patience": None,
        "validation_fraction": 0.25,
        "seed": 17,
    }


def _assert_finite_shape(values, shape):
    array = np.asarray(values, dtype=float)
    assert array.shape == shape
    assert np.isfinite(array).all()
    return array


def _assert_probabilities(values, shape):
    array = _assert_finite_shape(values, shape)
    assert ((array >= 0.0) & (array <= 1.0)).all()
    return array


def _assert_loss_history(model):
    _assert_finite_shape(model.train_loss, (N_EPOCHS,))
    _assert_finite_shape(model.val_loss, (N_EPOCHS,))


def test_deep_surv_fit_and_predictions():
    config = ml.DeepSurvConfig(
        hidden_layers=[4],
        dropout_rate=0.0,
        l2_reg=0.0,
        **_training_options(),
    )
    model = ml.deep_surv(FEATURES, N_OBS, N_FEATURES, TIME, STATUS, config)

    _assert_loss_history(model)
    assert model.n_features == N_FEATURES
    assert model.hidden_layers == config.hidden_layers
    _assert_finite_shape(model.predict_risk(FEATURES, N_OBS), (N_OBS,))
    _assert_probabilities(
        model.predict_survival(FEATURES, N_OBS),
        (N_OBS, len(model.unique_times)),
    )


def test_deephit_competing_risk_fit_and_predictions():
    config = ml.DeepHitConfig(
        shared_layers=[4],
        cause_specific_layers=[4],
        num_durations=N_DURATIONS,
        num_risks=2,
        dropout_rate=0.0,
        weight_decay=0.0,
        use_batch_norm=False,
        **_training_options(),
    )
    model = ml.deephit(FEATURES, N_OBS, N_FEATURES, TIME, CAUSES, config)

    _assert_loss_history(model)
    assert model.n_features == N_FEATURES
    assert model.num_risks == config.num_risks
    assert model.num_durations == config.num_durations
    assert model.config.shared_layers == config.shared_layers
    pmf = _assert_probabilities(model.predict_pmf(FEATURES, N_OBS), (N_OBS, 2 * N_DURATIONS))
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
    for risk_idx in range(2):
        _assert_probabilities(model.predict_cif(FEATURES, N_OBS, risk_idx), (N_OBS, N_DURATIONS))
    _assert_probabilities(model.predict_survival(FEATURES, N_OBS), (N_OBS, N_DURATIONS))


@pytest.mark.parametrize("temporal_type", ["lstm", "gru", "attention", "lstm_attention"])
def test_dynamic_deephit_history_fit(temporal_type):
    config = ml.DynamicDeepHitConfig(
        temporal_type=ml.TemporalType(temporal_type),
        embedding_dim=4,
        num_temporal_layers=1,
        shared_hidden_sizes=[4],
        cause_hidden_sizes=[4],
        num_durations=N_DURATIONS,
        num_causes=2,
        dropout_rate=0.0,
        **_training_options(),
    )
    model = ml.dynamic_deephit(
        HISTORY, SEQ_LENGTHS, N_OBS, MAX_SEQ_LEN, N_FEATURES, TIME, CAUSES, config
    )

    _assert_loss_history(model)
    assert model.num_causes == config.num_causes
    assert model.num_durations == config.num_durations
    assert model.max_seq_len == MAX_SEQ_LEN
    _assert_finite_shape(model.duration_cuts, (N_DURATIONS + 1,))


def test_tracer_missing_history_fit_and_predictions():
    mask = [1.0] * len(HISTORY)
    mask[3] = 0.0
    for row, length in enumerate(SEQ_LENGTHS):
        for step in range(length, MAX_SEQ_LEN):
            start = (row * MAX_SEQ_LEN + step) * N_FEATURES
            mask[start : start + N_FEATURES] = [0.0] * N_FEATURES
    time_delta = [float((index // N_FEATURES) % MAX_SEQ_LEN) for index in range(len(HISTORY))]
    config = ml.TracerConfig(
        embedding_dim=4,
        num_factorized_layers=1,
        num_attention_heads=2,
        num_durations=N_DURATIONS,
        num_events=2,
        mlp_hidden_size=4,
        dropout_rate=0.0,
        weight_decay=0.0,
        **_training_options(),
    )
    model = ml.tracer(
        HISTORY,
        mask,
        time_delta,
        SEQ_LENGTHS,
        N_OBS,
        MAX_SEQ_LEN,
        N_FEATURES,
        TIME,
        CAUSES,
        config,
    )

    _assert_loss_history(model)
    assert model.n_features == N_FEATURES
    assert model.num_events == config.num_events
    assert model.num_durations == config.num_durations
    assert model.embedding_dim == config.embedding_dim
    prediction_args = (HISTORY, mask, time_delta, SEQ_LENGTHS, N_OBS, MAX_SEQ_LEN)
    _assert_probabilities(model.predict_survival(*prediction_args), (N_OBS, N_DURATIONS))
    for event_idx in range(2):
        _assert_probabilities(
            model.predict_hazard(*prediction_args, event_idx), (N_OBS, N_DURATIONS)
        )
        _assert_probabilities(model.predict_cif(*prediction_args, event_idx), (N_OBS, N_DURATIONS))


def test_survtrace_categorical_fit_and_predictions():
    categorical = [0, 1, 1, 0]
    config = ml.SurvTraceConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=8,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        num_durations=N_DURATIONS,
        num_events=2,
        vocab_size=2,
        weight_decay=0.0,
        **_training_options(),
    )
    model = ml.survtrace(categorical, FEATURES, N_OBS, 1, N_FEATURES, [2], TIME, CAUSES, config)

    _assert_loss_history(model)
    assert model.cat_cardinalities == [2]
    assert model.hidden_size == config.hidden_size
    assert model.num_layers == config.num_hidden_layers
    assert model.num_events == config.num_events
    assert model.num_durations == config.num_durations
    for event_idx in range(2):
        _assert_probabilities(
            model.predict_hazard(categorical, FEATURES, N_OBS, event_idx),
            (N_OBS, N_DURATIONS),
        )
        _assert_probabilities(
            model.predict_survival(categorical, FEATURES, N_OBS, event_idx),
            (N_OBS, N_DURATIONS),
        )
    _assert_probabilities(
        model.predict_cumulative_incidence(categorical, FEATURES, N_OBS),
        (N_OBS, 2, N_DURATIONS),
    )


def test_galee_history_fit():
    config = ml.GALEEConfig(
        health_dim=4,
        hidden_sizes=[4],
        num_attention_heads=2,
        num_durations=N_DURATIONS,
        dropout_rate=0.0,
        **_training_options(),
    )
    model = ml.galee(HISTORY, SEQ_LENGTHS, N_OBS, MAX_SEQ_LEN, N_FEATURES, TIME, STATUS, config)

    _assert_loss_history(model)
    assert model.health_dim == config.health_dim
    assert model.num_durations == config.num_durations
    assert model.max_seq_len == MAX_SEQ_LEN
    _assert_finite_shape(model.duration_cuts, (N_DURATIONS + 1,))


def test_contrastive_surv_fit():
    config = ml.ContrastiveSurvConfig(
        embedding_dim=4,
        projection_dim=2,
        encoder_hidden_sizes=[4],
        num_attention_heads=2,
        num_encoder_layers=1,
        dropout_rate=0.0,
        **_training_options(),
    )
    model = ml.contrastive_surv(FEATURES, N_OBS, N_FEATURES, TIME, STATUS, config)

    _assert_loss_history(model)
    assert model.embedding_dim == config.embedding_dim
    assert model.projection_dim == config.projection_dim
    assert model.duration_cuts is None


def test_cox_time_fit_and_predictions():
    config = ml.CoxTimeConfig(
        hidden_dims=[4],
        time_embedding_dim=2,
        dropout_rate=0.0,
        learning_rate=0.001,
        batch_size=N_OBS,
        n_epochs=N_EPOCHS,
        seed=17,
    )
    covariates = np.asarray(FEATURES).reshape(N_OBS, N_FEATURES).tolist()
    model = ml.fit_cox_time(covariates, TIME, STATUS, config)

    assert "n_features=2" in repr(model)
    assert "hidden_dims=[4]" in repr(model)
    hazards = _assert_finite_shape(model.predict_hazard(covariates, TIME), (N_OBS,))
    assert (hazards > 0.0).all()
    curves = _assert_probabilities(model.predict_survival(covariates, [0.0, 2.0, 4.0]), (N_OBS, 3))
    np.testing.assert_array_equal(curves[:, 0], 1.0)
