# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from survival import _survival as _surv

from ._sklearn_common import (
    BaseEstimator,
    RegressorMixin,
    _compute_concordance_index,
    _validate_survival_data,
    check_array,
    check_is_fitted,
)
from ._sklearn_streaming import StreamingMixin

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class DeepSurvEstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible DeepSurv model.

    DeepSurv is a deep feedforward neural network for survival analysis
    using Cox partial likelihood loss.

    Parameters
    ----------
    hidden_layers : list of int, default=[64, 32]
        Number of neurons in each hidden layer.
    activation : str, default="selu"
        Activation function. One of "relu", "selu", or "tanh".
    dropout_rate : float, default=0.2
        Dropout rate applied after each hidden layer.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
    batch_size : int, default=256
        Mini-batch size for training.
    n_epochs : int, default=100
        Number of training epochs.
    l2_reg : float, default=0.0001
        L2 regularization (weight decay) coefficient.
    seed : int or None, default=None
        Random seed for reproducibility.
    early_stopping_patience : int or None, default=10
        Number of epochs without improvement before early stopping.
        Set to None to disable early stopping.
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation.

    Attributes
    ----------
    model_ : DeepSurv
        The underlying fitted model.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "selu",
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 100,
        l2_reg: float = 0.0001,
        seed: int | None = None,
        early_stopping_patience: int | None = 10,
        validation_fraction: float = 0.1,
    ):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DeepSurvEstimator":
        """Fit the DeepSurv model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : DeepSurvEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]
        n_obs = X.shape[0]

        activation = _surv.Activation(self.activation)
        config = _surv.DeepSurvConfig(
            hidden_layers=self.hidden_layers,
            activation=activation,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            l2_reg=self.l2_reg,
            seed=self.seed,
            early_stopping_patience=self.early_stopping_patience,
            validation_fraction=self.validation_fraction,
        )

        x_flat = X.flatten().tolist()
        self.model_ = _surv.DeepSurv.fit(
            x_flat, n_obs, self.n_features_in_, time.tolist(), status.tolist(), config
        )

        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict risk scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores (higher = higher risk).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        x_flat = X.flatten().tolist()
        return np.array(self.model_.predict_risk(x_flat, X.shape[0]))

    def predict_survival_function(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict survival function for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        times : ndarray of shape (n_times,)
            Time points.
        survival : ndarray of shape (n_samples, n_times)
            Survival probabilities.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        x_flat = X.flatten().tolist()
        survival = self.model_.predict_survival(x_flat, X.shape[0])
        return np.array(self.model_.unique_times), np.array(survival)

    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict median survival time for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        median_times : ndarray of shape (n_samples,)
            Predicted median survival times (NaN if survival never drops below 0.5).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        x_flat = X.flatten().tolist()
        result = self.model_.predict_median_survival_time(x_flat, X.shape[0])
        return np.array([t if t is not None else np.nan for t in result])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the concordance index on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples, 2)
            True target values.

        Returns
        -------
        score : float
            Concordance index (C-index), between 0 and 1.
        """
        check_is_fitted(self)
        X, time, status = _validate_survival_data(X, y)
        risk_scores = self.predict(X)
        return _compute_concordance_index(time, status, risk_scores)

    @property
    def train_loss(self) -> NDArray[np.float64]:
        """Training loss history."""
        check_is_fitted(self)
        return np.array(self.model_.train_loss)

    @property
    def val_loss(self) -> NDArray[np.float64]:
        """Validation loss history."""
        check_is_fitted(self)
        return np.array(self.model_.val_loss)


class StreamingDeepSurvEstimator(DeepSurvEstimator, StreamingMixin):
    """DeepSurv Estimator with streaming/batched prediction support.

    This class extends DeepSurvEstimator with methods for processing large
    datasets that don't fit in memory.

    See DeepSurvEstimator for full documentation.
    """

    pass
