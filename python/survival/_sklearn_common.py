from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from . import _survival as _surv

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    class BaseEstimator:
        def get_params(self, deep: bool = True) -> dict[str, Any]: ...
        def set_params(self, **params: Any) -> BaseEstimator: ...

    class RegressorMixin:
        pass

    class _Predictor(Protocol):
        def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...

    class _FlatModel(Protocol):
        """The Rust ML models take a flattened row-major design plus its row count."""

        unique_times: list[float]

        def predict_risk(self, x: list[float], n_obs: int) -> list[float]: ...
        def predict_survival(self, x: list[float], n_obs: int) -> list[list[float]]: ...
        def predict_median_survival_time(
            self, x: list[float], n_obs: int
        ) -> list[float | None]: ...

    def check_array(X: ArrayLike, **kwargs: Any) -> NDArray[np.float64]: ...
    def check_is_fitted(estimator: Any, attributes: Any = None) -> None: ...
else:
    try:
        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.utils.validation import check_array, check_is_fitted

        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False

        class BaseEstimator:
            def get_params(self, deep: bool = True) -> dict:
                return {
                    k: getattr(self, k)
                    for k in self.__init__.__code__.co_varnames[
                        1 : self.__init__.__code__.co_argcount
                    ]
                }

            def set_params(self, **params) -> "BaseEstimator":
                for key, value in params.items():
                    setattr(self, key, value)
                return self

        class RegressorMixin:
            pass

        def check_array(X, **kwargs):
            array = np.asarray(X, dtype=kwargs.get("dtype"))
            if kwargs.get("ensure_2d", True) and array.ndim != 2:
                shape = "scalar" if array.ndim == 0 else f"{array.ndim}D"
                raise ValueError(f"Expected 2D array, got {shape} array instead")
            return array

        def check_is_fitted(estimator, attributes=None):
            if not hasattr(estimator, "is_fitted_") or not estimator.is_fitted_:
                raise ValueError(f"{type(estimator).__name__} is not fitted yet.")


def _validate_survival_data(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    X_array = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    y_array = np.asarray(y)

    if y_array.ndim == 1:
        raise ValueError("y must be a 2D array with columns [time, status]")
    if y_array.shape[1] != 2:
        raise ValueError("y must have exactly 2 columns: [time, status]")

    time = y_array[:, 0].astype(np.float64)
    status = y_array[:, 1].astype(np.int32)

    if X_array.shape[0] != len(time):
        raise ValueError(f"X has {X_array.shape[0]} samples, but y has {len(time)} samples")

    return X_array, time, status


def _check_prediction_input(estimator: Any, X: ArrayLike) -> NDArray[np.float64]:
    """Validate ``X`` for prediction: fitted estimator, 2-D floats, matching feature count."""
    check_is_fitted(estimator)
    X_array = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    if X_array.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_array.shape[1]} features, but model expects {estimator.n_features_in_}"
        )
    return X_array


def _compute_concordance_index(
    time: NDArray[np.float64],
    status: NDArray[np.int32],
    risk_scores: NDArray[np.float64],
) -> float:
    """Harrell's C-index of ``risk_scores`` (higher = higher risk), via R's ``concordancefit``.

    ``reverse=True`` is R's convention for a risk score: a larger score should go with the
    shorter survival time; ties in the score count one half.
    """
    time = np.asarray(time, dtype=np.float64)
    status = np.asarray(status, dtype=np.int32)
    risk = np.asarray(risk_scores, dtype=np.float64)
    fit = _surv.concordancefit(
        _surv.SurvivalData(time.tolist(), status.tolist()),
        _surv.CovariateMatrix(risk.tolist(), len(risk), 1),
        reverse=True,
        std_err=False,
    )
    return float(fit.concordance[0])


class SurvivalScoreMixin:
    """Mixin providing concordance index scoring for survival models.

    ``predict`` must return a risk score (higher = higher risk); estimators whose ``predict``
    returns a survival time override :meth:`_risk_scores` to negate it.
    """

    def _risk_scores(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(cast("_Predictor", self).predict(X), dtype=np.float64)

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
        return _compute_concordance_index(time, status, self._risk_scores(X))


class FlatModelPredictMixin:
    """Prediction methods shared by the estimators wrapping a Rust ML model.

    The wrapped ``model_`` (DeepSurv, gradient boosting, survival forest) takes the design
    matrix flattened row-major together with its row count and exposes ``predict_risk``,
    ``predict_survival``, ``predict_median_survival_time`` and ``unique_times``.
    """

    model_: _FlatModel
    n_features_in_: int

    def _flat_input(self, X: ArrayLike) -> tuple[list[float], int]:
        X_array = _check_prediction_input(self, X)
        return X_array.flatten().tolist(), X_array.shape[0]

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
        x_flat, n_obs = self._flat_input(X)
        return np.array(self.model_.predict_risk(x_flat, n_obs))

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
        x_flat, n_obs = self._flat_input(X)
        survival = self.model_.predict_survival(x_flat, n_obs)
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
        x_flat, n_obs = self._flat_input(X)
        result = self.model_.predict_median_survival_time(x_flat, n_obs)
        return np.array([t if t is not None else np.nan for t in result], dtype=np.float64)
