from __future__ import annotations

from numbers import Number
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

    def check_is_fitted(estimator: Any, attributes: Any = None) -> None: ...
else:
    try:
        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.utils.validation import check_is_fitted

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

        def check_is_fitted(estimator, attributes=None):
            if not hasattr(estimator, "is_fitted_") or not estimator.is_fitted_:
                raise ValueError(f"{type(estimator).__name__} is not fitted yet.")


def check_array(
    X: ArrayLike, *, dtype: Any = np.float64, ensure_2d: bool = True
) -> NDArray[np.float64]:
    """Validate dense features consistently with or without scikit-learn."""
    if np.ma.is_masked(X):
        raise ValueError("X must not contain masked feature values")
    array = np.asarray(X)
    if ensure_2d and array.ndim != 2:
        shape = "scalar" if array.ndim == 0 else f"{array.ndim}D"
        raise ValueError(f"Expected 2D array, got {shape} array instead")
    if array.ndim > 0 and array.shape[0] == 0:
        raise ValueError("X must contain at least one sample")

    # NumPy complex scalars in object arrays can also lose their imaginary part
    # when cast to float64. Check before conversion to avoid that data loss.
    if np.iscomplexobj(array):
        raise ValueError("X must contain only real feature values")
    if array.dtype.kind == "O":
        simple_types = (float, int, bool, str, bytes)
        complex_types = (complex, np.complexfloating)
        temporal_types = (np.datetime64, np.timedelta64)
        for value in array.flat:
            if type(value) in simple_types:
                continue
            if isinstance(value, complex_types):
                raise ValueError("X must contain only real feature values")
            if isinstance(value, temporal_types) and np.isnat(value):
                raise ValueError("X must contain only finite feature values")
            if np.ma.is_masked(value):
                raise ValueError("X must not contain masked feature values")
    if array.dtype.kind in "mM" and np.isnat(array).any():
        raise ValueError("X must contain only finite feature values")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(dtype, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("X must contain only finite real feature values") from error
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite feature values")
    return array


def _validate_survival_data(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    X_array = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    if np.ma.is_masked(y):
        raise ValueError("y must not contain masked time or status values")
    y_array = np.asarray(y)

    if y_array.ndim != 2:
        raise ValueError("y must be a 2D array with columns [time, status]")
    if y_array.shape[1] != 2:
        raise ValueError("y must have exactly 2 columns: [time, status]")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"X has {X_array.shape[0]} samples, but y has {y_array.shape[0]} samples")

    real_values = y_array.dtype.kind in "biuf"
    if y_array.dtype.kind == "O":
        # Keep ordinary object arrays fast while admitting exact numeric types
        # such as Decimal and Fraction without first rounding them to float64.
        basic_real_types = (int, float, bool, np.bool_)
        real_values = all(
            type(value) in basic_real_types
            or (
                isinstance(value, Number)
                and not isinstance(value, complex | np.complexfloating | np.timedelta64)
            )
            for value in y_array.flat
        )
    if not real_values:
        raise ValueError("y must contain only real numeric values")

    # Check the original values: both integer casts and float64 casts can turn
    # invalid event indicators into an exact 0 or 1.
    raw_status = y_array[:, 1]
    try:
        binary_status = ((raw_status == 0) | (raw_status == 1)).all()
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ValueError("status must contain only 0 or 1") from error
    if not binary_status:
        raise ValueError("status must contain only 0 or 1")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            time = y_array[:, 0].astype(np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("time must contain only finite real values") from error
    if not np.isfinite(time).all():
        raise ValueError("time must contain only finite real values")
    status = raw_status.astype(np.int32)

    return X_array, time, status


def _compute_concordance_index(
    time: NDArray[np.float64],
    status: NDArray[np.int32],
    risk_scores: NDArray[np.float64],
) -> float:
    """Compute Harrell's concordance index (C-index) in Rust."""
    return float(
        _surv.concordance_index(
            np.asarray(time, dtype=np.float64).tolist(),
            np.asarray(status, dtype=np.int32).tolist(),
            np.asarray(risk_scores, dtype=np.float64).tolist(),
        )
    )


class SurvivalScoreMixin:
    """Mixin providing concordance index scoring for survival models."""

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
        risk_scores = cast("_Predictor", self).predict(X)
        return _compute_concordance_index(time, status, risk_scores)


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


def _check_prediction_input(estimator: Any, X: ArrayLike) -> NDArray[np.float64]:
    """Validate ``X`` for prediction: fitted estimator, 2-D floats, matching feature count."""
    check_is_fitted(estimator)
    X_array = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    if X_array.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_array.shape[1]} features, but model expects {estimator.n_features_in_}"
        )
    return X_array
