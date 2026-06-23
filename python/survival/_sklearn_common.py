# ruff: noqa: N803, N806, UP037
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
