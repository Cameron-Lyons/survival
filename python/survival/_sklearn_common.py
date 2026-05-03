# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from survival import _survival as _surv

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

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
                for k in self.__init__.__code__.co_varnames[1 : self.__init__.__code__.co_argcount]
            }

        def set_params(self, **params) -> "BaseEstimator":
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class RegressorMixin:
        pass

    def check_array(X, **kwargs):
        return np.asarray(X)

    def check_is_fitted(estimator, attributes=None):
        if not hasattr(estimator, "is_fitted_") or not estimator.is_fitted_:
            raise ValueError(f"{type(estimator).__name__} is not fitted yet.")


def _validate_survival_data(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    X = check_array(X, dtype=np.float64, ensure_2d=True)
    y = np.asarray(y)

    if y.ndim == 1:
        raise ValueError("y must be a 2D array with columns [time, status]")
    if y.shape[1] != 2:
        raise ValueError("y must have exactly 2 columns: [time, status]")

    time = y[:, 0].astype(np.float64)
    status = y[:, 1].astype(np.int32)

    if X.shape[0] != len(time):
        raise ValueError(f"X has {X.shape[0]} samples, but y has {len(time)} samples")

    return X, time, status


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
        risk_scores = self.predict(X)
        return _compute_concordance_index(time, status, risk_scores)
