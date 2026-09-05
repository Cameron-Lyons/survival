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
