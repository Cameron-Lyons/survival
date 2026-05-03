# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from survival import _survival as _surv

from ._sklearn_common import (
    BaseEstimator,
    RegressorMixin,
    SurvivalScoreMixin,
    _validate_survival_data,
    check_array,
    check_is_fitted,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class CoxPHEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Cox Proportional Hazards model.

    Parameters
    ----------
    n_iters : int, default=20
        Maximum number of iterations for the Newton-Raphson optimization.

    Attributes
    ----------
    model_ : CoxPHModel
        The underlying fitted Cox model.
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from survival.sklearn_compat import CoxPHEstimator
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = np.column_stack([np.random.exponential(10, 100), np.random.binomial(1, 0.7, 100)])
    >>> model = CoxPHEstimator()
    >>> model.fit(X, y)
    >>> risk_scores = model.predict(X)
    """

    def __init__(self, n_iters: int = 20):
        self.n_iters = n_iters

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CoxPHEstimator":
        """Fit the Cox PH model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : CoxPHEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]

        covariates = X.tolist()
        self.model_ = _surv.CoxPHModel.new_with_data(covariates, time.tolist(), status.tolist())
        self.model_.fit(self.n_iters)

        self.coef_ = np.array(self.model_.coefficients)
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

        return np.array(self.model_.predict(X.tolist()))

    def predict_survival_function(
        self, X: ArrayLike, times: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict survival function for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        times : array-like of shape (n_times,), optional
            Time points at which to evaluate the survival function.

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

        times_list = times.tolist() if times is not None else None
        t, surv = self.model_.survival_curve(X.tolist(), times_list)
        return np.array(t), np.array(surv)

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

        result = self.model_.predicted_survival_time(X.tolist(), 0.5)
        return np.array([t if t is not None else np.nan for t in result])
