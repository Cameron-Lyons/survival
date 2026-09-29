from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _survival as _surv
from ._sklearn_common import (
    BaseEstimator,
    RegressorMixin,
    SurvivalScoreMixin,
    _check_prediction_input,
    _validate_survival_data,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _median_survival_times(
    times: NDArray[np.float64], survival: NDArray[np.float64]
) -> NDArray[np.float64]:
    """R's median rule: the first time at which the survival curve reaches 0.5 or below."""
    medians = np.empty(survival.shape[0], dtype=np.float64)
    for row_idx, curve in enumerate(survival):
        crossing = np.flatnonzero(curve <= 0.5)
        medians[row_idx] = times[crossing[0]] if crossing.size else np.nan
    return medians


class CoxPHEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Cox Proportional Hazards model (R's ``coxph``).

    Parameters
    ----------
    n_iters : int, default=20
        Maximum number of iterations for the Newton-Raphson optimization.
    ties : str, default="efron"
        Tie handling, "efron" or "breslow".

    Attributes
    ----------
    model_ : CoxPHFit
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

    def __init__(self, n_iters: int = 20, ties: str = "efron"):
        self.n_iters = n_iters
        self.ties = ties

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

        self.model_ = _surv.coxph_fit(
            time,
            status,
            X,
            method=self.ties,
            iter_max=self.n_iters,
        )

        self.coef_ = np.asarray(self.model_.coefficients, dtype=np.float64)
        self.event_times_ = np.sort(np.unique(time))
        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict risk scores (centred linear predictors, R's ``predict(type = "lp")``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores (higher = higher risk).
        """
        X = _check_prediction_input(self, X)
        return np.asarray(self.model_.predict("lp", newdata=X).fit, dtype=np.float64)

    def predict_survival_function(
        self, X: ArrayLike, times: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict survival function for samples (R's ``survfit(fit, newdata)``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        times : array-like of shape (n_times,), optional
            Time points at which to evaluate the survival function; defaults to the
            unique training times.

        Returns
        -------
        times : ndarray of shape (n_times,)
            Time points.
        survival : ndarray of shape (n_samples, n_times)
            Survival probabilities.
        """
        X = _check_prediction_input(self, X)

        evaluation_times = (
            np.asarray(times, dtype=np.float64)
            if times is not None
            else np.asarray(self.event_times_, dtype=np.float64)
        )
        survival = self.model_.predict_survival_at(evaluation_times, newdata=X).T
        np.clip(survival, 0.0, 1.0, out=survival)
        return evaluation_times, survival

    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict median survival time for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        median_times : ndarray of shape (n_samples,)
            Predicted median survival times (NaN if survival never drops to 0.5).
        """
        times, survival = self.predict_survival_function(X)
        return _median_survival_times(times, survival)
