# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _survival as _surv
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

_EXP_CLAMP_MIN = -100.0
_EXP_CLAMP_MAX = 100.0


def _step_values_at(
    step_times: NDArray[np.float64],
    step_values: NDArray[np.float64],
    evaluation_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a cumulative step function at requested times."""
    if step_times.size == 0:
        return np.zeros_like(evaluation_times, dtype=np.float64)
    positions = np.searchsorted(step_times, evaluation_times, side="right") - 1
    values = np.zeros_like(evaluation_times, dtype=np.float64)
    valid = positions >= 0
    values[valid] = step_values[positions[valid]]
    return values


def _median_survival_times(
    times: NDArray[np.float64], survival: NDArray[np.float64]
) -> NDArray[np.float64]:
    medians = np.empty(survival.shape[0], dtype=np.float64)
    target = 0.5
    for row_idx, curve in enumerate(survival):
        crossing = np.flatnonzero(curve <= target)
        if crossing.size == 0:
            medians[row_idx] = np.nan
            continue
        idx = int(crossing[0])
        if idx == 0:
            medians[row_idx] = times[0]
            continue
        previous_survival = curve[idx - 1]
        current_survival = curve[idx]
        previous_time = times[idx - 1]
        current_time = times[idx]
        if previous_survival == current_survival:
            medians[row_idx] = current_time
        else:
            fraction = (previous_survival - target) / (previous_survival - current_survival)
            medians[row_idx] = previous_time + fraction * (current_time - previous_time)
    return medians


class CoxPHEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Cox Proportional Hazards model.

    Parameters
    ----------
    n_iters : int, default=20
        Maximum number of iterations for the Newton-Raphson optimization.

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
        self.model_ = _surv.coxph_fit(
            time.tolist(),
            status.tolist(),
            covariates,
            None,
            None,
            None,
            None,
            self.n_iters,
            None,
            None,
            "breslow",
            None,
        )

        self.coef_ = np.asarray(self.model_.coefficients[0], dtype=np.float64)
        self.event_times_ = np.sort(np.unique(time))
        baseline_times, baseline_hazard = self.model_.basehaz(True)
        self._baseline_times_ = np.asarray(baseline_times, dtype=np.float64)
        self._baseline_hazard_ = np.asarray(baseline_hazard, dtype=np.float64)
        self._center_ = float(np.mean(np.asarray(self.model_.linear_predictors, dtype=np.float64)))
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

        evaluation_times = (
            np.asarray(times, dtype=np.float64)
            if times is not None
            else np.asarray(self.event_times_, dtype=np.float64)
        )
        baseline_hazard = _step_values_at(
            self._baseline_times_, self._baseline_hazard_, evaluation_times
        )
        linear_predictors = np.asarray(self.model_.predict(X.tolist()), dtype=np.float64)
        risk_multipliers = np.exp(
            np.clip(linear_predictors - self._center_, _EXP_CLAMP_MIN, _EXP_CLAMP_MAX)
        )
        survival = np.exp(-np.outer(risk_multipliers, baseline_hazard))
        return evaluation_times, np.clip(survival, 0.0, 1.0)

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

        times, survival = self.predict_survival_function(X)
        return _median_survival_times(times, survival)
