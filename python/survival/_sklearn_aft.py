from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _survival as _surv
from ._sklearn_common import (
    BaseEstimator,
    RegressorMixin,
    _compute_concordance_index,
    _validate_survival_data,
    check_array,
    check_is_fitted,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_LOG_TIME_ERROR_DISTRIBUTIONS = {
    "weibull": "extreme",
    "exponential": "extreme",
    "rayleigh": "extreme",
    "lognormal": "gaussian",
    "loglogistic": "logistic",
}


class AFTEstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Accelerated Failure Time (AFT) model.

    Log-time distributions use log(T) = X @ beta + sigma * epsilon. Gaussian,
    logistic, extreme-value, and Student-t distributions instead model T directly
    and allow negative and zero responses.

    Parameters
    ----------
    distribution : str, default="weibull"
        Error distribution. One of:
        - "weibull": Weibull distribution (extreme value errors)
        - "lognormal": Log-normal distribution (Gaussian errors)
        - "loglogistic": Log-logistic distribution (logistic errors)
        - "exponential": Exponential distribution (special case of Weibull)
        - "rayleigh": Rayleigh distribution (special case of Weibull)
        - "gaussian": Gaussian distribution (for linear models)
        - "logistic": Logistic distribution (for linear models)
        - "extreme": Extreme-value distribution on the original response scale
        - "t": Student-t distribution with four degrees of freedom
        Native distribution aliases are also accepted.
    max_iter : int, default=200
        Maximum number of iterations for optimization.
    tol : float, default=1e-9
        Convergence tolerance.

    Attributes
    ----------
    model_ : SurvivalFit
        The underlying fitted AFT model.
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the model location.
    scale_ : float
        Estimated scale parameter (sigma).
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from survival.sklearn_compat import AFTEstimator
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = np.column_stack([np.random.exponential(10, 100), np.random.binomial(1, 0.7, 100)])
    >>> model = AFTEstimator(distribution="weibull")
    >>> model.fit(X, y)
    >>> predicted_times = model.predict(X)

    Notes
    -----
    For log-time distributions, coefficients describe acceleration factors:
    - Positive coefficients increase expected survival time
    - Negative coefficients decrease expected survival time
    - exp(coef) gives the multiplicative effect on survival time
    For distributions that model T directly, coefficients describe additive
    changes in the response.
    """

    def __init__(
        self,
        distribution: str = "weibull",
        max_iter: int = 200,
        tol: float = 1e-9,
    ):
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AFTEstimator":
        """Fit the AFT model using maximum likelihood estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : AFTEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]
        n = len(time)

        events = status == 1
        n_events = events.sum()

        if n_events < X.shape[1] + 1:
            raise ValueError(
                f"Not enough events ({n_events}) to fit model with {X.shape[1]} features"
            )

        X_with_intercept = np.column_stack([np.ones(n), X])

        self.model_ = _surv.survreg(
            time=time.tolist(),
            status=status.tolist(),
            covariates=X_with_intercept.tolist(),
            distribution=self.distribution,
            max_iter=self.max_iter,
            eps=self.tol,
        )

        location = np.array(self.model_.location_coefficients)
        self.intercept_ = location[0]
        self.coef_ = location[1:]
        self.scale_ = self.model_.scale
        self.converged_ = self.model_.convergence_flag == 0

        self.is_fitted_ = True
        return self

    def _prediction_linear_predictor(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        return self.intercept_ + X @ self.coef_

    def _inverse_response_transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model_.distribution in _LOG_TIME_ERROR_DISTRIBUTIONS:
            return np.exp(values)
        return values

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the fitted location on the response scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        survival_times : ndarray of shape (n_samples,)
            The linear predictor for distributions that model T directly, or
            its exponential for log-time distributions. This matches
            ``survreg`` response prediction; use ``predict_median`` for medians.
        """
        return self._inverse_response_transform(self._prediction_linear_predictor(X))

    def predict_median(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict median survival time for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        median_times : ndarray of shape (n_samples,)
            Predicted median survival times.
        """
        return self.predict_quantile(X, q=0.5)

    def predict_quantile(self, X: ArrayLike, q: float = 0.5) -> NDArray[np.float64]:
        """Predict survival time quantile for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        q : float, default=0.5
            Quantile to predict (0 < q < 1). Default is median (0.5).

        Returns
        -------
        quantile_times : ndarray of shape (n_samples,)
            Predicted survival times at the given quantile.
        """
        linear_pred = self._prediction_linear_predictor(X)

        if not np.isfinite(q) or not 0 < q < 1:
            raise ValueError("q must be between 0 and 1")

        distribution = self.model_.distribution
        error_distribution = _LOG_TIME_ERROR_DISTRIBUTIONS.get(distribution, distribution)
        parameters = self.model_.distribution_parameters
        quantile = _surv.survreg_distribution(
            values=[q],
            mean=[0.0],
            scale=[1.0],
            distribution=error_distribution,
            kind="quantile",
            parms=parameters[0] if parameters else None,
        )[0]
        return self._inverse_response_transform(linear_pred + self.scale_ * quantile)

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
        predicted_times = self.predict(X)
        return _compute_concordance_index(time, status, -predicted_times)

    @property
    def acceleration_factors(self) -> NDArray[np.float64]:
        """Return exponentiated coefficients.

        These describe multiplicative time effects for log-time distributions.

        Returns
        -------
        af : ndarray of shape (n_features,)
            exp(coef_) for every distribution. For log-time distributions,
            values > 1 increase survival time and values < 1 decrease it.
            Identity-time coefficients instead describe additive effects.
        """
        check_is_fitted(self)
        return np.exp(self.coef_)
