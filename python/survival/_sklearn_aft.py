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
    check_is_fitted,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class AFTEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Accelerated Failure Time (AFT) model.

    AFT models assume that covariates act multiplicatively on the survival time,
    i.e., log(T) = X @ beta + sigma * epsilon, where epsilon follows a specified
    error distribution. The fit is R's ``survreg``.

    Parameters
    ----------
    distribution : str, default="weibull"
        Error distribution, one of R's ``survreg.distributions`` names (partial
        matching as in R):
        - "weibull": Weibull distribution (extreme value errors on log time)
        - "exponential": Exponential distribution (Weibull with scale fixed at 1)
        - "rayleigh": Rayleigh distribution (Weibull with scale fixed at 0.5)
        - "lognormal" / "loggaussian": Log-normal distribution (Gaussian errors on log time)
        - "loglogistic": Log-logistic distribution (logistic errors on log time)
        - "gaussian": Gaussian distribution on the untransformed time
        - "logistic": Logistic distribution on the untransformed time
        - "extreme": extreme value distribution on the untransformed time
    max_iter : int, default=30
        Maximum number of Newton-Raphson iterations (R's ``iter.max``).
    tol : float, default=1e-9
        Relative convergence tolerance (R's ``rel.tolerance``).

    Attributes
    ----------
    model_ : SurvregFit
        The underlying fitted AFT model.
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (acceleration factors on the log scale for the
        log-transformed distributions).
    intercept_ : float
        Estimated intercept.
    scale_ : float
        Estimated scale parameter (sigma).
    converged_ : bool
        Whether the Newton-Raphson iteration converged.
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
    For the log-transformed distributions the coefficients are acceleration factors:
    - Positive coefficients increase expected survival time
    - Negative coefficients decrease expected survival time
    - exp(coef) gives the multiplicative effect on survival time
    For "gaussian", "logistic" and "extreme" the model is linear in the raw time and
    predictions are on the time scale directly (no exponentiation).
    """

    def __init__(
        self,
        distribution: str = "weibull",
        max_iter: int = 30,
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

        self.model_ = _surv.survreg_fit(
            _surv.SurvregData(time.tolist(), status.tolist(), X_with_intercept.tolist()),
            _surv.SurvregDistribution(self.distribution),
            control=_surv.SurvregControl(iter_max=self.max_iter, rel_tolerance=self.tol),
        )

        location = np.asarray(self.model_.coefficients[: self.n_features_in_ + 1], dtype=np.float64)
        self.intercept_ = float(location[0])
        self.coef_ = location[1:]
        self.scale_ = float(self.model_.scale[0])
        self.converged_ = bool(self.model_.converged)

        self.is_fitted_ = True
        return self

    def _design(self, X: ArrayLike) -> list[list[float]]:
        X_array = _check_prediction_input(self, X)
        return np.column_stack([np.ones(X_array.shape[0]), X_array]).tolist()

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict the expected response (R's ``predict(type = "response")``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        survival_times : ndarray of shape (n_samples,)
            Predicted survival times: exp(linear predictor) for the log-transformed
            distributions, the linear predictor itself for the others.
        """
        design = self._design(X)
        prediction = self.model_.predict(newdata=design, predict_type="response")
        return np.asarray(prediction.fit, dtype=np.float64).reshape(-1)

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
        return self.predict_quantile(X, 0.5)

    def predict_quantile(self, X: ArrayLike, q: float = 0.5) -> NDArray[np.float64]:
        """Predict survival time quantile for samples (R's ``predict(type = "quantile")``).

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
        design = self._design(X)

        if not np.isfinite(q) or not 0 < q < 1:
            raise ValueError("q must be between 0 and 1")

        prediction = self.model_.predict(newdata=design, predict_type="quantile", p=[float(q)])
        return np.asarray(prediction.fit, dtype=np.float64).reshape(-1)

    def _risk_scores(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.predict(X)

    @property
    def acceleration_factors(self) -> NDArray[np.float64]:
        """Return acceleration factors (exp of coefficients).

        Returns
        -------
        af : ndarray of shape (n_features,)
            Acceleration factors. Values > 1 increase survival time,
            values < 1 decrease survival time.
        """
        check_is_fitted(self)
        return np.exp(self.coef_)
