# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from collections.abc import Iterator
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
    """Compute concordance index (C-index) for survival predictions."""
    n = len(time)
    concordant = 0.0
    comparable = 0.0

    for i in range(n):
        if status[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j]:
                comparable += 1.0
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    return concordant / comparable if comparable > 0 else 0.5


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

        result = self.model_.predicted_survival_time(X.tolist(), 0.5)
        return np.array([t if t is not None else np.nan for t in result])


class GradientBoostSurvivalEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Gradient Boosting Survival model.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression trees.
    min_samples_split : int, default=10
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=5
        Minimum number of samples required at each leaf node.
    subsample : float, default=1.0
        Fraction of samples used for fitting individual trees.
    max_features : int or None, default=None
        Number of features to consider for splits.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : GradientBoostSurvival
        The underlying fitted model.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        max_features: int | None = None,
        seed: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientBoostSurvivalEstimator":
        """Fit the gradient boosting survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : GradientBoostSurvivalEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]
        n_obs = X.shape[0]

        config = _surv.GradientBoostSurvivalConfig(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            seed=self.seed,
        )

        x_flat = X.flatten().tolist()
        self.model_ = _surv.GradientBoostSurvival.fit(
            x_flat, n_obs, self.n_features_in_, time.tolist(), status.tolist(), config
        )

        self.feature_importances_ = np.array(self.model_.feature_importance)
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

        x_flat = X.flatten().tolist()
        return np.array(self.model_.predict_risk(x_flat, X.shape[0]))

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        survival = self.model_.predict_survival(x_flat, X.shape[0])
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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        result = self.model_.predict_median_survival_time(x_flat, X.shape[0])
        return np.array([t if t is not None else np.nan for t in result])


class SurvivalForestEstimator(SurvivalScoreMixin, BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Random Survival Forest model.

    Parameters
    ----------
    n_trees : int, default=500
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of trees.
    min_node_size : int, default=15
        Minimum number of samples at each leaf node.
    mtry : int or None, default=None
        Number of features to consider at each split (default: sqrt(n_features)).
    sample_fraction : float, default=0.632
        Fraction of samples used for each tree.
    seed : int or None, default=None
        Random seed for reproducibility.
    oob_error : bool, default=True
        Whether to compute out-of-bag error.

    Attributes
    ----------
    model_ : SurvivalForest
        The underlying fitted model.
    variable_importance_ : ndarray of shape (n_features,)
        Variable importances.
    oob_error_ : float or None
        Out-of-bag error (if computed).
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        n_trees: int = 500,
        max_depth: int | None = None,
        min_node_size: int = 15,
        mtry: int | None = None,
        sample_fraction: float = 0.632,
        seed: int | None = None,
        oob_error: bool = True,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.mtry = mtry
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.oob_error = oob_error

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SurvivalForestEstimator":
        """Fit the random survival forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : SurvivalForestEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]
        n_obs = X.shape[0]

        config = _surv.SurvivalForestConfig(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_node_size=self.min_node_size,
            mtry=self.mtry,
            sample_fraction=self.sample_fraction,
            seed=self.seed,
            oob_error=self.oob_error,
        )

        x_flat = X.flatten().tolist()
        self.model_ = _surv.SurvivalForest.fit(
            x_flat, n_obs, self.n_features_in_, time.tolist(), status.tolist(), config
        )

        self.variable_importance_ = np.array(self.model_.variable_importance)
        self.oob_error_ = self.model_.oob_error
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
            Predicted risk scores (cumulative hazard at last time point).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        x_flat = X.flatten().tolist()
        return np.array(self.model_.predict_risk(x_flat, X.shape[0]))

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        survival = self.model_.predict_survival(x_flat, X.shape[0])
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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        result = self.model_.predict_median_survival_time(x_flat, X.shape[0])
        return np.array([t if t is not None else np.nan for t in result])


class AFTEstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Accelerated Failure Time (AFT) model.

    AFT models assume that covariates act multiplicatively on the survival time,
    i.e., log(T) = X @ beta + sigma * epsilon, where epsilon follows a specified
    error distribution.

    Parameters
    ----------
    distribution : str, default="weibull"
        Error distribution. One of:
        - "weibull": Weibull distribution (extreme value errors)
        - "lognormal": Log-normal distribution (Gaussian errors)
        - "loglogistic": Log-logistic distribution (logistic errors)
        - "exponential": Exponential distribution (special case of Weibull)
        - "gaussian": Gaussian distribution (for linear models)
        - "logistic": Logistic distribution (for linear models)
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    tol : float, default=1e-9
        Convergence tolerance.

    Attributes
    ----------
    model_ : SurvivalFit
        The underlying fitted AFT model.
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (acceleration factors in log scale).
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
    The AFT model interprets coefficients as acceleration factors:
    - Positive coefficients increase expected survival time
    - Negative coefficients decrease expected survival time
    - exp(coef) gives the multiplicative effect on survival time
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

        coefs = np.array(self.model_.coefficients)
        self.intercept_ = coefs[0]
        self.coef_ = coefs[1:-1]
        self.scale_ = np.exp(coefs[-1])
        self.converged_ = self.model_.convergence_flag == 0

        self.is_fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict expected survival time for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        survival_times : ndarray of shape (n_samples,)
            Predicted survival times (median by default).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        linear_pred = self.intercept_ + X @ self.coef_
        return np.exp(linear_pred)

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        linear_pred = self.intercept_ + X @ self.coef_

        if self.distribution in ("weibull", "exponential", "extreme_value"):
            median_z = np.log(np.log(2))
        elif self.distribution in ("lognormal", "gaussian", "loglogistic", "logistic"):
            median_z = 0.0
        else:
            median_z = 0.0

        return np.exp(linear_pred + self.scale_ * median_z)

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        if not 0 < q < 1:
            raise ValueError("q must be between 0 and 1")

        linear_pred = self.intercept_ + X @ self.coef_

        if self.distribution in ("weibull", "exponential", "extreme_value"):
            z_q = np.log(-np.log(1 - q))
        elif self.distribution in ("lognormal", "gaussian"):
            from scipy.stats import norm

            z_q = norm.ppf(q)
        elif self.distribution in ("loglogistic", "logistic"):
            z_q = np.log(q / (1 - q))
        else:
            z_q = 0.0

        return np.exp(linear_pred + self.scale_ * z_q)

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
        """Return acceleration factors (exp of coefficients).

        Returns
        -------
        af : ndarray of shape (n_features,)
            Acceleration factors. Values > 1 increase survival time,
            values < 1 decrease survival time.
        """
        check_is_fitted(self)
        return np.exp(self.coef_)


def iter_chunks(X: ArrayLike, batch_size: int = 1000) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Iterate over an array in chunks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input array.
    batch_size : int, default=1000
        Number of samples per chunk.

    Yields
    ------
    start_idx : int
        Starting index of the chunk.
    chunk : ndarray
        Chunk of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(10000, 5)
    >>> for start_idx, chunk in iter_chunks(X, batch_size=1000):
    ...     print(f"Processing samples {start_idx} to {start_idx + len(chunk)}")
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield start_idx, X[start_idx:end_idx]


class StreamingMixin:
    """Mixin class providing streaming/batched prediction methods."""

    def predict_batched(
        self, X: ArrayLike, batch_size: int = 1000
    ) -> Iterator[NDArray[np.float64]]:
        """Predict risk scores in batches to handle large datasets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        batch_size : int, default=1000
            Number of samples per batch.

        Yields
        ------
        risk_scores : ndarray of shape (batch_size,) or smaller for last batch
            Predicted risk scores for each batch.

        Examples
        --------
        >>> model = GradientBoostSurvivalEstimator()
        >>> model.fit(X_train, y_train)
        >>> all_predictions = []
        >>> for batch_preds in model.predict_batched(X_large, batch_size=5000):
        ...     all_predictions.append(batch_preds)
        >>> predictions = np.concatenate(all_predictions)
        """
        for _, chunk in iter_chunks(X, batch_size):
            yield self.predict(chunk)

    def predict_survival_batched(
        self, X: ArrayLike, batch_size: int = 1000
    ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Predict survival functions in batches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        batch_size : int, default=1000
            Number of samples per batch.

        Yields
        ------
        times : ndarray of shape (n_times,)
            Time points (same for all batches).
        survival : ndarray of shape (batch_size, n_times)
            Survival probabilities for each batch.
        """
        for _, chunk in iter_chunks(X, batch_size):
            yield self.predict_survival_function(chunk)

    def predict_to_array(
        self, X: ArrayLike, batch_size: int = 1000, out: NDArray | None = None
    ) -> NDArray[np.float64]:
        """Predict risk scores with optional pre-allocated output array.

        This method is memory-efficient for large datasets as it can write
        directly to a pre-allocated array or memory-mapped file.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        batch_size : int, default=1000
            Number of samples per batch.
        out : ndarray of shape (n_samples,), optional
            Pre-allocated output array. If None, a new array is created.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores.

        Examples
        --------
        >>> # Using with memory-mapped array for very large datasets
        >>> import numpy as np
        >>> out = np.memmap('predictions.dat', dtype='float64', mode='w+', shape=(1000000,))
        >>> model.predict_to_array(X_large, batch_size=10000, out=out)
        >>> out.flush()  # Write to disk
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        if out is None:
            out = np.empty(n_samples, dtype=np.float64)
        elif out.shape[0] != n_samples:
            raise ValueError(f"out has shape {out.shape}, expected ({n_samples},)")

        for start_idx, chunk in iter_chunks(X, batch_size):
            end_idx = start_idx + chunk.shape[0]
            out[start_idx:end_idx] = self.predict(chunk)

        return out


class StreamingCoxPHEstimator(CoxPHEstimator, StreamingMixin):
    """Cox PH Estimator with streaming/batched prediction support.

    This class extends CoxPHEstimator with methods for processing large
    datasets that don't fit in memory.

    See CoxPHEstimator for full documentation.
    """

    pass


class StreamingGradientBoostSurvivalEstimator(GradientBoostSurvivalEstimator, StreamingMixin):
    """Gradient Boosting Survival Estimator with streaming support.

    This class extends GradientBoostSurvivalEstimator with methods for
    processing large datasets that don't fit in memory.

    See GradientBoostSurvivalEstimator for full documentation.
    """

    pass


class StreamingSurvivalForestEstimator(SurvivalForestEstimator, StreamingMixin):
    """Survival Forest Estimator with streaming support.

    This class extends SurvivalForestEstimator with methods for processing
    large datasets that don't fit in memory.

    See SurvivalForestEstimator for full documentation.
    """

    pass


class StreamingAFTEstimator(AFTEstimator, StreamingMixin):
    """AFT Estimator with streaming/batched prediction support.

    This class extends AFTEstimator with methods for processing large
    datasets that don't fit in memory.

    See AFTEstimator for full documentation.
    """

    pass


class DeepSurvEstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible DeepSurv model.

    DeepSurv is a deep feedforward neural network for survival analysis
    using Cox partial likelihood loss.

    Parameters
    ----------
    hidden_layers : list of int, default=[64, 32]
        Number of neurons in each hidden layer.
    activation : str, default="selu"
        Activation function. One of "relu", "selu", or "tanh".
    dropout_rate : float, default=0.2
        Dropout rate applied after each hidden layer.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
    batch_size : int, default=256
        Mini-batch size for training.
    n_epochs : int, default=100
        Number of training epochs.
    l2_reg : float, default=0.0001
        L2 regularization (weight decay) coefficient.
    seed : int or None, default=None
        Random seed for reproducibility.
    early_stopping_patience : int or None, default=10
        Number of epochs without improvement before early stopping.
        Set to None to disable early stopping.
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation.

    Attributes
    ----------
    model_ : DeepSurv
        The underlying fitted model.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "selu",
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 100,
        l2_reg: float = 0.0001,
        seed: int | None = None,
        early_stopping_patience: int | None = 10,
        validation_fraction: float = 0.1,
    ):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DeepSurvEstimator":
        """Fit the DeepSurv model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, 2)
            Target values where y[:, 0] is survival time and y[:, 1] is event status.

        Returns
        -------
        self : DeepSurvEstimator
            Fitted estimator.
        """
        X, time, status = _validate_survival_data(X, y)
        self.n_features_in_ = X.shape[1]
        n_obs = X.shape[0]

        activation = _surv.Activation(self.activation)
        config = _surv.DeepSurvConfig(
            hidden_layers=self.hidden_layers,
            activation=activation,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            l2_reg=self.l2_reg,
            seed=self.seed,
            early_stopping_patience=self.early_stopping_patience,
            validation_fraction=self.validation_fraction,
        )

        x_flat = X.flatten().tolist()
        self.model_ = _surv.DeepSurv.fit(
            x_flat, n_obs, self.n_features_in_, time.tolist(), status.tolist(), config
        )

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

        x_flat = X.flatten().tolist()
        return np.array(self.model_.predict_risk(x_flat, X.shape[0]))

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        survival = self.model_.predict_survival(x_flat, X.shape[0])
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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_2d=True)

        x_flat = X.flatten().tolist()
        result = self.model_.predict_median_survival_time(x_flat, X.shape[0])
        return np.array([t if t is not None else np.nan for t in result])

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

    @property
    def train_loss(self) -> NDArray[np.float64]:
        """Training loss history."""
        check_is_fitted(self)
        return np.array(self.model_.train_loss)

    @property
    def val_loss(self) -> NDArray[np.float64]:
        """Validation loss history."""
        check_is_fitted(self)
        return np.array(self.model_.val_loss)


class StreamingDeepSurvEstimator(DeepSurvEstimator, StreamingMixin):
    """DeepSurv Estimator with streaming/batched prediction support.

    This class extends DeepSurvEstimator with methods for processing large
    datasets that don't fit in memory.

    See DeepSurvEstimator for full documentation.
    """

    pass


def predict_large_dataset(
    estimator,
    X: ArrayLike,
    batch_size: int = 1000,
    output_file: str | None = None,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Predict on a large dataset using batched processing.

    This is a utility function for making predictions on datasets that may
    not fit in memory. It processes the data in batches and optionally
    writes results to a memory-mapped file.

    Parameters
    ----------
    estimator : fitted estimator
        A fitted survival estimator with a predict method.
    X : array-like of shape (n_samples, n_features)
        Samples to predict. Can be a numpy array or memory-mapped array.
    batch_size : int, default=1000
        Number of samples to process at once.
    output_file : str, optional
        Path to output file for memory-mapped results. If provided, results
        are written to this file and can exceed available RAM.
    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        Predicted risk scores. If output_file is provided, this is a
        memory-mapped array.

    Examples
    --------
    >>> # Process a very large dataset
    >>> predictions = predict_large_dataset(
    ...     model, X_huge, batch_size=10000,
    ...     output_file='predictions.mmap', verbose=True
    ... )
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    if output_file is not None:
        predictions = np.memmap(output_file, dtype=np.float64, mode="w+", shape=(n_samples,))
    else:
        predictions = np.empty(n_samples, dtype=np.float64)

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, (start_idx, chunk) in enumerate(iter_chunks(X, batch_size)):
        end_idx = start_idx + chunk.shape[0]
        predictions[start_idx:end_idx] = estimator.predict(chunk)

        if verbose:
            print(f"Processed batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx})")

    if output_file is not None:
        predictions.flush()

    return predictions


def survival_curves_to_disk(
    estimator,
    X: ArrayLike,
    output_file: str,
    batch_size: int = 100,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], np.memmap]:
    """Compute survival curves and write to disk for large datasets.

    This function computes survival curves in batches and stores them in
    a memory-mapped file, allowing processing of datasets larger than RAM.

    Parameters
    ----------
    estimator : fitted estimator
        A fitted survival estimator with predict_survival_function method.
    X : array-like of shape (n_samples, n_features)
        Samples to predict.
    output_file : str
        Path to output file for memory-mapped survival curves.
    batch_size : int, default=100
        Number of samples to process at once. Smaller values use less
        memory but are slower.
    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    times : ndarray of shape (n_times,)
        Time points for the survival curves.
    survival : memmap of shape (n_samples, n_times)
        Memory-mapped array of survival probabilities.

    Examples
    --------
    >>> times, survival_curves = survival_curves_to_disk(
    ...     model, X_huge, 'survival_curves.mmap',
    ...     batch_size=100, verbose=True
    ... )
    >>> # Access individual survival curves without loading all into memory
    >>> curve_0 = survival_curves[0]  # Loads only first curve
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    first_times, first_surv = estimator.predict_survival_function(X[:1])
    n_times = len(first_times)
    times = first_times

    survival = np.memmap(output_file, dtype=np.float64, mode="w+", shape=(n_samples, n_times))

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, (start_idx, chunk) in enumerate(iter_chunks(X, batch_size)):
        end_idx = start_idx + chunk.shape[0]
        _, batch_surv = estimator.predict_survival_function(chunk)
        survival[start_idx:end_idx] = batch_surv

        if verbose:
            print(f"Processed batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx})")

    survival.flush()
    return times, survival
