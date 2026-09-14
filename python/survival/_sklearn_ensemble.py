from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _survival as _surv
from ._sklearn_common import (
    BaseEstimator,
    FlatModelPredictMixin,
    RegressorMixin,
    SurvivalScoreMixin,
    _validate_survival_data,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class GradientBoostSurvivalEstimator(
    FlatModelPredictMixin, SurvivalScoreMixin, BaseEstimator, RegressorMixin
):
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


class SurvivalForestEstimator(
    FlatModelPredictMixin, SurvivalScoreMixin, BaseEstimator, RegressorMixin
):
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
