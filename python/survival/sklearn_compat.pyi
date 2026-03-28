# ruff: noqa: N803

from collections.abc import Iterator
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class SurvivalScoreMixin:
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def score(self, X: ArrayLike, y: ArrayLike) -> float: ...

class CoxPHEstimator(SurvivalScoreMixin):
    n_iters: int
    n_features_in_: int
    coef_: NDArray[np.float64]
    is_fitted_: bool

    def __init__(self, n_iters: int = 20) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_survival_function(
        self, X: ArrayLike, times: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]: ...

class GradientBoostSurvivalEstimator(SurvivalScoreMixin):
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    subsample: float
    max_features: int | None
    seed: int | None
    n_features_in_: int
    feature_importances_: NDArray[np.float64]
    is_fitted_: bool

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
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_survival_function(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]: ...

class SurvivalForestEstimator(SurvivalScoreMixin):
    n_trees: int
    max_depth: int | None
    min_node_size: int
    mtry: int | None
    sample_fraction: float
    seed: int | None
    oob_error: bool
    n_features_in_: int
    variable_importance_: NDArray[np.float64]
    oob_error_: float | None
    is_fitted_: bool

    def __init__(
        self,
        n_trees: int = 500,
        max_depth: int | None = None,
        min_node_size: int = 15,
        mtry: int | None = None,
        sample_fraction: float = 0.632,
        seed: int | None = None,
        oob_error: bool = True,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_survival_function(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]: ...

class AFTEstimator:
    distribution: str
    max_iter: int
    tol: float
    n_features_in_: int
    intercept_: float
    coef_: NDArray[np.float64]
    scale_: float
    converged_: bool
    is_fitted_: bool

    def __init__(
        self,
        distribution: str = "weibull",
        max_iter: int = 200,
        tol: float = 1e-9,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_median(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_quantile(self, X: ArrayLike, q: float = 0.5) -> NDArray[np.float64]: ...
    def score(self, X: ArrayLike, y: ArrayLike) -> float: ...
    @property
    def acceleration_factors(self) -> NDArray[np.float64]: ...

def iter_chunks(
    X: ArrayLike, batch_size: int = 1000
) -> Iterator[tuple[int, NDArray[np.float64]]]: ...

class StreamingMixin:
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_survival_function(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_batched(
        self, X: ArrayLike, batch_size: int = 1000
    ) -> Iterator[NDArray[np.float64]]: ...
    def predict_survival_batched(
        self, X: ArrayLike, batch_size: int = 1000
    ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]: ...
    def predict_to_array(
        self,
        X: ArrayLike,
        batch_size: int = 1000,
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]: ...

class StreamingCoxPHEstimator(CoxPHEstimator, StreamingMixin): ...
class StreamingGradientBoostSurvivalEstimator(GradientBoostSurvivalEstimator, StreamingMixin): ...
class StreamingSurvivalForestEstimator(SurvivalForestEstimator, StreamingMixin): ...
class StreamingAFTEstimator(AFTEstimator, StreamingMixin): ...

class DeepSurvEstimator:
    hidden_layers: list[int]
    activation: str
    dropout_rate: float
    learning_rate: float
    batch_size: int
    n_epochs: int
    l2_reg: float
    seed: int | None
    early_stopping_patience: int | None
    validation_fraction: float
    n_features_in_: int
    is_fitted_: bool

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
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def predict_survival_function(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_median_survival_time(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def score(self, X: ArrayLike, y: ArrayLike) -> float: ...
    @property
    def train_loss(self) -> NDArray[np.float64]: ...
    @property
    def val_loss(self) -> NDArray[np.float64]: ...

class StreamingDeepSurvEstimator(DeepSurvEstimator, StreamingMixin): ...

def predict_large_dataset(
    estimator: Any,
    X: ArrayLike,
    batch_size: int = 1000,
    output_file: str | None = None,
    verbose: bool = False,
) -> NDArray[np.float64]: ...
def survival_curves_to_disk(
    estimator: Any,
    X: ArrayLike,
    output_file: str,
    batch_size: int = 100,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
