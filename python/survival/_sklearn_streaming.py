# ruff: noqa: N803, N806, UP037
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from ._sklearn_aft import AFTEstimator
from ._sklearn_cox import CoxPHEstimator
from ._sklearn_ensemble import GradientBoostSurvivalEstimator, SurvivalForestEstimator

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


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
