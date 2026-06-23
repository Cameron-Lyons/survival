# ruff: noqa: N803
from __future__ import annotations

from ._sklearn_aft import AFTEstimator
from ._sklearn_streaming import StreamingMixin

__all__ = ["StreamingAFTEstimator"]


class StreamingAFTEstimator(AFTEstimator, StreamingMixin):
    """AFT estimator with streaming/batched prediction support.

    This class extends AFTEstimator with methods for processing large
    datasets that don't fit in memory.
    """
