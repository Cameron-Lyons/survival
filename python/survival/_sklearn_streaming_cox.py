# ruff: noqa: N803
from __future__ import annotations

from ._sklearn_cox import CoxPHEstimator
from ._sklearn_streaming import StreamingMixin

__all__ = ["StreamingCoxPHEstimator"]


class StreamingCoxPHEstimator(CoxPHEstimator, StreamingMixin):
    """Cox PH estimator with streaming/batched prediction support.

    This class extends CoxPHEstimator with methods for processing large
    datasets that don't fit in memory.
    """
