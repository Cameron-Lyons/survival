# ruff: noqa: N803
from __future__ import annotations

from ._sklearn_ensemble import GradientBoostSurvivalEstimator, SurvivalForestEstimator
from ._sklearn_streaming import StreamingMixin

__all__ = [
    "StreamingGradientBoostSurvivalEstimator",
    "StreamingSurvivalForestEstimator",
]


class StreamingGradientBoostSurvivalEstimator(GradientBoostSurvivalEstimator, StreamingMixin):
    """Gradient boosting survival estimator with streaming support.

    This class extends GradientBoostSurvivalEstimator with methods for
    processing large datasets that don't fit in memory.
    """


class StreamingSurvivalForestEstimator(SurvivalForestEstimator, StreamingMixin):
    """Survival forest estimator with streaming support.

    This class extends SurvivalForestEstimator with methods for processing
    large datasets that don't fit in memory.
    """
