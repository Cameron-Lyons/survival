from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

_PUBLIC_EXPORTS = (
    "AFTEstimator",
    "CoxPHEstimator",
    "DeepSurvEstimator",
    "GradientBoostSurvivalEstimator",
    "StreamingAFTEstimator",
    "StreamingCoxPHEstimator",
    "StreamingDeepSurvEstimator",
    "StreamingGradientBoostSurvivalEstimator",
    "StreamingMixin",
    "StreamingSurvivalForestEstimator",
    "SurvivalForestEstimator",
    "SurvivalScoreMixin",
    "iter_chunks",
    "predict_large_dataset",
    "survival_curves_to_disk",
)

_EXPORT_MODULES = {
    "_HAS_SKLEARN": "._sklearn_common",
    "SurvivalScoreMixin": "._sklearn_common",
    "AFTEstimator": "._sklearn_aft",
    "CoxPHEstimator": "._sklearn_cox",
    "DeepSurvEstimator": "._sklearn_deep",
    "StreamingDeepSurvEstimator": "._sklearn_deep",
    "GradientBoostSurvivalEstimator": "._sklearn_ensemble",
    "SurvivalForestEstimator": "._sklearn_ensemble",
    "StreamingAFTEstimator": "._sklearn_streaming_aft",
    "StreamingCoxPHEstimator": "._sklearn_streaming_cox",
    "StreamingGradientBoostSurvivalEstimator": "._sklearn_streaming_ensemble",
    "StreamingMixin": "._sklearn_streaming",
    "StreamingSurvivalForestEstimator": "._sklearn_streaming_ensemble",
    "iter_chunks": "._sklearn_streaming",
    "predict_large_dataset": "._sklearn_streaming",
    "survival_curves_to_disk": "._sklearn_streaming",
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORT_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_path, __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"_HAS_SKLEARN"})
