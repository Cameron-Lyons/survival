# ruff: noqa: F401

from . import bayesian as _bayesian_module
from . import causal as _causal_module
from . import core as _core_module
from . import data_prep as _data_prep_module
from . import datasets as _datasets_module
from . import interpretability as _interpretability_module
from . import interval as _interval_module
from . import joint as _joint_module
from . import missing as _missing_module
from . import ml as _ml_module
from . import monitoring as _monitoring_module
from . import population as _population_module
from . import pybridge as _pybridge_module
from . import qol as _qol_module
from . import recurrent as _recurrent_module
from . import regression as _regression_module
from . import relative as _relative_module
from . import reliability_tools as _reliability_tools_module
from . import residuals as _residuals_module
from . import spatial as _spatial_module
from . import surv_analysis as _surv_analysis_module
from . import validation as _validation_module
from .sklearn_compat import (
    AFTEstimator,
    CoxPHEstimator,
    DeepSurvEstimator,
    GradientBoostSurvivalEstimator,
    StreamingAFTEstimator,
    StreamingCoxPHEstimator,
    StreamingDeepSurvEstimator,
    StreamingGradientBoostSurvivalEstimator,
    StreamingMixin,
    StreamingSurvivalForestEstimator,
    SurvivalForestEstimator,
    iter_chunks,
    predict_large_dataset,
    survival_curves_to_disk,
)

_PUBLIC_MODULES = {
    "bayesian": _bayesian_module,
    "causal": _causal_module,
    "core": _core_module,
    "data_prep": _data_prep_module,
    "datasets": _datasets_module,
    "interpretability": _interpretability_module,
    "interval": _interval_module,
    "joint": _joint_module,
    "missing": _missing_module,
    "ml": _ml_module,
    "monitoring": _monitoring_module,
    "population": _population_module,
    "pybridge": _pybridge_module,
    "qol": _qol_module,
    "recurrent": _recurrent_module,
    "regression": _regression_module,
    "relative": _relative_module,
    "reliability_tools": _reliability_tools_module,
    "residuals": _residuals_module,
    "spatial": _spatial_module,
    "surv_analysis": _surv_analysis_module,
    "validation": _validation_module,
}

_DOMAIN_MODULES = (
    _datasets_module,
    _data_prep_module,
    _core_module,
    _monitoring_module,
    _population_module,
    _regression_module,
    _residuals_module,
    _bayesian_module,
    _causal_module,
    _interpretability_module,
    _interval_module,
    _joint_module,
    _missing_module,
    _ml_module,
    _pybridge_module,
    _qol_module,
    _recurrent_module,
    _relative_module,
    _reliability_tools_module,
    _spatial_module,
    _surv_analysis_module,
    _validation_module,
)

_SKLEARN_EXPORTS = [
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
    "iter_chunks",
    "predict_large_dataset",
    "survival_curves_to_disk",
]

for _module in _DOMAIN_MODULES:
    globals().update({name: getattr(_module, name) for name in _module.__all__})

for _name, _module in _PUBLIC_MODULES.items():
    globals()[_name] = _module

__all__ = []
for _module in _DOMAIN_MODULES:
    __all__.extend(_module.__all__)
__all__.extend(_SKLEARN_EXPORTS)
__all__.extend(_PUBLIC_MODULES)
__all__ = list(dict.fromkeys(__all__))

del _name
del _module
del _DOMAIN_MODULES
del _PUBLIC_MODULES
del _SKLEARN_EXPORTS
