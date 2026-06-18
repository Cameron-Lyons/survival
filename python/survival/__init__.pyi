# ruff: noqa: F401

from typing import Any, Final

from . import bayesian as bayesian
from . import causal as causal
from . import core as core
from . import data_prep as data_prep
from . import datasets as datasets
from . import interpretability as interpretability
from . import interval as interval
from . import joint as joint
from . import missing as missing
from . import ml as ml
from . import monitoring as monitoring
from . import population as population
from . import pybridge as pybridge
from . import qol as qol
from . import r_api as r_api
from . import recurrent as recurrent
from . import regression as regression
from . import relative as relative
from . import reliability_tools as reliability_tools
from . import residuals as residuals
from . import spatial as spatial
from . import surv_analysis as surv_analysis
from . import validation as validation
from .r_api import (
    Surv,
    anova,
    basehaz,
    concordance,
    cox_zph,
    coxph,
    coxph_detail,
    is_surv,
    predict,
    survdiff,
    survfit,
    survreg,
)
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

__preferred__: Final[tuple[str, ...]]
__legacy_root_exports__: Final[tuple[str, ...]]
__deprecated_root_exports__: Final[tuple[str, ...]]
__deprecated_root_export_reason__: Final[str]
__version__: Final[str]
__all__: list[str]

def __getattr__(name: str) -> Any: ...
