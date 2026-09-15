"""Survival analysis in Rust with Python bindings.

The root package exposes three things, all loaded lazily on first access:

* the domain modules (``survival.regression``, ``survival.surv_analysis``, ...), each binding
  a documented subset of the Rust extension ``survival._survival``;
* the R-style API of :mod:`survival.r` under the R names (``Surv``, ``survfit``, ``coxph``,
  ...), listed in ``_R_EXPORTS``;
* the scikit-learn compatible estimators of :mod:`survival.sklearn_compat`, listed in
  ``_SKLEARN_EXPORTS``.

``__all__`` is derived from these three lists; ``__init__.pyi`` mirrors them and
``python/tests/test_binding_contract.py`` checks that the two agree.
"""

from importlib import import_module as _import_module

__version__ = "1.3.0"

_PUBLIC_MODULES = {
    "bayesian": ".bayesian",
    "causal": ".causal",
    "core": ".core",
    "data_prep": ".data_prep",
    "datasets": ".datasets",
    "interpretability": ".interpretability",
    "interval": ".interval",
    "joint": ".joint",
    "missing": ".missing",
    "ml": ".ml",
    "monitoring": ".monitoring",
    "population": ".population",
    "pybridge": ".pybridge",
    "qol": ".qol",
    "r": ".r",
    "recurrent": ".recurrent",
    "regression": ".regression",
    "relative": ".relative",
    "reliability_tools": ".reliability_tools",
    "residuals": ".residuals",
    "r_api": ".r_api",
    "sklearn_compat": ".sklearn_compat",
    "spatial": ".spatial",
    "surv_analysis": ".surv_analysis",
    "validation": ".validation",
}

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

_R_EXPORTS = [
    "AaregModelResult",
    "StrataFactor",
    "CchModelResult",
    "Surv",
    "Surv2",
    "Surv2data",
    "FineGrayFrame",
    "FineGrayOutput",
    "RateTable",
    "PyearsResult",
    "SurvExpResult",
    "SurvfitConfidenceIntervalResult",
    "SurvfitMultiStateResult",
    "TMergeFrame",
    "TMergeOperation",
    "TcutResult",
    "YatesResult",
    "aic",
    "aareg",
    "aeqSurv",
    "anova",
    "as_data_frame",
    "basehaz",
    "bcloglog",
    "bic",
    "blog",
    "blogit",
    "bprobit",
    "cipoisson",
    "coef",
    "coef_names",
    "confint",
    "concordance",
    "clogit",
    "cch",
    "coxph",
    "coxph_detail",
    "coxph_wtest",
    "cox_zph",
    "cumevent",
    "cumtdc",
    "degrees_freedom",
    "dsurvreg",
    "df_residual",
    "extract_aic",
    "event",
    "fitted",
    "finegray",
    "fromtimeline",
    "format_surv",
    "is_surv",
    "is_na_surv",
    "is_ratetable",
    "loglik",
    "lvcf",
    "model_formula",
    "model_frame",
    "model_matrix",
    "model_term_names",
    "model_weights",
    "model_summary",
    "nobs",
    "neardate",
    "nostutter",
    "nsk",
    "pyears",
    "pspline",
    "pseudo",
    "predict",
    "psurvreg",
    "qsurvreg",
    "ratetableDate",
    "rsurvreg",
    "rttright",
    "strata",
    "survdiff",
    "survConcordance",
    "survConcordance_fit",
    "survcondense",
    "survcheck",
    "survexp",
    "survexp_individual",
    "survexp_mn",
    "survobrien",
    "survexp_us",
    "survexp_usr",
    "survfit",
    "survfit0",
    "survfit_confint",
    "survfit_residuals",
    "survfitkm_counting_influence",
    "survfitkm_influence",
    "survSplit",
    "survreg",
    "tdc",
    "tcut",
    "tmerge",
    "totimeline",
    "vcov",
    "yates",
]

__all__ = list(dict.fromkeys([*_PUBLIC_MODULES, *_R_EXPORTS, *_SKLEARN_EXPORTS]))


def _load_public_module(name):
    module = globals().get(name)
    if module is None:
        module = _import_module(_PUBLIC_MODULES[name], __name__)
        globals()[name] = module
    return module


def __getattr__(name):
    if name == "_survival":
        value = _import_module("._survival", __name__)
        globals()[name] = value
        return value

    if name in _PUBLIC_MODULES:
        return _load_public_module(name)

    if name in _R_EXPORTS:
        value = getattr(_load_public_module("r"), name)
        globals()[name] = value
        return value

    if name in _SKLEARN_EXPORTS:
        value = getattr(_load_public_module("sklearn_compat"), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*__all__, "__version__"})
