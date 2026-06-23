from importlib import import_module as _import_module

from ._binding_manifest import MODULE_BINDINGS

__version__ = "2.0.0"

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
    "recurrent": ".recurrent",
    "regression": ".regression",
    "relative": ".relative",
    "reliability_tools": ".reliability_tools",
    "residuals": ".residuals",
    "r_api": ".r_api",
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
    "Surv",
    "aic",
    "anova",
    "as_data_frame",
    "basehaz",
    "bic",
    "coef",
    "coef_names",
    "confint",
    "concordance",
    "coxph",
    "coxph_detail",
    "cox_zph",
    "degrees_freedom",
    "df_residual",
    "extract_aic",
    "fitted",
    "is_surv",
    "loglik",
    "model_formula",
    "model_frame",
    "model_matrix",
    "model_weights",
    "model_summary",
    "nobs",
    "predict",
    "survdiff",
    "survfit",
    "survreg",
    "vcov",
]

_PREFERRED_EXPORTS = list(_PUBLIC_MODULES) + _R_EXPORTS + _SKLEARN_EXPORTS

_LEGACY_EXPORT_MODULES = {
    name: module_name
    for module_name in _PUBLIC_MODULES
    for name in MODULE_BINDINGS.get(module_name, ())
    if name not in _PREFERRED_EXPORTS
}

__preferred__ = tuple(_PREFERRED_EXPORTS)
__legacy_root_exports__ = tuple(_LEGACY_EXPORT_MODULES)
__deprecated_root_exports__ = __legacy_root_exports__
__deprecated_root_export_reason__ = (
    "Root-level algorithm and result exports are retained for compatibility. "
    "Prefer importing from domain modules such as survival.regression, "
    "survival.surv_analysis, or survival.validation."
)

__all__ = list(dict.fromkeys(_PREFERRED_EXPORTS))


def _load_public_module(name):
    module_path = _PUBLIC_MODULES[name]
    module = globals().get(name)
    if module is None:
        module = _import_module(module_path, __name__)
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
        value = getattr(_load_public_module("r_api"), name)
        globals()[name] = value
        return value

    if name in _SKLEARN_EXPORTS:
        value = getattr(_import_module(".sklearn_compat", __name__), name)
        globals()[name] = value
        return value

    module_name = _LEGACY_EXPORT_MODULES.get(name)
    if module_name is not None:
        return getattr(_load_public_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | {"__preferred__", "__deprecated_root_exports__", "__version__"})
