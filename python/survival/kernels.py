"""The native kernels of the R-survival core.

Survival curves, Cox and AFT fitting, residuals, population rates, data
preparation and validation statistics take arrays and return typed native
results. ``survival.r`` and the package root hold R's API built on them, under
R's names; a kernel that shares a name with an R function (``survdiff``,
``pyears``, ``coxph_fit``, ...) is available only here.
"""

from importlib import import_module

from ._binding_utils import bind_names

_DOMAIN_MODULES = (
    "core",
    "data_prep",
    "interval",
    "population",
    "regression",
    "surv_analysis",
    "validation",
)

_OWN = bind_names(
    globals(),
    [
        "concordancefit",
        "nsk",
        "cluster",
        "strata",
        "lvcf",
        "neardate",
        "nostutter",
        "rttright",
        "survcondense",
        "tcut",
        "totimeline",
        "is_ratetable",
        "match_ratetable",
        "PyearsResult",
        "pyears",
        "PyearsSummary",
        "summary_pyears",
        "SurvExpResult",
        "survexp",
        "summary_survexp",
        "survexp_mn",
        "survexp_us",
        "survexp_usr",
        "coxph_fit",
        "cox_zph",
        "coxph_wtest",
        "coxph_detail",
        "finegray",
        "survreg_fit",
        "SurvregFitResult",
        "survpenal_fit",
        "SurvpenalFitResult",
        "dsurvreg",
        "psurvreg",
        "qsurvreg",
        "rsurvreg",
        "aggregate_survfit",
        "coxsurv_fit",
        "SurvDiffResult",
        "survdiff",
        "pseudo",
        "StateFigResult",
        "statefig",
        "SurvfitKMResult",
        "survfit0",
        "survfit_confint",
        "summary_survfit",
        "quantile_survfit",
        "BrierResult",
        "brier",
        "cipoisson",
        "royston",
        "SurvCheckProblem",
        "SurvCheckResult",
        "survcheck",
        "survobrien",
        "YatesResult",
        "yates",
        "agmart",
        "SurvregResiduals",
    ],
)

_SHARED: list[str] = []
for _name in _DOMAIN_MODULES:
    _module = vars(import_module(f"survival.{_name}"))
    for _binding in _module["__all__"]:
        if _binding in _module:
            globals()[_binding] = _module[_binding]
            _SHARED.append(_binding)

__all__ = list(dict.fromkeys([*_SHARED, *_OWN]))

del _name, _module, _binding
