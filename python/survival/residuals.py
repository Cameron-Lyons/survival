"""Martingale residual kernels and the parametric residual container.

Model-based residuals live on the fits: ``CoxPHFit.martingale_residuals()`` and friends
(``survival.regression``) and ``SurvregFit.residuals()``; R's ``residuals.survfit`` is
``survival.surv_analysis.survfitresid``.
"""

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "coxmart",
        "agmart",
        "SurvregResiduals",
    ],
)
