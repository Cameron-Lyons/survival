"""Data preparation kernels (R's ``aeqSurv``, ``tmerge``, ``survSplit``); those named like an R
function (``neardate``, ``tcut``, ...) are in ``survival.kernels``.

All row indices returned by these bindings are zero-based.
"""

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "AeqSurvResult",
        "aeq_surv",
        "ClusterResult",
        "StrataResult",
        "RttrightResult",
        "SurvcondenseResult",
        "SurvSplitResult",
        "survsplit",
        "TcutResult",
        "TmergeStep",
        "tmerge_step",
        "Surv2CountingResult",
        "surv2counting",
        "TotimelineResult",
    ],
)
