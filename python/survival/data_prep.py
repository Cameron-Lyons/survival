"""Data preparation kernels (R's ``aeqSurv``, ``cluster``, ``strata``, ``neardate``, ``tmerge``).

All row indices returned by these bindings are zero-based.
"""

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "AeqSurvResult",
        "aeq_surv",
        "ClusterResult",
        "cluster",
        "StrataResult",
        "strata",
        "lvcf",
        "neardate",
        "nostutter",
        "RttrightResult",
        "rttright",
        "SurvcondenseResult",
        "survcondense",
        "SurvSplitResult",
        "survsplit",
        "TcutResult",
        "tcut",
        "TmergeStep",
        "tmerge_step",
        "Surv2CountingResult",
        "surv2counting",
        "TotimelineResult",
        "totimeline",
    ],
)
