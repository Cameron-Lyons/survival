"""Internal hooks for R interoperability and native-result reconstruction."""

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "CoxPenaltyTerms",
        "_survpenal_fit_from_state",
        "_unpickle",
        "cox_callback",
    ],
)


def _call_fit_with_warnings(
    function: Callable[..., Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a fit and its warnings for R's condition system."""
    with warnings.catch_warnings(record=True) as recorded:
        # Each R fit call should signal its diagnostics, including when Python
        # has already emitted the same warning from this source line.
        warnings.simplefilter("always", RuntimeWarning)
        result = function(**arguments)
    return {"result": result, "warnings": [str(issue.message) for issue in recorded]}


def _surv_columns(response: Any) -> dict[str, Any]:
    """Bulk normalized response columns for R's native model-frame adapter."""
    import numpy as np

    from .r._surv import Surv

    if not isinstance(response, Surv):
        raise TypeError("argument is not a Surv object")
    return {
        **{
            name: None
            if (values := getattr(response, name)) is None
            else np.asarray(values, dtype=float)
            for name in ("time", "event", "start", "time2")
        },
        "type": response.type,
        "states": list(response.states),
        "clabel": response.clabel,
    }


def _r_subset(rows: list[int]) -> Any:
    """Preserve R's missing selected rows until the shared na.action step."""
    from .r._coerce import _RSubset

    return _RSubset(rows)


def _yates_model_metadata(fit: Any) -> dict[str, Any]:
    """Factor levels needed to rebuild a fitted Python model's R model frame."""
    from .r._fit import _formula_design_for_fit
    from .r._formula import _covariate_term_name
    from .r._types import _CategoricalDesignTerm
    from .r._yates import _design_factors

    design = _formula_design_for_fit(fit)
    if design is None:
        raise TypeError("Yates requires fitted formula metadata")
    factors = [part for part in _design_factors(design) if isinstance(part, _CategoricalDesignTerm)]
    return {
        "xlevels": {_covariate_term_name(part.term): list(part.levels) for part in factors},
        "raw_levels": {
            part.term.column: list(part.levels) for part in factors if not part.term.strata
        },
    }


def _concordance_lm_data(
    data: list[dict[str, Any]], names: list[str], options: dict[str, Any], newdata: bool = False
) -> Any:
    """R evaluates external model frames and predictions; Rust scores them."""
    from .r._concordance import _concordance_from_data, _FitData
    from .r._surv import Surv

    prepared = [
        _FitData(
            y=Surv(value["y"]),
            x=list(value["x"]),
            strata=None,
            strata_levels=(),
            weights=None if value.get("weights") is None else list(value["weights"]),
            cluster=value.get("cluster"),
            timefix=None if newdata else False,
        )
        for value in data
    ]
    return _concordance_from_data(prepared, options=options, names=names)


def _concordance_survival_models(
    fits: list[Any],
    names: list[str],
    options: dict[str, Any],
    newdata: Any | None,
    clusters: list[Any],
) -> Any:
    """Keep fitted rows in Python; R supplies cluster codes in its sort order."""
    from .r._concordance import _concordance_from_data, _fit_data

    need_weights = any(fit.weights is not None for fit in fits)
    prepared = [
        _fit_data(fit, newdata, need_weights, cluster)
        for fit, cluster in zip(fits, clusters, strict=True)
    ]
    return _concordance_from_data(prepared, options=options, names=names)


def _survexp_cox_fit(
    fit: Any, data: Any, group: Any, weights: Any, y: Any, times: Any, method: str
) -> Any:
    """Prepared R population rows applied to a Python-backed Cox rate model."""
    import numpy as np

    from .r._coxph import _check_interaction_margins, _survfit_newdata, predict_coxph

    if method.startswith("individual"):
        hazard = np.asarray(predict_coxph(fit, data, type="expected", na_action="na.fail"))
        return hazard if method == "individual.h" else np.exp(-hazard)
    _check_interaction_margins(fit)
    new, _, _ = _survfit_newdata(fit, data, individual=False, id=None, na_action="na.fail")
    engine = fit.penalized if fit.penalized is not None else fit.fit
    return engine.expected_survival(
        new.x,
        group,
        weights,
        new_strata=new.strata,
        new_offset=new.offset,
        y=y,
        times=times,
        method=method,
    ).to_arrays()
