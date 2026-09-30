"""Hooks between Python and Rust: R's ``coxpenal.fit`` penalty callback and the pickle
reconstructors of the native result classes."""

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
