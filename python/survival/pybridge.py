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
