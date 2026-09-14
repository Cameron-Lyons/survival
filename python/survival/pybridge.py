"""Hooks that call back into Python from Rust (R's ``coxpenal.fit`` penalty callback)."""

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "CoxPenaltyTerms",
        "cox_callback",
    ],
)
