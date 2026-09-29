"""Adapter for population contrasts of a model fitted outside this package."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ._coerce import _float_vector
from ._fit import _r_factor_design
from ._formula import _design_term_output_names, _fit_formula_design, model_frame
from ._types import _FormulaDesign


@dataclass(frozen=True)
class YatesModel:
    """Supply an external linear model or GLM to :func:`yates`.

    Coefficients and covariance follow the R formula's design-column order,
    including its intercept, with ``lm``'s factor levels: those that occur in
    ``data`` (``drop.unused.levels = TRUE``). ``sigma2`` is the residual variance
    for sum-of-squares tests. ``family`` supplies a GLM's vectorized inverse
    link, using ``linkinv`` or ``link.inverse``. Optional ``weights`` are the
    fitted model's case weights, used for linear data-population means.
    No model is refitted by this adapter.
    """

    formula: str
    data: Any
    coefficients: list[float]
    variance: list[list[float]]
    sigma2: float | None = None
    family: Any = field(default=None, kw_only=True, repr=False)
    weights: Any = field(default=None, kw_only=True, repr=False)
    design: _FormulaDesign = field(init=False, repr=False)
    model: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        frame = model_frame(self.formula, self.data, na_action="fail")
        design = _r_factor_design(
            frame.data,
            _fit_formula_design(
                frame.data, frame.spec, frame.terms, frame.n, include_intercept=True
            ),
            drop_unused_levels=True,
        )
        beta = _float_vector(self.coefficients, "coefficients")
        names = (["(Intercept)"] if design.intercept else []) + [
            name for term in design.covariates for name in _design_term_output_names(term)
        ]
        if len(beta) != len(names):
            raise ValueError(f"coefficients must follow formula columns {names}")
        object.__setattr__(self, "coefficients", beta)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "model", dict(frame.data))
        if self.family is not None:
            from ._yates_setup import _inverse_link

            if _inverse_link(self) is None:
                raise ValueError("family must supply a callable linkinv or link.inverse")
        if self.weights is not None:
            weights = _float_vector(self.weights, "weights")
            if len(weights) != frame.n or any(not math.isfinite(w) or w < 0 for w in weights):
                raise ValueError("weights must be finite, nonnegative and match the model rows")
            if not any(w > 0 for w in weights):
                raise ValueError("weights must have a positive total")
            self.model["(weights)"] = weights
            object.__setattr__(self, "weights", weights)
