"""The penalty branch of R's ``survreg`` (``survreg.R:203-229``): ``ridge()`` and
``pspline()`` terms are fitted by ``survpenal.fit``, whose port is the Rust
``survpenal_fit``; this module finds the penalised terms of the design and lays out R's
``assign``/``pcols`` for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .. import _survival as _core
from ._types import _FormulaDesign, _PenaltyDesignTerm

if TYPE_CHECKING:
    from ._survreg import _SurvregFrame


def _term_index(design: _FormulaDesign, position: int) -> int:
    """The 1-based model term of the design's covariate at ``position``."""

    if len(design.term_assignments) == len(design.covariates):
        return design.term_assignments[position]
    return position + 1


def penalty_terms(design: _FormulaDesign | None) -> list[tuple[int, _PenaltyDesignTerm]]:
    """The penalised terms of the design (``pterms``) with their 1-based model term (R's
    ``attr(X, "assign")`` value).  Only main effects are searched: the design builder
    refuses a penalty term inside an interaction."""

    if design is None:
        return []
    return [
        (_term_index(design, position), term)
        for position, term in enumerate(design.covariates)
        if isinstance(term, _PenaltyDesignTerm) and term.penalized
    ]


def assign_list(frame: _SurvregFrame) -> tuple[list[str], list[list[int]]]:
    """R's ``attrassign(X, newTerms)``: the labels of ``(Intercept)`` and of every
    non-strata term, with the 0-based design columns of each, in column order."""

    labels: list[str] = []
    columns: list[list[int]] = []
    if 0 in frame.assign:
        labels.append("(Intercept)")
        columns.append([j for j, term in enumerate(frame.assign) if term == 0])
    for term_index, label in enumerate(frame.term_labels, start=1):
        term_columns = [j for j, term in enumerate(frame.assign) if term == term_index]
        if term_index != frame.strata_term and term_columns:
            labels.append(label)
            columns.append(term_columns)
    return labels, columns


def fit_penalized(
    frame: _SurvregFrame,
    terms: Sequence[tuple[int, _PenaltyDesignTerm]],
    data: Any,
    distribution: Any,
    *,
    init: Sequence[float] | None,
    scale: float,
    control: Any,
    robust: bool | None,
) -> tuple[Any, tuple[str, ...]]:
    """``survpenal.fit`` on the frame's ``SurvregData`` for the ``penalty_terms`` of
    its design: the ``SurvpenalFit`` and the labels of its ``assign2`` terms (``"sigma"``
    last when the scale is estimated)."""

    labels, columns = assign_list(frame)
    pcols = [
        [j for j, assigned in enumerate(frame.assign) if assigned == term_index]
        for term_index, _ in terms
    ]
    fit = _core.survpenal_fit(
        data,
        distribution,
        [term.penalty for _, term in terms],
        pcols,
        assign=columns,
        init=None if init is None else list(init),
        scale=scale,
        control=control,
        robust=robust,
    )
    if len(fit.assign2) > len(labels):
        labels.append("sigma")
    return fit, tuple(labels)
