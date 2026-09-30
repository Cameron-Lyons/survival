"""The penalty branch of R's ``survreg`` (``survreg.R:203-229``): ``ridge()`` and
``pspline()`` terms are fitted by ``survpenal.fit``, whose port is the Rust
``survpenal_fit``; this module finds the penalised terms of the design and lays out R's
``assign``/``pcols`` for it.  The fit's print method is in ``_survpenal_print``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import _survival as _core
from ._terms import _column_groups
from ._types import _FormulaDesign, _PenaltyDesignTerm


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


def assign_list(
    assign: Sequence[int], term_labels: Sequence[str], strata_term: int
) -> tuple[list[str], list[list[int]]]:
    """R's ``attrassign(X, newTerms)``: the labels of ``(Intercept)`` and of every
    non-strata term, with the 0-based design columns of each, in column order.  ``assign``
    is ``attr(X, "assign")`` per column and ``strata_term`` the 1-based strata term (0
    for none)."""

    groups = _column_groups(assign, ("(Intercept)", *term_labels), exclude=strata_term or None)
    return list(groups), list(groups.values())


def fit_penalized(
    assign: Sequence[int],
    term_labels: Sequence[str],
    strata_term: int,
    terms: Sequence[tuple[int, _PenaltyDesignTerm]],
    data: Any,
    distribution: Any,
    *,
    init: Sequence[float] | None,
    scale: float,
    control: Any,
    robust: bool | None,
    nstrat: int | None,
) -> tuple[Any, tuple[str, ...]]:
    """``survpenal.fit`` on a ``SurvregData`` whose design columns are laid out by
    ``assign`` (see :func:`assign_list`), for its ``penalty_terms``: the ``SurvpenalFit``
    and the labels of its ``assign2`` terms (``"sigma"`` last when the scale is
    estimated)."""

    labels, columns = assign_list(assign, term_labels, strata_term)
    pcols = [
        [j for j, assigned in enumerate(assign) if assigned == term_index]
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
        nstrat=nstrat,
    )
    if len(fit.assign2) > len(labels):
        labels.append("sigma")
    return fit, tuple(labels)
