"""Compact penalized AFT fitting on prepared matrices (R's survpenal.fit)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import _survival as _core
from ._coerce import _float_vector, _integer_scalar
from ._survreg import _resolve_control
from ._survreg_lowlevel import _fitting_distribution, _prepared_data
from ._types import PsplineResult


@dataclass(frozen=True)
class _PenaltyPrintInfo:
    kind: str
    distribution: str | None = None
    degree: int = 3
    nterm: int = 0
    boundary: tuple[float, float] | None = None
    intercept: bool = False


@dataclass(frozen=True)
class SurvpenalFitResult:
    """Numerical fit components without retained design, response or callbacks.

    Coefficients include estimated log-scales; sparse frailties are separate.
    Column assignments are zero-based. Numeric properties return copies.
    """

    _fit: _core.SurvpenalFitResult = field(repr=False)
    coefficient_names: tuple[str, ...]
    assign2_labels: tuple[str, ...]
    _print_info: tuple[_PenaltyPrintInfo | None, ...] = field(repr=False)

    @property
    def coefficients(self) -> list[float]:
        return self._fit.coefficients

    @property
    def icoef(self) -> list[float]:
        return self._fit.icoef

    @property
    def var(self) -> list[list[float]]:
        return self._fit.var

    @property
    def var2(self) -> list[list[float]]:
        return self._fit.var2

    @property
    def loglik(self) -> list[float]:
        return self._fit.loglik

    @property
    def iter(self) -> list[int]:
        return self._fit.iter

    @property
    def linear_predictors(self) -> list[float]:
        return self._fit.linear_predictors

    @property
    def df(self) -> list[float]:
        return self._fit.df

    @property
    def df2(self) -> None:
        return None

    @property
    def penalty(self) -> list[float]:
        return self._fit.penalty[1:] if 2 in self._fit.pterms else self._fit.penalty

    @property
    def score(self) -> list[float]:
        return self._fit.score

    @property
    def frail(self) -> list[float] | None:
        return self._fit.frail

    @property
    def fvar(self) -> list[float] | None:
        return self._fit.fvar

    @property
    def pterms(self) -> dict[str, int]:
        return dict(zip(self.assign2_labels, self._fit.pterms, strict=False))

    @property
    def assign2(self) -> dict[str, list[int]]:
        return dict(zip(self.assign2_labels, self._fit.assign2, strict=True))

    @property
    def history(self) -> dict[str, _core.PenaltyHistory]:
        return {self.assign2_labels[value.term]: value for value in self._fit.history}

    @property
    def n(self) -> int:
        return self._fit.n

    @property
    def nvar(self) -> int:
        return self._fit.nvar

    @property
    def n_eff(self) -> float:
        return self._fit.n_eff

    @property
    def scale(self) -> list[float]:
        return self._fit.scale

    @property
    def converged(self) -> bool:
        return self._fit.converged

    @property
    def inner_failures(self) -> list[int]:
        return self._fit.inner_failures


def _column_groups(value: Any, name: str) -> list[list[int]]:
    try:
        groups = [[_integer_scalar(j, name) for j in group] for group in value]
    except TypeError as error:
        raise TypeError(f"{name} must contain sequences of zero-based column indices") from error
    if any(not group or any(j < 0 for j in group) for group in groups):
        raise ValueError(f"{name} must contain nonempty groups of zero-based column indices")
    return groups


def _penalty(value: Any) -> tuple[_core.CoxPenalty, _PenaltyPrintInfo]:
    if isinstance(value, PsplineResult):
        if not value.penalty:
            raise ValueError("pattr requires a penalized spline")
        return _core.CoxPenalty.pspline(
            df=value.df,
            theta=value.theta,
            nterm=value.nterm,
            eps=value.eps,
            method=value.method,
            intercept=value.intercept,
        ), _PenaltyPrintInfo(
            "pspline",
            degree=value.degree,
            nterm=value.nterm,
            boundary=value.boundary_knots,
            intercept=value.intercept,
        )
    if not isinstance(value, _core.CoxPenalty):
        raise TypeError("pattr must contain CoxPenalty or PsplineResult objects")
    return value, _PenaltyPrintInfo(value.kind, value.distribution)


def survpenal_fit(
    x: Any,
    y: Any,
    weights: Any = None,
    offset: Any = None,
    init: Any = None,
    controlvals: Any = None,
    dist: Any = "extreme",
    scale: Any = 0,
    nstrat: Any = 1,
    strata: Any = None,
    pcols: Any = None,
    pattr: Any = None,
    assign: Any = None,
    parms: Any = None,
    *,
    column_names: Sequence[str] | None = None,
) -> SurvpenalFitResult:
    """Fit prepared AFT matrices with ridge, spline, frailty or callback penalties.

    ``pattr`` contains native ``CoxPenalty`` objects or ``pspline`` results;
    ``pcols`` holds their zero-based design columns. ``assign`` maps term names
    to column groups, or is a sequence of groups. It defaults to the penalized
    groups plus each remaining column. Every column must belong to one term.
    Responses and distributions follow ``survreg_fit``: prepared response scale,
    numeric censoring codes, base densities and one-based stratum codes.
    """
    matrix, data, scale_value, count, names = _prepared_data(
        x, y, weights, offset, scale, nstrat, strata, column_names
    )
    if pcols is None or pattr is None:
        raise ValueError("Invalid pcols or pattr arg")
    groups = _column_groups(pcols, "pcols")
    values = list(pattr)
    entries = [_penalty(value) for value in values]
    if not entries or len(entries) != len(groups):
        raise ValueError("Invalid pcols or pattr arg")
    for value, group in zip(values, groups, strict=True):
        if isinstance(value, PsplineResult) and value.n_cols != len(group):
            raise ValueError("pspline basis and pcols must have the same number of columns")
    if assign is None:
        used = {j for group in groups for j in group}
        assignments = sorted([*groups, *([j] for j in range(matrix.shape[1]) if j not in used)])
    else:
        assignments = _column_groups(
            assign.values() if isinstance(assign, Mapping) else assign, "assign"
        )
    if isinstance(assign, Mapping):
        labels = list(assign)
        if any(not isinstance(label, str) for label in labels):
            raise TypeError("assign names must be strings")
    else:
        # Validate bounds before using column names. The native layer checks
        # term coverage, duplicate columns and penalty/assignment agreement.
        if any(j >= len(names) for group in assignments for j in group):
            raise ValueError("assign refers to a column outside x")
        labels = [
            names[group[0]] if len(group) == 1 else f"term {i + 1}"
            for i, group in enumerate(assignments)
        ]
    if len(set(labels)) != len(labels) or (scale_value == 0 and "sigma" in labels):
        raise ValueError("assign names must be unique and reserve sigma for estimated scales")
    result = _core.survpenal_fit_raw(
        data,
        _fitting_distribution(dist, parms),
        [value[0] for value in entries],
        groups,
        assign=assignments,
        init=None if init is None else _float_vector(init, "init"),
        scale=scale_value,
        control=_resolve_control(controlvals, {}),
        nstrat=count,
    )
    info = {tuple(group): entry[1] for group, entry in zip(groups, entries, strict=True)}
    print_info = tuple(info.get(tuple(group)) for group in assignments)
    sparse = {group[0] for group, entry in zip(groups, entries, strict=True) if entry[0].sparse}
    names = tuple(name for j, name in enumerate(names) if j not in sparse)
    if scale_value == 0:
        names += ("Log(scale)",) * count
        labels.append("sigma")
    return SurvpenalFitResult(result, names, tuple(labels), print_info)
