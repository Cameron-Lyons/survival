"""The remaining R functions of survival: ``survcheck``, ``survobrien``, ``royston``, ``brier``,
``cipoisson``, the bounded links, ``nsk``, ``pspline`` and ``statefig``.

Each function does what the R code of its namesake does (argument checking, the model frame,
labelling of the result) and calls the Rust kernel of the same name for the numbers.  The
Cox-model functions (``royston``, ``brier``) read R's ``coxph`` components from the
Rust fit behind the result (``coefficients``, ``var``, ``means``, ``loglik``, ``nevent``,
``linear_predictors``, ``time``/``status``/``entry`` = ``fit$y``, ``weights``, ``x``,
``strata``, ``offset``, ``method``) and use the R generics of ``_models`` (``coef``, ``vcov``,
``model_frame``, ``model_formula``, ``predict``) plus the formula design of the fit for
``model.matrix`` on new or population data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _survival as _core
from . import _models
from ._coerce import (
    _coerce_array_like,
    _finite_float,
    _float_vector,
    _int_vector,
    _integer_scalar,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _normalize_bool_option,
    _normalize_na_action,
    _r_factor_levels,
    _scalar_or_vector,
    _subset_indices,
    _subset_sequence,
)
from ._coxph import _has_strata, _prediction_newdata
from ._coxphms import CoxphmsModel
from ._fit import _formula_design_for_fit
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _column_source,
    _covariate_term_columns,
    _covariate_term_name,
    _formula_columns,
    _parse_formula,
    _strata_keep,
    _strata_specs,
    _subset_formula_inputs,
    _term_values,
    _timeline_counting,
    _timeline_response,
)
from ._models import model_formula
from ._names import _make_names_unique, _make_unique
from ._penalties import _combine_basis, _pspline_boundary, _pspline_cbase, _pspline_combine
from ._surv import Surv, _complete_codes, _subset_surv
from ._types import (
    _MISSING,
    BrierResult,
    PsplineResult,
    StateFigResult,
    SurvCheckCodes,
    SurvCheckProblem,
    SurvCheckResult,
    _InteractionTerm,
)

# ---------------------------------------------------------------------------
# Access to a fitted coxph object
# ---------------------------------------------------------------------------


def _coxph_engine(fit: Any, message: str) -> Any:
    """The Rust ``CoxPHFit`` behind an R-style coxph result (R's ``inherits(fit, "coxph")``)."""

    engine = fit if isinstance(fit, _core.CoxPHFit) else getattr(fit, "fit", None)
    if not isinstance(engine, _core.CoxPHFit):
        raise TypeError(message)
    return engine


def _fit_response(engine: Any) -> Surv:
    """``fit$y``: the (timefixed) response the Cox model was fitted to."""

    if engine.entry is None:
        return Surv(list(engine.time), list(engine.status))
    return Surv(list(engine.entry), list(engine.time), list(engine.status))


def _newdata_response(fit: Any, newdata: Any) -> Surv:
    """``model.response(model.frame(formula(fit), data = newdata))``."""

    response, _terms = _parse_formula(model_formula(fit), newdata)
    return response


def _call_column(fit: Any, name: str, newdata: Any, n: int) -> list[Any] | None:
    """A ``weights=``/``id=`` argument of the original call, re-evaluated on ``newdata``.

    The fit records the column name the argument referred to (``weights_column``,
    ``id_column``); an argument given as a vector cannot be re-evaluated, as in R.
    """

    column = getattr(fit, f"{name}_column", None)
    if column is None:
        return None
    try:
        values = _materialize_labels(_column(newdata, column), name)
    except KeyError as exc:
        raise ValueError(f"newdata is missing the {name} column {column!r}") from exc
    if len(values) != n:
        raise ValueError(f"wrong length for {name}")
    return values


def _unique_in_order(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


# ---------------------------------------------------------------------------
# statefig
# ---------------------------------------------------------------------------


def _statefig_state_names(connect: Any, states: Any | None) -> list[str]:
    if states is not None:
        return [str(value) for value in _materialize_1d(states, "states")]
    if isinstance(connect, Mapping):
        return [str(value) for value in connect]
    if hasattr(connect, "columns"):  # a data frame: its row labels
        return [str(value) for value in connect.index]
    raise ValueError("connect must have the state names as dimnames")


def _statefig_layout(layout: Any) -> dict[str, Any]:
    """Interpret R's ``layout``: a vector of box counts, a one-column matrix (top to bottom), a
    two-column matrix of coordinates, or any other matrix read column-wise as a vector."""

    rows = _coerce_array_like(layout, "layout")
    if not rows:
        raise ValueError("layout must be a numeric vector or matrix")
    is_matrix = isinstance(rows[0], list | tuple)
    matrix = [[float(v) for v in row] for row in rows] if is_matrix else [[float(v) for v in rows]]
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("layout must be a numeric vector or matrix")
    if is_matrix and width == 2 and len(matrix) > 1:
        return {"coordinates": matrix}
    counts = [row[column] for column in range(width) for row in matrix]
    if any(value <= 0.0 or value != math.floor(value) for value in counts):
        raise ValueError("non-integer number of states in layout argument")
    return {"layout": [int(value) for value in counts], "column": is_matrix and width == 1}


def statefig(
    layout: Any,
    connect: Any,
    states: Any | None = None,
    margin: Any = 0.03,
    box: Any = True,
    cex: Any = 1,
    col: Any = 1,
    lwd: Any = 1,
    lty: Any = 1,
    bcol: Any | None = None,
    acol: Any | None = None,
    alwd: Any | None = None,
    alty: Any | None = None,
    offset: Any = 0,
) -> StateFigResult:
    """Box coordinates and arrows of R's ``statefig`` state-space figure.

    R draws the figure and returns the box centres invisibly; this returns them as
    ``positions`` together with the arrows.  The graphical parameters are accepted for
    compatibility with R's signature.  ``states`` names the states when ``connect`` is not a
    mapping keyed by state name (R's ``dimnames``).
    """

    del box, cex, col, lwd, lty, bcol, acol, alwd, alty
    _finite_float(margin, "margin")
    _finite_float(offset, "offset")
    if isinstance(connect, Mapping):
        rows = [[float(v) for v in _materialize_1d(row, "connect")] for row in connect.values()]
    else:
        rows = [[float(v) for v in row] for row in _coerce_array_like(connect, "connect")]
    figure = _core.statefig(
        rows, _statefig_state_names(connect, states), **_statefig_layout(layout)
    )
    return StateFigResult(
        states=list(figure.states),
        positions=list(zip(figure.x, figure.y, strict=True)),
        arrows=list(figure.arrows),
    )


# ---------------------------------------------------------------------------
# cipoisson and the bounded links
# ---------------------------------------------------------------------------


def _numeric_or_nan(values: Any, name: str) -> list[float]:
    return [math.nan if _is_missing_value(v) else float(v) for v in _scalar_or_vector(values, name)]


def cipoisson(
    k: Any, time: Any = 1, p: Any = 0.95, method: Any = "exact"
) -> tuple[float, float] | list[tuple[float, float]]:
    """Confidence limits for Poisson rates, like R's ``cipoisson``.

    ``k``, ``time`` and ``p`` recycle like R vectors.  As in R the result is one
    ``(lower, upper)`` pair when every argument has length one and otherwise one pair per
    element (R's two-column matrix).
    """

    method_value = _match_string_arg(method, "method", ["exact", "anscombe"], "Invalid method")
    limits = _core.cipoisson(
        _numeric_or_nan(k, "k"),
        _numeric_or_nan(time, "time"),
        _numeric_or_nan(p, "p"),
        method_value,
    )
    pairs = list(zip(limits.lower, limits.upper, strict=True))
    return pairs[0] if len(pairs) == 1 else pairs


def _bounded_link(x: Any, edge: Any, name: str) -> float | list[float]:
    link = _core.LinkFunctionParams(_finite_float(edge, "edge"))
    transform = getattr(link, f"{name}_many")
    try:
        values = _materialize_1d(x, "x")
    except TypeError:
        return float(transform([None if _is_missing_value(x) else float(x)])[0])
    return [
        float(v) for v in transform([None if _is_missing_value(v) else float(v) for v in values])
    ]


def blogit(x: Any, edge: Any = 0.05, *, inverse: bool = False) -> float | list[float]:
    """R's ``blogit(edge)$linkfun``: the logit of ``x`` bounded away from 0 and 1."""

    return _bounded_link(x, edge, "blogit_inverse" if inverse else "blogit")


def bprobit(x: Any, edge: Any = 0.05, *, inverse: bool = False) -> float | list[float]:
    """R's ``bprobit(edge)$linkfun``: the probit of ``x`` bounded away from 0 and 1."""

    return _bounded_link(x, edge, "bprobit_inverse" if inverse else "bprobit")


def bcloglog(x: Any, edge: Any = 0.05, *, inverse: bool = False) -> float | list[float]:
    """R's ``bcloglog(edge)$linkfun``: the complementary log-log of bounded ``x``."""

    return _bounded_link(x, edge, "bcloglog_inverse" if inverse else "bcloglog")


def blog(x: Any, edge: Any = 0.05, *, inverse: bool = False) -> float | list[float]:
    """R's ``blog(edge)$linkfun``: the log of ``x`` bounded below by ``edge``."""

    return _bounded_link(x, edge, "blog_inverse" if inverse else "blog")


# ---------------------------------------------------------------------------
# nsk and pspline
# ---------------------------------------------------------------------------


def _quantile_type7(sorted_values: list[float], probability: float) -> float:
    """R's default ``quantile(x, p)`` (type 7)."""

    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _nsk_boundary(x: list[float], knots: list[float], b: Any, boundary: Any) -> tuple[float, float]:
    """R's ``Boundary.knots`` handling in ``nsk``: the default quantiles, ``TRUE`` for the
    range, ``FALSE``/``NULL`` for the outer knots, or an explicit pair.  The Rust basis widens
    the pair to enclose the knots as ``nsk.R`` does."""

    if boundary is _MISSING:
        b_value = _finite_float(b, "b")
        ordered = sorted(x)
        return (_quantile_type7(ordered, b_value), _quantile_type7(ordered, 1.0 - b_value))
    if isinstance(boundary, bool):
        boundary = (min(x), max(x)) if boundary else None
    if boundary is None:
        if len(knots) < 2:
            raise ValueError("wrong length for Boundary.knots")
        return (knots[0], knots[-1])
    values = _float_vector(boundary, "Boundary.knots")
    if len(values) != 2:
        raise ValueError("wrong length for Boundary.knots")
    return (values[0], values[1])


def nsk(
    x: Any,
    df: Any | None = None,
    knots: Any | None = None,
    intercept: Any = False,
    b: Any = 0.05,
    Boundary_knots: Any = _MISSING,
) -> _core.SplineBasisResult:
    """Natural spline basis whose coefficients are the values at the knots (R's ``nsk``).

    ``Boundary_knots`` is R's ``Boundary.knots``: the ``b``/``1 - b`` quantiles of ``x`` by
    default, ``True`` for the range of ``x``, ``False``/``None`` for the outer ``knots``.  Missing
    ``x`` values give rows of ``NaN``.
    """

    x_values = _numeric_or_nan(x, "x")
    observed = [value for value in x_values if not math.isnan(value)]
    if not observed:
        raise ValueError("x must contain at least one non-missing value")
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("x must contain only finite values")
    knot_values = sorted(set(_float_vector(knots, "knots"))) if knots is not None else []
    boundary = _nsk_boundary(observed, knot_values, b, Boundary_knots)
    if not boundary[0] < boundary[1]:
        raise ValueError("Boundary.knots must be finite and strictly increasing")
    spline = _core.NaturalSplineKnot(
        knot_values or None,
        boundary,
        None if df is None else _integer_scalar(df, "df"),
        _normalize_bool_option(intercept, "intercept"),
    )
    return spline.basis(x_values)


def _pspline_method(
    df: Any, theta: Any | None, nterm: Any | None, eps: Any | None, method: Any | None
) -> tuple[int | float, float | None, int, float, str]:
    """The ``fixed``/``aic``/``df`` selection at the top of R's ``pspline``."""

    df_value = _finite_float(df, "df")
    df_value = int(df_value) if df_value.is_integer() else df_value
    nterm_value = None if nterm is None else int(round(_finite_float(nterm, "nterm")))
    eps_value = None if eps is None else _finite_float(eps, "eps")
    if theta is not None:
        theta_value = _finite_float(theta, "theta")
        if theta_value <= 0.0 or theta_value >= 1.0:
            raise ValueError("Invalid value for theta")
        nterm_value = int(round(2.5 * df_value)) if nterm_value is None else nterm_value
        return df_value, theta_value, nterm_value, 0.1 if eps_value is None else eps_value, "fixed"
    if df_value == 0 or (method is not None and str(method) == "aic"):
        return df_value, None, 15, 1e-5 if eps_value is None else eps_value, "aic"
    if df_value <= 1:
        raise ValueError("Too few degrees of freedom")
    nterm_value = int(round(2.5 * df_value)) if nterm_value is None else nterm_value
    if df_value > nterm_value:
        raise ValueError(f"`nterm' too small for df={df_value:g}")
    return df_value, None, nterm_value, 0.1 if eps_value is None else eps_value, "df"


def _second_difference_penalty(nvar: int) -> list[list[float]]:
    """R's ``t(D) %*% D`` for the second-difference matrix ``D`` of ``nvar`` coefficients."""

    penalty = [[0.0] * nvar for _ in range(nvar)]
    # Each difference row has only three nonzero entries. Accumulate its
    # outer product directly instead of multiplying dense matrices in Python.
    for i in range(nvar - 2):
        for j, left in enumerate((1.0, -2.0, 1.0)):
            for k, right in enumerate((1.0, -2.0, 1.0)):
                penalty[i + j][i + k] += left * right
    return penalty


def pspline(
    x: Any,
    df: Any = 4,
    theta: Any | None = None,
    nterm: Any | None = None,
    degree: Any = 3,
    eps: Any | None = None,
    method: Any | None = None,
    Boundary_knots: Any | None = None,
    intercept: Any = False,
    penalty: Any = True,
    combine: Any | None = None,
    *,
    boundary_knots: Any | None = None,
) -> PsplineResult:
    """The P-spline basis of R's ``pspline`` term with its penalty attributes.

    ``Boundary_knots`` is R's ``Boundary.knots`` (the range of ``x`` by default); the
    keyword ``boundary_knots`` is the same argument under the name the R bridge uses.  The
    smoothing-parameter control functions live in the Cox fitter; ``method`` records which one
    R would use (``fixed`` for a given ``theta``, ``aic`` for ``df = 0``, else ``df``).
    """

    if Boundary_knots is not None and boundary_knots is not None:
        raise ValueError("use only one of Boundary_knots or boundary_knots")
    boundary_arg = Boundary_knots if boundary_knots is None else boundary_knots
    df_value, theta_value, nterm_value, eps_value, method_value = _pspline_method(
        df, theta, nterm, eps, method
    )
    x_values = _numeric_or_nan(x, "x")
    observed = [value for value in x_values if not math.isnan(value)]
    if not observed:
        raise ValueError("x must contain at least one non-missing value")
    if nterm_value < 3:
        raise ValueError("Too few basis functions")
    boundary = _pspline_boundary(boundary_arg, observed)
    intercept_value = _normalize_bool_option(intercept, "intercept")
    basis = _core.pspline_basis(x_values, nterm_value, _integer_scalar(degree, "degree"), boundary)

    matrix = [list(row) for row in basis.basis]
    combine_codes = None
    if combine is not None:
        groups = _pspline_combine(combine, len(matrix[0]), intercept_value)
        matrix = _combine_basis(matrix, groups)
        combine_codes = list(groups if intercept_value else groups[1:])
    nvar = len(matrix[0])
    dmat = _second_difference_penalty(nvar)
    if not intercept_value:
        matrix = [row[1:] for row in matrix]
        dmat = [row[1:] for row in dmat[1:]]
    return PsplineResult(
        basis=matrix,
        knots=list(basis.knots),
        nterm=basis.nterm,
        degree=basis.degree,
        boundary_knots=basis.boundary_knots,
        intercept=intercept_value,
        penalty=_normalize_bool_option(penalty, "penalty"),
        df=df_value,
        eps=eps_value,
        method=method_value,
        dmat=dmat,
        cbase=_pspline_cbase(basis.nterm, basis.degree, boundary, nvar),
        theta=theta_value,
        combine=combine_codes,
    )


def predict_pspline(object: PsplineResult, newx: Any = _MISSING, **_kwargs: Any) -> PsplineResult:
    """Evaluate a P-spline at ``newx`` using its original basis settings.

    Omitting ``newx`` returns ``object`` unchanged. New values use the stored
    boundaries, degree, number of terms, intercept, and column combinations;
    extrapolation is linear beyond the boundaries. The returned basis has
    ``penalty=False``, as in R's ``predict.pspline``.
    """

    if not isinstance(object, PsplineResult):
        raise TypeError("predict_pspline requires a PsplineResult")
    if newx is _MISSING:
        return object
    return pspline(
        newx,
        nterm=object.nterm,
        degree=object.degree,
        boundary_knots=object.boundary_knots,
        intercept=object.intercept,
        combine=object.combine,
        penalty=False,
    )


@_models._predict.register(PsplineResult)
def _predict_pspline(
    object: PsplineResult, newdata: Any | None = None, **kwargs: Any
) -> PsplineResult:
    newx = kwargs.pop("newx", _MISSING)
    if newdata is not None:
        if newx is not _MISSING:
            raise ValueError("use only one of newdata or newx")
        newx = newdata
    return predict_pspline(object, newx, **kwargs)


def _frailty_encoding(
    x: Any,
    *,
    levels: Any | None = None,
    sparse: Any | None = None,
) -> dict[str, Any]:
    """The ``as.factor`` step of R's ``frailty`` terms: 1-based codes, levels and the sparse
    default ``nclass > 5`` (used by the R bridge)."""

    values = _materialize_labels(x, "x")
    level_values = (
        [str(v) for v in _materialize_1d(levels, "levels")]
        if levels is not None
        else sorted({str(v) for v in values if not _is_missing_value(v)})
    )
    level_index = {level: idx + 1 for idx, level in enumerate(level_values)}
    codes: list[int | None] = []
    for value in values:
        if _is_missing_value(value):
            codes.append(None)
        elif str(value) not in level_index:
            raise ValueError(f"x contains value {str(value)!r} outside supplied levels")
        else:
            codes.append(level_index[str(value)])
    sparse_value = (
        len(level_values) > 5 if sparse is None else _normalize_bool_option(sparse, "sparse")
    )
    return {
        "codes": codes,
        "levels": level_values,
        "nclass": len(level_values),
        "sparse": sparse_value,
    }


# ---------------------------------------------------------------------------
# The model frame of survcheck
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelFrame:
    response: Surv
    extras: dict[str, list[Any] | None]
    kept: list[int]
    omitted: list[int]


def _missing_rows(frame: Mapping[str, Any], n: int) -> set[int]:
    """Rows with a missing value in any column; a ``Surv`` column is missing where R's ``Surv``
    made it ``NA`` (missing time or status)."""

    rows: set[int] = set()
    for name, values in frame.items():
        if isinstance(values, Surv):
            rows.update(idx for idx, event in enumerate(values.event) if event is None)
            rows.update(idx for idx, time in enumerate(values.time) if math.isnan(time))
            if values.start is not None:
                rows.update(idx for idx, time in enumerate(values.start) if math.isnan(time))
        else:
            rows.update(_missing_row_indices([(name, values)], n))
    return rows


def _take_rows(frame: Mapping[str, Any], rows: list[int]) -> dict[str, Any]:
    return {
        name: _subset_surv(values, rows)
        if isinstance(values, Surv)
        else _subset_sequence(values, rows, name)
        for name, values in frame.items()
    }


def _model_frame(
    formula: Any, data: Any, subset: Any, na_action: Any, **extras: Any
) -> _ModelFrame:
    """R's ``model.frame`` for a formula with a ``Surv`` response (or a ``Surv`` object) plus
    row-aligned arguments such as ``id`` and ``istate``, given as vectors or column names.

    ``subset`` is applied first, then ``na.action`` to the formula variables, the response and
    the extras; ``kept`` and ``omitted`` record the 0-based rows of the subset that ``na.omit``
    kept and dropped.
    """

    action = _normalize_na_action(na_action)
    if isinstance(formula, Surv):
        frame: dict[str, Any] = {"(response)": formula}
        n = len(formula)
    else:
        if data is None:
            raise ValueError("a data argument is required to evaluate the formula")
        frame = {name: _column_source(data, name) for name in _formula_columns(formula, data)}
        n = len(_column(data, next(iter(frame))))
    for name in [name for name, values in extras.items() if values is not None]:
        frame[name] = _column_or_values(data, extras[name], name)
        if len(_materialize_labels(frame[name], name)) != n:
            raise ValueError(f"wrong length for {name}")
    if subset is not None:
        frame = _take_rows(frame, _subset_indices(subset, n))
        n = len(next(iter(frame.values())))
    omitted: list[int] = []
    kept = list(range(n))
    # na.omit before building the response (a missing variable), then on the response itself
    for build_response in (True, False):
        missing = _missing_rows(frame, len(kept))
        if missing and action == "fail":
            raise ValueError("missing values in object")
        if missing and action in {"omit", "exclude"}:
            rows = [idx for idx in range(len(kept)) if idx not in missing]
            omitted.extend(kept[idx] for idx in missing)
            kept = [kept[idx] for idx in rows]
            frame = _take_rows(frame, rows)
        if build_response and not isinstance(formula, Surv):
            frame["(response)"] = _parse_formula(formula, frame)[0]
    return _ModelFrame(
        response=frame["(response)"],
        extras={
            name: None if name not in frame else _materialize_labels(frame[name], name)
            for name in extras
        },
        kept=kept,
        omitted=sorted(omitted),
    )


# ---------------------------------------------------------------------------
# survcheck
# ---------------------------------------------------------------------------


def _survcheck_states(response: Surv) -> list[str]:
    """The state names R forces onto the response (``event`` for 0/1 data)."""

    if response.type in {"right", "counting"}:
        return ["event"]
    if response.type not in {"mright", "mcounting"}:
        raise ValueError("response must be right censored")
    return list(response.states)


def _survcheck_problem(
    problem: Any, row_numbers: list[int], id_levels: list[Any]
) -> SurvCheckProblem | None:
    if problem is None:
        return None
    return SurvCheckProblem(
        row=[row_numbers[row] for row in problem.row],
        id=[id_levels[code - 1] for code in problem.id],
    )


def _survcheck_codes(
    id: Any, time2: Any, status: Any, time1: Any | None = None, istate: Any | None = None
) -> SurvCheckCodes:
    """``survcheck``'s kernel for a response given as integer codes, the R bridge's entry
    point after it evaluates the model frame in R: ``id`` as ``match(id, unique(id))``,
    ``status`` as ``0`` (censored) or the code of the target state and ``istate`` as codes of
    the same states.  The states are only known by their codes, so they are named after them.
    """

    status_codes = _int_vector(status, "status")
    istate_codes = None if istate is None else _int_vector(istate, "istate")
    n_states = max([0, *status_codes, *(istate_codes or [])])
    states = [str(code) for code in range(1, n_states + 1)]
    raw = _core.survcheck(
        _int_vector(id, "id"),
        _float_vector(time2, "time2"),
        status_codes,
        states,
        time1=None if time1 is None else _float_vector(time1, "time1"),
        istate=istate_codes,
        istate_levels=None if istate_codes is None else states,
        timefix=False,
    )

    def rows(problem: Any) -> list[int]:
        return [] if problem is None else list(problem.row)

    return SurvCheckCodes(
        current_states=[0 if state == "(s0)" else int(state) for state in raw.istate],
        overlap_rows=rows(raw.overlap),
        gap_rows=rows(raw.gap),
        jump_rows=rows(raw.jump),
        teleport_rows=rows(raw.teleport),
        n_transitions=raw.n_transitions,
    )


def survcheck(
    formula: Any = _MISSING,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "na.omit",
    id: Any | None = None,
    istate: Any | None = None,
    istate0: str = "(s0)",
    timefix: bool = True,
) -> SurvCheckResult:
    """Consistency checks of (multi-state) survival data, like R's ``survcheck``.

    ``formula`` is ``Surv(...) ~ ...`` evaluated in ``data`` (or a ``Surv`` object); ``id`` and
    ``istate`` are column names of ``data`` or vectors.  Problem rows are reported as 1-based
    row numbers of ``data`` after ``subset``, as R does (of the counting-process rows for
    ``Surv2`` timeline data, which is converted first).
    """

    if formula is _MISSING:
        raise ValueError("a formula argument is required")
    if not isinstance(timefix, bool):
        raise ValueError("invalid value for timefix option")
    extras = {"id": id, "istate": istate}
    if _timeline_response(formula):
        # survcheck.R converts timeline data before its na.action
        formula, data, extras = _timeline_counting(formula, data, subset, extras)
        subset = None
    frame = _model_frame(formula, data, subset, na_action, **extras)
    response = frame.response
    n = len(response)
    if n == 0:
        raise ValueError("No (non-missing) observations")
    states = _survcheck_states(response)
    id_values = frame.extras["id"]
    if id_values is None:
        raise ValueError("an id argument is required")
    if len(id_values) != n:
        raise ValueError("wrong length for id")
    istate_values = frame.extras["istate"]
    if istate_values is not None and len(istate_values) != n:
        raise ValueError("wrong length for istate")

    id_levels = _unique_in_order(id_values)
    id_codes = {value: code for code, value in enumerate(id_levels, start=1)}
    istate_levels = None if istate_values is None else _r_factor_levels(istate_values)
    raw = _core.survcheck(
        [id_codes[value] for value in id_values],
        list(response.time),
        [int(event) for event in response.event],
        states,
        time1=None if response.start is None else list(response.start),
        istate=None
        if istate_levels is None
        else [istate_levels.index(value) + 1 for value in istate_values],
        istate_levels=None if istate_levels is None else [str(v) for v in istate_levels],
        istate0=istate0,
        censor_label=response.clabel if response.clabel is not None else "censor",
        timefix=timefix,
    )
    # R reports rows of the data before missing values were removed.
    row_numbers = [idx + 1 for idx in frame.kept]
    return SurvCheckResult(
        states=raw.states,
        transitions=raw.transitions,
        events=raw.events,
        flag=raw.flag,
        istate=raw.istate,
        n_id=raw.n_id,
        n_observations=raw.n_observations,
        n_transitions=raw.n_transitions,
        overlap=_survcheck_problem(raw.overlap, row_numbers, id_levels),
        gap=_survcheck_problem(raw.gap, row_numbers, id_levels),
        jump=_survcheck_problem(raw.jump, row_numbers, id_levels),
        teleport=_survcheck_problem(raw.teleport, row_numbers, id_levels),
        y=response,
        id=list(id_values),
        na_action=[idx + 1 for idx in frame.omitted] or None,
    )


# ---------------------------------------------------------------------------
# survobrien
# ---------------------------------------------------------------------------


def _survobrien_columns(
    data: Any, covariates: Sequence[Any], n: int
) -> tuple[list[str], list[tuple[str, list[float]]]]:
    """Split the model terms into the ones R leaves alone (``keepers <- factors | protected``:
    factors, non-numeric terms and ``I()`` (AsIs) terms) and the continuous ones it transforms.

    A kept term contributes every data column it references (R's ``all.vars``), once per term.
    """

    keepers: list[str] = []
    continuous: list[tuple[str, list[float]]] = []
    for term in covariates:
        if isinstance(term, _InteractionTerm):
            raise ValueError("This function cannot deal with iteraction terms")
        numeric = None
        if term.transform != "I" and not term.categorical:
            try:
                numeric = [float(value) for value in _term_values(data, term, n)]
            except (TypeError, ValueError):
                numeric = None
        if numeric is None:
            keepers.extend(_covariate_term_columns(term))
        else:
            continuous.append((_covariate_term_name(term), numeric))
    if not continuous:
        raise ValueError("No continuous variables to modify")
    return keepers, continuous


def _survobrien_transformed(
    transform: Callable[..., Any] | None,
    continuous: list[tuple[str, list[float]]],
    expansion: Any,
) -> list[tuple[str, list[float]]]:
    """The transformed columns: the Rust logit-rank default, or ``transform`` applied to the
    values of every risk set (R's ``lapply(indx, function(x) transform(z[x]))``)."""

    if transform is None:
        return [
            (name, list(column))
            for (name, _values), column in zip(continuous, expansion.transformed, strict=True)
        ]
    blocks: dict[int, list[int]] = {}
    for position, block in enumerate(expansion.strata):
        blocks.setdefault(block, []).append(position)
    out: list[tuple[str, list[float]]] = []
    for name, values in continuous:
        column = [0.0] * len(expansion.row)
        for positions in blocks.values():
            transformed = transform([values[expansion.row[p]] for p in positions])
            if isinstance(transformed, int | float):
                # a length-one R vector comes back from reticulate as a scalar
                transformed = [transformed]
            result = _float_vector(transformed, "transform")
            if len(result) != len(positions):
                raise ValueError("Transform function must be 1 to 1")
            for position, value in zip(positions, result, strict=True):
                column[position] = value
        out.append((name, column))
    return out


def survobrien(
    formula: str,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "na.omit",
    transform: Callable[..., Any] | None = None,
) -> dict[str, list[Any]]:
    """O'Brien's logit-rank expansion of a data set, like R's ``survobrien``.

    Returns the expanded data frame (a mapping of columns): the response, the untransformed
    variables of the factor and ``I()`` terms, the ``strata`` and ``cluster`` columns (or
    ``.id.``, the source row), the transformed continuous variables and the risk-set number
    ``.strata.``.  The column names are made syntactic and unique as R's ``data.frame`` does
    (``log(z)`` becomes ``log.z.``, a repeated ``z`` becomes ``z.1``).  String columns count
    as factors.
    """

    if (
        transform is not None
        and len(_materialize_1d(transform(list(range(1, 11))), "transform")) != 10
    ):
        raise ValueError("Transform function must be 1 to 1")
    if data is None:
        raise ValueError("a data argument is required to evaluate the formula")
    if subset is not None:
        data, _aligned = _subset_formula_inputs(formula, data, subset)
    data, _aligned, _removed = _apply_formula_na_action(formula, data, na_action)
    response, terms = _parse_formula(formula, data)
    n = len(response)
    if response.type not in {"right", "counting"}:
        raise ValueError("Response must be right censored or (start, stop] data")
    keepers, continuous = _survobrien_columns(data, terms.covariates, n)
    strata_codes = None
    if terms.strata:
        strata_codes = _complete_codes(
            _strata_keep(data, _strata_specs(terms)), "missing values in the strata"
        )
    expansion = _core.survobrien(
        list(response.time),
        [int(event) for event in response.event],
        [values for _name, values in continuous],
        start=None if response.start is None else list(response.start),
        strata=strata_codes,
    )
    rows = list(expansion.row)
    columns: list[tuple[str, list[Any]]] = []
    if expansion.start is not None:
        columns += [("start", list(expansion.start)), ("stop", list(expansion.time))]
    else:
        columns.append(("time", list(expansion.time)))
    columns.append(("status", list(expansion.status)))
    # data[knames]: `[.data.frame` makes the names of the kept and strata variables unique
    knames = [*keepers, *terms.strata]
    columns += [
        (label, list(_subset_sequence(_column(data, name), rows, name)))
        for label, name in zip(
            [*_make_unique(knames), *terms.clusters], [*knames, *terms.clusters], strict=True
        )
    ]
    if not terms.clusters:
        columns.append((".id.", [row + 1 for row in rows]))
    columns += _survobrien_transformed(transform, continuous, expansion)
    columns.append((".strata.", list(expansion.strata)))
    # data.frame()'s check.names: make.names(unique = TRUE)
    names = _make_names_unique([name for name, _values in columns])
    return {name: values for name, (_label, values) in zip(names, columns, strict=True)}


# ---------------------------------------------------------------------------
# royston
# ---------------------------------------------------------------------------


def royston(
    fit: Any, newdata: Any | None = None, ties: Any = True, adjust: Any = False
) -> dict[str, float]:
    """Royston and Sauerbrei's D and the related R-squared measures of a Cox model.

    Returns R's named vector as a dict: ``D``, ``se(D)``, ``R.D``, ``R.KO``, ``R.N`` and
    ``C.GH``; ``R.N`` (Nagelkerke's R-squared) is not defined for ``newdata`` and is omitted
    then, as in R.
    """

    if isinstance(fit, CoxphmsModel):
        raise ValueError("not defined for multi-state models")
    engine = _coxph_engine(fit, "function defined only for coxph models")
    ties_value = _normalize_bool_option(ties, "ties")
    adjust_value = _normalize_bool_option(adjust, "adjust")
    if newdata is None:
        eta = list(engine.linear_predictors)
        response = _fit_response(engine)
    else:
        if engine.strata is not None:
            raise ValueError("cannot use newdata for a stratified model")
        response = _newdata_response(fit, newdata)
        eta = _float_vector(_models.predict(fit, newdata, type="lp"), "eta")
        # rescale: eta <- coxph(y2 ~ eta)$linear.predictor
        eta = list(
            _core.coxph_fit(
                list(response.time),
                [int(event) for event in response.event],
                [[value] for value in eta],
                entry=None if response.start is None else list(response.start),
            ).linear_predictors
        )
    result = _core.royston(
        eta,
        list(response.time),
        [int(event) for event in response.event],
        list(engine.loglik),
        engine.nevent,
        len(engine.coefficients),
        entry_times=None if response.start is None else list(response.start),
        ties=ties_value,
        adjust=adjust_value,
    )
    values = {"D": result.d, "se(D)": result.se_d, "R.D": result.r_d, "R.KO": result.r_ko}
    if newdata is None:
        values["R.N"] = result.r_n
    values["C.GH"] = result.c_gh
    return values


# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------


def _brier_weights(weights: Sequence[Any] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    try:
        values = [float(value) for value in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    return values


def _brier_is_simple(response: Surv, id_values: Sequence[Any] | None) -> bool:
    """R's ``survcheck2`` gate: no data problems, one starting state and a common entry time."""

    if id_values is None:
        return True
    levels = _unique_in_order(id_values)
    codes = {value: code for code, value in enumerate(levels, start=1)}
    check = _core.survcheck(
        [codes[value] for value in id_values],
        list(response.time),
        [int(event) for event in response.event],
        list(response.states) or ["event"],
        time1=None if response.start is None else list(response.start),
    )
    flags = check.flag
    if flags.overlap or flags.gap or flags.jump or flags.teleport or flags.duplicate:
        raise ValueError("one or more flags are >0 in survcheck")
    n_startstate = sum(1 for row in check.transitions.counts if row and row[0] > 1)
    if response.start is None:
        return n_startstate == 1
    first_entry: dict[Any, float] = {}
    for value, start in zip(id_values, response.start, strict=True):
        first_entry[value] = min(first_entry.get(value, math.inf), start)
    entries = list(first_entry.values())
    return n_startstate == 1 and all(entry == entries[0] for entry in entries)


def _newdata_design(
    fit: Any, newdata: Any
) -> tuple[list[list[float]], list[int] | None, list[float] | None]:
    """``model.matrix`` of the fit's terms on ``newdata`` with its strata codes (coded as the
    fit's ``strata.keep``) and offsets, as ``survfit.coxph(fit, newdata)`` builds them."""

    if _formula_design_for_fit(fit) is None:
        raise TypeError("newdata requires a model fitted from a formula")
    stratified = _has_strata(fit)
    new = _prediction_newdata(
        fit, newdata, need_strata=stratified, need_response=False, na_action="na.fail"
    )
    if stratified and new.strata is None:
        raise ValueError("New data must contain the strata variable(s) of the model")
    return new.x, new.strata, new.offset


def _brier_model_predictions(
    fit: Any, engine: Any, newdata: Any | None, times: list[float]
) -> np.ndarray:
    """``1 - summary(survfit(fit, newdata), times, extend = TRUE)$surv``: one row per time.

    Evaluate the Cox baseline at the requested times before expanding the observations.
    Training rows stay in Rust; new data use the fit's formula design and strata codes.
    """

    if newdata is None:
        probabilities = engine.predict_survival_at(times)
    else:
        rows, strata, offsets = _newdata_design(fit, newdata)
        probabilities = engine.predict_survival_at(
            times, newdata=rows, new_strata=strata, new_offset=offsets
        )
    np.subtract(1.0, probabilities, out=probabilities)
    return probabilities


def brier(
    fit: Any,
    times: Any | None = None,
    newdata: Any | None = None,
    ties: Any = True,
    detail: Any = False,
    timefix: Any = True,
    efron: Any = False,
) -> BrierResult:
    """Brier score of a Cox model with inverse-probability-of-censoring weights (R's ``brier``)."""

    if isinstance(fit, CoxphmsModel):
        # R fails later with "times contains missing or infinite values"
        raise ValueError("brier is not defined for multi-state coxph fits")
    engine = _coxph_engine(fit, "fit must be a coxph object")
    if not isinstance(timefix, bool):
        raise ValueError("invalid value for timefix option")
    ties_value = _normalize_bool_option(ties, "ties")
    if newdata is None:
        response = _fit_response(engine)
        weights = _brier_weights(engine.weights, len(response))
        id_values = getattr(fit, "id", None)
    else:
        response = _newdata_response(fit, newdata)
        n = len(response)
        weights = _brier_weights(_call_column(fit, "weights", newdata, n), n)
        id_values = _call_column(fit, "id", newdata, n)
    if response.type not in {"right", "mright", "counting", "mcounting"}:
        raise ValueError("response must be right censored")
    if response.start is not None and id_values is None:
        raise ValueError("id is required for start-stop data")
    if id_values is not None:
        id_values = _materialize_labels(id_values, "id")
    if not _brier_is_simple(response, id_values):
        raise ValueError("delayed entry is not yet implemented")

    use_efron = _normalize_bool_option(efron, "efron") and engine.method == _core.TieMethod.Efron
    dtime = list(response.time)
    dstat = [int(event) for event in response.event]
    if times is None:
        null_curve = _core.survfitkm(
            dtime,
            dstat,
            start=None if response.start is None else list(response.start),
            weights=weights,
            stype=2 if use_efron else 1,
            ctype=2 if use_efron else 1,
            se_fit=False,
            timefix=timefix,
        )
        eval_times = [t for t, d in zip(null_curve.time, null_curve.n_event, strict=True) if d > 0]
    else:
        eval_times = _float_vector(times, "times")
    phat = _brier_model_predictions(fit, engine, newdata, eval_times)
    result = _core.brier(
        dtime,
        dstat,
        eval_times,
        phat,
        weights=weights,
        ties=ties_value,
        efron=use_efron,
        timefix=timefix,
        start=None if response.start is None else list(response.start),
    )
    if not _normalize_bool_option(detail, "detail"):
        return BrierResult(rsquared=result.rsquared, brier=result.brier, times=result.times)
    return BrierResult(
        rsquared=result.rsquared,
        brier=result.brier,
        times=result.times,
        p0=result.p0,
        phat=phat.tolist(),
        eff_n=result.eff_n,
    )
