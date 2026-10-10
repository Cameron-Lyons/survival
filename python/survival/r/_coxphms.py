"""The multi-state Cox model: R's ``coxph`` for a ``Surv(time, state)`` or
``Surv(start, stop, state)`` response (R/coxph.R, parsecovar.R, multimiss.R,
stacker.R, print.coxph.R's ``coef.coxphms``, xtras.R's ``vcov.coxphms``,
predict.coxphms.R and residuals.coxphms.R).

:func:`coxph` hands a multi-state model frame to :func:`fit_multistate`, which does
what coxph.R's multi-state sections do: the missing values of a formula list,
``survcheck2``, the transition maps (``parsecovar1``-``parsecovar3``), then the Rust
``coxphms_fit`` stacks the data, fits it and computes ``share``.  The methods work
on the Cox fit of the stacked data and put its rows back on the data rows through
``rmap``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _categories,
    _float_vector,
    _is_missing_value,
    _match_string_arg,
    _materialize_labels,
    _normalize_bool_option,
    _normalize_na_action,
    _normalize_optional_bool_option,
    _pop_dotted_keyword,
    _r_factor_levels,
    _rows_of,
    _start_time_value,
    _warn_outside_package,
)
from ._coxph import (
    _LOG_DOUBLE_MAX,
    CoxphModel,
    _check_interaction_margins,
    _cluster_codes,
    _concordance_summary,
    _cox_fit_diagnostic_messages,
    _fit_frame,
    _prediction_newdata,
    _row_names,
    _survfit_types,
)
from ._data_prep import aeqSurv
from ._fit import _excluded_rows, _ModelFrame, _newdata_columns, _pad_rows, _tt_terms
from ._formula import (
    _column_source,
    _covariate_factors,
    _formula_model_term_degree,
    _formula_tokens,
    _prepare_formula_inputs,
    _scan,
    _split_terms,
    _strata_specs,
    _strata_term,
    _top_level,
)
from ._surv import Surv, _missing_rows
from ._types import (
    CoxSurvfitMultiStateResult,
    NaAction,
    NamedMatrix,
    _CategoricalDesignTerm,
    _FormulaModelTerm,
    _ModelClusterTerm,
    _ModelCovariateTerm,
    _ModelOffsetTerm,
    _ModelStrataTerm,
    _NumericDesignTerm,
    _PenaltyDesignTerm,
)

# survcheck2's flags, the names coxph.control's survcheckallow may list
SURVCHECK_FLAGS = ("overlap", "gap", "jump", "teleport")


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxphmsShare:
    """``fit$share`` of a model with shared baseline hazards: per row of ``cmap``,
    ``vtype`` is 0 for a fixed covariate, 1 for a time-dependent one and 2 for a
    coefficient that is constant within each transition of the shared hazard;
    ``scale`` is the hazard multiplier those give each transition."""

    vtype: tuple[int, ...]
    scale: tuple[float, ...]


@dataclass(frozen=True)
class _MsData:
    """The unstacked data of a multi-state fit, which its methods rebuild from."""

    x: np.ndarray
    x_names: tuple[str, ...]
    # 1-based position in the model terms of each column of x
    x_assign: tuple[int, ...]
    means: tuple[float, ...]
    # 1-based current state and target state (0 when censored) of each row
    istate: np.ndarray
    endpoint: np.ndarray
    # the istate argument's values, which survcheck2 read
    istate_values: Any
    id: np.ndarray
    weights: np.ndarray | None
    offset: np.ndarray | None
    # the codes of each strata() term (-1 missing), and the transitions each stratifies
    strata_terms: tuple[np.ndarray, ...]
    strata_use: np.ndarray
    # the 0-based transition of each stacked row, and each transition's reference
    # transition (1-based) when its baseline is proportional to another's, else 0
    hazard: np.ndarray
    phbaseline: tuple[int, ...]
    # the user strata (strata(mf[stangle$vars], shortlabel = TRUE)); -1 missing
    strata: np.ndarray | None
    strata_levels: tuple[str, ...]
    # rownames(mf) of every row before the na.action (after subset)
    row_labels: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CoxphmsModel(CoxphModel):
    """R's ``coxphms`` object (class ``c("coxphms", "coxph")``).

    ``fit`` is the Cox fit of the stacked data, so the inherited components are R's:
    ``coefficients``, ``var``, ``loglik``, ``linear_predictors`` and ``residuals``
    (one per stacked row), ``nevent``, the concordance, ...  ``y``, ``n``, ``id`` and
    ``x`` are the unstacked data.  ``states`` are the states, ``transitions`` the
    observed transitions (survcheck's table), ``cmap`` maps each covariate and
    transition to its coefficient (0 for none), ``smap`` each transition to its
    baseline hazard and strata terms, ``rmap`` gives each stacked row its 1-based data
    row and baseline block, and ``share`` is set for shared baseline hazards.
    """

    states: tuple[str, ...]
    cmap: NamedMatrix
    smap: NamedMatrix
    rmap: np.ndarray = field(repr=False, compare=False)
    transitions: NamedMatrix
    share: CoxphmsShare | None
    n_id: int
    ms: _MsData = field(repr=False, compare=False)

    @property
    def x(self) -> list[list[float]]:
        return self.ms.x.tolist()

    @property
    def means(self) -> list[float]:
        return list(self.ms.means)

    @property
    def strata(self) -> list[str | None] | None:
        codes = self.ms.strata
        if codes is None:
            return None
        return [None if code < 0 else self.ms.strata_levels[code] for code in codes]


# ---------------------------------------------------------------------------
# parsecovar1: the formula list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StateSet:
    """R's ``list(stateid = name, values = c(...))``: the states whose ``statedata``
    column ``name`` takes one of ``values``."""

    stateid: str
    values: tuple[Any, ...] | None


_Side = tuple[Any, ...] | _StateSet


@dataclass(frozen=True)
class _CovariateLine:
    """One ``lhs ~ rhs / options`` element of a formula list."""

    pairs: tuple[tuple[_Side, _Side], ...]
    rhs: str
    common: bool
    shared: bool


@dataclass(frozen=True)
class _FormulaList:
    """R's formula list: the master formula the model frame is made from (the first
    formula's terms, then those of every line), the first formula's right-hand side
    and the covariate lines."""

    master: str
    dformula_rhs: str
    lines: tuple[_CovariateLine, ...]
    statedata: dict[str, list[Any]] | None


def _statedata_columns(statedata: Any) -> dict[str, list[Any]]:
    if isinstance(statedata, Mapping):
        columns = dict(statedata)
    elif hasattr(statedata, "columns") and hasattr(statedata, "__getitem__"):
        columns = {name: statedata[name] for name in statedata.columns}
    else:
        raise ValueError("statedata must be a data frame")
    if "state" not in columns:
        raise ValueError("statedata data frame must contain a 'state' variable")
    return {str(name): _materialize_labels(values, str(name)) for name, values in columns.items()}


def _unquote(text: str) -> str | None:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return None


def _enclosed(text: str) -> bool:
    """Whether the parentheses at the ends of *text* match each other."""

    if not (text.startswith("(") and text.endswith(")")):
        return False
    return not any(depth == 0 for idx, _c, depth in _scan(text) if 0 < idx < len(text) - 1)


def _split_top(text: str, separator: str) -> list[str]:
    positions = [idx for idx, char in _top_level(text) if char == separator]
    bounds = [-1, *positions, len(text)]
    return [text[a + 1 : b].strip() for a, b in zip(bounds, bounds[1:], strict=False)]


def _r_number(text: str) -> float | None:
    try:
        value = float(text[:-1] if text.endswith("L") else text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _lhs_value(text: str, names: Sequence[str]) -> Any:
    """Evaluate one side of a transition (parsecovar1's ``eval`` in its environments)
    with a literal-only evaluator: integers, ``a:b``, quoted state names, ``c(...)``,
    parentheses, and a ``statedata`` column name, alone or called with values."""

    text = text.strip()
    if _enclosed(text):
        return _lhs_value(text[1:-1], names)
    colons = [idx for idx, char in _top_level(text) if char == ":"]
    if colons:
        # a:b, R's `:` being left associative
        left = _lhs_value(text[: colons[-1]], names)
        right = _lhs_value(text[colons[-1] + 1 :], names)
        if not (isinstance(left, tuple) and isinstance(right, tuple)) or not left or not right:
            raise ValueError(f"invalid term: {text}")
        start, stop = left[0], right[0]
        step = 1 if stop >= start else -1
        return tuple(start + step * k for k in range(int(abs(stop - start)) + 1))
    quoted = _unquote(text)
    if quoted is not None:
        return (quoted,)
    number = _r_number(text)
    if number is not None:
        return (number,)
    if "(" in text and text.endswith(")"):
        name, _sep, arguments = text[:-1].partition("(")
        name = name.strip()
        values: list[Any] = []
        for argument in _split_top(arguments, ",") if arguments.strip() else []:
            value = _lhs_value(argument, names)
            if not isinstance(value, tuple):
                raise ValueError(f"invalid term: {text}")
            values.extend(value)
        if name == "c":
            return tuple(values)
        if name in names:
            return _StateSet(name, tuple(values))
        raise ValueError(f"invalid term: {text}")
    if text in names:
        return _StateSet(text, None)
    raise ValueError(f"object '{text}' not found (quote state names, e.g. \"death\")")


def _parse_lhs(lhs: str, names: Sequence[str]) -> tuple[tuple[_Side, _Side], ...]:
    """parsecovar1's ``pcut``: the ``+``-separated terms of a left-hand side, each
    split at its last top-level ``:``."""

    pairs = []
    for term in _split_top(lhs, "+"):
        colons = [idx for idx, char in _top_level(term) if char == ":"]
        if not colons:
            raise ValueError(f"term found without a ':' {term}")
        pairs.append(
            (
                _lhs_value(term[: colons[-1]], names),
                _lhs_value(term[colons[-1] + 1 :], names),
            )
        )
    return tuple(pairs)


def _parse_rhs(rhs: str) -> tuple[str, bool, bool]:
    """rightslash + parse_rightside: the covariates before the last top-level ``/``
    and the options after it (``common``, ``shared``; ``init`` is accepted and, as in
    R, never used)."""

    slashes = [idx for idx, char in _top_level(rhs) if char == "/"]
    if not slashes:
        return rhs.strip(), False, False
    options = [
        option
        for part in _split_top(rhs[slashes[-1] + 1 :], "+")
        for piece in _split_top(part, "*")
        for option in _split_top(piece, ":")
    ]
    unknown = [option for option in options if option not in {"common", "shared", "init"}]
    if unknown:
        raise ValueError("option not recognized in a covariates formula: " + ", ".join(unknown))
    return rhs[: slashes[-1]].strip(), "common" in options, "shared" in options


def _formula_sides(formula: Any) -> tuple[str, str]:
    if not isinstance(formula, str):
        raise ValueError("an element of the formula list is not a formula")
    lhs, separator, rhs = formula.partition("~")
    if not separator or not lhs.strip() or not rhs.strip():
        raise ValueError("all formulas must have a left and right side")
    return lhs.strip(), rhs.strip()


def _formula_list(formulas: Sequence[Any], statedata: Any | None) -> _FormulaList:
    """parsecovar1 and coxph.R's master formula."""

    if not formulas:
        raise ValueError("all formulas must have a left and right side")
    sides = [_formula_sides(formula) for formula in formulas]
    columns = None if statedata is None else _statedata_columns(statedata)
    names = ("state",) if columns is None else tuple(columns)
    response, dformula_rhs = sides[0]
    lines = []
    for lhs, rhs in sides[1:]:
        covariates, common, shared = _parse_rhs(rhs)
        lines.append(_CovariateLine(_parse_lhs(lhs, names), covariates, common, shared))
    for rhs in [dformula_rhs, *(line.rhs for line in lines)]:
        if any(isinstance(term, _ModelOffsetTerm) for term in _split_terms(rhs).model_terms):
            raise ValueError(
                "offset() terms are not supported in a list of formulas (use the offset argument)"
            )
    for line in lines:
        if any(isinstance(term, _ModelClusterTerm) for term in _split_terms(line.rhs).model_terms):
            raise ValueError("cluster() terms are only allowed in the first formula of a list")
    # the term labels of each line; a removal ("- x") adds none
    added = [
        term
        for line in lines
        for op, term in _formula_tokens(line.rhs)
        if op == "+" and term not in {"0", "1"}
    ]
    master = f"{response} ~ {' + '.join([dformula_rhs, *added])}"
    return _FormulaList(master, dformula_rhs, tuple(lines), columns)


# ---------------------------------------------------------------------------
# parsecovar2 / parsecovar3
# ---------------------------------------------------------------------------


def _term_key(term: _FormulaModelTerm) -> Any:
    """What R's termmatch compares: a covariate term's set of variables (so ``a:b`` is
    ``b:a``), or a strata term's variables."""

    if isinstance(term, _ModelStrataTerm):
        return ("strata", term.spec)
    if isinstance(term, _ModelCovariateTerm):
        return frozenset(_covariate_factors(term.term))
    return term


def _model_terms(rhs: str) -> list[_FormulaModelTerm]:
    """The terms parsecovar2 matches: a ``cluster()`` term is the cluster argument and
    an ``offset()`` the offset, neither one a transition's term."""

    return [
        term
        for term in _split_terms(rhs).model_terms
        if not isinstance(term, _ModelOffsetTerm | _ModelClusterTerm)
    ]


def _termmatch(terms: Sequence[_FormulaModelTerm], allterm: Sequence[Any]) -> list[int]:
    positions = []
    for term in terms:
        if _term_key(term) not in allterm:
            raise ValueError("termmatch failure 1")
        positions.append(allterm.index(_term_key(term)))
    return positions


def _state_indices(side: _Side, statedata: dict[str, list[Any]], left: bool) -> list[int]:
    """The 1-based states one side of a transition names (parsecovar2)."""

    nstate = len(statedata["state"])
    if isinstance(side, _StateSet):
        if side.values is None:
            raise ValueError(f"state variable with no list of values: {side.stateid}")
        if side.stateid not in statedata:
            raise ValueError(f"{side.stateid}: state variable not found")
        column = statedata[side.stateid]
        unknown = [value for value in side.values if value not in column]
        if unknown:
            raise ValueError("".join(map(str, unknown)) + ": state value not found")
        return [i + 1 for i, value in enumerate(column) if value in side.values]
    if len(side) == 1 and not isinstance(side[0], str) and side[0] == 0:
        return list(range(1, nstate + 1))
    if all(isinstance(value, int | float) for value in side):
        if any(value != int(value) for value in side):
            raise ValueError("non-integer state number")
        if any(value < 1 or value > nstate for value in side):
            raise ValueError("numeric state is out of range")
        return [int(value) for value in side]
    states = [str(value) for value in statedata["state"]]
    labels = [str(value) for value in side]
    missing = [label for label in labels if label not in states]
    if missing:
        raise ValueError("".join(missing if left else labels) + ": state not found")
    return [i + 1 for i, state in enumerate(states) if state in labels]


def _line_pairs(line: _CovariateLine, statedata: dict[str, list[Any]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left, right in line.pairs:
        from_states = _state_indices(left, statedata, True)
        to_states = _state_indices(right, statedata, False)
        pairs.extend((s1, s2) for s2 in to_states for s1 in from_states)
    return pairs


@dataclass(frozen=True)
class _TransitionMap:
    """parsecovar2's result: ``tmap`` (row 0 the baseline hazard of each transition,
    then one row per model term), the transition labels, their states and the
    proportional baselines."""

    tmap: np.ndarray
    labels: tuple[str, ...]
    trans_from: np.ndarray
    trans_to: np.ndarray
    phbaseline: np.ndarray


def _parsecovar2(
    formulas: _FormulaList | None,
    dformula_rhs: str,
    allterm: list[Any],
    states: list[str],
    transitions: NamedMatrix,
) -> _TransitionMap:
    """parsecovar2: which terms (and which baseline hazard) each observed transition
    has, from the default formula's terms and each covariate line."""

    if formulas is None or formulas.statedata is None:
        statedata: dict[str, list[Any]] = {"state": list(states)}
    else:
        given = formulas.statedata
        position = {str(state): i for i, state in reversed(list(enumerate(given["state"])))}
        absent = [state for state in states if state not in position]
        if absent:
            raise ValueError(
                "statedata does not contain all the possible states: " + "".join(absent)
            )
        order = [position[state] for state in states]
        statedata = {name: [values[i] for i in order] for name, values in given.items()}

    nterm = len(allterm)
    nstate = len(states)
    shape = (nterm + 1, nstate, nstate)
    dmap = np.arange(1, math.prod(shape) + 1, dtype=np.int64).reshape(shape, order="F")
    tmap = np.zeros(shape, dtype=np.int64)
    rows = [0, *(1 + t for t in _termmatch(_model_terms(dformula_rhs), allterm))]
    tmap[rows] = dmap[rows]
    dformula_terms = rows[1:]
    for line in () if formulas is None else formulas.lines:
        pairs = _line_pairs(line, statedata)
        s1 = np.array([pair[0] - 1 for pair in pairs])
        s2 = np.array([pair[1] - 1 for pair in pairs])
        rindex = [1 + t for t in _termmatch(_model_terms(line.rhs), allterm)]
        # R's update(dformula, ~ . + rhs): the default terms the line removes
        joiner = " " if line.rhs.lstrip().startswith("-") else " + "
        kept = {1 + t for t in _termmatch(_model_terms(dformula_rhs + joiner + line.rhs), allterm)}
        for row in (row for row in dformula_terms if row not in kept):
            tmap[row, s1, s2] = 0
        # "~ -1 + (rhs)" has an intercept when the line asks for its own baseline
        if _split_terms("-1 + " + line.rhs).intercept:
            rindex = [0, *rindex]
        if rindex:
            for k in range(len(pairs)):
                source = (s1[0], s2[0]) if line.common else (s1[k], s2[k])
                tmap[rindex, s1[k], s2[k]] = dmap[(rindex, *source)]
        if line.shared and len(pairs) > 1:
            tmap[0, s1[1:], s2[1:]] = -dmap[0, s1[0], s2[0]]

    # the observed transitions: rows with any count (censoring included) and the
    # columns that are states
    counts = np.array(transitions.values, dtype=float)
    keep_rows = counts.sum(axis=1) > 0
    keep_cols = np.array([name in states for name in transitions.colnames])
    t2 = counts[np.ix_(keep_rows, keep_cols)]
    rownames = transitions.rownames or []
    indx1 = np.array([states.index(name) for name, k in zip(rownames, keep_rows, strict=True) if k])
    indx2 = np.array(
        [states.index(name) for name, k in zip(transitions.colnames, keep_cols, strict=True) if k]
    )
    # a shared hazard credits each of its transitions with all of its events
    baseline = tmap[0][np.ix_(indx1, indx2)]
    for value in np.unique(baseline):
        cells = baseline == value
        if cells.sum() > 1:
            t2[cells] = t2[cells].sum()
    tcol, trow = np.nonzero(t2.T > 0)  # the cells in column-major order
    tmap2 = tmap[:, indx1[trow], indx2[tcol]]
    labels = tuple(f"{indx1[i] + 1}:{indx2[j] + 1}" for i, j in zip(trow, tcol, strict=True))

    temp = tmap2[0].copy()
    positive = np.flatnonzero(temp > 0)
    reference = {int(temp[k]): int(k) for k in positive[::-1]}
    if any(int(abs(value)) not in reference for value in temp):
        raise ValueError("a shared baseline hazard has no observed reference transition")
    tmap2[0] = [reference[int(abs(value))] + 1 for value in temp]
    phbaseline = np.where(temp < 0, tmap2[0], 0)
    tmap2[0] = _match_first(tmap2[0], tmap2[0])
    if nterm:
        flat = tmap2[1:].flatten(order="F")
        tmap2[1:] = (_match_first(flat, np.concatenate([[0], flat])) - 1).reshape(
            tmap2[1:].shape, order="F"
        )
    return _TransitionMap(
        tmap=tmap2,
        labels=labels,
        trans_from=indx1[trow] + 1,
        trans_to=indx2[tcol] + 1,
        phbaseline=phbaseline,
    )


def _match_first(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    """R's ``match(values, unique(table))``."""

    order: dict[int, int] = {}
    for value in table.tolist():
        order.setdefault(value, len(order) + 1)
    return np.array([order[value] for value in values.tolist()], dtype=np.int64)


def _parsecovar3(tmap: np.ndarray, x_assign: Sequence[int], phbaseline: np.ndarray) -> np.ndarray:
    """parsecovar3: ``cmap``, one row per column of X, then one per transition that
    is the reference of proportional baselines."""

    ntrans = tmap.shape[1]
    references = list(dict.fromkeys(int(p) for p in phbaseline if p != 0))
    cmap = np.zeros((len(x_assign) + len(references), ntrans), dtype=np.int64)
    counts = np.bincount(np.asarray(x_assign, dtype=np.int64), minlength=max(x_assign) + 1)[1:]
    mult = 1 + int(counts.max())
    row = 0
    for term in dict.fromkeys(a for a in x_assign if a != 0):
        for k in range(1, counts[term - 1] + 1):
            cmap[row + k - 1] = np.where(tmap[term] == 0, 0, tmap[term] * mult + k)
        row += int(counts[term - 1])
    for reference in references:
        members = np.flatnonzero(phbaseline == reference)
        cmap[row, members] = cmap.max() + np.arange(1, len(members) + 1)
        row += 1
    values = np.unique(np.concatenate([[0], cmap.ravel()]))
    return np.searchsorted(values, cmap).astype(np.int64)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def _survcheckallow(value: Any) -> tuple[str, ...]:
    """coxph.control's ``survcheckallow``: the survcheck flags a fit lets through."""

    names = (
        (value,) if isinstance(value, str) else tuple(_materialize_labels(value, "survcheckallow"))
    )
    if any(name not in SURVCHECK_FLAGS for name in names):
        raise ValueError("survcheckallow must be a subset of " + ", ".join(SURVCHECK_FLAGS))
    return names


def _check_terms(frame: _ModelFrame) -> None:
    """coxph.R's refusals for a multi-state model."""

    for term in frame.design.covariates:
        if isinstance(term, _PenaltyDesignTerm):
            what = {"ridge": "ridge penalties", "frailty": "frailty terms"}.get(
                term.kind, "pspline terms"
            )
            raise ValueError(f"multi-state models do not currently support {what}")
    if _tt_terms(frame.design):
        raise ValueError("the tt() transform is not implemented for multi-state models")
    if not frame.names:
        raise ValueError("a multi-state coxph model needs at least one covariate")
    if frame.id is None:
        raise ValueError("an id statement is required for multi-state models")


def _first_level_drop(frame: _ModelFrame) -> np.ndarray:
    """The rows a formula list loses before survcheck: a missing response, id, weight,
    istate or cluster."""

    drop = _missing_rows(frame.y)
    for values in (frame.id, frame.istate, frame.cluster):
        if values is not None:
            drop |= np.fromiter(map(_is_missing_value, values), bool, len(drop))
    if frame.weights is not None:
        drop |= np.isnan(np.asarray(frame.weights, dtype=float))
    return drop


def _unique_codes(values: Sequence[Any]) -> np.ndarray:
    """``match(x, unique(x))``, 1-based."""

    index: dict[Any, int] = {}
    return np.fromiter(
        (index.setdefault(value, len(index) + 1) for value in values), np.int32, len(values)
    )


@dataclass(frozen=True)
class _Survcheck:
    states: list[str]
    istate: np.ndarray
    transitions: NamedMatrix


def _survcheck2(
    y: Surv, events: np.ndarray, id_codes: np.ndarray, istate: Any | None, allow: Sequence[str]
) -> _Survcheck:
    """coxph.R's ``survcheck2`` call: the states, each row's 1-based current state and
    the transitions table; a flagged problem not in ``allow`` is an error."""

    codes = levels = None
    if istate is not None:
        levels = _r_factor_levels(istate)
        index = {level: code for code, level in enumerate(levels, 1)}
        codes = [index[value] for value in istate]
    check = _core.survcheck(
        id_codes.tolist(),
        list(y.time),
        events.tolist(),
        list(y.states),
        time1=None if y.start is None else list(y.start),
        istate=codes,
        istate_levels=None if levels is None else [str(level) for level in levels],
        istate0="(s0)",
        censor_label=y.clabel if y.clabel is not None else "censored",
        timefix=False,
    )
    if any(getattr(check.flag, name) > 0 for name in SURVCHECK_FLAGS if name not in allow):
        raise ValueError("data set fails survcheck for one or more subjects")
    states = list(check.states)
    position = {state: code for code, state in enumerate(states, 1)}
    table = check.transitions
    return _Survcheck(
        states=states,
        istate=np.array([position[state] for state in check.istate], dtype=np.int32),
        transitions=NamedMatrix(
            rownames=list(table.from_states),
            colnames=list(table.to_states),
            values=[list(row) for row in table.counts],
        ),
    )


def _term_missing(
    x: np.ndarray, x_assign: np.ndarray, strata_codes: dict[int, np.ndarray], nterm: int
) -> np.ndarray:
    """``n x nterm``: whether each row misses a variable of each model term."""

    missing = np.zeros((x.shape[0], nterm), dtype=bool)
    nan = np.isnan(x)
    for term in range(nterm):
        if term in strata_codes:
            missing[:, term] = strata_codes[term] < 0
        else:
            missing[:, term] = nan[:, x_assign == term + 1].any(axis=1)
    return missing


def _multimiss(
    term_missing: np.ndarray,
    istate: np.ndarray,
    endpoint: np.ndarray,
    tmap: _TransitionMap,
    nstate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """R's ``multimiss``: the rows with no transition whose terms they all have, and
    per (from, to) state the events lost from a transition to a missing term."""

    uses = (tmap.tmap[1:] > 0).astype(np.int64)
    ismiss = term_missing.astype(np.int64) @ uses
    ispart = istate[:, None] == tmap.trans_from[None, :]
    omit = ~np.any(ispart & (ismiss == 0), axis=1)
    count = np.zeros((nstate, nstate), dtype=np.int64)
    for k, (s1, s2) in enumerate(zip(tmap.trans_from, tmap.trans_to, strict=True)):
        count[s1 - 1, s2 - 1] = np.sum((istate == s1) & (endpoint == s2) & (ismiss[:, k] > 0))
    return omit, count


def _subtract_counts(transitions: NamedMatrix, count: np.ndarray, states: list[str]) -> NamedMatrix:
    values = [list(row) for row in transitions.values]
    for i, source in enumerate(transitions.rownames or []):
        for j, target in enumerate(transitions.colnames):
            if source in states and target in states:
                values[i][j] -= int(count[states.index(source), states.index(target)])
    return NamedMatrix(transitions.rownames, transitions.colnames, values)


def _coef_names(
    cmap: np.ndarray, rownames: Sequence[str], labels: Sequence[str], phbaseline: np.ndarray
) -> list[str]:
    """coxph.R's coefficient names: each coefficient's covariate, suffixed by its first
    transition unless the covariate has one coefficient for all of them, and
    ``ph(<transition>/<reference>)`` for the proportional baselines."""

    ncoef = int(cmap.max())
    flat = cmap.flatten(order="F")
    names = []
    for coef in range(1, ncoef + 1):
        position = int(np.argmax(flat == coef))
        row, col = position % cmap.shape[0], position // cmap.shape[0]
        values = cmap[row]
        single = bool(np.all((values == 0) | (values == values.max())))
        names.append(rownames[row] + ("" if single else f"_{labels[col]}"))
    children = [labels[k] for k in np.flatnonzero(phbaseline > 0).tolist()]
    bases = [labels[p - 1] for p in phbaseline if p > 0]
    for i, (child, base) in enumerate(zip(children, bases, strict=True)):
        names[ncoef - len(bases) + i] = f"ph({child}/{base})"
    return names


def _means(x: np.ndarray, nocenter: Sequence[float]) -> tuple[float, ...]:
    """``fit$means`` of the unstacked X: each column's mean over its non-missing
    values, 0 when they are all ``nocenter`` values."""

    means = []
    for column in x.T:
        values = column[~np.isnan(column)]
        means.append(0.0 if np.isin(values, nocenter).all() else float(values.mean()))
    return tuple(means)


def fit_multistate(
    frame: _ModelFrame,
    *,
    row_labels: Sequence[str],
    formulas: _FormulaList | None,
    na_action: str | None,
    method: str,
    init: Any | None,
    iter_max: int,
    eps: float | None,
    toler_chol: float | None,
    toler_inf: float | None,
    timefix: bool,
    robust: bool | None,
    singular_ok: bool,
    nocenter: list[float],
    keep_model: bool,
    survcheckallow: Sequence[str],
) -> CoxphmsModel:
    """The multi-state sections of coxph.R on the model frame (made with
    ``na.action = na.pass`` for a formula list, whose missing values are handled here)."""

    _check_terms(frame)
    n0 = frame.n
    kept = np.arange(n0)
    action = _normalize_na_action(na_action)
    if formulas is not None:
        drop = _first_level_drop(frame)
        if drop.all():
            raise ValueError("all observations deleted due to missing")
        kept = np.flatnonzero(~drop)
    ids = frame.id or []
    y = frame.y if len(kept) == n0 else frame.y.subset(kept.tolist())
    if timefix:
        y = aeqSurv(y)
    istate_values = (
        None
        if frame.istate is None
        else _rows_of(frame.istate, [frame.istate[i] for i in kept.tolist()])
    )
    events = np.array([0 if e is None else int(e) for e in y.event])
    id_codes = _unique_codes([ids[i] for i in kept.tolist()])
    check = _survcheck2(y, events, id_codes, istate_values, survcheckallow)
    states = check.states
    if not events.any():
        raise ValueError("a multi-state coxph model needs at least one event")
    state_index = [states.index(state) + 1 for state in y.states]
    endpoint = np.array([0 if e == 0 else state_index[e - 1] for e in events], dtype=np.int32)

    # the model terms in R's order (attr(Terms, "factors")) and each X column's term
    allterm_terms = sorted(
        (
            term
            for term in frame.terms.model_terms
            if isinstance(term, _ModelCovariateTerm | _ModelStrataTerm)
        ),
        key=_formula_model_term_degree,
    )
    allterm = [_term_key(term) for term in allterm_terms]
    x_assign = np.array(
        [
            index
            for index, columns in zip(
                frame.design.term_assignments, frame.assign.values(), strict=True
            )
            for _column in columns
        ],
        dtype=np.int32,
    )
    strata_columns = {
        t: term.spec for t, term in enumerate(allterm_terms) if isinstance(term, _ModelStrataTerm)
    }
    strata_positions = list(strata_columns)
    dformula_rhs = frame.formula.partition("~")[2] if formulas is None else formulas.dformula_rhs
    tmap = _parsecovar2(formulas, dformula_rhs, allterm, states, check.transitions)
    transitions = check.transitions

    x = np.asarray(frame.x, dtype=np.float64).reshape(n0, frame.nvar)[kept]
    strata_codes = {
        t: np.array(
            [-1 if code is None else code for code in _strata_term(frame.data, columns).codes],
            dtype=np.int32,
        )[kept]
        for t, columns in strata_columns.items()
    }
    istate = check.istate
    if formulas is not None:
        omit, count = _multimiss(
            _term_missing(x, x_assign, strata_codes, len(allterm)),
            istate,
            endpoint,
            tmap,
            len(states),
        )
        if omit.all():
            raise ValueError("all observations deleted due to missing values")
        if omit.any():
            transitions = _subtract_counts(transitions, count, states)
            keep = ~omit
            kept, x, istate, endpoint = kept[keep], x[keep], istate[keep], endpoint[keep]
            id_codes = id_codes[keep]
            y = y.subset(np.flatnonzero(keep).tolist())
            strata_codes = {t: codes[keep] for t, codes in strata_codes.items()}
    na_record = frame.na_action
    if len(kept) < n0:
        if action == "fail":
            raise ValueError("missing values in object")
        removed = sorted(set(range(n0)) - set(kept.tolist()))
        na_record = (
            NaAction(tuple(row + 1 for row in removed), action)
            if action in {"omit", "exclude"}
            else None
        )
        frame = frame.take(kept.tolist())
    if method == "exact":
        raise ValueError("ties='exact' not supported for multistate")
    if frame.weights is not None and not all(map(math.isfinite, frame.weights)):
        raise ValueError("weights must be finite")
    offset = None
    if frame.offset is not None and any(value != 0.0 for value in frame.offset):
        if any(not math.isfinite(v) or v > _LOG_DOUBLE_MAX for v in frame.offset):
            raise ValueError("offsets must lead to a finite risk score")
        offset = np.asarray(frame.offset, dtype=np.float64)

    smap_rows = [0, *(1 + t for t in strata_positions)]
    smap = tmap.tmap[smap_rows].copy()
    smap[1:] = (smap[1:] > 0).astype(np.int64)
    cmap = _parsecovar3(tmap.tmap, x_assign.tolist(), tmap.phbaseline)
    references = dict.fromkeys(int(p) for p in tmap.phbaseline if p != 0)
    cmap_rows = [*frame.names, *(f"ph({tmap.labels[p - 1]})" for p in references)]

    use_robust = True if robust is None else robust
    cluster: Sequence[int] | np.ndarray | None = None
    if frame.cluster is not None:
        if not use_robust:
            warnings.warn(
                "cluster specified with robust=FALSE, cluster ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        cluster = _cluster_codes(frame.cluster)
    elif use_robust:
        cluster = id_codes
    strata_terms = [strata_codes[t] for t in strata_positions]
    engine, rindex, block, hazard, _, vtype, scale = _core.coxphms_fit(
        np.asarray(y.time, dtype=np.float64),
        endpoint,
        istate,
        x,
        cmap.flatten(order="F"),
        cmap.shape[0],
        tmap.tmap[0],
        tmap.trans_from,
        tmap.trans_to,
        id_codes,
        x_assign,
        entry=None if y.start is None else np.asarray(y.start, dtype=np.float64),
        strata_terms=strata_terms or None,
        strata_use=smap[1:].flatten(order="F") if strata_terms else None,
        weights=None if frame.weights is None else np.asarray(frame.weights, dtype=np.float64),
        offset=offset,
        cluster=cluster,
        method=method,
        init=None if init is None else _float_vector(init, "init"),
        iter_max=iter_max,
        eps=eps,
        toler_chol=toler_chol,
        nocenter=nocenter,
        robust=use_robust,
    )
    aliased = [idx for idx, value in enumerate(engine.coefficients) if math.isnan(value)]
    if aliased and not singular_ok:
        columns = " ".join(str(idx + 1) for idx in aliased)
        raise ValueError(f"X matrix deemed to be singular; variable {columns}")
    for message in _cox_fit_diagnostic_messages(engine, iter_max, eps, toler_inf):
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    labels = list(tmap.labels)
    user_strata = None if frame.strata is None else np.asarray(frame.strata, dtype=np.int32)
    ms = _MsData(
        x=x,
        x_names=tuple(frame.names),
        x_assign=tuple(x_assign.tolist()),
        means=_means(x, nocenter),
        istate=istate,
        endpoint=endpoint,
        istate_values=frame.istate,
        id=id_codes,
        weights=None if frame.weights is None else np.asarray(frame.weights, dtype=np.float64),
        offset=offset,
        strata_terms=tuple(strata_terms),
        strata_use=smap[1:] > 0,
        hazard=np.asarray(hazard),
        phbaseline=tuple(int(p) for p in tmap.phbaseline),
        strata=user_strata,
        strata_levels=tuple(frame.strata_levels),
        row_labels=tuple(row_labels),
    )
    strata_labels = [spec.call for spec in strata_columns.values()]
    return CoxphmsModel(
        fit=engine,
        formula=frame.formula,
        design=frame.design,
        terms=frame.terms,
        coef_names=tuple(_coef_names(cmap, cmap_rows, labels, tmap.phbaseline)),
        assign=dict(frame.assign),
        y=y,
        strata_levels=tuple(frame.strata_levels),
        concordance=_concordance_summary(engine.concordance),
        n=frame.n,
        timefix=timefix,
        tt=False,
        id=tuple(frame.id or ()),
        cluster=None if frame.cluster is None else tuple(frame.cluster),
        cluster_levels=frame.cluster_levels,
        model=frame.model_frame() if keep_model else None,
        weights_column=frame.weights_column,
        id_column=frame.id_column,
        na_action=na_record,
        _frame=replace(frame, x=[]),
        states=tuple(states),
        cmap=NamedMatrix(cmap_rows, labels, cmap.tolist()),
        smap=NamedMatrix(["(Baseline)", *strata_labels], labels, smap.tolist()),
        rmap=np.column_stack([np.asarray(rindex) + 1, np.asarray(block)]).astype(np.int64),
        transitions=transitions,
        share=None
        if vtype is None or scale is None
        else CoxphmsShare(tuple(int(v) for v in vtype), tuple(float(v) for v in scale)),
        n_id=int(np.unique(id_codes).size),
        ms=ms,
    )


# ---------------------------------------------------------------------------
# coef.coxphms / vcov.coxphms
# ---------------------------------------------------------------------------


def coef_coxphms(fit: CoxphmsModel, *, matrix: bool = False) -> Any:
    """R's ``coef.coxphms``: the coefficients, or with ``matrix`` a matrix shaped like
    ``cmap`` holding each transition's coefficient of each covariate (0 where it has
    none); the states are ``fit.states``."""

    if not matrix:
        return fit.coefficients
    beta = np.asarray(fit.coefficients, dtype=np.float64)
    cmap = np.asarray(fit.cmap.values, dtype=np.int64)
    values = np.where(cmap > 0, beta[np.maximum(cmap, 1) - 1], 0.0)
    return NamedMatrix(list(fit.cmap.rownames or []), list(fit.cmap.colnames), values.tolist())


def vcov_coxphms(fit: CoxphmsModel, *, complete: bool = True, matrix: bool = False) -> Any:
    """R's ``vcov.coxphms``: the variance, or with ``matrix`` one ``nrow(cmap)``-square
    matrix per transition, the covariances of that transition's coefficients (0 where
    a covariate has none).  R's own result recycles an index and is not that."""

    var = np.asarray(fit.var, dtype=np.float64)
    if not matrix:
        if complete:
            return var.tolist()
        keep = [i for i, b in enumerate(fit.coefficients) if not math.isnan(b)]
        return var[np.ix_(keep, keep)].tolist()
    cmap = np.asarray(fit.cmap.values, dtype=np.int64)
    rownames = list(fit.cmap.rownames or [])
    blocks: dict[str, NamedMatrix] = {}
    for k, label in enumerate(fit.cmap.colnames):
        index = cmap[:, k]
        present = index > 0
        block = np.zeros((len(index), len(index)))
        rows = np.flatnonzero(present)
        block[np.ix_(rows, rows)] = var[np.ix_(index[rows] - 1, index[rows] - 1)]
        blocks[label] = NamedMatrix(rownames, rownames, block.tolist())
    return blocks


# ---------------------------------------------------------------------------
# predict.coxphms
# ---------------------------------------------------------------------------

_PREDICT_INCOMPLETE = "predict.coxphms not complete for type expected, survival and terms"


def _fitted_row_labels(fit: CoxphmsModel, *, padded: bool = False) -> list[str]:
    """``rownames(model.frame(fit))``: the data's row names of the rows the fit kept,
    or with ``padded`` of every row (``naresid.exclude`` puts the others back)."""

    labels = fit.ms.row_labels
    if padded or fit.na_action is None:
        return list(labels)
    gone = {row - 1 for row in fit.na_action.rows}
    return [label for row, label in enumerate(labels) if row not in gone]


def predict_coxphms(
    fit: CoxphmsModel,
    newdata: Any | None = None,
    *,
    type: str = "lp",
    se_fit: Any = False,
    na_action: Any | None = None,
    terms: Any | None = None,
    collapse: Any | None = None,
    reference: str | None = None,
) -> NamedMatrix:
    """R's ``predict.coxphms``: the linear predictor (``lp``) or risk score of every
    row for every transition, one column per transition.

    The rows are those of the unstacked design (``model.matrix(fit)``, NaN where a
    formula list left a covariate missing) or of ``newdata`` (its complete rows, R's
    ``na.omit``).  Types expected, survival and terms, and the strata reference, are
    not available in R; ``se_fit``, ``na_action``, ``terms`` and ``collapse`` are
    accepted and ignored, as R does.  Unlike R, the ``ph()`` rows of
    ``coef(fit, matrix=True)``, which scale baseline hazards, are left out of the
    linear predictor rather than failing as non-conformable.
    """

    newdata, _ = _prepare_formula_inputs(newdata)
    predict_type = _match_string_arg(
        type,
        "type",
        ("lp", "risk", "expected", "terms", "survival"),
        "type must be one of lp, risk, expected, terms, survival",
    )
    if reference is None:
        reference_name = (
            "sample" if predict_type == "terms" or len(fit.smap.values) == 1 else "strata"
        )
    else:
        reference_name = _match_string_arg(
            reference,
            "reference",
            ("strata", "sample", "zero"),
            "reference must be one of strata, sample, zero",
        )
    if predict_type not in {"lp", "risk"}:
        raise ValueError(_PREDICT_INCOMPLETE)
    if reference_name == "strata":
        raise ValueError("strata reference unfinished")
    if newdata is None:
        x = fit.ms.x
        rownames = _fitted_row_labels(fit)
    else:
        new = _prediction_newdata(
            fit, newdata, need_strata=False, need_response=False, na_action="na.omit"
        )
        x = np.asarray(new.x, dtype=np.float64).reshape(new.n, len(fit.ms.x_names))
        missing = set(new.missing)
        rownames = _row_names(
            newdata, [row for row in range(new.n + len(missing)) if row not in missing]
        )
    if reference_name == "sample":
        x = x - np.asarray(fit.ms.means)
    beta = np.asarray(coef_coxphms(fit, matrix=True).values, dtype=np.float64)
    eta = x @ beta[: len(fit.ms.x_names)]
    values = eta if predict_type == "lp" else np.exp(eta)
    return NamedMatrix(rownames, list(fit.cmap.colnames), values.tolist())


# ---------------------------------------------------------------------------
# residuals.coxphms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxphmsSchoenfeldResiduals:
    """``residuals(fit, type = "schoenfeld" | "scaledsch")`` of a multi-state fit: one
    row of ``values`` per event of the stacked data, in (stratum, time) order.  ``time``
    holds the event times (R's row names), ``transition`` the transition of each event,
    ``strata`` the 1-based stacked stratum of each event when the model has
    ``strata()`` terms (``attr(, "strata")``, else ``None``) and ``colnames`` the
    coefficient names."""

    values: list[list[float]] = field(repr=False)
    time: list[float] = field(repr=False)
    transition: list[str] = field(repr=False)
    strata: list[int] | None = field(repr=False)
    colnames: list[str]


def _collapse_groups(
    fit: CoxphmsModel, collapse: Any, *, by_level: bool
) -> tuple[np.ndarray, list[str]] | None:
    """residuals.coxphms's groups: ``TRUE`` means the cluster (else the id), a vector
    has one value per row of the model frame.  They are numbered in order of first
    appearance, R's ``factor(cluster, unique(cluster))``, except that with ``by_level``
    (``rowsum(reorder = TRUE)`` of the score family) the groups of a factor vector
    follow its level order and the missing values' group goes last.  All missing
    values form one group labelled ``"NA"``; the other labels are the values."""

    if collapse is None or collapse is False:
        return None
    declared = None
    if collapse is True:
        values = list(fit.cluster if fit.cluster is not None else fit.id or ())
    else:
        values = _materialize_labels(collapse, "collapse")
        if len(values) != fit.n:
            raise ValueError("collapse vector not the same length as the model frame")
        if by_level:
            declared = _categories(collapse)
    if declared is None:
        # every missing value is the one NA group (NaNs never compare equal)
        missing = [_is_missing_value(value) for value in values]
        keys = [None if miss else value for value, miss in zip(values, missing, strict=True)]
        codes = _unique_codes(keys) - 1
        first = np.unique(codes, return_index=True)[1]
        labels = [_as_character(values[row]) for row in first.tolist()]
        if by_level and any(missing):
            na_code = codes[missing.index(True)]
            codes = np.where(codes == na_code, len(labels) - 1, codes - (codes > na_code))
            labels.append(labels.pop(int(na_code)))
        return codes, labels
    position = {level: code for code, level in enumerate(declared)}
    level_codes = np.array([position.get(value, len(declared)) for value in values])
    present, codes = np.unique(level_codes, return_inverse=True)
    labels = [*map(_as_character, declared), "NA"]
    return codes, [labels[code] for code in present.tolist()]


def _rowsum_codes(values: np.ndarray, codes: np.ndarray, nrow: int) -> np.ndarray:
    total = np.zeros((nrow, values.shape[1]))
    np.add.at(total, codes, values)
    return total


def residuals_coxphms(
    fit: CoxphmsModel,
    *,
    type: str = "martingale",
    collapse: Any = False,
    weighted: Any | None = None,
    na_action: Any | None = None,
) -> NamedMatrix | CoxphmsSchoenfeldResiduals:
    """R's ``residuals.coxphms``: martingale residuals with one column per transition,
    score, dfbeta and dfbetas residuals with one column per coefficient (one row per
    data row, or per group with ``collapse``), or the Schoenfeld and scaled Schoenfeld
    residuals of the events as a :class:`CoxphmsSchoenfeldResiduals`.

    The residuals of the stacked fit are put back on the data row each stacked row
    came from.  ``weighted`` (default for dfbeta and dfbetas) multiplies each stacked
    row by its own data row's case weight; ``na_action`` ("na.omit" or "na.exclude")
    changes how a fit's removed rows are padded.  Schoenfeld residuals ignore
    ``collapse``, as in R.  Unlike R, every data row the stacking leaves out gets
    zero score residuals, and the columns of the martingale residuals follow the
    transition of each stacked row, so shared and common baselines keep one column
    per transition.
    """

    omit = fit.na_action
    if omit is not None and na_action is not None:
        kind = _normalize_na_action(na_action)
        if kind not in {"omit", "exclude"}:
            raise ValueError("changing to an unrecognized na.action type")
        omit = replace(omit, kind=kind)
    types = ("martingale", "score", "schoenfeld", "dfbeta", "dfbetas", "scaledsch")
    otype = _match_string_arg(type, "type", types, f"type must be one of {', '.join(types)}")
    weighted_value = _normalize_optional_bool_option(weighted, "weighted")
    if weighted_value is None:
        weighted_value = otype in {"dfbeta", "dfbetas"}
    engine = fit.fit
    rindex = fit.rmap[:, 0] - 1
    colnames = list(fit.coef_names)

    if otype in {"schoenfeld", "scaledsch"}:
        result = (
            engine.schoenfeld_residuals(weighted=weighted_value)
            if otype == "schoenfeld"
            else engine.scaled_schoenfeld_residuals(weighted=weighted_value)
        )
        labels = fit.cmap.colnames
        return CoxphmsSchoenfeldResiduals(
            values=[list(row) for row in result.residuals],
            time=list(result.time),
            transition=[labels[k] for k in fit.ms.hazard[np.asarray(result.rows)].tolist()],
            strata=None
            if result.strata is None or not fit.ms.strata_terms
            else [int(code) + 1 for code in result.strata],
            colnames=colnames,
        )

    groups = _collapse_groups(fit, collapse, by_level=otype != "martingale")
    if otype == "martingale":
        hazard = fit.ms.hazard
        present = np.unique(hazard)
        values = np.zeros((fit.n, len(present)))
        values[rindex, np.searchsorted(present, hazard)] = engine.residuals
        if weighted_value and fit.ms.weights is not None:
            values *= fit.ms.weights[:, None]
        colnames = [fit.cmap.colnames[k] for k in present.tolist()]
        if groups is not None:
            codes, labels = groups
            return NamedMatrix(labels, colnames, _rowsum_codes(values, codes, len(labels)).tolist())
        rows = values.tolist()
        rownames = None
    else:
        method = {
            "score": engine.score_residuals,
            "dfbeta": engine.dfbeta,
            "dfbetas": engine.dfbetas,
        }[otype]
        if groups is None:
            codes, nrow = rindex, fit.n
        else:
            codes, nrow = groups[0][rindex], len(groups[1])
        # the engine sums the stacked rows by code, in increasing code order
        summed = np.asarray(method(weighted=weighted_value, collapse=codes.tolist()))
        values = np.zeros((nrow, len(colnames)))
        values[np.unique(codes)] = summed.reshape(-1, len(colnames))
        if groups is not None:
            return NamedMatrix(groups[1], colnames, values.tolist())
        rows = values.tolist()
        rownames = _fitted_row_labels(fit)
    excluded = _excluded_rows(omit)
    if excluded:
        rows = _pad_rows(rows, excluded)
        rownames = None if rownames is None else _fitted_row_labels(fit, padded=True)
    return NamedMatrix(rownames, colnames, rows)


# ---------------------------------------------------------------------------
# cox.zph / coxph.detail on the stacked data
# ---------------------------------------------------------------------------


def _zph_assign(fit: CoxphmsModel) -> list[tuple[str, list[int]]]:
    """cox.zph's ``asgn`` with ``terms = TRUE`` for a multi-state fit (attrassign.R's
    ``expandassign``): the 0-based coefficients of each term, a model term within one
    transition, named ``<term label>_<transition>``, found in column-major order of
    ``cmap``; each ``ph()`` coefficient is a term of its own, named by the coefficient
    (R fails on those)."""

    labels = {col: label for label, cols in fit.assign.items() for col in cols}
    cmap = np.asarray(fit.cmap.values, dtype=np.int64)
    nx = len(fit.ms.x_names)
    groups: dict[str, list[int]] = {}
    seen: set[int] = set()
    for k, transition in enumerate(fit.cmap.colnames):
        for row in range(nx):
            coef = int(cmap[row, k])
            if coef > 0 and coef not in seen:
                seen.add(coef)
                groups.setdefault(f"{labels[row]}_{transition}", []).append(coef - 1)
    ph = sorted({int(coef) for coef in cmap[nx:].ravel() if coef > 0})
    return [*groups.items(), *((fit.coef_names[coef - 1], [coef - 1]) for coef in ph)]


def _stacked_strata_labels(codes: Sequence[int]) -> list[str]:
    """The 0-based stacked strata codes as the 1-based labels ``"1"``, ``"2"``, ...
    (R's integer stacker strata)."""

    return [str(int(code) + 1) for code in codes]


# ---------------------------------------------------------------------------
# survfit.coxphms
# ---------------------------------------------------------------------------


def _as_frame_columns(newdata: Any) -> dict[str, list[Any]]:
    """``newdata`` as named columns; a mapping of scalars is one row (R allows a named
    list there)."""

    columns = {str(name): _column_source(newdata, name) for name in _newdata_columns(newdata)}
    if isinstance(newdata, Mapping) and all(np.ndim(value) == 0 for value in columns.values()):
        return {name: [value] for name, value in columns.items()}
    return {name: list(values) for name, values in columns.items()}


def _share_dummies(fit: CoxphmsModel, newdata: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """survfit.coxphms's stand-ins for the variables of shared-hazard "gamma" terms that
    ``newdata`` leaves out (their coefficients are set to 0): a factor's first level,
    or a simple numeric variable's mean."""

    if fit.share is None:
        return newdata
    nx = len(fit.ms.x_names)
    gamma = {column for column, vtype in enumerate(fit.share.vtype[:nx]) if vtype == 2}
    nrow = len(next(iter(newdata.values()), []))
    filled = dict(newdata)
    for columns, term in zip(fit.assign.values(), fit.design.covariates, strict=True):
        if not gamma & set(columns):
            continue
        if isinstance(term, _CategoricalDesignTerm):
            value = term.levels[0]
        elif isinstance(term, _NumericDesignTerm) and term.term.transform is None:
            value = fit.ms.means[columns[0]]
        else:
            continue
        name = term.term.column
        if name not in filled:
            filled[name] = [value] * nrow
    return filled


def _check_newdata_strata(fit: CoxphmsModel, newdata: dict[str, list[Any]]) -> None:
    """Strata variables in ``newdata`` must take levels of the fit (``model.frame``'s
    ``xlev`` check); they select no curves."""

    terms = _strata_specs(fit.terms)
    if not terms or not all(column in newdata for spec in terms for column in spec.columns):
        return
    data = _fit_frame(fit).data
    for columns in terms:
        fitted = set(_strata_term(data, columns).levels)
        for level in _strata_term(newdata, columns).levels:
            if level not in fitted:
                raise ValueError(f"factor {columns.call} has new level {level}")


def _coxms_newdata(
    fit: CoxphmsModel, newdata: Any, na_action: Any | None
) -> tuple[np.ndarray, np.ndarray | None, dict[str, list[Any]]]:
    """``model.frame(Terms2, newdata, na.action)`` and ``model.matrix`` for the curves:
    the design and offset of the complete rows, and those rows of ``newdata``."""

    columns = _share_dummies(fit, _as_frame_columns(newdata))
    _check_newdata_strata(fit, columns)
    new = _prediction_newdata(
        fit, columns, need_strata=True, need_response=False, na_action="na.omit"
    )
    action = "omit" if na_action is None else _normalize_na_action(na_action)
    if new.missing:
        if action == "fail":
            raise ValueError("missing values in object")
        if action == "pass":
            raise ValueError("newdata rows with missing values need na_action='na.omit'")
    if new.n == 0:
        raise ValueError("all rows of newdata have missing values")
    missing = set(new.missing)
    kept = [row for row in range(new.n + len(missing)) if row not in missing]
    x = np.asarray(new.x, dtype=np.float64).reshape(new.n, len(fit.ms.x_names))
    offset = None if new.offset is None else np.asarray(new.offset, dtype=np.float64)
    used = {name: [values[row] for row in kept] for name, values in columns.items()}
    return x, offset, used


def survfit_coxphms(
    fit: CoxphmsModel,
    newdata: Any | None = None,
    *,
    se_fit: Any = False,
    conf_int: Any = 0.95,
    individual: Any = False,
    stype: Any | None = None,
    ctype: Any | None = None,
    conf_type: Any = "log",
    censor: Any = True,
    start_time: Any | None = None,
    id: Any | None = None,
    influence: Any = False,
    na_action: Any | None = None,
    type: Any | None = None,
    p0: Any | None = None,
    time0: Any = False,
    **kwargs: Any,
) -> CoxSurvfitMultiStateResult:
    """R's ``survfit.coxphms``: probability-in-state curves from a multi-state Cox
    model, one per ``newdata`` row (every stratum gets a curve for every row).

    The time grid, the counts and ``p0`` are those of the Aalen-Johansen estimate of the
    data (``survfitAJ``); ``pstate`` and ``cumhaz`` are the model's.  ``stype`` 1
    updates the state probabilities by ``p (I + A)``, 2 (the default) by ``p expm(A)``;
    ``ctype`` defaults to 2 for an Efron fit, else 1.  ``start_time`` drops the rows
    that end by then, and ``p0`` / ``time0`` are survfitAJ's.  Standard errors are not
    available: ``se_fit=True`` warns; ``conf_int``, ``conf_type``, ``censor`` and
    ``influence`` are accepted and ignored.  Incomplete ``newdata`` rows are left out
    (``na.omit``; ``na.fail`` refuses them).  Unlike R the offsets are used, aliased
    coefficients count as 0 and ``newdata`` keeps only the rows used.
    """

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, None)
    if kwargs:
        raise TypeError(f"survfit got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if newdata is None:
        raise ValueError("multi-state survival requires a newdata argument")
    if id is not None or _normalize_bool_option(individual, "individual"):
        raise ValueError("using a covariate path is not supported for multi-state")
    if _normalize_bool_option(se_fit, "se_fit"):
        _warn_outside_package("se.fit not yet implemented for multistate coxph models")
    stype_value, ctype_value = _survfit_types(fit, type, stype, ctype)
    ms = fit.ms
    if ms.strata_terms:
        _check_interaction_margins(fit)
    start = _start_time_value(start_time)
    time0_value = _normalize_bool_option(time0, "time0")
    p0_value = None if p0 is None else _float_vector(p0, "p0")

    # the rows of the curves: none with a missing user stratum, none ending by start.time
    y = fit.y
    time = np.asarray(y.time, dtype=np.float64)
    keep = np.ones(len(time), dtype=bool) if ms.strata is None else ms.strata >= 0
    if start is not None:
        keep &= time > start
        if not keep.any():
            raise ValueError("start.time has removed all observations")
        survfit_start = start
    else:
        # survfitAJ's start.time <- min(Y[, 2], 0): the status column of right-censored
        # data, the stop time of counting-process data
        survfit_start = 0.0 if y.start is None else min(0.0, float(time.min()))
    rows = np.flatnonzero(keep)
    subset = len(rows) < len(time)
    y_used = y.subset(rows.tolist()) if subset else y
    events = np.array([0 if e is None else int(e) for e in y_used.event])
    istate_values = (
        None
        if ms.istate_values is None
        else _rows_of(ms.istate_values, [ms.istate_values[i] for i in rows.tolist()])
    )
    check = _survcheck2(y_used, events, ms.id[rows], istate_values, SURVCHECK_FLAGS)
    if check.states != list(fit.states):
        raise ValueError("failed to rebuild the data set")

    x2, offset2, used = _coxms_newdata(fit, newdata, na_action)
    beta = np.nan_to_num(np.asarray(fit.coefficients, dtype=np.float64), nan=0.0)
    cmap = np.asarray(fit.cmap.values, dtype=np.int32)
    if fit.share is not None:
        gamma = cmap[np.asarray(fit.share.vtype) == 2]
        beta[gamma[gamma > 0] - 1] = 0.0
    labels = list(fit.cmap.colnames)
    trans_from, trans_to = zip(*(label.split(":") for label in labels), strict=True)
    engine, pstate, cumhaz = _core.coxphms_curves(
        time[rows],
        ms.endpoint[rows],
        check.istate,
        ms.x[rows],
        cmap.flatten(order="F"),
        cmap.shape[0],
        np.asarray(fit.smap.values[0], dtype=np.int32),
        np.asarray(trans_from, dtype=np.int32),
        np.asarray(trans_to, dtype=np.int32),
        ms.id[rows],
        list(fit.states),
        beta,
        np.asarray(ms.means, dtype=np.float64),
        x2,
        stype_value,
        ctype_value,
        entry=None if y.start is None else np.asarray(y.start, dtype=np.float64)[rows],
        weights=None if ms.weights is None else ms.weights[rows],
        offset=None if ms.offset is None else ms.offset[rows],
        strata=None if ms.strata is None else ms.strata[rows],
        strata_terms=[codes[rows] for codes in ms.strata_terms] or None,
        strata_use=ms.strata_use.flatten(order="F") if ms.strata_terms else None,
        share_scale=None if fit.share is None else np.asarray(fit.share.scale),
        newoffset=offset2,
        start_time=survfit_start,
        p0=p0_value,
        time0=time0_value,
    )
    strata = None
    if engine.strata is not None:
        names = [ms.strata_levels[code] for code in engine.strata_codes or ()]
        strata = dict(zip(names, engine.strata, strict=True))
    return CoxSurvfitMultiStateResult(
        n=[int(value) for value in engine.n],
        time=list(engine.time),
        n_risk=engine.n_risk,
        n_event=engine.n_event,
        n_censor=engine.n_censor,
        n_transition=engine.n_transition,
        n_id=[int(value) for value in engine.n_id],
        pstate=np.ascontiguousarray(pstate),
        cumhaz=np.ascontiguousarray(cumhaz),
        cumhaz_names=labels,
        p0=engine.p0,
        states=list(fit.states),
        transitions=check.transitions,
        type=engine.type,
        t0=engine.t0,
        start_time=engine.start_time,
        strata=strata,
        newdata=used,
        stype=stype_value,
        ctype=ctype_value,
        time0=time0_value,
        engine=engine,
    )
