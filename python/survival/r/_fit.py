"""The shared model-frame path (R's ``model.frame`` + ``model.matrix`` for a survival
formula) used by ``coxph``, ``cch``, ``aareg`` and ``concordance``, and the ``newdata``
and ``naresid`` helpers the model methods share."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import compress, product
from typing import Any

from ._coerce import (
    _DEFAULT_NA_ACTION,
    _categories,
    _float_vector,
    _floats_or_nan,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _mstate_categories,
    _normalize_na_action,
    _optional_float_vector,
    _rows_of,
    _strata_level_sort_key,
    _strata_value_label,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _column_source,
    _covariate_term_name,
    _data_row_count,
    _data_row_labels,
    _data_rows,
    _design_rows_from_spec,
    _design_term_name,
    _fit_formula_design,
    _formula_cluster_values,
    _formula_data_rows,
    _formula_design_columns,
    _formula_design_row_count,
    _formula_missing_rows,
    _formula_model_frame,
    _formula_response_spec,
    _formula_response_values,
    _made_nan_rows,
    _na_action_record,
    _offset_vector,
    _parse_formula,
    _penalty_arguments,
    _strata_covariate,
    _strata_keep,
    _strata_specs,
    _strata_term_values,
    _subset_formula_inputs,
    _with_strata_cache,
)
from ._penalties import fit_penalty
from ._surv import Surv, _complete_codes, _strata
from ._types import (
    NaAction,
    StrataFactor,
    _CategoricalDesignTerm,
    _CovariateTerm,
    _DesignTerm,
    _FormulaDesign,
    _FormulaTerms,
    _InteractionDesignTerm,
    _MatrixDesignTerm,
    _PenaltyDesignTerm,
    _SingleDesignTerm,
    _StrataSpec,
    _SurvResponseSpec,
)

# ---------------------------------------------------------------------------
# model.frame / model.matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelFrame:
    """What R's ``model.frame`` + ``model.matrix`` give a survival model function.

    ``x`` is the model matrix without its intercept column, ``names`` its column
    names, ``assign`` R's ``attrassign`` (term label -> 0-based columns); the
    ``strata``, ``offset``, ``weights``, ``cluster``, ``id`` and ``istate`` specials are
    already row-aligned with ``y`` (after ``subset`` and ``na.action``); ``na_action``
    records the rows the ``na.action`` removed.
    """

    formula: str
    data: Any
    y: Surv
    x: list[list[float]]
    design: _FormulaDesign
    terms: _FormulaTerms
    names: list[str]
    assign: dict[str, tuple[int, ...]]
    strata: list[int] | None
    strata_levels: tuple[str, ...]
    offset: list[float] | None
    weights: list[float] | None
    cluster: list[Any] | None
    id: list[Any] | None
    # a factor keeps its levels (survcheck's states come from them)
    istate: Sequence[Any] | None
    row_names: tuple[str, ...] | None = None
    extra: dict[str, list[Any]] = field(default_factory=dict)
    # the column names the weights= / id= arguments referred to (R keeps the call's
    # expressions, so brier's newdata can re-evaluate them); None for vector arguments
    weights_column: str | None = None
    id_column: str | None = None
    na_action: NaAction | None = None
    cluster_levels: tuple[Any, ...] | None = None
    id_levels: tuple[Any, ...] | None = None

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def nvar(self) -> int:
        return len(self.names)

    def model_frame(self) -> dict[str, Any]:
        """R's ``fit$model``: the response and every variable the formula uses."""

        return _formula_model_frame(
            self.data,
            self.y,
            self.design,
            extra_columns=tuple(self.terms.clusters),
            weights=self.weights,
            offset=self.offset,
            strata=self.strata_labels(),
            cluster=self.cluster,
            id=self.id,
            istate=self.istate,
        )

    def strata_labels(self) -> list[str | None] | None:
        if self.strata is None:
            return None
        return [None if code < 0 else self.strata_levels[code] for code in self.strata]

    def take(self, rows: Sequence[int]) -> _ModelFrame:
        """The frame at the 0-based *rows* (R's ``mf[rows, ]``), every row-aligned
        piece subset together."""

        def pick(values: Sequence[Any] | None) -> Any:
            if values is None:
                return None
            return _rows_of(values, [values[row] for row in rows])

        return replace(
            self,
            data=_formula_data_rows(self.formula, self.data, list(rows), self.n),
            y=self.y.subset(rows),
            x=pick(self.x) if self.x else self.x,
            strata=pick(self.strata),
            offset=pick(self.offset),
            weights=pick(self.weights),
            cluster=pick(self.cluster),
            id=pick(self.id),
            istate=pick(self.istate),
            extra={name: pick(values) for name, values in self.extra.items()},
        )


def _model_frame_levels(values: Any, levels: Sequence[Any]) -> tuple[Any, ...]:
    """``levels(x)`` of a model-frame variable: the column's own categories when it
    carries them (a pandas Categorical / R factor), unused ones included, since
    ``model.frame`` keeps them (``drop.unused.levels = FALSE``); else the sorted distinct
    values *levels*."""

    categories = _mstate_categories(values)
    if categories is not None:
        return tuple(_materialize_1d(categories, "categories"))
    return tuple(sorted(levels, key=_strata_level_sort_key))


def _r_levels(values: Any, levels: Sequence[Any]) -> tuple[Any, ...]:
    """``levels(factor(x))``: the levels of :func:`_model_frame_levels` among the distinct
    values *levels* (``factor()`` drops the unused ones)."""

    present = set(levels)
    return tuple(level for level in _model_frame_levels(values, levels) if level in present)


def _r_factor_design(
    data: Any,
    design: _FormulaDesign,
    *,
    drop_unused_levels: bool = False,
    drop_unused_strata: bool = True,
) -> _FormulaDesign:
    """Give every categorical term R's factor level order (the formula module
    keeps first-appearance order): ``model.frame``'s levels, a factor's unused ones
    included, or with *drop_unused_levels* only those that occur, as ``lm``'s
    ``model.frame(drop.unused.levels = TRUE)`` has them. AFT also retains evaluated
    strata levels with *drop_unused_strata=False*, including interaction columns."""

    levels_of = _r_levels if drop_unused_levels else _model_frame_levels

    def relevel(term: _SingleDesignTerm) -> _SingleDesignTerm:
        if not isinstance(term, _CategoricalDesignTerm):
            return term
        # a logical expression (I(sex == 2)) has no column to declare levels
        column = term.term.column
        source = (
            _strata_term_values(data, term.term.strata, drop_unused=drop_unused_strata)
            if term.term.strata
            else None
            if term.term.arithmetic is not None
            else _column_source(data, column)
        )
        return replace(term, levels=levels_of(source, term.levels))

    covariates: list[_DesignTerm] = []
    for term in design.covariates:
        if isinstance(term, _InteractionDesignTerm):
            covariates.append(replace(term, factors=tuple(relevel(f) for f in term.factors)))
        else:
            covariates.append(relevel(term))
    return replace(design, covariates=tuple(covariates))


def _column_names(term: _SingleDesignTerm) -> list[str]:
    """``colnames(model.matrix)`` for one term: ``factor(x)level`` with R's labels."""

    if isinstance(term, _PenaltyDesignTerm | _MatrixDesignTerm):
        return list(term.names)
    prefix = _covariate_term_name(term.term)
    if isinstance(term, _CategoricalDesignTerm):
        if term.contrasts and not term.full:
            return [f"{prefix}{name}" for name in term.contrast_names]
        levels = term.levels if term.full else term.levels[1:]
        return [f"{prefix}{_strata_value_label(level)}" for level in levels]
    return [prefix]


def _design_names_and_assign(
    design: _FormulaDesign,
) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    names: list[str] = []
    assign: dict[str, tuple[int, ...]] = {}
    for term in design.covariates:
        if isinstance(term, _InteractionDesignTerm):
            parts = [_column_names(factor) for factor in term.factors]
            columns = [":".join(reversed(combo)) for combo in product(*reversed(parts))]
        else:
            columns = _column_names(term)
        assign[_design_term_name(term)] = tuple(range(len(names), len(names) + len(columns)))
        names.extend(columns)
    return names, assign


def _model_matrix_column_names(term: _SingleDesignTerm) -> list[str]:
    """Formula/basis labels, before a penalty replaces the coefficient labels."""
    if not isinstance(term, _PenaltyDesignTerm):
        return _column_names(term)
    if term.matrix_names:
        return list(term.matrix_names)
    prefix = _covariate_term_name(term.term)
    width = len(term.names)
    if width == 1:
        return [prefix]
    suffixes = (
        term.columns
        if term.kind == "ridge" and term.term.transform != "tt"
        else tuple(str(j + 1) for j in range(width))
    )
    return [prefix + suffix for suffix in suffixes]


def _model_matrix_names_and_assign(design: _FormulaDesign) -> tuple[list[str], list[int]]:
    names: list[str] = ["(Intercept)"] if design.intercept else []
    assign = [0] if design.intercept else []
    for term, code in zip(design.covariates, design.term_assignments, strict=True):
        if isinstance(term, _InteractionDesignTerm):
            parts = [_model_matrix_column_names(factor) for factor in term.factors]
            columns = [":".join(reversed(combo)) for combo in product(*reversed(parts))]
        else:
            columns = _model_matrix_column_names(term)
        names.extend(columns)
        assign.extend([code] * len(columns))
    return names, assign


def _model_matrix_newdata_design(design: _FormulaDesign, data: Any) -> _FormulaDesign:
    """Reevaluate frailty constructors before NA omission, then restore fitted factors.

    Sparse groups have no fitted factor levels. R therefore recodes them locally,
    and the automatic sparse default can produce a dense basis on a small new set.
    Dense terms retain their original full identity contrasts and factor levels.
    """
    covariates: list[_DesignTerm] = []
    for term in design.covariates:
        if isinstance(term, _PenaltyDesignTerm) and term.term.transform == "tt":
            raise ValueError("an evaluated matrix is required for a matrix time transform")
        if not isinstance(term, _PenaltyDesignTerm) or term.kind != "frailty":
            covariates.append(term)
            continue
        if term.term.call is None:
            raise ValueError("a frailty term must have a function call")
        columns, options = _penalty_arguments(term.term.call)
        source = _column_source(data, columns[0])
        fresh = fit_penalty(
            term.term,
            columns,
            {column: _column(data, column) for column in columns},
            options,
            _mstate_categories(source),
        )
        if not fresh.penalty.sparse and len(fresh.levels) < 2:
            raise ValueError("not enough degrees of freedom to define contrasts")
        if not term.penalty.sparse:
            if fresh.penalty.sparse:
                raise ValueError("contrasts apply only to factors")
            unknown = [level for level in fresh.levels if level not in term.levels]
            if unknown:
                raise ValueError(
                    f"factor {term.term.call} has new levels "
                    + ", ".join(_strata_value_label(level) for level in unknown)
                )
            fresh = replace(fresh, levels=term.levels, names=term.names)
        covariates.append(fresh)
    return replace(design, covariates=tuple(covariates))


def _model_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
    weights: Any | None = None,
    offset: Any | None = None,
    strata_arg: Any | None = None,
    cluster: Any | None = None,
    id: Any | None = None,
    istate: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    deferred_na: bool = False,
    defer_tt: bool = False,
) -> _ModelFrame:
    """Evaluate a survival formula on ``data`` the way ``model.frame`` does.

    Vector arguments may name a column of ``data``; ``subset`` and ``na.action`` are
    applied to the formula's variables and to every vector argument together (``extra``
    carries any further row-aligned vectors, e.g. ``cch``'s ``subcoh``).
    ``deferred_na`` (with ``na_action="pass"``) leaves the missing values for the caller
    to drop, as coxph.R does for a formula list: a missing stratum has code -1 and a
    missing weight stays NaN.
    """

    if not isinstance(formula, str):
        raise TypeError("a formula argument is required")
    if data is None:
        raise ValueError("a data argument is required with a formula")
    full_data = data
    aligned = {
        "weights": _column_or_values(data, weights, "weights"),
        "offset": _column_or_values(data, offset, "offset"),
        "strata": _column_or_values(data, strata_arg, "strata"),
        "cluster": _column_or_values(data, cluster, "cluster"),
        "id": _column_or_values(data, id, "id"),
        "istate": _column_or_values(data, istate, "istate"),
        **{name: _column_or_values(data, value, name) for name, value in (extra or {}).items()},
    }
    if subset is not None:
        data, aligned = _subset_formula_inputs(formula, data, subset, **aligned)
    row_names = _data_row_labels(data, _data_row_count(data, formula))
    data, aligned, removed = _apply_formula_na_action(formula, data, na_action, **aligned)

    y, terms = _parse_formula(formula, data)
    n = len(y)
    if n == 0:
        raise ValueError("No (non-missing) observations")

    strata_codes: list[int] | None = None
    strata_levels: tuple[str, ...] = ()
    factor: StrataFactor | None = None
    if terms.strata:
        if aligned["strata"] is not None:
            raise ValueError("use only one of formula strata(...) or strata")
        factor = _strata_keep(data, _strata_specs(terms))
    elif aligned["strata"] is not None:
        factor = _strata([("strata", aligned["strata"])])
    if factor is not None:
        if len(factor.codes) != n:
            raise ValueError("strata columns must have the same length as the Surv response")
        if deferred_na:
            strata_codes = [-1 if code is None else code for code in factor.codes]
        else:
            strata_codes = _complete_codes(factor, "missing values in the strata")
        strata_levels = tuple(factor.levels)

    offset_values = _offset_vector(data, terms.offsets, n) if terms.offsets else None
    if aligned["offset"] is not None:
        if offset_values is not None:
            raise ValueError("use only one of formula offset(...) or offset")
        offset_values = _float_vector(aligned["offset"], "offset")
        if len(offset_values) != n:
            raise ValueError("offset must have the same length as the Surv response")

    cluster_values = aligned["cluster"]
    if terms.clusters:
        if cluster_values is not None:
            warnings.warn(
                "cluster appears both in a formula and as an argument, formula term ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            cluster_values = _formula_cluster_values(data, terms, n)
    cluster_levels = None if cluster_values is None else _categories(cluster_values)
    if cluster_values is not None:
        cluster_values = _materialize_labels(cluster_values, "cluster")
        if len(cluster_values) != n:
            raise ValueError("cluster must have the same length as the Surv response")

    id_values = aligned["id"]
    id_levels = None if id_values is None else _categories(id_values)
    if id_values is not None:
        id_values = _materialize_labels(id_values, "id")
        if len(id_values) != n:
            raise ValueError("id must have the same length as the Surv response")
    istate_values = aligned["istate"]
    if istate_values is not None:
        istate_values = _rows_of(istate_values, _materialize_labels(istate_values, "istate"))
        if len(istate_values) != n:
            raise ValueError("istate must have the same length as the Surv response")

    weight_values: list[float] | None
    if deferred_na and aligned["weights"] is not None:
        weight_values = _floats_or_nan(_materialize_1d(aligned["weights"], "weights"))
        if len(weight_values) != n:
            raise ValueError(f"weights must have length {n}")
        if any(math.isinf(value) for value in weight_values):
            raise ValueError("weights must be finite")
    else:
        weight_values = _optional_float_vector(aligned["weights"], "weights", n)
        if weight_values is not None and not all(map(math.isfinite, weight_values)):
            raise ValueError("weights must be finite")

    design = _r_factor_design(
        data,
        _fit_formula_design(
            data,
            _formula_response_spec(formula),
            terms,
            n,
            full_data=full_data,
            strata_margins=True,
        ),
    )
    names, assign = _design_names_and_assign(design)
    return _ModelFrame(
        formula=formula,
        data=data,
        y=y,
        x=_design_rows_from_spec(
            data,
            design,
            n,
            evaluated={term: [0.0] * n for term in _tt_terms(design)} if defer_tt else None,
        ),
        design=design,
        terms=terms,
        names=names,
        assign=assign,
        strata=strata_codes,
        strata_levels=strata_levels,
        offset=offset_values,
        weights=weight_values,
        cluster=cluster_values,
        cluster_levels=None if cluster_levels is None else tuple(cluster_levels),
        id=id_values,
        id_levels=None if id_levels is None else tuple(id_levels),
        istate=istate_values,
        extra={
            name: _materialize_labels(aligned[name], name)
            for name in (extra or {})
            if aligned[name] is not None
        },
        weights_column=weights if isinstance(weights, str) else None,
        id_column=id if isinstance(id, str) else None,
        na_action=_na_action_record(na_action, removed),
        row_names=row_names,
    )


def _tt_terms(design: _FormulaDesign) -> list[_CovariateTerm]:
    """The distinct ``tt(x)`` variables, in formula variable order."""

    used = list(
        dict.fromkeys(
            factor.term
            for term in design.covariates
            for factor in (term.factors if isinstance(term, _InteractionDesignTerm) else (term,))
            if factor.term.transform == "tt"
        )
    )
    return [term for term in design.variables if term in used] if design.variables else used


# ---------------------------------------------------------------------------
# newdata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NewData:
    """``model.frame(Terms2, newdata)``: the pieces a prediction needs, at the rows of
    ``newdata`` retained by its NA action; ``missing`` lists omitted rows (0-based).
    Prediction may keep NaN covariates and offsets so independent outputs remain
    available. ``data`` holds the model's variables at the kept rows."""

    data: Any
    x: list[list[float]]
    strata: list[int] | None
    offset: list[float] | None
    y: Surv | None
    missing: tuple[int, ...] = ()

    @property
    def n(self) -> int:
        return len(self.x)


def _newdata_columns(newdata: Any) -> list[Any]:
    if isinstance(newdata, Mapping):
        return list(newdata)
    columns = getattr(newdata, "columns", None)
    if columns is None:
        raise TypeError("newdata must be a data frame (a mapping of columns)")
    return list(columns)


def _newdata_response(newdata: Any, spec: _SurvResponseSpec | None) -> Surv | None:
    """The response evaluated on ``newdata`` when all of its columns are present."""

    if spec is None or not set(spec.columns) <= set(_newdata_columns(newdata)):
        return None
    args = _formula_response_values(newdata, spec)
    return Surv(*args, type=spec.type, origin=spec.origin)


def _newdata_frame(
    design: _FormulaDesign,
    strata_terms: Sequence[_StrataSpec],
    strata_levels: Sequence[str],
    newdata: Any,
    *,
    need_strata: bool,
    need_response: bool,
    na_action: str | None,
    allow_missing_predictors: bool = False,
    allow_missing_strata: bool = False,
    extra_missing: Sequence[int] = (),
) -> _NewData:
    """Evaluate the model terms on ``newdata`` (R's ``model.frame(Terms2, newdata,
    na.action)``).

    The ``strata()`` terms (the columns of each) are looked up only when ``need_strata``
    (R's ``found.strata``) and coded as the fit coded them (``strata.keep``);
    the response only when ``need_response`` (``predict(type='expected')``,
    ``survfit(id=)``); either is ``None`` when absent from ``newdata``.  A row with a
    missing value in one of these variables or in a covariate or offset variable (a
    NaN that ``log``, ``sqrt`` or arithmetic made included) is left out and listed in
    ``missing``: ``na.fail`` refuses it, and a prediction pads it back as NaN for
    ``na.pass`` and ``na.exclude``.
    With ``allow_missing_predictors``, ``na.pass`` carries covariate and offset
    NaNs into the prediction kernel. ``allow_missing_strata`` is used only when
    strata do not affect the requested prediction; missing codes then use a
    valid placeholder. Expected-count predictions ignore a missing event code.
    """

    present = set(_newdata_columns(newdata))
    strata_columns = [column for term in strata_terms for column in term.columns]
    if not (need_strata and set(strata_columns) <= present):
        strata_columns = []
    response_columns = (
        list(design.response.columns)
        if need_response and design.response is not None and set(design.response.columns) <= present
        else []
    )
    pass_missing = _normalize_na_action(na_action) == "pass"
    columns = list(
        dict.fromkeys(
            [
                *_formula_design_columns(design, include_unused=not pass_missing),
                *strata_columns,
                *response_columns,
            ]
        )
    )
    n = _formula_design_row_count(newdata, design)
    strata_specs = dict.fromkeys(
        [spec for spec in strata_terms if strata_columns]
        + [
            part.term.strata
            for term in design.covariates
            for part in (term.factors if isinstance(term, _InteractionDesignTerm) else (term,))
            if part.term.strata is not None
        ]
    )
    newdata = _with_strata_cache(newdata, tuple(strata_specs), n)
    # na.pass leaves missing unused variables in the model frame without
    # propagating them into the design. They still have to exist and align.
    if pass_missing:
        used = set(columns)
        _missing_row_indices(
            [
                (name, _column_source(newdata, name))
                for name in _formula_design_columns(design, include_unused=True)
                if name not in used
            ],
            n,
        )
    variables = [
        part.term
        for term in design.covariates
        for part in (term.factors if isinstance(term, _InteractionDesignTerm) else (term,))
    ]
    if not pass_missing:
        variables.extend(term for term in design.variables if term.strata is None)
    variables.extend(_strata_covariate(spec) for spec in strata_terms if strata_columns)
    missing = _formula_missing_rows(
        newdata, columns, [*variables, *design.offsets], n, required=response_columns
    )
    keep_missing = pass_missing and allow_missing_predictors
    response = None
    if keep_missing:
        missing = (
            _formula_missing_rows(
                newdata,
                strata_columns,
                [_strata_covariate(spec) for spec in strata_terms] if strata_columns else [],
                n,
            )
            if not allow_missing_strata
            else set()
        )
        if response_columns:
            # Expected counts use follow-up, not the event indicator.
            response = _newdata_response(newdata, design.response)
            if response is None:
                raise ValueError("newdata does not contain a survival response")
            missing.update(i for i, value in enumerate(response.time) if math.isnan(value))
            if response.start is not None:
                missing.update(i for i, value in enumerate(response.start) if math.isnan(value))
    missing.update(extra_missing)
    made, evaluated = _made_nan_rows(newdata, [*variables, *design.offsets], missing, n)
    if made and not keep_missing:
        # the design reads the evaluated variables at the rows that stay
        stays = [row not in made for row in range(n) if row not in missing]
        evaluated = {term: list(compress(values, stays)) for term, values in evaluated.items()}
        missing.update(made)
    if missing and _normalize_na_action(na_action) == "fail":
        raise ValueError("missing values in newdata")
    m = n - len(missing)
    if missing:
        kept = [row for row in range(n) if row not in missing]
        newdata = _data_rows(newdata, columns, kept, n)
        if response is not None:
            response = response.subset(kept)
    rows = _design_rows_from_spec(
        newdata, design, m, evaluated=evaluated, allow_missing=keep_missing
    )
    offset = _offset_vector(newdata, list(design.offsets), m, evaluated)
    strata_codes: list[int] | None = None
    if strata_columns:
        factor = _strata_keep(newdata, strata_terms)
        level_index = {level: idx for idx, level in enumerate(strata_levels)}
        try:
            remap = [level_index[level] for level in factor.levels]
        except KeyError as exc:
            raise ValueError("New data has a strata not found in the original model") from exc
        strata_codes = (
            [0 if code is None else remap[code] for code in factor.codes]
            if keep_missing and allow_missing_strata
            else [remap[code] for code in _complete_codes(factor, "missing values in the strata")]
        )
    y = (
        response
        if response is not None
        else (_newdata_response(newdata, design.response) if response_columns else None)
    )
    return _NewData(
        data=newdata,
        x=rows,
        strata=strata_codes,
        offset=offset,
        y=y,
        missing=tuple(sorted(missing)),
    )


# ---------------------------------------------------------------------------
# naresid / napredict
# ---------------------------------------------------------------------------


def _prediction_row_labels(
    labels: tuple[str, ...] | None, n: int, missing: Sequence[int], restore: bool
) -> list[str]:
    """Apply the numerical prediction's row mask to its original labels once."""
    source = labels if labels is not None else tuple(str(row + 1) for row in range(n))
    if restore or not missing:
        return list(source)
    omitted = set(missing)
    return [label for row, label in enumerate(source) if row not in omitted]


def _prediction_row_result(
    values: Any, fit_names: list[str] | None, se_names: list[str] | None
) -> dict[str, Any]:
    """Transfer a shared fit/error label vector to R once."""
    shared = fit_names is se_names
    return {
        "values": values,
        "fit_names": fit_names,
        "se_names": None if shared else se_names,
        "shared_names": shared,
    }


def _na_entry(width: int | None) -> Any:
    """``NA`` for one entry of a vector (``width`` None) or one row of a matrix."""

    return math.nan if width is None else [math.nan] * width


def _row_width(values: list[Any]) -> int | None:
    return len(values[0]) if values and isinstance(values[0], list) else None


def _pad_rows(values: list[Any], rows: Sequence[int], width: int | None = None) -> list[Any]:
    """R's ``naresid.exclude``: ``values`` (a vector, or a matrix as a list of rows)
    with NaN, or a row of NaN, inserted at the sorted 0-based ``rows`` of the result.
    ``width`` is the column count of a matrix without rows (``None`` for a vector)."""

    if not rows:
        return values
    if values:
        width = _row_width(values)
    gaps = set(rows)
    kept = iter(values)
    return [
        _na_entry(width) if row in gaps else next(kept) for row in range(len(values) + len(gaps))
    ]


def _excluded_rows(na_action: NaAction | None) -> list[int]:
    """The 0-based rows ``naresid``/``napredict`` give back to a fit's residuals and
    predictions (as NA): those ``na.exclude`` removed, none for ``na.omit``."""

    if na_action is None or na_action.kind != "exclude":
        return []
    return [row - 1 for row in na_action.rows]


def _rowsum_excluded(values: list[Any], codes: Sequence[int], excluded: Sequence[int]) -> list[Any]:
    """``rowsum(naresid(fit$na.action, rr), collapse)`` from ``values``, the rowsum of
    the fit's rows: ``codes`` are the 0-based groups of every row of the padded
    residuals, and a group with an excluded row sums to NA."""

    gaps = set(excluded)
    na_groups = {codes[row] for row in gaps}
    fitted = sorted({code for row, code in enumerate(codes) if row not in gaps})
    sums = dict(zip(fitted, values, strict=True))
    width = _row_width(values)
    return [
        _na_entry(width) if group in na_groups else sums[group] for group in range(max(codes) + 1)
    ]


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def _formula_design_for_fit(fit: Any) -> _FormulaDesign | None:
    return getattr(fit, "design", None)
