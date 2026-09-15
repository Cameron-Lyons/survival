"""The shared model-frame path (R's ``model.frame`` + ``model.matrix`` for a survival
formula) used by ``coxph``, ``cch``, ``aareg`` and ``concordance``, plus the survreg
accessors ``_survreg``/``_models`` still share."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _float_vector,
    _is_missing_value,
    _label_levels,
    _materialize_1d,
    _materialize_labels,
    _mstate_categories,
    _optional_float_vector,
    _strata_level_sort_key,
    _strata_value_label,
)
from ._formula import (
    _apply_formula_na_action,
    _column_or_values,
    _column_source,
    _combined_columns,
    _covariate_term_name,
    _design_rows_from_spec,
    _design_term_name,
    _design_term_output_names,
    _fit_formula_design,
    _formula_design_row_count,
    _formula_model_frame,
    _formula_response_spec,
    _formula_response_values,
    _offset_vector,
    _parse_formula,
    _subset_formula_inputs,
)
from ._surv import Surv
from ._types import (
    _CategoricalDesignTerm,
    _CovariateTerm,
    _DesignTerm,
    _FormulaDesign,
    _FormulaFit,
    _FormulaTerms,
    _InteractionDesignTerm,
    _NumericDesignTerm,
    _SingleDesignTerm,
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
    already row-aligned with ``y`` (after ``subset`` and ``na.action``).
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
    istate: list[Any] | None
    extra: dict[str, list[Any]] = field(default_factory=dict)

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
        )

    def strata_labels(self) -> list[str] | None:
        if self.strata is None:
            return None
        return [self.strata_levels[code] for code in self.strata]


def _r_levels(values: Any, levels: Sequence[Any]) -> tuple[Any, ...]:
    """``levels(factor(x))``: the column's own categories when it carries them
    (a pandas Categorical / R factor), else the sorted distinct values."""

    categories = _mstate_categories(values)
    if categories is not None:
        present = set(levels)
        return tuple(
            level for level in _materialize_1d(categories, "categories") if level in present
        )
    return tuple(sorted(levels, key=_strata_level_sort_key))


def _r_factor_design(data: Any, design: _FormulaDesign) -> _FormulaDesign:
    """Give every categorical term R's factor level order (the formula module
    keeps first-appearance order)."""

    def relevel(term: _SingleDesignTerm) -> _SingleDesignTerm:
        if not isinstance(term, _CategoricalDesignTerm):
            return term
        return replace(term, levels=_r_levels(_column_source(data, term.term.column), term.levels))

    covariates: list[_DesignTerm] = []
    for term in design.covariates:
        if isinstance(term, _InteractionDesignTerm):
            covariates.append(replace(term, factors=tuple(relevel(f) for f in term.factors)))
        else:
            covariates.append(relevel(term))
    return replace(design, covariates=tuple(covariates))


def _column_names(term: _SingleDesignTerm) -> list[str]:
    """``colnames(model.matrix)`` for one term: ``factor(x)level`` with R's labels."""

    prefix = _covariate_term_name(term.term)
    if isinstance(term, _CategoricalDesignTerm):
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


def _factor(column: Any, name: str) -> tuple[list[str], list[int | None]]:
    """R's ``factor(x)``: the levels (as character) and 0-based codes (None = NA)."""

    values = _materialize_1d(column, name)
    present = [value for value in values if not _is_missing_value(value)]
    levels = _r_levels(column, _label_levels(present, name))
    index = {level: idx for idx, level in enumerate(levels)}
    codes = [None if _is_missing_value(value) else index[value] for value in values]
    return [_strata_value_label(level) for level in levels], codes


def _is_character(column: Any) -> bool:
    """``is.character(x) | is.factor(x)``: what makes ``strata()`` drop the ``name=`` prefix."""

    if _mstate_categories(column) is not None:
        return True
    values = _materialize_1d(column, "strata")
    return all(isinstance(value, str) or _is_missing_value(value) for value in values)


def _strata_factor(
    columns: Mapping[str, Any], n: int, *, shortlabel: bool | None = None
) -> _core.StrataResult:
    """``strata(mf[, vars])``: R's labels (``name=level``, or the bare level when
    every variable is character/factor) and compact codes."""

    if any(len(_materialize_1d(column, name)) != n for name, column in columns.items()):
        raise ValueError("strata columns must have the same length as the Surv response")
    factors = [_factor(column, name) for name, column in columns.items()]
    if shortlabel is None:
        shortlabel = all(_is_character(column) for column in columns.values())
    return _core.strata(
        list(columns),
        [levels for levels, _codes in factors],
        [codes for _levels, codes in factors],
        shortlabel=shortlabel,
    )


def _model_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None = None,
    na_action: str | None = "fail",
    weights: Any | None = None,
    offset: Any | None = None,
    strata_arg: Any | None = None,
    cluster: Any | None = None,
    id: Any | None = None,
    istate: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> _ModelFrame:
    """Evaluate a survival formula on ``data`` the way ``model.frame`` does.

    Vector arguments may name a column of ``data``; ``subset`` and ``na.action`` are
    applied to the data and to every vector argument together (``extra`` carries any
    further row-aligned vectors, e.g. ``cch``'s ``subcoh``).
    """

    if not isinstance(formula, str):
        raise TypeError("a formula argument is required")
    if data is None:
        raise ValueError("a data argument is required with a formula")
    if isinstance(data, Mapping):  # the bundled datasets carry _nrow/_ncol metadata
        data = {key: value for key, value in data.items() if not str(key).startswith("_")}
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
    data, aligned = _apply_formula_na_action(formula, data, na_action, **aligned)

    y, terms = _parse_formula(formula, data)
    n = len(y)
    if n == 0:
        raise ValueError("No (non-missing) observations")

    strata_codes: list[int] | None = None
    strata_levels: tuple[str, ...] = ()
    if terms.strata:
        if aligned["strata"] is not None:
            raise ValueError("use only one of formula strata(...) or strata")
        factor = _strata_factor({name: _column_source(data, name) for name in terms.strata}, n)
        strata_codes = [int(code) for code in factor.codes]
        strata_levels = tuple(factor.levels)
    elif aligned["strata"] is not None:
        factor = _strata_factor({"strata": _materialize_labels(aligned["strata"], "strata")}, n)
        strata_codes = [int(code) for code in factor.codes]
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
            cluster_values = _combined_columns(data, terms.clusters, n)
    if cluster_values is not None:
        cluster_values = _materialize_labels(cluster_values, "cluster")
        if len(cluster_values) != n:
            raise ValueError("cluster must have the same length as the Surv response")

    id_values = aligned["id"]
    if id_values is not None:
        id_values = _materialize_labels(id_values, "id")
        if len(id_values) != n:
            raise ValueError("id must have the same length as the Surv response")
    istate_values = aligned["istate"]
    if istate_values is not None:
        istate_values = _materialize_labels(istate_values, "istate")
        if len(istate_values) != n:
            raise ValueError("istate must have the same length as the Surv response")

    weight_values = _optional_float_vector(aligned["weights"], "weights", n)
    if weight_values is not None and not all(math.isfinite(value) for value in weight_values):
        raise ValueError("weights must be finite")

    design = _r_factor_design(
        data, _fit_formula_design(data, _formula_response_spec(formula), terms, n)
    )
    names, assign = _design_names_and_assign(design)
    return _ModelFrame(
        formula=formula,
        data=data,
        y=y,
        x=_design_rows_from_spec(data, design, n),
        design=design,
        terms=terms,
        names=names,
        assign=assign,
        strata=strata_codes,
        strata_levels=strata_levels,
        offset=offset_values,
        weights=weight_values,
        cluster=cluster_values,
        id=id_values,
        istate=istate_values,
        extra={
            name: _materialize_labels(aligned[name], name)
            for name in (extra or {})
            if aligned[name] is not None
        },
    )


def _tt_terms(design: _FormulaDesign) -> list[_CovariateTerm]:
    """The ``tt(x)`` terms of a design, in model-matrix column order."""

    return [
        term.term
        for term in design.covariates
        if isinstance(term, _NumericDesignTerm) and term.term.transform == "tt"
    ]


# ---------------------------------------------------------------------------
# newdata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NewData:
    """``model.frame(Terms2, newdata)``: the pieces a prediction needs."""

    x: list[list[float]]
    strata: list[int] | None
    offset: list[float] | None
    y: Surv | None

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


def _newdata_response(newdata: Any, spec: _SurvResponseSpec) -> Surv | None:
    """The response evaluated on ``newdata`` when all of its columns are present."""

    if not set(spec.columns) <= set(_newdata_columns(newdata)):
        return None
    args = _formula_response_values(newdata, spec)
    return Surv(*args, type=spec.type, origin=spec.origin)


def _newdata_frame(
    design: _FormulaDesign,
    strata_terms: Sequence[str],
    strata_levels: Sequence[str],
    newdata: Any,
    *,
    need_strata: bool,
    need_response: bool,
) -> _NewData:
    """Evaluate the model terms on ``newdata`` (R's ``model.frame(Terms2, newdata)``).

    Strata columns are looked up only when ``need_strata`` (R's ``found.strata``),
    the response only when ``need_response`` (``predict(type='expected')``,
    ``survfit(id=)``); either is ``None`` when absent from ``newdata``.
    """

    n = _formula_design_row_count(newdata, design)
    rows = _design_rows_from_spec(newdata, design, n)
    offset = _offset_vector(newdata, list(design.offsets), n)
    strata_codes: list[int] | None = None
    if need_strata and strata_terms and set(strata_terms) <= set(_newdata_columns(newdata)):
        factor = _strata_factor({name: _column_source(newdata, name) for name in strata_terms}, n)
        level_index = {level: idx for idx, level in enumerate(strata_levels)}
        try:
            strata_codes = [level_index[factor.levels[int(code)]] for code in factor.codes]
        except KeyError as exc:
            raise ValueError("New data has a strata not found in the original model") from exc
    y = _newdata_response(newdata, design.response) if need_response else None
    return _NewData(x=rows, strata=strata_codes, offset=offset, y=y)


# ---------------------------------------------------------------------------
# survreg accessors (transitional: shared with _survreg/_models until survreg moves
# to its typed wrapper)
# ---------------------------------------------------------------------------


def _unwrap_formula_fit(fit: Any) -> Any:
    return fit.fit if isinstance(fit, _FormulaFit) else fit


def _formula_design_for_fit(fit: Any) -> _FormulaDesign | None:
    return getattr(fit, "design", None)


def _is_survreg_fit(fit: Any) -> bool:
    return hasattr(fit, "n_covariates") and hasattr(fit, "location_coefficients")


def _location_beta(fit: Any) -> list[float]:
    values = getattr(fit, "location_coefficients", None)
    if values is None:
        values = fit.coefficients
    return [float(value) for value in values]


def _fallback_coef_names(width: int) -> list[str]:
    return [f"x{idx + 1}" for idx in range(width)]


def _formula_design_output_names(design: _FormulaDesign) -> list[str]:
    names = [name for term in design.covariates for name in _design_term_output_names(term)]
    if design.intercept:
        names.insert(0, "(Intercept)")
    return names


def _fit_location_coef_names(fit: Any, width: int) -> list[str]:
    design = _formula_design_for_fit(fit)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            return names
    coefficient_names = getattr(fit, "coefficient_names", None)
    if coefficient_names is not None and len(coefficient_names) == width:
        return list(coefficient_names)
    return _fallback_coef_names(width)


def _survreg_scale_coef_names(fit: Any, width: int) -> list[str]:
    if width <= 0:
        return []
    if width == 1:
        return ["Log(scale)"]
    design = _formula_design_for_fit(fit)
    if design is not None and len(design.strata_levels) == width:
        return [f"Log(scale:{level})" for level in design.strata_levels]
    return [f"Log(scale{idx + 1})" for idx in range(width)]


def _cox_training_rows(fit: Any, nvar: int) -> list[list[float]]:
    covariates = getattr(fit, "x", None)
    if covariates is None:
        return []
    rows = [[float(value) for value in row] for row in covariates]
    if any(len(row) != nvar for row in rows):
        return []
    return rows


def _location_variance_matrix(fit: Any, nvar: int) -> list[list[float]]:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:nvar]] for row in list(raw_variance)[:nvar]]
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        raise ValueError("fitted survreg variance matrix does not match coefficient width")
    return variance


def _survreg_has_variance_width(fit: Any, width: int) -> bool:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        return False
    matrix = list(raw_variance)
    return len(matrix) >= width and all(len(row) >= width for row in matrix[:width])


def _survreg_variance_matrix(fit: Any, width: int) -> list[list[float]]:
    if width == 0:
        return []
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:width]] for row in list(raw_variance)[:width]]
    if len(variance) != width or any(len(row) != width for row in variance):
        raise ValueError("fitted survreg variance matrix does not match residual width")
    return variance


def _survreg_scales(fit: Any) -> list[float]:
    values = getattr(fit, "scales", None)
    if values is None:
        values = [getattr(fit, "scale", 1.0)]
    scales = [_finite_float(value, "scale") for value in values]
    if not scales:
        raise ValueError("fitted survreg model does not expose scale values")
    return scales


def _survreg_strata(fit: Any, n: int, nstrata: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata_codes = [int(value) for value in values]
    if len(strata_codes) != n:
        raise ValueError("fitted survreg strata do not match training rows")
    if any(value < 0 or value >= nstrata for value in strata_codes):
        raise ValueError("fitted survreg strata reference missing scale values")
    return strata_codes
