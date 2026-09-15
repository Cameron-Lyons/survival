"""Parametric accelerated failure time models: R's ``survreg`` and its methods.

Mirrors ``survreg.R``, ``survreg.control.R``, ``survreg.distributions.R``, ``survregDtest.R``,
``predict.survreg.R``, ``residuals.survreg.R``, ``anova.survreg.R``/``anova.survreglist.R``,
``summary.survreg.R`` and ``dsurvreg.R`` from R survival 3.8.  Every numeric kernel
(``survreg.fit``/``survreg6``, the prediction and residual derivative matrices, the
location-scale d/p/q/r functions) lives in the Rust core; this module does what R's R code
does: it builds the model frame and design, resolves the distribution and control arguments,
calls the kernel and labels the result.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_rows,
    _control_mapping,
    _encode_labels,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _keep_rows_after_na_action,
    _materialize_1d,
    _materialize_labels,
    _matrix_input_column_names,
    _missing_row_indices,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_optional_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
    _quantile_vector,
    _strata_level_sort_key,
    _strata_value_label,
    _subset_data,
    _subset_optional_sequence,
)
from ._fit import (
    _fallback_coef_names,
    _fit_location_coef_names,
    _formula_design_for_fit,
    _formula_design_output_names,
    _location_beta,
    _unwrap_formula_fit,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _design_rows_from_spec,
    _design_term_name,
    _design_term_output_names,
    _fit_formula_design,
    _formula_columns,
    _formula_design_row_count,
    _formula_model_frame,
    _formula_model_term_degree,
    _formula_response_spec,
    _formula_response_values,
    _offset_vector,
    _parse_formula,
    _subset_formula_inputs,
)
from ._surv import Surv, _survreg_response_arrays
from ._types import (
    PredictResult,
    _FormulaDesign,
    _FormulaFit,
    _ModelCovariateTerm,
    _ModelStrataTerm,
)

SurvregDistribution = _core.SurvregDistribution
SurvregControl = _core.SurvregControl

# R's ``survreg.distributions``: the built-in location-scale families keyed by the names
# ``survreg(dist=)`` matches against.  Users may add entries (a ``SurvregDistribution``).
_BUILTIN_DISTRIBUTIONS = (
    "extreme",
    "logistic",
    "gaussian",
    "weibull",
    "exponential",
    "rayleigh",
    "loggaussian",
    "lognormal",
    "loglogistic",
    "t",
)
survreg_distributions: dict[str, Any] = {
    name: _core.SurvregDistribution(name) for name in _BUILTIN_DISTRIBUTIONS
}

_TRANSFORMS = {"log": _core.SurvregTransform.Log, "identity": _core.SurvregTransform.Identity}


# --- results -----------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class SurvregModelResult(_FormulaFit):
    """R's ``survreg`` object: the Rust ``SurvregFit`` plus the metadata R keeps on it.

    Attribute access falls through to the Rust fit (``scale``, ``linear_predictors``,
    ``icoef``, ``means``, ``df``, ``df_residual``, ``iterations``, ``converged``, ...).
    ``coefficients`` are the location coefficients as in R (``NaN`` where singular);
    the full vector with the ``Log(scale)`` entries is ``fit.coefficients``.
    """

    dist: Any = "weibull"
    control: Any = None
    assign: tuple[int, ...] = ()
    term_labels: tuple[str, ...] = ()
    strata_term: int = 0
    strata_columns: tuple[str, ...] = ()
    strata_levels: tuple[str, ...] = ()

    @property
    def coefficients(self) -> list[float]:
        return _location_beta(self.fit)

    @property
    def loglik(self) -> list[float]:
        return [float(self.fit.intercept_only_log_likelihood), float(self.fit.log_likelihood)]

    @property
    def var(self) -> list[list[float]]:
        return self.fit.variance_matrix

    @property
    def robust(self) -> bool:
        return self.fit.naive_variance_matrix is not None

    @property
    def parms(self) -> list[float] | None:
        return list(self.fit.distribution.parms) or None

    @property
    def idf(self) -> int:
        return 1 + _estimated_scale_count(self.fit)

    @property
    def iter(self) -> int:
        return int(self.fit.iterations)

    def __repr__(self) -> str:
        formula = f"formula={self.formula!r}, " if self.formula is not None else ""
        return f"SurvregModelResult({formula}{self.fit!r})"


@dataclass(frozen=True)
class SurvregAnovaResult:
    """R's ``anova.survreg``/``anova.survreglist`` table (an ``anova`` data frame)."""

    terms: list[str]
    df: list[float]
    deviance: list[float]
    resid_df: list[int]
    loglik: list[float]
    p: list[float] | None
    heading: list[str]
    test_labels: list[str] | None = None

    def frame(self) -> dict[str, list[Any]]:
        """The table as columns, with R's column names."""

        columns: dict[str, list[Any]] = {}
        if self.test_labels is None:
            columns["Df"] = self.df
            columns["Deviance"] = self.deviance
            columns["Resid. Df"] = self.resid_df
            columns["-2*LL"] = self.loglik
        else:
            columns["Terms"] = self.terms
            columns["Resid. Df"] = self.resid_df
            columns["-2*LL"] = self.loglik
            columns["Test"] = self.test_labels
            columns["Df"] = self.df
            columns["Deviance"] = self.deviance
        if self.p is not None:
            columns["Pr(>Chi)"] = self.p
        return columns


# --- distributions and control -------------------------------------------------------------


def _match_arg(value: Any, name: str, choices: Sequence[str]) -> str:
    """R's ``match.arg``: exact or unique-prefix match, with R's error message."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value in choices:
        return value
    matches = [choice for choice in choices if choice.startswith(value)]
    if len(matches) != 1:
        options = ", ".join(f'"{choice}"' for choice in choices)
        raise ValueError(f"'{name}' should be one of {options}")
    return matches[0]


def _parms_vector(parms: Any | None) -> list[float] | None:
    if parms is None:
        return None
    values = list(parms.values()) if isinstance(parms, Mapping) else parms
    return _quantile_vector(values, "parms")


def _distribution_from_list(dlist: Mapping[str, Any]) -> Any:
    """A user distribution given as R's list: a transform of a built-in family."""

    name = dlist.get("name")
    if not isinstance(name, str):
        raise ValueError("Invalid distribution object: Missing a name")
    base = dlist.get("dist")
    if not isinstance(base, str):
        raise ValueError("custom densities are not supported; give 'dist' (a built-in name)")
    reference = _resolve_distribution(base, None)
    trans = dlist.get("trans", "identity")
    if not isinstance(trans, str) or trans not in _TRANSFORMS:
        raise ValueError("trans must be 'log' or 'identity'")
    scale = dlist.get("scale")
    return _core.SurvregDistribution.custom(
        name,
        reference.family,
        _TRANSFORMS[trans],
        None if scale is None else _finite_float(scale, "scale"),
        _parms_vector(dlist.get("parms")),
    )


def _resolve_distribution(dist: Any, parms: Any | None) -> Any:
    """``survreg``'s distribution lookup: name (``match.arg``), list, or object; then parms."""

    if isinstance(dist, str):
        key = _match_arg(dist, "dist", tuple(survreg_distributions))
        dist = survreg_distributions[key]
        if parms is not None and key in _BUILTIN_DISTRIBUTIONS:
            return _core.SurvregDistribution(key, _parms_vector(parms))
    elif isinstance(dist, Mapping):
        dist = _distribution_from_list(dist)
    elif not isinstance(dist, _core.SurvregDistribution):
        raise TypeError("Invalid distribution object")
    errors = dist.dtest()
    if errors:
        raise ValueError("Invalid distribution object: " + "; ".join(errors))
    if parms is None:
        return dist
    if not dist.parms:
        raise ValueError(f"{dist.name} distribution has no optional parameters")
    return _core.SurvregDistribution.custom(
        dist.name, dist.family, dist.transform, dist.scale, _parms_vector(parms)
    )


def survregDtest(dlist: Any, verbose: bool = False) -> bool | list[str]:
    """Check a distribution object; ``True``, or the problems when ``verbose``."""

    try:
        distribution = _distribution_from_list(dlist) if isinstance(dlist, Mapping) else dlist
        errors = list(_core.survreg_dtest(distribution))
    except (TypeError, ValueError) as exc:
        errors = [str(exc)]
    if not errors:
        return True
    return errors if verbose else False


def survreg_control(
    maxiter: Any = 30,
    rel_tolerance: Any = 1e-9,
    toler_chol: Any = 1e-10,
    iter_max: Any | None = None,
    debug: Any = 0,
    outer_max: Any = 10,
    **dotted: Any,
) -> Any:
    """R's ``survreg.control`` (``max_iter``/``eps``/``tol_chol`` are accepted aliases);
    ``debug`` and ``outer.max`` are accepted and unused as in R."""

    for alias in ("rel.tolerance", "eps"):
        rel_tolerance = _pop_dotted_keyword(dotted, alias, "rel_tolerance", rel_tolerance, 1e-9)
    for alias in ("toler.chol", "tol_chol"):
        toler_chol = _pop_dotted_keyword(dotted, alias, "toler_chol", toler_chol, 1e-10)
    for alias in ("iter.max", "max_iter"):
        iter_max = _pop_dotted_keyword(dotted, alias, "iter_max", iter_max, None)
    _pop_dotted_keyword(dotted, "outer.max", "outer_max", outer_max, 10)
    if dotted:
        raise TypeError(f"unused argument(s): {', '.join(sorted(dotted))}")
    iterations = _integer_scalar(maxiter if iter_max is None else iter_max, "iter.max")
    if iterations < 0:
        raise ValueError("iter.max must be non-negative")
    return _core.SurvregControl(
        iter_max=iterations,
        rel_tolerance=_finite_float(rel_tolerance, "rel.tolerance"),
        toler_chol=_finite_float(toler_chol, "toler.chol"),
    )


def _resolve_control(control: Any | None, options: dict[str, Any]) -> Any:
    if control is None:
        return survreg_control(**options)
    if options:
        raise TypeError(f"unused argument(s) with control: {', '.join(sorted(options))}")
    if isinstance(control, _core.SurvregControl):
        return control
    return survreg_control(**_control_mapping(control, "control"))


# --- the model frame -----------------------------------------------------------------------


@dataclass(frozen=True)
class _SurvregFrame:
    """What R's ``model.frame`` + ``model.matrix`` step yields inside ``survreg``."""

    response: Surv
    x: list[list[float]]
    names: tuple[str, ...]
    design: _FormulaDesign | None = None
    assign: tuple[int, ...] = ()
    term_labels: tuple[str, ...] = ()
    strata_term: int = 0
    strata: list[int] | None = None
    strata_levels: tuple[str, ...] = ()
    weights: list[float] | None = None
    offset: list[float] | None = None
    cluster: list[Any] | None = None
    model: dict[str, Any] | None = None


def _is_categorical(values: Sequence[Any]) -> bool:
    return hasattr(values, "categories") or any(isinstance(value, str) for value in values)


def _column_levels(values: Sequence[Any]) -> list[Any]:
    categories = getattr(values, "categories", None)
    if categories is not None:
        return list(categories)
    return sorted({value for value in values if value is not None}, key=_strata_level_sort_key)


def _strata_factor(data: Any, columns: Sequence[str], n: int) -> tuple[list[int], tuple[str, ...]]:
    """R's ``strata(m[, vars])``: 0-based codes and level labels (``name=level, ...``)."""

    values = [_column(data, name) for name in columns]
    shortlabel = all(_is_categorical(column) for column in values)
    codes = [0] * n
    labels = [""] * n
    for term, (name, column) in enumerate(zip(columns, values, strict=True)):
        levels = _column_levels(column)
        level_labels = [_strata_value_label(level) for level in levels]
        if not shortlabel:
            level_labels = [f"{name}={label}" for label in level_labels]
            if term:
                width = max(len(label) for label in level_labels)
                level_labels = [label.ljust(width) for label in level_labels]
        position = {level: idx for idx, level in enumerate(levels)}
        for row in range(n):
            if column[row] is None:
                raise ValueError("strata contains missing values")
            code = position[column[row]]
            codes[row] = codes[row] * len(levels) + code
            labels[row] = level_labels[code] if not term else f"{labels[row]}, {level_labels[code]}"
    observed = sorted(set(codes))
    lookup = {code: idx for idx, code in enumerate(observed)}
    level_names = tuple(labels[codes.index(code)] for code in observed)
    return [lookup[code] for code in codes], level_names


def _term_structure(
    design: _FormulaDesign, model_terms: Sequence[Any]
) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    """R's ``attr(X, "assign")`` per column, ``term.labels`` (strata included) and the
    1-based position of the strata term (0 when absent)."""

    ordered = sorted(model_terms, key=_formula_model_term_degree)
    labels = [""] * len(ordered)
    assignments = (
        design.term_assignments
        if len(design.term_assignments) == len(design.covariates)
        else tuple(range(1, len(design.covariates) + 1))
    )
    for term, term_index in zip(design.covariates, assignments, strict=True):
        labels[term_index - 1] = _design_term_name(term)
    strata_term = 0
    for term_index, model_term in enumerate(ordered, start=1):
        if isinstance(model_term, _ModelStrataTerm):
            labels[term_index - 1] = f"strata({', '.join(model_term.columns)})"
            strata_term = term_index
    assign = [0] * int(design.intercept)
    for term, term_index in zip(design.covariates, assignments, strict=True):
        assign.extend([term_index] * len(_design_term_output_names(term)))
    return tuple(assign), tuple(labels), strata_term


def _drop_interval2_missing(
    spec: Any, data: Any, na_action: str | None, **row_aligned: Any
) -> tuple[Any, dict[str, Any]]:
    """An interval2 endpoint that is NA means censoring: only a row with both endpoints
    missing is an NA response (``is.na.Surv``).  Those rows, and rows with a missing
    weight/offset/cluster, go through ``na.action`` here; the covariates go through the
    shared formula path with the response columns excluded."""

    if spec.type != "interval2":
        return data, row_aligned
    left, right = _formula_response_values(data, spec)[:2]
    n = len(left)
    missing = {
        row for row, ends in enumerate(zip(left, right, strict=True)) if ends == (None, None)
    }
    missing |= _missing_row_indices(
        [(name, values) for name, values in row_aligned.items() if values is not None], n
    )
    keep = _keep_rows_after_na_action(missing, n, na_action, "formula data")
    if keep is None:
        return data, row_aligned
    return _subset_data(data, keep), {
        name: _subset_optional_sequence(values, keep, name) for name, values in row_aligned.items()
    }


def _formula_frame(
    formula: str,
    data: Any,
    *,
    weights: Any | None,
    subset: Any | None,
    na_action: str | None,
    offset: Any | None,
    cluster: Any | None,
    keep_model: bool,
) -> _SurvregFrame:
    spec = _formula_response_spec(formula)
    weights = _column_or_values(data, weights, "weights")
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula, data, subset, weights=weights, offset=offset, cluster=cluster
        )
        weights, offset, cluster = aligned["weights"], aligned["offset"], aligned["cluster"]
    data, aligned = _drop_interval2_missing(
        spec, data, na_action, weights=weights, offset=offset, cluster=cluster
    )
    excluded = spec.columns if spec.type == "interval2" else ()
    if any(column not in excluded for column in _formula_columns(formula, data)):
        data, aligned = _apply_formula_na_action(
            formula, data, na_action, exclude_columns=excluded, **aligned
        )
    weights, offset, cluster = aligned["weights"], aligned["offset"], aligned["cluster"]
    response, terms = _parse_formula(formula, data)
    n = len(response)
    design = _fit_formula_design(data, spec, terms, n, include_intercept=True)
    if terms.offsets:
        if offset is not None:
            raise ValueError("use only one of formula offset(...) or offset")
        offset = _offset_vector(data, terms.offsets, n)
    if terms.clusters:
        if len(terms.clusters) > 1:
            raise ValueError("a formula cannot have multiple cluster terms")
        if cluster is not None:
            warnings.warn(
                "cluster appears both in a formula and as an argument, formula term ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            cluster = _column(data, terms.clusters[0])
    strata, strata_levels = _strata_factor(data, terms.strata, n) if terms.strata else (None, ())
    model_terms = [
        term
        for term in terms.model_terms
        if isinstance(term, _ModelCovariateTerm | _ModelStrataTerm)
    ]
    assign, term_labels, strata_term = _term_structure(design, model_terms)
    return _SurvregFrame(
        response=response,
        x=_design_rows_from_spec(data, design, n),
        names=tuple(_formula_design_output_names(design)),
        design=design,
        assign=assign,
        term_labels=term_labels,
        strata_term=strata_term,
        strata=strata,
        strata_levels=strata_levels,
        weights=_optional_float_vector(weights, "weights", n),
        offset=_optional_float_vector(offset, "offset", n),
        cluster=_materialize_labels(cluster, "cluster") if cluster is not None else None,
        model=_formula_model_frame(
            data,
            response,
            design,
            extra_columns=tuple(terms.clusters),
            weights=weights,
            offset=offset,
            strata=strata,
            cluster=cluster,
        )
        if keep_model
        else None,
    )


def _matrix_frame(
    response: Surv, x: Any, *, weights: Any | None, offset: Any | None, cluster: Any | None
) -> _SurvregFrame:
    """``survreg(<Surv>, x=<matrix or data frame>)``: the design is used as given."""

    rows = _as_rows(x, "x")
    n = len(response)
    if len(rows) != n:
        raise ValueError("x must have the same number of rows as the Surv response")
    names = _matrix_input_column_names(x)
    if names is None or len(names) != len(rows[0]):
        names = tuple(f"x{idx + 1}" for idx in range(len(rows[0])))
    return _SurvregFrame(
        response=response,
        x=rows,
        names=names,
        assign=tuple(range(1, len(names) + 1)),
        term_labels=names,
        weights=_optional_float_vector(weights, "weights", n),
        offset=_optional_float_vector(offset, "offset", n),
        cluster=_materialize_labels(cluster, "cluster") if cluster is not None else None,
    )


# --- survreg -------------------------------------------------------------------------------


def survreg(
    formula: str | Surv | None = None,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    dist: Any = "weibull",
    init: Any | None = None,
    scale: Any = 0,
    control: Any | None = None,
    parms: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = True,
    robust: Any | None = None,
    cluster: Any | None = None,
    score: Any = False,
    offset: Any | None = None,
    **kwargs: Any,
) -> SurvregModelResult:
    """Fit a parametric survival regression model, like R's ``survreg``.

    ``formula`` is an R formula string (``"Surv(time, status) ~ age + strata(sex)"``) or a
    ``Surv`` response with the design given as ``x`` (the R bridge passes it as
    ``response=``); the remaining arguments are R's (``dist`` accepts a name, a
    ``SurvregDistribution`` or an R-style list; extra keywords are ``survreg.control``
    options).
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    formula = _pop_dotted_keyword(kwargs, "response", "formula", formula, None)
    control = _resolve_control(control, kwargs)
    keep_model = _normalize_bool_option_with_default(model, "model", False)
    keep_y = _normalize_bool_option_with_default(y, "y", True)
    keep_score = _normalize_bool_option_with_default(score, "score", False)
    robust_value = _normalize_optional_bool_option(robust, "robust")
    scale_value = _finite_float(scale, "scale")

    if isinstance(formula, str):
        keep_x = _normalize_bool_option_with_default(x, "x", False)
        frame = _formula_frame(
            formula,
            data,
            weights=weights,
            subset=subset,
            na_action=na_action,
            offset=offset,
            cluster=cluster,
            keep_model=keep_model,
        )
    elif isinstance(formula, Surv):
        if subset is not None or na_action not in {None, "fail", "pass"}:
            raise ValueError("subset and na_action require a formula")
        keep_x = True
        frame = _matrix_frame(formula, x, weights=weights, offset=offset, cluster=cluster)
    else:
        raise TypeError("a formula argument is required")
    response = frame.response
    if response.type == "counting":
        raise ValueError("start-stop type Surv objects are not supported")
    if response.type in {"mright", "mcounting"}:
        raise ValueError("multi-state survival is not supported")

    distribution = _resolve_distribution(dist, parms)
    if distribution.scale is not None and scale_value != 0.0:
        warnings.warn(
            f"{distribution.name} has a fixed scale, user specified value ignored",
            RuntimeWarning,
            stacklevel=2,
        )
        scale_value = 0.0
    if scale_value < 0.0:
        raise ValueError("Invalid scale value")
    if scale_value > 0.0 and frame.strata is not None and len(frame.strata_levels) > 1:
        raise ValueError("The scale argument is not valid with multiple strata")

    time, status, time2 = _survreg_response_arrays(response)
    cluster_codes = (  # as.numeric(as.factor(cluster)); unused when robust = FALSE
        _encode_labels(frame.cluster, "cluster")
        if frame.cluster is not None and robust_value is not False
        else None
    )
    fit = _core.survreg_fit(
        _core.SurvregData(
            time,
            [int(code) for code in status],
            frame.x,
            time2=time2,
            weights=frame.weights,
            offset=frame.offset,
            strata=frame.strata,
            cluster=cluster_codes,
        ),
        distribution,
        init=_float_vector(init, "init") if init is not None else None,
        scale=scale_value,
        control=control,
        robust=robust_value,
    )
    if control.iter_max > 1 and not fit.converged:
        warnings.warn("Ran out of iterations and did not converge", RuntimeWarning, stacklevel=2)

    return SurvregModelResult(
        fit=fit,
        design=frame.design,
        formula=formula if isinstance(formula, str) else None,
        coefficient_names=frame.names,
        case_weights=frame.weights,
        naive_variance=fit.naive_variance_matrix,
        cluster=frame.cluster if fit.naive_variance_matrix is not None else None,
        x_matrix=frame.x if keep_x else None,
        y_response=response if keep_y else None,
        model_frame=frame.model,
        score_values=list(fit.score) if keep_score else None,
        n_observations=len(response),
        dist=dist if isinstance(dist, str) else distribution,
        control=control,
        assign=frame.assign,
        term_labels=frame.term_labels,
        strata_term=frame.strata_term,
        strata_columns=tuple(frame.design.strata) if frame.design is not None else (),
        strata_levels=frame.strata_levels,
    )


# --- accessors used by the generics ----------------------------------------------------------


def _estimated_scale_count(model: Any) -> int:
    return len(model.coefficients) - len(model.means)


def survreg_scale_names(fit: Any) -> list[str]:
    """Names of the ``Log(scale)`` coefficients: ``survreg.fit`` repeats ``Log(scale)``."""

    return ["Log(scale)"] * _estimated_scale_count(_unwrap_formula_fit(fit))


def survreg_summary_names(fit: Any) -> list[str]:
    """Row names of ``summary.survreg``'s table: the scale rows carry the strata labels."""

    model = _unwrap_formula_fit(fit)
    names = _fit_location_coef_names(fit, len(model.means))
    scales = _estimated_scale_count(model)
    levels = getattr(fit, "strata_levels", ())
    if scales > 1 and len(levels) == scales:
        return names + list(levels)
    return names + ["Log(scale)"] * scales


def survreg_vcov(fit: Any, complete: bool = True) -> list[list[float]]:
    """``fit$var``; the location block only when ``complete`` is false."""

    model = _unwrap_formula_fit(fit)
    variance = [[float(value) for value in row] for row in model.variance_matrix]
    if complete:
        return variance
    nvar = len(model.means)
    return [row[:nvar] for row in variance[:nvar]]


def survreg_summary(fit: Any) -> dict[str, Any]:
    """The pieces of ``summary.survreg`` that are not the coefficient table."""

    model = _unwrap_formula_fit(fit)
    distribution = model.distribution
    parms = list(distribution.parms)
    summary: dict[str, Any] = {
        "location_coefficients": _location_beta(model),
        "location_coefficient_names": _fit_location_coef_names(fit, len(model.means)),
        "scale": model.scale[0] if len(model.scale) == 1 else list(model.scale),
        "scales": list(model.scale),
        "distribution": distribution.name,
        "parms": (
            f"{distribution.name} distribution: parmameters= {' '.join(map(str, parms))}"
            if parms
            else f"{distribution.name} distribution"
        ),
        "chi": 2.0 * (model.log_likelihood - model.intercept_only_log_likelihood),
        "iter": int(model.iterations),
        "idf": 1 + _estimated_scale_count(model),
    }
    if parms:
        summary["distribution_parameters"] = parms
    return summary


# --- predict.survreg -------------------------------------------------------------------------


def _newdata_strata(fit: Any, newdata: Any, n: int) -> list[int]:
    """``match(strata(newdata), levels(strata.keep))`` for a stratified fit."""

    codes, levels = _strata_factor(newdata, fit.strata_columns, n)
    position = {level: idx for idx, level in enumerate(fit.strata_levels)}
    for level in levels:
        if level not in position:
            raise ValueError(f"newdata contains unknown strata level {level!r}")
    return [position[levels[code]] for code in codes]


def _newdata_inputs(
    fit: Any, newdata: Any
) -> tuple[list[list[float]], list[int] | None, list[float] | None]:
    """``model.matrix(object, newframe)`` plus the per-row strata and the newdata offset."""

    design = _formula_design_for_fit(fit)
    if design is None:
        names = getattr(fit, "coefficient_names", None)
        if names is not None and (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
            columns = [_column(newdata, name) for name in names]
            rows = [[float(col[row]) for col in columns] for row in range(len(columns[0]))]
            return rows, None, None
        return _as_rows(newdata, "newdata"), None, None
    if not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        raise TypeError("newdata must be a data frame with the model's columns")
    n = _formula_design_row_count(newdata, design)
    rows = _design_rows_from_spec(newdata, design, n)
    strata = _newdata_strata(fit, newdata, n) if fit.strata_levels else None
    offset = _offset_vector(newdata, design.offsets, n) if design.offsets else None
    return rows, strata, offset


def _term_selection(terms: Any | None, names: Sequence[str]) -> list[int] | None:
    """``pred[, terms]``: term names or 1-based positions, as 0-based indices."""

    if terms is None:
        return None
    requested = [terms] if isinstance(terms, str | int) else list(terms)
    selected = []
    for value in requested:
        if isinstance(value, str):
            if value not in names:
                raise ValueError(f"terms contains unknown model term {value!r}")
            selected.append(names.index(value))
        else:
            position = index(value) - 1
            if position < 0 or position >= len(names):
                raise ValueError("terms indices must be between 1 and the number of model terms")
            selected.append(position)
    return selected


def _drop(values: list[list[float]], keep_matrix: bool) -> Any:
    """R's ``drop``: one column (or one quantile row) becomes a vector."""

    if keep_matrix:
        return values
    if values and len(values[0]) == 1:
        return [row[0] for row in values]
    if len(values) == 1:
        return values[0]
    return values


def predict_survreg(
    fit: Any,
    newdata: Any | None = None,
    type: str = "response",
    se_fit: bool = False,
    terms: Any | None = None,
    p: Any = (0.1, 0.9),
) -> Any:
    """R's ``predict.survreg``: response, lp, terms, quantile and uquantile predictions.

    Vectors for lp/response (and single-``p`` quantiles), row-per-observation matrices for
    terms and several ``p``; with ``se_fit`` a ``PredictResult(fit, se_fit)``.
    """

    model = _unwrap_formula_fit(fit)
    predict_type = _match_arg(
        type, "type", ("response", "link", "lp", "linear", "terms", "quantile", "uquantile")
    )
    include_se = _normalize_bool_option(se_fit, "se.fit")
    rows = strata = offset = None
    if newdata is not None:
        rows, strata, offset = _newdata_inputs(fit, newdata)
    assign = list(fit.assign) if isinstance(fit, SurvregModelResult) else None
    term_names = [
        label
        for position, label in enumerate(getattr(fit, "term_labels", ()), start=1)
        if position != getattr(fit, "strata_term", 0)
    ]
    result = model.predict(
        newdata=rows,
        predict_type=predict_type,
        se_fit=include_se,
        p=_quantile_vector(p, "p"),
        offset=offset,
        strata=strata,
        assign=assign,
        terms=_term_selection(terms, term_names),
    )
    keep_matrix = predict_type == "terms"
    if not include_se:
        return _drop(result.fit, keep_matrix)
    return PredictResult(_drop(result.fit, keep_matrix), _drop(result.se_fit, keep_matrix))


# --- residuals.survreg -----------------------------------------------------------------------

_RESIDUAL_TYPES = (
    "response",
    "deviance",
    "dfbeta",
    "dfbetas",
    "working",
    "ldcase",
    "ldresp",
    "ldshape",
    "matrix",
)


def _collapse_codes(collapse: Any, n: int) -> list[int] | None:
    """``rowsum(rr, collapse)`` groups, in R's sorted-unique order."""

    if collapse is None or collapse is False:
        return None
    values = _materialize_labels(collapse, "collapse")
    if len(values) != n:
        raise ValueError("Wrong length for 'collapse'")
    try:
        levels = sorted(set(values))
    except TypeError:
        levels = list(dict.fromkeys(values))
    position = {level: idx for idx, level in enumerate(levels)}
    return [position[value] for value in values]


def residuals_survreg(
    fit: Any,
    type: str = "response",
    rsigma: bool = True,
    collapse: Any = False,
    weighted: bool = False,
) -> Any:
    """R's ``residuals.survreg``: the nine residual types, optionally weighted and collapsed."""

    model = _unwrap_formula_fit(fit)
    residual_type = _match_arg(type, "type", _RESIDUAL_TYPES)
    result = model.residuals(
        residual_type,
        rsigma=_normalize_bool_option(rsigma, "rsigma"),
        collapse=_collapse_codes(collapse, int(model.n)),
        weighted=_normalize_bool_option(weighted, "weighted"),
    )
    return _drop(result.values, residual_type in {"dfbeta", "dfbetas", "matrix"})


# --- anova.survreg ---------------------------------------------------------------------------


def _refit_terms(fit: SurvregModelResult, keep: int) -> Any:
    """``update(fit, ~ . - <dropped terms>)``: refit with the first ``keep`` terms."""

    model = fit.fit
    columns = [column for column, term in enumerate(fit.assign) if term <= keep]
    fixed_scale = model.scale[0] if _estimated_scale_count(model) == 0 else 0.0
    return _core.survreg_fit(
        _core.SurvregData(
            model.time,
            model.status,
            [[row[column] for column in columns] for row in model.covariates],
            time2=model.time2,
            weights=model.weights,
            offset=model.offset,
            strata=model.strata if 0 < fit.strata_term <= keep else None,
            cluster=model.cluster,
        ),
        model.distribution,
        scale=fixed_scale,
        control=fit.control,
    )


def _chisq_p_values(deviance: list[float], df: list[float]) -> list[float]:
    """``stat.anova(test="Chisq")``: ``pchisq(dev, |df|, lower=FALSE)``, NA otherwise."""

    p_values = []
    for value, degrees in zip(deviance, df, strict=True):
        if math.isnan(value) or math.isnan(degrees) or degrees == 0:
            p_values.append(math.nan)
            continue
        statistic = value * math.copysign(1.0, degrees)
        if statistic < 0.0:
            p_values.append(math.nan)
        else:
            p_values.append(float(_core.lrt_test(statistic / 2.0, 0.0, int(abs(degrees))).p_value))
    return p_values


def _anova_single(fit: SurvregModelResult, with_test: bool) -> SurvregAnovaResult:
    model = fit.fit
    labels = list(fit.term_labels)
    loglik = [0.0] * (len(labels) + 1)
    resid_df = [0] * (len(labels) + 1)
    loglik[-1] = -2.0 * float(model.log_likelihood)
    resid_df[-1] = int(model.df_residual)
    for keep in range(len(labels) - 1, -1, -1):
        refit = _refit_terms(fit, keep)
        loglik[keep] = -2.0 * float(refit.log_likelihood)
        resid_df[keep] = int(refit.df_residual)
    df = [math.nan] + [float(resid_df[k - 1] - resid_df[k]) for k in range(1, len(loglik))]
    deviance = [math.nan] + [loglik[k - 1] - loglik[k] for k in range(1, len(loglik))]
    heading = [
        "Analysis of Deviance Table",
        f"Response: {fit.formula.partition('~')[0].strip() if fit.formula else 'y'}",
        f"Scale fixed at {model.scale[0]:g}"
        if _estimated_scale_count(model) == 0
        else "Scale estimated",
        "Terms added sequentially (first to last)",
    ]
    return SurvregAnovaResult(
        terms=["NULL", *labels],
        df=df,
        deviance=deviance,
        resid_df=resid_df,
        loglik=loglik,
        p=_chisq_p_values(deviance, df) if with_test else None,
        heading=heading,
    )


def _diff_term(previous: Sequence[str], current: Sequence[str], position: int) -> str:
    """``anova.survreglist``'s ``diff.term``: how model ``position`` differs from the last."""

    in_current = all(label in current for label in previous)
    in_previous = all(label in previous for label in current)
    if in_current and in_previous:
        return "="
    if in_current:
        return "+".join(["", *[label for label in current if label not in previous]])
    if in_previous:
        return "-".join(["", *[label for label in previous if label not in current]])
    return f"{position - 1} vs. {position}"


def _anova_list(fits: Sequence[SurvregModelResult], with_test: bool) -> SurvregAnovaResult:
    responses = [fit.formula.partition("~")[0].strip() if fit.formula else "" for fit in fits]
    keep = [response == responses[0] for response in responses]
    if not all(keep):
        warnings.warn(
            "Some fit objects deleted because response differs from the first model",
            RuntimeWarning,
            stacklevel=3,
        )
    fits = [fit for fit, same in zip(fits, keep, strict=True) if same]
    if len(fits) == 1:
        raise ValueError("The first model has a different response from the rest")
    resid_df = [int(fit.fit.df_residual) for fit in fits]
    loglik = [-2.0 * float(fit.fit.log_likelihood) for fit in fits]
    labels = [list(fit.term_labels) for fit in fits]
    tests = [""] + [_diff_term(labels[i - 1], labels[i], i + 1) for i in range(1, len(fits))]
    df = [math.nan] + [float(resid_df[i - 1] - resid_df[i]) for i in range(1, len(fits))]
    deviance = [math.nan] + [loglik[i - 1] - loglik[i] for i in range(1, len(fits))]
    return SurvregAnovaResult(
        terms=[fit.formula.partition("~")[2].strip() if fit.formula else "" for fit in fits],
        df=df,
        deviance=deviance,
        resid_df=resid_df,
        loglik=loglik,
        p=_chisq_p_values(deviance, df) if with_test else None,
        heading=["Analysis of Deviance Table", f"Response: {responses[0]}"],
        test_labels=tests,
    )


def anova_survreg(*fits: Any, test: str = "Chisq") -> SurvregAnovaResult:
    """R's ``anova.survreg`` (one fit: terms added sequentially) and ``anova.survreglist``."""

    if len(fits) == 1 and isinstance(fits[0], list | tuple):
        fits = tuple(fits[0])
    if not fits:
        raise TypeError("anova requires at least one fitted model")
    for fit in fits:
        if not isinstance(fit, SurvregModelResult):
            raise TypeError("anova.survreg requires survreg model fits")
    with_test = _match_arg(test, "test", ("Chisq", "none")) == "Chisq"
    if len(fits) == 1:
        return _anova_single(fits[0], with_test)
    return _anova_list(fits, with_test)


# --- dsurvreg / psurvreg / qsurvreg / rsurvreg -----------------------------------------------


def dsurvreg(
    x: Any, mean: Any, scale: Any = 1, distribution: str = "weibull", parms: Any | None = None
) -> list[float]:
    """Density of the ``survreg`` location-scale distributions (R's ``dsurvreg``)."""

    return _core.dsurvreg(
        _quantile_vector(x, "x"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
        distribution,
        _parms_vector(parms),
    )


def psurvreg(
    q: Any, mean: Any, scale: Any = 1, distribution: str = "weibull", parms: Any | None = None
) -> list[float]:
    """Distribution function of the ``survreg`` distributions (R's ``psurvreg``)."""

    return _core.psurvreg(
        _quantile_vector(q, "q"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
        distribution,
        _parms_vector(parms),
    )


def qsurvreg(
    p: Any, mean: Any, scale: Any = 1, distribution: str = "weibull", parms: Any | None = None
) -> list[float]:
    """Quantiles of the ``survreg`` distributions (R's ``qsurvreg``)."""

    return _core.qsurvreg(
        _quantile_vector(p, "p"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
        distribution,
        _parms_vector(parms),
    )


def rsurvreg(
    n: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
    seed: int | None = None,
) -> list[float]:
    """Random draws from the ``survreg`` distributions (R's ``rsurvreg``; ``seed`` is ours)."""

    count = _integer_scalar(n, "n")
    if count < 0:
        raise ValueError("n must be non-negative")
    return _core.rsurvreg(
        count,
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
        distribution,
        _parms_vector(parms),
        None if seed is None else _integer_scalar(seed, "seed"),
    )


# ---------------------------------------------------------------------------
# survreg methods of the shared generics in ``_models``
#
# ``_models`` keeps the coxph branch inline and routes every other fit here by
# looking up ``<generic>_survreg``; these are R's ``*.survreg`` methods.
# ---------------------------------------------------------------------------


def _normal_two_sided_p_value(statistic: float) -> float:
    """``2 * pnorm(-abs(z))``."""

    if math.isnan(statistic):
        return math.nan
    if math.isinf(statistic):
        return 0.0
    return 2.0 * NormalDist().cdf(-abs(statistic))


def coef_survreg(fit: Any) -> list[float]:
    """``coef.survreg``: the location coefficients only."""

    return _location_beta(fit)


def coef_names_survreg(fit: Any, *, complete: Any | None = None) -> list[str]:
    """``names(coef(fit))``; ``complete`` appends the ``Log(scale)`` rows."""

    names = _fit_location_coef_names(fit, len(_location_beta(fit)))
    if complete is None:
        return names
    if _normalize_bool_option(complete, "complete"):
        names.extend(survreg_scale_names(fit))
    return names


def vcov_survreg(fit: Any, *, complete: Any = True) -> list[list[float]]:
    """``vcov.survreg``: ``fit$var``, or its location block."""

    return survreg_vcov(fit, _normalize_bool_option_with_default(complete, "complete", True))


def confint_survreg(
    fit: Any, parm: Any | None = None, *, level: Any = 0.95
) -> list[dict[str, float | str]]:
    """``confint.survreg``: normal-approximation intervals for the location coefficients."""

    z = NormalDist().inv_cdf(1.0 - (1.0 - _normalize_conf_level(level, "level")) / 2.0)
    names = coef_names_survreg(fit)
    coefficients = coef_survreg(fit)
    variance = survreg_vcov(fit, False)
    from ._models import _coefficient_selection

    return [
        {
            "name": names[idx],
            "lower": coefficients[idx] - z * math.sqrt(max(float(variance[idx][idx]), 0.0)),
            "upper": coefficients[idx] + z * math.sqrt(max(float(variance[idx][idx]), 0.0)),
        }
        for idx in _coefficient_selection(parm, names)
    ]


def degrees_freedom_survreg(fit: Any) -> int:
    """``sum(fit$df)``: the coefficients plus the estimated scales."""

    return int(_unwrap_formula_fit(fit).df)


def df_residual_survreg(fit: Any) -> int:
    """``fit$df.residual``."""

    return int(_unwrap_formula_fit(fit).df_residual)


def loglik_survreg(fit: Any) -> float:
    """``fit$loglik[2]``, on the original response scale."""

    return float(_unwrap_formula_fit(fit).log_likelihood)


def nobs_survreg(fit: Any) -> int:
    """``nobs``: the number of observations in the fitted model frame."""

    if isinstance(fit, _FormulaFit) and fit.n_observations is not None:
        return fit.n_observations
    model = _unwrap_formula_fit(fit)
    values = getattr(model, "status", None)
    if values is None:
        values = getattr(model, "event_times", None)
    if values is None:
        raise TypeError("model does not expose stored observations")
    return len(list(values))


def model_formula_survreg(fit: Any) -> str:
    """``formula.survreg``."""

    if isinstance(fit, _FormulaFit) and fit.formula is not None:
        return fit.formula
    raise TypeError("model_formula requires a formula-based fitted model")


def model_weights_survreg(fit: Any) -> list[float] | None:
    """``weights.survreg``: ``None`` when every case weight is 1."""

    if isinstance(fit, _FormulaFit) and fit.case_weights is not None:
        return list(fit.case_weights)
    values = getattr(_unwrap_formula_fit(fit), "weights", None)
    if values is None:
        return None
    weights = [float(value) for value in _materialize_1d(values, "weights")]
    if all(abs(value - 1.0) <= 1e-12 for value in weights):
        return None
    return weights


def model_term_names_survreg(fit: Any, terms: Any | None = None) -> list[str]:
    """``attr(terms(fit), 'term.labels')``, optionally the subset ``terms`` selects."""

    design = _formula_design_for_fit(fit)
    if design is None:
        raise TypeError("model_term_names requires a formula-based fitted model")
    names = [_design_term_name(term) for term in design.covariates]
    return [names[idx] for idx in _term_selection(terms, names)]


def model_matrix_survreg(fit: Any) -> dict[str, Any]:
    """``model.matrix.survreg``: the design matrix, its column names and ``assign``."""

    model = _unwrap_formula_fit(fit)
    rows = getattr(model, "covariates", None)
    if rows is None:
        rows = getattr(model, "x", None)
    if rows is None:
        raise TypeError("model_matrix requires a fitted model with stored covariates")
    matrix = [[float(value) for value in row] for row in rows]
    width = len(matrix[0]) if matrix else 0
    if any(len(row) != width for row in matrix):
        raise ValueError("stored model matrix must be rectangular")

    design = _formula_design_for_fit(fit)
    columns = _fallback_coef_names(width)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            columns = names
    elif len(coef_names_survreg(fit)) == width:
        columns = coef_names_survreg(fit)

    assign = list(range(1, width + 1))
    if design is not None and len(design.term_assignments) == len(design.covariates):
        built = [0] if design.intercept else []
        for term, assignment in zip(design.covariates, design.term_assignments, strict=True):
            built.extend([assignment] * len(_design_term_output_names(term)))
        if len(built) == width:
            assign = built
    return {"data": matrix, "columns": columns, "assign": assign}


def model_frame_survreg(fit: Any) -> dict[str, list[Any]]:
    """``model.frame.survreg`` for a fit made with ``model=TRUE``."""

    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("model_frame requires a stored model frame")
    if not isinstance(frame, Mapping):
        raise TypeError("stored model frame must be mapping-like")

    columns: dict[str, list[Any]] = {}
    for name, values in frame.items():
        if isinstance(values, Surv):
            existing = set(columns)
            if values.start is not None:
                if "start" not in existing:
                    columns["start"] = list(values.start)
                if "stop" not in existing:
                    columns["stop"] = list(values.time)
            elif "time" not in existing:
                columns["time"] = list(values.time)
            if values.time2 is not None and "time2" not in existing:
                columns["time2"] = list(values.time2)
            if "status" not in existing:
                columns["status"] = list(values.event)
            continue
        if isinstance(values, Mapping):
            continue
        text_name = str(name)
        if text_name in {"group", "(id)", "(cluster)", "(strata)"}:
            columns[text_name] = _materialize_labels(values, text_name)
            continue
        materialized = _materialize_1d(values, text_name)
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns[text_name] = list(materialized)
    return columns


def model_summary_survreg(fit: Any) -> dict[str, Any]:
    """``summary.survreg``: the coefficient table plus the fit's scalar pieces."""

    model = _unwrap_formula_fit(fit)
    coefficients = [float(value) for value in model.coefficients]
    names = survreg_summary_names(fit)
    variance = survreg_vcov(fit, True)
    naive_variance = model.naive_variance_matrix
    robust = naive_variance is not None
    if naive_variance is None:
        naive_variance = variance

    rows: list[dict[str, float | str]] = []
    for idx, value in enumerate(coefficients):
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        naive_standard_error = math.sqrt(max(float(naive_variance[idx][idx]), 0.0))
        if math.isnan(value):
            statistic = math.nan
        elif standard_error > 0.0:
            statistic = value / standard_error
        elif value == 0.0:
            statistic = math.nan
        else:
            statistic = math.copysign(math.inf, value)
        row: dict[str, float | str] = {
            "name": names[idx],
            "coef": value,
            "value": value,
            "se": standard_error,
            "naive_se": naive_standard_error,
            "statistic": statistic,
            "z": statistic,
            "p": _normal_two_sided_p_value(statistic),
        }
        if robust:
            row["robust_se"] = standard_error
        rows.append(row)

    result: dict[str, Any] = {
        "model_type": "survreg",
        "coefficients": rows,
        "coefficient_names": names,
        "loglik": loglik_survreg(fit),
        "df": degrees_freedom_survreg(fit),
        "n": nobs_survreg(fit),
        "robust": robust,
    }
    result.update(survreg_summary(fit))
    return result
