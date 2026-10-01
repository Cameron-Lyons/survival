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
from dataclasses import dataclass, field
from itertools import chain
from operator import index
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
    _as_matrix_rows,
    _as_rows,
    _coefficient_selection,
    _control_mapping,
    _encode_labels,
    _finite_float,
    _float_or_nan,
    _float_vector,
    _integer_scalar,
    _materialize_labels,
    _matrix_input_column_names,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_na_action,
    _normalize_optional_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
    _quantile_vector,
)
from ._fit import (
    _excluded_rows,
    _model_matrix_names_and_assign,
    _NewData,
    _newdata_frame,
    _pad_rows,
    _r_factor_design,
    _rowsum_excluded,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _column_source,
    _design_rows_from_spec,
    _design_term_name,
    _design_term_output_names,
    _fit_formula_design,
    _formula_cluster_values,
    _formula_model_frame,
    _formula_model_term_degree,
    _formula_response_spec,
    _na_action_record,
    _offset_vector,
    _parse_formula,
    _strata_keep,
    _strata_specs,
    _subset_formula_inputs,
)
from ._surv import Surv, _complete_codes, _survreg_response_arrays
from ._survpenal import fit_penalized, penalty_terms
from ._types import (
    NaAction,
    PredictResult,
    _FormulaDesign,
    _ModelCovariateTerm,
    _ModelStrataTerm,
    _StrataSpec,
)

SurvregDistribution = _core.SurvregDistribution
SurvregControl = _core.SurvregControl

# R's ``survreg.distributions``: the built-in location-scale families keyed by the names
# ``survreg(dist=)`` matches against. Users may add objects or distribution dictionaries.
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


@dataclass(frozen=True)
class SurvregModelResult:
    """R's ``survreg`` object: the Rust ``SurvregFit`` plus the metadata R keeps on it.

    Most properties are R's components: ``coefficients`` are the location coefficients
    (``NaN`` where singular; the full vector with the ``Log(scale)`` entries is
    ``fit.coefficients``), ``loglik`` is (intercept-only, full), ``var`` the variance of
    every coefficient and ``naive_var`` (R's ``naive.var``) the model-based one of a
    robust fit.  ``weights``, ``x``, ``y``, ``model`` and ``score`` are ``None`` unless
    the call kept them; ``na_action`` (``fit$na.action``) records the rows the
    ``na.action`` removed.  ``n``, ``converged``, ``robust`` and ``distribution`` (the
    resolved distribution object) are conveniences with no ``survreg`` component in R.

    A model with ``ridge()`` or ``pspline()`` terms is R's ``survreg.penal`` object:
    ``penalized`` holds the ``SurvpenalFit`` (``fit`` is its ``survreg``), and ``df``,
    ``iter``, ``var2``, ``penalty``, ``pterms``, ``assign2`` and ``history`` are its
    components.
    """

    fit: _core.SurvregFit = field(repr=False)
    formula: str | None
    coefficient_names: tuple[str, ...]
    dist: Any
    control: _core.SurvregControl = field(repr=False)
    design: _FormulaDesign | None = field(default=None, repr=False)
    weights: list[float] | None = field(default=None, repr=False)
    cluster: list[Any] | None = field(default=None, repr=False)
    x: list[list[float]] | None = field(default=None, repr=False)
    y: Surv | None = field(default=None, repr=False)
    model: dict[str, Any] | None = field(default=None, repr=False)
    score: list[float] | None = field(default=None, repr=False)
    assign: tuple[int, ...] = field(default=(), repr=False)
    term_labels: tuple[str, ...] = ()
    strata_term: int = field(default=0, repr=False)
    strata_terms: tuple[_StrataSpec, ...] = field(default=(), repr=False)
    strata_levels: tuple[str, ...] = ()
    na_action: NaAction | None = field(default=None, repr=False)
    penalized: Any | None = field(default=None, repr=False)
    assign2_labels: tuple[str, ...] = field(default=(), repr=False)

    @property
    def is_penalized(self) -> bool:
        """R's ``inherits(fit, "survreg.penal")``."""
        return self.penalized is not None

    @property
    def coefficients(self) -> list[float]:
        return [float(value) for value in self.fit.coefficients[: len(self.fit.means)]]

    @property
    def icoef(self) -> list[float]:
        return self.fit.icoef

    @property
    def var(self) -> list[list[float]]:
        return self.fit.variance_matrix

    @property
    def naive_var(self) -> list[list[float]] | None:
        return self.fit.naive_variance_matrix

    @property
    def robust(self) -> bool:
        return self.fit.naive_variance_matrix is not None

    @property
    def loglik(self) -> list[float]:
        return [float(self.fit.intercept_only_log_likelihood), float(self.fit.log_likelihood)]

    @property
    def iter(self) -> int | list[int]:
        """``fit$iter``: the iterations, or (outer, total inner) for a penalized fit."""
        if self.penalized is not None:
            return list(self.penalized.iter)
        return int(self.fit.iterations)

    @property
    def converged(self) -> bool:
        return bool(self.fit.converged)

    @property
    def linear_predictors(self) -> list[float]:
        return self.fit.linear_predictors

    @property
    def scale(self) -> list[float]:
        return self.fit.scale

    @property
    def idf(self) -> int:
        return 1 + _estimated_scale_count(self.fit)

    @property
    def df(self) -> int | list[float]:
        """``fit$df``: the number of coefficients, the estimated scales included, or the
        degrees of freedom of every term of ``assign2`` for a penalized fit."""
        if self.penalized is not None:
            return list(self.penalized.df)
        return int(self.fit.df)

    @property
    def df_residual(self) -> int | float:
        """``fit$df.residual``: ``n - sum(df)``, fractional for a penalized fit (``NaN``
        where R has ``NA``)."""
        if self.penalized is None:
            return int(self.fit.df_residual)
        return float(self.fit.df_residual)

    @property
    def var2(self) -> list[list[float]] | None:
        """``fit$var2`` of a penalized fit: the sandwich ``H^-1 I H^-1``."""
        return None if self.penalized is None else self.penalized.var2

    @property
    def penalty(self) -> list[float] | None:
        """``fit$penalty`` of a penalized fit: ``c(0, P)``."""
        return None if self.penalized is None else list(self.penalized.penalty)

    @property
    def inner_failures(self) -> list[int]:
        """The outer iterations of a penalized fit whose inner loop did not converge."""
        return [] if self.penalized is None else list(self.penalized.inner_failures)

    @property
    def pterms(self) -> dict[str, int] | None:
        """``fit$pterms``: 0 for an ordinary term, 1 for a penalized one."""
        if self.penalized is None:
            return None
        return dict(zip(self.assign2_labels, self.penalized.pterms, strict=False))

    @property
    def assign2(self) -> dict[str, list[int]] | None:
        """``fit$assign2``: the 0-based coefficients of every term, ``sigma`` last."""
        if self.penalized is None:
            return None
        return dict(zip(self.assign2_labels, self.penalized.assign2, strict=True))

    @property
    def history(self) -> dict[str, Any] | None:
        """``fit$history``: the ``PenaltyHistory`` of every penalized term."""
        if self.penalized is None:
            return None
        return {self.assign2_labels[entry.term]: entry for entry in self.penalized.history}

    @property
    def frail(self) -> None:
        """``fit$frail``: survreg() refuses sparse frailty terms, so always ``None``."""
        return None

    @property
    def fvar(self) -> None:
        return None

    @property
    def means(self) -> list[float]:
        return self.fit.means

    @property
    def n(self) -> int:
        return int(self.fit.n)

    @property
    def parms(self) -> list[float] | None:
        return list(self.fit.distribution.parms) or None

    @property
    def distribution(self) -> _core.SurvregDistribution:
        """The ``SurvregDistribution`` ``dist`` resolved to (``survreg.distributions[[dist]]``)."""
        return self.fit.distribution


@dataclass(frozen=True)
class SurvregAnovaResult:
    """R's ``anova.survreg``/``anova.survreglist`` table (an ``anova`` data frame)."""

    terms: list[str]
    df: list[float]
    deviance: list[float]
    resid_df: list[float]
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


def _distribution_issues(dlist: Mapping[str, Any]) -> list[str]:
    """Structural checks before constructing a native distribution."""
    issues = []
    name = dlist.get("name")
    if not isinstance(name, str):
        issues.append("Missing a name")
    if dlist.get("dist") is None:
        for key in ("init", "deviance", "density", "quantile"):
            if not callable(dlist.get(key)):
                issues.append(f"Missing or invalid {key} function")
    trans = dlist.get("trans", "identity")
    if callable(trans):
        for key in ("dtrans", "itrans"):
            if not callable(dlist.get(key)):
                issues.append(f"Missing or invalid {key} component")
    elif not isinstance(trans, str) or trans not in _TRANSFORMS:
        issues.append("trans must be 'log' or 'identity', or a callable with dtrans and itrans")
    return issues


def _distribution_from_list(dlist: Mapping[str, Any]) -> Any:
    """A callback-defined family or a response transform of another distribution."""

    issues = _distribution_issues(dlist)
    if issues:
        raise ValueError("Invalid distribution object: " + "; ".join(issues))
    name = dlist["name"]
    base = dlist.get("dist")
    trans = dlist.get("trans", "identity")
    scale = dlist.get("scale")
    scale = None if scale is None else _finite_float(scale, "scale")
    transform = _TRANSFORMS["identity"] if callable(trans) else _TRANSFORMS[trans]
    parms = dlist.get("parms")
    if base is None:
        reference = _core.SurvregDistribution.from_callbacks(
            name,
            dlist["init"],
            dlist["density"],
            dlist["deviance"],
            dlist["quantile"],
            variance=dlist.get("variance"),
            transform=transform,
            scale=scale,
            parms=_parms_vector(parms),
            parm_names=list(parms) if isinstance(parms, Mapping) else None,
        )
    else:
        reference = _resolve_distribution(base, parms, probe=False).derived(name, transform, scale)
    if callable(trans):
        reference = reference.with_transform(trans, dlist["dtrans"], dlist["itrans"])
    return reference


def _resolve_distribution(dist: Any, parms: Any | None, *, probe: bool = True) -> Any:
    """``survreg``'s distribution lookup: name (``match.arg``), list, or object; then parms."""

    if isinstance(dist, str):
        key = _match_arg(dist, "dist", tuple(survreg_distributions))
        dist = survreg_distributions[key]
    if isinstance(dist, Mapping):
        dist = _distribution_from_list(dist)
    elif not isinstance(dist, _core.SurvregDistribution):
        raise TypeError("Invalid distribution object")
    errors = dist.dtest() if probe else []
    if errors:
        raise ValueError("Invalid distribution object: " + "; ".join(errors))
    if parms is None:
        return dist
    if not dist.parms:
        raise ValueError(f"{dist.name} distribution has no optional parameters")
    if isinstance(parms, Mapping):
        names = dist.parm_names
        if not names or any(name not in names for name in parms):
            raise ValueError("Invalid parameter names")
        values = dict(zip(names, dist.parms, strict=True))
        values.update(parms)
        parms = [values[name] for name in names]
    return dist.with_parms(_parms_vector(parms))


def survregDtest(dlist: Any, verbose: bool = False) -> bool | list[str]:
    """Check a distribution object; ``True``, or the problems when ``verbose``."""

    try:
        if isinstance(dlist, Mapping) and (issues := _distribution_issues(dlist)):
            return issues if verbose else False
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
    ``outer.max`` bounds the outer iterations of a penalized fit and must be at least 1
    (R does not check it), ``debug`` is accepted and unused as in R."""

    for alias in ("rel.tolerance", "eps"):
        rel_tolerance = _pop_dotted_keyword(dotted, alias, "rel_tolerance", rel_tolerance, 1e-9)
    for alias in ("toler.chol", "tol_chol"):
        toler_chol = _pop_dotted_keyword(dotted, alias, "toler_chol", toler_chol, 1e-10)
    for alias in ("iter.max", "max_iter"):
        iter_max = _pop_dotted_keyword(dotted, alias, "iter_max", iter_max, None)
    outer_max = _pop_dotted_keyword(dotted, "outer.max", "outer_max", outer_max, 10)
    if dotted:
        raise TypeError(f"unused argument(s): {', '.join(sorted(dotted))}")
    iterations = _integer_scalar(maxiter if iter_max is None else iter_max, "iter.max")
    if iterations < 0:
        raise ValueError("iter.max must be non-negative")
    outer = _integer_scalar(outer_max, "outer.max")
    if outer < 1:
        raise ValueError("invalid value for outer.max")
    return _core.SurvregControl(
        iter_max=iterations,
        rel_tolerance=_finite_float(rel_tolerance, "rel.tolerance"),
        toler_chol=_finite_float(toler_chol, "toler.chol"),
        outer_max=outer,
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
    strata_terms: tuple[_StrataSpec, ...] = ()
    strata: list[int] | None = None
    strata_levels: tuple[str, ...] = ()
    weights: list[float] | None = None
    offset: list[float] | None = None
    cluster: list[Any] | None = None
    model: dict[str, Any] | None = None
    na_action: NaAction | None = None


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
            labels[term_index - 1] = model_term.spec.call
            strata_term = term_index
    assign = [0] * int(design.intercept)
    for term, term_index in zip(design.covariates, assignments, strict=True):
        assign.extend([term_index] * len(_design_term_output_names(term)))
    return tuple(assign), tuple(labels), strata_term


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
    full_data = data
    weights = _column_or_values(data, weights, "weights")
    offset = _column_or_values(data, offset, "offset")
    cluster = _column_or_values(data, cluster, "cluster")
    aligned = {"weights": weights, "offset": offset, "cluster": cluster}
    if subset is not None:
        data, aligned = _subset_formula_inputs(formula, data, subset, **aligned)
    data, aligned, removed = _apply_formula_na_action(formula, data, na_action, **aligned)
    weights, offset, cluster = aligned["weights"], aligned["offset"], aligned["cluster"]
    response, terms = _parse_formula(formula, data)
    n = len(response)
    design = _r_factor_design(
        data,
        _fit_formula_design(data, spec, terms, n, include_intercept=True, full_data=full_data),
        drop_unused_strata=False,
    )
    if terms.offsets:
        if offset is not None:
            raise ValueError("use only one of formula offset(...) or offset")
        offset = _offset_vector(data, terms.offsets, n)
    if terms.clusters:
        if cluster is not None:
            warnings.warn(
                "cluster appears both in a formula and as an argument, formula term ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            cluster = _formula_cluster_values(data, terms, n)
    strata_terms = _strata_specs(terms)
    strata: list[int] | None = None
    strata_levels: tuple[str, ...] = ()
    if strata_terms:
        factor = _strata_keep(data, strata_terms, drop_unused=False)
        strata = _complete_codes(factor, "strata contains missing values")
        strata_levels = tuple(factor.levels)
    model_terms = [
        term
        for term in terms.model_terms
        if isinstance(term, _ModelCovariateTerm | _ModelStrataTerm)
    ]
    assign, term_labels, strata_term = _term_structure(design, model_terms)
    return _SurvregFrame(
        response=response,
        x=_design_rows_from_spec(data, design, n),
        names=tuple(_design_output_names(design)),
        design=design,
        assign=assign,
        term_labels=term_labels,
        strata_term=strata_term,
        strata_terms=strata_terms,
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
        na_action=_na_action_record(na_action, removed),
    )


def _design_output_names(design: _FormulaDesign) -> list[str]:
    """``colnames(model.matrix)``: ``(Intercept)`` then each term's columns."""

    names = [name for term in design.covariates for name in _design_term_output_names(term)]
    if design.intercept:
        names.insert(0, "(Intercept)")
    return names


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
    na_action: str | None = _DEFAULT_NA_ACTION,
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

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
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
        if subset is not None or _normalize_na_action(na_action) not in {"fail", "pass", "omit"}:
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
    if not all(map(math.isfinite, chain.from_iterable(frame.x))):
        raise ValueError("data contains an infinite predictor")

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

    penalized_terms = penalty_terms(frame.design)
    if any(term.kind == "frailty" for _, term in penalized_terms):
        raise ValueError("survreg does not support frailty terms")

    time, status, time2 = _survreg_response_arrays(response)
    cluster_codes = (  # as.numeric(as.factor(cluster)); unused when robust = FALSE
        _encode_labels(frame.cluster, "cluster")
        if frame.cluster is not None and robust_value is not False
        else None
    )
    data_value = _core.SurvregData(
        time,
        [int(code) for code in status],
        frame.x,
        time2=time2,
        weights=frame.weights,
        offset=frame.offset,
        strata=frame.strata,
        cluster=cluster_codes,
    )
    init_value = _float_vector(init, "init") if init is not None else None
    penalized = None
    assign2_labels: tuple[str, ...] = ()
    if penalized_terms:
        # survpenal.fit warns about nothing: its inner-loop warning is never reached
        penalized, assign2_labels = fit_penalized(
            frame.assign,
            frame.term_labels,
            frame.strata_term,
            penalized_terms,
            data_value,
            distribution,
            init=init_value,
            scale=scale_value,
            control=control,
            robust=robust_value,
            nstrat=len(frame.strata_levels) or None,
        )
        fit = penalized.survreg
    else:
        fit = _core.survreg_fit(
            data_value,
            distribution,
            init=init_value,
            scale=scale_value,
            control=control,
            robust=robust_value,
            nstrat=len(frame.strata_levels) or None,
        )
        if control.iter_max > 1 and not fit.converged:
            warnings.warn(
                "Ran out of iterations and did not converge", RuntimeWarning, stacklevel=2
            )

    return SurvregModelResult(
        fit=fit,
        formula=formula if isinstance(formula, str) else None,
        coefficient_names=frame.names,
        dist=dist if isinstance(dist, str) else distribution,
        control=control,
        design=frame.design,
        weights=frame.weights,
        cluster=frame.cluster if fit.naive_variance_matrix is not None else None,
        x=frame.x if keep_x else None,
        y=response if keep_y else None,
        model=frame.model,
        # fit$u of a penalized fit, frailties included
        score=list(fit.score if penalized is None else penalized.score) if keep_score else None,
        assign=frame.assign,
        term_labels=frame.term_labels,
        strata_term=frame.strata_term,
        strata_terms=frame.strata_terms,
        strata_levels=frame.strata_levels,
        na_action=frame.na_action,
        penalized=penalized,
        assign2_labels=assign2_labels,
    )


# --- accessors used by the generics ----------------------------------------------------------


def _estimated_scale_count(model: Any) -> int:
    return len(model.coefficients) - len(model.means)


def survreg_df(fit: SurvregModelResult) -> int | float:
    """``sum(fit$df)``: the number of coefficients, the estimated scales included, or the
    fractional total of a penalized fit."""

    if fit.penalized is None:
        return int(fit.fit.df)
    return float(fit.fit.df)


def _location_names(fit: SurvregModelResult, complete: bool = True) -> list[str]:
    """``names(coef(fit, complete))``: without ``complete`` the aliased (``NA``)
    coefficients are left out."""

    names = list(fit.coefficient_names)
    if complete:
        return names
    return [
        name for name, value in zip(names, fit.coefficients, strict=True) if not math.isnan(value)
    ]


def _scale_labels(fit: SurvregModelResult, stratum_template: str) -> list[str]:
    """The labels of the estimated scales: ``Log(scale)`` for a single scale; for a scale
    per stratum, ``stratum_template`` filled with each stratum (``names(fit$scale)``)."""

    scales = _estimated_scale_count(fit.fit)
    if scales > 1:
        return [stratum_template.format(level) for level in fit.strata_levels]
    return ["Log(scale)"] * scales


def survreg_summary_names(fit: SurvregModelResult) -> list[str]:
    """Row names of ``summary.survreg``'s table: a stratum's scale row is named by the
    stratum."""

    return _location_names(fit) + _scale_labels(fit, "{}")


def survreg_vcov_names(fit: SurvregModelResult, complete: bool = True) -> list[str]:
    """``dimnames(vcov(fit, complete))``: the location names (the aliased ones left out
    without ``complete``), then ``Log(scale)``, or ``Log(scale[<stratum>])`` per stratum."""

    return _location_names(fit, complete) + _scale_labels(fit, "Log(scale[{}])")


def survreg_vcov(fit: SurvregModelResult, complete: bool = True) -> list[list[float]]:
    """R's ``vcov.survreg``: ``fit$var``; without ``complete`` the rows and columns of the
    aliased location coefficients are dropped and the ``Log(scale)`` rows kept."""

    model = fit.fit
    variance = [[float(value) for value in row] for row in model.variance_matrix]
    if complete:
        return variance
    nvar = len(model.means)
    keep = [
        idx for idx, value in enumerate(model.coefficients) if idx >= nvar or not math.isnan(value)
    ]
    return [[variance[row][column] for column in keep] for row in keep]


def survreg_summary(fit: SurvregModelResult) -> dict[str, Any]:
    """The pieces of ``summary.survreg`` that are not the coefficient table."""

    model = fit.fit
    distribution = model.distribution
    parms = list(distribution.parms)
    loglik = fit.loglik
    summary: dict[str, Any] = {
        "location_coefficients": fit.coefficients,
        "location_coefficient_names": _location_names(fit),
        "scale": model.scale[0] if len(model.scale) == 1 else list(model.scale),
        "scales": list(model.scale),
        "scale_names": list(fit.strata_levels) if len(model.scale) > 1 else [],
        "fixed_scale": _estimated_scale_count(model) == 0,
        "distribution": distribution.name,
        "parms": (
            f"{distribution.name} distribution: parmameters= {' '.join(map(str, parms))}"
            if parms
            else f"{distribution.name} distribution"
        ),
        "loglik": loglik,
        "chi": 2.0 * (loglik[1] - loglik[0]),
        # print.summary.survreg's df: sum(x$df) - x$idf
        "chi_df": survreg_df(fit) - fit.idf,
        "iter": fit.iter,
        "idf": fit.idf,
    }
    if parms:
        summary["distribution_parameters"] = parms
    return summary


# --- predict.survreg -------------------------------------------------------------------------


def _newdata_inputs(
    fit: SurvregModelResult,
    newdata: Any,
    na_action: str,
    *,
    allow_missing_predictors: bool = False,
    allow_missing_strata: bool = False,
) -> _NewData:
    """``model.frame(Terms, newdata, na.action)`` and ``model.matrix(object, newframe)``:
    the rows, strata and offset of the complete ``newdata`` rows."""

    design = fit.design
    if design is None:
        # a fit on a design matrix: newdata's columns named like its coefficients, or a
        # matrix
        if isinstance(newdata, Mapping) or hasattr(newdata, "columns"):
            columns = [_column(newdata, name) for name in fit.coefficient_names]
            rows = [list(map(_float_or_nan, values)) for values in zip(*columns, strict=True)]
        else:
            rows = _as_matrix_rows(
                newdata, "newdata", allow_empty_columns=False, convert=_float_or_nan
            )
        missing = [row for row, values in enumerate(rows) if any(map(math.isnan, values))]
        if missing and na_action == "fail":
            raise ValueError("missing values in newdata")
        gaps = set(missing)
        if na_action == "pass" and allow_missing_predictors:
            gaps = set()
            missing = []
        return _NewData(
            data=None,
            x=[values for row, values in enumerate(rows) if row not in gaps],
            strata=None,
            offset=None,
            y=None,
            missing=tuple(missing),
        )
    if not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        raise TypeError("newdata must be a data frame with the model's columns")
    # Terms keeps the strata() term, so its variables are required
    for term in fit.strata_terms:
        for name in term.columns:
            _column_source(newdata, name)
    return _newdata_frame(
        design,
        fit.strata_terms,
        fit.strata_levels,
        newdata,
        need_strata=bool(fit.strata_levels),
        need_response=False,
        na_action=na_action,
        allow_missing_predictors=allow_missing_predictors,
        allow_missing_strata=allow_missing_strata,
    )


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
    fit: SurvregModelResult,
    newdata: Any | None = None,
    type: str = "response",
    se_fit: bool = False,
    terms: Any | None = None,
    p: Any = (0.1, 0.9),
    na_action: str | None = "na.pass",
) -> Any:
    """R's ``predict.survreg``: response, lp, terms, quantile and uquantile predictions.

    Vectors for lp/response (and single-``p`` quantiles), row-per-observation matrices for
    terms and several ``p``; with ``se_fit`` a ``PredictResult(fit, se_fit)``.  Without
    ``newdata`` a ``na.exclude`` fit's predictions are NaN at the rows it removed
    (``naresid``); ``na_action`` applies to ``newdata``, whose incomplete rows are NaN
    (``na.pass``, ``na.exclude``), dropped (``na.omit``) or refused (``na.fail``).
    """

    predict_type = _match_arg(
        type, "type", ("response", "link", "lp", "linear", "terms", "quantile", "uquantile")
    )
    include_se = _normalize_bool_option(se_fit, "se.fit")
    action = _normalize_na_action(na_action)
    new = (
        None
        if newdata is None
        else _newdata_inputs(
            fit,
            newdata,
            action,
            allow_missing_predictors=True,
            allow_missing_strata=predict_type not in {"quantile", "uquantile"},
        )
    )
    term_names = [fit.term_labels[code - 1] for code in sorted(set(fit.assign) - {0})]
    quantiles = _quantile_vector(p, "p")
    selection = _term_selection(terms, term_names)
    predictions: list[list[float]] = []
    se_values: list[list[float]] = []
    if new is None or new.n:
        result = fit.fit.predict(
            newdata=None if new is None else new.x,
            predict_type=predict_type,
            se_fit=include_se,
            p=quantiles,
            offset=None if new is None else new.offset,
            strata=None if new is None else new.strata,
            assign=list(fit.assign),
            terms=selection,
        )
        predictions = result.fit
        if include_se:
            standard_errors = result.se_fit
            if standard_errors is None:
                raise RuntimeError("native predictor did not return requested standard errors")
            se_values = standard_errors
    # naresid restores omitted rows; na.pass predictor NaNs have already
    # propagated through just the outputs that use them.
    if new is None:
        gaps = _excluded_rows(fit.na_action)
    else:
        gaps = [] if action == "omit" else list(new.missing)
    if predict_type in {"quantile", "uquantile"}:
        width = len(quantiles)
    elif predict_type == "terms":
        width = len(term_names if selection is None else selection)
    else:
        width = 1
    keep_matrix = predict_type == "terms"
    fitted = _drop(_pad_rows(predictions, gaps, width), keep_matrix)
    if not include_se:
        return fitted
    return PredictResult(fitted, _drop(_pad_rows(se_values, gaps, width), keep_matrix))


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
    fit: SurvregModelResult,
    type: str = "response",
    rsigma: bool = True,
    collapse: Any = False,
    weighted: bool = False,
) -> Any:
    """R's ``residuals.survreg``: the nine residual types, optionally weighted and collapsed."""

    model = fit.fit
    residual_type = _match_arg(type, "type", _RESIDUAL_TYPES)
    # naresid comes before the collapse: the engine sums the fit's rows, and a group
    # holding a row na.exclude removed sums to NA
    excluded = _excluded_rows(fit.na_action)
    codes = _collapse_codes(collapse, int(model.n) + len(excluded))
    fit_codes = codes
    if codes is not None and excluded:
        gaps = set(excluded)
        fit_codes = [code for row, code in enumerate(codes) if row not in gaps]
    result = model.residuals(
        residual_type,
        rsigma=_normalize_bool_option(rsigma, "rsigma"),
        collapse=fit_codes,
        weighted=_normalize_bool_option(weighted, "weighted"),
    )
    values = result.values
    if codes is None:
        values = _pad_rows(values, excluded)
    elif excluded:
        values = _rowsum_excluded(values, codes, excluded)
    return _drop(values, residual_type in {"dfbeta", "dfbetas", "matrix"})


# --- anova.survreg ---------------------------------------------------------------------------


def _refit_terms(fit: SurvregModelResult, keep: int) -> Any:
    """``update(fit, ~ . - <dropped terms>)``: refit with the first ``keep`` terms, through
    survpenal.fit while a ``ridge()`` or ``pspline()`` term remains.  The stored penalty
    objects and basis columns are what ``update`` rebuilds from the same data."""

    model = fit.fit
    columns = [column for column, term in enumerate(fit.assign) if term <= keep]
    strata_term = fit.strata_term if fit.strata_term <= keep else 0
    fixed_scale = model.scale[0] if _estimated_scale_count(model) == 0 else 0.0
    data = _core.SurvregData(
        model.time,
        model.status,
        [[row[column] for column in columns] for row in model.covariates],
        time2=model.time2,
        weights=model.weights,
        offset=model.offset,
        strata=model.strata if strata_term else None,
        cluster=model.cluster,
    )
    penalized_terms = [
        (term_index, term) for term_index, term in penalty_terms(fit.design) if term_index <= keep
    ]
    if penalized_terms:
        refit, _ = fit_penalized(
            [fit.assign[column] for column in columns],
            fit.term_labels[:keep],
            strata_term,
            penalized_terms,
            data,
            model.distribution,
            init=None,
            scale=fixed_scale,
            control=fit.control,
            robust=None,
            nstrat=len(fit.strata_levels) if strata_term else None,
        )
        return refit.survreg
    return _core.survreg_fit(
        data,
        model.distribution,
        scale=fixed_scale,
        control=fit.control,
        nstrat=len(fit.strata_levels) if strata_term else None,
    )


def _chisq_p_values(deviance: list[float], df: list[float]) -> list[float]:
    """``stat.anova(test="Chisq")``: ``pchisq(dev * sign(df), |df|, lower=FALSE)``, NA at a
    zero df or a negative statistic (``pchisq`` carries a NaN deviance or df through)."""

    p_values = []
    for value, degrees in zip(deviance, df, strict=True):
        statistic = value * math.copysign(1.0, degrees)
        if degrees == 0 or statistic < 0.0:
            p_values.append(math.nan)
        else:
            p_values.append(_core.pchisq(statistic, abs(degrees), lower_tail=False))
    return p_values


def _anova_single(fit: SurvregModelResult, with_test: bool) -> SurvregAnovaResult:
    model = fit.fit
    labels = list(fit.term_labels)
    loglik = [0.0] * (len(labels) + 1)
    resid_df = [0.0] * (len(labels) + 1)
    loglik[-1] = -2.0 * float(model.log_likelihood)
    resid_df[-1] = model.df_residual
    for keep in range(len(labels) - 1, -1, -1):
        refit = _refit_terms(fit, keep)
        loglik[keep] = -2.0 * float(refit.log_likelihood)
        resid_df[keep] = refit.df_residual
    df = [math.nan] + [resid_df[k - 1] - resid_df[k] for k in range(1, len(loglik))]
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
    resid_df = [fit.fit.df_residual for fit in fits]
    loglik = [-2.0 * float(fit.fit.log_likelihood) for fit in fits]
    labels = [list(fit.term_labels) for fit in fits]
    tests = [""] + [_diff_term(labels[i - 1], labels[i], i + 1) for i in range(1, len(fits))]
    df = [math.nan] + [resid_df[i - 1] - resid_df[i] for i in range(1, len(fits))]
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


def _dpqr_distribution(distribution: Any, parms: Any | None) -> Any:
    """Density functions use a case-folded exact registry lookup, without prefixes."""

    if isinstance(distribution, str):
        key = distribution.lower()
        if key not in survreg_distributions:
            raise ValueError("Distribution not found")
        distribution = survreg_distributions[key]
        # As R does, families with no parameters ignore the supplied parms.
        if (
            key in _BUILTIN_DISTRIBUTIONS
            and key != "t"
            and isinstance(distribution, SurvregDistribution)
            and not distribution.parms
        ):
            parms = None
    return _resolve_distribution(distribution, parms, probe=False)


def dsurvreg(
    x: Any, mean: Any, scale: Any = 1, distribution: Any = "weibull", parms: Any | None = None
) -> list[float]:
    """Density of the ``survreg`` location-scale distributions (R's ``dsurvreg``)."""

    return _dpqr_distribution(distribution, parms).pdf_values(
        _quantile_vector(x, "x"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
    )


def psurvreg(
    q: Any, mean: Any, scale: Any = 1, distribution: Any = "weibull", parms: Any | None = None
) -> list[float]:
    """Distribution function of the ``survreg`` distributions (R's ``psurvreg``)."""

    return _dpqr_distribution(distribution, parms).cdf_values(
        _quantile_vector(q, "q"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
    )


def qsurvreg(
    p: Any, mean: Any, scale: Any = 1, distribution: Any = "weibull", parms: Any | None = None
) -> list[float]:
    """Quantiles of the ``survreg`` distributions (R's ``qsurvreg``)."""

    return _dpqr_distribution(distribution, parms).quantile_values(
        _quantile_vector(p, "p"),
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
    )


def rsurvreg(
    n: Any,
    mean: Any,
    scale: Any = 1,
    distribution: Any = "weibull",
    parms: Any | None = None,
    seed: int | None = None,
) -> list[float]:
    """Random draws from the ``survreg`` distributions (R's ``rsurvreg``,
    ``qsurvreg(runif(n), ...)``).

    ``seed=s`` draws R's uniforms, so the result equals R's ``set.seed(s);
    rsurvreg(n, ...)``; without a seed the uniforms come from a clock-seeded generator
    whose stream is not R's.
    """

    count = _integer_scalar(n, "n")
    if count < 0:
        raise ValueError("n must be non-negative")
    return _dpqr_distribution(distribution, parms).sample(
        count,
        _quantile_vector(mean, "mean"),
        _quantile_vector(scale, "scale"),
        None if seed is None else _integer_scalar(seed, "seed"),
    )


# ---------------------------------------------------------------------------
# R's ``*.survreg`` methods of the generics in ``_models``, which registers them
# ---------------------------------------------------------------------------


def _normal_two_sided_p_value(statistic: float) -> float:
    """``2 * pnorm(-abs(z))``."""

    if math.isnan(statistic):
        return math.nan
    if math.isinf(statistic):
        return 0.0
    # erfc directly: NormalDist.cdf goes through 1 + erf on older Pythons and
    # cancels for large |z|
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def coef_names_survreg(fit: SurvregModelResult, *, complete: Any | None = None) -> list[str]:
    """``names(coef(fit))``; ``complete=False`` drops the aliased coefficients, as
    ``coef(fit, complete=FALSE)``, and ``complete=True`` gives ``vcov(fit)``'s names,
    the ``Log(scale)`` rows appended."""

    if complete is None:
        return _location_names(fit)
    if _normalize_bool_option(complete, "complete"):
        return survreg_vcov_names(fit)
    return _location_names(fit, complete=False)


def vcov_survreg(fit: SurvregModelResult, *, complete: Any = True) -> list[list[float]]:
    """``vcov.survreg``: ``fit$var``, less the aliased coefficients without ``complete``."""

    return survreg_vcov(fit, _normalize_bool_option_with_default(complete, "complete", True))


def confint_survreg(
    fit: SurvregModelResult, parm: Any | None = None, *, level: Any = 0.95
) -> list[dict[str, float | str]]:
    """``confint.survreg``: normal-approximation intervals for the location coefficients."""

    z = NormalDist().inv_cdf(1.0 - (1.0 - _normalize_conf_level(level, "level")) / 2.0)
    names = _location_names(fit)
    coefficients = fit.coefficients
    variance = survreg_vcov(fit)  # the location block leads fit$var
    return [
        {
            "name": names[idx],
            "lower": coefficients[idx] - z * math.sqrt(max(float(variance[idx][idx]), 0.0)),
            "upper": coefficients[idx] + z * math.sqrt(max(float(variance[idx][idx]), 0.0)),
        }
        for idx in _coefficient_selection(parm, names)
    ]


def model_term_names_survreg(fit: SurvregModelResult, terms: Any | None = None) -> list[str]:
    """``attr(terms(fit), 'term.labels')``, optionally the subset ``terms`` selects."""

    names = list(fit.term_labels)
    selection = _term_selection(terms, names)
    return names if selection is None else [names[idx] for idx in selection]


def model_matrix_survreg(fit: SurvregModelResult, data: Any | None = None) -> dict[str, Any]:
    """``model.matrix.survreg``: the design matrix, its column names and ``assign``."""

    strata_names = {spec.call for spec in fit.strata_terms}
    removed = [i for i, label in enumerate(fit.term_labels, start=1) if label in strata_names]
    return {
        "data": [[float(value) for value in row] for row in fit.fit.covariates]
        if data is None
        else _newdata_inputs(fit, data, "na.omit").x,
        "columns": list(fit.coefficient_names)
        if fit.design is None
        else _model_matrix_names_and_assign(fit.design)[0],
        "assign": [code - sum(index < code for index in removed) for code in fit.assign],
    }


def model_summary_survreg(fit: SurvregModelResult, correlation: Any = False) -> dict[str, Any]:
    """``summary.survreg``: the coefficient table, the pieces of the fit R copies
    (``loglik`` as (intercept-only, full), ``var``, ``scale``, ...) and, with
    ``correlation``, the correlation matrix of the coefficients that are not ``NA``
    (its rows follow theirs in ``coefficient_names``)."""

    model = fit.fit
    coefficients = [float(value) for value in model.coefficients]
    names = survreg_summary_names(fit)
    variance = survreg_vcov(fit)
    naive_variance = model.naive_variance_matrix
    robust = naive_variance is not None
    if naive_variance is None:
        naive_variance = variance

    rows: list[dict[str, float | str]] = []
    standard_errors = []
    for idx, value in enumerate(coefficients):
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        standard_errors.append(standard_error)
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
            "se": standard_error,
            "naive_se": naive_standard_error,
            "z": statistic,
            "p": _normal_two_sided_p_value(statistic),
        }
        if robust:
            row["robust_se"] = standard_error
        rows.append(row)

    correl = None
    if _normalize_bool_option(correlation, "correlation"):
        # diag(1/stds) %*% var[!nas, !nas] %*% diag(1/stds)
        keep = [idx for idx, value in enumerate(coefficients) if not math.isnan(value)]
        correl = [
            [variance[i][j] / (standard_errors[i] * standard_errors[j]) for j in keep] for i in keep
        ]

    result: dict[str, Any] = {
        "model_type": "survreg",
        "coefficients": rows,
        "coefficient_names": names,
        "var": variance,
        "correlation": correl,
        "df": fit.df,
        "n": fit.n,
        "robust": robust,
        "na_action": fit.na_action,
    }
    result.update(survreg_summary(fit))
    return result
