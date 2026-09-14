"""``aareg`` Aalen additive regression."""

from __future__ import annotations

import warnings
from numbers import Real
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _encode_labels,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _label_levels,
    _materialize_labels,
    _normalize_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
)
from ._fit import _formula_design_output_names
from ._formula import (
    _apply_formula_na_action,
    _combined_columns,
    _design_rows_from_spec,
    _dot_terms,
    _fit_formula_design,
    _formula_model_frame,
    _formula_response_spec,
    _offset_vector,
    _parse_formula,
    _split_terms,
    _subset_formula_inputs,
)
from ._types import AaregModelResult


def _normalize_aareg_test(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("test must be a string")
    normalized = value.strip().casefold()
    choices = ("aalen", "variance", "nrisk")
    if normalized in choices:
        return normalized
    matches = [choice for choice in choices if choice.startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("test must be one of aalen, variance, or nrisk")


def _aareg_taper_values(value: Any) -> list[float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        values = [_finite_float(value, "taper")]
    else:
        values = _float_vector(value, "taper")
    if not values or any(item <= 0.0 for item in values):
        raise ValueError("taper must contain positive finite values")
    return values


def aareg(
    formula: Any,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    qrtol: Any = 1e-7,
    nmin: Any | None = None,
    dfbeta: Any = False,
    taper: Any = 1.0,
    test: Any = "aalen",
    cluster: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = False,
    **kwargs: Any,
) -> Any:
    """Fit Aalen's additive hazards model with R-compatible risk sets."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"aareg got unexpected keyword argument(s): {unexpected}")

    if isinstance(formula, _core.AaregOptions):
        if (
            data is not None
            or weights is not None
            or subset is not None
            or cluster is not None
            or nmin is not None
            or na_action != "fail"
            or qrtol != 1e-7
            or dfbeta is not False
            or taper != 1.0
            or test != "aalen"
            or model is not False
            or x is not False
            or y is not False
        ):
            raise TypeError("AaregOptions input cannot be combined with formula-style options")
        return _core.aareg(formula)
    if not isinstance(formula, str):
        raise TypeError("aareg formula must be a string or AaregOptions")

    response_spec = _formula_response_spec(formula)
    _lhs, _separator, rhs = formula.partition("~")
    formula_terms = _split_terms(rhs, _dot_terms(data, response_spec.columns))
    if len(formula_terms.clusters) > 1:
        raise ValueError("a formula cannot have multiple cluster terms")
    ignored_formula_clusters: tuple[str, ...] = ()
    if formula_terms.clusters and cluster is not None:
        ignored_formula_clusters = tuple(formula_terms.clusters)
        warnings.warn(
            "cluster appears both in a formula and as an argument, formula term ignored",
            RuntimeWarning,
            stacklevel=2,
        )

    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            weights=weights,
            cluster=cluster,
        )
        weights = aligned["weights"]
        cluster = aligned["cluster"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        exclude_columns=ignored_formula_clusters,
        weights=weights,
        cluster=cluster,
    )
    weights = aligned["weights"]
    cluster = aligned["cluster"]

    response, terms = _parse_formula(formula, data)
    if response.type not in {"right", "counting"}:
        raise ValueError(f'Aalen model does not support "{response.type}" survival data')
    if terms.strata:
        raise ValueError("strata terms are not allowed in aareg formulas")
    if terms.clusters and cluster is None:
        cluster = _combined_columns(data, terms.clusters, len(response))
    if terms.offsets:
        _offset_vector(data, terms.offsets, len(response))

    design = _fit_formula_design(data, response_spec, terms, len(response))
    rows = _design_rows_from_spec(data, design, len(response))
    coefficient_names = ["Intercept", *_formula_design_output_names(design)]
    fit_weights = _optional_float_vector(weights, "weights", len(response))
    cluster_values = None if cluster is None else _materialize_labels(cluster, "cluster")
    if cluster_values is not None and len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")
    cluster_levels = (
        None if cluster_values is None else list(_label_levels(cluster_values, "cluster"))
    )
    cluster_codes = None if cluster_values is None else _encode_labels(cluster_values, "cluster")
    keep_dfbeta = _normalize_bool_option(dfbeta, "dfbeta") or cluster_values is not None
    keep_model = _normalize_bool_option(model, "model")
    keep_x = _normalize_bool_option(x, "x")
    keep_y = _normalize_bool_option(y, "y")
    qrtol_value = _finite_float(qrtol, "qrtol")
    if qrtol_value <= 0.0:
        raise ValueError("qrtol must be positive")
    nmin_value = None
    if nmin is not None:
        nmin_value = _integer_scalar(nmin, "nmin")
        if nmin_value < 0:
            raise ValueError("nmin must be non-negative")
    test_name = _normalize_aareg_test(test)

    raw = _core.aareg_fit(
        list(response.time),
        [int(value) for value in response.event],
        rows,
        start=None if response.start is None else list(response.start),
        weights=fit_weights,
        cluster=cluster_codes,
        qrtol=qrtol_value,
        nmin=nmin_value,
        dfbeta=keep_dfbeta,
        taper=_aareg_taper_values(taper),
        test=test_name,
    )
    test_names = (
        coefficient_names[1:]
        if test_name == "variance" and len(coefficient_names) > 2
        else coefficient_names
    )
    model_frame = (
        _formula_model_frame(
            data,
            response,
            design,
            extra_columns=() if ignored_formula_clusters else terms.clusters,
            weights=weights,
            cluster=cluster_values,
        )
        if keep_model
        else None
    )
    return AaregModelResult(
        n=[int(value) for value in raw.n],
        times=[float(value) for value in raw.times],
        n_risk=[float(value) for value in raw.n_risk],
        coefficient=[[float(value) for value in row] for row in raw.coefficient],
        coefficient_names=coefficient_names,
        test_statistic=[float(value) for value in raw.test_statistic],
        test_statistic_names=test_names,
        test_variance=[[float(value) for value in row] for row in raw.test_variance],
        test=str(raw.test),
        time_weights=[[float(value) for value in row] for row in raw.time_weights],
        dfbeta=(
            None
            if raw.dfbeta is None
            else [
                [[float(value) for value in time_values] for time_values in cluster_rows]
                for cluster_rows in raw.dfbeta
            ]
        ),
        robust_test_variance=(
            None
            if raw.robust_test_variance is None
            else [[float(value) for value in row] for row in raw.robust_test_variance]
        ),
        formula=formula,
        weights=fit_weights,
        cluster=cluster_values,
        cluster_levels=cluster_levels,
        model=model_frame,
        x=[list(row) for row in rows] if keep_x else None,
        y=response if keep_y else None,
    )
