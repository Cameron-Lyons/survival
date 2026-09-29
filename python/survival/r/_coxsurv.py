"""Prepared Cox survival curves: R's coxsurv.fit and survfitcoxph.fit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, overload

import numpy as np
from numpy.typing import NDArray

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _encode_labels,
    _factor,
    _integer_scalar,
    _is_missing_value,
    _label_levels,
    _materialize_labels,
    _normalize_bool_option,
    _numeric_design_matrix,
    _pop_dotted_keyword,
)
from ._surv import Surv
from ._types import StrataFactor


@dataclass(frozen=True)
class CoxSurvFitResult:
    """Bare curves with read-only NumPy views and explicit labels.

    Ordinary predictions have one matrix column per new row (a vector for
    one row); individual predictions have vectors. ``n`` always lists counts
    per curve. ``strata`` is absent for one curve. List-mode results also
    expose ``hazard``, ``varhaz``, ``ndeath`` and ``xbar`` for ordinary curves.
    No fitted model, response, or design matrix is retained.
    """

    _fit: _core.CoxSurvRawResult = field(repr=False)
    labels: tuple[str, ...]
    row_names: tuple[str, ...] | None
    individual: bool
    _range: slice = field(repr=False)
    _counts: tuple[int, ...] = field(repr=False)

    @property
    def n(self) -> list[int]:
        return list(self._counts)

    @property
    def strata(self) -> dict[str, int] | None:
        return (
            dict(zip(self.labels, self._fit.lengths, strict=True)) if len(self.labels) > 1 else None
        )

    def _values(self, name: str) -> NDArray[np.float64] | None:
        values = getattr(self._fit, name)
        if values is None:
            return None
        out = values[self._range]
        if name in {"surv", "cumhaz", "std_err"} and out.shape[1] == 1:
            return out[:, 0]
        return out

    @property
    def time(self) -> NDArray[np.float64]:
        return self._fit.time[self._range]

    @property
    def n_risk(self) -> NDArray[np.float64]:
        return self._fit.n_risk[self._range]

    @property
    def n_event(self) -> NDArray[np.float64]:
        return self._fit.n_event[self._range]

    @property
    def n_censor(self) -> NDArray[np.float64]:
        return self._fit.n_censor[self._range]

    @property
    def surv(self) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._values("surv"))

    @property
    def cumhaz(self) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._values("cumhaz"))

    @property
    def std_err(self) -> NDArray[np.float64] | None:
        return self._values("std_err")

    @property
    def hazard(self) -> NDArray[np.float64] | None:
        return self._values("hazard")

    @property
    def varhaz(self) -> NDArray[np.float64] | None:
        return self._values("varhaz")

    @property
    def ndeath(self) -> NDArray[np.float64] | None:
        return self._values("ndeath")

    @property
    def xbar(self) -> NDArray[np.float64] | None:
        return self._values("xbar")


@dataclass(frozen=True)
class CoxSurvFitList(Sequence[CoxSurvFitResult]):
    """Named per-curve views, returned by ``unlist=False``.

    Names identify strata for ordinary curves or IDs for individual curves.
    Every view shares the same native owner; selecting one retains its siblings.
    """

    curves: tuple[CoxSurvFitResult, ...]
    names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.curves)

    @overload
    def __getitem__(self, index: int) -> CoxSurvFitResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CoxSurvFitResult, ...]: ...

    def __getitem__(self, index: int | slice) -> CoxSurvFitResult | tuple[CoxSurvFitResult, ...]:
        return self.curves[index]


def _response(value: Any, name: str, *, prediction: bool = False) -> NDArray[np.float64]:
    if isinstance(value, Surv):
        if value.type not in ({"counting"} if prediction else {"right", "counting"}):
            raise ValueError(
                f"{name} requires {'counting' if prediction else 'right or counting'} data"
            )
        columns: list[Any] = [value.time] if prediction else [value.time, value.event]
        if value.start is not None:
            columns.insert(0, value.start)
        value = np.column_stack(columns)
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 columns")
    if matrix.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric with no missing status values")
    return matrix.astype(np.float64, copy=False)


def _vector(value: Any, n: int, name: str) -> NDArray[np.float64]:
    if value is None:
        raise ValueError(f"{name} is required")
    values = np.asarray(value)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must have {n} entries")
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric")
    return values.astype(np.float64, copy=False)


def _strata(values: Any, n: int) -> tuple[list[int], tuple[str, ...]]:
    if values is None or len(values) == 0:
        return [0] * n, ("0",)
    if isinstance(values, StrataFactor):
        raw, levels = values.codes, values.levels
    else:
        raw, levels = _factor(values, "strata")
    if len(raw) != n or any(code is None for code in raw):
        raise ValueError("strata must have one non-missing value per response row")
    if len(set(levels)) != len(levels):
        raise ValueError("strata must have unique level labels")
    return [int(code) for code in raw if code is not None], tuple(levels)


def coxsurv_fit(
    ctype: Any = 1,
    stype: Any = 2,
    se_fit: Any = True,
    varmat: Any = None,
    cluster: Any = None,
    y: Any = None,
    x: Any = None,
    wt: Any = None,
    risk: Any = None,
    position: Any = None,
    strata: Any = None,
    oldid: Any = None,
    y2: Any = None,
    x2: Any = None,
    risk2: Any = None,
    strata2: Any = None,
    id2: Any = None,
    unlist: Any = True,
    *,
    rownames: Any = None,
    **kwargs: Any,
) -> CoxSurvFitResult | CoxSurvFitList:
    """Compute curves from prepared Cox responses, covariates and relative risks.

    Designs must already use the same centering convention as the supplied
    risks. ``strata2`` uses R's one-based positions into the original stratum
    levels. ``id2`` enables (start, stop] trajectories, preserving input order.
    ``cluster``, ``position`` and ``oldid`` are accepted and unused, as in R.
    The caller supplies ``varmat`` when requesting standard errors.
    """
    se_fit = _normalize_bool_option(
        _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True), "se.fit"
    )
    if kwargs:
        raise TypeError(
            f"coxsurv_fit got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    unlist = _normalize_bool_option(unlist, "unlist")
    stype, ctype = _integer_scalar(stype, "stype"), _integer_scalar(ctype, "ctype")
    if stype not in (1, 2) or ctype not in (1, 2):
        raise ValueError("stype and ctype must be 1 or 2")
    response = _response(y, "y")
    n = response.shape[0]
    design = _numeric_design_matrix(x, n, empty=True)
    p = design.shape[1]
    # A vector represents one prepared prediction row, including p=0.
    if isinstance(x2, dict):
        new_x = np.column_stack(list(x2.values())) if x2 else np.empty((1, 0))
    else:
        new_x = np.asarray(x2.to_numpy() if hasattr(x2, "to_numpy") else x2)
        if new_x.ndim == 1:
            new_x = new_x.reshape(1, -1)
    if new_x.ndim != 2 or new_x.shape[0] == 0 or new_x.shape[1] != p:
        raise ValueError("x2 needs at least one row and the same columns as x")
    new_x = _numeric_design_matrix(new_x, new_x.shape[0])
    m = new_x.shape[0]
    weights = np.ones(n) if wt is None else _vector(wt, n, "wt")
    original_risk, new_risk = _vector(risk, n, "risk"), _vector(risk2, m, "risk2")
    codes, levels = _strata(strata, n)
    variance = None
    if se_fit:
        if varmat is None:
            raise ValueError("varmat is required when se.fit is true")
        variance = np.asarray(varmat, dtype=float)
        if variance.shape != (p, p):
            raise ValueError(f"varmat must have shape ({p}, {p})")
    ids = None
    new_response = None
    new_strata = None
    labels = levels
    if id2 is not None:
        values = _materialize_labels(id2, "id2")
        if len(values) != m or any(_is_missing_value(value) for value in values):
            raise ValueError("id2 must have one non-missing value per x2 row")
        ids = _encode_labels(values, "id2")
        labels = tuple(_as_character(value) for value in _label_levels(values, "id2"))
        if len(set(labels)) != len(labels):
            raise ValueError("id2 must have unique printed labels")
        new_response = _response(y2, "y2", prediction=True)
        if strata2 is not None:
            if isinstance(strata2, StrataFactor):
                new_strata = [-1 if code is None else code for code in strata2.codes]
            elif hasattr(strata2, "categories") or hasattr(strata2, "cat"):
                factor_codes, _ = _factor(strata2, "strata2")
                new_strata = [-1 if code is None else code for code in factor_codes]
            else:
                new_strata = [
                    _integer_scalar(value, "strata2") - 1
                    for value in _materialize_labels(strata2, "strata2")
                ]
    names = rownames
    if names is None and hasattr(x2, "index") and not callable(x2.index):
        names = x2.index
    row_names = (
        None
        if names is None
        else tuple(_as_character(v) for v in _materialize_labels(names, "rownames"))
    )
    if row_names is not None and len(row_names) != m:
        raise ValueError("rownames must have one label per x2 row")
    fitted = _core.coxsurv_fit(
        response,
        design,
        weights,
        original_risk,
        codes,
        len(levels),
        new_x,
        new_risk,
        stype=stype,
        ctype=ctype,
        varmat=variance,
        y2=new_response,
        strata2=new_strata,
        id2=ids,
        keep_details=not unlist,
    )
    counts = tuple(fitted.n)
    result = CoxSurvFitResult(fitted, labels, row_names, ids is not None, slice(None), counts)
    if unlist:
        return result
    curves = []
    start = 0
    for index, (label, length) in enumerate(zip(labels, fitted.lengths, strict=True)):
        curves.append(
            CoxSurvFitResult(
                fitted,
                (label,),
                row_names,
                ids is not None,
                slice(start, start + length),
                (counts[index],),
            )
        )
        start += length
    return CoxSurvFitList(tuple(curves), labels)


def survfitcoxph_fit(
    y: Any,
    x: Any,
    wt: Any = None,
    x2: Any = None,
    risk: Any = None,
    newrisk: Any = None,
    strata: Any = None,
    se_fit: Any = True,
    survtype: Any = 1,
    vartype: Any = None,
    varmat: Any = None,
    id: Any = None,
    y2: Any = None,
    strata2: Any = None,
    unlist: Any = True,
    *,
    rownames: Any = None,
    **kwargs: Any,
) -> CoxSurvFitResult | CoxSurvFitList:
    """Legacy direct Cox curves. ``vartype`` is unused, as in R's wrapper.

    ``survtype`` selects 1=Kalbfleisch–Prentice, 2=Breslow or 3=Efron.
    ``newrisk`` must supply one relative risk per prediction row.
    """
    code = _integer_scalar(survtype, "survtype")
    if code not in (1, 2, 3):
        raise ValueError("survtype must be 1, 2, or 3")
    return coxsurv_fit(
        ctype=2 if code == 3 else 1,
        stype=1 if code == 1 else 2,
        se_fit=se_fit,
        varmat=varmat,
        y=y,
        x=x,
        wt=wt,
        risk=risk,
        strata=strata,
        y2=y2,
        x2=x2,
        risk2=newrisk,
        strata2=strata2,
        id2=id,
        unlist=unlist,
        rownames=rownames,
        **kwargs,
    )
