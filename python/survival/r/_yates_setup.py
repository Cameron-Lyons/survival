"""Public prediction setup for Yates marginal means."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _survival as _core
from ._coerce import _match_string_arg, _warn_outside_package
from ._coxph import CoxphModel, survfit_coxph
from ._coxphms import CoxphmsModel
from ._types import _MISSING, CoxSurvfitResult

_COXPH_PREDICT = ["lp", "risk", "expected", "terms", "survival", "linear"]


def _numeric_eta(eta: Any) -> NDArray[np.float64]:
    values = np.asarray(eta)
    if values.dtype.kind not in "biuf":
        raise TypeError("eta must be numeric")
    return values.astype(np.float64, copy=False)


@dataclass(frozen=True)
class YatesLinkPrediction:
    """Callable inverse link; ``X`` is accepted and unused, as in R.

    Cox risk uses the shared native exponential. GLM response calls the
    external family's inverse link with a numeric NumPy input. Output is a
    NumPy array, retaining the inverse link's shape (including scalar shape).
    """

    _inverse_link: Callable[[Any], Any] | None = field(default=None, repr=False)
    _native: _core.YatesPrediction | None = field(default=None, repr=False)

    def __call__(self, eta: Any, X: Any = None) -> NDArray[np.float64]:
        values = _numeric_eta(eta)
        if self._inverse_link is not None:
            return np.asarray(self._inverse_link(values), dtype=np.float64)
        if self._native is None:
            raise ValueError("prediction has no inverse link")
        return self._native.predict(values.reshape(-1)).reshape(values.shape)


def _cox_survival_baseline(fit: Any, options: Any) -> tuple[CoxSurvfitResult, float]:
    if isinstance(fit, CoxphmsModel):
        raise ValueError("multi-state coxph not yet supported")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = survfit_coxph(fit, censor=False)
    if not isinstance(baseline, CoxSurvfitResult):
        raise ValueError("multi-state coxph not yet supported")
    if baseline.strata is not None:
        raise ValueError("stratified models not yet supported")
    horizon = (options or {}).get("rmean")
    try:
        horizon = max(baseline.time, default=-math.inf) if horizon is None else float(horizon)
    except (TypeError, ValueError) as exc:
        raise TypeError("rmean must be numeric") from exc
    if math.isnan(horizon):
        raise ValueError("rmean must not be NaN")
    return baseline, horizon


def _yates_survival_summary(
    baseline: CoxSurvfitResult, curves: _core.YatesCurves, ncurve: int | None = None
) -> CoxSurvfitResult:
    return replace(
        baseline,
        surv=curves.surv,
        cumhaz=curves.cumhaz,
        std_err=curves.std_err,
        std_chaz=curves.std_err,
        lower=curves.lower,
        upper=curves.upper,
        colnames=[str(i + 1) for i in range(ncurve)]
        if not baseline.time and ncurve is not None
        else baseline.colnames,
    )


@dataclass(frozen=True)
class YatesSurvivalSetup(Mapping[str, Callable[..., Any]]):
    """Prepared Cox survival prediction and summary functions.

    ``predict(eta, X=None)`` returns rows containing restricted mean, time-zero
    survival and survival at each baseline time. ``summary(surv, var)`` accepts
    per-population means and variances in that layout and returns curves at the
    original baseline times. It corrects R's extra time-zero summary row and
    stale cumulative hazard, consistently with :func:`yates`.

    Both functions are also accessible as mapping entries. The setup retains
    the baseline curve and prepared increments; it does not retain the fit.
    """

    _baseline: CoxSurvfitResult = field(repr=False)
    _native: _core.YatesPrediction = field(repr=False)
    rmean: float

    def predict(self, eta: Any, X: Any = None) -> NDArray[np.float64]:
        values = _numeric_eta(eta)
        if values.squeeze().ndim > 1:
            raise ValueError("eta must be a scalar, vector, or single-row/column matrix")
        return self._native.predict(values.reshape(-1))

    def summary(self, surv: Any, var: Any) -> CoxSurvfitResult:
        mean, variance = np.asarray(surv, dtype=float), np.asarray(var, dtype=float)
        expected = len(self._baseline.time) + 2
        if mean.ndim != 2 or mean.shape[1] != expected:
            raise ValueError(
                f"surv must have {expected} columns: restricted mean, time zero, times"
            )
        curves = _core.yates_survival_summary(mean, variance, self._baseline.conf_int or 0.95)
        return _yates_survival_summary(self._baseline, curves, mean.shape[0])

    def __getitem__(self, key: str) -> Callable[..., Any]:
        if key == "predict":
            return self.predict
        if key == "summary":
            return self.summary
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("predict", "summary"))

    def __len__(self) -> int:
        return 2


def _inverse_link(fit: Any) -> Callable[[Any], Any] | None:
    family = getattr(fit, "family", None)
    if family is None:
        family = getattr(getattr(fit, "model", None), "family", None)
    function = (
        family.get("linkinv") if isinstance(family, Mapping) else getattr(family, "linkinv", None)
    )
    if not callable(function):
        function = getattr(getattr(family, "link", None), "inverse", None)
    return function if callable(function) else None


def yates_setup(
    fit: Any,
    predict: Any = None,
    options: Any = None,
    *,
    type: Any = _MISSING,
    **kwargs: Any,
) -> None | YatesLinkPrediction | YatesSurvivalSetup:
    """R's ``yates_setup`` prediction dispatcher.

    Cox ``lp``/``linear`` and GLM ``link``/``linear`` return ``None``. Cox
    ``risk`` or GLM ``response`` returns a callable. Cox ``survival`` returns
    the prediction/summary mapping, with ``options['rmean']`` as the horizon.
    Unique abbreviations are accepted. Other setup options and ``**kwargs``
    are unused, as in R; simulation seeds belong to :func:`yates`.

    A GLM supplies ``fit.family.linkinv`` (or a family mapping), or
    ``fit.model.family.link.inverse`` as in statsmodels. A default-class fit
    returns ``None`` and warns only for an explicit nonlinear ``type``;
    ``predict`` is ignored by that default method, matching R.
    """
    if isinstance(fit, CoxphModel):
        kind = _match_string_arg(
            "lp" if predict is None else predict,
            "predict",
            _COXPH_PREDICT,
            "invalid Cox prediction type",
        )
        if kind in ("lp", "linear"):
            return None
        if kind == "risk":
            return YatesLinkPrediction(_native=_core.YatesPrediction())
        if kind in ("expected", "terms"):
            raise ValueError("type expected is not supported")
        baseline, horizon = _cox_survival_baseline(fit, options)
        native = _core.YatesPrediction(baseline.time, baseline.cumhaz, horizon)
        return YatesSurvivalSetup(baseline, native, horizon)
    inverse = _inverse_link(fit)
    if inverse is not None:
        kind = _match_string_arg(
            "link" if predict is None else predict,
            "predict",
            ["link", "response", "terms", "linear"],
            "invalid GLM prediction type",
        )
        if kind in ("link", "linear"):
            return None
        if kind == "terms":
            raise ValueError("type terms not yet supported")
        return YatesLinkPrediction(_inverse_link=inverse)
    if type is not _MISSING and type not in ("linear", "link"):
        _warn_outside_package(
            f"no yates_setup method exists for a model of class {fit.__class__.__name__} "
            f"and estimate type {type}, linear predictor estimate used by default"
        )
    return None
