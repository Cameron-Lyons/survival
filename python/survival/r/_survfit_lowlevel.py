"""R's direct survfitKM interface without a retained formula model frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import _survival as _core
from ._coerce import _factor, _mstate_categories, _pop_dotted_keyword
from ._surv import Surv, _complete_codes
from ._survfit import _aligned, _case_weights, _km_engine, _logical, _SurvfitData
from ._types import StrataFactor, SurvfitInfluenceMatrix


@dataclass(frozen=True)
class SurvfitKMResult:
    """Bare numerical curves, preserving every declared factor level.

    ``strata`` includes zero-row curves and ``n``/``n_id`` include zero counts
    for unused levels. Influence matrices use the same per-curve list layout
    as other Python survival results, with ``None`` for unused levels.
    Numeric curve properties return copies; influence values are read-only
    NumPy views of native storage. No observations or model frame are retained.
    """

    _fit: _core.SurvfitKMResult = field(repr=False)
    levels: tuple[str, ...]
    stype: int
    ctype: int
    _observed: tuple[int, ...] = field(repr=False)
    _clname: tuple[Any, ...] | None = field(repr=False)
    _se_fit: bool = field(repr=False)

    def _counts(self, values: Sequence[int]) -> list[int]:
        out = [0] * len(self.levels)
        for code, value in zip(self._observed, values, strict=True):
            out[code] = value
        return out

    @property
    def n(self) -> list[int]:
        return self._counts(self._fit.n)

    @property
    def n_id(self) -> list[int] | None:
        values = self._fit.n_id
        return None if values is None else self._counts(values)

    @property
    def strata(self) -> dict[str, int] | None:
        if len(self.levels) == 1:
            return None
        out = dict.fromkeys(self.levels, 0)
        codes = self._fit.strata_codes or self._observed
        lengths = self._fit.strata
        if lengths is None:
            lengths = [len(self._fit.time)]
        for code, size in zip(codes, lengths, strict=True):
            out[self.levels[code]] = size
        return out

    def _influence(
        self,
        curves: list[_core.SurvfitInfluence] | None,
    ) -> list[SurvfitInfluenceMatrix | None] | None:
        if curves is None:
            return None
        out: list[SurvfitInfluenceMatrix | None] = [None] * len(self.levels)
        codes = self._fit.strata_codes or self._observed
        for code, curve in zip(codes, curves, strict=True):
            out[code] = SurvfitInfluenceMatrix(curve, self._clname)
        return out

    @property
    def influence_surv(self) -> list[SurvfitInfluenceMatrix | None] | None:
        return self._influence(self._fit.influence_surv)

    @property
    def influence_chaz(self) -> list[SurvfitInfluenceMatrix | None] | None:
        return self._influence(self._fit.influence_chaz)

    @property
    def time(self) -> list[float]:
        return self._fit.time

    @property
    def n_risk(self) -> list[float]:
        return self._fit.n_risk

    @property
    def n_event(self) -> list[float]:
        return self._fit.n_event

    @property
    def n_censor(self) -> list[float]:
        return self._fit.n_censor

    @property
    def n_enter(self) -> list[float] | None:
        return self._fit.n_enter

    @property
    def counts(self) -> _core.SurvfitCounts | None:
        return self._fit.counts

    @property
    def surv(self) -> list[float]:
        return self._fit.surv

    @property
    def cumhaz(self) -> list[float]:
        return self._fit.cumhaz

    @property
    def std_err(self) -> list[float] | None:
        return self._fit.std_err

    @property
    def std_chaz(self) -> list[float] | None:
        return self._fit.std_chaz

    @property
    def lower(self) -> list[float] | None:
        return self._fit.lower

    @property
    def upper(self) -> list[float] | None:
        return self._fit.upper

    @property
    def type(self) -> str:
        return self._fit.type

    @property
    def t0(self) -> float:
        return self._fit.t0

    @property
    def logse(self) -> bool | None:
        return None if not self._se_fit else self._fit.logse

    @property
    def conf_int(self) -> float | None:
        return None if not self._se_fit else self._fit.conf_int

    @property
    def conf_type(self) -> str | None:
        return None if not self._se_fit else self._fit.conf_type

    @property
    def conf_lower(self) -> str | None:
        return None if not self._se_fit or self._fit.conf_lower == "usual" else self._fit.conf_lower


def survfitKM(
    x: Any,
    y: Surv,
    weights: Any = None,
    stype: Any = 1,
    ctype: Any = 1,
    se_fit: Any = True,
    conf_int: Any = 0.95,
    conf_type: Any = "log",
    conf_lower: Any = "usual",
    start_time: Any = None,
    id: Any = None,
    cluster: Any = None,
    robust: Any = None,
    influence: Any = False,
    type: Any = None,
    entry: Any = False,
    time0: Any = False,
    **kwargs: Any,
) -> SurvfitKMResult:
    """Fit a factor and prepared right/counting ``Surv`` without formula processing.

    ``x`` must be a ``StrataFactor`` (from ``strata``) or categorical array/series.
    Declared level order and unused levels are preserved. Close times remain
    distinct; no missing rows are omitted. Old-style ``type`` overrides
    ``stype``/``ctype``. ``time0`` is accepted and unused, as in R's direct fitter.
    """
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    conf_lower = _pop_dotted_keyword(kwargs, "conf.lower", "conf_lower", conf_lower, "usual")
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    if kwargs:
        raise TypeError(
            f"survfitKM got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    if y.type not in {"right", "counting"}:
        raise ValueError("Can only handle right censored or counting data")
    if isinstance(x, StrataFactor):
        codes = _complete_codes(x, "x contains missing values")
        levels = list(x.levels)
        if any(code < 0 or code >= len(levels) for code in codes):
            raise ValueError("x contains invalid factor codes")
    elif _mstate_categories(x) is not None:
        raw_codes, levels = _factor(x, "x")
        if any(code is None for code in raw_codes):
            raise ValueError("x contains missing values")
        codes = [int(code) for code in raw_codes if code is not None]
    else:
        raise TypeError("x must be a factor")
    if len(codes) != len(y):
        raise ValueError("x and y have different lengths")
    if not codes:
        raise ValueError("data set has no non-missing observations")
    if len(set(levels)) != len(levels):
        raise ValueError("x factor levels must have unique labels")
    n = len(y)

    # R treats empty id/cluster vectors as absent. Alignment and missing-value
    # checks remain explicit because no formula model frame runs here.
    def optional(value: Any, name: str) -> list[Any] | None:
        return None if value is None or len(value) == 0 else _aligned(value, n, name)

    frame = _SurvfitData(
        y,
        codes,
        levels,
        _case_weights(weights, n),
        optional(id, "id"),
        optional(cluster, "cluster"),
        None,
        {},
        (),
    )
    engine, call, clname = _km_engine(
        frame,
        stype=stype,
        ctype=ctype,
        type_=type,
        se_fit=se_fit,
        conf_int=conf_int,
        conf_type=conf_type,
        conf_lower=conf_lower,
        start_time=start_time,
        robust=robust,
        influence=influence,
        entry=_logical(entry, "entry argument must be TRUE/FALSE"),
        reverse=False,
        timefix=False,
        id_name=None,
    )
    return SurvfitKMResult(
        engine,
        tuple(levels),
        call.stype,
        call.ctype,
        tuple(sorted(set(codes))),
        clname,
        bool(se_fit),
    )
