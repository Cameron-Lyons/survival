"""Backward-compatible façade over :mod:`survival.r`.

The R-style API now lives in the ``survival.r`` package; this module re-exports its public
surface (same ``__all__``) so ``survival.r_api`` keeps working for the R bridge and callers.
The private names below are the ones ``r/survivalr/R/bridge.R`` reaches through ``python_attr``.
"""

from __future__ import annotations

from . import r as _r
from .r import *
from .r._coerce import _r_factor, _r_term_subscript
from .r._coxph import _cox_fit_diagnostic_messages
from .r._misc import _frailty_encoding, _survcheck_codes
from .r._models import (
    _prediction_term_names,
    _subset_coxms_curves,
    _subset_survfit_multistate,
    _survfit_multistate_structure,
    _survfit_strata_curves,
)
from .r._penalties import _penalty_control
from .r._survfit import _subset_cox_survfit
from .r._survreg import survreg_vcov_names as _survreg_vcov_names

__all__ = _r.__all__
