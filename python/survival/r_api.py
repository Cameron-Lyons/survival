"""Backward-compatible façade over :mod:`survival.r`.

The R-style API now lives in the ``survival.r`` package; this module re-exports its public
surface (same ``__all__``) so ``survival.r_api`` keeps working for the R bridge and callers.
The private names below are the ones ``r/survivalr/R/bridge.R`` reaches through ``python_attr``.
"""

from __future__ import annotations

from . import r as _r
from .r import *  # noqa: F403
from .r import (  # noqa: F401
    ConcordanceResult,
    predict_terms_constant,
)
from .r._coerce import _r_factor  # noqa: F401
from .r._misc import _frailty_encoding  # noqa: F401
from .r._models import (  # noqa: F401
    _subset_survfit_multistate,
    _survfit_multistate_structure,
)

__all__ = _r.__all__
