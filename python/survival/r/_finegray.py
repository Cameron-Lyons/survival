"""``finegray``: the Fine-Gray weighted data set for competing risks.

A port of the R code of ``R/finegray.R``: the model frame, the multi-state
response checks, the censoring distribution (``survfitkm`` on the ranked
times) and the per-stratum split (the ``finegray`` kernel).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _int_vector,
    _is_missing_value,
    _materialize_1d,
    _normalize_bool_option,
    _pop_dotted_keyword,
)
from ._formula import _model_variables, _strata_keep, _strata_specs, model_frame
from ._names import _make_names
from ._surv import Surv, _complete_codes
from ._types import FineGrayFrame, ModelFrame


def _etype_index(states: Sequence[str], etype: Any) -> int:
    """R's ``match(etype, states)[1]`` (one-based) with its checks and warning."""

    if etype is None:
        return 1
    requested = (
        [etype] if isinstance(etype, str) else [str(v) for v in _materialize_1d(etype, "etype")]
    )
    index = [states.index(value) + 1 if value in states else None for value in requested]
    if any(value is None for value in index) or not index:
        raise ValueError("etype argument has a state that is not in the data")
    if len(index) > 1:
        warnings.warn("only the first endpoint was used", stacklevel=3)
    return int(index[0] or 1)


def _finegray_inputs(mf: ModelFrame) -> tuple[Surv, list[int], list[float] | None]:
    """The checked response, the stratum of each row and the user weights."""

    response = mf.response
    if not isinstance(response, Surv):
        raise ValueError("Response must be a survival object")
    if response.type not in {"mright", "mcounting"}:
        raise ValueError("Fine-Gray model requires a multi-state survival")
    if len(response.states) < 2:
        raise ValueError("survival time has only a single state")
    if any(value is None for value in response.event) or any(
        math.isnan(value) for value in (*response.time, *(response.start or ()))
    ):
        raise ValueError("missing values in the response")
    if mf.terms.clusters:
        raise ValueError("a cluster() term is not valid")
    if mf.terms.strata:
        factor = _strata_keep(mf.data, _strata_specs(mf.terms))
        istrat = _complete_codes(factor, "strata must not contain missing values")
    else:
        istrat = [0] * mf.n
    weights = None if mf.weights is None else [_finite_float(v, "weights") for v in mf.weights]
    return response, istrat, weights


def finegray(
    formula: str,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.pass",
    etype: Any | None = None,
    prefix: str = "fg",
    count: str | None = None,
    id: Any | None = None,
    timefix: bool = True,
    **kwargs: Any,
) -> FineGrayFrame:
    """R's ``finegray``: expand a multi-state ``Surv`` response into Fine-Gray weighted rows.

    The result carries the model-frame columns plus ``<prefix>start``,
    ``<prefix>stop``, ``<prefix>status``, ``<prefix>wt`` (and ``count``), with the
    selected endpoint as its ``event`` attribute.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.pass")
    if kwargs:
        raise TypeError(f"finegray got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if not isinstance(formula, str):
        raise ValueError("A formula argument is required")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    fix_time = _normalize_bool_option(timefix, "timefix")
    mf = model_frame(formula, data, subset=subset, na_action=na_action, weights=weights, id=id)
    if mf.n == 0:
        raise ValueError("No (non-missing) observations")
    response, istrat, user_weights = _finegray_inputs(mf)
    enum = _etype_index(list(response.states), etype)
    count_name = None if count is None else _make_names(str(count))
    output_names = [f"{prefix}{suffix}" for suffix in ("start", "stop", "status", "wt")]

    # Preserve the existing subject-label convention; only compact integer codes
    # cross the native boundary. Covariate objects remain in Python.
    ids = None
    if mf.id is not None:
        if any(_is_missing_value(value) for value in mf.id):
            raise ValueError("id must not contain missing values")
        labels = [str(value) for value in mf.id]
        codes = {value: i for i, value in enumerate(dict.fromkeys(labels))}
        ids = [codes[value] for value in labels]
    status = _int_vector(response.event, "status")
    split = _core.finegray_expand(
        response.time,
        status,
        enum,
        start=response.start,
        strata=istrat,
        id=ids,
        weights=user_weights,
        timefix=fix_time,
    )
    source = [row - 1 for row in split.row]
    variables = [
        (name, values) for name, values in _model_variables(mf) if not name.startswith("strata(")
    ]
    if mf.weights is not None:
        variables.append(("(weights)", list(mf.weights)))
    columns: dict[str, list[Any]] = {
        name: [values[idx] for idx in source] for name, values in variables
    }
    columns[output_names[0]] = split.start
    columns[output_names[1]] = split.end
    columns[output_names[2]] = [int(status[idx] == enum) for idx in source]
    columns[output_names[3]] = split.wt
    if count_name is not None:
        columns[count_name] = split.add
    return FineGrayFrame(columns, event=response.states[enum - 1])


def _finegray_frame(result: _core.FineGrayOutput) -> dict[str, list[Any]]:
    """``as_data_frame`` of a raw ``FineGrayOutput``."""

    return {
        "row": [int(value) for value in result.row],
        "start": [float(value) for value in result.start],
        "end": [float(value) for value in result.end],
        "wt": [float(value) for value in result.wt],
        "add": [int(value) for value in result.add],
    }
