"""The survreg response arrays against a reference copy of their element-by-element
definition."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
r_surv = importlib.import_module("survival.r._surv")
Surv = r_surv.Surv


def _reference_response_lists(response: Any) -> tuple[list[float], list[int], list[float] | None]:
    if None in response.event:
        raise ValueError("missing values in the response")
    if response.type == "right":
        status = [float(value or 0) for value in response.event]
        time2 = None
    elif response.type == "left":
        status = [1.0 if value == 1 else 2.0 for value in response.event]
        time2 = None
    elif response.type == "interval":
        status = [float(value or 0) for value in response.event]
        time2 = list(response.time2 or ())
    else:
        raise NotImplementedError("survreg supports right, left, and interval Surv responses")
    return list(response.time), [int(code) for code in status], time2


def _responses() -> list[Any]:
    rng = np.random.default_rng(3)
    n = 200
    time = np.round(rng.exponential(2.0, n), 2) + 0.01
    event = rng.integers(0, 2, n)
    upper = time + np.where(rng.random(n) < 0.3, 0.0, rng.exponential(1.0, n))
    codes = rng.integers(0, 4, n)
    interval_lower = np.where(codes == 2, np.nan, time)
    interval_upper = np.where(codes == 0, np.nan, np.where(codes == 1, time, upper))
    return [
        Surv(time, event),
        Surv(time, event, type="left"),
        Surv(time, upper, codes, type="interval"),
        Surv(interval_lower, interval_upper, type="interval2"),
        Surv(time),
    ]


@pytest.mark.parametrize("index", range(5))
def test_response_arrays_carry_the_reference_columns(index):
    response = _responses()[index]
    time, status, time2 = r_surv._survreg_response_arrays(response)
    ref_time, ref_status, ref_time2 = _reference_response_lists(response)

    assert time.dtype == np.float64
    assert status.dtype == np.int32
    assert time.tobytes() == np.asarray(ref_time, dtype=np.float64).tobytes()
    assert status.tolist() == ref_status
    assert (time2 is None) == (ref_time2 is None)
    if time2 is not None:
        assert time2.tobytes() == np.asarray(ref_time2, dtype=np.float64).tobytes()

    x = np.ones((len(response), 1))
    data = core.SurvregData(time, status, x, time2=time2)
    reference = core.SurvregData(ref_time, ref_status, x, time2=ref_time2)
    assert data.time == reference.time
    assert data.status == reference.status
    assert data.time2 == reference.time2


def test_response_arrays_reject_a_missing_status():
    response = Surv([1.0, 2.0, 3.0], [1, None, 0])
    assert None in response.event
    with pytest.raises(ValueError, match="missing values in the response"):
        r_surv._survreg_response_arrays(response)
