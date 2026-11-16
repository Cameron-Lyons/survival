"""The tt() expansion's design and row names against the eager row-by-row pipeline.

The expanded rows are coded straight into a float array, without row names or a
response subset; ``model_matrix`` rebuilds the names on demand.  The reference fits
here run the earlier pipeline (a list-of-rows design from rows that carry their
names), and every fitted number must match it bit for bit.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import replace

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_coxph = importlib.import_module("survival.r._coxph")
r_formula = importlib.import_module("survival.r._formula")


def tt_log(x, t, riskset, weights):
    return np.asarray(x, dtype=float) * np.log(t)


def tt_root(x, t, riskset, weights):
    return np.asarray(x, dtype=float) * np.sqrt(t)


def tt_matrix(x, t, riskset, weights):
    return np.column_stack([tt_log(x, t, riskset, weights), tt_root(x, t, riskset, weights)])


def tt_factor(x, t, riskset, weights):
    return r.Factor(["hi" if v > 0 else "lo" for v in tt_log(x, t, riskset, weights)], ["lo", "hi"])


def tt_spline(x, t, riskset, weights):
    return r.pspline(tt_log(x, t, riskset, weights), df=2)


def _data(n=120, seed=4):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    time = np.round(rng.exponential(np.exp(-0.4 * x1)), 2) + 0.01
    time[::17] = time[0]
    start = np.round(time * rng.uniform(0.0, 0.5, n), 3)
    return {
        "time": time,
        "start": start,
        "status": rng.integers(0, 2, n),
        "x1": x1,
        "x2": x2,
        "w": 0.5 + rng.integers(1, 4, n),
        "cl": rng.integers(0, 15, n),
        "g": r.Factor([("a", "b", "c")[i % 3] for i in range(n)], ["a", "b", "c"]),
    }


CASES = {
    "plain": ("Surv(time, status) ~ x1 + tt(x2)", {}),
    "strata": ("Surv(time, status) ~ x1 + tt(x2) + strata(g)", {}),
    "factor_term": ("Surv(time, status) ~ g + tt(x2)", {}),
    "interaction": ("Surv(time, status) ~ x1 + tt(x2):g", {}),
    "counting": ("Surv(start, time, status) ~ x1 + tt(x2)", {}),
    "breslow_weights": ("Surv(time, status) ~ x1 + tt(x2)", {"ties": "breslow", "weights": "w"}),
    "cluster": ("Surv(time, status) ~ x1 + tt(x2) + cluster(cl)", {}),
    "no_timefix": ("Surv(time, status) ~ x1 + tt(x2)", {"timefix": False}),
    "two_terms": ("Surv(time, status) ~ tt(x1) + tt(x2)", {"tt": [tt_log, tt_root]}),
    "matrix": ("Surv(time, status) ~ x1 + tt(x2)", {"tt": tt_matrix}),
    "factor_tt": ("Surv(time, status) ~ x1 + tt(x2)", {"tt": tt_factor}),
    "spline_tt": ("Surv(time, status) ~ x1 + tt(x2)", {"tt": tt_spline}),
    "subset": ("Surv(time, status) ~ x1 + tt(x2)", {"subset": [i for i in range(120) if i % 5]}),
}


def _eager_pipeline(monkeypatch):
    """The expansion as it was: named rows with their response, and a row-list design."""

    def named_rows(formula, data, rows, n, **_flags):
        return r_formula._data_rows(data, r_formula._formula_columns(formula, data), rows, n)

    monkeypatch.setattr(r_coxph, "_formula_data_rows", named_rows)
    monkeypatch.setattr(r_coxph, "_design_array_from_spec", r_formula._design_rows_from_spec)


def _bits(values):
    return np.asarray(values, dtype=float).tobytes()


def _expanded_names(fit):
    """The names the eager expansion gave its rows: ``data[tindex, ]`` of the fit's frame."""

    frame = r_coxph._fit_frame(fit)
    y = r.aeqSurv(frame.y) if fit.timefix else frame.y
    core = survival._survival
    if y.start is None:
        counts = core.coxcount1(core.SurvivalData(list(y.time), y._event_codes()), frame.strata)
    else:
        counts = core.coxcount2(
            core.CountingProcessData(list(y.start), list(y.time), y._event_codes()), frame.strata
        )
    rows = [int(row) for row in counts.index]
    return list(r_formula._formula_data_rows(fit.formula, frame.data, rows, frame.n).row_names)


def _fit(data, case):
    formula, options = CASES[case]
    options = {"tt": tt_log, **options}
    return r.coxph(formula, data, **options)


@pytest.mark.parametrize("case", sorted(CASES))
def test_expanded_fit_matches_the_eager_expansion_bitwise(case, monkeypatch):
    data = _data()
    fit = _fit(data, case)
    with monkeypatch.context() as patched:
        _eager_pipeline(patched)
        reference = _fit(data, case)
    assert fit.coefficient_names == reference.coefficient_names
    for name in ("coefficients", "var", "loglik", "means", "linear_predictors"):
        assert _bits(getattr(fit, name)) == _bits(getattr(reference, name)), name
    for kind in ("martingale", "score", "dfbeta"):
        assert _bits(r.residuals(fit, type=kind)) == _bits(r.residuals(reference, type=kind))
    schoenfeld = r.residuals(fit, type="schoenfeld")
    expected_schoenfeld = r.residuals(reference, type="schoenfeld")
    assert _bits(schoenfeld.values) == _bits(expected_schoenfeld.values)
    assert schoenfeld.time == expected_schoenfeld.time
    matrix = r.model_matrix(fit, _with_metadata=True)
    expected = r.model_matrix(reference, _with_metadata=True)
    assert _bits(matrix["data"]) == _bits(expected["data"])
    assert matrix["row_names"] == expected["row_names"] == _expanded_names(fit)
    response = r.Surv(list(fit.y.time), list(fit.y.event))
    for name in ("time", "event", "start", "time2", "type", "states", "clabel"):
        assert getattr(fit.y, name) == getattr(response, name), name


def test_expanded_row_names_follow_the_data_row_names():
    pd = pytest.importorskip("pandas")
    columns = {key: value for key, value in _data().items() if key != "g"}
    frame = pd.DataFrame(columns)
    frame.loc[[4, 9], "x1"] = math.nan
    variants = {
        "filtered": frame[frame.x2 > -1.2],
        "named": frame.set_axis([f"p{i}" for i in range(len(frame))]),
        "default": frame,
    }
    for name, data in variants.items():
        fit = r.coxph("Surv(time, status) ~ x1 + tt(x2)", data, tt=tt_log, na_action="na.exclude")
        names = r.model_matrix(fit, _with_metadata=True)["row_names"]
        assert names == _expanded_names(fit), name
        assert len(set(names)) == len(names), name


def test_names_stored_by_earlier_fits_are_still_read():
    fit = _fit(_data(), "plain")
    rows = len(r.model_matrix(fit)["data"])
    stored = tuple(f"row{i}" for i in range(rows))
    legacy = replace(fit, _matrix_rows=stored)
    assert r.model_matrix(legacy, _with_metadata=True)["row_names"] == list(stored)
