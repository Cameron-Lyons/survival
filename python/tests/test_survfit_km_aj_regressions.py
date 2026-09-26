"""survfit (Kaplan-Meier / Aalen-Johansen) regressions against R 4.5.3 with survival 3.8-12.

Curves derived from a fit (a stratum, ``fit[, states]``, ``survfit0``) are rebuilt from the
engine.
"""

from __future__ import annotations

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
# the R bridge's fit[k] and fit[, states]
_survfit_strata_curves = r._survfit_strata_curves
_subset_survfit_multistate = r._subset_survfit_multistate


def _close(actual, expected, rel=1e-10):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, float) and math.isnan(e):
            assert math.isnan(a), (actual, expected)
        else:
            assert a == pytest.approx(e, rel=rel, abs=1e-12), (actual, expected)


def _close_rows(actual, expected, rel=1e-10):
    assert len(actual) == len(expected), (actual, expected)
    for row, expected_row in zip(actual, expected, strict=True):
        _close(row, expected_row, rel)


def _two_groups():
    return {"t": [1, 2, 3, 4, 5, 6, 7, 8], "e": [1] * 8, "g": [1, 1, 1, 1, 2, 2, 2, 2]}


def _three_states():
    events = ["censor", "a", "b", "a", "b", "censor", "a", "b"]
    return {**_two_groups(), "e": r._r_factor(events, ["censor", "a", "b"])}


# ---------------------------------------------------------------------------
# curves derived from a fit
# ---------------------------------------------------------------------------


def test_methods_of_a_split_stratum_use_that_stratum_only():
    # k <- survfit(Surv(t, e) ~ g); quantile(k[2]), survfit0(k[2]), summary(k[2], c(5.5, 7))
    fit = r.survfit("Surv(t, e) ~ g", _two_groups())
    second = _survfit_strata_curves(fit)["2"]

    assert second.strata is None
    assert second.n == [4]
    quantile = r.quantile_survfit(second)
    _close(quantile.quantile[0], [5.5, 6.5, 7.5])
    _close(quantile.lower[0], [5, 5, 6])
    _close(quantile.upper[0], [math.nan] * 3)
    fit0 = r.survfit0(second)
    _close(fit0.time, [0, 5, 6, 7, 8])
    _close(fit0.surv, [1, 0.75, 0.5, 0.25, 0])
    summary = r.summary_survfit(second, times=[5.5, 7])
    _close(summary.surv, [0.75, 0.25])
    _close(summary.std_err, [0.21650635094611, 0.21650635094611], rel=1e-12)
    _close(summary.n_risk, [3, 2])
    _close(summary.table.values[0], [4, 4, 4, 4, 6.5, 0.559016994374947, 6.5, 5, math.nan])


def test_a_stratum_emptied_by_start_time_keeps_its_own_counts():
    data = {**_two_groups(), "e": [1, 1, 0, 1, 0, 1, 1, 0]}
    fit = r.survfit("Surv(t, e) ~ g", data, start_time=5)
    assert fit.n == [0, 4]

    (only,) = _survfit_strata_curves(fit).values()
    # R's fit[1] reports n = 0 here: it indexes n by the position among the fitted curves
    assert only.n == [4]
    _close(only.surv, [1, 2 / 3, 1 / 3, 1 / 3])


def test_methods_of_a_state_subset_keep_only_those_states():
    # f <- survfit(Surv(t, e) ~ 1); survfit0(f[, "a"]), summary(f[, "a"], times = c(3, 6))
    fit = r.survfit("Surv(t, e) ~ 1", _three_states())
    only_a = _subset_survfit_multistate(fit, [1])

    assert only_a.states == ["a"]
    assert only_a.oldstate == ("(s0)", "a", "b")
    assert only_a.n_id is None
    assert only_a.transitions is None
    assert only_a.hazard_names == []
    assert only_a.std_chaz is None
    fit0 = r.survfit0(only_a)
    assert fit0.states == ["a"]
    assert fit0.oldstate == ("(s0)", "a", "b")
    assert [len(row) for row in fit0.pstate] == [1] * 9
    _close(
        [row[0] for row in fit0.pstate],
        [0, 0, 1 / 7, 1 / 7, 2 / 7, 2 / 7, 2 / 7, 0.5, 0.5],
    )
    summary = r.summary_survfit(only_a, times=[3, 6])
    _close([row[0] for row in summary.pstate], [1 / 7, 2 / 7])
    _close([row[0] for row in summary.std_err], [0.132260014253222, 0.170746944190628], 1e-12)
    assert summary.table.rownames == ["a"]
    _close(summary.table.values[0], [8, 3, 1.64285714285714, 0.84493862664624], 1e-12)
    _close(r.summary_survfit(only_a).time, [2, 4, 7])

    # every state in its original order keeps the transitions and records no oldstate
    every = _subset_survfit_multistate(fit, [0, 1, 2])
    assert every.oldstate is None
    assert every.hazard_names == fit.hazard_names
    assert every.cumhaz == fit.cumhaz


def test_methods_of_a_split_multistate_stratum_match_r():
    # f <- survfit(Surv(t, e) ~ g); survfit0(f[2, ]), summary(f[2, ], times = c(5.5, 7))
    fit = r.survfit("Surv(t, e) ~ g", _three_states())
    second = _survfit_strata_curves(fit)["2"]

    assert second.p0 == [[1.0, 0.0, 0.0]]
    assert second.n == [4]
    fit0 = r.survfit0(second)
    _close(fit0.time, [0, 5, 6, 7, 8])
    _close_rows(
        fit0.pstate,
        [[1, 0, 0], [0.75, 0, 0.25], [0.75, 0, 0.25], [0.375, 0.375, 0.25], [0, 0.375, 0.625]],
    )
    summary = r.summary_survfit(second, times=[5.5, 7])
    _close_rows(summary.pstate, [[0.75, 0, 0.25], [0.375, 0.375, 0.25]])
    _close_rows(
        summary.table.values,
        [
            [4, 0, 6.875, 0.602728172562060],
            [4, 1, 0.375, 0.286410980934740],
            [4, 2, 0.75, 0.649519052838329],
        ],
        1e-12,
    )


def test_split_turnbull_strata_take_their_rows():
    data = {"l": [1, 2, None, 4, 1, 3], "r": [3, 4, 2, None, 2, None], "g": [1, 1, 1, 1, 2, 2]}
    fit = r.survfit("Surv(l, r, type = 'interval2') ~ g", data)
    first = _survfit_strata_curves(fit)["1"]
    alone = r.survfit("Surv(l, r, type = 'interval2') ~ 1", {k: v[:4] for k, v in data.items()})

    assert first.strata is None
    assert (first.n, first.time, first.surv) == (alone.n, alone.time, alone.surv)
