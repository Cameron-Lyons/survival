"""survfit (Kaplan-Meier / Aalen-Johansen) regressions against R 4.5.3 with survival 3.8-12.

Curves derived from a fit (a stratum, ``fit[, states]``, ``survfit0``) are rebuilt from the
engine, KM fits keep R's ``time0``/``start.time`` semantics, and influence rows carry R's
row names.
"""

from __future__ import annotations

import math
import warnings

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


def _labelled():
    return {
        "time": [1, 2, 3, 4, 5, 6],
        "status": [1, 0, 1, 1, 0, 1],
        "e": r._r_factor(["censor", "a", "b", "a", "censor", "b"], ["censor", "a", "b"]),
        "cl": ["z", "z", "q", "q", "m", "m"],
        "id": [101, 102, 203, 204, 305, 309],
        "g": [1, 2, 1, 2, 1, 2],
    }


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


# ---------------------------------------------------------------------------
# time0 and start.time: survfitKM and survfitTurnbull set neither
# ---------------------------------------------------------------------------


def test_survfit0_adds_the_time_0_row_to_a_time0_km_fit():
    # survfit0(survfit(Surv(c(1, 2, 3, 4, 5), c(1, 0, 1, 1, 0)) ~ 1, time0 = TRUE))
    fit = r.survfit(r.Surv([1, 2, 3, 4, 5], [1, 0, 1, 1, 0]), time0=True)
    fit0 = r.survfit0(fit)

    _close(fit0.time, [0, 1, 2, 3, 4, 5])
    _close(fit0.surv, [1, 0.8, 0.8, 0.533333333333333, 0.266666666666667, 0.266666666666667])
    se = [0, 0.223606797749979, 0.223606797749979, 0.465474668125631] + [0.84656167328002] * 2
    _close(fit0.std_err, se, 1e-12)


def test_rmean_of_a_start_time_km_fit_is_checked_against_its_first_time():
    # f <- survfit(Surv(time, status) ~ 1, lung, start.time = 100): f$start.time is NULL
    fit = r.survfit("Surv(time, status) ~ 1", survival.datasets.load_lung(), start_time=100)

    assert not hasattr(fit, "start_time")
    assert fit.t0 == 100.0
    with pytest.raises(ValueError, match="Truncation point for the mean time in state"):
        r.summary_survfit(fit, rmean=100.5)
    table = r.summary_survfit(fit, rmean=200).table.values[0]
    _close(table[4:6], [90.8389344412138, 1.5456137540005], 1e-12)
    assert r.survfit0(fit).time[:2] == [100.0, 105.0]


def test_turnbull_fits_record_neither_start_time_nor_time0():
    data = {"l": [1, 2, None, 4], "r": [3, 4, 2, None]}
    fit = r.survfit("Surv(l, r, type = 'interval2') ~ 1", data, start_time=1, time0=True)

    assert not hasattr(fit, "start_time")
    assert (fit.time0, fit.t0) == (False, 1.0)
    _close(fit.time, [1.5, 2.5, 4])


# ---------------------------------------------------------------------------
# influence row names and survfitAJ's cluster warning
# ---------------------------------------------------------------------------


def test_km_influence_rows_are_named_by_cluster_id_or_row_number():
    data = _labelled()

    fit = r.survfit("Surv(time, status) ~ 1", data, cluster="cl", influence=True)
    assert isinstance(fit.influence_surv[0], r.SurvfitInfluence)
    assert fit.influence_surv[0].cluster == ["z", "q", "m"]
    assert fit.influence_chaz[0].cluster == ["z", "q", "m"]
    _close(
        fit.influence_surv[0].values[0],
        [-1 / 9, -1 / 9, -0.0833333333333333, -1 / 18, -1 / 18, 0],
        1e-12,
    )
    assert r.survfit0(fit).influence_surv[0].cluster == ["z", "q", "m"]
    by_id = r.survfit("Surv(time, status) ~ 1", data, id="id", influence=True)
    assert by_id.influence_surv[0].cluster == [101, 102, 203, 204, 305, 309]
    late = r.survfit("Surv(time, status) ~ 1", data, id="id", influence=True, start_time=3)
    assert late.influence_surv[0].cluster == [203, 204, 305, 309]
    by_row = r.survfit("Surv(time, status) ~ 1", data, influence=True, start_time=2)
    assert by_row.influence_surv[0].cluster == [1, 2, 3, 4, 5]
    by_group = r.survfit("Surv(time, status) ~ g", data, id="id", influence=1)
    assert [curve.cluster for curve in by_group.influence_surv] == [
        [101, 203, 305],
        [102, 204, 309],
    ]
    assert _survfit_strata_curves(by_group)["2"].influence_surv[0].cluster == [102, 204, 309]


def test_aj_influence_rows_are_numbered_as_in_r():
    # dimnames(fit$influence.pstate)[[1]]: survfitAJ names the rows by the cluster numbers
    data = _labelled()

    def rows(formula, **kwargs):
        fit = r.survfit(formula, data, influence=True, **kwargs)
        return [curve.cluster for curve in fit.influence_pstate]

    assert rows("Surv(time, e) ~ 1", cluster="cl") == [[1, 2, 3]]
    assert rows("Surv(time, e) ~ 1", id="id") == [[1, 2, 3, 4, 5, 6]]
    assert rows("Surv(time, e) ~ 1", start_time=2) == [[1, 2, 3, 4, 5]]
    assert rows("Surv(time, e) ~ 1", id="id", start_time=3) == [[1, 2, 3, 4]]
    assert rows("Surv(time, e) ~ g") == [[1, 3, 5], [2, 4, 6]]
    assert rows("Surv(time, e) ~ g", cluster="cl") == [[1, 2, 3], [1, 2, 3]]


def test_an_id_on_two_clusters_warns():
    events = r._r_factor(["cens", "a", "cens", "b", "cens", "a"], ["cens", "a", "b"])
    data = {
        "id": [1, 1, 2, 2, 3, 3],
        "cl": ["x", "y", "x", "x", "y", "y"],
        "t1": [0, 1, 0, 3, 0, 5],
        "t2": [1, 2, 3, 4, 5, 6],
        "st": events,
    }
    # subject 1 lies in clusters x and y; R's Ctwoclust misses it (survfitAJ passes it
    # the 1-based order(id)), the intended check does not
    with pytest.warns(UserWarning, match="an id value appears on more than one cluster"):
        fit = r.survfit("Surv(t1, t2, st) ~ 1", data, id="id", cluster="cl")
    _close_rows(fit.pstate, [[2 / 3, 1 / 3, 0], [1 / 3, 1 / 3, 1 / 3], [0, 2 / 3, 1 / 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.survfit(
            "Surv(t1, t2, st) ~ 1",
            {**data, "cl": ["x", "x", "y", "y", "z", "z"]},
            id="id",
            cluster="cl",
        )
