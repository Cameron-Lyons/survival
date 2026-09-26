"""``residuals.survfit`` and ``pseudo`` against R's ``survival`` (3.8.11) on small data."""

from __future__ import annotations

import math

import pytest

from .helpers import setup_survival_import
from .r_api_support import _toy_data

survival = setup_survival_import()
r = survival.r_api


def _close(actual, expected, rel=1e-8):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e, rel=rel, abs=1e-12), (actual, expected)


def _group_fit():
    return r.survfit("Surv(time, status) ~ group", _toy_data())


def test_residuals_survfit_matches_r_pstate_and_cumhaz():
    result = r.survfit_residuals(_group_fit(), times=[5, 2])

    assert isinstance(result, r.SurvfitResidualsResult)
    assert result.time == [2.0, 5.0]
    assert result.id == [1, 2, 3, 4, 5, 6, 7, 8]
    assert result.curve == [1, 1, 1, 1, 2, 2, 2, 2]
    assert result.columns is None
    assert result.id_name is None
    _close([row[0] for row in result.resid], [-0.125, -0.125, 0.125, 0.125, 0, 0, 0, 0])
    _close([row[1] for row in result.resid], [0] * 8)
    cumhaz = r.survfit_residuals(_group_fit(), times=[2, 5], type="chaz")
    _close(
        [row[0] for row in cumhaz.resid],
        [0.1875, 0.159722222222222, -0.173611111111111, -0.173611111111111, 0, 0, 0, 0],
    )


def test_residuals_survfit_collapse_and_weights_follow_r_rules():
    data = _toy_data()
    data["id"] = [1, 1, 2, 2, 3, 3, 4, 4]
    fit = r.survfit("Surv(time, status) ~ group", data, id="id", weights=[1, 2, 1, 2, 1, 2, 1, 2])

    collapsed = r.survfit_residuals(fit, times=[2], collapse=True)
    assert collapsed.id == [1, 2, 3, 4]
    assert collapsed.curve == [1, 1, 2, 2]
    assert collapsed.id_name == "id"
    per_row = r.survfit_residuals(fit, times=[2], weighted=True)
    _close(
        [row[0] for row in collapsed.resid],
        [
            per_row.resid[0][0] + per_row.resid[1][0],
            per_row.resid[2][0] + per_row.resid[3][0],
            0.0,
            0.0,
        ],
    )
    with pytest.raises(ValueError, match="collapse=TRUE and weighted=FALSE"):
        r.survfit_residuals(fit, times=[2], collapse=True, weighted=False)
    # ids that do not repeat cannot be collapsed: R silently turns collapse off
    plain = r.survfit_residuals(_group_fit(), times=[2], collapse=True)
    assert len(plain.id) == 8


def test_residuals_survfit_collapse_without_weights_matches_r():
    # residuals(fit, times, collapse=TRUE) on counting-process data with repeated ids and
    # no case weights: R's `weighted <- FALSE` for missing weights must not reach the kernel
    data = {
        "id": [1, 1, 2, 2, 3, 3, 4, 4],
        "t1": [0, 1, 0, 2, 0, 1, 0, 3],
        "t2": [1, 4, 2, 5, 1, 3, 3, 6],
        "st": [0, 1, 1, 1, 0, 1, 0, 1],
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
    }
    fit = r.survfit("Surv(t1, t2, st) ~ g", data, id="id")

    collapsed = r.survfit_residuals(fit, times=[2, 5], collapse=True)
    assert collapsed.id == [1, 2, 3, 4]
    assert collapsed.curve == [1, 1, 2, 2]
    assert collapsed.id_name == "id"
    expected = [[0.25, 0.0], [-0.25, 0.0], [0.0, -0.25], [0.0, 0.25]]
    for row, want in zip(collapsed.resid, expected, strict=True):
        _close(row, want)
    explicit = r.survfit_residuals(fit, times=[2, 5], collapse=True, weighted=True)
    assert explicit.resid == collapsed.resid
    cumhaz = r.survfit_residuals(fit, times=[2, 5], collapse=True, type="cumhaz")
    for row, want in zip(cumhaz.resid, [[-0.25, 0], [0.25, 0], [0, 0.25], [0, -0.25]], strict=True):
        _close(row, want)
    per_row = r.survfit_residuals(fit, times=[2, 5])
    assert per_row.id == [1, 1, 2, 2, 3, 3, 4, 4]
    _close([row[0] for row in per_row.resid], [0, 0.25, -0.25, 0, 0, 0, 0, 0])

    states = {
        "id": [1, 1, 2, 2, 3, 3, 4],
        "t1": [0, 2, 0, 1, 0, 3, 0],
        "t2": [2, 5, 1, 4, 3, 6, 4],
        "ev": r._r_factor(["b", "a", "b", "censor", "b", "censor", "a"], ["censor", "a", "b"]),
    }
    multi = r.survfit_residuals(
        r.survfit("Surv(t1, t2, ev) ~ 1", states, id="id"), times=[2, 4], collapse=True
    )
    assert multi.id == [1, 2, 3, 4]
    assert multi.columns == ["(s0)", "a", "b"]
    expected_states = [
        [[-0.125, 0], [0, -0.0625], [0.125, 0.0625]],
        [[-0.125, 0], [0, -0.0625], [0.125, 0.0625]],
        [[0.125, 0], [0, -0.0625], [-0.125, 0.0625]],
        [[0.125, 0], [0, 0.1875], [-0.125, -0.1875]],
    ]
    for row, want in zip(multi.resid, expected_states, strict=True):
        for column, want_column in zip(row, want, strict=True):
            _close(column, want_column)


def test_residuals_survfit_data_frame_layout_matches_r():
    frame = r.survfit_residuals(_group_fit(), times=[2, 5], data_frame=True)

    assert list(frame) == ["(id)", "time", "resid", "curve"]
    assert frame["(id)"] == [1, 2, 3, 4, 5, 6, 7, 8] * 2
    assert frame["time"] == [2.0] * 8 + [5.0] * 8
    assert frame["curve"] == [1, 1, 1, 1, 2, 2, 2, 2] * 2
    _close(frame["resid"][:4], [-0.125, -0.125, 0.125, 0.125])


def test_residuals_survfit_argument_errors_match_r():
    fit = _group_fit()
    with pytest.raises(ValueError, match="the times argument is required"):
        r.survfit_residuals(fit)
    with pytest.raises(ValueError, match="times must be a numeric vector"):
        r.survfit_residuals(fit, times=["a"])
    with pytest.raises(ValueError, match="'type' should be one of"):
        r.survfit_residuals(fit, times=[2], type="bogus")
    with pytest.raises(ValueError, match="collapse must be TRUE/FALSE"):
        r.survfit_residuals(fit, times=[2], collapse="yes")
    with pytest.raises(TypeError, match="argument must be a survfit object"):
        r.survfit_residuals([1, 2], times=[2])
    interval = r.survfit("Surv(l, r, type = 'interval2') ~ 1", {"l": [1, 2, None], "r": [3, 4, 2]})
    with pytest.raises(ValueError, match="residuals for interval-censored data are not available"):
        r.survfit_residuals(interval, times=[2])


def test_residuals_survfit_multistate_reports_states_and_transitions():
    data = {
        "id": [1, 2, 3, 4, 5, 6],
        "time": [1, 2, 3, 4, 5, 6],
        "ev": r._r_factor(["a", "b", "censor", "a", "censor", "b"], ["censor", "a", "b"]),
    }
    fit = r.survfit("Surv(time, ev) ~ 1", data, id="id")
    result = r.survfit_residuals(fit, times=[2, 4])

    assert result.columns == ["(s0)", "a", "b"]
    assert result.column_name == "state"
    assert result.id == [1, 2, 3, 4, 5, 6]
    assert result.curve is None
    assert (len(result.resid), len(result.resid[0]), len(result.resid[0][0])) == (6, 3, 2)
    _close([sum(row[k][0] for row in result.resid) for k in range(3)], [0, 0, 0])
    cumhaz = r.survfit_residuals(fit, times=[2, 4], type="cumhaz")
    assert cumhaz.columns == ["1:2", "1:3"]
    assert cumhaz.column_name == "transition"
    frame = r.survfit_residuals(fit, times=[2], data_frame=True)
    assert list(frame) == ["id", "state", "time", "resid"]
    assert frame["state"] == ["(s0)"] * 6 + ["a"] * 6 + ["b"] * 6


def test_pseudo_matches_r_and_drops_single_time():
    fit = _group_fit()
    with pytest.warns(UserWarning, match="beyond the end of one or more curves"):
        values = r.pseudo(fit, times=[2, 5])

    _close([row[0] for row in values], [0, 0, 1, 1, 1, 1, 1, 1], rel=1e-6)
    _close([row[1] for row in values], [0, 0, 0, 0, 1, 1, 1, 1], rel=1e-6)
    with pytest.warns(UserWarning, match="beyond the end"):
        rmst = r.pseudo(fit, times=5, type="rmst")
    _close(rmst, [1, 2, 4, 4, 5, 5, 5, 5], rel=1e-6)


def test_pseudo_collapses_by_subject_and_returns_r_data_frame():
    data = _toy_data()
    data["id"] = [1, 1, 2, 2, 3, 3, 4, 4]
    fit = r.survfit("Surv(time, status) ~ 1", data, id="id")

    values = r.pseudo(fit, times=[2, 5])
    assert len(values) == 4
    frame = r.pseudo(fit, times=[2, 5], data_frame=True)
    assert list(frame) == ["id", "time", "resid", "pseudo"]
    assert frame["id"] == [1, 2, 3, 4] * 2
    assert len(frame["pseudo"]) == 8
    # collapse = FALSE keeps one row per observation
    assert len(r.pseudo(fit, times=[2], collapse=False)) == 8
    with pytest.raises(ValueError, match="the times argument is required"):
        r.pseudo(fit)
    with pytest.raises(ValueError, match="'type' should be one of"):
        r.pseudo(fit, times=[2], type="bogus")


def test_pseudo_multistate_has_subject_state_time_layout():
    data = {
        "id": [1, 2, 3, 4, 5, 6],
        "time": [1, 2, 3, 4, 5, 6],
        "ev": r._r_factor(["a", "b", "censor", "a", "censor", "b"], ["censor", "a", "b"]),
    }
    fit = r.survfit("Surv(time, ev) ~ 1", data)
    values = r.pseudo(fit, times=[2, 4])

    assert len(values) == 6
    assert len(values[0]) == 3
    assert len(values[0][0]) == 2
    single = r.pseudo(fit, times=[4])
    assert len(single[0]) == 3
    assert not isinstance(single[0][0], list)
    assert all(not math.isnan(v) for row in single for v in row)


# R 4.5.3, survival 3.8-12 for the tests below


def _mgus2_competing_risks(n=60):
    # R: m <- mgus2 with m$etime <- ifelse(pstat == 0, futime, ptime) and
    # m$event <- factor(ifelse(pstat == 0, 2 * death, 1), 0:2, c("censor", "pcm", "death"))
    mgus2 = survival.datasets.load_mgus2()
    labels = ["censor", "pcm", "death"]
    etime, event = [], []
    for row in range(n):
        progressed = mgus2["pstat"][row] != 0
        etime.append(mgus2["ptime"][row] if progressed else mgus2["futime"][row])
        event.append(labels[1] if progressed else labels[2 * int(mgus2["death"][row])])
    return {"etime": etime, "event": r._r_factor(event, labels)}


def test_residuals_and_pseudo_of_a_start_time_fit_match_r():
    # R: the residuals and pseudo values at times c(20, 40) of
    # fit <- survfit(Surv(time, status) ~ 1, aml, start.time = 10)
    fit = r.survfit("Surv(time, status) ~ 1", survival.datasets.load_aml(), start_time=10)

    result = r.survfit_residuals(fit, times=[20, 40])
    assert len(result.resid) == 23
    # the row ending at 9 is not part of the curve
    assert result.resid[0] == [0.0, 0.0]
    _close(result.resid[1], [-0.04585537918871252, -0.01959631589261219])
    _close(result.resid[3], [-0.05322499370118418, -0.02274572380392486])
    cumhaz = r.survfit_residuals(fit, times=[20, 40], type="cumhaz")
    _close(cumhaz.resid[1], [0.05227690204622154, 0.05227690204622154])
    values = r.pseudo(fit, times=[20, 40])
    _close(values[0], [0.825396825396825, 0.352733686067019])
    assert values[1][0] == pytest.approx(0.0, abs=1e-12)
    assert values[1][1] == pytest.approx(0.0, abs=1e-12)
    _close(values[3], [-0.132653061224490, -0.0566893424036282])
    _close(r.pseudo(fit, times=[20, 40], type="rmst")[0], [9.03968253968254, 20.30599647266314])

    # survfit(Surv(time, status) ~ x, aml, start.time = 10): rows 1, 2, 12, 13
    by_group = r.survfit("Surv(time, status) ~ x", survival.datasets.load_aml(), start_time=10)
    resid = r.survfit_residuals(by_group, times=[20, 40]).resid
    _close(resid[1], [-0.07875, -0.0405])
    assert resid[11] == [0.0, 0.0]
    group_values = r.pseudo(by_group, times=[20, 40])
    _close(group_values[0], [0.7875, 0.405])
    _close(group_values[12], [0.875, 0.291666666666667])


def test_multistate_residuals_and_pseudo_of_a_start_time_fit_match_r():
    # R: fit <- survfit(Surv(etime, event) ~ 1, data = m[1:60, ], start.time = 12), then
    # resid(fit, times = c(24, 60))[i, , ] and pseudo(fit, times = c(24, 60))[i, , ]
    fit = r.survfit("Surv(etime, event) ~ 1", _mgus2_competing_risks(), start_time=12)
    assert fit.n == [49]
    assert fit.start_time == 12.0

    result = r.survfit_residuals(fit, times=[24, 60])
    assert len(result.resid) == 60
    assert result.columns == ["(s0)", "pcm", "death"]
    first = result.resid[0]
    _close(first[0], [0.00444444444444444, -0.008602150537634409])
    _close(first[1], [0.0, -0.000277777777777778])
    _close(first[2], [-0.00444444444444444, 0.008879928315412186])
    # row 5 ends at 8, before the start, and still takes part as in R
    _close(result.resid[4][0], [-0.0122222222222222, -0.008602150537634403])
    values = r.pseudo(fit, times=[24, 60])
    _close(values[0][0], [1.115736961451247, 0.21048935703313554])
    _close(values[0][2], [-0.115736961451247, 0.78271359081266911])
    _close(values[4][2], [0.700929705215419, 0.78271359081266878])
