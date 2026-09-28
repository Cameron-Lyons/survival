"""survfit0, summary.survfit and quantile.survfit on Turnbull and survfit.coxph curves.

R 4.5.3 with survival 3.8-12.  R's survfitTurnbull and survfit.coxph objects inherit from
``survfit``, so the three methods take them as they take a Kaplan-Meier fit; the Turnbull
fit also follows R on ``robust`` and ``start.time``.
"""

from __future__ import annotations

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api


def _close(actual, expected, rel=1e-8):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, float) and math.isnan(e):
            assert math.isnan(a), (actual, expected)
        else:
            assert a == pytest.approx(e, rel=rel, abs=1e-12), (actual, expected)


def _close_rows(actual, expected, rel=1e-8):
    assert len(actual) == len(expected), (actual, expected)
    for row, expected_row in zip(actual, expected, strict=True):
        _close(row, expected_row, rel)


NA = math.nan
INF = math.inf


# ---------------------------------------------------------------------------
# Turnbull curves: d <- data.frame(l = c(1,2,3,4,5,6,2,7), r = c(1,NA,3,6,5,8,4,NA))
# ---------------------------------------------------------------------------


def _interval_data():
    return {"l": [1, 2, 3, 4, 5, 6, 2, 7], "r": [1, None, 3, 6, 5, 8, 4, None]}


def _turnbull(data=None, formula="Surv(l, r, type = 'interval2') ~ 1", **kwargs):
    return r.survfit(formula, _interval_data() if data is None else data, **kwargs)


def test_turnbull_quantiles_match_r():
    quantiles = r.quantile_survfit(_turnbull())

    assert quantiles.quantile == [[3.0, 5.0, 7.5]]
    assert quantiles.lower == [[1.0, 3.0, 5.0]]
    assert quantiles.upper == [[7.5, 7.5, 7.5]]


def test_turnbull_survfit0_fills_in_what_r_derives():
    fit0 = r.survfit0(_turnbull())

    _close(fit0.time, [0, 1, 2, 3, 3.5, 5, 5.5, 7, 7.5])
    se = [
        0,
        0.116925805875275,
        0.116925805875275,
        0.185546918046783,
        0.185546918544061,
        0.172840380546378,
        0.172840379860608,
        0.172840379860608,
        0,
    ]
    _close(fit0.std_err, se)
    # R fills in cumhaz = -log(surv) and std.chaz = std.err (logse is taken as TRUE)
    _close(fit0.std_chaz, se)
    _close(
        fit0.cumhaz,
        [
            0,
            0.133531392624523,
            0.133531392624523,
            0.538937372868664,
            0.538996500732687,
            1.232025429060484,
            1.232143681292632,
            1.232143681292632,
            INF,
        ],
    )
    _close(fit0.lower[:4], [1, 0.6733834259428694, 0.6733834259428694, 0.3127576401728233])


def test_turnbull_summary_matches_r():
    fit = _turnbull()
    summary = r.summary_survfit(fit)

    assert summary.table.colnames[4:] == ["rmean", "se(rmean)", "median", "0.95LCL", "0.95UCL"]
    _close(
        summary.table.values[0],
        [8, 8, 8, 6, 4.645867825607064, 0.818407326871296, 5, 3, 7.5],
    )
    _close(summary.time, [1, 3, 3.5, 5, 5.5, 7.5])
    # std.err is se(S) from the robust variance, read as se(log S): 0.1023 = 0.1169 * 0.875
    _close(
        summary.std_err,
        [
            0.1023100801408659,
            0.1082421021290439,
            0.1082357024840356,
            0.0504177393170779,
            0.0504117774593440,
            0,
        ],
    )
    at = r.summary_survfit(fit, times=[2, 4, 6])
    _close(at.surv, [0.875, 0.583333333333333, 0.291666666666667])
    _close(at.std_err, [0.102310080140866, 0.108235702484036, 0.050411777459344])
    _close(at.n_risk, [7, 4, 2])
    _close(at.n_event, [1, 2, 2])
    _close(at.n_censor, [1, 0, 0])


def test_turnbull_methods_by_stratum_match_r():
    data = {
        "l": [1, 2, 3, 4, 5, 6, 2, 7, 1, 3, 2, 5],
        "r": [1, None, 3, 6, 5, 8, 4, None, 2, None, 6, 7],
        "g": ["a"] * 8 + ["b"] * 4,
    }
    fit = _turnbull(data, "Surv(l, r, type = 'interval2') ~ g")

    assert fit.strata == {"g=a": 8, "g=b": 3}
    quantiles = r.quantile_survfit(fit)
    assert quantiles.strata == ["g=a", "g=b"]
    _close_rows(quantiles.quantile, [[3, 5, 7.5], [3.5, 5.5, 5.5]])
    _close_rows(quantiles.lower, [[1, 3, 5], [1.5, 1.5, NA]])
    _close_rows(quantiles.upper, [[7.5, 7.5, 7.5], [NA, NA, NA]])
    table = r.summary_survfit(fit).table
    assert table.rownames == ["g=a", "g=b"]
    _close(table.values[1], [4, 4, 4, 3, 4.5, 0.866025403784439, 5.5, 1.5, NA])
    _close(r.survfit0(fit).time, [0, 1, 2, 3, 3.5, 5, 5.5, 7, 7.5, 0, 1.5, 3, 5.5])
    at = r.summary_survfit(fit, times=[2, 5])
    assert at.strata == ["g=a", "g=a", "g=b", "g=b"]
    _close(at.surv, [0.875, 0.291701158940397, 0.75, 0.75])
    _close(
        at.std_err,
        [0.1023100801408659, 0.0504177393170779, 0.2165063509461096, 0.2165063509461096],
    )


def test_turnbull_robust_false_gives_the_greenwood_error_of_log_surv():
    fit = _turnbull(robust=False)

    _close(
        fit.std_err,
        [
            0.133630620956212,
            0.133630620956212,
            0.318081270529207,
            0.318104505140176,
            0.592563375150846,
            0.592613260221602,
            0.592613260221602,
            INF,
        ],
    )
    _close(fit.lower[:3], [0.6733819365059553, 0.6733819365059553, 0.3127455972529785])
    assert math.isnan(fit.lower[7])
    assert math.isnan(fit.upper[7])
    _close(
        r.summary_survfit(fit).std_err,
        [0.116926793336686, 0.185558379154956, 0.185560961331769, 0.172851423277135]
        + [0.172845534231301, NA],
    )
    with pytest.raises(ValueError, match="robust must be TRUE/FALSE"):
        _turnbull(robust="yes")


def test_turnbull_default_variance_follows_survfit_km_rule():
    # l = c(1,2,3,5), r = c(NA,NA,3,NA): integer pseudo-weights, so not robust by default
    data = {"l": [1, 2, 3, 5], "r": [None, None, 3, None]}

    _close(_turnbull(data).std_err, [0, 0, 0.816496580927726, 0.816496580927726])
    _close(_turnbull(data, robust=True).std_err, [0, 0, 0.272165526975909, 0.272165526975909])


def test_turnbull_start_time_keeps_intervals_ending_after_it():
    # every row interval censored: R keeps y[, 2] >= start.time, so (2,4] stays and (1,2] goes
    data = {"l": [1, 4, 6, 2, 0], "r": [2, 6, 8, 4, 3]}
    fit = _turnbull(data, start_time=3)

    assert fit.n == [4]
    _close(fit.time, [2.5, 5, 7])
    _close(fit.surv, [0.5, 0.25, 0])
    _close(fit.std_err, [0.5, 0.866025403784439, INF])
    _close(fit.n_risk, [4, 2, 1])
    # R compares its placeholder 1 for exact and censored rows; here their times count
    assert _turnbull(start_time=3).n == [6]


def test_turnbull_refuses_a_cluster():
    data = {**_interval_data(), "cl": [1, 1, 2, 2, 3, 3, 4, 4]}
    with pytest.raises(ValueError, match="cluster is not supported for interval-censored data"):
        _turnbull(data, cluster="cl")


# ---------------------------------------------------------------------------
# survfit.coxph curves on lung
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


def _cox_curves(lung, formula, **kwargs):
    return r.survfit(r.coxph(formula, lung), **kwargs)


def test_cox_curve_summary_quantile_and_survfit0_match_r(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age")

    at = r.summary_survfit(curves, times=[100, 300])
    _close(at.surv, [0.864906320408970, 0.533365021945315])
    _close(at.std_err, [0.0225985909793118, 0.0346462093877946])
    _close(at.lower, [0.821728903778967, 0.469604625636010])
    _close(at.upper, [0.910352477128639, 0.605782462746066])
    _close(at.n_risk, [196, 92])
    _close(at.n_event, [31, 70])
    assert at.table.rownames is None
    _close(
        at.table.values[0],
        [228, 228, 228, 165, 379.4407945651065, 20.1489545632073, 310, 285, 363],
    )
    quantiles = r.quantile_survfit(curves)
    assert quantiles.strata is None
    assert quantiles.quantile == [[175, 310, 558]]
    assert quantiles.lower == [[145, 285, 460]]
    assert quantiles.upper == [[197, 363, 654]]

    fit0 = r.survfit0(curves)
    assert isinstance(fit0, r.CoxSurvfitResult)
    _close(fit0.time[:3], [0, 5, 11])
    _close(fit0.surv[:3], [1, 0.995684569684275, 0.982721473037965])
    _close(fit0.std_err[:3], [0, 0.00432517163249581, 0.00871795972358942])
    _close(fit0.std_chaz[:3], [0, 0.00432517163249581, 0.00871795972358942])
    _close(fit0.cumhaz[:3], [0, 0.0043247686607984, 0.0174295427911605])
    _close(fit0.lower[:3], [1, 0.987279647096595, 0.966072467112922])
    _close(fit0.n_risk[:3], [228, 228, 227])
    assert len(r.survfit0(fit0).time) == len(fit0.time)


def test_cox_curves_for_newdata_rows_are_labelled_by_row(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age", newdata={"age": [50, 70]})

    assert curves.colnames == ["1", "2"]
    quantiles = r.quantile_survfit(curves)
    assert quantiles.strata == ["1", "2"]
    assert quantiles.quantile == [[194, 364, 654], [163, 288, 477]]
    assert quantiles.lower == [[166, 306, 533], [132, 239, 428]]
    assert quantiles.upper == [[269, 524, 883], [182, 350, 624]]
    at = r.summary_survfit(curves, times=[100, 300])
    _close_rows(
        at.surv,
        [[0.891395980359339, 0.846051052832331], [0.607806270135450, 0.484805765273381]],
    )
    _close_rows(
        at.std_err,
        [[0.022363490121411, 0.0268229146008075], [0.048654878537338, 0.0419143495310236]],
    )
    _close(at.n_risk, [196, 92])
    assert at.table.rownames == ["1", "2"]
    _close_rows(
        [row[4:7] for row in at.table.values],
        [[440.941992691214, 27.5969232301886, 364], [344.466623563749, 16.4580571028554, 288]],
    )
    _close_rows(r.survfit0(curves).surv[:2], [[1, 1], [0.99658003861439, 0.995030816627018]])


def test_cox_curves_of_two_covariates_match_r(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age + sex")

    _close(r.survfit0(curves).time[:3], [0, 5, 11])
    _close(r.summary_survfit(curves, times=100).surv, [0.867920595821706])
    quantiles = r.quantile_survfit(curves)
    assert quantiles.quantile == [[176, 320, 558]]
    assert quantiles.lower == [[147, 285, 460]]
    assert quantiles.upper == [[199, 363, 654]]


def test_stratified_cox_curves_match_r(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age + strata(sex)")

    quantiles = r.quantile_survfit(curves)
    assert quantiles.strata == ["sex=1", "sex=2"]
    assert quantiles.quantile == [[147, 283, 460], [226, 426, 705]]
    assert quantiles.lower == [[110, 218, 387], [186, 348, 550]]
    _close_rows(quantiles.upper, [[179, 320, 583], [340, 550, NA]])
    at = r.summary_survfit(curves, times=[100, 300])
    assert at.strata == ["sex=1", "sex=1", "sex=2", "sex=2"]
    _close(at.surv, [0.828884075886690, 0.448587511916771, 0.921569379106663, 0.672954127568657])
    assert at.table.rownames == ["sex=1", "sex=2"]
    _close(
        at.table.values[0],
        [138, 138, 138, 112, 332.060477281240, 23.9431132739084, 283, 218, 320],
    )
    fit0 = r.survfit0(curves)
    assert fit0.strata == {"sex=1": 120, "sex=2": 88}
    _close([fit0.time[k] for k in (0, 1, 120, 121)], [0, 11, 0, 5])


def test_stratified_cox_curves_for_newdata_rows_are_labelled_stratum_then_row(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age + strata(sex)", newdata={"age": [50, 70]})

    # R: quantile() gives a stratum x curve x probability array, summary() a table whose
    # rows run over the strata first
    labels = ["sex=1, 1", "sex=2, 1", "sex=1, 2", "sex=2, 2"]
    quantiles = r.quantile_survfit(curves)
    assert quantiles.strata == labels
    assert quantiles.quantile == [
        [166, 306, 567],
        [268, 473, 731],
        [132, 239, 429],
        [201, 363, 654],
    ]
    assert quantiles.lower == [[132, 239, 442], [201, 361, 654], [93, 189, 353], [167, 310, 520]]
    _close_rows(
        quantiles.upper, [[218, 442, 814], [361, 728, NA], [175, 303, 558], [310, 524, 765]]
    )
    at = r.summary_survfit(curves, times=[100, 300])
    _close_rows(
        at.surv,
        [
            [0.857808518515627, 0.808863729105679],
            [0.519370955235684, 0.404101679374167],
            [0.935429588904698, 0.911815210656278],
            [0.723475554201420, 0.639110297399010],
        ],
    )
    assert at.table.rownames == labels
    _close(
        [row[4] for row in at.table.values],
        [384.053301058900, 517.248987590434, 302.862956801733, 434.386905284629],
    )


def test_cox_curves_from_a_start_time_match_r(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age", start_time=100)

    quantiles = r.quantile_survfit(curves)
    assert quantiles.quantile == [[218, 361, 613]]
    assert quantiles.lower == [[194, 320, 524]]
    assert quantiles.upper == [[269, 433, 705]]
    # a probability of 0 reports start.time; the curve itself starts at min(0, time)
    assert r.quantile_survfit(curves, probs=[0, 0.5]).quantile == [[100, 361]]
    _close(r.survfit0(curves).time[:3], [0, 105, 107])
    _close(r.summary_survfit(curves, times=[50, 200]).surv, [1, 0.788389886647259])
    _close(
        r.summary_survfit(curves).table.values[0],
        [196, 196, 196, 134, 430.459457196032, 21.057231250079, 361, 320, 433],
    )
    with pytest.raises(ValueError, match="Truncation point"):
        r.summary_survfit(curves, rmean=99)


def test_aggregated_cox_curves_are_labelled_by_group(lung):
    curves = _cox_curves(lung, "Surv(time, status) ~ age", newdata={"age": [50, 60, 70]})
    grouped = r.aggregate_survfit(curves, by=["x", "y", "x"])

    assert grouped.colnames == ["1", "2"]
    table = r.summary_survfit(grouped).table
    assert table.rownames == ["1", "2"]
    assert table.colnames == [
        "records",
        "n.max",
        "n.start",
        "events",
        "rmean",
        "se(rmean)",
        "median",
    ]
    _close_rows(
        [row[4:7] for row in table.values],
        [[392.704308127482, 21.9211646742765, 337], [391.193981282588, 21.4802700794743, 337]],
    )
    quantiles = r.quantile_survfit(grouped, conf_int=False)
    assert quantiles.quantile == [[176, 337, 574], [177, 337, 574]]
    assert quantiles.lower is None
