import math
from math import exp, log

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_compute_baseline_survival_steps():
    ndeath = [1, 1, 0, 1, 0]
    risk = [1.0, 1.0, 1.0, 1.0, 1.0]
    wt = [1.0, 1.0, 1.0, 1.0]
    sn = 5
    denom = [5.0, 4.0, 3.0, 2.0, 1.0]

    result = survival.surv_analysis.compute_baseline_survival_steps(ndeath, risk, wt, sn, denom)
    assert isinstance(result, list)
    assert len(result) == sn


def test_compute_baseline_survival_steps_validates_inputs():
    with pytest.raises(ValueError, match="ndeath length must be 2"):
        survival.surv_analysis.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[1.0],
            wt=[1.0],
            sn=2,
            denom=[2.0, 1.0],
        )

    with pytest.raises(ValueError, match="risk length must be at least 2"):
        survival.surv_analysis.compute_baseline_survival_steps(
            ndeath=[2],
            risk=[1.0],
            wt=[1.0, 1.0],
            sn=1,
            denom=[2.0],
        )

    with pytest.raises(ValueError, match="risk must be positive"):
        survival.surv_analysis.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[0.0],
            wt=[1.0],
            sn=1,
            denom=[2.0],
        )

    with pytest.raises(ValueError, match="denom contains non-finite"):
        survival.surv_analysis.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[1.0],
            wt=[1.0],
            sn=1,
            denom=[float("inf")],
        )

    with pytest.raises(ValueError, match="death contribution must not exceed denom"):
        survival.surv_analysis.compute_baseline_survival_steps(
            ndeath=[1],
            risk=[2.0],
            wt=[1.0],
            sn=1,
            denom=[1.0],
        )


def test_agsurv4_alias_matches_validated_baseline_steps():
    ndeath = [1, 2, 0]
    risk = [1.0, 1.0, 1.0]
    wt = [0.2, 0.3, 0.4]
    denom = [5.0, 4.0, 3.0]

    direct = survival.surv_analysis.compute_baseline_survival_steps(ndeath, risk, wt, 3, denom)
    alias = survival.surv_analysis.agsurv4(ndeath, risk, wt, 3, denom)

    assert alias == pytest.approx(direct)

    with pytest.raises(ValueError, match="risk length must be at least 1"):
        survival.surv_analysis.agsurv4([1], [], [1.0], 1, [2.0])


def test_compute_tied_baseline_summaries():
    n = 5
    nvar = 2
    dd = [1, 1, 2, 1, 1]
    x1 = [10.0, 9.0, 8.0, 7.0, 6.0]
    x2 = [5.0, 4.0, 3.0, 2.0, 1.0]
    xsum = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    xsum2 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.5, 2.0, 1.5, 1.0, 0.5]

    result = survival.surv_analysis.compute_tied_baseline_summaries(
        n, nvar, dd, x1, x2, xsum, xsum2
    )
    assert isinstance(result, dict)
    assert "sum1" in result
    assert "sum2" in result
    assert "xbar" in result


def test_compute_tied_baseline_summaries_validates_inputs():
    with pytest.raises(ValueError, match="dd length must be 2"):
        survival.surv_analysis.compute_tied_baseline_summaries(
            2,
            1,
            [1],
            [10.0, 9.0],
            [5.0, 4.0],
            [10.0, 9.0],
            [5.0, 4.0],
        )

    with pytest.raises(ValueError, match="xsum length must be 2"):
        survival.surv_analysis.compute_tied_baseline_summaries(
            2,
            1,
            [1, 1],
            [10.0, 9.0],
            [5.0, 4.0],
            [10.0],
            [5.0, 4.0],
        )

    with pytest.raises(ValueError, match="positive event counts"):
        survival.surv_analysis.compute_tied_baseline_summaries(
            1,
            1,
            [0],
            [10.0],
            [5.0],
            [10.0],
            [5.0],
        )

    with pytest.raises(ValueError, match="x1 contains non-finite"):
        survival.surv_analysis.compute_tied_baseline_summaries(
            1,
            1,
            [1],
            [float("nan")],
            [5.0],
            [10.0],
            [5.0],
        )

    with pytest.raises(ValueError, match="tied denominator must be positive"):
        survival.surv_analysis.compute_tied_baseline_summaries(
            1,
            1,
            [2],
            [1.0],
            [3.0],
            [1.0],
            [0.5],
        )


def test_agsurv5_alias_matches_validated_tied_baseline_summaries():
    args = (
        2,
        1,
        [1, 2],
        [10.0, 9.0],
        [0.0, 1.0],
        [10.0, 9.0],
        [0.0, 0.5],
    )

    direct = survival.surv_analysis.compute_tied_baseline_summaries(*args)
    alias = survival.surv_analysis.agsurv5(*args)

    assert alias == direct

    with pytest.raises(ValueError, match="positive event counts"):
        survival.surv_analysis.agsurv5(1, 1, [0], [1.0], [0.0], [1.0], [0.0])


def test_cox_survfit_baseline_handles_ties_weights_and_delayed_entry():
    result = survival.surv_analysis.cox_survfit_baseline(
        y=[
            [0.0, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [2.0, 4.0, 0.0],
        ],
        x=[[0.0], [1.0], [2.0]],
        weights=[1.0, 1.0, 1.0],
        risk=[1.0, 2.0, 4.0],
        survtype=3,
        vartype=3,
    )

    assert result["time"] == pytest.approx([2.0, 3.0, 4.0])
    assert result["n_risk"] == pytest.approx([2.0, 2.0, 1.0])
    assert result["n_event"] == pytest.approx([1.0, 1.0, 0.0])
    assert result["n_censor"] == pytest.approx([0.0, 0.0, 1.0])
    assert result["hazard"] == pytest.approx([1.0 / 3.0, 1.0 / 6.0, 0.0])
    assert result["cumhaz"] == pytest.approx([1.0 / 3.0, 0.5, 0.5])

    with pytest.raises(ValueError, match="start must be less than stop"):
        survival.surv_analysis.cox_survfit_baseline(
            [[1.0, 1.0, 0.0]],
            [[0.0]],
            [1.0],
            [1.0],
            2,
            2,
        )


# ---------------------------------------------------------------------------------------------
# Kaplan-Meier family (survfitkm and its summaries), against R survival 3.8.11
# ---------------------------------------------------------------------------------------------

_KM_TIME = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_KM_STATUS = [1, 1, 0, 1, 0, 1, 1, 0]


def test_survfitkm_matches_r_survfit():
    fit = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS)

    # survfit(Surv(time, status) ~ 1)
    assert isinstance(fit, survival.surv_analysis.SurvfitKMResult)
    assert fit.n == [8]
    assert fit.time == pytest.approx(_KM_TIME)
    assert fit.n_risk == pytest.approx([8, 7, 6, 5, 4, 3, 2, 1])
    assert fit.n_event == pytest.approx([1, 1, 0, 1, 0, 1, 1, 0])
    assert fit.n_censor == pytest.approx([0, 0, 1, 0, 1, 0, 0, 1])
    assert fit.surv == pytest.approx([0.875, 0.75, 0.75, 0.6, 0.6, 0.4, 0.2, 0.2])
    assert fit.std_err[:2] == pytest.approx([0.13363062095621220, 0.20412414523193151])
    assert fit.lower[:2] == pytest.approx([0.673381936505955347, 0.502701841294046936])
    assert fit.upper == pytest.approx([1.0] * 8)
    assert fit.cumhaz[:2] == pytest.approx([0.125, 0.26785714285714285])
    assert fit.std_chaz[:2] == pytest.approx([0.125, 0.18982403237026158])
    assert fit.cumhaz[-1] == pytest.approx(1.30119047619047623)
    assert fit.logse is True
    assert (fit.conf_type, fit.conf_int, fit.type) == ("log", 0.95, "right")
    assert fit.strata is None
    assert fit.influence_surv is None


def test_survfitkm_confidence_intervals_match_r():
    plain = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, conf_type="plain", conf_int=0.9)
    assert plain.lower[:2] == pytest.approx([0.682672539892347441, 0.498184244525166409])
    assert plain.lower[-2:] == pytest.approx([0.0, 0.0])
    assert plain.upper[3:5] == pytest.approx([0.89880249996469019, 0.89880249996469019])

    loglog = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, conf_type="log-log")
    assert loglog.lower[:2] == pytest.approx([0.3870000140320252191, 0.3148071137055191149])
    assert loglog.upper[:2] == pytest.approx([0.98139296587896596, 0.93089831709062754])

    none = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, conf_type="none")
    assert none.lower is None
    assert none.upper is None

    bands = survival.surv_analysis.survfit_confint(
        [0.875, 0.75], [0.1336306209562122, 0.2041241452319315]
    )
    assert isinstance(bands, survival.surv_analysis.ConfidenceBands)
    assert bands.lower == pytest.approx([0.673381936505955347, 0.502701841294046936])
    assert bands.upper == pytest.approx([1.0, 1.0])

    with pytest.raises(ValueError, match="conf.type must be one of"):
        survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, conf_type="weird")


def test_survfitkm_weights_and_hazard_types_match_r():
    weighted = survival.surv_analysis.survfitkm(
        _KM_TIME, _KM_STATUS, weights=[1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
    )
    # R: survfit with case weights w
    assert weighted.n_risk == pytest.approx([10, 9, 7, 6, 5, 3, 2, 1])
    assert weighted.surv[:2] == pytest.approx([0.9, 0.7])
    assert weighted.std_err[:2] == pytest.approx([0.10540925533894598, 0.20701966780270625])

    fleming = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, stype=2, ctype=2)
    # R: survfit with stype = 2, ctype = 2 (Fleming-Harrington)
    assert fleming.surv[:2] == pytest.approx([0.88249690258459546, 0.76501706144857473])
    assert fleming.cumhaz[:2] == pytest.approx([0.125, 0.26785714285714285])

    nelson = survival.surv_analysis.nelson_aalen(_KM_TIME, _KM_STATUS)
    assert isinstance(nelson, survival.surv_analysis.NelsonAalenResult)
    assert nelson.time == pytest.approx([1.0, 2.0, 4.0, 6.0, 7.0])
    assert nelson.cumulative_hazard == pytest.approx(
        [0.125, 0.26785714285714285, 0.46785714285714286, 0.80119047619047623, 1.30119047619047623]
    )
    assert nelson.variance[:2] == pytest.approx([0.015625, 0.18982403237026158**2])
    assert nelson.n_risk == [8, 7, 5, 3, 2]
    assert nelson.survival()[:2] == pytest.approx([0.88249690258459546, 0.76501706144857473])


def test_survfitkm_cluster_robust_variance_and_influence_match_r():
    fit = survival.surv_analysis.survfitkm(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 0, 1, 0, 1], cluster=[0, 0, 1, 1, 2, 2], influence=3
    )

    # survfit(Surv(time, status) ~ 1, cluster = id, influence = TRUE)
    assert fit.surv == pytest.approx([5 / 6, 2 / 3, 2 / 3, 4 / 9, 4 / 9, 0.0])
    assert fit.std_err == pytest.approx(
        [0.1360828, 0.2721655, 0.2721655, 0.2771598, 0.2771598, 0.0], abs=1e-7
    )
    assert fit.std_chaz == pytest.approx(
        [0.1360828, 0.3320419, 0.3320419, 0.4571841, 0.4571841, 0.4571841], abs=1e-7
    )
    (influence,) = fit.influence_surv
    assert isinstance(influence, survival.surv_analysis.SurvfitInfluence)
    (chaz_influence,) = fit.influence_chaz
    surv_rows = np.asarray(influence.values).reshape(-1, 6)
    chaz_rows = np.asarray(chaz_influence.values).reshape(-1, 6)
    assert surv_rows[0] == pytest.approx([-1 / 9, -2 / 9, -2 / 9, -4 / 27, -4 / 27, 0.0])
    assert chaz_rows[2] == pytest.approx(
        [-1 / 18, -0.13555555555555554, -0.13555555555555554, -0.35777777777777775]
        + [-0.35777777777777775] * 2
    )


def test_survfitkm_delayed_entry_reverse_and_near_ties_match_r():
    delayed = survival.surv_analysis.survfitkm(
        [2.0, 4.0, 3.0, 5.0, 5.0], [1, 0, 1, 1, 0], start=[0.0, 0.0, 1.0, 2.0, 3.0]
    )
    # survfit(Surv(start, stop, status) ~ 1)
    assert delayed.time == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert delayed.n_risk == pytest.approx([3.0, 3.0, 3.0, 2.0])
    assert delayed.n_event == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert delayed.surv == pytest.approx([2 / 3, 4 / 9, 4 / 9, 2 / 9])
    assert delayed.cumhaz == pytest.approx([1 / 3, 2 / 3, 2 / 3, 7 / 6])
    assert delayed.std_chaz == pytest.approx(
        [0.33333333333333331, 0.47140452079103168, 0.47140452079103168, 0.68718427093627676]
    )

    # reverse = TRUE follows survfitkm.c: tied deaths leave the risk set before the censorings
    reverse = survival.surv_analysis.survfitkm(
        [1.0, 2.0, 2.0, 3.0, 4.0], [1, 0, 1, 0, 1], reverse=True
    )
    assert reverse.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert reverse.n_risk == pytest.approx([5.0, 4.0, 2.0, 1.0])
    assert reverse.surv == pytest.approx([1.0, 2 / 3, 1 / 3, 1 / 3])

    near = survival.surv_analysis.survfitkm([1.0 + 5e-10, 2.0, 1.0], [1, 0, 1])
    # aeqSurv folds the near tie
    assert near.time == pytest.approx([1.0, 2.0])
    assert near.n_risk == pytest.approx([3.0, 1.0])
    assert near.n_event == pytest.approx([2.0, 0.0])
    assert near.surv == pytest.approx([1 / 3, 1 / 3])

    with pytest.raises(ValueError, match="status"):
        survival.surv_analysis.survfitkm([1.0, 2.0, 3.0], [1, 2, 0])


def test_survfitkm_strata_stack_curves_like_r():
    fit = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS, strata=[1, 1, 1, 1, 2, 2, 2, 2])

    # survfit(Surv(time, status) ~ g)
    assert fit.strata == [4, 4]
    assert fit.strata_codes == [1, 2]
    assert fit.n == [4, 4]
    assert fit.time == pytest.approx(_KM_TIME)
    assert fit.surv == pytest.approx([0.75, 0.5, 0.5, 0.0, 1.0, 2 / 3, 1 / 3, 1 / 3])


def test_survfit_summaries_match_r():
    fit = survival.surv_analysis.survfitkm(_KM_TIME, _KM_STATUS)

    at_times = survival.surv_analysis.summary_survfit(fit, times=[0.0, 2.5, 5.0, 10.0])
    # summary(fit, times = c(0, 2.5, 5, 10)) drops the time beyond the last follow-up
    assert at_times.time == pytest.approx([0.0, 2.5, 5.0])
    assert at_times.surv == pytest.approx([1.0, 0.75, 0.6])
    assert at_times.n_risk == pytest.approx([8.0, 6.0, 4.0])
    assert at_times.n_event == pytest.approx([0.0, 2.0, 1.0])
    assert at_times.std_err == pytest.approx([0.0, 0.15309310892394862, 0.18165902124584954])
    assert at_times.lower == pytest.approx([1.0, 0.50270184129404694, 0.33146462434329693])

    extended = survival.surv_analysis.summary_survfit(fit, times=[0.0, 2.5, 5.0, 10.0], extend=True)
    assert extended.time == pytest.approx([0.0, 2.5, 5.0, 10.0])
    assert extended.surv == pytest.approx([1.0, 0.75, 0.6, 0.2])

    table = survival.surv_analysis.survmean(fit)
    # summary(fit)$table
    assert isinstance(table, survival.surv_analysis.SurvmeanTable)
    assert (table.records, table.n_max, table.n_start, table.events) == ([8.0], [8.0], [8.0], [5.0])
    assert table.rmean == pytest.approx([5.175])
    assert table.se_rmean == pytest.approx([0.90141382006268356])
    assert table.median == pytest.approx([6.0])
    assert table.lower == pytest.approx([4.0])
    assert math.isnan(table.upper[0])
    assert table.end_time == pytest.approx([8.0])

    restricted = survival.surv_analysis.survmean(fit, rmean="6")
    # summary(fit, rmean = 6)$table
    assert restricted.rmean == pytest.approx([4.575])
    assert restricted.se_rmean == pytest.approx([0.68832904558793684])

    quantiles = survival.surv_analysis.quantile_survfit(fit, probs=[0.25, 0.5, 0.75])
    # R: quantile of the survfit object at 25/50/75%
    assert isinstance(quantiles, survival.surv_analysis.SurvfitQuantiles)
    assert quantiles.probs == pytest.approx([0.25, 0.5, 0.75])
    assert quantiles.quantile[0] == pytest.approx([3.0, 6.0, 7.0])
    assert quantiles.lower[0] == pytest.approx([1.0, 4.0, 6.0])
    assert all(math.isnan(value) for value in quantiles.upper[0])

    with_origin = survival.surv_analysis.survfit0(fit)
    # survfit0(fit) prepends the time-0 row
    assert with_origin.time == pytest.approx([0.0, *_KM_TIME])
    assert with_origin.surv[:2] == pytest.approx([1.0, 0.875])
    assert with_origin.n_risk[:2] == pytest.approx([8.0, 8.0])


def test_survmean_and_quantile_curve_kernels_match_r():
    fit = survival.surv_analysis.survfitkm([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 1])

    rows = survival.validation.survmean_curves(
        fit.time,
        fit.surv,
        fit.n_risk,
        fit.n_event,
        fit.n,
        lower=fit.lower,
        upper=fit.upper,
        rmean="numeric",
        rmean_at=4.0,
    )
    # summary(survfit(Surv(1:4, c(1,1,0,1)) ~ 1), rmean = 4)$table
    (row,) = rows
    assert isinstance(row, survival.validation.SurvfitSummaryRow)
    assert (row.records, row.n_max, row.events) == (4.0, 4.0, 3.0)
    assert row.rmean == pytest.approx(2.75)
    assert row.se_rmean == pytest.approx(0.649519052838329)
    assert row.median == pytest.approx(3.0)
    assert row.lower == pytest.approx(1.0)
    assert math.isnan(row.upper)

    quantiles = survival.validation.quantile_survfit_curves(
        fit.time, fit.surv, lower=fit.lower, upper=fit.upper, probs=[0.5]
    )
    assert isinstance(quantiles, survival.validation.SurvfitCurveQuantiles)
    assert quantiles.quantile[0] == pytest.approx([3.0])
    assert quantiles.lower[0] == pytest.approx([1.0])


def test_pseudo_values_and_survfit_residuals_match_r():
    values = survival.surv_analysis.pseudo(_KM_TIME, _KM_STATUS, [2.0, 5.0])

    # pseudo(survfit(Surv(time, status) ~ 1), times = c(2, 5))
    assert isinstance(values, survival.surv_analysis.SurvfitResid)
    assert values.id == list(range(8))
    assert values.curve == [0] * 8
    assert values.times == pytest.approx([2.0, 5.0])
    np.testing.assert_allclose(
        values.values,
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.8],
            [1.0, -0.16],
            [1.0, 1.04],
            [1.0, 1.04],
            [1.0, 1.04],
            [1.0, 1.04],
        ],
    )

    cumhaz = survival.surv_analysis.pseudo(_KM_TIME, _KM_STATUS, [2.0, 5.0], type_="cumhaz")
    np.testing.assert_allclose(
        cumhaz.values[:2],
        [[1.14285714285714279, 1.34285714285714297], [1.12244897959183665, 1.32244897959183660]],
    )

    rmst = survival.surv_analysis.pseudo(_KM_TIME, _KM_STATUS, [5.0], type_="rmst")
    assert [row[0] for row in rmst.values] == pytest.approx(
        [1.0, 2.0, 4.8, 3.84, 5.04, 5.04, 5.04, 5.04]
    )

    residuals = survival.surv_analysis.survfitresid(_KM_TIME, _KM_STATUS, [2.0, 5.0])
    # R: residuals of the survfit object at times 2 and 5
    np.testing.assert_allclose(
        residuals.values[:3], [[-0.09375, -0.075], [-0.09375, -0.075], [0.03125, 0.025]]
    )

    with pytest.raises(ValueError, match="type"):
        survival.surv_analysis.pseudo(_KM_TIME, _KM_STATUS, [2.0], type_="weird")

    gee = survival.surv_analysis.pseudo_gee_regression(
        pseudo_values=[[0.8], [0.7], [0.6]],
        covariates=[[1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
        cluster_id=None,
        config=survival.surv_analysis.GEEConfig(),
    )
    assert len(gee.coefficients) == len(gee.std_errors) == 2
    assert len(gee.confidence_intervals) == 2

    with pytest.raises(ValueError, match="correlation_structure"):
        survival.surv_analysis.GEEConfig(correlation_structure="weird")
    with pytest.raises(ValueError, match="pseudo_values row 1"):
        survival.surv_analysis.pseudo_gee_regression(
            [[0.8], [0.7, 0.6]], [[1.0], [1.0]], None, None
        )


def test_survfit_matrix_public_apis_validate_shapes_and_values():
    result = survival.surv_analysis.survfit_from_hazard(
        [1.0, 2.0],
        [0.1, 0.2],
        n_risk=[10.0, 8.0],
        n_event=[1.0, 2.0],
    )

    assert result.n_states == 1
    assert result.get_cumhaz_at_state(0) == pytest.approx([0.1, 0.3])
    assert result.get_surv_at_state(0) == pytest.approx([exp(-0.1), exp(-0.3)])

    multistate = survival.surv_analysis.survfit_multistate(
        [1.0],
        [[[0.0, 0.25], [0.10, 0.0]]],
        0,
    )
    assert multistate.n_states == 2
    assert multistate.surv[0] == pytest.approx([0.75, 0.25])
    assert sum(multistate.surv[0]) == pytest.approx(1.0)

    with pytest.raises(IndexError, match="out of range"):
        result.get_surv_at_state(1)

    with pytest.raises(ValueError, match="surv length must match time length"):
        survival.surv_analysis.SurvfitMatrixResult([1.0], [], [[0.1]], None, [], [], 1)

    with pytest.raises(ValueError, match="surv values must be between 0 and 1"):
        survival.surv_analysis.SurvfitMatrixResult([1.0], [[1.2]], [[0.1]], None, [], [], 1)

    with pytest.raises(ValueError, match="hazard contains non-finite"):
        survival.surv_analysis.survfit_from_hazard([1.0], [float("nan")])

    with pytest.raises(ValueError, match="n_risk must have the same length as time"):
        survival.surv_analysis.survfit_from_hazard([1.0, 2.0], [0.1, 0.2], n_risk=[10.0])

    with pytest.raises(ValueError, match="hazard_matrix must be non-negative"):
        survival.surv_analysis.survfit_from_matrix([1.0], [[-0.1]])

    with pytest.raises(ValueError, match="hazard_matrix length must match time length"):
        survival.surv_analysis.survfit_from_matrix([1.0, 2.0], [[0.1]])

    with pytest.raises(ValueError, match="hazard_matrix must have at least one column"):
        survival.surv_analysis.survfit_from_matrix([1.0], [[]])

    with pytest.raises(ValueError, match="off-diagonal entries must be non-negative"):
        survival.surv_analysis.survfit_multistate(
            [1.0],
            [[[0.0, -0.1], [0.0, 0.0]]],
            0,
        )

    with pytest.raises(ValueError, match="transition_hazards must have at least one state"):
        survival.surv_analysis.survfit_multistate([1.0], [[]], 0)

    with pytest.raises(ValueError, match="outgoing row sums"):
        survival.surv_analysis.survfit_multistate(
            [1.0],
            [[[0.0, 0.8, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            0,
        )

    core = survival._survival

    with pytest.raises(ValueError, match="base_hazards must be non-decreasing"):
        core.cox_survfit_from_baseline(
            [1.0, 2.0],
            [0.2, 0.1],
            [0.0],
            0.0,
            None,
            None,
            None,
        )

    with pytest.raises(ValueError, match="no baseline hazard"):
        core.cox_survfit_from_baseline(
            [1.0],
            [0.2],
            [0.0],
            0.0,
            [0],
            [1],
            None,
        )


# ---------------------------------------------------------------------------------------------
# survdiff, survcheck, aggregate_survfit
# ---------------------------------------------------------------------------------------------

_SD_TIME = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]
_SD_STATUS = [1, 1, 1, 0, 1, 0, 1, 1, 0]
_SD_GROUP = [1, 1, 2, 1, 3, 2, 3, 2, 3]


def test_survdiff_matches_r():
    result = survival.surv_analysis.survdiff(_SD_TIME, _SD_STATUS, _SD_GROUP)

    # survdiff(Surv(t, s) ~ g)
    assert isinstance(result, survival.surv_analysis.SurvDiffResult)
    assert result.n == [3, 3, 3]
    assert result.group_codes == [1, 2, 3]
    assert [row[0] for row in result.obs] == pytest.approx([2.0, 2.0, 2.0])
    assert [row[0] for row in result.exp] == pytest.approx([1.0, 2.25, 2.75])
    np.testing.assert_allclose(
        result.var,
        [
            [0.68253968253968256, -0.32738095238095233, -0.35515873015873012],
            [-0.32738095238095238, 1.31845238095238093, -0.99107142857142860],
            [-0.35515873015873017, -0.99107142857142860, 1.34623015873015861],
        ],
    )
    assert result.chisq == pytest.approx(1.5105257668985863)
    assert result.pvalue == pytest.approx(0.46988707295818827)
    assert result.df == 2
    assert result.strata is None

    rho = survival.surv_analysis.survdiff(_SD_TIME, _SD_STATUS, _SD_GROUP, rho=1.0)
    assert rho.chisq == pytest.approx(1.8983088156678258)
    assert [row[0] for row in rho.exp] == pytest.approx(
        [0.88888888888888884, 1.59259259259259234, 1.88888888888888862]
    )

    stratified = survival.surv_analysis.survdiff(
        _SD_TIME, _SD_STATUS, _SD_GROUP, strata=[1, 1, 1, 2, 2, 2, 1, 2, 1]
    )
    # survdiff(Surv(t, s) ~ g + strata(st)): obs/exp are group x stratum
    np.testing.assert_allclose(stratified.obs, [[2.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(stratified.exp, [[0.9, 0.25], [0.7, 1.5], [2.4, 0.25]])
    assert stratified.chisq == pytest.approx(1.1151131290777976)
    assert stratified.var[0] == pytest.approx(
        [0.67749999999999999, -0.28833333333333333, -0.38916666666666666]
    )

    one_sample = survival.surv_analysis.survdiff_one_sample(
        _SD_STATUS, [0.5, 0.3, 0.8, 0.4, 0.9, 0.2, 0.6, 0.7, 0.5]
    )
    # survdiff(Surv(t, s) ~ offset(expected))
    assert one_sample.obs[0] == pytest.approx([6.0])
    assert one_sample.exp[0] == pytest.approx([6.3120004444308409])
    assert one_sample.chisq == pytest.approx(0.015422096082222288)

    with pytest.raises(ValueError, match="length"):
        survival.surv_analysis.survdiff(_SD_TIME, _SD_STATUS, _SD_GROUP[:-1])


def test_logrank_test_matches_survdiff():
    result = survival.validation.logrank_test(_SD_TIME, _SD_STATUS, _SD_GROUP)

    assert isinstance(result, survival.validation.LogRankResult)
    assert result.statistic == pytest.approx(1.5105257668985863)
    assert result.p_value == pytest.approx(0.46988707295818827)
    assert result.df == 2
    assert result.observed == pytest.approx([2.0, 2.0, 2.0])
    assert result.expected == pytest.approx([1.0, 2.25, 2.75])
    assert result.variance[0] == pytest.approx(
        [0.68253968253968256, -0.32738095238095233, -0.35515873015873012]
    )
    assert result.rho == pytest.approx(0.0)
    assert survival.validation.logrank_test(_SD_TIME, _SD_STATUS, _SD_GROUP, rho=1.0).statistic == (
        pytest.approx(1.8983088156678258)
    )


def test_survcheck_matches_r():
    result = survival.validation.survcheck(
        [1, 1, 2, 3, 4],
        [2.0, 4.0, 1.0, 3.0, 5.0],
        [1, 2, 2, 1, 0],
        ["a", "b"],
        time1=[0.0, 2.0, 0.0, 1.0, 0.0],
    )

    # survcheck(Surv(t1, t2, state) ~ 1, id = id) on a two-state process
    assert isinstance(result, survival.validation.SurvCheckResult)
    assert result.states == ["(s0)", "a", "b"]
    transitions = result.transitions
    assert isinstance(transitions, survival.validation.SurvCheckTransitions)
    assert transitions.from_states == ["(s0)", "a", "b"]
    assert transitions.to_states == ["a", "b", "(censored)"]
    assert transitions.counts == [[2, 1, 1], [0, 1, 0], [0, 0, 0]]
    events = result.events
    assert isinstance(events, survival.validation.SurvCheckEvents)
    assert events.states == ["a", "b", "(any)"]
    assert events.count == [0, 1, 2]
    assert events.subjects == [[2, 2, 0], [2, 2, 0], [1, 2, 1]]
    flag = result.flag
    assert isinstance(flag, survival.validation.SurvCheckFlags)
    assert (flag.overlap, flag.gap, flag.jump, flag.teleport, flag.duplicate) == (0, 0, 0, 0, 0)
    assert (result.n_id, result.n_observations, result.n_transitions) == (4, 5, 4)
    assert result.istate == ["(s0)", "a", "(s0)", "(s0)", "(s0)"]
    assert result.overlap is None
    assert result.gap is None

    overlapping = survival.validation.survcheck(
        [2, 2, 1, 1, 1],
        [10.0, 15.0, 10.0, 15.0, 25.0],
        [0, 1, 0, 1, 0],
        ["event"],
        time1=[0.0, 5.0, 0.0, 5.0, 20.0],
    )
    assert overlapping.flag.overlap == 2
    assert overlapping.flag.gap == 1
    assert isinstance(overlapping.overlap, survival.validation.SurvCheckProblem)
    assert sorted(overlapping.overlap.id) == [1, 2]
    assert sorted(overlapping.overlap.row) == [1, 3]
    assert overlapping.gap.id == [1]
    assert overlapping.gap.row == [4]

    with pytest.raises(ValueError, match="length"):
        survival.validation.survcheck([1], [1.0, 2.0], [1], ["event"])


def test_aggregate_survfit_matches_r_aggregate():
    # aggregate(survfit) collapses the data columns of a curve matrix within `by` groups
    surv = [[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]]
    by = [survival.surv_analysis.GroupingFactor([0, 0, 1], ["a", "b"], name="grp")]

    mean = survival.surv_analysis.aggregate_survfit(surv=surv, by=by)
    assert isinstance(mean, survival.surv_analysis.AggregateSurvfitResult)
    np.testing.assert_allclose(mean.surv, [[0.85, 0.7], [0.45, 0.3]])
    assert mean.pstate is None
    assert isinstance(mean.newdata, survival.surv_analysis.AggregateGroups)
    assert mean.newdata.names == ["grp"]
    assert mean.newdata.labels == [["a"], ["b"]]

    maximum = survival.surv_analysis.aggregate_survfit(surv=surv, by=by, fun="max")
    np.testing.assert_allclose(maximum.surv, [[0.9, 0.7], [0.5, 0.3]])

    pstate = survival.surv_analysis.aggregate_survfit(
        pstate=[[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]], by=by, fun="median"
    )
    np.testing.assert_allclose(pstate.pstate, [[[0.85, 0.15], [0.7, 0.3]]])

    with pytest.raises(ValueError, match="FUN must be one of"):
        survival.surv_analysis.aggregate_survfit(surv=surv, by=by, fun="weird")
    with pytest.raises(ValueError, match="surv"):
        survival.surv_analysis.aggregate_survfit(by=by)


def test_life_table_public_api_and_validation():
    result = survival.validation.life_table(
        time=[1.0, 2.0, 4.0],
        status=[1, 0, 1],
        breaks=[0.0, 2.0, 4.0],
    )

    assert result.interval_start == pytest.approx([0.0, 2.0])
    assert result.interval_end == pytest.approx([2.0, 4.0])
    assert result.n_deaths == pytest.approx([1.0, 1.0])
    assert result.n_censored == pytest.approx([0.0, 1.0])

    with pytest.raises(ValueError, match="time and status must have same length"):
        survival.validation.life_table([1.0], [], [0.0, 2.0])
    with pytest.raises(ValueError, match="breaks must define at least one interval"):
        survival.validation.life_table([1.0], [1], [0.0])
    with pytest.raises(ValueError, match="time contains NaN"):
        survival.validation.life_table([float("nan")], [1], [0.0, 2.0])
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.validation.life_table([1.0], [2], [0.0, 2.0])
    with pytest.raises(ValueError, match="breaks must be strictly increasing"):
        survival.validation.life_table([1.0], [1], [0.0, 0.0])
    with pytest.raises(ValueError, match="time values must fall within the break range"):
        survival.validation.life_table([3.0], [1], [0.0, 2.0])


# ---------------------------------------------------------------------------------------------
# royston, brier, yates, anova, cipoisson, survobrien, hypothesis tests
# ---------------------------------------------------------------------------------------------


def _lung40_cox():
    """coxph(Surv(time, status) ~ age + sex, lung[1:40, ]) with R's status - 1 coding."""
    data = survival.datasets.load_lung()
    n = 40
    time = [float(t) for t in data["time"][:n]]
    status = [int(s) - 1 for s in data["status"][:n]]
    x = [[float(a), float(s)] for a, s in zip(data["age"][:n], data["sex"][:n], strict=True)]
    return time, status, x, survival.regression.coxph_fit(time, status, x)


def test_royston_matches_r():
    time, status, _, fit = _lung40_cox()

    result = survival.validation.royston(
        fit.linear_predictors, time, status, fit.loglik, fit.nevent, 2
    )
    # royston(fit): D, se(D), R.D, R.KO, R.N, C.GH
    assert isinstance(result, survival.validation.RoystonResult)
    assert result.d == pytest.approx(0.03981824901943797901)
    assert result.se_d == pytest.approx(0.25603657261269513468)
    assert result.r_d == pytest.approx(0.00037836534700899557)
    assert result.r_ko == pytest.approx(0.00250860021026843042)
    assert result.r_n == pytest.approx(0.00430570107636293279)
    assert result.c_gh == pytest.approx(0.51865217566019261586)

    adjusted = survival.validation.royston(
        fit.linear_predictors, time, status, fit.loglik, fit.nevent, 2, adjust=True
    )
    assert adjusted.d == pytest.approx(-0.3689820154379666595)
    assert adjusted.se_d == pytest.approx(0.0261363709160594947)

    with pytest.raises(ValueError, match="length"):
        survival.validation.royston(fit.linear_predictors[:-1], time, status, fit.loglik, 37, 2)


def test_brier_matches_r():
    time, status, x, fit = _lung40_cox()
    times = [100.0, 200.0, 300.0]

    (curve,) = fit.survfit(newdata=x, se_fit=False)
    curve_times = np.asarray(curve.time)
    curve_surv = np.asarray(curve.surv)
    positions = np.searchsorted(curve_times, times, side="right") - 1
    phat = [(1.0 - curve_surv[position]).tolist() for position in positions]

    result = survival.validation.brier(time, status, times, phat)
    # R: brier score of the Cox fit at 100, 200, 300
    assert isinstance(result, survival.validation.BrierResult)
    assert result.times == pytest.approx(times)
    assert result.brier == pytest.approx(
        [0.15976459951163186, 0.22733644682597290, 0.23750409195705183]
    )
    assert result.rsquared == pytest.approx(
        [0.00147125305230089154, 0.00071891505066856709, 0.01039961684561729882]
    )
    assert result.eff_n == pytest.approx([40.0, 40.0, 40.0])
    assert result.p0 == pytest.approx([0.2, 0.35, 0.4])

    with pytest.raises(ValueError, match="phat"):
        survival.validation.brier(time, status, times, phat[:2])


def test_yates_linear_predictor_matches_r():
    data = survival.datasets.load_lung()
    n = 40
    time = [float(t) for t in data["time"][:n]]
    status = [int(s) - 1 for s in data["status"][:n]]
    # sex as a treatment contrast (R's factor coding), age at its mean 63.25
    x = [
        [float(a), 1.0 if s == 2 else 0.0]
        for a, s in zip(data["age"][:n], data["sex"][:n], strict=True)
    ]
    fit = survival.regression.coxph_fit(time, status, x)
    assert fit.means == pytest.approx([63.25, 0.0])

    cmat = survival.validation.yates_population_means([[[63.25, 0.0]], [[63.25, 1.0]]])
    np.testing.assert_allclose(cmat, [[63.25, 0.0], [63.25, 1.0]])

    result = survival.validation.yates(
        cmat,
        fit.coefficients,
        fit.var,
        offset=-sum(m * b for m, b in zip(fit.means, fit.coefficients, strict=True)),
    )
    # yates(coxph(Surv(time, status) ~ age + factor(sex)), ~ factor(sex), predict = "linear")
    assert isinstance(result, survival.validation.YatesResult)
    estimates = result.estimate
    assert all(isinstance(item, survival.validation.YatesEstimate) for item in estimates)
    assert [item.pmm for item in estimates] == pytest.approx([0.0, -0.094528607850935142])
    assert [item.std for item in estimates] == pytest.approx(
        [1.5783638980241397, 1.4980223726078221]
    )
    (test,) = result.test
    assert isinstance(test, survival.validation.YatesContrast)
    assert (test.name, test.df) == ("global", 1)
    assert test.chisq == pytest.approx(0.061185962294992374)
    assert test.ss is None
    np.testing.assert_allclose(
        result.mvar,
        [[2.4912325945859566, 2.2946313232915623], [2.2946313232915623, 2.2440710288335688]],
    )
    np.testing.assert_allclose(result.cmat, cmat)

    with pytest.raises(ValueError, match="test"):
        survival.validation.yates(cmat, fit.coefficients, fit.var, test="weird")


def test_anova_coxph_and_hypothesis_tests_match_r():
    time, status, _, fit = _lung40_cox()
    loglik = [-108.52888024552935, -108.47379191989565, -108.44296072448500]

    anova = survival.validation.anova_coxph(loglik, [0, 1, 2], names=["NULL", "age", "sex"])
    # anova(coxph(Surv(time, status) ~ age + sex))
    assert isinstance(anova, survival.validation.AnovaCoxphResult)
    assert anova.test == "Chisq"
    assert [row.name for row in anova.rows] == ["NULL", "age", "sex"]
    assert [row.loglik for row in anova.rows] == pytest.approx(loglik)
    assert anova.rows[0].chisq is None
    assert [row.chisq for row in anova.rows[1:]] == pytest.approx(
        [0.110176651267408943, 0.061662390821282997]
    )
    assert [row.df for row in anova.rows[1:]] == [1, 1]
    assert [row.p_value for row in anova.rows[1:]] == pytest.approx(
        [0.739943110443081253, 0.803887498076046536]
    )

    wald = survival.validation.wald_test(fit.coefficients, fit.var)
    assert isinstance(wald, survival.validation.TestResult)
    assert wald.statistic == pytest.approx(0.17139103502639053)
    assert wald.df == 2
    lrt = survival.validation.lrt_test(fit.loglik[1], fit.loglik[0], 2)
    assert lrt.statistic == pytest.approx(0.17183904208869194)
    assert lrt.p_value == pytest.approx(0.9176680812172392)
    assert lrt.test_name == "LikelihoodRatioTest"
    score = survival.validation.score_test([1.0, 2.0], [[4.0, 0.0], [0.0, 2.0]])
    assert score.statistic == pytest.approx(0.25 + 2.0)
    assert score.df == 2

    with pytest.raises(ValueError, match="length"):
        survival.validation.anova_coxph([-1.0, -0.5], [1])


def test_cipoisson_and_survobrien_match_r():
    exact = survival.validation.cipoisson([0, 5, 20], time=[1.0, 2.0, 10.0])
    # R: exact Poisson limits for counts 0, 5, 20 over 1, 2, 10 time units
    assert isinstance(exact, survival.validation.CipoissonResult)
    assert exact.lower == pytest.approx([0.0, 0.81174319505921044, 1.22165195854039443])
    assert exact.upper == pytest.approx(
        [3.68887945411393536, 5.83416603966133351, 3.08883779026745930]
    )
    anscombe = survival.validation.cipoisson([5], time=[2.0], method="anscombe")
    assert anscombe.lower == pytest.approx([0.75394070032766036])
    assert anscombe.upper == pytest.approx([5.79300183486583187])

    expansion = survival.validation.survobrien(
        _SD_TIME, _SD_STATUS, [[0.5, 0.3, 0.8, 0.4, 0.9, 0.2, 0.6, 0.7, 0.5]]
    )
    # survobrien(Surv(t, s) ~ x): one block of rows per event time (.id. = row + 1)
    assert isinstance(expansion, survival.validation.SurvObrienExpansion)
    assert expansion.row == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        3,
        4,
        5,
        6,
        7,
        8,
        6,
        7,
        8,
        7,
        8,
    ]
    assert expansion.time[:9] == pytest.approx(_SD_TIME)
    assert expansion.status[:11] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert expansion.strata[:10] == [1] * 9 + [2]
    assert expansion.event_times == pytest.approx([1.0, 2.0, 3.0, 5.0, 6.0])
    assert expansion.transformed[0][:3] == pytest.approx(
        [-0.22314355131420985, -1.60943791243410050, 1.60943791243410073]
    )
    assert expansion.start is None


def test_concordance_metric_public_apis_and_validation():
    uno = survival.validation.uno_c_index(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    comparison = survival.validation.compare_uno_c_indices(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score_1=[0.9, 0.7, 0.4, 0.2],
        risk_score_2=[0.2, 0.4, 0.7, 0.9],
    )
    decomposition = survival.validation.c_index_decomposition(
        time=[1.0, 2.0, 3.0, 4.0],
        status=[1, 1, 0, 1],
        risk_score=[0.9, 0.7, 0.4, 0.2],
    )
    gonen = survival.validation.gonen_heller_concordance([0.9, 0.7, 0.4, 0.2])

    assert uno.c_index == pytest.approx(1.0)
    assert uno.comparable_pairs > 0.0
    assert 0.0 <= uno.ci_lower <= uno.ci_upper <= 1.0
    assert comparison.c_index_1 > comparison.c_index_2
    assert comparison.difference > 0.0
    assert 0.0 <= comparison.p_value <= 1.0
    assert decomposition.c_index == pytest.approx(1.0)
    assert 0.0 <= decomposition.alpha <= 1.0
    assert decomposition.n_event_event_pairs > 0
    assert 0.0 <= gonen.cpe <= 1.0
    assert gonen.std_error >= 0.0

    with pytest.raises(ValueError, match="linear_predictor must not be empty"):
        survival.validation.gonen_heller_concordance([])
    with pytest.raises(ValueError, match="linear_predictor contains NaN"):
        survival.validation.gonen_heller_concordance([0.1, float("nan")])
    with pytest.raises(ValueError, match="linear_predictor contains non-finite"):
        survival.validation.gonen_heller_concordance([0.1, float("inf")])

    exact_time = [1.0, 2.0, 2.0, 3.0, 4.0]
    near_time = [1.0, 2.0 + 5e-10, 2.0, 3.0, 4.0]
    boundary_status = [1, 1, 1, 0, 0]
    boundary_risk = [0.9, 0.7, 0.8, 0.4, 0.2]
    reverse_boundary_risk = [0.2, 0.4, 0.3, 0.7, 0.9]

    exact_uno = survival.validation.uno_c_index(exact_time, boundary_status, boundary_risk, 2.0)
    near_uno = survival.validation.uno_c_index(near_time, boundary_status, boundary_risk, 2.0)
    assert near_uno.comparable_pairs == pytest.approx(exact_uno.comparable_pairs)
    assert near_uno.concordant == pytest.approx(exact_uno.concordant)
    assert near_uno.discordant == pytest.approx(exact_uno.discordant)
    assert near_uno.tied_risk == pytest.approx(exact_uno.tied_risk)
    assert near_uno.c_index == pytest.approx(exact_uno.c_index)

    exact_comparison = survival.validation.compare_uno_c_indices(
        exact_time, boundary_status, boundary_risk, reverse_boundary_risk, 2.0
    )
    near_comparison = survival.validation.compare_uno_c_indices(
        near_time, boundary_status, boundary_risk, reverse_boundary_risk, 2.0
    )
    assert near_comparison.c_index_1 == pytest.approx(exact_comparison.c_index_1)
    assert near_comparison.c_index_2 == pytest.approx(exact_comparison.c_index_2)
    assert near_comparison.difference == pytest.approx(exact_comparison.difference)
    assert near_comparison.variance_diff == pytest.approx(exact_comparison.variance_diff)

    exact_decomposition = survival.validation.c_index_decomposition(
        exact_time, boundary_status, boundary_risk, 2.0
    )
    near_decomposition = survival.validation.c_index_decomposition(
        near_time, boundary_status, boundary_risk, 2.0
    )
    assert near_decomposition.n_event_event_pairs == exact_decomposition.n_event_event_pairs
    assert near_decomposition.n_event_censored_pairs == exact_decomposition.n_event_censored_pairs
    assert near_decomposition.c_index == pytest.approx(exact_decomposition.c_index)
    assert near_decomposition.c_index_ee == pytest.approx(exact_decomposition.c_index_ee)
    assert near_decomposition.c_index_ec == pytest.approx(exact_decomposition.c_index_ec)
    assert near_decomposition.alpha == pytest.approx(exact_decomposition.alpha)

    with pytest.raises(ValueError, match="time, status, and risk_score must have the same length"):
        survival.validation.uno_c_index([1.0], [1], [0.1, 0.2])
    with pytest.raises(ValueError, match="time cannot be empty"):
        survival.validation.uno_c_index([], [], [])
    with pytest.raises(ValueError, match="time contains NaN"):
        survival.validation.uno_c_index([float("nan")], [1], [0.5])
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.validation.uno_c_index([1.0], [2], [0.5])
    with pytest.raises(ValueError, match="risk_score contains non-finite"):
        survival.validation.uno_c_index([1.0], [1], [float("inf")])
    with pytest.raises(ValueError, match="tau must be non-negative"):
        survival.validation.uno_c_index([1.0], [1], [0.5], -1.0)
    with pytest.raises(ValueError, match="risk_score_2 contains NaN"):
        survival.validation.compare_uno_c_indices([1.0], [1], [0.5], [float("nan")])
    with pytest.raises(ValueError, match="tau must be finite"):
        survival.validation.c_index_decomposition([1.0], [1], [0.5], float("inf"))


def test_time_dependent_auc_public_apis_and_validation():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    marker = [0.9, 0.2, 0.7, 0.1]

    auc = survival.validation.time_dependent_auc(time, status, marker, 2.5)
    cumulative = survival.validation.cumulative_dynamic_auc(time, status, marker, [1.5, 2.5, 3.5])

    assert 0.0 <= auc.auc <= 1.0
    assert auc.n_cases == 1
    assert auc.n_controls == 2
    assert len(cumulative.auc) == 3
    assert 0.0 <= cumulative.mean_auc <= 1.0

    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    boundary_status = [1, 1, 0, 0]
    boundary_marker = [0.9, 0.8, 0.2, 0.1]

    exact_auc = survival.validation.time_dependent_auc(
        exact_time, boundary_status, boundary_marker, 1.0
    )
    near_auc = survival.validation.time_dependent_auc(
        near_time, boundary_status, boundary_marker, 1.0
    )
    assert near_auc.n_cases == exact_auc.n_cases
    assert near_auc.n_controls == exact_auc.n_controls
    assert near_auc.auc == pytest.approx(exact_auc.auc)
    assert near_auc.std_error == pytest.approx(exact_auc.std_error)

    exact_cumulative = survival.validation.cumulative_dynamic_auc(
        exact_time, boundary_status, boundary_marker, [1.0, 2.0]
    )
    near_cumulative = survival.validation.cumulative_dynamic_auc(
        near_time, boundary_status, boundary_marker, [1.0, 2.0]
    )
    assert near_cumulative.n_cases == exact_cumulative.n_cases
    assert near_cumulative.n_controls == exact_cumulative.n_controls
    assert near_cumulative.auc == pytest.approx(exact_cumulative.auc)
    assert near_cumulative.mean_auc == pytest.approx(exact_cumulative.mean_auc)
    assert near_cumulative.integrated_auc == pytest.approx(exact_cumulative.integrated_auc)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.validation.time_dependent_auc([float("nan")], [1], [0.5], 1.0)

    with pytest.raises(ValueError, match="time contains negative value"):
        survival.validation.time_dependent_auc([-1.0], [1], [0.5], 1.0)

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.validation.time_dependent_auc([1.0], [2], [0.5], 1.0)

    with pytest.raises(ValueError, match="marker contains non-finite"):
        survival.validation.time_dependent_auc([1.0], [1], [float("inf")], 1.0)

    with pytest.raises(ValueError, match="t contains non-finite"):
        survival.validation.time_dependent_auc([1.0], [1], [0.5], float("nan"))

    with pytest.raises(ValueError, match="times must be sorted"):
        survival.validation.cumulative_dynamic_auc(time, status, marker, [2.5, 1.5])


def test_landmark_summary_apis_group_near_tied_event_times():
    exact_time = [1.0, 1.0, 2.0, 3.0]
    near_time = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [1, 1, 0, 0]

    exact_conditional = survival.validation.conditional_survival(exact_time, status, 1.0, 2.0)
    near_conditional = survival.validation.conditional_survival(near_time, status, 1.0, 2.0)
    assert near_conditional.conditional_survival == pytest.approx(
        exact_conditional.conditional_survival
    )
    assert near_conditional.n_at_risk == exact_conditional.n_at_risk

    exact_survival = survival.validation.survival_at_times(exact_time, status, [1.0, 2.0])
    near_survival = survival.validation.survival_at_times(near_time, status, [1.0, 2.0])
    for actual, expected in zip(near_survival, exact_survival, strict=True):
        assert actual.survival == pytest.approx(expected.survival)
        assert actual.n_at_risk == expected.n_at_risk
        assert actual.n_events == expected.n_events

    group = [0, 1, 0, 1, 0, 1]
    grouped_status = [1, 1, 1, 0, 0, 0]
    exact_hazard = survival.validation.hazard_ratio(
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], grouped_status, group
    )
    near_hazard = survival.validation.hazard_ratio(
        [1.0, 1.0 + 5e-10, 2.0, 2.0 + 5e-10, 3.0, 3.0],
        grouped_status,
        group,
    )
    assert near_hazard.hazard_ratio == pytest.approx(exact_hazard.hazard_ratio)
    assert near_hazard.se_log_hr == pytest.approx(exact_hazard.se_log_hr)
    assert near_hazard.p_value == pytest.approx(exact_hazard.p_value)


def test_landmark_summary_apis_validate_public_inputs():
    with pytest.raises(ValueError, match="time and status must have same length"):
        survival.validation.landmark_analysis([1.0], [], 0.5)
    with pytest.raises(ValueError, match="time contains NaN"):
        survival.validation.landmark_analysis([float("nan")], [1], 0.5)
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.validation.landmark_analysis([1.0], [2], 0.5)
    with pytest.raises(ValueError, match="landmark_times contains non-finite"):
        survival.validation.landmark_analysis_batch([1.0], [1], [float("inf")])
    with pytest.raises(ValueError, match="given_time must be finite"):
        survival.validation.conditional_survival([1.0], [1], float("nan"), 2.0)
    with pytest.raises(ValueError, match="confidence_level"):
        survival.validation.conditional_survival([1.0], [1], 0.5, 2.0, 1.0)
    with pytest.raises(ValueError, match="group must have same length"):
        survival.validation.hazard_ratio([1.0], [1], [])
    with pytest.raises(ValueError, match="eval_times contains NaN"):
        survival.validation.survival_at_times([1.0], [1], [float("nan")])


def test_rcll_public_apis_and_validation():
    result = survival.validation.rcll(
        survival_predictions=[
            [0.95, 0.85, 0.70],
            [0.90, 0.75, 0.55],
            [0.98, 0.92, 0.80],
        ],
        prediction_times=[1.0, 2.0, 3.0],
        event_times=[2.5, 1.5, 3.0],
        status=[1, 1, 0],
        weights=[1.0, 2.0, 1.0],
    )
    single_time = survival.validation.rcll_single_time(
        survival_probs=[0.8, 0.7, 0.9],
        event_times=[1.0, 2.0, 3.0],
        status=[1, 0, 1],
        prediction_time=2.0,
        weights=[1.0, 2.0, 1.0],
    )

    assert result.n_events == 2
    assert result.n_censored == 1
    assert result.mean_rcll > 0.0
    assert result.event_contribution > result.censored_contribution
    assert single_time.n_events == 1
    assert single_time.n_censored == 2
    assert single_time.rcll > 0.0

    with pytest.raises(ValueError, match="survival_predictions row 0 has 1 elements, expected 2"):
        survival.validation.rcll([[0.9]], [1.0, 2.0], [1.0], [1], None)

    duplicate_times = survival.validation.rcll(
        survival_predictions=[
            [0.95, 0.8, 0.7, 0.5],
            [0.95, 0.8, 0.7, 0.5],
        ],
        prediction_times=[1.0, 2.0, 2.0, 3.0],
        event_times=[2.0, 2.0],
        status=[1, 0],
    )

    assert duplicate_times.event_contribution == pytest.approx(-log(0.15))
    assert duplicate_times.censored_contribution == pytest.approx(-log(0.7))

    with pytest.raises(ValueError, match="prediction_times must be sorted"):
        survival.validation.rcll([[0.9, 0.8]], [2.0, 1.0], [1.0], [1])

    with pytest.raises(ValueError, match="status.*0/1"):
        survival.validation.rcll([[0.9]], [1.0], [1.0], [2])

    with pytest.raises(ValueError, match="probabilities between 0 and 1"):
        survival.validation.rcll([[1.2]], [1.0], [1.0], [1])

    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.validation.rcll([[0.9]], [1.0], [1.0], [1], [-1.0])

    with pytest.raises(ValueError, match="prediction_time contains non-finite"):
        survival.validation.rcll_single_time([0.9], [1.0], [1], float("nan"))


# ---------------------------------------------------------------------------------------------
# RMST family
# ---------------------------------------------------------------------------------------------


def test_rmst_family_matches_r_survmean():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1]
    group = [0, 0, 1, 1]

    comparison = survival.validation.rmst_comparison(time, status, group, 4.0)
    # summary(survfit(Surv(time, status) ~ group), rmean = 4)$table per group
    assert isinstance(comparison, survival.validation.RmstComparisonResult)
    assert comparison.tau == pytest.approx(4.0)
    first, second = comparison.groups
    assert isinstance(first, survival.validation.RmstGroupResult)
    assert (first.group, first.n, first.events) == (0, 2, 2.0)
    assert first.rmean == pytest.approx(1.5)
    assert first.se_rmean == pytest.approx(0.35355339059327379)
    assert (second.group, second.n, second.events) == (1, 2, 1.0)
    assert second.rmean == pytest.approx(4.0)
    assert second.se_rmean == pytest.approx(0.0)
    assert comparison.difference == pytest.approx([2.5])
    assert comparison.difference_se == pytest.approx([0.35355339059327379])
    assert comparison.df == 1
    assert comparison.p_value < 1e-6

    nnt = survival.validation.number_needed_to_treat(
        [1.0, 1.0, 3.0, 2.0, 2.0, 3.0], [1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1], 2.0
    )
    assert nnt.time_horizon == pytest.approx(2.0)
    # S_control(2) = 1/3, S_treated(2) = 2/3: ARR = 1/3, NNT = 3
    assert nnt.absolute_risk_reduction == pytest.approx(1 / 3)
    assert nnt.nnt == pytest.approx(3.0)

    threshold = survival.validation.rmst_optimal_threshold(
        [1.0, 1.0, 2.0, 3.0, 4.0, 4.5], [1, 1, 1, 1, 0, 0], alpha=0.999, min_events_per_interval=2
    )
    assert isinstance(threshold, survival.validation.RMSTOptimalThresholdResult)
    assert threshold.max_followup == pytest.approx(4.5)
    assert threshold.optimal_tau <= threshold.max_followup
    assert threshold.rmean > 0.0

    near = survival.validation.rmst_optimal_threshold(
        [1.0, 1.0 + 5e-10, 2.0, 3.0, 4.0, 4.5],
        [1, 1, 1, 1, 0, 0],
        alpha=0.999,
        min_events_per_interval=2,
    )
    assert near.optimal_tau == pytest.approx(threshold.optimal_tau)
    assert near.rmean == pytest.approx(threshold.rmean)

    with pytest.raises(ValueError, match="length"):
        survival.validation.rmst_comparison([1.0], [1], [0, 1], 1.0)
    with pytest.raises(ValueError, match="group length mismatch"):
        survival.validation.number_needed_to_treat([1.0], [1], [0, 1], 1.0)
    with pytest.raises(ValueError, match="alpha must be greater than 0"):
        survival.validation.rmst_optimal_threshold([1.0], [1], alpha=0.0)
    with pytest.raises(ValueError, match="min_events_per_interval must be at least 2"):
        survival.validation.rmst_optimal_threshold([1.0], [1], min_events_per_interval=1)


def test_survfitaj_extended_public_apis_and_validation():
    config = survival.surv_analysis.AalenJohansenExtendedConfig()
    result = survival.surv_analysis.survfitaj_extended(
        from_state=[0, 0, 0],
        to_state=[1, 2, 0],
        time=[1.0, 1.0 + 5e-10, 2.0],
        config=config,
        weights=None,
    )

    assert result.n_obs == 3
    assert result.n_events == 2
    assert result.time == pytest.approx([1.0, 2.0])
    assert result.transition_matrices[0].n_transitions[0][1] == 1
    assert result.transition_matrices[0].n_transitions[0][2] == 1
    assert result.transition_matrices[0].matrix[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result.get_cif(1) == pytest.approx([1 / 3, 1 / 3])
    assert result.get_state_prob(0, 1) == pytest.approx([1 / 3, 1 / 3])
    assert result.interpolate_at(1.0)[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    with pytest.raises(ValueError, match="from_state, to_state, and time must have equal length"):
        survival.surv_analysis.survfitaj_extended([0], [1, 2], [1.0], config, None)

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.surv_analysis.survfitaj_extended([0, 0], [1, 2], [1.0, float("inf")], config, None)

    with pytest.raises(ValueError, match="weights must have length n_obs"):
        survival.surv_analysis.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, [1.0])

    with pytest.raises(ValueError, match="weights contains negative value"):
        survival.surv_analysis.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, [1.0, -1.0])

    config.confidence_level = float("nan")
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        survival.surv_analysis.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, None)

    config = survival.surv_analysis.AalenJohansenExtendedConfig()
    config.variance_estimator = survival.surv_analysis.VarianceEstimator("bootstrap")
    config.n_bootstrap = 0
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        survival.surv_analysis.survfitaj_extended([0, 0], [1, 2], [1.0, 2.0], config, None)


def test_basehaz_binding_matches_coxph_fit_baseline():
    fit = survival.regression.coxph_fit(
        _KM_TIME, _KM_STATUS, [[0.0], [0.2], [0.1], [-0.1], [0.3], [0.0], [0.2], [0.1]]
    )
    times, hazard = survival.surv_analysis.basehaz(
        time=_KM_TIME,
        status=_KM_STATUS,
        linear_predictors=fit.linear_predictors,
        centered=False,
    )

    # the fit's linear predictors are already centred, so this is basehaz(fit, centered = TRUE)
    expected = fit.basehaz(centered=True)
    assert times == pytest.approx([1.0, 2.0, 4.0, 6.0, 7.0])
    assert hazard == pytest.approx([expected.hazard[i] for i in (0, 1, 3, 5, 6)])

    with pytest.raises(
        ValueError,
        match="time, status, and linear_predictors must have the same length",
    ):
        survival.surv_analysis.basehaz([1.0], [1, 0], [0.1], False)


def test_basehaz_counts_same_time_censors_in_event_risk_set():
    times, hazard = survival.surv_analysis.basehaz(
        time=[2.0, 2.0, 3.0],
        status=[0, 1, 1],
        linear_predictors=[0.0, 0.0, 0.0],
        centered=False,
    )

    assert times == pytest.approx([2.0, 3.0])
    assert hazard == pytest.approx([1.0 / 3.0, 1.0 / 3.0 + 1.0])
