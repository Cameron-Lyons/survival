"""``survival.regression``: Cox (``coxph_fit`` / ``CoxPHFit``) and parametric (``survreg_fit`` /
``SurvregFit``) models against R survival 3.8.11 references."""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
regression = survival.regression
core = survival.core

# R: d <- data.frame(t = c(1,2,2,3,4,4,5,6,7,8), s = c(1,1,1,0,1,1,0,1,0,1), x1, x2)
_TIED_TIME = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_TIED_STATUS = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]
_TIED_X1 = [0.0, 0.4, 0.8, 0.2, 1.0, 1.4, 0.6, 1.2, 1.6, 1.8]
_TIED_X2 = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97]
_TIED_X = [[a, b] for a, b in zip(_TIED_X1, _TIED_X2, strict=True)]
_NEWDATA = [[0.5, 0.3], [1.5, 0.9]]

# R: d <- data.frame(t = 1:8, s = c(1,1,0,1,1,1,0,1), x = c(.5,.2,.9,.1,.7,.3,.8,.4))
_AFT_TIME = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_AFT_STATUS = [1, 1, 0, 1, 1, 1, 0, 1]
_AFT_X = [0.5, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.4]
_AFT_DESIGN = [[1.0, value] for value in _AFT_X]


def _tied_fit(**kwargs):
    return regression.coxph_fit(_TIED_TIME, _TIED_STATUS, _TIED_X, **kwargs)


def test_coxph_fit_efron_matches_r():
    fit = _tied_fit()

    # coxph(Surv(t, s) ~ x1 + x2, d): ties = "efron"
    assert isinstance(fit, regression.CoxPHFit)
    assert fit.method == regression.TieMethod.Efron
    assert fit.coefficients == pytest.approx([-2.34686780701378028, 0.57759281933864315])
    np.testing.assert_allclose(
        fit.var,
        [[3.9806704210981687, -4.1165383592668485], [-4.1165383592668485, 6.0567373235724258]],
    )
    assert fit.naive_var is None
    assert fit.rscore is None
    assert fit.loglik == pytest.approx([-11.079060882340368, -9.002136268091796])
    assert fit.score == pytest.approx(3.9327366994909028)
    assert fit.wald_test == pytest.approx(3.284067646975064)
    assert fit.iter == 4
    assert fit.flag == 2
    assert fit.means == pytest.approx([0.9, 0.585])
    assert (fit.n, fit.nevent, fit.nvar) == (10, 7, 2)
    assert fit.linear_predictors[:3] == pytest.approx(
        [1.889807790867024995, 0.927956955287966956, 0.254902529378230547]
    )
    assert fit.hazard_ratios() == pytest.approx([0.095668345141275812, 1.781744284046125726])
    assert fit.first == pytest.approx([0.0, 0.0], abs=1e-7)
    assert fit.strata is None
    assert fit.cluster is None
    assert fit.entry is None
    assert fit.nocenter == [False, False]
    assert fit.x[0] == pytest.approx([0.0, 0.2])


def test_coxph_fit_breslow_matches_r():
    fit = _tied_fit(method="breslow")

    assert fit.method == regression.TieMethod.Breslow
    assert fit.coefficients == pytest.approx([-2.31840202040787569, 0.49851177402429914])
    np.testing.assert_allclose(
        fit.var,
        [[3.9261720323257716, -3.9888855557265210], [-3.9888855557265210, 5.9305727482653090]],
    )
    assert fit.loglik == pytest.approx([-11.3791654747907067, -9.3511047589307736])
    assert fit.score == pytest.approx(3.796050528232672)
    assert fit.wald_test == pytest.approx(3.2051239607042534)
    assert fit.hazard_ratios() == pytest.approx([0.098430750328168962, 1.646269425780562790])
    assert fit.martingale_residuals()[:3] == pytest.approx(
        [0.636525000172903588, 0.416174237936349778, 0.709513473868391364]
    )


def test_coxph_fit_exact_ties_match_r():
    fit = _tied_fit(method="exact")

    assert fit.method == regression.TieMethod.Exact
    assert fit.coefficients == pytest.approx([-2.71433867809912455, 0.48496541470257887])
    np.testing.assert_allclose(
        fit.var,
        [[4.4111228049234628, -4.1915089564114147], [-4.1915089564114147, 6.5231955014425296]],
    )
    assert fit.loglik == pytest.approx([-9.6927665212204754, -7.3450470379782935])


def test_coxph_fit_iter_max_zero_evaluates_at_init():
    fit = _tied_fit(iter_max=0)
    assert fit.coefficients == pytest.approx([0.0, 0.0])
    assert fit.loglik == pytest.approx([-11.079060882340368, -11.079060882340368])
    assert fit.iter == 0

    started = _tied_fit(init=[-2.34686780701378028, 0.57759281933864315], iter_max=0)
    assert started.loglik[1] == pytest.approx(-9.002136268091796)


def test_coxph_fit_residual_methods_match_r():
    fit = _tied_fit()

    assert fit.residuals == pytest.approx(fit.martingale_residuals())
    assert fit.martingale_residuals() == pytest.approx(
        [
            0.635149405512543730,
            0.509438247825749979,
            0.749740909915125431,
            -0.871048977506360678,
            0.386232255905764044,
            0.802746854126972886,
            -1.475373020118189826,
            0.050719555006668249,
            -0.513068378290874816,
            -0.274536852377399221,
        ]
    )
    assert fit.deviance_residuals()[:3] == pytest.approx(
        [0.863849447365589551, 0.636813730529640520, 1.127401993537265223]
    )
    np.testing.assert_allclose(
        fit.score_residuals()[:2],
        [
            [-0.233452232810407168, -0.054675243151784514],
            [-0.123661810977466757, -0.097791564978628287],
        ],
    )
    np.testing.assert_allclose(
        fit.dfbeta()[:2],
        [
            [-0.704223662151148178, 0.629861485547724920],
            [-0.089694184730174861, -0.083239233071467883],
        ],
    )
    np.testing.assert_allclose(fit.dfbetas()[:1], [[-0.352965698671396322, 0.255932644177769064]])

    schoenfeld = fit.schoenfeld_residuals()
    assert isinstance(schoenfeld, regression.SchoenfeldResiduals)
    assert schoenfeld.time == pytest.approx([1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 8.0])
    assert schoenfeld.rows == [0, 1, 2, 4, 5, 7, 9]
    assert schoenfeld.strata is None
    np.testing.assert_allclose(
        schoenfeld.residuals[:2],
        [
            [-0.367554831641573987, -0.086082491264657890],
            [-0.183018714657403714, -0.177531782621244111],
        ],
    )
    scaled = fit.scaled_schoenfeld_residuals()
    np.testing.assert_allclose(
        scaled.residuals[:2],
        [
            [-10.10813719067957273, 7.51931450020224279],
            [-2.33091334257708471, -1.67542588266199388],
        ],
    )

    # collapse sums residuals within the given groups
    collapsed = fit.martingale_residuals(collapse=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    full = fit.martingale_residuals()
    assert collapsed == pytest.approx([full[i] + full[i + 1] for i in range(0, 10, 2)])


def test_coxph_fit_predictions_match_r():
    fit = _tied_fit()

    lp = fit.predict("lp", newdata=_NEWDATA, se_fit=True)
    assert isinstance(lp, regression.CoxPrediction)
    assert lp.fit == pytest.approx([0.77413316929399911, -1.22617894611659528])
    assert lp.se_fit == pytest.approx([0.43622816343060195, 0.69135346438992074])

    risk = fit.predict("risk", newdata=_NEWDATA)
    assert risk.fit == pytest.approx([2.1687114065283661, 0.2934115798922754])
    assert risk.se_fit is None

    expected = fit.predict("expected")
    assert expected.fit[:3] == pytest.approx(
        [0.36485059448745627, 0.49056175217425002, 0.25025909008487457]
    )

    terms = fit.predict_terms(newdata=_NEWDATA, se_fit=True)
    assert isinstance(terms, regression.CoxTermsPrediction)
    np.testing.assert_allclose(
        terms.fit,
        [[0.93874712280551242, -0.16461395351151328], [-1.40812068420826786, 0.18194173809167263]],
    )
    np.testing.assert_allclose(
        terms.se_fit,
        [[0.79806470124652640, 0.70139752573499303], [1.19709705186978876, 0.77522884423341354]],
    )

    with pytest.raises(ValueError, match="type must be"):
        fit.predict("bogus")
    with pytest.raises(ValueError, match="follow-up time"):
        fit.predict("expected", newdata=_NEWDATA)


def test_coxph_fit_basehaz_and_survfit_match_r():
    fit = _tied_fit()

    basehaz = fit.basehaz()
    assert isinstance(basehaz, regression.Basehaz)
    assert basehaz.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert basehaz.hazard[:3] == pytest.approx(
        [0.055129234588335407, 0.245971097299985619, 0.245971097299985619]
    )
    assert basehaz.hazard[-1] == pytest.approx(8.435007882087591113)
    assert basehaz.strata is None
    assert fit.basehaz(centered=False).hazard[:2] == pytest.approx(
        [0.3250468662387635, 1.4502674481097746]
    )

    (curve,) = fit.survfit(newdata=_NEWDATA)
    assert isinstance(curve, regression.CoxSurvfitCurve)
    assert (curve.stratum, curve.n) == (0, 10)
    assert curve.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert curve.n_risk == pytest.approx([10, 9, 7, 6, 4, 3, 2, 1])
    assert curve.n_event == pytest.approx([1, 2, 0, 2, 0, 1, 0, 1])
    assert curve.n_censor == pytest.approx([0, 0, 1, 0, 1, 0, 1, 0])
    # survfit(fit, newdata)$surv, one column per newdata row
    assert [row[0] for row in curve.surv][:2] == pytest.approx(
        [0.88731130006455339, 0.58658345696823611]
    )
    assert [row[1] for row in curve.surv][:2] == pytest.approx(
        [0.98395456594223285, 0.93037200423029276]
    )
    assert [row[0] for row in curve.std_err][:2] == pytest.approx(
        [0.122851277492371569, 0.317907389155939857]
    )
    assert [row[1] for row in curve.cumhaz][:2] == pytest.approx(
        [0.016175555818815357, 0.072170768266625329]
    )


def test_coxph_fit_counting_process_strata_weights_offset_match_r():
    start = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 3.0, 0.0, 1.0]
    stop = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    x = [[value] for value in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.2, 0.8, 1.1, 0.4]]
    strata = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    weights = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0]
    offset = [0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0]

    fit = regression.coxph_fit(
        stop, status, x, entry=start, strata=strata, weights=weights, offset=offset
    )
    # coxph(Surv(start, stop, status) ~ x + strata(g) + offset(off), weights = w)
    assert fit.coefficients == pytest.approx([-0.78302485927205856])
    assert fit.var[0] == pytest.approx([2.2334762314262075])
    assert fit.loglik == pytest.approx([-6.5337342596707160, -6.3940130870789798])
    assert fit.means == pytest.approx([1.3])
    assert fit.nevent == 6
    assert fit.strata == strata
    assert fit.entry == pytest.approx(start)
    assert fit.linear_predictors[:3] == pytest.approx(
        [0.726419887417647070, 0.234907457781617646, 0.043395028145588543]
    )
    assert fit.martingale_residuals()[:3] == pytest.approx(
        [0.55023859220750193, -0.57974372141854302, 0.74846876019198305]
    )

    clustered = regression.coxph_fit(stop, status, x, entry=start, cluster=strata)
    # coxph(Surv(start, stop, status) ~ x + cluster(g)): robust sandwich variance
    assert clustered.coefficients == pytest.approx([0.48868151624787642])
    assert clustered.var[0] == pytest.approx([0.018306739858682395])
    assert clustered.naive_var[0] == pytest.approx([0.6478055354496437])
    assert clustered.rscore == pytest.approx(1.2162778262460221)

    exact = regression.coxph_fit(stop, status, x, entry=start, method="exact")
    # coxph(..., ties = "exact") on counting-process data (agexact)
    assert exact.coefficients == pytest.approx([0.56988220267373335])
    assert exact.var[0] == pytest.approx([0.70878875738958225])
    assert exact.loglik == pytest.approx([-6.0684255882441098, -5.8190977265322319])

    agexact = regression.agexact(start, stop, status, x)
    assert isinstance(agexact, regression.AgexactFit)
    assert agexact.coefficients == pytest.approx([0.56988220267373335])
    assert agexact.loglik == pytest.approx([-6.0684255882441098, -5.8190977265322319])
    assert agexact.var[0] == pytest.approx([0.70878875738958225])


def test_coxph_fit_validates_inputs():
    with pytest.raises(ValueError, match="status length mismatch"):
        regression.coxph_fit(_TIED_TIME, _TIED_STATUS[:-1], _TIED_X)
    with pytest.raises(ValueError, match="ties must be"):
        _tied_fit(method="bogus")
    with pytest.raises(ValueError, match="x has 9 rows"):
        regression.coxph_fit(_TIED_TIME, _TIED_STATUS, _TIED_X[:-1])


def test_coxph_detail_matches_r():
    fit = _tied_fit()
    detail = regression.coxph_detail(fit)

    # R: coxph.detail(fit) fields
    assert isinstance(detail, regression.CoxphDetail)
    assert detail.time == pytest.approx([1.0, 2.0, 4.0, 6.0, 8.0])
    assert detail.nevent == [1, 2, 2, 1, 1]
    assert detail.nrisk == [10, 9, 6, 3, 1]
    assert detail.hazard == pytest.approx(
        [
            0.055129234588335407,
            0.190841862711650212,
            0.524866033793095821,
            1.046074250800156813,
            6.618096500194352139,
        ]
    )
    assert detail.varhaz[:2] == pytest.approx([0.0030392325062957165, 0.0183590433439632765])
    assert detail.means[0] == pytest.approx([0.36755483164157399, 0.28608249126465790])
    assert detail.score[1] == pytest.approx([0.033962570685192539, 0.104936434757511715])
    np.testing.assert_allclose(
        detail.imat[0],
        [
            [0.174591449272772953, 0.102190627784997268],
            [0.102190627784997268, 0.090067315124623792],
        ],
    )
    assert detail.wtrisk[:2] == pytest.approx([18.13919615367898430, 11.52109965348462595])
    assert detail.nevent_wt == pytest.approx([1.0, 2.0, 2.0, 1.0, 1.0])
    assert detail.strata is None
    assert detail.riskmat is None

    with_riskmat = regression.coxph_detail(fit, riskmat=True)
    assert len(with_riskmat.riskmat) == 10
    assert sum(with_riskmat.riskmat[0]) == 1  # the first subject is only at risk at time 1


def test_cox_zph_matches_r():
    fit = _tied_fit()
    zph = regression.cox_zph(fit)

    # cox.zph(fit)$table and the scaled Schoenfeld residuals
    assert isinstance(zph, regression.CoxZph)
    assert zph.transform == "km"
    assert [row.chisq for row in zph.table] == pytest.approx(
        [0.15195475713205253, 0.14087118516631569]
    )
    assert [row.df for row in zph.table] == [1, 1]
    assert [row.p for row in zph.table] == pytest.approx([0.69667427791706926, 0.70741646786387680])
    assert isinstance(zph.global_test, regression.CoxZphTest)
    assert (zph.global_test.chisq, zph.global_test.df, zph.global_test.p) == pytest.approx(
        (1.54291722545947430, 2, 0.46233820385834629)
    )
    assert zph.time == pytest.approx([1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 8.0])
    assert zph.x == pytest.approx(
        [0.0, 0.1, 0.1, 0.3, 0.3, 0.533333333333333326, 0.688888888888888884]
    )
    assert zph.y[0] == pytest.approx([-10.10813719067957273, 7.51931450020224101])
    assert zph.strata is None

    identity = regression.cox_zph(fit, transform="identity", terms=False, global_test=False)
    assert identity.transform == "identity"
    assert [row.chisq for row in identity.table] == pytest.approx(
        [0.20965874456275058, 0.11082021786863967]
    )
    assert [row.p for row in identity.table] == pytest.approx(
        [0.64703500805248726, 0.73921225486361186]
    )
    assert identity.global_test is None


def test_coxph_wtest_matches_r():
    fit = _tied_fit()
    result = regression.coxph_wtest(fit.var, [fit.coefficients])

    # coxph.wtest(fit$var, coef(fit))
    assert isinstance(result, regression.CoxphWtest)
    assert result.test == pytest.approx([3.284067646975064])
    assert result.df == 2
    assert [row[0] for row in result.solve] == pytest.approx(
        [-1.6522473815825811, -1.0276072039183648]
    )

    breslow = _tied_fit(method="breslow")
    assert regression.coxph_wtest(breslow.var, [breslow.coefficients]).test == pytest.approx(
        [3.2051239607042534]
    )


def _aft_fit(distribution="weibull", **data_kwargs):
    return regression.survreg_fit(
        regression.SurvregData(_AFT_TIME, _AFT_STATUS, _AFT_DESIGN, **data_kwargs),
        regression.SurvregDistribution(distribution),
    )


def test_survreg_fit_matches_r():
    fit = _aft_fit()

    # survreg(Surv(t, s) ~ x, d)
    assert isinstance(fit, regression.SurvregFit)
    assert fit.coefficients == pytest.approx(
        [1.1636656987433951, 1.3788867351883287, math.log(0.497638477465639)]
    )
    assert fit.scale == pytest.approx([0.497638477465639])
    assert fit.log_likelihood == pytest.approx(-14.307025676163946)
    assert fit.intercept_only_log_likelihood == pytest.approx(-15.423608565500084)
    np.testing.assert_allclose(
        fit.variance_matrix,
        [
            [0.184525152366485423, -0.380847439814775035, -0.040806686884472650],
            [-0.380847439814775035, 1.018571428722077066, 0.083390468128268641],
            [-0.040806686884472650, 0.083390468128268641, 0.115733388634409073],
        ],
    )
    assert fit.naive_variance_matrix is None
    assert fit.icoef == pytest.approx([1.76106531439882441, -0.63333639546646736])
    assert (fit.iterations, fit.df, fit.df_residual, fit.n) == (7, 3, 5, 8)
    assert fit.converged
    assert fit.means == pytest.approx([1.0, 0.4875])
    assert fit.linear_predictors[:2] == pytest.approx([1.8531090663375596, 1.4394430457810610])
    assert fit.distribution.name == "Weibull"
    assert fit.distribution.family == regression.SurvregFamily.ExtremeValue
    assert fit.distribution.transform == regression.SurvregTransform.Log
    assert max(abs(value) for value in fit.score) < 1e-8


def test_survreg_fit_low_level_wrapper_matches_survreg_fit():
    legacy = regression.survreg(
        time=_AFT_TIME,
        status=[float(value) for value in _AFT_STATUS],
        covariates=_AFT_DESIGN,
        distribution="weibull",
    )
    fit = _aft_fit()
    assert legacy.coefficients == pytest.approx(fit.coefficients)
    assert legacy.log_likelihood == pytest.approx(fit.log_likelihood)


def test_survreg_fit_predictions_match_r():
    fit = _aft_fit()
    newdata = [[1.0, 0.25], [1.0, 0.75]]

    lp = fit.predict(newdata=newdata, predict_type="lp", se_fit=True)
    assert isinstance(lp, regression.SurvregPrediction)
    assert lp.predict_type == regression.SurvregPredictType.Lp
    assert [row[0] for row in lp.fit] == pytest.approx([1.5083873825404772, 2.1978307501346417])
    assert [row[0] for row in lp.se_fit] == pytest.approx(
        [0.24033756833717804, 0.43150946837872656]
    )

    quantile = fit.predict(newdata=newdata, predict_type="quantile", p=[0.1, 0.5], se_fit=True)
    np.testing.assert_allclose(
        quantile.fit,
        [[1.4747935472633338, 3.7659360558805890], [2.9386825822052023, 7.5040270644327540]],
    )
    np.testing.assert_allclose(
        quantile.se_fit,
        [[0.73385962335949151, 0.98846239504509181], [1.56237123886765339, 3.20239577005181753]],
    )

    uquantile = fit.predict(newdata=newdata, predict_type="uquantile", p=[0.5])
    assert [row[0] for row in uquantile.fit] == pytest.approx(
        [1.3259964507707331, 2.0154398183648974]
    )

    response = fit.predict(predict_type="response", se_fit=True)
    assert [row[0] for row in response.fit][:2] == pytest.approx(
        [6.3796233932137438, 4.2183457371586082]
    )
    assert [row[0] for row in response.se_fit][:2] == pytest.approx(
        [1.5406568557700573, 1.1391801296374089]
    )

    with pytest.raises(ValueError, match="prediction type 'bogus'"):
        fit.predict(predict_type="bogus")
    with pytest.raises(ValueError, match="probabilities between 0 and 1"):
        fit.predict(newdata=newdata, predict_type="quantile", p=[1.5])
    with pytest.raises(ValueError, match="newdata row 0 length mismatch"):
        fit.predict(newdata=[[1.0]], predict_type="lp")


def test_survreg_fit_residuals_match_r():
    fit = _aft_fit()

    def column(residual_type):
        result = fit.residuals(residual_type=residual_type)
        assert isinstance(result, survival.residuals.SurvregResiduals)
        assert result.residual_type == getattr(
            regression.SurvregResidType, residual_type.capitalize()
        )
        return [row[0] for row in result.values]

    assert column("response")[:3] == pytest.approx(
        [-5.37962339321374383, -2.21834573715860817, -8.07470590153972267]
    )
    assert column("deviance")[:3] == pytest.approx(
        [-2.34433263343717346, -1.20239552918718373, 0.38072585863811953]
    )
    assert column("working")[:3] == pytest.approx(
        [-20.115412566486959633, -1.731897200161101047, 0.497638477465638940]
    )
    assert column("ldcase")[:2] == pytest.approx([1.0179850595745871811, 0.1932737001492679518])
    assert column("ldresp")[:2] == pytest.approx([0.1314999730545083345, 0.1817180514737036234])
    assert column("ldshape")[:2] == pytest.approx([2.075233179897426172, 0.865101372213592956])

    np.testing.assert_allclose(
        fit.residuals(residual_type="dfbeta").values[:2],
        [
            [-0.095914018114840099, -0.032221968157953827, 0.303088449952158834],
            [-0.175870407208253726, 0.290253708135824240, 0.056753379002578902],
        ],
    )
    np.testing.assert_allclose(
        fit.residuals(residual_type="dfbetas").values[:1],
        [[-0.223282300929045940, -0.031926868183771190, 0.890922756312198971]],
    )
    matrix = fit.residuals(residual_type="matrix").values
    assert len(matrix) == 8
    np.testing.assert_allclose(
        matrix[0],
        [
            -3.050066333649101846,
            -1.960977962021287935,
            -0.097486340662400917,
            2.633906040309799579,
            -3.968675422108914219,
            2.141630783746856004,
        ],
    )

    with pytest.raises(ValueError, match="residual type 'bogus'"):
        fit.residuals(residual_type="bogus")


def test_survreg_fit_interval_strata_weights_and_cluster_match_r():
    # survreg(Surv(t1, t2, type = "interval2") ~ x + strata(g), weights = w, dist = "lognormal")
    # with R's interval status coding 0 right / 1 exact / 2 left / 3 interval
    fit = regression.survreg_fit(
        regression.SurvregData(
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 7.0, 2.0],
            [3, 1, 2, 3, 1, 0, 1, 3],
            _AFT_DESIGN,
            time2=[2.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0, 4.0],
            weights=[1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            strata=[0, 0, 0, 0, 1, 1, 1, 1],
        ),
        regression.SurvregDistribution("lognormal"),
    )
    assert fit.coefficients[:2] == pytest.approx([0.79486508969647773, 1.30538829240094301])
    assert fit.scale == pytest.approx([0.86326971587297341, 0.08607663656883309])
    assert fit.log_likelihood == pytest.approx(-11.362428108182179)
    assert fit.variance_matrix[0] == pytest.approx(
        [
            0.0240311839313740692,
            -0.0364926896367259446,
            -0.0015813742715430944,
            0.0234594693742770921,
        ]
    )

    clustered = _aft_fit(cluster=[1, 1, 2, 2, 3, 3, 4, 4])
    # survreg(Surv(t, s) ~ x + cluster(g)): robust variance keeps the naive one alongside
    assert clustered.coefficients[:2] == pytest.approx([1.1636656987433951, 1.3788867351883287])
    assert clustered.variance_matrix[0] == pytest.approx(
        [0.13301265160388523, -0.23361217142580540, -0.14495353454168899]
    )
    assert clustered.naive_variance_matrix[0] == pytest.approx(
        [0.184525152366485423, -0.380847439814775035, -0.040806686884472650]
    )


def test_survreg_distribution_helpers_match_r():
    weibull = regression.SurvregDistribution("weibull")
    assert (weibull.name, weibull.scale, weibull.parms) == ("Weibull", None, [])
    assert weibull.dtest() == []
    assert regression.survreg_dtest(weibull) == []
    assert weibull.variance() == pytest.approx(math.pi**2 / 6)

    exponential = regression.SurvregDistribution("exp")  # R's match.arg partial matching
    assert exponential.name == "Exponential"
    assert exponential.scale == pytest.approx(1.0)

    with pytest.raises(ValueError, match="'normal' should be one of"):
        regression.SurvregDistribution("normal")

    # dsurvreg / psurvreg / qsurvreg with R's recycling of mean and scale
    assert regression.dsurvreg([1.0, 2.0, 5.0], [1.5], [0.5]) == pytest.approx(
        [0.094738019355815842, 0.163187748073130218, 0.143403690391142008]
    )
    assert regression.psurvreg([1.0, 2.0, 5.0], [1.5], [0.5]) == pytest.approx(
        [0.048568007099546562, 0.180571615166323918, 0.711965988172734487]
    )
    assert regression.qsurvreg([0.1, 0.5, 0.9], [1.5], [0.5]) == pytest.approx(
        [1.4547242101138491, 3.7312509012850112, 6.8006365807998757]
    )
    assert regression.qsurvreg([0.1, 0.5, 0.9], [1.5], [0.5], distribution="lognormal") == (
        pytest.approx([2.3613281052886124, 4.4816890703380645, 8.5060339044805140])
    )
    assert regression.dsurvreg([1.0, 2.0], [1.5], [0.5], distribution="t", parms=[4]) == (
        pytest.approx([0.42932505167995955, 0.42932505167995955])
    )
    draws = regression.rsurvreg(5, [1.5], [0.5], seed=7)
    assert len(draws) == 5
    assert draws == regression.rsurvreg(5, [1.5], [0.5], seed=7)


def test_survreg_fit_validates_inputs():
    with pytest.raises(ValueError, match="weights must contain positive values"):
        _aft_fit(weights=[0.0] + [1.0] * 7)
    with pytest.raises(ValueError, match="status length mismatch"):
        regression.SurvregData(_AFT_TIME, _AFT_STATUS[:-1], _AFT_DESIGN)
    control = regression.SurvregControl(iter_max=1)
    limited = regression.survreg_fit(
        regression.SurvregData(_AFT_TIME, _AFT_STATUS, _AFT_DESIGN),
        regression.SurvregDistribution("weibull"),
        control=control,
    )
    assert limited.iterations == 1
    assert not limited.converged


def test_spline_config_validates_public_inputs():
    config = survival.regression.SplineConfig(3, 3, " Uniform ", (1.0, 10.0))
    assert config.knot_placement == "equal"

    with pytest.raises(ValueError, match="n_knots"):
        survival.regression.SplineConfig(0, 3, "quantile", None)
    with pytest.raises(ValueError, match="degree"):
        survival.regression.SplineConfig(3, 0, "quantile", None)
    with pytest.raises(ValueError, match="knot_placement"):
        survival.regression.SplineConfig(3, 3, "unknown", None)
    with pytest.raises(ValueError, match="boundary_knots"):
        survival.regression.SplineConfig(3, 3, "quantile", (0.0, 10.0))
    with pytest.raises(ValueError, match="boundary_knots"):
        survival.regression.SplineConfig(3, 3, "quantile", (10.0, 10.0))


def test_flexible_parametric_model_revalidates_mutated_spline_config():
    config = survival.regression.SplineConfig(3, 3, "quantile", None)
    config.knot_placement = "unknown"
    time = [float(value) for value in range(1, 21)]
    event = [1 if idx % 3 == 0 else 0 for idx in range(20)]
    covariates = [[idx * 0.1] for idx in range(20)]

    with pytest.raises(ValueError, match="knot_placement"):
        survival.regression.flexible_parametric_model(time, event, covariates, config)


def test_restricted_cubic_spline_validates_public_inputs():
    x = [float(value) for value in range(1, 6)]

    with pytest.raises(ValueError, match="x must contain only finite"):
        survival.regression.restricted_cubic_spline([1.0, 2.0, float("nan"), 4.0, 5.0], 4, None)
    with pytest.raises(ValueError, match="n_knots"):
        survival.regression.restricted_cubic_spline(x, 2, None)
    with pytest.raises(ValueError, match="knots must contain only finite"):
        survival.regression.restricted_cubic_spline(x, None, [1.0, 2.0, float("inf")])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.regression.restricted_cubic_spline(x, None, [1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.regression.restricted_cubic_spline([1.0] * 5, 4, None)


def test_predict_hazard_spline_validates_public_inputs():
    time = [float(value) for value in range(1, 21)]
    event = [1 if idx % 3 == 0 else 0 for idx in range(20)]
    covariates = [[idx * 0.1] for idx in range(20)]
    config = survival.regression.SplineConfig(3, 3, "quantile", None)
    model = survival.regression.flexible_parametric_model(time, event, covariates, config)

    with pytest.raises(ValueError, match="eval_times"):
        survival.regression.predict_hazard_spline(model, [], [0.5])
    with pytest.raises(ValueError, match="eval_times must contain only finite"):
        survival.regression.predict_hazard_spline(model, [1.0, float("nan")], [0.5])
    with pytest.raises(ValueError, match="non-negative"):
        survival.regression.predict_hazard_spline(model, [-1.0, 2.0], [0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        survival.regression.predict_hazard_spline(model, [1.0, 1.0], [0.5])
    with pytest.raises(ValueError, match="covariate_values length"):
        survival.regression.predict_hazard_spline(model, [1.0, 2.0], [0.5, 1.0])
    with pytest.raises(ValueError, match="covariate_values must contain only finite"):
        survival.regression.predict_hazard_spline(model, [1.0, 2.0], [float("nan")])

    bad_coefficients = survival.regression.FlexibleParametricResult(
        [float("nan")],
        model.spline_coefficients,
        model.std_errors,
        model.knots,
        model.log_likelihood,
        model.aic,
        model.bic,
        model.n_iterations,
        model.converged,
    )
    with pytest.raises(ValueError, match="coefficients must contain only finite"):
        survival.regression.predict_hazard_spline(bad_coefficients, [1.0, 2.0], [0.5])

    bad_spline_coefficients = survival.regression.FlexibleParametricResult(
        model.coefficients,
        model.spline_coefficients[:-1],
        model.std_errors,
        model.knots,
        model.log_likelihood,
        model.aic,
        model.bic,
        model.n_iterations,
        model.converged,
    )
    with pytest.raises(ValueError, match="spline_coefficients length"):
        survival.regression.predict_hazard_spline(bad_spline_coefficients, [1.0, 2.0], [0.5])


def test_aareg_public_api():
    options = survival.regression.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        variable_names=["time", "x1"],
        max_iter=20,
    )

    result = survival.aareg(options)

    assert len(result.coefficients) == 2
    assert len(result.standard_errors) == 2
    assert len(result.confidence_intervals) == 2
    assert len(result.p_values) == 2
    assert result.fit_details is not None
    assert result.fit_details.iterations <= 20
    assert result.fit_details.converged is True
    assert len(result.residuals) == 4
    assert math.isfinite(result.goodness_of_fit)

    weighted_subset = survival.regression.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        variable_names=["time", "x1"],
        max_iter=20,
    )
    weighted_subset.subset = [0, 1, 2]
    weighted_subset.weights = [1.0, 2.0, 1.0, 99.0]
    weighted_result = survival.aareg(weighted_subset)

    assert len(weighted_result.residuals) == 3


def test_aareg_rejects_invalid_formula():
    options = survival.regression.AaregOptions(
        formula="time",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
        max_iter=5,
    )

    with pytest.raises(ValueError, match="Formula Error"):
        survival.aareg(options)


def test_aareg_validates_public_inputs():
    with pytest.raises(ValueError, match="data cannot be empty"):
        survival.aareg(
            survival.regression.AaregOptions(
                formula="time ~ x1",
                data=[],
                variable_names=["time", "x1"],
            )
        )

    with pytest.raises(ValueError, match="data row 1 has 1 columns"):
        survival.aareg(
            survival.regression.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, 2.0], [2.0]],
                variable_names=["time", "x1"],
            )
        )

    with pytest.raises(ValueError, match="variable_names length"):
        survival.aareg(
            survival.regression.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, 2.0], [2.0, 3.0]],
                variable_names=["time"],
            )
        )

    with pytest.raises(ValueError, match="data contains non-finite"):
        survival.aareg(
            survival.regression.AaregOptions(
                formula="time ~ x1",
                data=[[1.0, float("inf")], [2.0, 3.0]],
                variable_names=["time", "x1"],
            )
        )

    missing = survival.regression.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, float("nan")], [2.0, 3.0], [3.0, 4.0]],
        variable_names=["time", "x1"],
    )
    with pytest.raises(ValueError, match="missing values in data"):
        survival.aareg(missing)

    missing.na_action = "Exclude"
    excluded = survival.aareg(missing)
    assert len(excluded.residuals) == 2

    bad_weights = survival.regression.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
    )
    bad_weights.weights = [1.0, float("inf")]
    with pytest.raises(ValueError, match="weights contains non-finite"):
        survival.aareg(bad_weights)

    bad_iter = survival.regression.AaregOptions(
        formula="time ~ x1",
        data=[[1.0, 2.0], [2.0, 3.0]],
        variable_names=["time", "x1"],
        max_iter=0,
    )
    with pytest.raises(ValueError, match="max_iter must be positive"):
        survival.aareg(bad_iter)


def test_recurrent_event_regression_validates_public_inputs():
    with pytest.raises(ValueError, match="x length"):
        survival.recurrent.gap_time_model(
            [0, 1], [0.0, 0.0], [1.0, 1.0], [1, 0], [0.5], 2, 1, 10, 1e-6
        )
    with pytest.raises(ValueError, match="stop_time"):
        survival.recurrent.gap_time_model([0], [1.0], [1.0], [1], [0.5], 1, 1, 10, 1e-6)
    with pytest.raises(ValueError, match="event_status"):
        survival.recurrent.pwp_gap_time([0], [1.0], [2], [0.5], 1, 1, False)
    with pytest.raises(ValueError, match="max_iter"):
        survival.recurrent.gap_time_model([0], [0.0], [1.0], [1], [0.5], 1, 1, 0, 1e-6)

    gap = survival.recurrent.gap_time_model(
        [10, 10, 42],
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 0, 1],
        [1.0, 0.5, 0.0],
        3,
        1,
        20,
        1e-4,
    )
    assert gap.n_subjects == 2

    method = survival.recurrent.MarginalMethod("andersen_gill")
    with pytest.raises(ValueError, match="x length"):
        survival.recurrent.marginal_recurrent_model(
            [0, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            2,
            1,
            method,
            10,
            1e-6,
        )
    with pytest.raises(ValueError, match="event_status"):
        survival.recurrent.wei_lin_weissfeld([0], [1.0], [2], [0.5], 1, 1)

    marginal = survival.recurrent.marginal_recurrent_model(
        [10, 10, 42],
        [0.0, 2.0, 0.0],
        [2.0, 5.0, 3.0],
        [1, 0, 1],
        [1.0, 0.5, 0.0],
        3,
        1,
        method,
        20,
        1e-4,
    )
    assert marginal.n_subjects == 2

    with pytest.raises(ValueError, match="covariates length"):
        survival.regression.anderson_gill_model(
            [1, 2],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            10,
            1e-6,
        )

    pwp_config = survival.regression.PWPConfig(
        survival.regression.PWPTimescale("gap"), 10, 1e-6, True, True
    )
    with pytest.raises(ValueError, match="stop must be greater than start"):
        survival.regression.pwp_model([1], [0.0], [0.0], [1], [1], [], pwp_config)
    with pytest.raises(ValueError, match="event_number"):
        survival.regression.pwp_model([1], [0.0], [1.0], [1], [0], [], pwp_config)

    wlw_config = survival.regression.WLWConfig(10, 1e-6, True, False)
    with pytest.raises(ValueError, match="event must contain only 0/1"):
        survival.regression.wlw_model([1], [1.0], [2], [1], [], wlw_config)

    bad_wlw_config = survival.regression.WLWConfig(0, 1e-6, True, False)
    with pytest.raises(ValueError, match="max_iter"):
        survival.regression.wlw_model([1], [1.0], [1], [1], [], bad_wlw_config)

    nb_config = survival.regression.NegativeBinomialFrailtyConfig(10, 1e-6, 10)
    with pytest.raises(ValueError, match="same length"):
        survival.regression.negative_binomial_frailty(
            [1, 2], [1.0, 1.0], [1, 0], [], [0.0], nb_config
        )
    with pytest.raises(ValueError, match="event counts"):
        survival.regression.negative_binomial_frailty([1], [1.0], [-1], [], None, nb_config)

    bad_nb_config = survival.regression.NegativeBinomialFrailtyConfig(10, 1e-6, 0)
    with pytest.raises(ValueError, match="em_max_iter"):
        survival.regression.negative_binomial_frailty([1], [1.0], [1], [], None, bad_nb_config)

    with pytest.raises(ValueError, match="x length"):
        survival.recurrent.joint_frailty_model(
            [0, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [1, 0],
            [0.5],
            2,
            1,
            [1.0, 1.0],
            [1, 0],
            [0.5, 0.2],
            2,
            1,
        )
    with pytest.raises(ValueError, match="subject_id values"):
        survival.recurrent.joint_frailty_model(
            [2],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0, 1.0],
            [1, 0],
            [0.5, 0.2],
            2,
            1,
        )
    with pytest.raises(ValueError, match="term_status"):
        survival.recurrent.joint_frailty_model(
            [0],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0],
            [2],
            [0.5],
            1,
            1,
        )
    with pytest.raises(ValueError, match="max_iter"):
        survival.recurrent.joint_frailty_model(
            [0],
            [0.0],
            [1.0],
            [1],
            [0.5],
            1,
            1,
            [1.0],
            [1],
            [0.5],
            1,
            1,
            survival.recurrent.FrailtyDistribution("gamma"),
            0,
            1e-4,
        )
