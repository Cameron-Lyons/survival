"""Post-fit methods of penalized and null Cox models, against R 4.5.3 with survival 3.8-12.

``summary`` of a penalized fit is summary.coxph.penal; ``anova`` counts a penalized fit's
``sum(fit$df)`` and, as R's anova.coxph does (anova.coxph.penal is not registered), refits
the leading terms of one model unpenalized; a sparse frailty has its column in
``predict(type="terms")``; a null model has R's logLik, AIC and residual methods; and the
chi-square tails come from the nmath ``pchisq``.
"""

import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12, nan_ok=True)


def _complete(data, column):
    keep = [value is not None for value in data[column]]
    return {
        key: [v for v, k in zip(values, keep, strict=True) if k] for key, values in data.items()
    }


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def lung_inst(lung):
    """``lung[!is.na(lung$inst), ]``: R sizes a frailty's df search from the data before
    na.omit, so the reference fits use data without missing values."""

    return _complete(lung, "inst")


@pytest.fixture(scope="module")
def lung_ecog(lung):
    """``lung[!is.na(lung$ph.ecog), ]`` (ridge() scales by variances taken before na.omit)."""

    return _complete(lung, "ph.ecog")


@pytest.fixture(scope="module")
def kidney():
    return datasets.load_kidney()


def _fit(formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return r.coxph(f"Surv(time, status) ~ {formula}", data)


def _table(result):
    rows = list(result.rows)
    return (
        [row.name for row in rows],
        [row.loglik for row in rows],
        [math.nan if row.chisq is None else row.chisq for row in rows],
        [math.nan if row.df is None else row.df for row in rows],
        [math.nan if row.p_value is None else row.p_value for row in rows],
    )


def _columns(summary):
    rows = summary["coefficients"]
    return [row["name"] for row in rows], [
        [row[key] for row in rows] for key in ("coef", "se", "se2", "chisq", "df", "p")
    ]


def test_pchisq_is_r_nmath_pchisq():
    pchisq = survival.validation.pchisq
    # R: pchisq(3.6432229393649322, 3.0886431277535298, lower.tail = FALSE)
    assert pchisq(3.6432229393649322, 3.0886431277535298, lower_tail=False) == approx(
        0.31611266250077846, rel=1e-14
    )
    assert pchisq(2.9406959209324954, 3.092186074265781, False, True) == approx(
        -0.87504842297161112, rel=1e-14
    )
    assert pchisq(0.5, 0.25) == approx(0.8696646545502863, rel=1e-14)
    assert math.isnan(pchisq(4.05, -6.9, lower_tail=False))
    assert math.isnan(pchisq(math.nan, 1.0))
    assert pchisq(1.0, 0.0, lower_tail=False) == 0.0
    assert pchisq(0.0, 0.0, lower_tail=False) == 1.0
    assert pchisq(-1.0, 2.0) == 0.0


def test_anova_of_model_lists_counts_the_penalized_df(lung):
    linear, spline = _fit("age + sex", lung), _fit("pspline(age, df=4) + sex", lung)
    names, loglik, chisq, df, p = _table(r.anova(linear, spline))
    assert names == ["1", "2"]
    assert loglik == approx([-742.84824578377038, -741.02663431408791])
    assert chisq[1] == approx(3.6432229393649322)
    assert df[1] == approx(3.0886431277535298)
    assert p[1] == approx(0.31611266250077846)

    _, loglik, chisq, df, p = _table(r.anova(_fit("ridge(age, theta=1) + sex", lung), linear))
    assert loglik == approx([-742.84832835847430, -742.84824578377038])
    assert df[1] == approx(0.0069725687156259042)
    assert p[1] == approx(0.030306562917279255)

    _, loglik, chisq, df, p = _table(
        r.anova(_fit("sex", lung), _fit("sex + pspline(age, df=4)", lung))
    )
    assert chisq[1] == approx(7.1327294217333019)
    assert df[1] == approx(4.0886431277534676)
    assert p[1] == approx(0.13551148995915155)


def test_anova_of_one_penalized_model_refits_its_terms_unpenalized(lung):
    names, loglik, chisq, df, p = _table(r.anova(_fit("ridge(age, sex, theta=1)", lung)))
    assert names == ["NULL", "ridge(age, sex, theta=1)"]
    assert loglik == approx([-749.9098013903947, -742.8485237040278])
    assert chisq[1] == approx(14.122555372733814)
    assert df[1] == approx(1.9863723944321767)
    assert p[1] == approx(0.00084225868726461655)

    # the pspline basis is refitted as 12 plain columns; the sex row then loses df
    names, loglik, chisq, df, p = _table(r.anova(_fit("pspline(age, df=4) + sex", lung)))
    assert names == ["NULL", "pspline(age, df=4)", "sex"]
    assert loglik == approx([-749.90980139039470, -743.05294395802741, -741.02663431408791])
    assert chisq[1:] == approx([13.7137148647345839, 4.0526192878789971])
    assert df[1:] == approx([12.0, -6.9113568722464702])
    assert p[1] == approx(0.31936232295178951)
    assert math.isnan(p[2])

    names, loglik, chisq, df, p = _table(r.anova(_fit("sex + pspline(age, df=4)", lung)))
    assert loglik == approx([-749.90980139039470, -744.59299902495457, -741.02663431408791])
    assert df[1:] == approx([1.0, 4.0886431277534676])
    assert p[1:] == approx([0.0011105103779009628, 0.1355114899591515487])


def test_anova_rows_of_a_frailty(lung_inst, lung_ecog, kidney):
    # a sparse frailty last: the full model's row absorbs it
    names, loglik, chisq, df, p = _table(r.anova(_fit("age + frailty(inst)", lung_inst)))
    assert names == ["NULL", "age", "frailty(inst)"]
    assert loglik == approx([-744.79993370264594, -742.70538003012962, -742.70531888212656])
    assert df[1:] == approx([1.0, 7.3832232073023363e-05])
    assert p[1:] == approx([0.04068451432977959054, 0.00033680611962798372])

    # in the middle it is refitted from the group codes of R's model matrix
    names, loglik, chisq, df, p = _table(r.anova(_fit("age + frailty(inst) + sex", lung_inst)))
    assert names == ["NULL", "age", "frailty(inst)", "sex"]
    assert loglik == approx(
        [-744.79993370264594, -742.70538003012962, -742.11137013208190, -737.81086073847212]
    )
    assert df[1:] == approx([1.0, 1.0, 7.3349093505648e-05])
    assert p[3] == approx(9.6509724895308492e-08)

    names, loglik, chisq, df, p = _table(r.anova(_fit("frailty(id)", kidney)))
    assert names == ["NULL", "frailty(id)"]
    assert loglik == approx([-187.90276158886371, -178.98686446619644])
    assert df[1] == approx(7.5518147173535981)
    assert p[1] == approx(0.017612824973733617)

    names, loglik, chisq, df, p = _table(
        r.anova(_fit("age + frailty(ph.ecog, sparse=False)", lung_ecog))
    )
    assert loglik == approx([-744.48045576144023, -742.3175214459211, -735.68179282968492])
    assert df[1:] == approx([1.0, 1.6648874602441985])
    assert p[1:] == approx([0.037537250724902695, 0.00082773172903428916])


def test_summary_of_a_pspline_fit_splits_linear_and_nonlinear(lung):
    fit = _fit("pspline(age, df=4) + sex", lung)
    summary = r.model_summary(fit, conf_int=0.9, scale=2)
    assert summary["model_type"] == "coxph.penal"
    assert summary["coefficient_columns"] == ["coef", "se(coef)", "se2", "Chisq", "DF", "p"]
    names, (coef, se, se2, chisq, df, p) = _columns(summary)
    assert names == ["pspline(age, df=4), linear", "pspline(age, df=4), nonlin", "sex"]
    assert coef == approx([0.016899678546633299, math.nan, -0.51821168761417724])
    assert se == approx([0.0090327919259855904, math.nan, 0.16842538712078162])
    assert se2 == approx([0.0090323624432969568, math.nan, 0.16812676131333218])
    assert chisq == approx([3.5003613135858807, 2.9406959209324954, 9.4667149194842661])
    assert df == approx([1.0, 3.092186074265781, 1.0])
    assert p == approx([0.061355441791301049, 0.41684183446948153, 0.0020923373437396484])
    assert summary["print2"] == ["Theta= 0.8092459"]
    logtest = summary["logtest"]
    assert [logtest["test"], logtest["df"], logtest["pvalue"]] == approx(
        [17.766334152613581, 5.0886431277535298, 0.0034911276990234039]
    )
    assert summary["iter"] == [4, 11]
    assert summary["df"] == approx([4.092186074265781, 0.996457053487749])
    assert [summary["n"], summary["nevent"]] == [228, 165]
    assert summary["concordance"] == approx({"C": 0.5990556610372739, "se(C)": 0.0256990004118581})
    for key in ("sctest", "waldtest", "rsq"):
        assert key not in summary
    conf = summary["conf_int"]
    assert [row["name"] for row in conf] == [f"ps(age){j}" for j in range(3, 15)] + ["sex"]
    assert [conf[0][key] for key in ("exp(coef)", "exp(-coef)", "lower", "upper")] == approx(
        [2.2903921066606721, 0.43660646449658447, 0.21449943680468256, 24.456455832237371]
    )
    assert [conf[-1][key] for key in ("exp(coef)", "exp(-coef)", "lower", "upper")] == approx(
        [0.35472112016701302, 2.8191160411569824, 0.20382498032113774, 0.61732900890898079]
    )
    assert fit.summary()["coefficients"] == r.model_summary(fit)["coefficients"]


def test_summary_of_frailty_fits(lung_inst, lung_ecog, kidney):
    summary = r.model_summary(_fit("age + frailty(inst, df=4)", lung_inst))
    names, (coef, se, se2, chisq, df, p) = _columns(summary)
    assert names == ["age", "frailty(inst, df=4)"]
    assert coef == approx([0.019367969408404077, math.nan])
    assert se2 == approx([0.0092541407835533499, math.nan])
    assert chisq == approx([4.3114635850672576, 3.3342518467451399])
    assert df == approx([1.0, 3.9867495786142921])
    assert p == approx([0.037856378786794223, 0.50145826465272547])
    assert summary["print2"] == ["Variance of random effect= 0.03794502   I-likelihood = -743.6"]
    assert summary["logtest"]["df"] == approx(4.9710524357490957)
    assert [row["name"] for row in summary["conf_int"]] == ["age"]

    summary = r.model_summary(_fit("age + frailty(inst, dist='gauss')", lung_inst))
    _, (_, _, _, chisq, df, p) = _columns(summary)
    assert chisq == approx([4.103233918079586, 0.15683551993110612])
    assert df == approx([1.0, 0.19052186031386142])
    assert p == approx([0.042801271481548803, 0.4251357699680971])
    assert summary["print2"] == ["Variance of random effect= 0.001302083"]

    # a dense frailty's test is coxph.wtest on its block of the variance
    summary = r.model_summary(_fit("age + frailty(ph.ecog, sparse=False)", lung_ecog))
    names, (_, _, se2, chisq, df, p) = _columns(summary)
    assert names == ["age", "frailty(ph.ecog, sparse=False)"]
    assert se2 == approx([0.0092838167905875523, math.nan])
    assert chisq == approx([1.9536402420389809, 10.932387579766241])
    assert df == approx([1.0, 1.6751540273442893])
    assert p == approx([0.16219511945173021, 0.0027813352116346376])
    assert summary["print2"] == ["Variance of random effect= 0.09365458   I-likelihood = -739.6"]
    assert [row["name"] for row in summary["conf_int"]] == ["age"] + [
        f"gamma:{level}" for level in range(4)
    ]
    assert summary["conf_int"][1]["exp(coef)"] == approx(0.6848777960028396)

    # a frailty alone: no coefficients, so no conf.int
    summary = r.model_summary(_fit("frailty(id)", kidney))
    names, (_, _, _, chisq, df, p) = _columns(summary)
    assert names == ["frailty(id)"]
    assert [chisq[0], df[0], p[0]] == approx(
        [10.416260706123129, 7.5518147173535981, 0.20383368450046599]
    )
    assert summary["print2"] == ["Variance of random effect= 0.1760715   I-likelihood = -187.7"]
    logtest = summary["logtest"]
    assert [logtest["test"], logtest["df"], logtest["pvalue"]] == approx(
        [17.831794245334549, 7.5518147173535981, 0.017612824973733617]
    )
    assert "conf_int" not in summary


def test_summary_of_ridge_fits_and_terms(lung_ecog):
    fit = _fit("ridge(age, ph.ecog, theta=1) + sex", lung_ecog)
    summary = r.model_summary(fit)
    names, (coef, se, se2, chisq, df, p) = _columns(summary)
    assert names == ["ridge(age)", "ridge(ph.ecog)", "sex"]
    assert coef == approx([0.011031697757053862, 0.46083111175087338, -0.55239541421339844])
    assert se == approx([0.0092326678445759838, 0.11318679065795338, 0.16773780638669011])
    assert se2 == approx([0.0091991781226002552, 0.11280072185334226, 0.16773648114492348])
    assert chisq == approx([1.4276780619728764, 16.576471501908458, 10.845216116600573])
    assert df == [1.0, 1.0, 1.0]
    assert p == approx([0.23214375406087454, 4.6727147274456751e-05, 0.00099051318622226585])
    assert summary["print2"] == []
    assert summary["logtest"]["df"] == approx(2.9863354762019778)

    # terms=TRUE: one Wald row for the multi-column ridge term, p on 1 df as R
    names, (coef, _, _, chisq, df, p) = _columns(r.model_summary(fit, terms=True))
    assert names == ["ridge(age, ph.ecog, theta=1)", "sex"]
    assert math.isnan(coef[0])
    assert chisq == approx([20.274274310834301, 10.845216116600573])
    assert df == approx([1.986351277488045, 1.0])
    assert p == approx([6.7096720375027704e-06, 0.00099051318622226585])

    fit = _fit("factor(ph.ecog) + ridge(age, theta=1)", lung_ecog)
    names, (_, _, se2, chisq, df, p) = _columns(r.model_summary(fit, terms=True))
    assert names == ["factor(ph.ecog)", "ridge(age)"]
    assert se2 == approx([math.nan, 0.0093254411779851772])
    assert chisq == approx([16.650829842797805, 1.3100109501850026])
    assert df == approx([2.9995284707807173, 1.0])
    assert p == approx([4.4930647735491849e-05, 0.25239267277898991])


def test_terms_predictions_carry_a_sparse_frailty(kidney):
    fit = _fit("age + sex + frailty(id)", kidney)
    assert r.model_term_names(fit) == ["age", "sex", "frailty(id)"]
    pred = r.predict(fit, type="terms", se_fit=True)
    assert len(pred.fit) == 76
    assert pred.fit[0] == approx([-0.0824536042289293, 1.169728429127768, 0.370387746942514])
    assert pred.fit[2] == approx([0.0226004430702933, -0.417760153259917, 0.256513807025264])
    assert [row[2] for row in pred.se_fit[:3]] == approx(
        [0.482224270265309, 0.482224270265309, 0.541118392353473]
    )
    assert r.predict(fit, type="terms", terms="frailty(id)")[2] == approx([0.256513807025264])
    assert r.predict(fit, type="terms", terms=[3, 1])[0] == approx(
        [0.370387746942514, -0.0824536042289293]
    )
    # new data get no frailty
    newdata = {key: values[:3] for key, values in kidney.items()}
    new = r.predict(fit, newdata, type="terms", se_fit=True)
    assert [row[2] for row in new.fit] == [0.0, 0.0, 0.0]
    assert [row[2] for row in new.se_fit] == [0.0, 0.0, 0.0]
    assert new.fit[0][:2] == approx([-0.0824536042289293, 1.169728429127768])

    # the frailty alone predicts the linear predictor, with the frailty's se (for the
    # risk too, as R)
    alone = _fit("frailty(id)", kidney)
    assert r.model_term_names(alone) == ["frailty(id)"]
    pred = r.predict(alone, type="terms", se_fit=True)
    assert pred.fit[:4] == approx(
        [0.2581914834224425, 0.2581914834224425, 0.0970654531403064, 0.0970654531403064]
    )
    assert pred.se_fit[:3] == approx([0.361078078326962, 0.361078078326962, 0.387287607839414])
    assert r.predict(alone, newdata, type="terms") == [0.0, 0.0, 0.0]
    lp = r.predict(alone, type="lp", se_fit=True)
    assert lp.fit[:2] == approx([0.258191483422443, 0.258191483422443])
    assert lp.se_fit[:2] == approx([0.361078078326962, 0.361078078326962])
    risk = r.predict(alone, type="risk", se_fit=True)
    assert risk.fit[:2] == approx([1.294586686781592, 1.294586686781592])
    assert risk.se_fit[:2] == approx([0.361078078326962, 0.361078078326962])
    risk = r.predict(alone, newdata, type="risk", se_fit=True)
    assert (risk.fit, risk.se_fit) == ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("formula", "value"),
    [
        ("1", -749.9098013903947),
        ("strata(sex)", -643.43701866939364),
        ("offset(age/100)", -748.24377722315035),
    ],
)
def test_null_model_loglik_and_information_criteria(lung, formula, value):
    fit = _fit(formula, lung)
    assert r.loglik(fit) == approx(value, rel=1e-12)
    assert r.degrees_freedom(fit) == 0
    assert r.aic(fit) == approx(-2 * value, rel=1e-12)
    assert r.aic(fit, k=3) == approx(-2 * value, rel=1e-12)
    assert r.bic(fit) == approx(-2 * value, rel=1e-12)
    assert r.extract_aic(fit) == approx([0.0, -2 * value], rel=1e-12)
    assert r.extract_aic(fit, k=5) == approx([0.0, -2 * value], rel=1e-12)


def test_null_model_residuals(lung):
    fit = _fit("1", lung)
    assert r.residuals(fit, type="deviance")[:3] == approx(
        [0.36635099102308333, -0.12292668139053035, -2.40481385447901452]
    )
    for kind in ("score", "schoenfeld", "dfbeta", "partial"):
        with pytest.raises(
            ValueError, match=f"'{kind}' residuals are not defined for a null model"
        ):
            r.residuals(fit, type=kind)


def test_frailty_alone_keeps_both_logliks(kidney):
    fit = _fit("frailty(id)", kidney)
    assert fit.loglik == approx([-187.90276158886371, -178.9868644661964368])
    assert r.aic(fit) == approx(373.0773583671000893)
    assert r.extract_aic(fit) == approx([7.5518147173535981, 373.0773583671000893])
