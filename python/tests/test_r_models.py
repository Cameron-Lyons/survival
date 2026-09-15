"""``survival.r`` model generics (coef, vcov, logLik, AIC, confint, summary, model.frame,
model.matrix, as_data_frame) and their dispatch on Cox / cch / aareg fits."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def ovarian():
    return datasets.load_ovarian()


@pytest.fixture(scope="module")
def fit(ovarian):
    return r.coxph("Surv(futime, fustat) ~ age + factor(rx)", ovarian)


def test_coefficient_accessors(fit):
    assert r.coef(fit) == approx([0.147326595469114, -0.803973012665114])
    assert r.coef_names(fit) == ["age", "factor(rx)2"]
    assert r.coef_names(fit, complete=False) == ["age", "factor(rx)2"]
    assert r.vcov(fit) == fit.var
    assert r.vcov(fit, complete=False) == fit.var
    assert r.loglik(fit) == approx(-27.0418988630084)
    assert r.nobs(fit) == 12  # the number of events, R's nobs attribute of logLik.coxph
    assert r.degrees_freedom(fit) == 2
    assert r.aic(fit) == approx(-2 * -27.0418988630084 + 4)
    assert r.bic(fit) == approx(-2 * -27.0418988630084 + 2 * math.log(12))
    assert r.extract_aic(fit) == approx([2.0, -2 * -27.0418988630084 + 4])
    assert r.extract_aic(fit, k=3) == approx([2.0, -2 * -27.0418988630084 + 6])
    with pytest.raises(TypeError, match="only defined for fitted survreg"):
        r.df_residual(fit)


def test_aliased_coefficients_are_nan(ovarian):
    duplicated = {**ovarian, "age2": ovarian["age"]}
    fit = r.coxph("Surv(futime, fustat) ~ age + age2 + rx", duplicated)
    assert math.isnan(r.coef(fit)[1])
    assert r.coef_names(fit, complete=False) == ["age", "rx"]
    assert len(r.vcov(fit, complete=False)) == 2
    assert r.degrees_freedom(fit) == 2
    with pytest.raises(ValueError, match="singular"):
        r.coxph("Surv(futime, fustat) ~ age + age2 + rx", duplicated, singular_ok=False)


def test_confint_and_summary(fit):
    intervals = r.confint(fit)
    assert [row["name"] for row in intervals] == ["age", "factor(rx)2"]
    z = 1.959963984540054
    assert intervals[0]["lower"] == approx(0.147326595469114 - z * 0.0461470477408026, rel=1e-6)
    assert intervals[0]["upper"] == approx(0.147326595469114 + z * 0.0461470477408026, rel=1e-6)
    assert [row["name"] for row in r.confint(fit, "age")] == ["age"]
    assert [row["name"] for row in r.confint(fit, [2])] == ["factor(rx)2"]
    assert r.confint(fit, level=0.9)[0]["upper"] < intervals[0]["upper"]
    with pytest.raises(ValueError, match="unknown coefficient name"):
        r.confint(fit, "sex")
    summary = r.model_summary(fit)
    assert summary["model_type"] == "coxph"
    assert summary["coefficient_names"] == ["age", "factor(rx)2"]


def test_model_frame_matrix_and_terms(ovarian, fit):
    matrix = r.model_matrix(fit)
    assert matrix["columns"] == ["age", "factor(rx)2"]
    assert matrix["assign"] == [1, 2]
    assert matrix["data"][0] == [72.3315, 0.0]
    assert len(matrix["data"]) == 26
    assert r.model_term_names(fit) == ["age", "factor(rx)"]
    assert r.model_term_names(fit, [2]) == ["factor(rx)"]
    assert r.model_formula(fit) == "Surv(futime, fustat) ~ age + factor(rx)"
    assert r.model_weights(fit) is None
    weighted = r.coxph("Surv(futime, fustat) ~ age", ovarian, weights=[1, 2] * 13)
    assert r.model_weights(weighted) == [1.0, 2.0] * 13
    with pytest.raises(TypeError, match="model=TRUE"):
        r.model_frame(fit)
    frame = r.model_frame(
        r.coxph("Surv(futime, fustat) ~ age + strata(rx)", ovarian, model=True, weights=[1, 2] * 13)
    )
    assert frame["time"][:2] == [59.0, 115.0]
    assert frame["status"][:2] == [1, 1]
    assert frame["(strata)"][:2] == ["rx=1", "rx=1"]
    assert frame["(weights)"][:2] == [1.0, 2.0]
    assert r.fitted(fit) == r.predict(fit)


def test_as_data_frame_shapes(ovarian, fit):
    curve = r.as_data_frame(r.survfit(fit))
    assert list(curve) == [
        "curve",
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "surv",
        "cumhaz",
        "std.err",
        "std.chaz",
        "lower",
        "upper",
    ]
    assert len(curve["time"]) == 26
    assert curve["curve"] == [1] * 26
    assert list(r.as_data_frame(r.basehaz(fit))) == ["hazard", "time"]
    zph = r.as_data_frame(r.cox_zph(fit))
    assert list(zph) == ["name", "chisq", "df", "p"]
    assert zph["name"][-1] == "GLOBAL"
    detail = r.as_data_frame(r.coxph_detail(fit))
    assert list(detail) == ["time", "nevent", "nrisk", "hazard", "varhaz", "cumhaz", "wtrisk"]
    anova = r.as_data_frame(r.anova(fit))
    assert anova["model"] == ["NULL", "age", "factor(rx)"]
    assert math.isnan(anova["Chisq"][0])
    concordance = r.as_data_frame(r.concordance(fit))
    assert list(concordance) == [
        "score",
        "concordance",
        "concordant",
        "discordant",
        "tied.x",
        "tied.y",
        "tied.xy",
        "n",
        "var",
    ]
    assert concordance["concordant"] == [174.0]
    assert concordance["n"] == [26]
    response = r.as_data_frame(fit.y)
    assert list(response) == ["time", "status", "type"]
    assert response["type"][0] == "right"
    with pytest.raises(TypeError, match="survival result object"):
        r.as_data_frame(object())


def test_generics_dispatch_to_survreg_or_fail_clearly():
    for generic in (r.coef, r.vcov, r.loglik, r.nobs, r.model_summary, r.residuals):
        with pytest.raises(TypeError, match="requires a fitted coxph or survreg model"):
            generic(object())
    with pytest.raises(TypeError, match="requires a fitted coxph or survreg model"):
        r.predict(object(), type="lp")
    with pytest.raises(TypeError, match="requires fitted coxph or survreg models"):
        r.anova(object())
