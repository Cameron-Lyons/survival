"""R's na.action semantics in the formula fitters and their methods.

Reference values come from R 4.5.3 with survival 3.8-12 on the bundled ``lung`` data
(one missing ``ph.ecog``, row 14), with R's default ``options(na.action = "na.omit")``.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets

NAN = math.nan


def approx(values, rel=1e-9):
    return pytest.approx(values, rel=rel, abs=1e-12, nan_ok=True)


def rows_approx(values, expected):
    assert len(values) == len(expected)
    for row, expected_row in zip(values, expected, strict=True):
        assert row == approx(expected_row)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def cox_exclude(lung):
    return r.coxph("Surv(time, status) ~ age + ph.ecog", lung, na_action="na.exclude")


@pytest.fixture(scope="module")
def weibull_exclude(lung):
    return r.survreg("Surv(time, status) ~ age + ph.ecog", lung, na_action="na.exclude")


# --- na.omit is the default ----------------------------------------------------------------


def test_coxph_omits_missing_rows_by_default(lung):
    fit = r.coxph("Surv(time, status) ~ age + ph.ecog", lung)
    assert fit.n == 227
    assert fit.na_action == r.NaAction(rows=(14,), kind="omit")
    assert len(fit.na_action) == 1
    assert fit.coefficients == approx([0.0112812387251644, 0.4434853528016963])
    assert r.model_summary(fit)["na_action"] == fit.na_action
    assert len(r.residuals(fit)) == len(r.predict(fit)) == 227
    with pytest.raises(ValueError, match="missing values in formula data"):
        r.coxph("Surv(time, status) ~ age + ph.ecog", lung, na_action="na.fail")
    complete = r.coxph("Surv(time, status) ~ age + sex", lung)
    assert complete.na_action is None


def test_survreg_omits_missing_rows_by_default(lung):
    fit = r.survreg("Surv(time, status) ~ age + ph.ecog", lung)
    assert fit.n == 227
    assert fit.na_action == r.NaAction(rows=(14,), kind="omit")
    assert r.coef(fit) == approx([6.8305035534691747, -0.0077197275613747, -0.3263819844626905])
    assert fit.scale == approx([0.738110535588488])
    assert r.model_summary(fit)["na_action"] == fit.na_action


def test_survreg_interval_response_and_covariates_share_one_na_action():
    data = {
        "left": [1, 2, None, 4, 5, 3, 6, None, 2, 7, 3],
        "right": [3, 4, 2, 6, 5, None, 8, 5, 3, None, 4],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", None],
    }
    fit = r.survreg('Surv(left, right, type = "interval2") ~ g', data)
    assert fit.n == 10  # the missing endpoints are censoring codes
    assert fit.na_action == r.NaAction(rows=(11,), kind="omit")
    assert r.coef(fit) == approx([1.398092952664020, 0.378164150584612])


def test_other_fitters_omit_missing_rows_by_default(lung):
    assert r.aareg("Surv(time, status) ~ age + ph.ecog", lung).n == [227, 137, 138]
    single = r.concordance("Surv(time, status) ~ ph.ecog", lung)
    assert single.concordance == approx(0.395537474099156)
    assert single.n == 227
    both = r.concordance("Surv(time, status) ~ age + ph.ecog", lung)
    assert both.concordance == approx([0.448855309041290, 0.395537474099156])
    with pytest.warns(DeprecationWarning, match="survConcordance is deprecated"):
        old = r.survConcordance("Surv(time, status) ~ ph.ecog", lung)
    assert old.concordance == approx(0.604462525900844)
    assert old.n == 227
    frame = survival.r._formula.model_frame("Surv(time, status) ~ age + ph.ecog", lung)
    assert frame.n == 227
    assert frame.na_action == r.NaAction(rows=(14,), kind="omit")


def test_rttright_omits_missing_rows_by_default(lung):
    data = {name: list(values[:30]) for name, values in lung.items()}
    data["age"][4] = None
    weights = r.rttright("Surv(time, status) ~ age", data)
    assert len(weights) == 29
    assert weights[:6] == approx([1 / 3, 0.25, 0.0, 0.25, 0.0, 0.25])


def test_clogit_builds_its_constant_time_after_na_action():
    data = {
        "case": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        "x": [1, 2, None, 3, 1, 2, 5, 1, 2, 3, 2, 1],
        "s": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    }
    fit = r.clogit("case ~ x + strata(s)", data)
    assert fit.n == 11
    assert fit.na_action == r.NaAction(rows=(3,), kind="omit")
    assert fit.coefficients == approx([0.180425282794039])
    assert fit.loglik == approx([-3.98898404656427, -3.91326510923087])


def test_cch_keeps_refusing_missing_values():
    # R's cch has no na.action argument and fails on a missing covariate
    nwtco = datasets.load_nwtco()
    keep = [
        i
        for i, (rel, sub) in enumerate(zip(nwtco["rel"], nwtco["in.subcohort"], strict=True))
        if rel == 1 or sub == 1
    ]
    data = {
        "seqno": [nwtco["seqno"][i] for i in keep],
        "edrel": [nwtco["edrel"][i] for i in keep],
        "rel": [int(nwtco["rel"][i]) for i in keep],
        "subcohort": [int(nwtco["in.subcohort"][i]) for i in keep],
        "age": [None] + [nwtco["age"][i] / 12 for i in keep[1:]],
    }
    with pytest.raises(ValueError, match="missing values"):
        r.cch("Surv(edrel, rel) ~ age", data, subcoh="subcohort", id="seqno", cohort_size=4028)


def test_tt_expansion_reads_only_the_formula_columns():
    ovarian = {name: list(values) for name, values in datasets.load_ovarian().items()}
    expected = [0.1829492632374148, -0.0112600994753437]
    ragged = r.coxph("Surv(futime, fustat) ~ age + tt(age)", {**ovarian, "meta": [1, 2]})
    assert ragged.coefficients == approx(expected)
    missing = {name: [*values, 1] for name, values in ovarian.items()}
    missing["age"][-1] = None
    fit = r.coxph("Surv(futime, fustat) ~ age + tt(age)", missing)
    assert fit.na_action == r.NaAction(rows=(27,), kind="omit")
    assert fit.coefficients == approx(expected)


# --- na.exclude: naresid / napredict -------------------------------------------------------


def test_coxph_na_exclude_pads_residuals_and_predictions(cox_exclude):
    assert cox_exclude.n == 227
    assert cox_exclude.na_action == r.NaAction(rows=(14,), kind="exclude")
    martingale = r.residuals(cox_exclude)
    assert len(martingale) == 228
    assert [martingale[i] for i in (0, 1, 2, 12, 13, 14)] == approx(
        [0.222668094717438, 0.216554641149297, -1.845501392135114, -1.355699994944682, NAN]
        + [-0.401299372399210]
    )
    lp = r.predict(cox_exclude)
    assert len(lp) == 228
    assert lp[:2] + lp[12:15] == approx(
        [0.1516968473160769, -0.3594759378366059, 0.0840094149650903, NAN, -0.0400842110117181]
    )
    assert r.predict(cox_exclude, se_fit=True).se_fit[12:15] == approx(
        [0.0506570519047983, NAN, 0.0524371171850168]
    )
    expected = r.predict(cox_exclude, type="expected")
    assert len(expected) == 228
    assert expected[12:15] == approx([2.35569999494468, NAN, 1.40129937239921])
    assert r.residuals(cox_exclude, type="deviance")[12:15] == approx(
        [-0.998861425698532, NAN, -0.357489700762621]
    )
    dfbeta = r.residuals(cox_exclude, type="dfbeta")
    assert len(dfbeta) == 228
    rows_approx(
        dfbeta[12:15],
        [
            [-0.000553385422377010, 0.003706320337551102],
            [NAN, NAN],
            [0.000279711633776675, 0.000783828674428051],
        ],
    )
    assert len(r.residuals(cox_exclude, type="score")) == 228
    rows_approx(
        r.residuals(cox_exclude, type="partial")[12:15],
        [
            [-1.293181059630775, -1.334209515293499],
            [NAN, NAN],
            [-0.462874063062112, -0.379808892748027],
        ],
    )
    terms = r.predict(cox_exclude, type="terms", se_fit=True)
    rows_approx(
        terms.fit[12:15],
        [
            [0.0625189353139068, 0.0214904796511835],
            [NAN, NAN],
            [-0.0615746906629018, 0.0214904796511835],
        ],
    )
    rows_approx(
        terms.se_fit[12:15],
        [
            [0.0516466177205868, 0.00561296651420406],
            [NAN, NAN],
            [0.0508665813639166, 0.00561296651420406],
        ],
    )
    # Schoenfeld residuals are per event, never padded
    assert len(r.residuals(cox_exclude, type="schoenfeld")) == 164


def test_coxph_na_exclude_collapses_after_padding(cox_exclude):
    pairs = [row // 2 + 1 for row in range(228)]
    martingale = r.residuals(cox_exclude, collapse=pairs)
    assert len(martingale) == 114
    assert [martingale[i] for i in (0, 5, 6, 7)] == approx(
        [0.439222735866735, -1.272785385561031, NAN, 0.381620590399595]
    )
    dfbeta = r.residuals(cox_exclude, type="dfbeta", collapse=pairs)
    assert len(dfbeta) == 114
    rows_approx(
        dfbeta[5:8],
        [
            [-0.000647657467688236, -0.022995200842009740],
            [NAN, NAN],
            [0.000548397607175879, -0.000715413046919201],
        ],
    )
    with pytest.raises(ValueError, match="Wrong length for 'collapse'"):
        r.residuals(cox_exclude, collapse=pairs[:227])


def test_survreg_na_exclude_pads_residuals_and_predictions(weibull_exclude):
    assert weibull_exclude.na_action == r.NaAction(rows=(14,), kind="exclude")
    response = r.residuals(weibull_exclude)
    assert len(response) == 228
    assert response[12:15] == approx([332.883722565781, NAN, 136.865866675888])
    fitted = r.predict(weibull_exclude)
    assert len(fitted) == 228
    assert fitted[12:15] == approx([395.116277434219, NAN, 430.134133324112])
    lp = r.predict(weibull_exclude, type="lp", se_fit=True)
    assert lp.fit[12:15] == approx([5.97918009483300, NAN, 6.06409709800813])
    assert lp.se_fit[12:15] == approx([0.0668758800535608, NAN, 0.0715594443520515])
    rows_approx(
        r.predict(weibull_exclude, type="quantile")[12:15],
        [[75.0504280504050, 731.274008170411], [NAN, NAN], [81.7019006017513, 796.084417907071]],
    )
    dfbeta = r.residuals(weibull_exclude, type="dfbeta")
    assert len(dfbeta) == 228
    assert dfbeta[13] == approx([NAN] * 4)
    pairs = [row // 2 + 1 for row in range(228)]
    collapsed = r.residuals(weibull_exclude, collapse=pairs)
    assert len(collapsed) == 114
    assert [collapsed[i] for i in (0, 5, 6, 7)] == approx(
        [-163.841809556079, 108.777807995626, NAN, -117.312404447319]
    )


# --- incomplete newdata: predict's na.pass -------------------------------------------------


@pytest.mark.parametrize("missing", [None, NAN])
def test_coxph_predict_gives_na_for_incomplete_newdata(lung, missing):
    fit = r.coxph("Surv(time, status) ~ age + sex", lung)
    newdata = {"age": [50, missing, 60], "sex": [1, 2, 1]}
    assert r.predict(fit, newdata) == approx([-0.00958326858562575, NAN, 0.16087004986848721])
    risk = r.predict(fit, newdata, type="risk", se_fit=True)
    assert risk.fit == approx([0.990462504597168, NAN, 1.174532328265733])
    assert risk.se_fit == approx([0.1349471091355843, NAN, 0.0769652951315515])
    with_response = {**newdata, "time": [100, 200, 300], "status": [1, 1, 0]}
    expected = r.predict(fit, with_response, type="expected", se_fit=True)
    assert expected.fit == approx([0.140304013605526, NAN, 0.730669249133447])
    assert expected.se_fit == approx([0.0316334084375922, NAN, 0.0884141712593211])
    assert r.predict(fit, with_response, type="survival") == approx(
        [0.869093978837999, NAN, 0.481586580814282]
    )
    assert r.predict(fit, newdata, na_action="na.omit") == approx(
        [-0.00958326858562575, 0.16087004986848721]
    )
    assert r.predict(fit, newdata, **{"na.action": "na.exclude"}) == approx(
        [-0.00958326858562575, NAN, 0.16087004986848721]
    )
    assert r.predict(fit, newdata, collapse=[1, 1, 2], na_action="na.omit") == approx(
        [-0.00958326858562575, 0.16087004986848721]
    )
    assert r.predict(fit, {"age": [missing], "sex": [2]}) == approx([NAN])
    with pytest.raises(ValueError, match="missing values in newdata"):
        r.predict(fit, newdata, na_action="na.fail")


def test_coxph_predict_checks_strata_only_when_the_prediction_uses_them(lung):
    fit = r.coxph("Surv(time, status) ~ age + strata(sex)", lung)
    newdata = {"age": [50, 60, 70], "sex": [1, None, 2]}
    by_stratum = r.predict(fit, newdata, se_fit=True)
    assert by_stratum.fit == approx([-0.216312855668989, NAN, 0.144670727189953])
    assert by_stratum.se_fit == approx([0.1225579660936628, NAN, 0.0819671582757146])
    assert r.predict(fit, newdata, reference="sample") == approx(
        [-0.2018297455750251, -0.0396832269312417, 0.1224632917125417]
    )


@pytest.mark.parametrize("missing", [None, NAN])
def test_survreg_predict_gives_na_for_incomplete_newdata(lung, missing):
    fit = r.survreg("Surv(time, status) ~ age + sex", lung)
    newdata = {"age": [50, missing], "sex": [1, 2]}
    assert r.predict(fit, newdata, type="lp") == approx([6.04408691863023, NAN])
    response = r.predict(fit, newdata, type="response", se_fit=True)
    assert response.fit == approx([421.612615052265, NAN])
    assert response.se_fit == approx([49.838755150192, NAN])
    quantiles = r.predict(fit, newdata, type="quantile", se_fit=True)
    rows_approx(quantiles.fit, [[77.2614637580455, 790.756335630527], [NAN, NAN]])
    rows_approx(quantiles.se_fit, [[12.2564368287329, 98.2166772845732], [NAN, NAN]])
    assert r.predict(fit, newdata, type="quantile", p=0.5) == approx([319.806940740502, NAN])
    assert r.predict(fit, newdata, type="lp", na_action="na.omit") == approx([6.04408691863023])
    assert r.predict(fit, newdata, type="lp", na_action="na.exclude") == approx(
        [6.04408691863023, NAN]
    )
    with pytest.raises(ValueError, match="missing values in newdata"):
        r.predict(fit, newdata, na_action="na.fail")
