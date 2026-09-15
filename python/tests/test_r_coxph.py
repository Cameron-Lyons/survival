"""``survival.r`` Cox models: coxph/clogit and their methods against R survival 3.8.11.

Reference values were computed with R (``ovarian`` and ``cgd`` are the bundled
datasets); vector comparisons use R's precision of 15 significant digits.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


class _Approx:
    """``pytest.approx`` that also accepts a matrix (list of rows) or a dict of scalars."""

    def __init__(self, values, rel=1e-8):
        self.values, self.rel = values, rel

    def __eq__(self, other):
        expected = self.values
        if isinstance(expected, dict):
            return set(other) == set(expected) and all(
                other[key] == _Approx(expected[key], self.rel) for key in expected
            )
        if isinstance(expected, list) and expected and isinstance(expected[0], list):
            return len(other) == len(expected) and all(
                row == _Approx(exp_row, self.rel)
                for row, exp_row in zip(other, expected, strict=True)
            )
        return other == pytest.approx(expected, rel=self.rel, abs=1e-12)

    def __repr__(self):
        return f"approx({self.values!r}, rel={self.rel})"


def approx(values, rel=1e-8):
    return _Approx(values, rel)


@pytest.fixture(scope="module")
def ovarian():
    return datasets.load_ovarian()


@pytest.fixture(scope="module")
def cgd():
    return datasets.load_cgd()


@pytest.fixture(scope="module")
def fit(ovarian):
    return r.coxph("Surv(futime, fustat) ~ age + rx", ovarian)


@pytest.fixture(scope="module")
def strata_fit(ovarian):
    return r.coxph("Surv(futime, fustat) ~ age + strata(rx)", ovarian)


# ---------------------------------------------------------------------------
# coxph
# ---------------------------------------------------------------------------


def test_coxph_matches_r_components(fit):
    assert isinstance(fit, r._coxph.CoxphModel)
    assert fit.coef_names == ("age", "rx")
    assert fit.assign == {"age": (0,), "rx": (1,)}
    assert fit.coefficients == approx([0.147326595469114, -0.803973012665114])
    assert fit.var == approx(
        [[0.00212955001519192, 0.00469692194666263], [0.00469692194666263, 0.399486408493207]]
    )
    assert fit.naive_var is None
    assert not fit.robust
    assert fit.loglik == approx([-34.9849403711626, -27.0418988630084])
    assert fit.score == approx(18.5571233836198, rel=1e-6)
    assert fit.wald_test == approx(13.467506895286, rel=1e-6)
    assert fit.rscore is None
    assert fit.iter == 5
    assert (fit.n, fit.nevent, fit.nvar, fit.method) == (26, 12, 2, "efron")
    assert fit.means == approx([56.1654423076923, 1.5])
    assert fit.linear_predictors[:3] == approx(
        [2.78367674829753, 3.10215264972311, 1.91950313725435]
    )
    assert fit.residuals[:3] == approx([0.818224210164987, 0.444577488815711, 0.69498818752843])
    assert fit.concordance["concordance"] == approx(0.798165137614679)
    assert fit.concordance["std"] == approx(0.0763261991426006, rel=1e-6)
    assert [fit.concordance[k] for k in ("concordant", "discordant", "tied.x")] == [174, 44, 0]
    assert fit.weights is None
    assert fit.offset is None
    assert fit.strata is None
    assert fit.y.type == "right"
    assert len(fit.y) == 26
    assert len(fit.x) == 26
    assert fit.x[0] == [72.3315, 1.0]
    assert fit.formula == "Surv(futime, fustat) ~ age + rx"
    assert fit.timefix
    assert not fit.tt
    assert fit.model is None


def test_coxph_ties_init_and_control(ovarian):
    breslow = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, ties="breslow")
    assert breslow.method == "breslow"
    assert breslow.coefficients == approx([0.147326595469114, -0.803973012665114])
    exact = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, method="exact")
    assert exact.method == "exact"
    assert exact.loglik == approx([-34.9849403711626, -27.0418988630084])
    init = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, init=[0.1, -0.5], iter_max=1)
    assert init.coefficients == approx([0.142430939255352, -0.830893914178188])
    assert init.wald_test == approx(1.33688420305505, rel=1e-6)  # coxph.wtest(var, coef - init)
    control = r.coxph(
        "Surv(futime, fustat) ~ age + rx",
        ovarian,
        control={"iter.max": 1, "eps": 1e-9},
        init=[0.1, -0.5],
    )
    assert control.coefficients == init.coefficients
    with pytest.raises(ValueError, match="wrong length for init"):
        r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, init=[0.1])
    with pytest.raises(ValueError, match="overflow or underflow"):
        r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, init=[50.0, 0.0])
    with pytest.raises(ValueError, match="Argument dats not matched"):
        r.coxph("Surv(futime, fustat) ~ age", ovarian, dats=1)
    with pytest.raises(ValueError, match="use only one of method or ties"):
        r.coxph("Surv(futime, fustat) ~ age", ovarian, ties="efron", method="breslow")


def test_coxph_robust_weights_offset_subset(ovarian):
    robust = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, robust=True)
    assert robust.robust
    assert robust.var == approx(
        [[0.00214683496629591, 0.0134510975088254], [0.0134510975088254, 0.374614593070843]],
        rel=1e-6,
    )
    assert robust.naive_var == approx(
        [[0.00212955001519192, 0.00469692194666263], [0.00469692194666263, 0.399486408493207]]
    )
    assert robust.rscore == approx(7.06193432522028, rel=1e-6)
    weighted = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, weights=[1, 2] * 13)
    assert weighted.coefficients == approx([0.182033024686623, -0.573931049701355])
    assert not weighted.robust
    assert weighted.weights[:3] == [1.0, 2.0, 1.0]
    fractional = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, weights=[1, 1.5] * 13)
    assert fractional.robust  # non-integer weights default to the robust variance, as in R
    assert fractional.var == approx(
        [[0.00235439823163678, 0.017170029798964], [0.017170029798964, 0.384191487657066]],
        rel=1e-6,
    )
    offset = r.coxph("Surv(futime, fustat) ~ age + offset(rx)", ovarian)
    assert offset.coefficients == approx([0.193789916928611])
    assert offset.linear_predictors[:3] == approx(
        [4.13281897725543, 4.55173464068001, 2.99610546152728]
    )
    assert offset.offset == [float(v) for v in ovarian["rx"]]
    subset = r.coxph(
        "Surv(futime, fustat) ~ age + rx", ovarian, subset=[age > 50 for age in ovarian["age"]]
    )
    assert subset.n == 20
    assert subset.coefficients == approx([0.178892437874408, -0.526206677564411])
    nocenter = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, nocenter=None)
    assert nocenter.means == approx([56.1654423076923, 1.5])


def test_coxph_strata_cluster_and_id(ovarian, cgd, strata_fit):
    assert strata_fit.coefficients == approx([0.137351719270484])
    assert strata_fit.strata_levels == ("rx=1", "rx=2")
    assert strata_fit.strata[:3] == ["rx=1", "rx=1", "rx=1"]
    cluster = r.coxph("Surv(futime, fustat) ~ age + rx + cluster(ecog.ps)", ovarian)
    assert cluster.robust
    assert cluster.cluster == tuple(ovarian["ecog.ps"])
    assert cluster.var == approx(
        [[0.00106619311130853, 0.0117209122641653], [0.0117209122641653, 0.128850752126559]],
        rel=1e-6,
    )
    assert cluster.concordance["std"] == approx(0.0246395259162723, rel=1e-6)
    with pytest.warns(RuntimeWarning, match="cluster specified with robust=FALSE"):
        ignored = r.coxph(
            "Surv(futime, fustat) ~ age + rx + cluster(ecog.ps)", ovarian, robust=False
        )
    assert not ignored.robust
    with pytest.warns(RuntimeWarning, match="cluster appears both"):
        r.coxph("Surv(futime, fustat) ~ age + cluster(rx)", ovarian, cluster="ecog.ps")
    by_term = r.coxph("Surv(tstart, tstop, status) ~ treat + age + cluster(id)", cgd)
    by_id = r.coxph("Surv(tstart, tstop, status) ~ treat + age", cgd, id="id")
    assert by_term.coefficients == approx([-1.1200824179883, -0.0305485974364017])
    expected_var = [
        [0.096057186086156, 0.000287264963587576],
        [0.000287264963587576, 0.000208728024603667],
    ]
    assert by_term.var == approx(expected_var, rel=1e-6)
    assert by_id.var == approx(expected_var, rel=1e-6)
    assert by_term.rscore == approx(11.0900515205177, rel=1e-6)
    assert by_id.id == tuple(cgd["id"])
    assert by_id.cluster is None
    with pytest.raises(ValueError, match="one of cluster or id is needed"):
        r.coxph("Surv(tstart, tstop, status) ~ treat + age", cgd, robust=True)


def test_coxph_formula_terms(ovarian):
    factor = r.coxph("Surv(futime, fustat) ~ age + factor(rx)", ovarian)
    assert factor.coef_names == ("age", "factor(rx)2")
    assert factor.assign == {"age": (0,), "factor(rx)": (1,)}
    assert factor.coefficients == approx([0.147326595469114, -0.803973012665114])
    interaction = r.coxph("Surv(futime, fustat) ~ age * rx", ovarian)
    assert interaction.coef_names == ("age", "rx", "age:rx")
    assert interaction.coefficients == approx(
        [-0.0122019497753204, -9.50426487155975, 0.146448733767626]
    )
    log = r.coxph("Surv(futime, fustat) ~ log(age) + rx", ovarian)
    assert log.coefficients == approx([8.12981930807953, -0.905203236611623])
    null = r.coxph("Surv(futime, fustat) ~ 1", ovarian)
    assert null.coef_names == ()
    assert null.loglik == approx([-34.9849403711626])
    assert (null.n, null.nevent) == (26, 12)
    assert null.score is None
    assert null.iter is None
    assert null.wald_test is None
    assert null.linear_predictors[:3] == [0.0, 0.0, 0.0]
    with_model = r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, model=True)
    assert list(r.model_frame(with_model)) == ["time", "status", "futime", "fustat", "age", "rx"]
    with pytest.raises(TypeError, match="a formula argument is required"):
        r.coxph()
    with pytest.raises(NotImplementedError, match="multi-state"):
        r.coxph("Surv(futime, fustat) ~ age", ovarian, istate="rx")


def test_coxph_time_transform(ovarian):
    custom = r.coxph(
        "Surv(futime, fustat) ~ age + tt(age)",
        ovarian,
        tt=lambda x, t, riskset, weights: [xi * math.log(ti) for xi, ti in zip(x, t, strict=True)],
    )
    assert custom.tt
    assert custom.coef_names == ("age", "tt(age)")
    assert custom.coefficients == approx([0.672048069354324, -0.0912792487479385])
    assert (custom.n, custom.nevent) == (26, 12)
    default = r.coxph("Surv(futime, fustat) ~ age + tt(age)", ovarian)  # O'Brien logit rank
    assert default.coefficients == approx([0.182949263237415, -0.0112600994753437])
    with pytest.raises(ValueError, match="function not defined for models with tt"):
        r.predict(custom)
    with pytest.raises(ValueError, match="model=TRUE"):
        r.coxph("Surv(futime, fustat) ~ age + tt(age)", ovarian, model=True)
    with pytest.raises(ValueError, match="Wrong length for tt"):
        r.coxph("Surv(futime, fustat) ~ age + tt(age)", ovarian, tt=[math.log, math.exp])


def test_clogit_matches_r():
    data = {
        "case": [1, 0, 0] * 8,
        "set": [s for s in range(1, 9) for _ in range(3)],
        "x": [1.2, 0.3, 0.8, 2.1, 1.0, 0.4, 0.9, 1.7, 0.2, 1.4, 0.6, 1.1]
        + [0.5, 2.2, 1.3, 0.7, 1.8, 0.1, 1.6, 0.9, 1.5, 0.3, 2.0, 1.2],
        "z": [0, 1] * 12,
    }
    exact = r.clogit("case ~ x + z + strata(set)", data)
    assert isinstance(exact, r._coxph.ClogitModel)
    assert isinstance(exact, r._coxph.CoxphModel)
    assert exact.method == "exact"
    assert exact.coefficients == approx([0.0375393088249208, 0.00297696745373074])
    assert exact.var == approx(
        [[0.374819012057062, 0.0290024905727018], [0.0290024905727018, 0.562401334960964]],
        rel=1e-6,
    )
    assert exact.loglik == approx([-8.78889830934488, -8.78702033632205])
    assert (exact.n, exact.nevent, exact.iter) == (24, 8, 3)
    assert exact.strata_levels[:2] == ("set=1", "set=2")
    efron = r.clogit("case ~ x + z + strata(set)", data, method="efron")
    assert efron.method == "efron"
    approximate = r.clogit("case ~ x + z + strata(set)", data, method="approximate")
    assert approximate.method == "breslow"
    assert approximate.coefficients == approx([0.037539308824921, 0.00297696745373081])
    with pytest.warns(RuntimeWarning, match="weights ignored"):
        r.clogit("case ~ x + z + strata(set)", data, weights=[1.0] * 24)
    with pytest.raises(ValueError, match="robust variance plus the exact method"):
        r.clogit("case ~ x + z + strata(set) + cluster(set)", data)
    with pytest.raises(ValueError, match="not defined for a clogit"):
        r.survfit(exact)
    with pytest.raises(ValueError, match="not defined for a clogit"):
        r.basehaz(exact)
    with pytest.raises(ValueError, match="score residuals are not available for the exact"):
        r.residuals(exact, type="score")


# ---------------------------------------------------------------------------
# summary / wtest
# ---------------------------------------------------------------------------


def test_summary_coxph_matches_r(fit):
    summary = r.model_summary(fit)
    rows = summary["coefficients"]
    assert summary["coefficient_columns"] == ["coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"]
    assert [row["name"] for row in rows] == ["age", "rx"]
    assert [rows[0][k] for k in ("coef", "exp_coef", "se", "z", "p")] == approx(
        [
            0.147326595469114,
            1.15873233797022,
            0.0461470477408026,
            3.1925464939082,
            0.0014102423942954,
        ],
        rel=1e-6,
    )
    assert [rows[1][k] for k in ("coef", "exp_coef", "se", "z", "p")] == approx(
        [
            -0.803973012665114,
            0.447547316050731,
            0.63204937187945,
            -1.27200982776778,
            0.203369627452367,
        ],
        rel=1e-6,
    )
    conf = summary["conf_int"][0]
    assert [conf[k] for k in ("exp(coef)", "exp(-coef)", "lower", "upper")] == approx(
        [1.15873233797022, 0.863012075551224, 1.05852882579423, 1.26842141502431], rel=1e-6
    )
    assert summary["logtest"] == approx(
        {"test": 15.8860830163083, "df": 2, "pvalue": 0.000355124719322007}, rel=1e-6
    )
    assert summary["waldtest"] == approx(
        {"test": 13.47, "df": 2, "pvalue": 0.00119005774521772}, rel=1e-6
    )
    assert summary["rsq"] == approx(
        {"rsq": 0.457193943356865, "maxrsq": 0.932197028619173}, rel=1e-6
    )
    assert summary["concordance"] == approx(
        {"C": 0.798165137614679, "se(C)": 0.0763261991426006}, rel=1e-6
    )
    assert (summary["n"], summary["nevent"], summary["df"], summary["used_robust"]) == (
        26,
        12,
        2,
        False,
    )
    assert "robscore" not in summary
    scaled = r.model_summary(fit, scale=2.0)
    assert scaled["coefficients"][0]["coef"] == approx(2 * 0.147326595469114)
    with pytest.raises(ValueError, match="null Cox model"):
        r.model_summary(r.coxph("Surv(futime, fustat) ~ 1", datasets.load_ovarian()))


def test_coxph_wtest_matches_r():
    result = r.coxph_wtest([[2.0, 0.5], [0.5, 1.0]], [1.0, 2.0])
    assert result.test == approx([4.0])
    assert result.df == 2
    assert result.solve == approx([0.0, 2.0])
    singular = r.coxph_wtest([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    assert singular.df == 1
    assert singular.test == approx([1.0])
    matrix = r.coxph_wtest([[1.0, 0.0], [0.0, 1.0]], [[1.0, 3.0], [2.0, 4.0]])
    assert matrix.test == approx([5.0, 25.0])
    assert matrix.solve == approx([[1.0, 3.0], [2.0, 4.0]])
    assert r.coxph_wtest([[1.0, 0.0], [0.0, 1.0]], [None, 2.0]).test == approx([4.0])
    scalar = r.coxph_wtest([2.0], [4.0])
    assert scalar.test == approx([8.0])
    assert scalar.solve == approx([2.0])
    assert r.coxph_wtest([], []).df == 0
    with pytest.raises(ValueError, match="Argument lengths do not match"):
        r.coxph_wtest([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="square matrix"):
        r.coxph_wtest([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="infinite argument"):
        r.coxph_wtest([[1.0, 0.0], [0.0, math.inf]], [1.0, 2.0])


# ---------------------------------------------------------------------------
# predict / residuals
# ---------------------------------------------------------------------------


def test_predict_types_match_r(fit):
    assert r.predict(fit)[:3] == approx([2.78367674829753, 3.10215264972311, 1.91950313725435])
    assert r.predict(fit, type="lp", reference="sample")[:3] == approx(r.predict(fit)[:3])
    assert r.predict(fit, type="lp", reference="zero")[:3] == approx(
        [9.85238062750911, 10.1708565289347, 8.98820701646592]
    )
    lp = r.predict(fit, type="lp", se_fit=True)
    assert lp.se_fit[:3] == approx(
        [0.761892721241284, 0.853883878048919, 0.526717456433125], rel=1e-6
    )
    assert r.predict(fit, type="risk")[:3] == approx(
        [16.1783956168244, 22.2457871635984, 6.81757023095213]
    )
    expected = r.predict(fit, type="expected", se_fit=True)
    assert expected.fit[:3] == approx([0.181775789835013, 0.555422511184289, 0.30501181247157])
    assert expected.se_fit[:3] == approx(
        [0.183369802890354, 0.415757832650625, 0.187371905283234], rel=1e-6
    )
    terms = r.predict(fit, type="terms", se_fit=True)
    assert terms.fit[:2] == approx(
        [[2.38169024196498, 0.401986506332557], [2.70016614339056, 0.401986506332557]]
    )
    assert terms.se_fit[:2] == approx(
        [[0.746015836107493, 0.316024685939725], [0.845771909208786, 0.316024685939725]],
        rel=1e-6,
    )
    assert r.predict_terms_constant(fit) == approx(7.06870387921157)
    assert r.predict(fit, type="terms", terms="rx")[:1] == approx([[0.401986506332557]])
    assert r.predict(fit, type="terms", terms=[2])[:1] == approx([[0.401986506332557]])
    assert r.predict(fit, type="lp", collapse=[1, 2] * 13) == approx(
        [0.573449612999897, -0.573449612999897]
    )
    assert list(r.predict(fit, type="lp", se_fit=True)) == [lp.fit, lp.se_fit]
    with pytest.raises(ValueError, match="terms argument not found"):
        r.predict(fit, type="terms", terms="sex")
    with pytest.raises(ValueError, match="Collapse vector is the wrong length"):
        r.predict(fit, collapse=[1, 2])


def test_predict_newdata_matches_r(fit, strata_fit):
    newdata = {"age": [50, 60], "rx": [1, 2]}
    assert r.predict(fit, newdata) == approx([-0.506347118420988, 0.162945823605037])
    assert r.predict(fit, newdata, type="risk") == approx([0.602693129302648, 1.1769729238112])
    assert r.predict(fit, newdata, type="lp", se_fit=True).se_fit == approx(
        [0.458017580700331, 0.386257384970718], rel=1e-6
    )
    with_response = {**newdata, "futime": [300, 500], "fustat": [1, 0]}
    assert r.predict(fit, with_response, type="expected") == approx(
        [0.0407365755473796, 0.503870491907558]
    )
    survival_pred = r.predict(fit, with_response, type="survival", se_fit=True)
    assert survival_pred.fit == approx([0.960082005716333, 0.604187624976049])
    assert survival_pred.se_fit == approx([0.0392046645871518, 0.153761400962293], rel=1e-6)
    with pytest.raises(ValueError, match="must contain the response"):
        r.predict(fit, newdata, type="expected")
    assert r.predict(strata_fit)[:3] == approx(
        [2.27992126265281, 2.57683447419981, 1.47425728292793]
    )
    assert r.predict(strata_fit, reference="sample")[:3] == approx(
        [2.22043581786429, 2.5173490294113, 1.41477183813941]
    )
    with pytest.raises(ValueError, match="strata variable"):
        r.predict(strata_fit, {"age": [50, 60]})
    with pytest.raises(ValueError, match="strata not found"):
        r.predict(strata_fit, {"age": [50], "rx": [3]})


def test_residual_types_match_r(fit):
    assert r.residuals(fit)[:3] == approx([0.818224210164987, 0.444577488815711, 0.69498818752843])
    assert r.residuals(fit, type="deviance")[:3] == approx(
        [1.331733505748, 0.535628013895036, 0.992387611955406]
    )
    assert r.residuals(fit, type="score")[:3] == approx(
        [
            [2.07633325949175, -0.108882403419631],
            [2.4807471017038, -0.0796928995703468],
            [0.124841548207016, -0.177183755645477],
        ]
    )
    assert r.residuals(fit, type="dfbeta")[:3] == approx(
        [
            [0.00391024337406709, -0.0337446650351247],
            [0.00490856369913549, -0.0201843547256554],
            [-0.000566361949602518, -0.0701961311785209],
        ]
    )
    assert r.residuals(fit, type="dfbetas")[:3] == approx(
        [
            [0.0847344210626438, -0.0533892865596595],
            [0.10636788135843, -0.0319347753888839],
            [-0.0122729833722764, -0.111061151710011],
        ],
        rel=1e-6,
    )
    schoenfeld = r.residuals(fit, type="schoenfeld")
    assert len(schoenfeld) == 12
    assert schoenfeld[:2] == approx(
        [[2.53760917080791, -0.133071598306384], [5.26306163081291, -0.162634638101886]]
    )
    assert r.residuals(fit, type="scaledsch")[:2] == approx(
        [[0.204673860321303, -1.29886912474649], [0.272655444767498, -1.28697466616326]],
        rel=1e-6,
    )
    assert r.residuals(fit, type="partial")[:2] == approx(
        [[3.19991445212996, 1.22021071649754], [3.14474363220627, 0.846563995148268]]
    )
    assert r.residuals(fit, weighted=True)[:2] == approx(r.residuals(fit)[:2])
    assert r.residuals(fit, collapse=[1, 2] * 13) == approx([0.120085288635876, -0.120085288635875])
    single = r.coxph("Surv(futime, fustat) ~ age", datasets.load_ovarian())
    assert isinstance(r.residuals(single, type="score")[0], float)  # R drops to a vector
    with pytest.raises(ValueError, match="type must be one of"):
        r.residuals(fit, type="response")
    with pytest.raises(ValueError, match="Wrong length for 'collapse'"):
        r.residuals(fit, collapse=[1, 2])


def test_residuals_collapse_true_uses_the_cluster(cgd):
    fit = r.coxph("Surv(tstart, tstop, status) ~ treat + age + cluster(id)", cgd)
    dfbeta = r.residuals(fit, type="dfbeta", collapse=True)
    assert len(dfbeta) == 128
    assert dfbeta[0] == approx([0.065667198724883, -0.000535244950828956])
    newdata = {
        "tstart": [0, 100, 200],
        "tstop": [100, 200, 400],
        "status": [1, 1, 0],
        "treat": ["placebo"] * 3,
        "age": [20] * 3,
    }
    assert r.predict(fit, newdata, type="expected") == approx(
        [0.172511140290613, 0.17889894508017, 1.1164435791036]
    )


# ---------------------------------------------------------------------------
# survfit / basehaz
# ---------------------------------------------------------------------------


def test_survfit_coxph_matches_r(fit):
    curve = r.survfit(fit)
    assert isinstance(curve, r.CoxSurvfitResult)
    assert curve.n == [26]
    assert curve.strata is None
    assert len(curve.time) == 26
    assert curve.time[:3] == [59.0, 115.0, 156.0]
    assert curve.n_risk[:3] == [26.0, 25.0, 24.0]
    assert curve.n_event[:3] == [1.0, 1.0, 1.0]
    assert curve.surv[:3] == approx([0.988827173113619, 0.975341574155817, 0.956246954292176])
    assert curve.cumhaz[:3] == approx([0.0112357117566082, 0.024967536868875, 0.0447390789003977])
    assert curve.std_err[:3] == approx(
        [0.0133053842638851, 0.023527686907885, 0.0366754618347347], rel=1e-6
    )
    assert curve.std_chaz == curve.std_err
    assert curve.logse
    assert curve.lower[:3] == approx(
        [0.963373794982104, 0.93138648789261, 0.889921819642363], rel=1e-6
    )
    assert curve.upper[:3] == [1.0, 1.0, 1.0]
    assert (curve.conf_type, curve.conf_int, curve.type) == ("log", 0.95, "right")
    assert len(r.survfit(fit, censor=False).time) == 12
    assert r.survfit(fit, conf_type="log-log").lower[:3] == approx(
        [0.891860812081944, 0.853591025128465, 0.800046601769798], rel=1e-6
    )
    assert r.survfit(fit, stype=1, ctype=1).surv[:3] == approx(
        [0.987676149376561, 0.971623612241242, 0.951206418569318]
    )
    without_se = r.survfit(fit, se_fit=False)
    assert without_se.std_err is None
    assert without_se.lower is None
    assert without_se.conf_type == "none"
    newdata = r.survfit(fit, {"age": [50, 60], "rx": [1, 2]})
    assert newdata.ncurve == 2
    assert len(newdata.surv) == 26
    assert newdata.surv[:2] == approx(
        [[0.993251189923049, 0.986862926107219], [0.985064888897564, 0.9710414585281]]
    )
    with pytest.raises(ValueError, match="stype must be 1 or 2"):
        r.survfit(fit, stype=3)
    with pytest.raises(NotImplementedError, match="start.time"):
        r.survfit(fit, start_time=100)


def test_survfit_strata_and_newdata_blocks(strata_fit):
    curve = r.survfit(strata_fit)
    assert curve.strata == {"rx=1": 13, "rx=2": 13}
    assert curve.n == [13, 13]
    found = r.survfit(strata_fit, {"age": [50, 60], "rx": [1, 2]})
    assert found.strata == {"1": 13, "2": 13}  # one block per newdata row, named by row
    assert found.surv[:3] == approx([0.990398488920438, 0.978417955484759, 0.960217536226302])
    every = r.survfit(strata_fit, {"age": [50, 60]})
    assert every.strata == {"rx=1": 13, "rx=2": 13}
    assert every.ncurve == 2
    assert every.surv[:2] == approx(
        [[0.990398488920438, 0.962615079946976], [0.978417955484759, 0.917442511566923]]
    )
    frame = r.as_data_frame(every)
    assert frame["curve"][:2] == [1, 1]
    assert len(frame["time"]) == 52
    assert frame["strata"][13] == "rx=2"


def test_survfit_individual_curves(cgd):
    fit = r.coxph("Surv(tstart, tstop, status) ~ treat + age + cluster(id)", cgd)
    newdata = {
        "tstart": [0, 100, 200],
        "tstop": [100, 200, 400],
        "status": [1, 1, 0],
        "treat": ["placebo"] * 3,
        "age": [20] * 3,
        "id": [1, 1, 1],
    }
    curve = r.survfit(fit, newdata, id="id")
    assert curve.n == [203]
    assert len(curve.time) == 137
    assert curve.strata is None
    assert curve.surv[:3] == approx([0.990474432220822, 0.981039600883159, 0.971694641670888])
    assert curve.surv[-1] == approx(0.230419512408258)
    with pytest.raises(ValueError, match="only makes sense with new data"):
        r.survfit(fit, id="id")


def test_basehaz_matches_r(fit, strata_fit):
    centered = r.basehaz(fit)
    assert centered.hazard[:3] == approx(
        [0.0112357117566082, 0.024967536868875, 0.0447390789003977]
    )
    assert centered.time[:3] == [59.0, 115.0, 156.0]
    assert centered.strata is None
    assert r.basehaz(fit, centered=False).hazard[:3] == approx(
        [9.56536397652442e-06, 2.12557586845906e-05, 3.80879807997079e-05]
    )
    newdata = r.basehaz(fit, {"age": [50, 60], "rx": [1, 2]})
    assert newdata.hazard[:2] == approx(
        [[0.00677168627853277, 0.0132241285172751], [0.0150477629264815, 0.0293861148689238]]
    )
    assert list(r.as_data_frame(newdata)) == ["hazard.1", "hazard.2", "time"]
    stratified = r.basehaz(strata_fit)
    assert stratified.strata[12:14] == ["rx=1", "rx=2"]
    assert len(stratified.time) == 26
    assert list(r.as_data_frame(stratified)) == ["hazard", "time", "strata"]
    with pytest.raises(TypeError, match="must be a coxph object"):
        r.basehaz("Surv(futime, fustat) ~ age")


# ---------------------------------------------------------------------------
# cox.zph / coxph.detail / anova
# ---------------------------------------------------------------------------


def test_cox_zph_matches_r(fit):
    zph = r.cox_zph(fit)
    assert isinstance(zph, r.CoxZPHResult)
    assert [row["name"] for row in zph.table] == ["age", "rx", "GLOBAL"]
    assert [row["chisq"] for row in zph.table] == approx(
        [0.224265642197981, 0.779606879884519, 0.918669928516088], rel=1e-6
    )
    assert [row["df"] for row in zph.table] == [1, 1, 2]
    assert [row["p"] for row in zph.table] == approx(
        [0.635808758676584, 0.377261396851857, 0.631703611325162], rel=1e-6
    )
    assert zph.transform == "km"
    assert zph.names == ("age", "rx")
    assert zph.strata is None
    assert zph.x[:3] == approx([0.0, 0.0384615384615384, 0.076923076923077])
    assert zph.time[:3] == [59.0, 115.0, 156.0]
    assert zph.y[:2] == approx(
        [[0.204673860321303, -1.29886912474649], [0.272655444767499, -1.28697466616325]],
        rel=1e-6,
    )
    assert zph.var == approx(
        [[0.025554600182303, 0.0563630633599516], [0.0563630633599516, 4.79383690191849]],
        rel=1e-6,
    )
    rank = r.cox_zph(fit, transform="rank", terms=False, global_test=False)
    assert [row["name"] for row in rank.table] == ["age", "rx"]
    assert [row["chisq"] for row in rank.table] == approx(
        [0.254858433308275, 0.688640780119428], rel=1e-6
    )
    assert rank.x[:3] == [1.0, 2.0, 3.0]
    assert r.cox_zph(fit, transform="identity").x[:3] == [59.0, 115.0, 156.0]
    assert r.cox_zph(fit, transform="log").x[:3] == approx(
        [4.07753744390572, 4.74493212836325, 5.04985600724954]
    )
    assert r.cox_zph(fit, **{"global": False}).table == r.cox_zph(fit, global_test=False).table
    subset = zph.subset([1])
    assert subset.names == ("rx",)
    assert subset.table == [zph.table[1]]
    assert subset.var == [[zph.var[1][1]]]
    assert r.as_data_frame(zph)["name"] == ["age", "rx", "GLOBAL"]
    with pytest.raises(ValueError, match="Unrecognized transform"):
        r.cox_zph(fit, transform="sqrt")
    with pytest.raises(ValueError, match="Null model"):
        r.cox_zph(r.coxph("Surv(futime, fustat) ~ 1", datasets.load_ovarian()))
    with pytest.raises(TypeError, match="result of a coxph fit"):
        r.cox_zph("Surv(futime, fustat) ~ age")


def test_coxph_detail_matches_r(fit):
    detail = r.coxph_detail(fit)
    assert isinstance(detail, r.CoxPHDetailResult)
    assert len(detail.time) == 12
    assert detail.time[:3] == [59.0, 115.0, 156.0]
    assert detail.nevent[:3] == [1, 1, 1]
    assert detail.nrisk[:3] == [26, 25, 24]
    assert detail.hazard[:3] == approx([0.0112357117566082, 0.0137318251122668, 0.0197715420315226])
    assert detail.varhaz[:3] == approx(
        [0.000126241218677585, 0.000188563020913881, 0.000390913874304265], rel=1e-6
    )
    assert detail.means[:2] == approx(
        [[69.7938908291921, 1.13307159830638], [69.2301383691871, 1.16263463810189]]
    )
    assert detail.score[:2] == approx(
        [[2.53760917080791, -0.133071598306384], [5.26306163081294, -0.162634638101886]]
    )
    assert detail.imat[0] == approx(
        [[47.5339058472271, -1.35072049310254], [-1.35072049310254, 0.115363548030568]],
        rel=1e-6,
    )
    assert detail.wtrisk[:3] == approx([89.0019272176374, 72.8235316008131, 50.5777444372147])
    assert detail.cumhaz[:2] == approx([0.0112357117566082, 0.024967536868875])
    assert detail.strata is None
    assert detail.riskmat is None
    assert detail.weights is None
    assert detail.y[0] == [-1.0, 59.0, 1.0]
    assert detail.x[0] == [72.3315, 1.0]
    by_time = r.coxph_detail(fit, riskmat=True, rorder="time")
    assert by_time.sortorder[:5] == [0, 1, 2, 21, 22]
    assert [sum(row) for row in by_time.riskmat[:5]] == [1, 2, 3, 4, 5]
    assert by_time.y[:2] == [[-1.0, 59.0, 1.0], [-1.0, 115.0, 1.0]]
    frame = r.as_data_frame(detail)
    assert list(frame) == ["time", "nevent", "nrisk", "hazard", "varhaz", "cumhaz", "wtrisk"]
    exact = r.coxph("Surv(futime, fustat) ~ age + rx", datasets.load_ovarian(), ties="exact")
    with pytest.raises(ValueError, match="not available for the exact method"):
        r.coxph_detail(exact)


def test_anova_matches_r(fit, ovarian):
    table = r.anova(fit)
    assert [row.name for row in table.rows] == ["NULL", "age", "rx"]
    assert [row.loglik for row in table.rows] == approx(
        [-34.9849403711626, -27.8381472910825, -27.0418988630084]
    )
    assert [row.chisq for row in table.rows][1:] == approx(
        [14.2935861601602, 1.59249685614817], rel=1e-6
    )
    assert [row.df for row in table.rows] == [None, 1, 1]
    assert [row.p_value for row in table.rows][1:] == approx(
        [0.000156396864322538, 0.206969765174687], rel=1e-6
    )
    nested = r.anova(r.coxph("Surv(futime, fustat) ~ age", ovarian), fit)
    assert [row.loglik for row in nested.rows] == approx([-27.8381472910825, -27.0418988630084])
    assert [row.chisq for row in nested.rows][1:] == approx([1.59249685614817], rel=1e-6)
    assert r.anova(fit, test=None).rows[1].p_value is None
    frame = r.as_data_frame(table)
    assert list(frame) == ["model", "loglik", "Chisq", "Df", "Pr(>|Chi|)"]
    with pytest.raises(ValueError, match="robust variances"):
        r.anova(r.coxph("Surv(futime, fustat) ~ age + rx", ovarian, robust=True))
    with pytest.raises(ValueError, match="same ties option"):
        r.anova(fit, r.coxph("Surv(futime, fustat) ~ age", ovarian, ties="breslow"))
    with pytest.raises(TypeError, match="All arguments must be Cox models"):
        r.anova(fit, "Surv(futime, fustat) ~ age")
