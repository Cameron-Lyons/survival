"""Methods of a multi-state ``coxph`` fit (R's ``coxphms``): ``predict``,
``residuals``, ``anova``, ``cox.zph`` and ``coxph.detail`` on the stacked data, the
model matrix/frame generics and the refusals.

Reference values are R 4.5.3 / survival 3.8-12 on the data sets of R's tests
multi2.R, residms.R and mstrata.R, rebuilt below with R's row order and factor
levels.  Where R is wrong or fails, the expected value is what the documented
deviation computes, checked against an ordinary coxph of R's stacked data.
"""

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


def cat(values, levels):
    return pd.Categorical(values, categories=levels)


def table(labels):
    return {label: labels.count(label) for label in dict.fromkeys(labels)}


def zph_rows(result):
    return [row["name"] for row in result.table], [row["chisq"] for row in result.table]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mg():
    m = pd.DataFrame(datasets.load_mgus2())
    m["etime"] = np.where(m.pstat == 1, m.ptime, m.futime)
    event = np.where(m.pstat == 1, "pcm", np.where(m.death == 1, "death", "censor"))
    m["event"] = cat(event, ["censor", "pcm", "death"])
    m["w"] = np.where(m.age > 70, 2, 1)
    return m


def _myeloid():
    raw = datasets.load_myeloid()
    base = {name: raw[name] for name in ("id", "trt", "sex", "flt3")}
    merged = r.tmerge(
        base,
        raw,
        "id",
        death=r.event("futime", "death"),
        priortx=r.tdc("txtime"),
        sct=r.event("txtime"),
    )
    data = pd.DataFrame(dict(merged))
    code = (data.sct + 2 * data.death).astype(int)
    data["event"] = cat(np.array(["censor", "sct", "death"])[code], ["censor", "sct", "death"])
    data["is_sct"] = (code == 1).astype(int)
    return data


@pytest.fixture(scope="module")
def my():
    return _myeloid()


@pytest.fixture(scope="module")
def my_w():
    data = _myeloid()
    data["w"] = np.where(data.id % 3 == 0, 2, 1)
    return data


@pytest.fixture(scope="module")
def my_na():
    data = _myeloid()
    data["sex"] = data["sex"].astype(object)
    data["flt3"] = data["flt3"].astype(object)
    data.loc[data.id.isin([273, 274, 275]), "sex"] = None
    data.loc[data.id.isin([271, 272, 273]), "flt3"] = None
    return data


@pytest.fixture(scope="module")
def lms():
    lung = pd.DataFrame(datasets.load_lung())
    data = lung[lung["ph.ecog"].notna() & (lung["ph.ecog"] < 3)].reset_index(drop=True)
    data["state"] = cat(np.where(data.status == 2, "death", "censor"), ["censor", "death"])
    ecog = data["ph.ecog"].astype(int)
    data["cstate"] = cat(np.array(["ph0", "ph1", "ph2"])[ecog], ["ph0", "ph1", "ph2"])
    data["id"] = np.arange(1, len(data) + 1)
    return data


@pytest.fixture(scope="module")
def fa1(mg):
    return r.coxph("Surv(etime, event) ~ age + sex", mg, id="id")


@pytest.fixture(scope="module")
def fm(my):
    return r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my, id="id")


NA_FORMULAS = ["Surv(tstart, tstop, event) ~ trt", "1:3 + 2:3 ~ sex", "1:2 + 2:3 ~ flt3"]


@pytest.fixture(scope="module")
def fna(my_na):
    return r.coxph(NA_FORMULAS, my_na, id="id")


@pytest.fixture(scope="module")
def fnax(my_na):
    return r.coxph(NA_FORMULAS, my_na, id="id", na_action="na.exclude")


@pytest.fixture(scope="module")
def fst(mg):
    return r.coxph("Surv(etime, event) ~ age + mspike + strata(sex)", mg, id="id")


@pytest.fixture(scope="module")
def fw(mg):
    return r.coxph("Surv(etime, event) ~ age + sex", mg, id="id", weights="w")


@pytest.fixture(scope="module")
def fwm(my_w):
    return r.coxph("Surv(tstart, tstop, event) ~ trt + sex", my_w, id="id", weights="w")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict(mg, fa1, fm, fna, fnax, fst):
    p = r.predict(fa1)
    assert len(p.values) == 1384
    assert p.colnames == ["1:2", "1:3"]
    assert p.rownames[:3] == ["1", "2", "3"]
    assert p.values[:5] == [
        approx([0.229159974829616, 1.13445990979215]),
        approx([0.0987820231876739, 0.489021894848329]),
        approx([0.28224978893409, 1.91329886581708]),
        approx([-0.0567328853349595, 0.235160026963153]),
        approx([0.255235565158005, 1.26354751278091]),
    ]
    assert np.sum(p.values, axis=0).tolist() == approx([-18.9281285311604, 294.856838735157])
    assert r.predict(fa1, type="risk").values[:3] == [
        approx([1.25754319817807, 3.10949368221866]),
        approx([1.10382566493364, 1.63072042360702]),
        approx([1.3261099262042, 6.77540311806512]),
    ]
    assert r.predict(fa1, reference="zero").values[:3] == [
        approx([1.14732597444909, 5.6798545315056]),
        approx([1.01694802280715, 5.03441651656178]),
        approx([1.20041578855357, 6.45869348753053]),
    ]
    new = r.predict(fa1, pd.DataFrame({"age": [60, 80], "sex": ["F", "M"]}))
    assert new.values == [
        approx([-0.135898289767822, -0.672766532050543]),
        approx([0.0997206566353712, 1.00968564489573]),
    ]
    incomplete = r.predict(fa1, pd.DataFrame({"age": [60, None, 70], "sex": ["F", "M", "M"]}))
    assert incomplete.rownames == ["1", "3"]
    assert incomplete.values == [
        approx([-0.135898289767822, -0.672766532050543]),
        approx([-0.0306572950065711, 0.364247629951917]),
    ]
    # se.fit is ignored, as in R
    assert r.predict(fa1, se_fit=True).values == p.values
    assert fa1.predict(**{"se.fit": True}).values == p.values

    pm = r.predict(fm)
    assert len(pm.values) == 1009
    assert pm.colnames == ["1:2", "1:3", "2:3"]
    assert pm.values[:4] == [
        approx([-0.138994059753283, -0.391675484853137, -0.289654524006076]),
        approx([-0.0312506555700437, 0.015635169227796, 0.215699959827038]),
        approx([-0.0312506555700437, 0.015635169227796, 0.215699959827038]),
        approx([0.0, 0.0, 0.0]),
    ]
    # no napredict padding under na.exclude; rows missing a covariate are NaN
    for fit in (fna, fnax):
        pn = r.predict(fit)
        assert (len(pn.values), len(pn.colnames)) == (1006, 3)
        assert pn.rownames[419:425] == ["420", "421", "422", "424", "426", "427"]
    assert np.isnan(r.predict(fna).values[421]).any()

    stratified = r.predict(fst, reference="sample")
    assert len(stratified.values) == 1373
    assert stratified.values[:2] == [
        approx([-0.283249718203009, 1.17437253626742]),
        approx([0.845330559554247, 0.438198714033526]),
    ]


def test_predict_refusals(fa1, fst):
    incomplete = "predict.coxphms not complete for type expected, survival and terms"
    for type_ in ("expected", "survival", "terms"):
        with pytest.raises(ValueError, match=incomplete):
            r.predict(fa1, type=type_)
    with pytest.raises(ValueError, match=incomplete):
        r.predict_terms_constant(fa1)
    for call in (lambda: r.predict(fst), lambda: r.predict(fa1, reference="strata")):
        with pytest.raises(ValueError, match="strata reference unfinished"):
            call()
    with pytest.raises(ValueError, match="type must be one of lp, risk"):
        r.predict(fa1, type="response")


def test_predict_leaves_out_ph_rows(mg):
    """Deviation: R stops with "non-conformable arguments" for proportional
    baselines; the ph() coefficient scales a baseline, not a row of X."""

    fsh = r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ 1 / shared"], mg, id="id")
    p = r.predict(fsh)
    assert len(p.values) == 1384
    beta = [0.0106778897372408, 0.0624086063114782]
    assert p.values[0] == approx([(88 - 70.4234104046243) * b for b in beta])


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------


def test_martingale_residuals(fa1, fm):
    rr = r.residuals(fa1)
    assert rr.rownames is None
    assert rr.colnames == ["1:2", "1:3"]
    assert len(rr.values) == 1384
    assert rr.values[:5] == [
        approx([-0.0285295525713872, 0.561886072698562]),
        approx([-0.021968450238322, 0.801849610433482]),
        approx([-0.0445459474894849, -0.39969285412818]),
        approx([-0.0725056061684486, 0.469051592084265]),
        approx([-0.00703826675277095, 0.770382305798317]),
    ]
    assert np.sum(rr.values, axis=0).tolist() == pytest.approx([0.0, 0.0], abs=1e-9)

    rows = r.residuals(fm)
    assert rows.values[:4] == [
        approx([-0.601339771345416, 0.866454035661883, 0.0]),
        approx([0.426878506253466, -0.154797984161297, 0.0]),
        approx([0.0, 0.0, 0.830177743149802]),
        approx([-1.27570312241568, -0.674915147349457, 0.0]),
    ]
    collapsed = r.residuals(fm, collapse=True)
    assert len(collapsed.values) == 646
    assert collapsed.rownames[:3] == ["1", "2", "3"]
    assert collapsed.values[:3] == [
        approx([-0.601339771345416, 0.866454035661883, 0.0]),
        approx([0.426878506253466, -0.154797984161297, 0.830177743149802]),
        approx([-1.27570312241568, -0.674915147349457, 0.0]),
    ]


def test_weighted_martingale_residuals(mg, fw):
    """Deviation: R stops with "argument is of length zero" for an uncollapsed
    weighted request on a fit without an na.action."""

    expected = [
        approx([-0.0578940144001094, 1.13809209004016]),
        approx([-0.0465762711306234, 1.62582510887526]),
        approx([-0.0895944567318889, -0.774530423125467]),
    ]
    weighted = r.residuals(fw, weighted=True)
    assert weighted.values[:3] == expected
    plain = np.array(r.residuals(fw).values)
    assert np.array(weighted.values) == pytest.approx(plain * mg.w.to_numpy()[:, None])
    collapsed = r.residuals(fw, weighted=True, collapse=True)
    assert collapsed.colnames == ["1:2", "1:3"]
    assert collapsed.values[:3] == expected


def test_martingale_columns_follow_the_transition(mg):
    """Deviation: R places the residuals by baseline block, so with a shared or
    common baseline one column holds every transition's values."""

    fcb = r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ 1 / common"], mg, id="id")
    rr = np.array(r.residuals(fcb).values)
    assert (rr != 0).sum(axis=0).tolist() == [1384, 1384]
    rindex = fcb.rmap[:, 0] - 1
    engine = np.array(fcb.residuals)
    for k in range(2):
        rows = fcb.ms.hazard == k
        assert rr[rindex[rows], k] == pytest.approx(engine[rows])

    fnc = r.coxph(["Surv(etime, event) ~ 1", "1:2 ~ age", "1:2 + 1:3 ~ 1 / common"], mg, id="id")
    assert fnc.coefficients == approx([-0.0296309018583241])
    assert fnc.loglik == approx([-7026.90746408329, -6716.66338877123])
    assert np.bincount(fnc.rmap[:, 1])[1:].tolist() == [2768]
    rnc = r.residuals(fnc)
    assert rnc.colnames == ["1:2", "1:3"]
    values = np.array(rnc.values)
    assert values.shape == (1384, 2)
    assert values[:3, 0].tolist() == approx(
        [-0.0157880320430367, -0.0184670292047806, -0.0189568135183752]
    )
    assert values[:3, 1].tolist() == approx(
        [0.785829687583495, 0.813729991091707, 0.692809463846103]
    )
    assert values.sum(axis=0).tolist() == approx([-11.1568550023307, 11.1568550023224], rel=1e-9)
    # deviation: with one coefficient R fails ("object 'rr' not found", "'dimnames'
    # applied to non-array"); the residuals stay one-column matrices
    score = np.array(r.residuals(fnc, type="score").values)
    assert score.shape == (1384, 1)
    assert score.sum() == pytest.approx(0.0, abs=1e-6)
    schoenfeld = np.array(r.residuals(fnc, type="schoenfeld").values)
    assert schoenfeld.shape == (975, 1)
    assert schoenfeld.sum() == pytest.approx(score.sum(), abs=1e-8)


def test_score_and_dfbeta_residuals(fa1, fm):
    score = r.residuals(fa1, type="score")
    assert score.colnames == ["age_1:2", "sexM_1:2", "age_1:3", "sexM_1:3"]
    assert score.rownames == [str(i) for i in range(1, 1385)]
    assert score.values[:3] == [
        approx([-0.466223042310077, 0.0149069837078832, 6.80355895540481, -0.329062261481475]),
        approx([-0.137507529240047, 0.0114923969196643, 1.12268245585108, -0.478855126900027]),
        approx([-1.00879638118934, -0.0214068348271224, -5.83759694014384, -0.150361145078565]),
    ]
    dfbeta = r.residuals(fa1, type="dfbeta")
    assert dfbeta.values[:3] == [
        approx(
            [
                -2.91579696201129e-05,
                0.000446716271971773,
                7.81742444371987e-05,
                -0.00137486515313077,
            ]
        ),
        approx(
            [
                -7.3410772063664e-06,
                0.000383759842447519,
                -1.05599543008788e-06,
                -0.00228926466339734,
            ]
        ),
        approx(
            [
                -7.26103764388033e-05,
                -0.000939221380664344,
                -8.12988354943413e-05,
                -0.000922311745971203,
            ]
        ),
    ]
    d = np.array(dfbeta.values)
    assert d.T @ d == pytest.approx(np.array(fa1.var), abs=1e-10)
    assert r.residuals(fa1, type="dfbeta", weighted=False).values == dfbeta.values
    assert r.residuals(fa1, type="dfbetas").values[:3] == [
        approx(
            [-0.00353040182287227, 0.00237042116920023, 0.0216151649073158, -0.0197261137541866]
        ),
        approx(
            [-0.000888846263606939, 0.00203635397119365, -0.000291982551632567, -0.0328456176671462]
        ),
        approx(
            [-0.00879155197288392, -0.00498381273076372, -0.0224791137878492, -0.0132330260727176]
        ),
    ]

    collapsed = r.residuals(fm, type="dfbeta", collapse=True)
    assert len(collapsed.values) == 646
    assert collapsed.rownames[:3] == ["1", "2", "3"]
    assert collapsed.values[:3] == [
        approx(
            [
                -0.00350358616768381,
                0.00318594192052251,
                0.0141423546168251,
                -0.0125009977728476,
                0,
                0,
            ]
        ),
        approx(
            [
                -0.00278887972457738,
                0.00292461341996025,
                0.00217651690048881,
                -0.00268781478319009,
                -0.00947204556541793,
                0.0108681847515926,
            ]
        ),
        approx(
            [
                0.00704314935303265,
                0.00576247891605673,
                0.00861622254678609,
                0.00805497513836497,
                0,
                0,
            ]
        ),
    ]
    c = np.array(collapsed.values)
    assert c.T @ c == pytest.approx(np.array(fm.var), abs=1e-10)
    assert r.residuals(fm, type="score").values[:3] == [
        approx(
            [-0.2953876495083322, 0.2613052709102068, 0.4528267355276457, -0.3905419392616868, 0, 0]
        ),
        approx(
            [-0.2324417163593817, 0.2428286408492979, 0.0674015834220177, -0.086558878406728, 0, 0]
        ),
        approx([0, 0, 0, 0, -0.3592163136580092, 0.4316357317877835]),
    ]
    # a collapse vector groups in order of first appearance
    groups = [3, 1, 2, 1] * 346
    by_vector = r.residuals(fa1, type="score", collapse=groups)
    assert by_vector.rownames == ["3", "1", "2"]
    s = np.array(score.values)
    assert by_vector.values[0] == approx(s[0::4].sum(axis=0).tolist(), rel=1e-10)


def test_weighted_residuals_on_counting_process_data(my_w, fwm):
    """Deviations: every weighted row carries its own data row's weight.  R reuses
    the stacked weights by data row (residuals.coxphms.R:238) and masks them in
    sorted order (line 164).  The expected values are an ordinary coxph of R's
    stacked data at fwm's coefficients."""

    assert fwm.coefficients == approx(
        [
            -0.133871145401897,
            0.0125996339764578,
            -0.346816602928315,
            -0.00905900567476805,
            -0.296942565695157,
            0.258904446192955,
        ]
    )
    assert fwm.loglik == approx([-5427.10493878249, -5419.01711222891])
    assert np.diag(fwm.var).tolist() == approx(
        [
            0.0123810115194183,
            0.0124221254396811,
            0.033081100070426,
            0.0322434526362473,
            0.0297067275775485,
            0.0297386884985181,
        ]
    )
    dfbeta = r.residuals(fwm, type="dfbeta")
    rows = [
        [-0.00254589630057676, 0.00233633101535726, 0.0104407853346827, -0.0091799503420803, 0, 0],
        [
            -0.00209044179868845,
            0.00212819679821624,
            0.00160071265112316,
            -0.00196872635117938,
            0,
            0,
        ],
        [0, 0, 0, 0, -0.0066166015296544, 0.00749760147987198],
        [0.010455618975984, 0.00867685638027888, 0.012964602704696, 0.0115433207114199, 0, 0],
    ]
    assert dfbeta.values[:4] == [approx(row) for row in rows]
    unweighted = np.array(r.residuals(fwm, type="dfbeta", weighted=False).values)
    assert np.array(dfbeta.values) == pytest.approx(unweighted * my_w.w.to_numpy()[:, None])

    collapsed = r.residuals(fwm, type="dfbeta", collapse=True)
    assert len(collapsed.values) == 646
    first = np.array(rows)
    assert collapsed.values[:3] == [
        approx(first[0].tolist()),
        approx((first[1] + first[2]).tolist()),
        approx(first[3].tolist()),
    ]
    c = np.array(collapsed.values)
    assert c.T @ c == pytest.approx(np.array(fwm.var), abs=1e-10)
    assert r.residuals(fwm, type="dfbetas").values[3] == approx(
        [0.114649741702125, 0.0948609968543434, 0.0881542759125359, 0.0782442906144059, 0, 0]
    )
    assert r.residuals(fwm, type="score", weighted=True).values[:3] == [
        approx(
            [-0.287067400031445, 0.258190836526462, 0.447498139296743, -0.380618276558001, 0, 0]
        ),
        approx(
            [-0.233851309776216, 0.237216993484075, 0.0661999415765481, -0.0843653566411128, 0, 0]
        ),
        approx([0, 0, 0, 0, -0.350992932440102, 0.409319765074261]),
    ]
    schoenfeld = r.residuals(fwm, type="schoenfeld", weighted=True)
    assert schoenfeld.time[:4] == [24, 50, 53, 60]
    assert schoenfeld.values[:4] == [
        approx([-0.484396496007661, 0.554007848423492, 0, 0, 0, 0]),
        approx([-0.483762221400378, -0.450316678786543, 0, 0, 0, 0]),
        approx([0.518057293270147, -0.448291662464279, 0, 0, 0, 0]),
        approx([1.03471108958435, -0.897348303387109, 0, 0, 0, 0]),
    ]
    plain = r.residuals(fwm, type="schoenfeld")
    assert plain.values[3] == approx([0.517355544792173, -0.448674151693554, 0, 0, 0, 0])


def test_weighted_schoenfeld_rows_carry_their_event_weight(mg, my_w, fw, fwm):
    for fit, weights in ((fw, mg.w.to_numpy()), (fwm, my_w.w.to_numpy())):
        engine = fit.fit.schoenfeld_residuals(False)
        event_weight = weights[fit.rmap[np.asarray(engine.rows), 0] - 1]
        weighted = np.array(r.residuals(fit, type="schoenfeld", weighted=True).values)
        plain = np.array(r.residuals(fit, type="schoenfeld").values)
        assert weighted == pytest.approx(plain * event_weight[:, None], abs=1e-14)


def test_missing_values(my_na, fna, fnax):
    mart = r.residuals(fna)
    assert (len(mart.values), len(mart.colnames)) == (1006, 3)
    assert mart.values[:3] == [
        approx([-0.67061032595808, 0.867255085743112, 0]),
        approx([0.35950847238352, -0.157723858242483, 0]),
        approx([0, 0, 0.836054884183869]),
    ]
    dfbeta = r.residuals(fna, type="dfbeta")
    assert (len(dfbeta.values), len(dfbeta.colnames)) == (1006, 9)
    assert dfbeta.rownames[417:426] == [
        "418",
        "419",
        "420",
        "421",
        "422",
        "424",
        "426",
        "427",
        "429",
    ]
    by_label = dict(zip(dfbeta.rownames, dfbeta.values, strict=True))
    assert [by_label[label] for label in ("421", "422", "424")] == [
        approx(
            [
                0,
                0,
                0,
                0,
                0,
                -0.0116964178777693,
                0.0126691931523404,
                -0.00129357585081853,
                0.0142394424309581,
            ]
        ),
        approx([0, 0, 0, 0.00218158135004756, 0.00222592923746637, 0, 0, 0, 0]),
        approx([0, 0, 0, 0.0142908851501649, 0.0138493470475762, 0, 0, 0, 0]),
    ]
    assert len(r.residuals(fna, type="dfbeta", collapse=True).values) == 645

    for padded in (
        r.residuals(fnax),
        r.residuals(fnax, type="dfbeta"),
        r.residuals(fna, type="dfbeta", na_action="na.exclude"),
        r.residuals(fna, **{"na.action": "na.exclude"}),
    ):
        values = np.array(padded.values)
        assert values.shape[0] == 1009
        assert (np.flatnonzero(np.isnan(values[:, 0])) + 1).tolist() == [423, 425, 428]
    assert r.residuals(fnax, type="dfbeta").rownames[422] == "423"
    assert len(r.residuals(fnax, na_action="na.omit").values) == 1006
    with pytest.raises(ValueError, match="changing to an unrecognized na.action type"):
        r.residuals(fna, na_action="na.pass")


def test_residuals_match_the_single_transition_fits(my_na, fna):
    """residms.R: the 1:2 transition of fna is the fit of the entry:sct rows."""

    subset = [p == 0 for p in my_na["priortx"]]
    fit12 = r.coxph(
        "Surv(tstart, tstop, is_sct) ~ trt + flt3",
        my_na,
        subset=subset,
        id="id",
        iter_max=4,
        ties="breslow",
    )
    kept = my_na.drop(index=[422, 424, 427]).reset_index(drop=True)
    rows = np.flatnonzero((kept.priortx == 0).to_numpy() & kept.flt3.notna().to_numpy())
    assert len(rows) == len(fit12.residuals)
    mart = np.array(r.residuals(fna).values)
    assert mart[rows, 0] == pytest.approx(r.residuals(fit12), rel=1e-6, abs=1e-9)
    dfbeta = np.array(r.residuals(fna, type="dfbeta").values)
    columns = [fna.coef_names.index(name) for name in ("trtB_1:2", "flt3B_1:2", "flt3C_1:2")]
    assert dfbeta[np.ix_(rows, columns)] == pytest.approx(
        np.array(r.residuals(fit12, type="dfbeta")), rel=1e-6, abs=1e-9
    )


def test_schoenfeld_residuals(mg, fa1, fm, fst):
    sch = r.residuals(fa1, type="schoenfeld")
    assert isinstance(sch, r.CoxphmsSchoenfeldResiduals)
    assert len(sch.values) == 975
    assert sch.colnames == list(fa1.coef_names)
    assert sch.time[:5] == [2, 2, 4, 5, 6]
    assert sch.values[:3] == [
        approx([4.73159381623526, -0.533363894227398, 0, 0]),
        approx([4.73159381623526, -0.533363894227398, 0, 0]),
        approx([4.87472848904315, -0.528979038321383, 0, 0]),
    ]
    assert sch.values[-2:] == [
        approx([0, 0, 12.1004776452153, -0.302402959981258]),
        approx([0, 0, 0, 0]),
    ]
    assert table(sch.transition) == {"1:2": 115, "1:3": 860}
    assert sch.strata is None
    assert r.residuals(fa1, type="schoenfeld", collapse=True).values == sch.values

    msch = r.residuals(fm, type="schoenfeld")
    assert len(msch.values) == 684
    assert table(msch.transition) == {"1:2": 364, "1:3": 140, "2:3": 180}
    assert msch.time[:6] == [24, 50, 53, 60, 62, 63]
    assert msch.values[:2] == [
        approx([-0.482527763322982, 0.568330243343097, 0, 0, 0, 0]),
        approx([-0.483193722318972, -0.436252844038486, 0, 0, 0, 0]),
    ]
    assert r.residuals(fm, type="scaledsch").values[:2] == [
        approx(
            [
                -4.13665045729857,
                4.61462990360071,
                -0.391675484853137,
                0.015635169227796,
                -0.289654524006076,
                0.215699959827038,
            ]
        ),
        approx(
            [
                -3.55268788750361,
                -3.09696499947694,
                -0.391675484853137,
                0.015635169227796,
                -0.289654524006076,
                0.215699959827038,
            ]
        ),
    ]
    assert r.residuals(fa1, type="scaledsch").values[:3] == [
        approx([0.235473269213753, -17.6756429834196, 0.0645438014943843, 0.391576147058636]),
        approx([0.235473269213753, -17.6756429834196, 0.0645438014943843, 0.391576147058636]),
        approx([0.245751213911377, -17.499051059238, 0.0645438014943844, 0.391576147058635]),
    ]

    stratified = r.residuals(fst, type="schoenfeld")
    assert len(stratified.values) == 969
    assert stratified.strata[:5] == [1, 1, 1, 1, 1]

    # deviation: each event is labelled by its own transition, not its block's first
    fsh = r.coxph(["Surv(etime, event) ~ age", "1:2 + 1:3 ~ 1 / shared"], mg, id="id")
    shared = r.residuals(fsh, type="schoenfeld")
    assert table(sorted(shared.transition)) == {"1:2": 115, "1:3": 860}


def test_residual_errors(fa1):
    for type_ in ("deviance", "partial"):
        with pytest.raises(ValueError, match="type must be one of martingale, score, schoenfeld"):
            r.residuals(fa1, type=type_)
    with pytest.raises(ValueError, match="collapse vector not the same length as the model frame"):
        r.residuals(fa1, type="score", collapse=[1, 2, 3])


def test_anova_refuses(mg, fa1):
    f0 = r.coxph("Surv(etime, event) ~ age", mg, id="id")
    single = r.coxph("Surv(etime, death) ~ age", mg)
    for call in (
        lambda: r.anova(fa1),
        lambda: r.anova(f0, fa1),
        lambda: r.anova([f0, fa1]),
        lambda: r.anova(single, fa1),
    ):
        with pytest.raises(NotImplementedError, match="anova not yet available for multistate"):
            call()


# ---------------------------------------------------------------------------
# cox.zph / coxph.detail
# ---------------------------------------------------------------------------


def test_cox_zph(my, fa1, fm):
    z = r.cox_zph(fm, transform="identity")
    names, chisq = zph_rows(z)
    assert names == ["trt_1:2", "sex_1:2", "trt_1:3", "sex_1:3", "trt_2:3", "sex_2:3", "GLOBAL"]
    assert chisq == approx(
        [
            0.842333658165971,
            0.380257238113855,
            0.0478979955238052,
            0.0114916088910122,
            0.501329704417726,
            0.885768504813342,
            2.64791332206217,
        ]
    )
    assert [row["df"] for row in z.table] == [1, 1, 1, 1, 1, 1, 6]
    assert [row["p"] for row in z.table] == approx(
        [
            0.358730195286573,
            0.537465599023187,
            0.826762139167709,
            0.91463117057401,
            0.47891644358803,
            0.34662658485404,
            0.85156041240509,
        ]
    )
    assert (len(z.y), len(z.y[0])) == (684, 6)
    assert z.var[0][:2] == approx([4.04137266721796, -0.312025057114868])
    assert z.strata[:3] == ["1", "1", "1"]

    km = [
        0.0229608737916389,
        0.273084920692436,
        0.315090192896323,
        0.807806418004537,
        3.07384670294809,
    ]
    assert zph_rows(r.cox_zph(fm))[1] == approx([1.7149769484353, *km, 5.90768958455487])
    names, chisq = zph_rows(r.cox_zph(fm, terms=False, global_test=False))
    assert names == ["trtB_1:2", "sexm_1:2", "trtB_1:3", "sexm_1:3", "trtB_2:3", "sexm_2:3"]
    assert chisq == approx([1.7149769484353, *km])
    assert zph_rows(r.cox_zph(fa1, transform="log"))[1] == approx(
        [3.83990742474756, 2.01505036148748, 44.5122912407049, 0.317446482915003, 49.9496644808415]
    )


def test_cox_zph_matches_the_single_transition_fit(my, fm):
    """multi2.R: the 1:2 rows of the multi-state test are those of the entry:sct fit."""

    fit12 = r.coxph(
        "Surv(tstart, tstop, is_sct) ~ trt + sex",
        my,
        subset=[p == 0 for p in my["priortx"]],
        ties="breslow",
        init=fm.coefficients[:2],
        iter_max=0,
    )
    single = zph_rows(r.cox_zph(fit12, transform="log", global_test=False))[1]
    assert single == approx([1.58427800065214, 0.0648680923581096], rel=1e-7)
    multi = zph_rows(r.cox_zph(fm, transform="log", global_test=False))[1]
    assert multi[:2] == approx(single, rel=1e-7)
    assert multi[2:] == approx(
        [0.615830052571201, 0.215631619856629, 0.693856353168168, 2.29706549866508]
    )


def test_cox_zph_common_and_ph_coefficients(my, lms):
    fcm = r.coxph(["Surv(tstart, tstop, event) ~ trt", "1:2 + 2:3 ~ sex / common"], my, id="id")
    names, chisq = zph_rows(r.cox_zph(fcm))
    assert names == ["trt_1:2", "sex_1:2", "trt_1:3", "trt_2:3", "GLOBAL"]
    assert chisq == approx(
        [1.67685697390792, 1.74514715563413, 0.271449948954891, 0.911366844209396, 4.26740061394963]
    )
    # deviation: R fails with terms=TRUE ("length of 'dimnames' [2] not equal to array extent")
    fl2 = r.coxph(
        ["Surv(time, state) ~ 1", "1:4 + 2:4 + 3:4 ~ age + sex / common + shared"],
        lms,
        id="id",
        istate="cstate",
        ties="breslow",
    )
    expected = [
        0.16326571453227,
        2.4296760683156,
        1.12068483824111,
        2.86397056973532,
        4.97518714606762,
    ]
    names, chisq = zph_rows(r.cox_zph(fl2, terms=False))
    assert names == ["age", "sex", "ph(2:4/1:4)", "ph(3:4/1:4)", "GLOBAL"]
    assert chisq == approx(expected, rel=1e-7)
    names, chisq = zph_rows(r.cox_zph(fl2))
    assert names == ["age_1:4", "sex_1:4", "ph(2:4/1:4)", "ph(3:4/1:4)", "GLOBAL"]
    assert chisq == approx(expected, rel=1e-7)


def test_cox_zph_and_detail_with_strata_terms(fst):
    """Deviation: R fails with "subscript out of bounds"; the values are R's with
    coxph.getdata passing the model frame to stacker."""

    z = r.cox_zph(fst, transform="identity")
    names, chisq = zph_rows(z)
    assert names == ["age_1:2", "mspike_1:2", "age_1:3", "mspike_1:3", "GLOBAL"]
    assert chisq == approx(
        [2.04961946687477, 0.245190374256471, 21.2167714282854, 0.133600589423014, 24.6741303425182]
    )
    assert [row["p"] for row in z.table] == approx(
        [
            0.152244238084909,
            0.620482231028513,
            4.10159679008849e-06,
            0.71472765077863,
            5.84979903604056e-05,
        ]
    )
    assert (len(z.y), len(z.y[0])) == (969, 4)
    assert table(z.strata) == {"1": 59, "2": 56, "3": 368, "4": 486}
    assert zph_rows(r.cox_zph(fst))[1] == approx(
        [2.11002183275138, 0.409969816901824, 27.3762098382133, 0.267114599042807, 31.3224546218397]
    )

    detail = r.coxph_detail(fst)
    assert len(detail.time) == 436
    assert sum(detail.hazard) == approx(6.7141411830998)
    assert sum(detail.nevent) == 969
    assert sum(detail.varhaz) == approx(3.4212753499391)
    assert detail.strata == {"1": 52, "2": 50, "3": 161, "4": 173}
    assert (len(detail.x), len(detail.x[0])) == (2746, 4)


def test_coxph_detail(fa1):
    detail = r.coxph_detail(fa1)
    assert len(detail.time) == 291
    assert detail.time[:6] == [2, 4, 5, 6, 8, 9]
    assert sum(detail.hazard) == approx(7.20723191662297)
    assert sum(detail.nevent) == 975
    assert sum(detail.varhaz) == approx(9.29665731054213)
    assert (len(detail.x), len(detail.x[0])) == (2768, 4)
    assert (len(detail.y), len(detail.y[0])) == (2768, 3)
    assert detail.strata == {"1": 88, "2": 203}


# ---------------------------------------------------------------------------
# the other generics
# ---------------------------------------------------------------------------


def test_model_generics(mg, my_na, fa1, fna, fnax, fw):
    mm = r.model_matrix(fa1)
    assert mm["columns"] == ["age", "sexM"]
    assert mm["assign"] == [1, 2]
    assert len(mm["data"]) == 1384
    assert mm["data"][:3] == [[88, 0], [78, 0], [94, 1]]
    new = r.model_matrix(fa1, pd.DataFrame({"age": [60, 80], "sex": ["F", "M"]}))
    assert new["data"] == [[60, 0], [80, 1]]
    mmn = np.array(r.model_matrix(fna)["data"])
    assert mmn.shape == (1006, 4)
    assert (np.flatnonzero(np.isnan(mmn).any(axis=1)) + 1).tolist() == [422, 423, 424, 425]

    assert len(r.model_frame(fa1)["age"]) == 1384
    frame = r.model_frame(fna)
    expected_start = my_na.tstart.drop(index=[422, 424, 427]).tolist()
    assert frame["tstart"] == expected_start
    assert r.model_term_names(fa1) == ["age", "sex"]

    fitted = r.fitted(fa1)
    assert len(fitted) == 2768
    assert fitted[:3] == approx([-1.58445433621737, -1.71483228785931, -1.5313645221129])
    assert len(r.fitted(fnax)) == 1647
    weights = r.model_weights(fw)
    assert len(weights) == 2768
    assert weights[:6] == [2, 2, 2, 1, 2, 2]
    assert r.model_weights(fa1) is None

    assert r.loglik(fa1) == approx(-6157.72345973042)
    assert r.nobs(fa1) == 975
    assert r.aic(fa1) == approx(12323.4469194608)
    assert r.bic(fa1) == approx(12342.9766693448)


def test_refusals(mg, fa1):
    refused = [
        (lambda fit: r.basehaz(fit), "the basehaz function is not implemented for multi-state"),
        (lambda fit: r.yates(fit, "sex"), "multi-state coxph not yet supported"),
        (lambda fit: r.royston(fit), "not defined for multi-state models"),
        (lambda fit: r.concordance(fit), "concordance is not available for multi-state"),
        (lambda fit: r.brier(fit), "brier is not defined for multi-state coxph fits"),
        (
            lambda fit: r.survexp("~ 1", mg, ratetable=fit, rmap={"age": "age"}),
            "Invalid rate table",
        ),
        (
            lambda fit: r.pyears("Surv(etime, death) ~ 1", mg, ratetable=fit),
            "Invalid rate table",
        ),
    ]
    for call, message in refused:
        with pytest.raises(ValueError, match=message):
            call(fa1)
    new = pd.DataFrame({"age": [60], "sex": ["F"]})
    with pytest.raises(NotImplementedError, match="multi-state coxph fits yet"):
        r.survfit(fa1, newdata=new)
