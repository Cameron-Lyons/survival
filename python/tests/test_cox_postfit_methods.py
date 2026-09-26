"""Cox post-fit methods against R survival 3.8-12: labelled Schoenfeld residuals,
``fitted.coxph``, ``predict.coxph``'s newdata offset rule, ``type = "terms"`` term
selection and ``model.matrix.coxph``."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def stratified(lung):
    return r.coxph("Surv(time, status) ~ age + strata(sex)", lung)


@pytest.fixture(scope="module")
def offset_fit(lung):
    return r.coxph("Surv(time, status) ~ age + offset(sex)", lung)


OFFSET_NEWDATA = {"age": [60, 70], "sex": [0, 2], "time": [300, 300], "status": [1, 1]}


def test_schoenfeld_residuals_carry_times_and_strata(stratified):
    # residuals(f, "schoenfeld"): names(rr) are the death times, attr(rr, "strata")
    # is table(strata[deaths])
    schoenfeld = r.residuals(stratified, type="schoenfeld")
    assert isinstance(schoenfeld, r.CoxSchoenfeldResiduals)
    assert len(schoenfeld.values) == len(schoenfeld.time) == 165
    assert schoenfeld.values[:4] == approx(
        [9.450173354360132, 16.450173354360132, 2.450173354360132, 9.624988485320273]
    )
    assert schoenfeld.time[:6] == [11.0, 11.0, 11.0, 12.0, 13.0, 13.0]
    assert schoenfeld.strata == {"sex=1": 112, "sex=2": 53}
    assert schoenfeld.colnames == ["age"]
    scaled = r.residuals(stratified, type="scaledsch")
    assert scaled.values[:3] == approx([0.1478151112704106, 0.2452951448421968, 0.0503350776986243])
    assert scaled.time == schoenfeld.time
    assert scaled.strata == {"sex=1": 112, "sex=2": 53}


def test_schoenfeld_matrix_keeps_its_labels(lung):
    fit = r.coxph("Surv(time, status) ~ age + ph.ecog + strata(sex)", lung, na_action="na.omit")
    schoenfeld = r.residuals(fit, type="schoenfeld")
    assert len(schoenfeld.values) == 164
    assert schoenfeld.values[:3] == [
        approx([9.58462088675799, 0.793202402278552]),
        approx([16.58462088675799, -1.206797597721448]),
        approx([2.58462088675799, -0.206797597721448]),
    ]
    assert schoenfeld.time[:3] == [11.0, 11.0, 11.0]
    assert schoenfeld.colnames == ["age", "ph.ecog"]
    assert schoenfeld.strata == {"sex=1": 111, "sex=2": 53}
    # R loses attr(, "strata") of a multi-column scaledsch through %*%; it is kept here
    scaled = r.residuals(fit, type="scaledsch")
    assert scaled.values[:3] == [
        approx([0.1218043049192844, 1.8976807271602172], rel=1e-7),
        approx([0.2778523562948556, -2.6251341755962865], rel=1e-7),
        approx([0.0527643793612939, -0.0591982838565021], rel=1e-7),
    ]
    assert scaled.strata == {"sex=1": 111, "sex=2": 53}


def test_schoenfeld_residuals_of_an_unstratified_fit(offset_fit):
    schoenfeld = r.residuals(offset_fit, type="schoenfeld")
    assert schoenfeld.strata is None
    assert len(schoenfeld.time) == len(schoenfeld.values) == offset_fit.nevent


def test_fitted_is_the_linear_predictor(lung, stratified):
    # fitted.coxph is object$linear.predictors, centred at the overall means even for
    # a stratified fit (predict's default reference = "strata" is not)
    assert r.fitted(stratified)[:3] == approx(
        [0.1873218991700549, 0.0900339879837848, -0.1045418343887552]
    )
    assert r.fitted(stratified, type="terms") == stratified.linear_predictors
    # and it is not padded by naresid
    excluded = r.coxph("Surv(time, status) ~ age + ph.ecog", lung, na_action="na.exclude")
    assert len(r.fitted(excluded)) == 227
    assert len(r.predict(excluded)) == 228


def test_newdata_offset_is_centred_only_with_se_fit(lung, offset_fit):
    # predict.coxph without se.fit: newx %*% beta + newoffset
    assert r.predict(offset_fit, OFFSET_NEWDATA) == approx(
        [-0.0540711030038905, 2.1668645866894258], rel=1e-7
    )
    assert r.predict(offset_fit, OFFSET_NEWDATA, type="risk") == approx(
        [0.947364743627923, 8.730866205965397], rel=1e-7
    )
    # with se.fit the offset is centred at mean(offset)
    with_se = r.predict(offset_fit, OFFSET_NEWDATA, se_fit=True)
    assert with_se.fit == approx([-1.448807945109154, 0.772127744584163], rel=1e-7)
    assert with_se.se_fit == approx([0.0225138459850939, 0.0694782128787308], rel=1e-7)
    risk = r.predict(offset_fit, OFFSET_NEWDATA, type="risk", se_fit=True)
    assert risk.fit == approx([0.234850075480797, 2.164366577155446], rel=1e-7)
    assert risk.se_fit == approx([0.0109105097779419, 0.1022148624105543], rel=1e-7)
    # reference = "zero" with non-zero means takes the se.fit branch
    assert r.predict(offset_fit, OFFSET_NEWDATA, reference="zero") == approx(
        [-0.0691227039453663, 2.1518129857479495], rel=1e-7
    )
    # so the training rows as newdata give back the linear predictors
    rows = {name: lung[name][:3] for name in ("age", "sex")}
    assert r.predict(offset_fit, rows) == approx(
        [1.255238862566752, 1.122677448750762, 0.857554621118783], rel=1e-7
    )
    assert r.predict(offset_fit, rows) == approx(offset_fit.linear_predictors[:3])


def test_expected_keeps_the_offset_inside_exp(offset_fit):
    # R computes chaz * (exp(x beta) + offset) and gives 0.125540986978578,
    # 0.421612145805566; the expected count is chaz * exp(x beta + offset)
    expected = r.predict(offset_fit, OFFSET_NEWDATA, type="expected")
    assert expected[0] == approx(0.125540986978578, rel=1e-7)
    assert expected == approx([0.12554098697857782, 1.1569794717896869], rel=1e-7)
    survival_prob = r.predict(offset_fit, OFFSET_NEWDATA, type="survival")
    assert survival_prob[0] == approx(0.882019612367199, rel=1e-7)


def test_predict_terms_returns_the_selected_terms(lung):
    fit = r.coxph("Surv(time, status) ~ pspline(age) + sex", lung)
    picked = r.predict(fit, type="terms", terms=[2, 1], se_fit=True)
    assert picked.fit[:2] == [
        approx([0.204557245110859, 0.250852836526523], rel=1e-6),
        approx([0.204557245110859, 0.019412381543379], rel=1e-6),
    ]
    assert picked.se_fit[:2] == [
        approx([0.0664837054424138, 0.145678364265347], rel=1e-6),
        approx([0.0664837054424138, 0.109796047244423], rel=1e-6),
    ]
    assert r.predict(fit, type="terms", terms="sex") == [
        [row[1]] for row in r.predict(fit, type="terms")
    ]


def test_predict_terms_places_a_sparse_frailty_among_the_selection(lung):
    fit = r.coxph("Surv(time, status) ~ sex + frailty(inst, sparse=TRUE)", lung)
    full = r.predict(fit, type="terms", se_fit=True)
    swapped = r.predict(fit, type="terms", terms=[2, 1], se_fit=True)
    assert swapped.fit == [row[::-1] for row in full.fit]
    assert swapped.se_fit == [row[::-1] for row in full.se_fit]
    frailty = r.predict(fit, type="terms", terms=[2])
    assert frailty == [[row[1]] for row in full.fit]
    # R's predict.coxph.penal fails on a repeated sparse term; each repeat is a column
    repeated = r.predict(fit, type="terms", terms=[2, 1, 2], se_fit=True)
    assert repeated.fit == [[row[1], row[0], row[1]] for row in full.fit]
    assert repeated.se_fit == [[row[1], row[0], row[1]] for row in full.se_fit]


def test_predict_terms_pads_a_repeated_sparse_frailty(lung):
    fit = r.coxph(
        "Surv(time, status) ~ sex + frailty(inst, sparse=TRUE)", lung, na_action="na.exclude"
    )
    padded = r.predict(fit, type="terms", terms=[2, 2])
    assert len(padded) == 228
    assert {len(row) for row in padded} == {2}


def test_model_matrix_numbers_terms_as_r(lung):
    fit = r.coxph("Surv(time, status) ~ age + strata(sex) + ph.ecog", lung, na_action="na.omit")
    matrix = r.model_matrix(fit)
    # the strata term keeps its place in the numbering
    assert matrix["assign"] == [1, 3]
    assert matrix["columns"] == ["age", "ph.ecog"]
    assert len(matrix["data"]) == 227
    assert matrix["strata"] == fit.strata
    assert matrix["strata"][:3] == ["sex=1", "sex=1", "sex=1"]
    # coxph drops cluster() from the terms, so it leaves no gap
    clustered = r.coxph(
        "Surv(time, status) ~ age + cluster(inst) + ph.ecog", lung, na_action="na.omit"
    )
    assert r.model_matrix(clustered)["assign"] == [1, 2]
    assert r.model_matrix(clustered)["strata"] is None


def test_model_matrix_of_new_data(lung):
    fit = r.coxph("Surv(time, status) ~ age + strata(sex) + ph.ecog", lung, na_action="na.omit")
    data = {"age": [50, 60, None], "sex": [1, 2, 1], "ph.ecog": [0, 1, 2]}
    matrix = r.model_matrix(fit, data)
    # model.frame's default na.omit leaves out the incomplete row
    assert matrix == {
        "data": [[50.0, 0.0], [60.0, 1.0]],
        "columns": ["age", "ph.ecog"],
        "assign": [1, 3],
        "strata": ["sex=1", "sex=2"],
    }
    with pytest.raises(ValueError, match="strata"):
        r.model_matrix(fit, {"age": [50], "ph.ecog": [0]})
