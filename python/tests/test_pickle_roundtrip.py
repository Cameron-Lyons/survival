"""Pickle, copy and deepcopy of the survival.r results, as R's saveRDS/readRDS of a fit.

The native classes the results keep (the Cox, penalised Cox, survreg, survfit and
concordance objects of ``survival._survival``) reduce to ``_unpickle`` with their encoded
state, so a restored result answers predict, summary, residuals and survfit exactly as the
original.  The reference values are R 4.5.3 with survival 3.8-12 on ``lung``.
"""

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
native = survival._survival


def round_trip(obj):
    return pickle.loads(pickle.dumps(obj))  # noqa: S301 - the test's own pickle


COPIES = {
    "pickle": round_trip,
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


def approx(values):
    if isinstance(values, list):
        values = np.asarray(values, dtype=float)
    return pytest.approx(values, rel=1e-8, abs=1e-12, nan_ok=True)


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


@pytest.fixture(scope="module")
def mgus2_states():
    data = survival.datasets.load_mgus2()
    pcm = [flag == 1 for flag in data["pstat"]]
    etime = [
        ptime if is_pcm else futime
        for ptime, futime, is_pcm in zip(data["ptime"], data["futime"], pcm, strict=True)
    ]
    event = [
        "pcm" if is_pcm else ("death" if death == 1 else "censor")
        for death, is_pcm in zip(data["death"], pcm, strict=True)
    ]
    weights = [1.0 + (i % 3) for i in range(len(etime))]
    return {
        **data,
        "etime": etime,
        "event": RFactor(event, ["censor", "pcm", "death"]),
        "w": weights,
    }


@pytest.fixture(scope="module")
def cox_fits(lung):
    # (start, stop] data, whose rows the fitter sorts into its own order
    split = r.survSplit(
        "Surv(time, status) ~ age + sex + ph.ecog", lung, cut=[100], start="t0", end="t1"
    )
    return {
        "counting": r.coxph("Surv(t0, t1, status) ~ age + sex + strata(ph.ecog)", split),
        "exact": r.coxph("Surv(time, status) ~ age + sex", lung, ties="exact"),
        "plain": r.coxph("Surv(time, status) ~ age + sex", lung),
        "strata": r.coxph("Surv(time, status) ~ age + strata(sex)", lung),
        "robust": r.coxph("Surv(time, status) ~ age + sex + cluster(inst)", lung),
        "pspline": r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", lung),
        "frailty": r.coxph("Surv(time, status) ~ age + frailty(inst)", lung),
    }


def assert_same_native(original, restored):
    """A copy of a native object is a new object with every attribute unchanged."""
    assert type(restored) is type(original)
    if hasattr(original, "__int__"):  # a fieldless enum
        assert restored == original
        return
    assert restored is not original
    for name in dir(original):
        if name.startswith("_"):
            continue
        value = getattr(original, name)
        if callable(value):
            continue
        again = getattr(restored, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(again, value)
            assert again.flags.f_contiguous == value.flags.f_contiguous
        elif type(value).__module__ == "survival._survival":
            assert_same_native(value, again)
        elif (
            isinstance(value, list) and value and type(value[0]).__module__ == "survival._survival"
        ):
            for item, item_again in zip(value, again, strict=True):
                assert_same_native(item, item_again)
        else:
            np.testing.assert_equal(again, value, err_msg=name)


def test_result_classes_live_in_the_extension_module():
    for cls in (
        native.CoxPHFit,
        native.CoxpenalFit,
        native.CoxPenalty,
        native.TieMethod,
        native.ConcordanceFit,
        native.SurvregFit,
        native.SurvregControl,
        native.SurvregDistribution,
        native.SurvfitKMResult,
        native.SurvfitCounts,
        native.SurvfitInfluence,
        native.SurvfitAJResult,
        native.SurvfitAJCounts,
        native.SurvfitAJInfluence,
        native.AnovaCoxphResult,
        native.AnovaRow,
        native.YatesContrast,
        native.SurvCheckFlags,
        native.SurvCheckTransitions,
        native.SurvCheckEvents,
        native.TcutResult,
        native.SplineBasisResult,
    ):
        assert cls.__module__ == "survival._survival"
        assert round_trip(cls) is cls


@pytest.mark.parametrize("how", COPIES)
@pytest.mark.parametrize(
    "kind", ["plain", "strata", "counting", "exact", "robust", "pspline", "frailty"]
)
def test_coxph_round_trip_keeps_every_method(cox_fits, kind, how):
    fit = cox_fits[kind]
    again = COPIES[how](fit)

    assert type(again) is type(fit)
    assert_same_native(fit.fit, COPIES[how](fit.fit))
    assert r.coef(again) == approx(r.coef(fit))
    assert r.vcov(again) == approx(r.vcov(fit))
    for type_ in ("lp", "risk", "expected", "terms"):
        assert np.asarray(r.predict(again, type=type_)) == approx(
            np.asarray(r.predict(fit, type=type_))
        )
    for type_ in ("martingale", "deviance", "score", "dfbeta"):
        if kind == "exact" and type_ not in ("martingale", "deviance"):
            continue  # R has no score residuals for the exact partial likelihood
        assert np.asarray(r.residuals(again, type=type_)) == approx(
            np.asarray(r.residuals(fit, type=type_))
        )
    if kind != "exact":
        schoenfeld = r.residuals(fit, type="schoenfeld")
        assert r.residuals(again, type="schoenfeld").values == approx(np.asarray(schoenfeld.values))
    np.testing.assert_equal(r.model_summary(again), r.model_summary(fit))
    if kind != "frailty":
        curve, curve_again = r.survfit(fit), r.survfit(again)
        assert curve_again.surv == approx(curve.surv)
        assert curve_again.std_err == approx(curve.std_err)


def test_restored_coxph_reproduces_r(cox_fits):
    fit = round_trip(cox_fits["plain"])
    # R's coefficients, first three linear predictors and deviance residuals.
    assert r.coef(fit) == approx([0.0170453318454113, -0.5132185171083844])
    assert list(r.predict(fit, type="lp"))[:3] == approx(
        [0.3995046957042455, 0.2972327046317776, 0.0926887224868421]
    )
    assert list(r.residuals(fit, type="deviance"))[:3] == approx(
        [0.00439643487827233, -0.43923325853091705, -2.50192695341419524]
    )
    pspline = round_trip(cox_fits["pspline"])
    assert pspline.loglik == approx([-749.909801390395, -741.026634314088])
    counting = round_trip(cox_fits["counting"])
    assert r.coef(counting) == approx([0.0107721611223054, -0.5535169257963722])
    assert counting.loglik == approx([-566.906517584651, -560.674008643026])
    assert list(r.residuals(counting, type="martingale"))[:3] == approx(
        [-0.137898486011657, 0.154439304726364, -0.153778766711812]
    )
    exact = copy.deepcopy(cox_fits["exact"])
    assert r.coef(exact) == approx([0.0170603236756335, -0.5138634696379791])
    assert exact.loglik == approx([-731.077044479620, -724.016323095385])
    assert list(r.residuals(exact, type="deviance"))[:2] == approx(
        [0.00526424745514941, -0.43784124918713996]
    )
    frailty = copy.deepcopy(cox_fits["frailty"])
    assert frailty.loglik == approx([-744.799933702646, -742.705318882127])
    assert r.coef(frailty) == approx([0.0186357548299227])


def test_penalized_fit_keeps_its_penalty_state(cox_fits):
    for kind in ("pspline", "frailty"):
        penalized = cox_fits[kind].penalized
        again = round_trip(penalized)
        assert_same_native(penalized, again)


def test_callback_penalty_pickles_its_callable():
    penalty = native.CoxPenalty.callback(np.add, diag=False, sparse=True)
    again = round_trip(penalty)
    assert (again.kind, again.diag, again.sparse) == ("callback", False, True)
    ridge = native.CoxPenalty.ridge(theta=1.0)
    assert copy.deepcopy(ridge).kind == "ridge"


@pytest.mark.parametrize("how", COPIES)
@pytest.mark.parametrize("formula", ["age + sex", "age + strata(sex)"])
def test_survreg_round_trip(lung, formula, how):
    fit = r.survreg(f"Surv(time, status) ~ {formula}", lung)
    again = COPIES[how](fit)
    assert_same_native(fit.fit, COPIES[how](fit.fit))
    assert_same_native(fit.control, COPIES[how](fit.control))
    assert r.coef(again) == approx(r.coef(fit))
    for type_ in ("response", "lp", "quantile"):
        assert np.asarray(r.predict(again, type=type_)) == approx(
            np.asarray(r.predict(fit, type=type_))
        )
    for type_ in ("response", "deviance", "working", "dfbeta"):
        assert np.asarray(r.residuals(again, type=type_)) == approx(
            np.asarray(r.residuals(fit, type=type_))
        )
    np.testing.assert_equal(r.model_summary(again), r.model_summary(fit))


def test_restored_survreg_reproduces_r(lung):
    fit = round_trip(r.survreg("Surv(time, status) ~ age + sex", lung))
    assert r.coef(fit) == approx([6.2748530584187714, -0.0122570255889486, 0.3820851396588851])
    assert fit.scale == approx([0.75405094764082])
    assert list(r.predict(fit, type="response"))[:3] == approx(
        [314.164993369631, 338.140151189303, 391.719007478334]
    )


@pytest.mark.parametrize("how", COPIES)
def test_survfit_km_round_trip(lung, how):
    weighted = {**lung, "w": [1.0 + (i % 3) for i in range(len(lung["time"]))]}
    for fit in (
        r.survfit("Surv(time, status) ~ sex", lung, influence=True),
        r.survfit("Surv(time, status) ~ 1", weighted, weights="w"),
    ):
        again = COPIES[how](fit)
        assert_same_native(fit.engine, COPIES[how](fit.engine))
        summary = r.summary_survfit(fit, times=[100, 365])
        assert r.summary_survfit(again, times=[100, 365]).surv == approx(summary.surv)


def test_restored_survfit_reproduces_r(lung):
    fit = round_trip(r.survfit("Surv(time, status) ~ sex", lung, influence=True))
    assert r.summary_survfit(fit, times=[100, 365]).surv == approx(
        [0.826086956521739, 0.336087834639379, 0.922088353413655, 0.526463030185906]
    )
    # The influence matrices keep R's column-major layout.
    assert all(curve.influence.values.flags.f_contiguous for curve in fit.influence_surv)


@pytest.mark.parametrize("how", COPIES)
def test_survfit_aj_round_trip(mgus2_states, how):
    for fit in (
        r.survfit("Surv(etime, event) ~ sex", mgus2_states, id="id", influence=True),
        r.survfit("Surv(etime, event) ~ 1", mgus2_states, id="id", weights="w"),
    ):
        again = COPIES[how](fit)
        assert_same_native(fit.engine, COPIES[how](fit.engine))
        summary = r.summary_survfit(fit, times=[100, 200])
        assert r.summary_survfit(again, times=[100, 200]).pstate == approx(summary.pstate)


@pytest.mark.parametrize("how", COPIES)
def test_concordance_round_trip(cox_fits, how):
    result = r.concordance(cox_fits["plain"])
    again = COPIES[how](result)
    assert again.concordance == approx(result.concordance)
    assert again.var == approx(result.var)
    native_fit = cox_fits["plain"].fit.concordance
    assert_same_native(native_fit, COPIES[how](native_fit))


@pytest.mark.parametrize("how", COPIES)
def test_other_native_results_round_trip(lung, mgus2_states, how):
    anova = COPIES[how](r.anova(r.coxph("Surv(time, status) ~ age + sex", lung)))
    assert [row.name for row in anova.rows] == ["NULL", "age", "sex"]
    assert [row.loglik for row in anova.rows] == approx(
        [-749.909801390395, -747.789352207734, -742.848245783770]
    )
    assert [row.chisq for row in anova.rows[1:]] == approx([4.24089836532130, 9.88221284792735])
    assert [row.p_value for row in anova.rows[1:]] == approx(
        [0.03946128102607811, 0.00166884120417826]
    )
    yates = r.yates(r.coxph("Surv(time, status) ~ age + factor(ph.ecog)", lung), "ph.ecog")
    (test,) = COPIES[how](yates).test
    assert (test.name, test.df) == ("global", 3)
    assert test.chisq == approx(16.6275607944514)
    check = COPIES[how](r.survcheck("Surv(etime, event) ~ 1", mgus2_states, id="id"))
    assert check.transitions.counts == [[115, 860, 409], [0, 0, 0], [0, 0, 0]]
    assert check.flag.overlap == check.flag.gap == 0
    assert_same_native(check.events, COPIES[how](check.events))
    tcut = COPIES[how](r.tcut([1.0, 5.0, 10.0], [0, 4, 8, 12]))
    assert tcut.labels == ["0+ thru  4", "4+ thru  8", "8+ thru 12"]
    basis = COPIES[how](r.nsk(lung["age"], df=3))
    assert (basis.n_rows, basis.n_cols, basis.knots) == (228, 3, [59.0, 67.0])
    assert basis.basis[:3] == approx([-0.0417023472422705, 0.195461467161017, 0.842748609639831])


def test_tie_method_and_distribution_enums_round_trip():
    for value in (native.TieMethod.Efron, native.SurvregFamily.T, native.SurvregTransform.Log):
        assert round_trip(value) == value


def test_corrupt_state_is_refused():
    with pytest.raises(ValueError, match="not a pickled CoxPHFit"):
        native._unpickle(native.CoxPHFit, b"\x00")
    with pytest.raises(ValueError, match="cannot restore"):
        native._unpickle(native.SurvivalData, b"")


def test_fitted_coxph_goes_to_a_worker_process(cox_fits):
    fit = cox_fits["plain"]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        lp = pool.submit(r.predict, fit, type="lp").result(timeout=120)
    assert np.asarray(lp) == approx(np.asarray(r.predict(fit, type="lp")))
