"""``coxph()``'s assembly around the fitters, against R 4.5.3 with survival 3.8-12.

The offset is centred before every fitter, data without events get coxph's fit
skeleton, the concordance is the one the fit computed (penalized fits included, on
the final linear predictors and with the cluster), ``coxph.control`` options reach
the convergence warnings, and the penalized fit reports its inner-loop failures.
"""

import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
_coxph = survival.r._coxph


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


def _concordance(fit):
    keys = ("concordant", "discordant", "tied.x", "tied.y", "tied.xy", "concordance", "std")
    return [fit.concordance[key] for key in keys]


@pytest.fixture(scope="module")
def ovarian():
    data = {key: list(values) for key, values in datasets.load_ovarian().items()}
    data["o"] = [707.0 + value for value in data["ecog.ps"]]
    data["o_low"] = [-740.0 + value for value in data["ecog.ps"]]
    data["start"] = [value / 3.0 for value in data["futime"]]
    return data


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def separated():
    return {
        "start": [0.0] * 10,
        "t": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "s": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        "x": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "z": [0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.9, -0.2, 0.4, 0.0],
        "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    }


def _messages(record):
    return [str(item.message) for item in record]


def _fit_quietly(*args, **kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fit = r.coxph(*args, **kwargs)
    return fit, _messages(record)


# --- the offset is centred before the fit -------------------------------------


def test_offsets_near_the_exp_limit_fit_as_in_r(ovarian):
    # coxph(Surv(futime, fustat) ~ age + rx + offset(o), ovarian), o = 707 + ecog.ps
    fit, messages = _fit_quietly("Surv(futime, fustat) ~ age + rx + offset(o)", ovarian)
    assert messages == []
    assert fit.coefficients == approx([0.146610093918191, -0.907891623259919])
    assert fit.loglik == approx([-35.2529266085066, -28.0501454820662])
    assert fit.linear_predictors[:3] == approx(
        [710.824053048186, 711.140980088209, 710.964082220290]
    )
    assert r.residuals(fit)[:3] == approx([0.895219554461548, 0.695456121672056, 0.584419162782667])
    assert _concordance(fit) == approx([168, 50, 0, 0, 0, 0.7706422018348624, 0.0820389984083848])
    assert fit.offset == approx(ovarian["o"])
    # o = -740 + ecog.ps underflowed every risk score
    low = r.coxph("Surv(futime, fustat) ~ age + rx + offset(o_low)", ovarian)
    assert low.coefficients == approx([0.146610093918191, -0.907891623259919])
    assert low.loglik == approx([-35.2529266085066, -28.0501454820662])
    assert low.linear_predictors[:3] == approx(
        [-736.175946951814, -735.859019911791, -736.035917779710]
    )


def test_offset_centring_reaches_init_robust_counting_exact_and_penalized_fits(ovarian):
    robust = r.coxph(
        "Surv(futime, fustat) ~ age + rx + offset(o)", ovarian, init=[0.1, -0.5], robust=True
    )
    assert robust.coefficients == approx([0.146610093918182, -0.907891623259953])
    assert robust.var == [
        approx([0.00253673541304842, 0.01636229616508834], rel=1e-6),
        approx([0.01636229616508834, 0.32777387297796984], rel=1e-6),
    ]
    assert robust.naive_var == [
        approx([0.00231619256863716, 0.00194095632066125], rel=1e-6),
        approx([0.00194095632066125, 0.39538102971407296], rel=1e-6),
    ]
    assert robust.rscore == approx(2.60629303071705, rel=1e-6)
    counting = r.coxph("Surv(start, futime, fustat) ~ age + rx + offset(o)", ovarian)
    assert counting.coefficients == approx([0.119142473526917, -0.660084232893501])
    assert counting.loglik == approx([-29.3828810379053, -25.9652180257017])
    assert counting.linear_predictors[:3] == approx(
        [710.256106217087, 710.513656502110, 710.557252210120]
    )
    exact = r.coxph("Surv(futime, fustat) ~ age + rx + offset(o)", ovarian, ties="exact")
    assert exact.coefficients == approx([0.146610093918191, -0.907891623259918])
    assert exact.loglik == approx([-35.2529266085066, -28.0501454820662])
    ridge = r.coxph("Surv(futime, fustat) ~ ridge(age, rx, theta=1) + offset(o)", ovarian)
    assert ridge.coefficients == approx([0.121157944128344, -0.839313141425226])
    assert ridge.loglik == approx([-35.2529266085066, -28.2099900916321])
    assert ridge.linear_predictors[:3] == approx(
        [710.378302885373, 710.640210013195, 710.667626732499]
    )


@pytest.mark.parametrize("shift", [50.0, 707.0, -740.0])
def test_curves_are_on_survfit_scale_for_any_offset(ovarian, shift):
    # survfit.coxph's risk = exp(X %*% beta + offset - xcenter), with the mean offset in
    # xcenter, so R gives the same curves for every shift; risks of exp(lp) would round
    # the baseline survival to 1 (shift 50) or overflow (707, -740)
    data = dict(ovarian, o=[shift + value for value in ovarian["ecog.ps"]])
    fit = r.coxph("Surv(futime, fustat) ~ age + rx + offset(o)", data)
    curve = r.survfit(fit)
    assert curve.surv[:3] == approx([0.990179987806773, 0.979324577815282, 0.966546096122788])
    assert curve.cumhaz[:3] == approx(
        [0.00986854651275763, 0.02089215124030774, 0.03402628760372377]
    )
    assert curve.std_err[:3] == approx([0.0118712460770787, 0.0202149778939414, 0.0293319303754243])
    assert r.basehaz(fit).hazard[:3] == approx(
        [0.00986854651275763, 0.02089215124030774, 0.03402628760372377]
    )
    assert r.basehaz(fit, centered=False).hazard[:3] == approx(
        [1.02218186765594e-05, 2.16400441002742e-05, 3.52443535298833e-05]
    )
    newdata = {"age": [50, 60], "rx": [1, 2], "o": [shift, shift + 2.0]}
    curves = r.survfit(fit, newdata)
    assert [list(row) for row in curves.surv[:3]] == [
        approx([0.998541928933378, 0.981335052618955]),
        approx([0.996915712437477, 0.960897106751261]),
        approx([0.994981604782447, 0.937101155152480]),
    ]
    assert [list(row) for row in curves.std_err[:3]] == [
        approx([0.00201187062820124, 0.02286885101362282]),
        approx([0.00364002826848285, 0.03912943523690722]),
        approx([0.00548271019033175, 0.05709001852477361]),
    ]
    assert r.survfit(fit, stype=1, ctype=1).surv[:3] == approx(
        [0.989629373513725, 0.977807859795110, 0.963909470409273]
    )


def test_expected_counts_keep_their_scale_with_an_offset(ovarian):
    # R 3.8-12 gives these for shift 50; for 707 its agsurv call gets an infinite risk
    data = dict(ovarian, o=[50.0 + value for value in ovarian["ecog.ps"]])
    fit = r.coxph("Surv(futime, fustat) ~ age + rx + offset(o)", data)
    expected = r.predict(fit, type="expected", se_fit=True)
    assert expected.fit[:3] == approx([0.104780445538452, 0.304543878327944, 0.415580837217333])
    assert expected.se_fit[:3] == approx([0.106033726389833, 0.229533138188893, 0.250513978450509])
    survival_prob = r.predict(fit, type="survival", se_fit=True)
    assert survival_prob.se_fit[:3] == approx(
        [0.0954857261059137, 0.1692714321196184, 0.1653284140605831]
    )


# --- data without events --------------------------------------------------------


@pytest.fixture(scope="module")
def censored():
    return {
        "start": [0, 0, 1, 1],
        "stop": [2, 3, 4, 5],
        "status": [0, 0, 0, 0],
        "x": [1, 2, 3, 4],
        "w": [1, 2, 3, 4],
        "off": [0.1, 0.2, 0.3, 0.4],
        "x01": [0, 1, 1, 1],
        "xi": [1, math.inf, 3, 4],
        "w0": [0, 2, 3, 4],
    }


@pytest.mark.parametrize(
    "formula",
    ["Surv(start, stop, status) ~ x + offset(off)", "Surv(stop, status) ~ x + offset(off)"],
)
def test_data_without_events_get_coxph_skeleton(censored, formula):
    fit = r.coxph(formula, censored, weights="w")
    assert math.isnan(fit.coefficients[0])
    assert fit.var == [[0.0]]
    assert fit.loglik == [0.0, 0.0]
    assert (fit.iter, fit.score, fit.wald_test, fit.nevent) == (0, 0.0, 0.0, 0)
    # colMeans(X), unweighted; the linear predictors are the centred offset
    assert fit.means == [2.5]
    assert fit.linear_predictors == approx([-0.15, -0.05, 0.05, 0.15])
    assert fit.residuals == [0.0] * 4
    counts = _concordance(fit)
    assert counts[:5] == [0.0] * 5
    assert math.isnan(counts[5])
    assert math.isnan(counts[6])


def test_data_without_events_skip_init_penalties_and_nocenter(censored):
    # R returns before checking init, before coxpenal.fit and with plain colMeans
    fit = r.coxph("Surv(stop, status) ~ x", censored, init=[1.0, 2.0])
    assert math.isnan(fit.coefficients[0])
    ridge = r.coxph("Surv(start, stop, status) ~ ridge(x, theta=1)", censored)
    assert ridge.penalized is None
    assert math.isnan(ridge.coefficients[0])
    assert (ridge.means, ridge.iter) == ([2.5], 0)
    assert r.coxph("Surv(stop, status) ~ x01", censored).means == [0.75]


@pytest.mark.parametrize("response", ["Surv(stop, status)", "Surv(start, stop, status)"])
def test_data_without_events_skip_the_predictor_and_weight_checks(censored, response):
    # coxph() returns before its infinite-predictor check and the fitters' weights > 0
    infinite = r.coxph(f"{response} ~ xi", censored)
    assert math.isnan(infinite.coefficients[0])
    assert infinite.means == [math.inf]
    assert infinite.linear_predictors == [0.0] * 4
    ridge = r.coxph(f"{response} ~ ridge(xi, theta=1)", censored)
    assert math.isnan(ridge.coefficients[0])
    zero_weight = r.coxph(f"{response} ~ x", censored, weights="w0")
    assert math.isnan(zero_weight.coefficients[0])
    assert zero_weight.means == [2.5]
    with_event = dict(censored, status=[1, 0, 0, 0])
    with pytest.raises(ValueError, match="data contains an infinite predictor"):
        r.coxph(f"{response} ~ xi", with_event)
    with pytest.raises(ValueError, match="Invalid weights, must be >0"):
        r.coxph(f"{response} ~ x", with_event, weights="w0")
    with pytest.raises(ValueError, match="Invalid weights, must be >0"):
        r.coxph(f"{response} ~ ridge(x, theta=1)", with_event, weights="w0")


# --- one concordance ------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (
            "Surv(time, status) ~ age + frailty(inst)",
            [10948, 8832, 46, 28, 0, 0.553364269141531, 0.0250590245333370],
        ),
        (
            "Surv(time, status) ~ age + sex + frailty(inst)",
            [11993, 7810, 23, 28, 0, 0.605492787249067, 0.0255724474576494],
        ),
        (
            "Surv(time, status) ~ age + frailty(inst) + cluster(inst)",
            [10948, 8832, 46, 28, 0, 0.553364269141531, 0.0245803339668568],
        ),
        (
            "Surv(time, status) ~ age + pspline(meal.cal) + cluster(inst)",
            [7112, 5571, 15, 17, 0, 0.560678847062529, 0.0273120470896529],
        ),
        (
            "Surv(time, status) ~ ridge(age, sex, theta=1) + cluster(inst)",
            [11814, 7704, 308, 28, 0, 0.603651770402502, 0.0198478101998055],
        ),
    ],
)
def test_penalized_concordance_uses_the_final_predictors_and_the_cluster(lung, formula, expected):
    fit, _ = _fit_quietly(formula, lung, na_action="omit")
    assert _concordance(fit) == approx(expected, rel=1e-6)
    # the engine fit carries the same concordance
    stored = fit.fit.concordance
    assert stored.concordance[0] == fit.concordance["concordance"]
    assert stored.count[0].concordant == fit.concordance["concordant"]


def test_ignored_cluster_still_enters_the_concordance(lung):
    with pytest.warns(RuntimeWarning, match="cluster specified with robust=FALSE"):
        fit = r.coxph(
            "Surv(time, status) ~ age + sex + cluster(inst)", lung, robust=False, na_action="omit"
        )
    assert not fit.robust
    assert _concordance(fit) == approx(
        [11814, 7704, 308, 28, 0, 0.603651770402502, 0.0198478101998055], rel=1e-6
    )


# --- timefix ---------------------------------------------------------------------


def test_timefix_merges_near_tied_times(lung):
    times = [float(value) for value in lung["time"]]
    noisy = dict(lung, time=[t * (1.0 + 1e-12 * (i % 3)) for i, t in enumerate(times)])
    exact = r.coxph("Surv(time, status) ~ age + sex", lung)
    fixed = r.coxph("Surv(time, status) ~ age + sex", noisy)
    unfixed = r.coxph("Surv(time, status) ~ age + sex", noisy, timefix=False)
    assert fixed.coefficients == approx(exact.coefficients, rel=1e-10)
    assert len(set(fixed.y.time)) == len(set(exact.y.time)) < len(set(unfixed.y.time))
    assert unfixed.coefficients != approx(exact.coefficients, rel=1e-6)


# --- coxph.control ---------------------------------------------------------------


def test_coxph_control_defaults_and_checks(separated):
    assert r.coxph_control() == {
        "eps": 1e-9,
        "toler.chol": pytest.approx(1.818989403545856e-12, rel=1e-15),
        "iter.max": 20,
        "toler.inf": pytest.approx(math.sqrt(1e-9), rel=1e-15),
        "outer.max": 10,
        "timefix": True,
        "survcheckallow": "gap",
    }
    assert r.coxph_control(**{"toler.inf": 1e-3, "outer.max": 5})["outer.max"] == 5
    # as.integer() truncates
    truncated = r.coxph_control(iter_max=2.7, outer_max=5.9)
    assert (truncated["iter.max"], truncated["outer.max"]) == (2, 5)
    fit, messages = _fit_quietly("Surv(t, s) ~ x + z", separated, **{"iter.max": 2.7})
    assert messages == ["Ran out of iterations and did not converge"]
    assert fit.iter == 3
    assert fit.coefficients == approx([4.3477042340223733, 0.2261679508977732])
    with pytest.warns(RuntimeWarning, match="tolerance should be < eps"):
        r.coxph_control(eps=1e-12, toler_chol=1e-10)
    with pytest.raises(ValueError, match="Invalid value for iterations"):
        r.coxph_control(iter_max=-1)
    with pytest.raises(TypeError, match="Invalid value for iterations"):
        r.coxph_control(iter_max="20")
    with pytest.raises(TypeError, match="invalid value for outer.max"):
        r.coxph_control(outer_max=True)
    with pytest.raises(TypeError, match="Invalid convergence criteria"):
        r.coxph_control(eps="1e-9")
    with pytest.raises(ValueError, match="invalid value for toler.chol"):
        r.coxph_control(toler_chol=0)
    with pytest.raises(ValueError, match="The toler.inf setting must be >0"):
        r.coxph_control(toler_inf=0)
    with pytest.raises(ValueError, match="invalid value for outer.max"):
        r.coxph_control(outer_max=0)
    with pytest.raises(TypeError, match="timefix must be TRUE or FALSE"):
        r.coxph_control(timefix="yes")
    with pytest.raises(TypeError, match="unused argument"):
        r.coxph_control(bogus=1)


def test_toler_inf_reaches_the_convergence_warning(separated):
    loud = "Loglik converged before variable 1; coefficient may be infinite. "
    fit, messages = _fit_quietly("Surv(t, s) ~ x + z", separated)
    assert fit.coefficients == approx([22.442733892856403, 0.258527583204505], rel=1e-7)
    assert messages == [loud]
    for kwargs in ({"control": {"toler.inf": 1e6}}, {"toler.inf": 1e6}, {"toler_inf": 1e6}):
        assert _fit_quietly("Surv(t, s) ~ x + z", separated, **kwargs)[1] == []
    # with control given, coxph's ... options are ignored, as in R
    ignored = _fit_quietly(
        "Surv(t, s) ~ x + z", separated, control=r.coxph_control(), toler_inf=1e6
    )[1]
    assert ignored == [loud]
    assert _fit_quietly("Surv(t, s) ~ x + z", separated, **{"outer.max": 5})[1] == [loud]
    with pytest.raises(ValueError, match="Argument bogus not matched"):
        r.coxph("Surv(t, s) ~ x + z", separated, bogus=1)


def test_each_fitter_has_its_own_convergence_rule(separated):
    beta = "Loglik converged before variable 1; beta may be infinite. "
    for formula, ties in [
        ("Surv(start, t, s) ~ x + z", "efron"),
        ("Surv(start, t, s) ~ x + z", "breslow"),
        ("Surv(t, s) ~ x + z", "exact"),
        ("Surv(start, t, s) ~ x + z", "exact"),
    ]:
        assert _fit_quietly(formula, separated, ties=ties)[1] == [beta], (formula, ties)
    assert _fit_quietly("Surv(start, t, s) ~ x + z", separated, **{"toler.inf": 1e6})[1] == []
    for formula in ("Surv(start, t, s) ~ x + z", "Surv(t, s) ~ x + z"):
        messages = _fit_quietly(formula, separated, iter_max=3)[1]
        assert messages == ["Ran out of iterations and did not converge"]


def test_robust_fits_test_convergence_with_the_naive_variance(separated):
    coefficient = "Loglik converged before variable 1; coefficient may be infinite. "
    beta = "Loglik converged before variable 1; beta may be infinite. "
    for formula, kwargs, message, var in [
        (
            "Surv(t, s) ~ x + z + cluster(id)",
            {},
            coefficient,
            [0.44372942314433855, 0.19240479257421977, 0.4507052716267313],
        ),
        (
            "Surv(t, s) ~ x + z",
            {"robust": True},
            coefficient,
            [0.3201858520764012, 0.34557372333926617, 0.9167766801508698],
        ),
        (
            "Surv(start, t, s) ~ x + z + cluster(id)",
            {},
            beta,
            [0.4437300297044401, 0.1924055464165384, 0.450706103605473],
        ),
        # an id with repeated events makes the fit robust by default
        (
            "Surv(start, t, s) ~ x + z",
            {"id": "id"},
            beta,
            [0.4437300297044401, 0.1924055464165384, 0.450706103605473],
        ),
    ]:
        fit, messages = _fit_quietly(formula, separated, **kwargs)
        assert messages == [message], (formula, kwargs)
        assert [fit.var[0][0], fit.var[0][1], fit.var[1][1]] == approx(var, rel=1e-6)
        assert fit.naive_var[0][0] == approx(5.066063e8, rel=1e-6)


def test_eps_below_toler_chol_warns(separated):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fit = r.coxph("Surv(t, s) ~ x + z", separated, control={"eps": 1e-12, "toler.chol": 1e-10})
    assert _messages(record) == [
        "For numerical accuracy, tolerance should be < eps",
        "Ran out of iterations and did not converge",
    ]
    assert fit.iter == 21


# --- the inner loop of a penalized fit --------------------------------------------


def test_inner_loop_failures_warn_as_coxpenal_fit(lung):
    fit, messages = _fit_quietly("Surv(time, status) ~ pspline(age, df=4) + sex", lung, iter_max=2)
    assert messages == ["Inner loop failed to coverge for iterations 1 2 3 4"]
    assert fit.iter == [4, 8]
    assert _fit_quietly("Surv(time, status) ~ pspline(age, df=4) + sex", lung, iter_max=1)[1] == []
    kidney = datasets.load_kidney()
    fit, messages = _fit_quietly("Surv(time, status) ~ age + sex + frailty(id)", kidney)
    assert messages == ["Inner loop failed to coverge for iterations 3"]
    assert fit.iter == [7, 65]
    assert _concordance(fit) == approx(
        [1596, 357, 17, 11, 0, 0.814467005076142, 0.0331689748959826], rel=1e-6
    )
    for iter_max in (0, 1):
        formula = "Surv(time, status) ~ age + sex + frailty(id)"
        assert _fit_quietly(formula, kidney, iter_max=iter_max)[1] == []


# --- the model frame ---------------------------------------------------------------


def test_model_frame_is_rebuilt_without_model_true(ovarian):
    formula = "Surv(futime, fustat) ~ age + strata(rx) + offset(o)"
    subset = [age > 50 for age in ovarian["age"]]
    fit = r.coxph(formula, ovarian, subset=subset)
    kept = r.coxph(formula, ovarian, subset=subset, model=True)
    assert fit.model is None
    assert _coxph._coxph_model_frame(kept) is kept.model
    assert _coxph._coxph_model_frame(fit) == kept.model
    assert "_frame" not in repr(fit)
