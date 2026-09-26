"""survfit.coxph regressions against R 4.5.3 with survival 3.8-12.

``start.time`` builds the curves from the rows still at risk at that time, and penalized
fits give ``id`` curves.
"""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-10):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def lung_fit(lung):
    return r.coxph("Surv(time, status) ~ age + sex", lung)


@pytest.fixture(scope="module")
def heart_fit():
    return r.coxph("Surv(start, stop, event) ~ age + transplant", datasets.load_heart())


def _heart_newdata(**changes):
    newdata = {
        "start": [0, 50, 0, 20],
        "stop": [50, 400, 20, 1000],
        "event": [0, 0, 0, 0],
        "age": [-5, -5, 3, 3],
        "transplant": [0, 1, 0, 1],
        "pid": ["b", "b", "a", "a"],
    }
    return {**newdata, **changes}


def _strata_data():
    # set.seed(1); data.frame(time = c(1:10, 21:30), status = rep(c(1, 0, 1, 1, 0), 4),
    #                         g = rep(1:2, each = 10), x = rnorm(20))
    x = [
        -0.62645381074233242,
        0.18364332422208224,
        -0.83562861241004716,
        1.5952808021377916,
        0.32950777181536051,
        -0.82046838411801526,
        0.48742905242848528,
        0.73832470512921733,
        0.57578135165349231,
        -0.30538838715635602,
        1.511781168450848,
        0.38984323641143109,
        -0.62124058054180376,
        -2.2146998871774999,
        1.1249309181431082,
        -0.044933609015230851,
        -0.016190263098946087,
        0.94383621068529922,
        0.82122119509808855,
        0.59390132121750883,
    ]
    return {
        "time": [*range(1, 11), *range(21, 31)],
        "status": [1, 0, 1, 1, 0] * 4,
        "g": [1] * 10 + [2] * 10,
        "x": x,
    }


# ---------------------------------------------------------------------------
# start.time
# ---------------------------------------------------------------------------


def test_start_time_builds_the_curve_from_the_rows_at_risk(lung_fit):
    # survfit(fit, start.time = 100)
    curve = r.survfit(lung_fit, start_time=100)
    assert curve.start_time == 100.0
    assert curve.n == [196]
    assert len(curve.time) == 164
    assert curve.time[:3] == [105.0, 107.0, 110.0]
    assert curve.n_risk[:3] == [196.0, 194.0, 192.0]
    assert curve.surv[:3] == approx([0.995026715313351, 0.985024763886337, 0.980024944029252])
    assert curve.cumhaz[:3] == approx([0.00498569262283097, 0.0150884971247201, 0.0201772545503708])
    assert curve.std_err[:3] == approx(
        [0.00498652602650291, 0.0087159245458067, 0.0100957264656873]
    )
    assert curve.lower[:3] == approx([0.985349277672496, 0.968340598802159, 0.960823533542582])
    assert curve.upper[:3] == approx([1.0, 1.0, 0.999610081758028])
    assert r.survfit(lung_fit, **{"start.time": 100}, stype=1).surv[:3] == approx(
        [0.995011625582318, 0.984985370091693, 0.979969727221027]
    )
    newdata = r.survfit(lung_fit, {"age": [50, 60], "sex": [1, 2]}, start_time=100)
    assert newdata.surv[0] == approx([0.995074030974004, 0.996501018866574])
    assert newdata.surv[2] == approx([0.980213558640518, 0.985914819509285])
    assert newdata.std_err[2] == approx([0.010350446313988, 0.00729606222312575])
    # the cached curve of the whole fit is untouched
    assert r.survfit(lung_fit).time[0] == 5.0


def test_start_time_reads_the_stop_time_of_counting_data(heart_fit):
    curve = r.survfit(heart_fit, start_time=100)
    assert curve.n == [54]
    assert len(curve.time) == 52
    assert curve.time[:3] == [100.0, 102.0, 109.0]
    assert curve.n_risk[:3] == [50.0, 49.0, 48.0]
    assert curve.surv[:3] == approx([0.979448223850899, 0.958850707206296, 0.958850707206296])
    assert curve.std_err[:3] == approx([0.0214277667125728, 0.0315718279279994, 0.0315718279279994])
    # survfit(fit, newdata = nd, id = nd$pid, start.time = 100), numeric ids 1, 2
    subjects = r.survfit(heart_fit, _heart_newdata(pid=[1, 1, 2, 2]), id="pid", start_time=100)
    assert subjects.strata == {"1": 26, "2": 43}
    assert subjects.n == [54, 54]
    assert subjects.time[:3] == [100.0, 102.0, 109.0]
    assert subjects.surv[:3] == approx([0.981041874972634, 0.962010279902039, 0.962010279902039])
    assert subjects.std_err[:3] == approx(
        [0.0191920562348243, 0.0275374762167643, 0.0275374762167643]
    )


def test_start_time_keeps_a_stratum_it_empties():
    fit = r.coxph("Surv(time, status) ~ x + strata(g)", _strata_data())
    curve = r.survfit(fit, start_time=15)
    assert curve.strata == {"g=1": 0, "g=2": 10}
    assert curve.n == [0, 10]
    assert curve.time == [float(t) for t in range(21, 31)]
    assert curve.surv == approx(
        [0.911703185270467, 0.911703185270467, 0.818916920480482, 0.721769214450701]
        + [0.721769214450701, 0.579122152428905, 0.579122152428905, 0.378441860225577]
        + [0.203946105189285, 0.203946105189285]
    )
    assert curve.std_err[-1] == approx(0.819221646496032)
    assert r.survfit(fit, start_time=15, stype=1).surv[-1] == approx(0.139802086569338)
    uncensored = r.survfit(fit, start_time=15, censor=False)
    assert uncensored.strata == {"g=1": 0, "g=2": 6}
    assert uncensored.time == [21.0, 23.0, 24.0, 26.0, 28.0, 29.0]
    every = r.survfit(fit, {"x": [0.5, -1]}, start_time=15)
    assert every.strata == {"g=1": 0, "g=2": 10}
    assert every.surv[0] == approx([0.921861441044301, 0.859787664575014])
    assert every.surv[-1] == approx([0.246763315487031, 0.0744024345015147])
    # a newdata row in the emptied stratum gets an empty curve, and the row in g=2 its own
    # curve (x = -1, the second column above); R's split() drops the empty stratum here and
    # hands the second row the first row's column (0.921861441044301, ...)
    found = r.survfit(fit, {"x": [0.5, -1], "g": [1, 2]}, start_time=15)
    assert found.strata == {"1": 0, "2": 10}
    assert found.n == [0, 10]
    assert found.surv == [row[1] for row in every.surv]


def test_penalized_fits_take_start_time_and_id():
    fit = r.coxph("Surv(start, stop, event) ~ pspline(age) + transplant", datasets.load_heart())
    assert r.survfit(fit, start_time=100).surv[:3] == approx(
        [0.979207892618949, 0.958406550548767, 0.958406550548767]
    )
    subjects = r.survfit(fit, _heart_newdata(pid=[1, 1, 2, 2]), id="pid")
    assert subjects.strata == {"1": 85, "2": 102}
    assert subjects.surv[:3] == approx([0.992080652630845, 0.968123794410737, 0.94398730070863])


def test_start_time_errors_as_r(lung_fit):
    with pytest.raises(ValueError, match="start.time argument has removed all endpoints"):
        r.survfit(lung_fit, start_time=1000)
    with pytest.raises(ValueError, match="removed all endpoints"):
        r.survfit(r.coxph("Surv(time, status) ~ x + strata(g)", _strata_data()), start_time=30)
    for value in ("a", [1, 2]):
        with pytest.raises(ValueError, match="start.time must be a single numeric value"):
            lung_fit.survfit(start_time=value)
