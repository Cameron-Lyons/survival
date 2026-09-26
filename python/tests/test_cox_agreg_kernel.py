"""The (start, stop] Cox kernel (``agfit4.c``) and its iteration control against R.

Reference values were computed with R 4.5.3 and survival 3.8-12, calling
``coxph(..., timefix = FALSE)`` or ``agreg.fit`` directly on the data below.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
datasets = survival.datasets

# 11 (start, stop] rows whose risk scores span many orders of magnitude at the
# solution, so small risk sets follow large ones.
CASE_1520 = {
    "time": [15.0, 4.0, 25.0, 8.0, 18.0, 6.0, 9.0, 14.0, 11.0, 64.0, 19.0],
    "entry": [9.0, 2.0, 5.0, 1.0, 14.0, 3.0, 7.0, 6.0, 3.0, 39.0, 2.0],
    "status": [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    "x": [
        [-3.397587783734524, -1.2399609396099633],
        [-4.001456275604981, -0.9612195837228918],
        [1.2485501179741572, -0.36802848351666784],
        [-1.922243349085923, -0.29505277170923994],
        [-0.6191833991904704, -0.432759419947449],
        [0.9541844726560894, 3.636570321763567],
        [0.9722226187833938, -1.5241526403027048],
        [-2.083451024212451, 1.3906982107759198],
        [-0.557594058080056, 1.1836246992905952],
        [-6.582305367737362, 2.3943458650099303],
        [0.9175209607015593, -1.2770581196683075],
    ],
}
R_COEF_1520 = [-2.729510672, 2.391046607]

# 9 rows, all deaths: the coefficients run off to infinity.
CASE_2124 = {
    "time": [43.0, 7.0, 2.0, 1.0, 13.0, 3.0, 41.0, 4.0, 10.0],
    "entry": [28.0, 6.0, 1.0, 0.0, 4.0, 0.0, 31.0, 1.0, 5.0],
    "status": [1] * 9,
    "x": [
        [0.055792698117579004, -1.7304770998879826],
        [-1.3617180976588907, -0.9673886866286651],
        [0.8665565328654526, 0.9626229915832482],
        [0.25602914821193906, -0.9861502050896039],
        [-0.3219648826811489, 0.2535879462914172],
        [3.191390489555803, 2.1186597213685254],
        [-0.6752767149783483, -3.530622409131793],
        [1.538739604857286, 1.4570505303916683],
        [0.8826048048813887, 0.8028742413755462],
    ],
}


def _counting_fit(case, **kwargs):
    return core.coxph_fit(case["time"], case["status"], case["x"], entry=case["entry"], **kwargs)


def test_counting_fit_keeps_small_risk_sets_exact():
    at_r = _counting_fit(CASE_1520, init=R_COEF_1520, iter_max=0)
    assert at_r.loglik[0] == pytest.approx(-2.02753429303384, rel=1e-12)

    fit = _counting_fit(CASE_1520, init=[-0.10734736591853931, 0.5817618903234776])
    assert fit.iter == 7
    assert fit.flag == 2
    assert fit.info == [2, 0, 0, 0]
    assert fit.coefficients == pytest.approx([-2.72951067219589, 2.39104660699241], rel=1e-10)
    assert fit.loglik == pytest.approx([-6.73608164456893, -2.02753429303383], rel=1e-12)


def test_counting_fit_never_converges_while_step_halving():
    fit = _counting_fit(CASE_2124, init=[-1.7291049186754255, -1.9022957381326357])
    # R: iter 20, info = (rank 2, 26 recentrings, 2 step halvings, ran out).
    assert fit.iter == 20
    assert fit.flag == 1000
    assert fit.info == [2, 26, 2, 1]
    assert fit.loglik[0] == pytest.approx(-7.44594093298, rel=1e-10)
    assert -1e-5 < fit.loglik[1] < 0


def test_counting_fit_reports_exp_overflow_like_r():
    # agreg.fit(matrix(c(0, 0, 1)), Surv(c(0, 0, 0), 1:3, 1), init = b,
    # iter.max = 0), which centres every column (nocenter = NULL): the centre
    # of the risk scores moves by b / 2.
    def fit(beta):
        return core.coxph_fit(
            [1.0, 2.0, 3.0],
            [1, 1, 1],
            [[0.0], [0.0], [1.0]],
            entry=[0.0] * 3,
            init=[beta],
            iter_max=0,
            nocenter=[],
        )

    moved = fit(500.0)
    assert moved.loglik[0] == pytest.approx(-1000.0, rel=1e-12)
    assert moved.info == [1, 1, 0, 0]
    with pytest.raises(RuntimeError, match="exp overflow due to covariates"):
        fit(2000.0)


def test_heart_offset_is_absorbed_by_recentring():
    heart = datasets.load_heart()
    x = [
        [age, float(transplant)]
        for age, transplant in zip(heart["age"], heart["transplant"], strict=True)
    ]
    n = len(heart["stop"])
    fits = {
        offset: core.coxph_fit(
            heart["stop"],
            [int(event) for event in heart["event"]],
            x,
            entry=heart["start"],
            offset=[offset] * n,
        )
        for offset in (0.0, -750.0)
    }
    # R: agreg.fit(cbind(age, transplant), Surv(start, stop, event), offset = o).
    assert fits[0.0].info == [2, 0, 0, 0]
    assert fits[-750.0].info == [2, 5, 0, 0]
    for fit in fits.values():
        assert fit.iter == 4
        assert fit.coefficients == pytest.approx(
            [0.0307422562590614, -0.00417824732163875], rel=1e-9
        )
        assert fit.loglik == pytest.approx([-298.121355672984, -295.536672614928], rel=1e-12)


def test_detail_of_counting_fit_matches_coxdetail():
    fit = _counting_fit(CASE_1520, init=R_COEF_1520, iter_max=0)
    detail = core.coxph_detail(fit)
    assert detail.time == [4.0, 6.0, 14.0, 15.0, 18.0, 19.0, 25.0, 64.0]
    assert detail.hazard[5] == pytest.approx(4126.13694309023, rel=1e-12)
    assert detail.means[4][0] == pytest.approx(-0.60293423955203, rel=1e-12)


def test_aliased_column_is_na_after_converging_while_halving():
    # coxph.fit marks aliased coefficients when coxfit$flag < nvar, -2 included.
    x1 = [
        -0.40135649592727085,
        -1.1471693888359418,
        -2.7438279902033003,
        -1.2057427694082394,
        0.13299093024701425,
        1.3781793945651697,
        0.42305051733694915,
        -2.2958873961891935,
        -0.1785977507882967,
    ]
    fit = core.coxph_fit(
        [8.0, 3.0, 8.0, 3.0, 7.0, 3.0, 3.0, 5.0, 4.0],
        [1, 0, 1, 0, 0, 1, 1, 1, 0],
        [[value, 2 * value] for value in x1],
        method="breslow",
        init=[-0.6366924767849802, -9.949008576306692],
        eps=0.01,
    )
    assert fit.flag == -2
    assert fit.iter == 13
    assert fit.coefficients[0] == pytest.approx(12.9775454916007, rel=1e-10)
    assert math.isnan(fit.coefficients[1])
    assert fit.info is None


@pytest.mark.parametrize("counting", [False, True])
def test_exact_fit_marks_aliased_coefficients_without_iterations(counting):
    # coxexact.fit / agexact.fit mark which.sing even with iter.max = 0;
    # coxph.fit and agreg.fit leave iter.max = 0 fits alone.
    x1 = [0.5, 1.2, -0.3, 0.8, 2.1, -1.0, 0.4, 1.5]

    def fit(method):
        return core.coxph_fit(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 1, 0, 1, 1, 0, 1, 1],
            [[value, 2 * value] for value in x1],
            entry=[0.0] * 8 if counting else None,
            method=method,
            iter_max=0,
        )

    exact = fit("exact")
    assert exact.coefficients[0] == 0.0
    assert math.isnan(exact.coefficients[1])
    assert fit("efron").coefficients == [0.0, 0.0]


@pytest.mark.parametrize(
    ("method", "counting", "expected_iter", "coef"),
    [
        ("efron", False, 3, [0.0110683142976359, -0.552544811889215, 0.463625119105121]),
        ("breslow", False, 3, [0.0110426869650175, -0.551821520328325, 0.462844562494459]),
        ("exact", False, 2, [0.00950614811179832, -0.524522232936856, 0.485137497664694]),
        ("efron", True, 2, [0.011068314297636, -0.552544811889215, 0.463625119105121]),
        ("breslow", True, 2, [0.0110426869650175, -0.551821520328325, 0.462844562494459]),
        ("exact", True, 2, [0.011068571621021, -0.553365330787488, 0.464285095353364]),
    ],
)
def test_iterations_reported_when_they_run_out(method, counting, expected_iter, coef):
    # coxfit6.c's loop counter ends at iter.max + 1; the other fitters stop at
    # iter.max.  lung, age + sex + ph.ecog, iter.max = 2.
    lung = datasets.load_lung()
    rows = [
        i
        for i in range(len(lung["time"]))
        if all(lung[name][i] is not None for name in ("age", "sex", "ph.ecog"))
    ]
    fit = core.coxph_fit(
        [lung["time"][i] for i in rows],
        [int(lung["status"][i]) - 1 for i in rows],
        [[lung["age"][i], lung["sex"][i], lung["ph.ecog"][i]] for i in rows],
        entry=[0.0] * len(rows) if counting else None,
        method=method,
        iter_max=2,
    )
    assert fit.iter == expected_iter
    assert fit.flag == 1000
    assert fit.coefficients == pytest.approx(coef, rel=1e-9)
    assert fit.info == ([3, 0, 0, 1] if counting and method != "exact" else None)


def _split_data(nsubject=120):
    """survSplit-style data: subject ``i`` is followed in unit intervals up to
    ``1 + 53 i mod 97``, dies there unless ``i`` is a multiple of 3, and has the
    time-varying covariate ``z = x_i (t - 1) / 5`` with ``x_i = (37 i mod 101) / 50``.
    """

    start, stop, status, z = [], [], [], []
    for i in range(nsubject):
        x = ((i * 37) % 101) / 50.0
        last = 1 + (i * 53) % 97
        for t in range(1, last + 1):
            start.append(float(t - 1))
            stop.append(float(t))
            status.append(int(i % 3 != 0 and t == last))
            z.append([x * (t - 1) / 5.0])
    return start, stop, status, z


@pytest.mark.parametrize(
    ("beta", "efron", "breslow"),
    [
        (0.5, -478.751116295908, -478.888803621125),
        (1.0, -743.829243873272, -743.94983459056),
        (2.0, -1321.85189636427, -1321.94656960294),
    ],
)
def test_survsplit_log_likelihood_matches_r(beta, efron, breslow):
    start, stop, status, z = _split_data()
    for method, expected in (("efron", efron), ("breslow", breslow)):
        fit = core.coxph_fit(stop, status, z, entry=start, method=method, init=[beta], iter_max=0)
        assert fit.loglik[0] == pytest.approx(expected, rel=1e-12)
