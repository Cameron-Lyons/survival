"""Low-level kernels in ``survival.core``, checked against R survival 3.8.11.

The reference numbers were produced with the local R (``Ccoxcount1``/``Ccoxcount2`` through
``.Call``, ``residuals.coxph``, ``concordance``, ``nsk`` and ``pspline``/``spline.des``).
"""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival.core


def _lung30():
    """First 30 rows of ``lung`` with R's ``status - 1`` coding and the R fit's coefficients."""
    data = survival.datasets.load_lung()
    n = 30
    time = [float(t) for t in data["time"][:n]]
    status = [int(s) - 1 for s in data["status"][:n]]
    age = [float(a) for a in data["age"][:n]]
    sex = [float(s) for s in data["sex"][:n]]
    coef = [-0.020490827381780291, 0.24290418268151132]
    score = [math.exp(a * coef[0] + s * coef[1]) for a, s in zip(age, sex, strict=True)]
    covariates = core.CovariateMatrix(
        [value for a, s in zip(age, sex, strict=True) for value in (a, s)], n, 2
    )
    return core.SurvivalData(time, status), covariates, score, age


# R: residuals(coxph(Surv(time, status) ~ age + sex, lung[1:30, ]), type = "score")[1:6, ]
_LUNG30_SCORE_HEAD = [
    [5.583696396479894, -0.1922229604586857],
    [0.6815233759442858, -0.056232271403001799],
    [18.958929979475286, 0.6274187497584186],
    [-3.6147548299078629, -0.15227987847314284],
    [5.1746924061028219, 0.57804450885005298],
    [-21.064442043887876, 0.43388489815172615],
]
# R: residuals(fit, type = "schoenfeld")[1:3, ] and the last row; rownames are event times
_LUNG30_SCHOENFELD_TIMES = [12, 61, 71, 81, 88, 118, 144, 166, 170, 210, 218, 301, 306, 310]
_LUNG30_SCHOENFELD_HEAD = [
    [12.131001972201943, -0.27280947092307239],
    [-5.5615323897406057, 0.72027605175364529],
    [-1.8341490140041827, -0.24441727159565119],
]
_LUNG30_MARTINGALE_HEAD = [0.58080397699164643, 0.16678808437734227, -2.745507876254901]


def test_typed_inputs_validate_and_expose_their_fields():
    data = core.SurvivalData([1.0, 2.0, 3.0], [1, 0, 1])
    assert data.time == pytest.approx([1.0, 2.0, 3.0])
    assert data.status == [1, 0, 1]
    assert data.is_empty() is False

    counting = core.CountingProcessData([0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [1, 0, 1])
    assert counting.start == pytest.approx([0.0, 0.0, 1.0])
    assert counting.stop == pytest.approx([1.0, 2.0, 3.0])
    assert counting.event == [1, 0, 1]

    matrix = core.CovariateMatrix([1.0, 2.0, 3.0, 4.0], 2, 2)
    assert (matrix.n_obs, matrix.n_vars) == (2, 2)
    assert matrix.shape() == (2, 2)
    assert core.Weights.unit(3).values == pytest.approx([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="status length mismatch"):
        core.SurvivalData([1.0], [1, 0])
    with pytest.raises(ValueError, match="non-finite"):
        core.SurvivalData([float("nan")], [1])
    with pytest.raises(ValueError, match="status values must be 0 or 1"):
        core.coxcount1(core.SurvivalData([1.0], [2]))
    with pytest.raises(ValueError, match="stop 1 is before start 2"):
        core.coxcount2(core.CountingProcessData([2.0], [1.0], [1]))
    with pytest.raises(ValueError, match="non-finite"):
        core.CovariateMatrix([1.0, float("nan")], 2, 1)


def test_coxcount1_matches_r_risk_set_expansion():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5]
    status = [1, 1, 0, 1, 0, 1, 1, 0]
    strata = [1, 0, 0, 0, 0, 0, 0, 0]

    result = core.coxcount1(core.SurvivalData(time, status), strata)

    # .Call(Ccoxcount1, Y[sorted, ], newstrat), index mapped back to 0-based input rows
    assert result.time == pytest.approx([4.0, 2.5, 2.0, 1.5, 1.0])
    assert result.nrisk == [2, 5, 6, 7, 1]
    assert result.index == [4, 3, 4, 3, 7, 2, 6, 4, 3, 7, 2, 6, 1, 4, 3, 7, 2, 6, 1, 5, 0]
    assert result.status == [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    assert sum(result.nrisk) == len(result.index)

    unstratified = core.coxcount1(core.SurvivalData(time, status))
    assert unstratified.time == pytest.approx([4.0, 2.5, 2.0, 1.5, 1.0])
    assert unstratified.nrisk == [2, 5, 6, 7, 8]

    with pytest.raises(ValueError, match="strata length mismatch"):
        core.coxcount1(core.SurvivalData([1.0, 2.0], [1, 1]), [0])


def test_coxcount2_matches_r_risk_set_expansion():
    counting = core.CountingProcessData(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 2.0, 3.0, 3.0, 4.0], [1, 0, 1, 0, 1, 0]
    )

    result = core.coxcount2(counting, [1, 0, 0, 0, 0, 0])

    # .Call(Ccoxcount2, Y, sort.start - 1, sort.end - 1, newstrat)
    assert result.time == pytest.approx([3.0, 2.0, 1.0])
    assert result.nrisk == [3, 3, 1]
    assert result.index == [5, 3, 4, 1, 3, 2, 0]
    assert result.status == [0, 0, 1, 0, 0, 1, 1]

    with pytest.raises(ValueError, match="non-finite"):
        core.coxcount2(core.CountingProcessData([0.0], [float("inf")], [1]))


def test_coxscore2_matches_r_score_residuals():
    data, covariates, score, _ = _lung30()

    efron = core.coxscore2(data, covariates, score, ties="efron")
    breslow = core.coxscore2(data, covariates, score, ties="breslow")

    assert len(efron) == 30
    np.testing.assert_allclose(efron[:6], _LUNG30_SCORE_HEAD, rtol=1e-9)
    # lung[1:30, ] has no tied event times, so both tie rules agree
    np.testing.assert_allclose(breslow, efron)

    with pytest.raises(ValueError, match='ties must be "breslow" or "efron"'):
        core.coxscore2(data, covariates, score, ties="exact")
    with pytest.raises(ValueError, match="score contains negative value"):
        core.coxscore2(data, covariates, [-1.0] + score[1:])


def test_schoenfeld_residuals_match_r():
    data, covariates, score, _ = _lung30()

    result = core.schoenfeld_residuals(data, covariates, score)

    assert result.time[:14] == pytest.approx(_LUNG30_SCHOENFELD_TIMES)
    assert len(result.time) == sum(data.status) == 28
    np.testing.assert_allclose(result.residuals[:3], _LUNG30_SCHOENFELD_HEAD, rtol=1e-9)
    assert result.strata == [0] * 28
    # index points at the event rows of the input, in event-time order
    assert [data.time[i] for i in result.index] == pytest.approx(result.time)
    assert all(data.status[i] == 1 for i in result.index)

    with pytest.raises(ValueError, match="score length mismatch"):
        core.schoenfeld_residuals(data, covariates, score[:-1])


def test_coxmart_matches_r_martingale_residuals():
    data, _, score, _ = _lung30()
    residuals = survival.residuals.coxmart(
        core.CoxMartInput(data, score, core.Weights.unit(30), [0] * 30)
    )

    assert residuals[:3] == pytest.approx(_LUNG30_MARTINGALE_HEAD, rel=1e-9)
    assert sum(residuals) == pytest.approx(0.0, abs=1e-9)


def test_counting_process_kernels_match_r():
    # coxph(Surv(start, stop, status) ~ x, ties = "efron") on an 8-row data set
    counting = core.CountingProcessData(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 3.0],
        [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    )
    x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.2, 0.8]
    coef = 0.33940574727728401
    score = [math.exp(coef * value) for value in x]
    covariates = core.CovariateMatrix(x, 8, 1)

    score_residuals = core.agscore3(counting, covariates, score)
    schoenfeld = core.schoenfeld_residuals_counting(counting, covariates, score)
    martingale = survival.residuals.agmart(
        survival.core.AndersenGillInput(counting, score, core.Weights.unit(8), [0] * 8)
    )

    assert [row[0] for row in score_residuals] == pytest.approx(
        [
            -0.070399944106733892,
            -0.078659471901273481,
            0.13191746072176413,
            -0.16013037456165868,
            0.18063690216442208,
            -0.25597739958532717,
            0.40106836377595223,
            -0.14845553650713714,
        ],
        rel=1e-9,
    )
    assert schoenfeld.time == pytest.approx([1.0, 2.0, 3.0, 5.0, 6.0])
    assert [row[0] for row in schoenfeld.residuals] == pytest.approx(
        [-0.10411407906487369, 0.18128565020160825, 0.25326981673764015, -0.33044138787436744, 0.0],
        abs=1e-9,
    )
    assert martingale == pytest.approx(
        [
            0.67618082721423223,
            -0.61352772601271433,
            0.72767734553208374,
            -0.56470821990035924,
            0.71321922403229898,
            -0.33982166004466885,
            -0.048284144363593517,
            -0.55073564645727902,
        ],
        rel=1e-9,
    )


def test_concordancefit_matches_r_concordance():
    data, _, _, age = _lung30()
    x = core.CovariateMatrix(age, 30, 1)

    fit = core.concordancefit(data, x)
    # concordance(Surv(time, status) ~ age, lung[1:30, ])
    assert fit.concordance == pytest.approx([0.54147465437788023])
    counts = fit.count[0]
    assert (counts.concordant, counts.discordant, counts.tied_x) == (223.0, 187.0, 24.0)
    assert (counts.tied_y, counts.tied_xy) == (0.0, 0.0)
    assert fit.var[0] == pytest.approx([0.0047245346852800834])
    assert fit.n == 30
    assert fit.dfbeta is None
    assert fit.count_strata is None

    reversed_fit = core.concordancefit(data, x, reverse=True, timewt="S")
    # R concordance with reverse = TRUE and timewt = "S"
    assert reversed_fit.concordance == pytest.approx([0.45852534562211977])
    assert reversed_fit.count[0].concordant == pytest.approx(187.0)
    assert reversed_fit.count[0].discordant == pytest.approx(223.0)
    assert reversed_fit.var[0] == pytest.approx([0.0047245346852800843])

    with_influence = core.concordancefit(data, x, influence=1, ranks=True)
    assert len(with_influence.dfbeta) == 30
    assert len(with_influence.ranks[0].time) == sum(data.status)

    with pytest.raises(ValueError, match="timewt must be one of"):
        core.concordancefit(data, x, timewt="bogus")
    with pytest.raises(ValueError, match="y length mismatch"):
        core.concordancefit(data, core.CovariateMatrix(age + [1.0], 31, 1))


def test_concordancefit_counting_matches_r_concordance():
    counting = core.CountingProcessData(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 3.0],
        [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    )
    x = core.CovariateMatrix([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.2, 0.8], 8, 1)

    fit = core.concordancefit_counting(counting, x)

    # concordance(Surv(start, stop, status) ~ x)
    assert fit.concordance == pytest.approx([0.44444444444444442])
    assert (fit.count[0].concordant, fit.count[0].discordant) == (4.0, 5.0)
    assert fit.var[0] == pytest.approx([0.018289894833104711])
    assert fit.n == 8


def test_nsk_matches_r_natural_spline_basis():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    result = core.nsk(x, df=3)

    # nsk(x, df = 3): knots at the 33/67% quantiles, boundary knots at the 5/95% quantiles
    assert result.knots == pytest.approx([4.333333333333333, 6.6666666666666661])
    assert result.boundary_knots == pytest.approx((1.45, 9.5499999999999989))
    assert (result.n_rows, result.n_cols) == (10, 3)
    # the basis is stored row-major as one flat vector
    basis = np.asarray(result.basis).reshape(result.n_rows, result.n_cols)
    assert basis[0] == pytest.approx(
        [-0.27012195647814957, 0.078804550540842982, -0.010153756890049298]
    )
    assert basis[4] == pytest.approx(
        [0.79495246082857862, 0.29560127526764979, -0.035082252843865194]
    )
    assert basis[9] == pytest.approx(
        [0.078804550540842955, -0.27012195647815024, 1.2014711628273564]
    )


def test_pspline_basis_matches_r_spline_des():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    result = core.pspline_basis(x, 4, 3, (1.0, 10.0))

    # pspline(x, nterm = 4): evenly spaced knots, spline.des design (R drops the first column)
    assert (result.nterm, result.degree) == (4, 3)
    assert result.boundary_knots == pytest.approx((1.0, 10.0))
    assert result.knots == pytest.approx(
        [-5.75, -3.5, -1.25, 1.0, 3.25, 5.5, 7.75, 10.0, 12.25, 14.5, 16.75]
    )
    assert len(result.basis) == 10
    assert all(len(row) == 7 for row in result.basis)
    assert result.basis[0][1:] == pytest.approx([0.66666666666666663, 1 / 6, 0, 0, 0, 0])
    assert result.basis[3][1:] == pytest.approx(
        [
            0.049382716049382713,
            0.57407407407407407,
            0.37037037037037035,
            0.0061728395061728392,
            0,
            0,
        ]
    )
    assert result.basis[9][1:] == pytest.approx([0, 0, 0, 1 / 6, 0.66666666666666663, 1 / 6])
    assert all(sum(row) == pytest.approx(1.0) for row in result.basis)


def test_norisk_validates_public_inputs():
    with pytest.raises(ValueError, match="strata values must be strictly increasing"):
        survival.surv_analysis.norisk(
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [1, 0, 1],
            [0, 1, 2],
            [0, 1, 2],
            [2, 1],
        )
