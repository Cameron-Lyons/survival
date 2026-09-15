"""``concordancefit`` / ``concordancefit_counting`` against R survival 3.8.11 ``concordance()``."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival.core

TIME = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
STATUS = [1, 1, 0, 1, 1, 1, 0, 1]
X = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9]
W = [1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 0.5, 1.5]


def _counts(fit, index=0):
    count = fit.count[index]
    return (count.concordant, count.discordant, count.tied_x, count.tied_y, count.tied_xy)


@pytest.fixture
def data():
    return core.SurvivalData(TIME, STATUS), core.CovariateMatrix(X, 8, 1)


def test_concordancefit_matches_r_default(data):
    fit = core.concordancefit(*data)

    # concordance(Surv(time, status) ~ x): (C + Tx/2) / (C + D + Tx)
    assert fit.concordance == pytest.approx([0.5])
    assert _counts(fit) == pytest.approx((9.0, 9.0, 3.0, 1.0, 0.0))
    assert fit.var[0] == pytest.approx([0.029478458049886618])
    assert fit.cvar == pytest.approx([0.03020327178490444])
    assert fit.n == 8
    assert fit.count_strata is None
    assert fit.dfbeta is None
    assert fit.influence is None
    assert fit.ranks is None


def test_concordancefit_without_standard_errors_only_counts(data):
    fit = core.concordancefit(*data, std_err=False)

    assert fit.concordance == pytest.approx([0.5])
    assert _counts(fit) == pytest.approx((9.0, 9.0, 3.0, 1.0, 0.0))
    assert fit.var is None
    assert fit.cvar is None


def test_concordancefit_case_weights_match_r(data):
    fit = core.concordancefit(*data, weights=core.Weights(W))

    # R concordance with case weights w
    assert fit.concordance == pytest.approx([0.64615384615384619])
    assert _counts(fit) == pytest.approx((19.0, 9.5, 4.0, 2.0, 0.0))
    assert fit.var[0] == pytest.approx([0.020554798501453028])
    assert fit.cvar == pytest.approx([0.024967025115870081])


@pytest.mark.parametrize(
    ("timewt", "concordance", "counts", "var", "cvar"),
    [
        (
            "S",
            0.48672566371681419,
            (9.4, 10.0, 3.2, 1.2, 0.0),
            0.031057472926532059,
            0.027801008021659596,
        ),
        (
            "S/G",
            0.4730831973898858,
            (9.88, 11.2, 3.44, 1.44, 0.0),
            0.032793061463206923,
            0.02537403773518913,
        ),
        (
            "n/G2",
            0.4730831973898858,
            (9.88, 11.2, 3.44, 1.44, 0.0),
            0.032793061463206943,
            0.025374037735189141,
        ),
        (
            "I",
            0.47573306370070778,
            (1.4464285714285714, 1.6178571428571429, 0.46785714285714286, 0.25, 0.0),
            0.036680341889530892,
            0.17544301778614305,
        ),
    ],
)
def test_concordancefit_time_weights_match_r(data, timewt, concordance, counts, var, cvar):
    fit = core.concordancefit(*data, timewt=timewt)

    assert fit.concordance == pytest.approx([concordance])
    assert _counts(fit) == pytest.approx(counts)
    assert fit.var[0] == pytest.approx([var])
    assert fit.cvar == pytest.approx([cvar])


def test_concordancefit_ymax_truncates_pairs(data):
    fit = core.concordancefit(*data, ymax=3.0)

    # R concordance with ymax = 3
    assert fit.concordance == pytest.approx([0.5])
    assert _counts(fit) == pytest.approx((7.0, 7.0, 3.0, 0.0, 0.0))
    assert fit.var[0] == pytest.approx([0.020761245674740487])
    assert fit.cvar == pytest.approx([0.037438210578348986])


def test_concordancefit_strata_keeps_per_stratum_counts(data):
    fit = core.concordancefit(*data, strata=[1, 1, 1, 1, 2, 2, 2, 2])

    # concordance(Surv(time, status) ~ x + strata(g))
    assert fit.concordance == pytest.approx([0.61111111111111116])
    assert fit.count_strata == [1, 2]
    assert _counts(fit, 0) == pytest.approx((3.0, 1.0, 1.0, 0.0, 0.0))
    assert _counts(fit, 1) == pytest.approx((2.0, 2.0, 0.0, 1.0, 0.0))
    assert fit.var[0] == pytest.approx([0.037265660722450848])
    assert fit.cvar == pytest.approx([0.052983539094650201])


def test_concordancefit_cluster_uses_grouped_jackknife_variance(data):
    fit = core.concordancefit(*data, cluster=[1, 1, 2, 2, 3, 3, 4, 4])

    # concordance(Surv(time, status) ~ x + cluster(id)): counts unchanged, var collapsed by id
    assert fit.concordance == pytest.approx([0.5])
    assert _counts(fit) == pytest.approx((9.0, 9.0, 3.0, 1.0, 0.0))
    assert fit.var[0] == pytest.approx([0.0034013605442176865])
    assert fit.cvar == pytest.approx([0.03020327178490444])


def test_concordancefit_influence_and_ranks_match_r(data):
    with_dfbeta = core.concordancefit(*data, influence=1)
    assert [row[0] for row in with_dfbeta.dfbeta] == pytest.approx(
        [
            0.0,
            0.047619047619047616,
            0.023809523809523808,
            -0.023809523809523808,
            -0.047619047619047616,
            0.023809523809523808,
            -0.11904761904761904,
            0.095238095238095233,
        ]
    )

    with_influence = core.concordancefit(*data, influence=2)
    assert with_influence.dfbeta is None
    influence = with_influence.influence[0]
    assert len(influence) == 8
    assert influence[0] == pytest.approx([3.0, 3.0, 1.0, 0.0, 0.0])
    assert influence[6] == pytest.approx([0.0, 5.0, 0.0, 0.0, 0.0])

    with_ranks = core.concordancefit(*data, ranks=True)
    ranks = with_ranks.ranks[0]
    assert ranks.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 4.0, 6.0])
    assert ranks.rank == pytest.approx([0.0, 0.42857142857142855, -0.6, -0.25, 0.25, 0.0])
    assert ranks.timewt == pytest.approx([8.0, 7.0, 5.0, 4.0, 4.0, 1.0])
    assert ranks.casewt == pytest.approx([1.0] * 6)


def test_concordancefit_timefix_groups_near_tied_event_times():
    near = core.SurvivalData([1.0, 1.0 + 4e-9, 2.0, 3.0], [1, 1, 1, 0])
    x = core.CovariateMatrix([0.9, 0.1, 0.5, 0.2], 4, 1)

    fixed = core.concordancefit(near, x)
    # aeqSurv folds the near-tie; equals concordance(Surv(c(1, 1, 2, 3), status) ~ x)
    assert fixed.concordance == pytest.approx([0.4])
    assert _counts(fixed) == pytest.approx((2.0, 3.0, 0.0, 1.0, 0.0))
    assert fixed.var[0] == pytest.approx([0.0864])

    raw = core.concordancefit(near, x, timefix=False)
    # without timefix the pair is ordered: one more discordant pair, no tied time
    assert raw.concordance == pytest.approx([1 / 3])
    assert _counts(raw) == pytest.approx((2.0, 4.0, 0.0, 0.0, 0.0))


def test_concordancefit_several_predictors_share_one_variance_matrix(data):
    survival_data, _ = data
    x = core.CovariateMatrix([value for a, b in zip(X, W, strict=True) for value in (a, b)], 8, 2)

    fit = core.concordancefit(survival_data, x)

    # concordance(Surv(time, status) ~ x + w)
    assert fit.concordance == pytest.approx([0.5, 0.38095238095238093])
    assert _counts(fit, 0) == pytest.approx((9.0, 9.0, 3.0, 1.0, 0.0))
    assert _counts(fit, 1) == pytest.approx((6.0, 11.0, 4.0, 1.0, 0.0))
    assert fit.var[0] == pytest.approx([0.029478458049886618, 0.011904761904761904])
    assert fit.var[1] == pytest.approx([0.011904761904761904, 0.021361983947017958])
    assert fit.cvar == pytest.approx([0.03020327178490444, 0.029616132167152576])


def test_concordancefit_counting_matches_r():
    counting = core.CountingProcessData(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 3.0],
        [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        [1, 0, 1, 0, 1, 0, 1, 1],
    )
    x = core.CovariateMatrix([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.2, 0.8], 8, 1)

    plain = core.concordancefit_counting(counting, x)
    # concordance(Surv(start, stop, status) ~ x)
    assert plain.concordance == pytest.approx([0.44444444444444442])
    assert _counts(plain) == pytest.approx((4.0, 5.0, 0.0, 0.0, 0.0))
    assert plain.var[0] == pytest.approx([0.018289894833104711])
    assert plain.cvar == pytest.approx([0.042181069958847732])
    assert plain.n == 8

    weighted = core.concordancefit_counting(
        counting, x, weights=core.Weights([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0])
    )
    assert weighted.concordance == pytest.approx([0.52941176470588236])
    assert _counts(weighted) == pytest.approx((9.0, 8.0, 0.0, 0.0, 0.0))
    assert weighted.var[0] == pytest.approx([0.017456687539660683])

    survival_weighted = core.concordancefit_counting(counting, x, timewt="S")
    assert survival_weighted.concordance == pytest.approx([0.46987951807228923])
    assert _counts(survival_weighted) == pytest.approx((6.5, 7.3333333333333321, 0.0, 0.0, 0.0))
    assert survival_weighted.cvar == pytest.approx([0.026491508201480624])


def test_concordancefit_validates_inputs(data):
    survival_data, x = data

    with pytest.raises(ValueError, match="timewt must be one of"):
        core.concordancefit(survival_data, x, timewt="bogus")
    with pytest.raises(ValueError, match="length mismatch"):
        core.concordancefit(survival_data, x, weights=core.Weights([1.0, 2.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        core.concordancefit(survival_data, x, strata=[0, 1])
    with pytest.raises(ValueError, match="length mismatch"):
        core.concordancefit(survival_data, core.CovariateMatrix(X + [0.3], 9, 1))
