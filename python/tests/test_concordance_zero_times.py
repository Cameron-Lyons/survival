"""Concordance signed-zero ordering, checked against stock survival 3.8-12."""

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("timefix", [False, True])
@pytest.mark.parametrize("std_err", [False, True])
def test_zero_event_and_censor_times_match_stock_r(counting, timefix, std_err):
    # R: concordancefit(Surv(c(0, -0, 1), c(1, 0, 1)), matrix(1:3, ncol=1),
    #                   influence=3, ranks=TRUE)
    time = [0.0, -0.0, 1.0]
    status = [1, 0, 1]
    response = r.Surv([-1.0] * 3, time, status) if counting else r.Surv(time, status)
    fit = r.concordancefit(
        response, [1.0, 2.0, 3.0], timefix=timefix, std_err=std_err, influence=3, ranks=True
    )
    assert fit.concordance == 1.0
    assert fit.count == {
        "concordant": 2.0,
        "discordant": 0.0,
        "tied.x": 0.0,
        "tied.y": 0.0,
        "tied.xy": 0.0,
    }
    if std_err:
        assert fit.var == 0.0
        assert fit.cvar == pytest.approx(1 / 6)
        assert fit.dfbeta == [0.0] * 3
        assert fit.influence == [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        assert fit.ranks == {
            "time": [0.0, 1.0],
            "rank": [2 / 3, 0.0],
            "timewt": [3.0, 1.0],
            "casewt": [1.0, 1.0],
        }
    else:
        assert fit.var is None
        assert fit.cvar is None
        assert fit.dfbeta is None
        assert fit.influence is None
        assert fit.ranks is None


@pytest.mark.parametrize(
    ("counting", "timewt"),
    [(False, weight) for weight in ("n", "S", "S/G", "n/G2", "I")]
    + [(True, weight) for weight in ("n", "S", "I")],
)
@pytest.mark.parametrize("timefix", [False, True])
def test_zero_sign_does_not_change_joint_weighted_results(counting, timewt, timefix):
    stop = np.array([0.0, -0.0, 0.0, -0.0, 1.0, 2.0, 3.0, 3.0])
    start = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -0.0, 1.0, 0.0])
    status = [1, 1, 1, 1, 1, 0, 1, 1]
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, -0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [1.0, 2.0],
            [-0.0, 1.0],
            [2.0, -0.0],
            [1.0, 1.0],
        ]
    )

    def fit(start, stop, x):
        response = r.Surv(start, stop, status) if counting else r.Surv(stop, status)
        return r.concordancefit(
            response,
            x,
            weights=[1.0, 0.5, 1.5, 1.0, 2.0, 0.0, 3.0, 1.0],
            cluster=[0, 0, 1, 1, 2, 2, 3, 3],
            timewt=timewt,
            timefix=timefix,
            influence=3,
            ranks=True,
        )

    # Adding positive zero canonicalizes zero signs without changing any values.
    assert fit(start, stop, x) == fit(start + 0.0, stop + 0.0, x + 0.0)


def test_zero_predictors_keep_stock_rank_row_order_for_all_columns():
    response = r.Surv([1.0, 1.0, 2.0], [1, 1, 1])
    x = np.array([[0.0, 0.0], [-0.0, 0.0], [1.0, 1.0]])
    fit = r.concordancefit(response, x, weights=[1.0, 2.0, 1.0], ranks=True)
    for ranks in fit.ranks:
        assert ranks == {
            "time": [1.0, 1.0, 2.0],
            "rank": [0.25, 0.25, 0.0],
            "timewt": [4.0, 4.0, 1.0],
            "casewt": [2.0, 1.0, 1.0],
        }
