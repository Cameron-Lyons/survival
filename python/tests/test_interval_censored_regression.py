"""Check the interval regression binding against stock survival 3.8-11."""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

interval = setup_survival_import().interval

TIME = np.array([2, 5, 3, 2, 5, 3, 6, 7, 4, 10, 8, 5, 8, 4, 7, 3])
KIND = np.tile([0, 1, 2, 3], 4)
LEFT = np.where(KIND == 2, np.nan, TIME)
RIGHT = np.select([KIND == 2, KIND == 3], [TIME, TIME + 3], default=np.inf)
Z = [-1, 0.5, 1, 0, 0.5, -1, 0, 1, 0, 1, -1, 0.5, 1, 0, 0.5, -1]
G = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
DESIGN = np.column_stack((Z, G))

# f <- survreg(Surv(time, time + 3, c(1, 0, 2, 3)[kind+1], type="interval") ~ z + g,
#              dist=dist, control=survreg.control(rel.tolerance=1e-12))
# Columns below hold the baseline time scale and shape, slopes, slope SEs from
# vcov(f), logLik(f), AIC(f), BIC(f), and survival at the first four report times
# (one per censor type) from 1 - psurvreg(t, f$linear.predictors, f$scale, dist).
STOCK = {
    "weibull": (
        (5.660018753743, 3.711613435779),
        (0.453453242642, -0.180395855640),
        (0.129220779185, 0.184733046400),
        (-16.215545623316, 40.431091246633, 43.521446135592),
        (0.892918971904, 0.587757127368, 0.966184033565, 0.845389128680),
    ),
    "lognormal": (
        (5.006747627631, 0.405835585525),
        (0.411238898756, -0.186455002475),
        (0.166599590674, 0.244017226188),
        (-17.492329318499, 42.984658636998, 46.075013525957),
        (0.893947306457, 0.520156055891, 0.965307388318, 0.811162608885),
    ),
    "loglogistic": (
        (4.930023093500, 4.700125753231),
        (0.465349099636, -0.127233484649),
        (0.158146017203, 0.212510181539),
        (-17.106231422921, 42.212462845843, 45.302817734802),
        (0.886274359491, 0.605716429261, 0.980618906458, 0.833435318942),
    ),
    "exponential": (
        (6.775849046892, 1.0),
        (0.431368201843, -0.387021794833),
        (0.394205553282, 0.603378838066),
        (-21.883024179640, 49.766048359280, 52.083814525999),
        (0.634849487085, 0.416518284427, 0.654722623379, 0.596580893591),
    ),
}


def fit_model(name="weibull", design=DESIGN):
    return interval.interval_censored_regression(
        LEFT,
        RIGHT,
        KIND,
        design.ravel(),
        len(TIME),
        design.shape[1],
        interval.IntervalDistribution(name),
        tol=1e-12,
    )


@pytest.mark.parametrize("name", [*STOCK, "generalized"])
def test_mixed_interval_fits_match_stock_survreg(name):
    fit = fit_model(name)
    params, slopes, errors, fit_stats, survival = STOCK[
        "weibull" if name == "generalized" else name
    ]
    assert fit.converged
    assert fit.n_iter > 0
    np.testing.assert_allclose([fit.scale, fit.shape], params, rtol=1e-8)
    np.testing.assert_allclose(fit.coefficients, slopes, rtol=1e-8)
    np.testing.assert_allclose(fit.std_errors, errors, rtol=1e-8)
    np.testing.assert_allclose([fit.log_likelihood, fit.aic, fit.bic], fit_stats, rtol=1e-10)
    assert len(fit.survival_prob) == len(TIME)
    np.testing.assert_allclose(fit.survival_prob[:4], survival, rtol=1e-8)


def test_duplicate_column_reports_alias_and_preserves_predictions():
    reduced = fit_model()
    fit = fit_model(design=np.column_stack((DESIGN, Z)))
    np.testing.assert_allclose(fit.coefficients[:2], reduced.coefficients)
    np.testing.assert_allclose(fit.std_errors[:2], reduced.std_errors)
    assert math.isnan(fit.coefficients[2])
    assert fit.std_errors[2] == 0.0
    np.testing.assert_allclose(fit.survival_prob, reduced.survival_prob)
    assert fit.aic == pytest.approx(reduced.aic + 2)
    assert fit.bic == pytest.approx(reduced.bic + math.log(len(TIME)))


def test_intercept_only_exponential_preserves_the_right_tail():
    # The exponential MLE is total time on study divided by number of events.
    fit = interval.interval_censored_regression(
        [1.0] * 100 + [199.0],
        [np.inf] * 101,
        [0] * 100 + [1],
        [],
        101,
        0,
        interval.IntervalDistribution.Exponential,
        tol=1e-12,
    )
    assert fit.converged
    assert fit.coefficients == fit.std_errors == []
    assert fit.scale == pytest.approx(2.99)
    assert fit.shape == 1.0
    assert fit.survival_prob[-1] == pytest.approx(math.exp(-199 / 2.99), rel=1e-8, abs=0)


@pytest.mark.parametrize(
    ("code", "left", "right", "message"),
    [(4, 2, 4, "censor"), (3, 5, 4, "[Ii]nterval"), (2, np.nan, np.inf, "finite")],
)
def test_invalid_observations_are_rejected(code, left, right, message):
    with pytest.raises(ValueError, match=message):
        interval.interval_censored_regression(
            [1, left, 3],
            [1, right, 3],
            [0, code, 0],
            [],
            3,
            0,
            interval.IntervalDistribution.Weibull,
        )


def test_turnbull_aitken_convergence_matches_stock_r():
    # Lightly weighted exact events locate the jumps. Most interval observations
    # span the middle of the grid, covering 9,883 jump visits in total.
    rows = np.arange(150)
    times = rows + 1
    lower = (5 * rows % 97) % 75
    upper = 150 - (29 * rows % 97) % 75
    upper[::4] = lower[::4] + 1
    (curve,) = interval.turnbull(
        np.r_[times, lower],
        np.r_[times, upper],
        [1] * 150 + [3] * 150,
        weights=[0.01] * 150 + [1.0] * 150,
        timefix=False,
        se_fit=False,
    ).curves

    # Stock survival 3.8-11 called survfitKM once per iteration.
    assert curve.iterations == 167
    np.testing.assert_array_equal(curve.time, times)
    positions = [31, 78, 96]
    np.testing.assert_allclose(
        np.asarray(curve.surv)[positions],
        [0.8180503976609773, 0.023561787069174576, 0.008150261305521099],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(curve.n_event)[positions],
        [0.018260458876189933, 0.3568234023356351, 0.04682734930357703],
        rtol=1e-8,
    )
