"""Cross-feature contracts for the R formula and population interfaces."""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r


def test_penalized_prediction_uses_training_spline_knots_and_effective_df():
    data = survival.datasets.load_lung()
    fit = r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", data)
    new = {"age": [40, 65, 80], "sex": [1, 2, 1]}
    together = r.predict(fit, newdata=new, type="lp")
    individually = [
        r.predict(fit, newdata={key: [values[i]] for key, values in new.items()}, type="lp")[0]
        for i in range(3)
    ]
    assert together == pytest.approx(individually)
    assert r.degrees_freedom(fit) == pytest.approx(sum(fit.df))
    assert r.model_summary(fit)["logtest"]["df"] == pytest.approx(sum(fit.df))
    assert r.aic(fit) == pytest.approx(-2 * fit.loglik[-1] + 2 * sum(fit.df))
    with pytest.warns(RuntimeWarning, match="robust variance.*not defined"):
        robust = r.coxph("Surv(time, status) ~ ridge(age, theta=1)", data, robust=True)
    assert not robust.robust


def test_multistate_summary_integrates_state_occupancy_and_deduplicates_times():
    data = {"time": [1, 2, 3, 4], "event": RFactor(["a", "b", "a", "b"], ["censor", "a", "b"])}
    fit = r.survfit("Surv(time, event) ~ 1", data)
    summary = r.summary_survfit(fit, times=[5, 2, 2, 0], extend=True)
    assert summary.time == [0, 2, 5]
    assert summary.n_risk == [[4, 0, 0], [3, 0, 0], [0, 0, 0]]
    assert summary.n_event == [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
    assert np.allclose(summary.pstate, [[1, 0, 0], [0.5, 0.25, 0.25], [0, 0.5, 0.5]])
    assert summary.table.rownames == ["(s0)", "a", "b"]
    assert [row[2] for row in summary.table.values] == pytest.approx([2.5, 1, 0.5])
    assert [row[3] for row in summary.table.values] == pytest.approx(
        [math.sqrt(0.3125), math.sqrt(0.375), math.sqrt(0.1875)]
    )
    scaled = r.summary_survfit(fit, scale=2)
    assert [row[2] for row in scaled.table.values] == pytest.approx([1.25, 0.5, 0.25])
    assert r.summary_survfit(fit, rmean="none").table.colnames == ["n", "nevent"]
    for times in ([], [math.inf], [math.nan]):
        with pytest.raises(ValueError, match="times"):
            r.summary_survfit(fit, times=times)


def test_cox_population_weights_and_individual_survival():
    data = survival.datasets.load_lung()
    fit = r.coxph("Surv(time, status) ~ age + sex", data)
    population = {"age": [40, 75], "sex": [1, 2], "time": [100, 300], "status": [1, 1]}
    times = [0, 100, 300]
    first = r.survexp(
        "~1", {key: values[:1] for key, values in population.items()}, ratetable=fit, times=times
    )
    second = r.survexp(
        "~1", {key: values[1:] for key, values in population.items()}, ratetable=fit, times=times
    )
    average = r.survexp("~1", population, ratetable=fit, times=times, weights=[1, 3])
    assert average.surv == pytest.approx(
        [(a + 3 * b) / 4 for a, b in zip(first.surv, second.surv, strict=True)]
    )
    s = r.survexp("time ~ 1", population, ratetable=fit, method="individual.s")
    h = r.survexp("time ~ 1", population, ratetable=fit, method="individual.h")
    assert s == pytest.approx([math.exp(-value) for value in h])
    with pytest.raises(ValueError, match="positive.*weight"):
        r.survexp("~1", population, ratetable=fit, times=times, weights=[0, 0])


def test_yates_seeded_risk_is_reproducible_and_validates_simulation_count():
    data = survival.datasets.load_veteran()
    fit = r.coxph("Surv(time, status) ~ factor(trt) + karno", data, model=True)
    first = r.yates(fit, "factor(trt)", predict="risk", options={"seed": 12})
    again = r.yates(fit, "factor(trt)", predict="risk", options={"seed": 12})
    other = r.yates(fit, "factor(trt)", predict="risk", options={"seed": 13})
    assert first.estimate == again.estimate
    assert first.mvar == again.mvar
    assert first.estimate["pmm"] == other.estimate["pmm"]
    assert first.mvar != other.mvar
    with pytest.raises(ValueError, match="nsim"):
        r.yates(fit, "factor(trt)", predict="risk", nsim=1)
