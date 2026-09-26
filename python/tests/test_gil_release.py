"""The core kernels run without the GIL, and NumPy inputs cross the boundary as lists do.

Each heavy binding converts its arguments while attached and releases the GIL around the
kernel (``py.detach``), so Python threads keep running during a fit and fits on several
threads overlap.  The typed boundary (``FloatVec``/``IntVec``/``FloatMatrix``) reads NumPy
arrays of any layout without a ``.tolist()``, and gives the same fit as nested lists.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival.core
population = survival.population
regression = survival.regression
sa = survival.surv_analysis


def _cox_data(n: int, p: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    time_ = rng.exponential(size=n) * np.exp(-0.3 * x[:, 0])
    status = (rng.uniform(size=n) < 0.7).astype(np.int32)
    return time_, status, x


def _python_steps_during(call: Callable[[], object]) -> tuple[int, float]:
    """How many loop steps a Python thread completes while ``call`` runs, and its duration.

    A binding that holds the GIL for its kernel blocks the thread for the whole call; one
    that detaches lets it run throughout.
    """

    steps = 0
    running = threading.Event()
    done = threading.Event()

    def spin() -> None:
        nonlocal steps
        running.set()
        while not done.is_set():
            steps += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    running.wait()
    try:
        before = steps
        started = time.perf_counter()
        call()
        return steps - before, time.perf_counter() - started
    finally:
        done.set()
        spinner.join()


def _assert_detaches(call: Callable[[], object]) -> None:
    idle_steps, idle_time = _python_steps_during(lambda: time.sleep(0.02))
    steps, elapsed = _python_steps_during(call)
    # With the GIL held the thread only runs at the call's boundaries (a share near 0);
    # detached, it runs for most of the call.
    share = (steps / elapsed) / (idle_steps / idle_time)
    assert share > 0.1, f"the thread ran for {share:.1%} of a {elapsed:.3f} s call"


def test_cox_fits_on_four_threads_overlap():
    time_, status, x = _cox_data(100_000)
    regression.coxph_fit(time_, status, x)

    started = time.perf_counter()
    sequential = [regression.coxph_fit(time_, status, x) for _ in range(4)]
    sequential_time = time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda _: regression.coxph_fit(time_, status, x), range(4)))
    threaded_time = time.perf_counter() - started

    assert all(fit.coefficients == sequential[0].coefficients for fit in threaded)
    assert threaded_time < 0.75 * sequential_time, (sequential_time, threaded_time)


def test_heavy_kernels_release_the_gil():
    time_, status, x = _cox_data(50_000)
    fit = regression.coxph_fit(time_, status, x)
    design = np.column_stack([np.ones(len(time_)), x])
    survreg_data = regression.SurvregData(time_ + 0.01, status, design)
    weibull = regression.SurvregDistribution("weibull")
    right = core.SurvivalData(time_, status)
    predictor = core.CovariateMatrix(x[:, 0], len(time_), 1)
    group = (x[:, 0] > 0).astype(np.int32)
    aj_time, aj_status, aj_x = _cox_data(1_000)
    states = (aj_status * (1 + (aj_x[:, 1] > 0))).astype(np.int32)

    calls = {
        "coxph_fit": lambda: regression.coxph_fit(time_, status, x),
        "CoxPHFit.dfbeta": lambda: fit.dfbeta(),
        "CoxPHFit.survfit": lambda: fit.survfit(x[:5]),
        "cox_zph": lambda: regression.cox_zph(fit),
        "survreg_fit": lambda: regression.survreg_fit(survreg_data, weibull),
        "concordancefit": lambda: core.concordancefit(right, predictor),
        "survdiff": lambda: sa.survdiff(time_, status, group),
        "survfitaj": lambda: sa.survfitaj(aj_time, states, ["censor", "a", "b"]),
    }
    for name, call in calls.items():
        try:
            _assert_detaches(call)
        except AssertionError as err:
            pytest.fail(f"{name}: {err}")


def test_numpy_and_list_inputs_give_identical_cox_fits():
    time_, status, x = _cox_data(200, p=3)
    strata = (x[:, 2] > 0).astype(np.int64)
    weights = 0.5 + np.arange(200) % 3 / 2
    from_numpy = regression.coxph_fit(
        time_, status, np.asfortranarray(x), strata=strata, weights=weights
    )
    from_lists = regression.coxph_fit(
        time_.tolist(),
        status.tolist(),
        x.tolist(),
        strata=strata.tolist(),
        weights=weights.tolist(),
    )
    assert from_numpy.coefficients == from_lists.coefficients
    assert from_numpy.var == from_lists.var
    assert from_numpy.loglik == from_lists.loglik
    assert from_numpy.x == x.tolist()


def test_numpy_and_list_inputs_give_identical_survreg_fits():
    time_, status, x = _cox_data(200)
    design = np.column_stack([np.ones(200), x])
    weibull = regression.SurvregDistribution("weibull")
    from_numpy = regression.survreg_fit(
        regression.SurvregData(time_ + 0.01, status, design[:, ::1]), weibull
    )
    from_lists = regression.survreg_fit(
        regression.SurvregData((time_ + 0.01).tolist(), status.tolist(), design.tolist()),
        weibull,
    )
    assert from_numpy.coefficients == from_lists.coefficients
    assert from_numpy.variance_matrix == from_lists.variance_matrix
    assert from_numpy.covariates == design.tolist()
    assert regression.SurvregData(time_, status, design).covariates == design.tolist()


def test_numpy_and_list_inputs_give_identical_pyears():
    us = population.survexp_us()
    positions = population.match_ratetable(
        us, ["age", "sex", "year"], [[18262.5, 21915.0], [1.0, 2.0], [10957.0, 12949.0]]
    ).r
    kwargs = {"factors": [1], "dims": [2], "cuts": [[]], "ratetable": us, "scale": 365.25}
    from_numpy = population.pyears(
        np.array([365.25, 1826.25]),
        event=np.array([1.0, 0.0]),
        categories_data=np.array([[1.0], [2.0]]),
        ratetable_positions=np.array(positions),
        **kwargs,
    )
    from_lists = population.pyears(
        [365.25, 1826.25],
        event=[1.0, 0.0],
        categories_data=[[1.0], [2.0]],
        ratetable_positions=positions,
        **kwargs,
    )
    assert from_numpy.pyears == from_lists.pyears
    assert from_numpy.expected == from_lists.expected


def test_numpy_and_list_inputs_give_identical_concordance():
    time_, status, x = _cox_data(300)
    right = core.SurvivalData(time_, status)
    predictor = core.CovariateMatrix(x[:, 0], 300, 1)
    strata = (x[:, 1] > 0).astype(np.int32)
    from_numpy = core.concordancefit(right, predictor, strata=strata, influence=1)
    from_lists = core.concordancefit(right, predictor, strata=strata.tolist(), influence=1)
    assert from_numpy.concordance == from_lists.concordance
    assert from_numpy.count_strata == from_lists.count_strata
    assert from_numpy.var == from_lists.var
    assert from_numpy.dfbeta == from_lists.dfbeta
