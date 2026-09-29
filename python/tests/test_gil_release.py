"""The core kernels run without the GIL, and NumPy inputs cross the boundary as lists do.

Each heavy binding converts its arguments while attached and releases the GIL around the
kernel (``py.detach``), so Python threads keep running during a fit and fits on several
threads overlap.  The typed boundary (``FloatVec``/``IntVec``/``FloatMatrix``) reads NumPy
arrays of any layout without a ``.tolist()``, and gives the same fit as nested lists.
"""

from __future__ import annotations

import os
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
    best_share = 0.0
    elapsed = 0.0
    # Shared CI runners can briefly deschedule the spinner. Retry the complete
    # measurement, including its baseline, without lowering the GIL-release threshold.
    for _ in range(3):
        idle_steps, idle_time = _python_steps_during(lambda: time.sleep(0.02))
        steps, elapsed = _python_steps_during(call)
        share = (steps / elapsed) / (idle_steps / idle_time)
        best_share = max(best_share, share)
        if share > 0.1:
            return
    assert best_share > 0.1, (
        f"the thread ran for at most {best_share:.1%} across three measurements "
        f"(last call: {elapsed:.3f} s)"
    )


@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs four cores to overlap four fits")
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
    survreg = regression.survreg_fit(survreg_data, weibull)
    right = core.SurvivalData(time_, status)
    predictor = core.CovariateMatrix(x[:, 0], len(time_), 1)
    group = (x[:, 0] > 0).astype(np.int32)
    aj_time, aj_status, aj_x = _cox_data(1_000)
    states = (aj_status * (1 + (aj_x[:, 1] > 0))).astype(np.int32)
    transition_curve = sa.survfitkm(time_, status, se_fit=False)

    calls = {
        "coxph_fit": lambda: regression.coxph_fit(time_, status, x),
        "coxph_fit_raw": lambda: regression.coxph_fit_raw(time_, status, x, resid=False),
        "CoxPHFit.dfbeta": lambda: fit.dfbeta(),
        "CoxPHFit.survfit": lambda: fit.survfit(x[:5]),
        "CoxPHFit.predict_survival_at": lambda: fit.predict_survival_at(np.linspace(0.0, 3.0, 128)),
        "cox_zph": lambda: regression.cox_zph(fit),
        "cox_zph_smooth": lambda: regression.cox_zph_smooth(time_, x, [1.0, 1.0]),
        "survreg_fit": lambda: regression.survreg_fit(survreg_data, weibull),
        "SurvregFit.predict": lambda: survreg.predict(design, "quantile", se_fit=True),
        "SurvregFit.residuals": lambda: survreg.residuals("dfbeta"),
        "concordancefit": lambda: core.concordancefit(right, predictor),
        "survdiff": lambda: sa.survdiff(time_, status, group),
        "survfitaj": lambda: sa.survfitaj(aj_time, states, ["censor", "a", "b"]),
        "survfit_matrix": lambda: sa.survfit_matrix(
            [[transition_curve], [transition_curve]], [0, 1], [1, 2], ["a", "b", "c"]
        ),
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


def test_yates_sgtt_releases_the_gil():
    index = np.arange(200_000)
    a, b = index % 2, (index // 2) % 2
    x = np.column_stack(
        [
            np.ones(len(index)),
            a,
            1 - a,
            b,
            1 - b,
            a * b,
            (1 - a) * b,
            a * (1 - b),
            (1 - a) * (1 - b),
        ]
    )
    _assert_detaches(
        lambda: survival.validation.yates_sgtt(
            x,
            [0, 1, 1, 2, 2, 3, 3, 3, 3],
            [[3], [3], []],
            [2, 3, 4, 5],
            np.eye(4),
            [0, 1, 2, 3],
            [(1, "a")],
        )
    )


def test_rate_matching_releases_the_gil():
    labels = [f"group{i:04d}" for i in range(1024)]
    table = population.RateTable([1024], ["group"], [labels], [None], [1], [0.01] * 1024)
    observations = [labels[i % 1024] for i in range(500_000)]
    _assert_detaches(lambda: population.match_ratetable(table, ["group"], [observations]))


@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_numpy_and_list_inputs_give_identical_survreg_fits(layout):
    time_, status, x = _cox_data(200)
    design = np.column_stack([np.ones(200), x])
    if layout == "fortran":
        array = np.asfortranarray(design)
    else:
        array = np.column_stack([design, x])[:, :3]
    assert not array.flags.c_contiguous
    weibull = regression.SurvregDistribution("weibull")
    from_numpy = regression.survreg_fit(
        regression.SurvregData(time_ + 0.01, status, array), weibull
    )
    from_lists = regression.survreg_fit(
        regression.SurvregData((time_ + 0.01).tolist(), status.tolist(), design.tolist()),
        weibull,
    )
    assert from_numpy.coefficients == from_lists.coefficients
    assert from_numpy.variance_matrix == from_lists.variance_matrix
    assert from_numpy.covariates == design.tolist()
    assert regression.SurvregData(time_, status, array).covariates == design.tolist()

    new = array[:7]
    for kind in ("response", "quantile", "terms"):
        numpy_pred = from_numpy.predict(new, kind, se_fit=True)
        list_pred = from_lists.predict(new.tolist(), kind, se_fit=True)
        assert numpy_pred.fit == list_pred.fit
        assert numpy_pred.se_fit == list_pred.se_fit
    assert from_numpy.predict(np.empty((0, 3)), "lp").fit == []
    assert from_numpy.predict([], "lp").fit == []
    with pytest.raises(ValueError, match="newdata must be 2 x 3, got 2 x 2"):
        from_numpy.predict(x[:2])


def test_numpy_and_list_inputs_give_identical_pyears():
    n = 300
    rng = np.random.default_rng(3)
    stop = rng.uniform(30, 3650, size=n)
    event = (rng.uniform(size=n) < 0.3).astype(float)
    group = 1.0 + (rng.uniform(size=n) < 0.5)
    age = rng.uniform(40, 70, size=n) * 365.25
    year = rng.uniform(10957, 12949, size=n)
    us = population.survexp_us()
    positions = np.array(
        population.match_ratetable(us, ["age", "sex", "year"], [age, group, year]).r
    )
    kwargs = {"factors": [1], "dims": [2], "cuts": [[]], "ratetable": us, "scale": 365.25}
    from_numpy = population.pyears(
        stop,
        event=event,
        categories_data=group[:, None],
        ratetable_positions=np.asfortranarray(positions),
        **kwargs,
    )
    from_lists = population.pyears(
        stop.tolist(),
        event=event.tolist(),
        categories_data=[[g] for g in group.tolist()],
        ratetable_positions=positions.tolist(),
        **kwargs,
    )
    assert from_numpy.pyears == from_lists.pyears
    assert from_numpy.expected == from_lists.expected


def test_pyears_reads_an_empty_categories_matrix_as_no_categories():
    stop = [365.25, 1826.25, 730.5]
    without = population.pyears(stop, event=[1.0, 0.0, 1.0])
    for empty in ([], np.empty((0, 0)), np.empty((3, 0))):
        result = population.pyears(stop, event=[1.0, 0.0, 1.0], categories_data=empty)
        assert result.pyears == without.pyears
        assert result.event == without.event


def test_ragged_nested_lists_are_rejected_at_the_boundary():
    with pytest.raises(ValueError, match="row 1 length mismatch"):
        sa.aggregate_survfit(surv=[[0.9, 0.8], [0.7]])


def test_matrix_conversion_does_not_repeat_custom_float_coercion():
    class Number:
        calls = 0

        def __float__(self):
            self.calls += 1
            return 1.0

    value = Number()
    with pytest.raises(TypeError, match="a float matrix"):
        regression.SurvregData([1.0, 2.0], [1, 0], [[value, 1.0], ["invalid", 2.0]])
    assert value.calls == 1
    value.calls = 0
    data = regression.SurvregData([1.0, 2.0], [1, 0], [[value, 2.0], [3.0, 4.0]])
    assert data.covariates == [[1.0, 2.0], [3.0, 4.0]]
    assert value.calls == 1


def test_matrix_conversion_preserves_custom_row_iteration():
    class Row(list):
        def __iter__(self):
            return iter([3.0, 4.0])

    data = regression.SurvregData([1.0, 2.0], [1, 0], [[1.0, 2.0], Row([9.0, 8.0])])
    assert data.covariates == [[1.0, 2.0], [3.0, 4.0]]


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
