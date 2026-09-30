"""Expected survival from baselines, without subject-by-time curve storage."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
population = survival.population


@pytest.mark.parametrize("ties", ["efron", "breslow", "exact"])
@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("stratified", [False, True])
@pytest.mark.parametrize("method", ["ederer", "hakulinen", "conditional"])
def test_fitted_aggregation_matches_expanded_curves(ties, counting, stratified, method):
    from survival.r._coxph import _survfit_curves, _survfit_newdata

    rng = np.random.default_rng(38)
    n = 60
    data = {
        "time": rng.integers(2, 15, n),
        "status": rng.integers(0, 2, n),
        "start": rng.uniform(0, 1, n),
        "x": rng.normal(size=n),
        "z": np.arange(n) % 3,
        "off": rng.normal(size=n),
        "w": rng.uniform(0.5, 2, n),
    }
    formula = "Surv(start, time, status)" if counting else "Surv(time, status)"
    formula += " ~ x + offset(off)" + (" + strata(z)" if stratified else "")
    fit = r.coxph(formula, data, ties=ties, weights=None if ties == "exact" else "w")
    newdata = {key: value[:17] for key, value in data.items()}
    new, _, _ = _survfit_newdata(fit, newdata, individual=False, id=None, na_action="na.fail")
    curves, _, _ = _survfit_curves(
        fit,
        newdata,
        individual=False,
        id=None,
        stype=2,
        ctype=2 if ties == "efron" else 1,
        se_fit=False,
        censor=False,
    )
    group = np.arange(17, dtype=np.int32) % 4
    weights = np.linspace(0.5, 2, 17)
    response = newdata["time"]
    for times in [None, [0, 0.5, 2, 2, 3.5, 8, 20]]:
        expected = population.survexp_cox(
            curves,
            group.tolist(),
            weights.tolist(),
            y=response.tolist(),
            times=times,
            method=method,
        )
        actual = fit.fit.expected_survival(
            new.x,
            group,
            weights,
            new_strata=new.strata,
            new_offset=new.offset,
            y=response,
            times=times,
            method=method,
        )
        np.testing.assert_equal(actual.time, expected.time)
        np.testing.assert_equal(actual.n_risk, expected.n_risk)
        np.testing.assert_allclose(actual.surv, expected.surv, rtol=2e-14, atol=2e-14)


def _args():
    return {
        "time": [1.0, 3.0, 2.0, 4.0],
        "cumhaz": [0.2, 0.7, 0.1, 0.4],
        "lengths": [2, 2],
        "risk": [0.5, 2.0, 1.0, 3.0],
        "strata": [1, 0, 1, 0],
        "group": [0, 1, 0, 1],
        "weights": [1.0, 2.0, 0.5, 1.0],
        "y": [2.0, 4.0, 3.0, 4.0],
        "times": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
    }


@pytest.mark.parametrize("layout", ["list", "float32", "strided", "readonly"])
def test_numeric_baselines_and_array_ownership(layout):
    args = _args()
    expected = population.survexp_cox_prepared(**args)
    for name, values in args.items():
        dtype = (
            np.int32
            if name in {"lengths", "strata", "group"}
            else (np.float32 if layout == "float32" else np.float64)
        )
        if layout != "list":
            array = np.repeat(np.asarray(values, dtype=dtype), 2)[::2]
            if layout == "readonly":
                array.flags.writeable = False
            args[name] = array
    result = population.survexp_cox_prepared(**args)
    arrays = result.to_arrays()
    np.testing.assert_allclose(arrays["surv"], expected.surv, rtol=2e-7)
    assert arrays["surv"].shape == (6, 2)
    assert arrays["n_risk"].dtype == np.float64
    arrays["time"][:] = -8
    arrays["surv"][:] = -9
    assert result.time[0] == 0
    assert result.surv[0] == [1, 1]
    del result
    assert arrays["surv"][0, 0] == -9


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"lengths": [3, 2]}, "lengths"),
        ({"lengths": [-1, 5]}, "nonnegative"),
        ({"strata": [0, 0, 2, 0]}, "baseline"),
        ({"group": [0, 2, 0, 2]}, "positive"),
        ({"weights": [0, 0, 0, 0]}, "positive"),
        ({"risk": [-1, 2, 1, 3]}, "risk"),
        ({"time": [3, 1, 2, 4]}, "increase"),
        ({"cumhaz": [0.7, 0.2, 0.1, 0.4]}, "decrease"),
        ({"times": []}, "nonempty"),
        ({"times": [1, float("nan")]}, "times"),
    ],
)
def test_prepared_input_validation(updates, message):
    with pytest.raises(ValueError, match=message):
        population.survexp_cox_prepared(**{**_args(), **updates})


def test_prepared_empty_stratum_and_calls_are_independent():
    args = {**_args(), "time": [], "cumhaz": [], "lengths": [0, 0]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: population.survexp_cox_prepared(**args).surv, range(12)))
    assert results == [[[1, 1]] * 6] * 12


def test_formula_cohorts_do_not_expand_subject_curves(monkeypatch):
    import survival.r._coxph as cox

    data = {"time": [1, 2, 3, 4, 5, 6], "status": [1, 1, 0, 1, 0, 1], "x": [0, 1, 2, 0, 1, 2]}
    fit = r.coxph("Surv(time, status) ~ ridge(x, theta=1)", data)
    monkeypatch.setattr(cox, "_survfit_curves", lambda *a, **kw: pytest.fail("expanded curves"))
    result = r.survexp("~ 1", data, ratetable=fit, times=[0, 1, 4])
    assert result.surv[0] == 1
    assert result.surv[-1] < result.surv[1] < 1


def test_fitted_expected_survival_shares_a_cold_baseline_cache_safely():
    rng = np.random.default_rng(981)
    x = rng.normal(size=(100, 2))
    fit = survival.regression.coxph_fit(
        np.arange(1, 101, dtype=float), np.ones(100, dtype=np.int32), x
    )

    def call(_):
        return fit.expected_survival(x[:12], [0] * 12, [1.0] * 12, times=[0, 10, 50]).surv

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(call, range(12)))
    expected = population.survexp_cox(
        fit.survfit(newdata=x[:12], censor=False, se_fit=False),
        [0] * 12,
        [1.0] * 12,
        times=[0, 10, 50],
    ).surv
    for value in actual:
        np.testing.assert_allclose(value, expected, rtol=2e-14)
