"""Raw response and fitted-curve quantiles against R survival 3.8-12."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
core = survival._survival
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/surv_quantile_reference.json").read_text()
)


def response(case):
    return r.Surv(*zip(*case["response"], strict=True), type=case["type"])


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        atol=2e-10,
        rtol=2e-9,
        equal_nan=True,
    )


def check_result(result, expected, probs):
    assert isinstance(result, r.SurvfitQuantileResult)
    assert result.probs == list(probs)
    assert result.strata is None
    for name in ("quantile", "lower", "upper"):
        close(getattr(result, name)[0], np.atleast_1d(expected[name]))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_response_quantiles_and_medians_match_r(case):
    y = response(case)
    probs = REFERENCE["probs"]
    for function in (r.quantile_surv, r.quantile, survival.quantile):
        check_result(function(y, probs, na_rm=True), case["quantile"], probs)
    for function in (r.median_surv, r.median, survival.median):
        check_result(function(y, na_rm=True), case["median"], [0.5])
    plain = r.quantile(y, probs, na_rm=True, conf_int=False, scale=2)
    close(plain.quantile[0], case["plain"])
    assert plain.lower is None
    assert plain.upper is None
    without = r.median(y, na_rm=True, conf_int=False)
    assert without.lower is None
    assert without.upper is None
    close(without.quantile[0], np.atleast_1d(case["median"]["quantile"]))


def test_missing_response_rows_require_explicit_removal():
    y = response(REFERENCE["cases"][-1])
    for function in (r.quantile, r.median, r.quantile_surv, r.median_surv):
        with pytest.raises(ValueError, match="missing values and NaN's not allowed"):
            function(y)
        assert function(y, **{"na.rm": True}).quantile
        with pytest.raises(ValueError, match="use only one of na_rm or na.rm"):
            function(y, na_rm=True, **{"na.rm": True})
    with pytest.raises(ValueError, match="no non-missing observations"):
        r.quantile(r.Surv([math.nan], [1]), na_rm=True)
    with pytest.raises(ValueError, match="no non-missing observations"):
        r.median(r.Surv([], []))


def test_defaults_aliases_and_scalar_array_probabilities():
    y = r.Surv([1, 2, 3, 4])
    assert r.quantile(y).probs == [0.25, 0.5, 0.75]
    assert r.quantile(y).quantile == [[1.5, 2.5, 3.5]]
    for p in (0.5, np.float32(0.5), np.array(0.5)):
        assert r.quantile(y, p).quantile == [[2.5]]
    assert r.quantile(y, []).quantile == [[]]
    assert r.median(y, **{"conf.int": False}).lower is None
    assert r.median(y).lower is not None
    assert r.median(r.survfit(y)).lower is None
    with pytest.raises((ValueError, TypeError), match="conf"):
        r.median(r.survfit(y), conf_int=True)


@pytest.mark.parametrize("p", [math.nan, math.inf, -0.01, 1.01])
def test_invalid_probabilities_are_rejected(p):
    with pytest.raises(ValueError, match="probability"):
        r.quantile(r.Surv([1, 2, 3]), p)


def test_quantile_generics_refuse_unrelated_and_multistate_objects():
    states = pd.Categorical(["censor", "a", "b", "a"], categories=["censor", "a", "b"])
    for y in (r.Surv([1, 2, 3, 4], states), r.Surv([0, 0, 0, 0], [1, 2, 3, 4], states)):
        for function in (r.quantile, r.median):
            with pytest.raises(ValueError, match="not defined for multiple-endpoint"):
                function(y)
            with pytest.raises(ValueError, match="not a well defined quantity for multi-state"):
                function(r.survfit(y, id=list(range(len(y)))))
    for function in (r.quantile, r.median):
        with pytest.raises(TypeError, match="requires a Surv response or survfit object"):
            function([1, 2, 3])


@pytest.mark.parametrize("kind", ["km", "cox"])
def test_fitted_medians_preserve_strata_cox_columns_and_origin(kind):
    data = REFERENCE["data"]
    if kind == "km":
        fit = r.survfit("Surv(time, status) ~ g", data)
        labels = ["g=0", "g=1"]
    else:
        model = r.coxph("Surv(time, status) ~ x + strata(g)", data)
        fit = r.survfit(model, newdata={"x": [-0.5, 0.7]}, start_time=2)
        labels = ["g=0, 1", "g=1, 1", "g=0, 2", "g=1, 2"]
    expected = REFERENCE["fit_cases"][kind]
    quantiles = r.quantile(fit, REFERENCE["probs"], conf_int=False)
    close(quantiles.quantile, expected["quantile"])
    for function in (r.median_survfit, r.median, survival.median):
        result = function(fit)
        assert result.probs == [0.5]
        assert result.strata == labels
        assert result.lower is None
        assert result.upper is None
        close(result.quantile, expected["median"])
        close(function(fit, scale=2).quantile, np.asarray(expected["median"], dtype=float) / 2)


def test_nonmonotone_confidence_bands_follow_r_approx_ordering():
    fit = core.SurvfitKMResult.from_stacked(
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1],
        [1] * 6,
        [5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0],
        [6],
        upper=[0.95, 0.7, 0.85, 0.5, 0.4, 0.2],
        lower=[0.7, 0.55, 0.65, 0.3, 0.15, 0.05],
    )
    q = core.quantile_survfit(fit, probs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])
    close(q.quantile[0], [1, 2, 2, 3, 3.5, 4, 5, 6])
    close(q.lower[0], [1, 1, 2, 2, 4, 4, 5, 6])
    close(q.upper[0], [3, 2, 3, 4, 4.5, 5.5, 6, math.nan])
