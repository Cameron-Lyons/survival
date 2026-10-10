"""Scalar curve-quantile scaling against independent stock survival calls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from survival.r._survfit import _cox_engines, _engine_of
from survival.r._types import CoxSurvfitResult

from .test_curve_quantile_boundaries import (
    REFERENCE,
    assert_quantiles,
    core,
    fitted,
    number,
    r,
)


@pytest.mark.parametrize("case", REFERENCE["scale_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("interface", ["facade", "prepared", "stacked"])
def test_scalar_scales_match_stock_r(case, interface):
    scale = case["scale"] if case["scale_kind"] == "logical" else number(case["scale"])
    probs = np.asarray([number(value) for value in case["probs"]])
    fit = fitted(case["fit"])
    options = {"conf_int": case["conf_int"], "scale": scale}
    if interface == "facade":
        actual = (
            r.median(fit, scale=scale)
            if case["method"] == "median"
            else r.quantile(fit, probs, **options)
        )
    else:
        if isinstance(fit, CoxSurvfitResult):
            engines = _cox_engines(fit)
            options["start_time"] = 0.0 if fit.start_time is None else fit.start_time
        else:
            engines = [_engine_of(fit)]
        results = []
        for engine in engines:
            if interface == "prepared":
                results.append(core.quantile_survfit(engine, probs, **options))
            else:
                results.append(
                    core.quantile_survfit_curves(
                        np.asarray(engine.time),
                        np.asarray(engine.surv),
                        lower=None if engine.lower is None else np.asarray(engine.lower),
                        upper=None if engine.upper is None else np.asarray(engine.upper),
                        strata=engine.strata,
                        probs=probs,
                        **options,
                    )
                )
        fields = {}
        for field in ("quantile", "lower", "upper"):
            parts = [getattr(result, field) for result in results]
            fields[field] = None if parts[0] is None else [row for part in parts for row in part]
        actual = SimpleNamespace(**fields)
    assert_quantiles(actual, case["expected"])
    for field in ("quantile", "lower", "upper"):
        wanted = case["expected"].get(field)
        if wanted is not None:
            wanted = np.asarray(wanted, dtype=float)
            values = np.asarray(getattr(actual, field), dtype=float)
            observed = ~np.isnan(wanted)
            np.testing.assert_array_equal(
                np.signbit(values)[observed], np.signbit(wanted)[observed]
            )


@pytest.mark.parametrize("scale", [0.0, -0.0, -2.0, float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize(
    "container",
    [
        lambda value: value,
        lambda value: [value],
        lambda value: iter([value]),
        lambda value: np.asarray(value),
        lambda value: np.asarray([value]),
        lambda value: pd.Series([value]),
        lambda value: pl.Series([value]),
    ],
    ids=["scalar", "list", "iterator", "zero-dimensional", "array", "pandas", "polars"],
)
@pytest.mark.parametrize("response", [False, True], ids=["fit", "Surv"])
def test_numeric_scalar_scale_containers_preserve_nonfinite_results(scale, container, response):
    y = r.Surv([1, 2, 3, 4])
    fit = y if response else r.survfit(y)
    result = r.quantile(fit, [0, 0.5, 1], conf_int=False, scale=container(scale))
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.asarray([[0, 2.5, 4]]) / scale
    np.testing.assert_array_equal(result.quantile, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        np.signbit(np.asarray(result.quantile))[finite], np.signbit(expected)[finite]
    )


@pytest.mark.parametrize(
    "missing",
    [
        pd.NA,
        [None],
        np.asarray(pd.NA, dtype=object),
        np.asarray(None, dtype=object),
        pd.Series([pd.NA], dtype="Float64"),
        pl.Series([None], dtype=pl.Float64),
    ],
)
def test_missing_numeric_scalar_scales_produce_undefined_quantiles(missing):
    result = r.quantile(fitted("km_right"), [0, 0.5, 1], scale=missing)
    for field in ("quantile", "lower", "upper"):
        assert np.isnan(getattr(result, field)).all()


@pytest.mark.parametrize("scale", ["1", b"1", np.asarray("1"), ["1"], pd.Series(["1"])])
def test_character_scales_refuse_r_numeric_operator_coercion(scale):
    with pytest.raises(TypeError, match="scale must be numeric"):
        r.quantile(fitted("km_right"), scale=scale)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_summary_tables_keep_their_separate_scale_contract(scale):
    with pytest.raises(ValueError, match="scale"):
        r.summary_survfit(fitted("km_right"), scale=scale)
