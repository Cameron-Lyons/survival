"""Public AFT rank reporting checked against R survival 3.8.11."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _duplicate_fit(*, matrix=False, robust=False):
    data = {
        "time": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "status": [1] * 6,
        "x": list(range(6)),
        "duplicate": list(range(6)),
    }
    options = {
        "dist": "gaussian",
        "scale": 1.0,
        "init": [2.0, 1.0, 4.0],
        "control": {"maxiter": 0},
        "robust": robust,
    }
    if matrix:
        fit = survival.survreg(
            time=data["time"],
            status=data["status"],
            covariates=[[1.0, value, value] for value in data["x"]],
            **options,
        )
    else:
        fit = survival.survreg("Surv(time, status) ~ x + duplicate", data=data, **options)
    return fit, data


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("robust", [False, True])
def test_survreg_aliases_are_reported_without_discarding_fitted_values(matrix, robust):
    fit, data = _duplicate_fit(matrix=matrix, robust=robust)
    assert fit.coefficients[:2] == [2.0, 1.0]
    assert math.isnan(fit.coefficients[2])
    assert math.isnan(survival.coef(fit)[2])
    assert fit.location_coefficients == [2.0, 1.0, 4.0]
    assert fit.fit.coefficients == [2.0, 1.0, 4.0]
    assert fit.linear_predictors == [2.0, 7.0, 12.0, 17.0, 22.0, 27.0]
    assert survival.predict(fit, type="lp") == fit.linear_predictors
    assert survival.r_api.residuals(fit, type="response") == [-1.0, -5.0, -9.0, -12.0, -15.0, -17.0]
    assert all(row[2] == 0.0 for row in survival.r_api.residuals(fit, type="dfbeta"))
    assert all(math.isnan(row[2]) for row in survival.r_api.residuals(fit, type="dfbetas"))
    assert survival.degrees_freedom(fit) == 3
    assert survival.df_residual(fit) == len(data["time"]) - 3
    summary = survival.model_summary(fit)["coefficients"][2]
    assert math.isnan(summary["coef"])
    assert summary["se"] == 0.0
    assert math.isnan(summary["statistic"])
    assert math.isnan(summary["p"])
    interval = survival.confint(fit)[2]
    assert math.isnan(interval["lower"])
    assert math.isnan(interval["upper"])
    assert survival.vcov(fit)[2] == [0.0, 0.0, 0.0]
    assert len(survival.vcov(fit, complete=False)) == 2
    if not robust:
        for actual, expected in zip(
            survival.vcov(fit, complete=False),
            [[11.0 / 21.0, -1.0 / 7.0], [-1.0 / 7.0, 2.0 / 35.0]],
            strict=True,
        ):
            assert actual == pytest.approx(expected, abs=1e-13, rel=0.0)


@pytest.mark.parametrize("kind", ["lp", "response", "quantile", "uquantile"])
def test_survreg_aliased_newdata_predictions_follow_r(kind):
    fit, data = _duplicate_fit()
    options = {"p": [0.5]} if kind in {"quantile", "uquantile"} else {}
    training = survival.predict(fit, type=kind, se_fit=True, **options)
    newdata = survival.predict(fit, data, type=kind, se_fit=True, **options)
    assert training.fit == fit.linear_predictors
    assert all(math.isnan(value) for value in newdata.fit)
    assert newdata.se_fit == pytest.approx(training.se_fit)


def test_survreg_alias_term_predictions_preserve_other_terms():
    fit, data = _duplicate_fit()
    for newdata in (None, data):
        result = survival.predict(fit, newdata, type="terms", se_fit=True)
        for idx, (values, se) in enumerate(zip(result.fit, result.se_fit, strict=True)):
            assert values[0] == idx - 2.5
            assert math.isnan(values[1])
            assert se[0] == pytest.approx(abs(idx - 2.5) * math.sqrt(2.0 / 35.0))
            assert math.isnan(se[1])


def test_zero_sandwich_variance_marks_location_aliases_like_r():
    data = {"time": [2.0 + 3.0 * idx for idx in range(6)], "status": [1] * 6, "x": list(range(6))}
    fit = survival.survreg(
        "Surv(time, status) ~ x",
        data=data,
        dist="gaussian",
        scale=1.0,
        init=[2.0, 3.0],
        control={"maxiter": 0},
        robust=True,
    )
    assert fit.variance_matrix == [[0.0, 0.0], [0.0, 0.0]]
    assert all(math.isnan(value) for value in survival.coef(fit))
    assert fit.location_coefficients == [2.0, 3.0]
    assert fit.naive_var[0][0] > 0.0
    assert survival.coef_names(fit, complete=False) == []
    assert survival.vcov(fit, complete=False) == []
    assert survival.predict(fit, type="response") == data["time"]
    assert all(math.isnan(value) for value in survival.predict(fit, data))


def test_discarded_scale_pivot_does_not_alias_location_or_scale():
    data = {"time": [2.0 + 3.0 * idx for idx in range(6)], "status": [1] * 6, "x": list(range(6))}
    fit = survival.survreg(
        "Surv(time, status) ~ x",
        data=data,
        dist="gaussian",
        init=[2.0, 3.0, 1.0],
        control={"maxiter": 0},
    )
    assert fit.coefficients == [2.0, 3.0, 1.0]
    assert survival.coef(fit) == [2.0, 3.0]
    assert len(survival.vcov(fit, complete=False)) == 3
    row = survival.model_summary(fit)["coefficients"][2]
    assert row["coef"] == 1.0
    assert row["se"] == 0.0
    assert row["statistic"] == math.inf
    assert row["p"] == 0.0


@pytest.mark.parametrize("distribution", ["gaussian", "lognormal"])
@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("kind", ["lp", "response", "quantile", "uquantile", "terms"])
def test_survreg_newdata_offset_predictions_match_r(distribution, aliased, kind):
    data = {
        "time": [2.0 + 3.0 * idx for idx in range(6)],
        "status": [1] * 6,
        "x": list(range(6)),
        "duplicate": list(range(6)),
        "off": [(idx + 1) / 10.0 for idx in range(6)],
    }
    formula = "Surv(time, status) ~ x + " + ("duplicate + " if aliased else "") + "offset(off)"
    fit = survival.survreg(
        formula,
        data=data,
        dist=distribution,
        scale=1.0,
        init=[2.0, 1.0, 4.0] if aliased else [2.0, 1.0],
        control={"maxiter": 0},
    )
    options = {"p": [0.25, 0.75]} if kind in {"quantile", "uquantile"} else {}
    training = survival.predict(fit, type=kind, **options)
    newdata = survival.predict(fit, data, type=kind, **options)
    if kind == "terms":
        for idx, (original, new) in enumerate(zip(training, newdata, strict=True)):
            assert original[0] == idx - 2.5
            assert new[0] == original[0]
            if aliased:
                assert math.isnan(original[1])
                assert math.isnan(new[1])
        return

    # R survival 3.8.11, prescribed coefficients and qnorm(c(.25, .75)).
    # Stored training LP retains offsets and raw aliased coefficients;
    # newdata uses the reported coefficients and omits formula offsets.
    scores = [-0.6744897501960817, 0.6744897501960817] if options else [0.0]
    transform = (
        math.exp if distribution == "lognormal" and kind in {"response", "quantile"} else float
    )
    for idx, (original, new) in enumerate(zip(training, newdata, strict=True)):
        original = original if options else [original]
        new = new if options else [new]
        lp = 2.0 + (5.0 if aliased else 1.0) * idx + data["off"][idx]
        assert original == pytest.approx([transform(lp + score) for score in scores])
        if aliased:
            assert all(math.isnan(value) for value in new)
        else:
            assert new == pytest.approx([transform(2.0 + idx + score) for score in scores])
