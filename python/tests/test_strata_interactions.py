"""Stratum-specific covariate effects against R survival 3.8-12."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/strata_interaction_reference.json").read_text()
)


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-7, atol=2e-9
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_strata_interaction_fits_and_predictions_match_r(case):
    options = dict(case["options"] or {})
    fit = getattr(r, case["kind"])(case["formula"], REFERENCE["data"], model=True, **options)
    assert r.coef_names(fit) == case["beta_names"]
    assert_close(r.coef(fit), case["beta"])
    assert_close(r.vcov(fit), case["variance"])
    matrix = r.model_matrix(fit)
    assert matrix["assign"] == case["assign"]
    assert_close(matrix["data"], case["x"])
    assert_close(r.model_matrix(fit, REFERENCE["newdata"])["data"], case["newx"])
    assert_close(r.predict(fit, type="lp"), case["lp"])
    prediction = r.predict(fit, REFERENCE["newdata"], type="lp", se_fit=True)
    assert_close(prediction.fit, case["prediction"])
    assert_close(prediction.se_fit, case["prediction_se"])
    if case["kind"] == "coxph" and (case["prediction_error"] or isinstance(case["terms"], dict)):
        # R's predict.coxph drops a column by term number after a multi-column
        # factor. LP prediction errors; term prediction silently uses wrong
        # columns. The fixture also records the correct R model.matrix result.
        assert_close(
            r.predict(fit, REFERENCE["newdata"], type="terms"), case["terms_from_model_matrix"]
        )
    elif not isinstance(case["terms"], dict):
        assert_close(r.predict(fit, REFERENCE["newdata"], type="terms"), case["terms"])
    if case["kind"] == "survreg":
        assert_close(fit.scale, case["scale"])
        return
    assert_close(r.residuals(fit, type="martingale"), case["martingale"])
    assert_close(r.residuals(fit, type="schoenfeld").values, case["schoenfeld"])
    if "survfit" in case:
        curves = r.survfit(fit, newdata=REFERENCE["newdata"])
        for name, expected in case["survfit"].items():
            assert_close(getattr(curves, name), expected)
    if "zph" in case:
        zph = r.cox_zph(fit)
        for name, expected in case["zph"].items():
            actual = getattr(zph, name)
            if name == "table":
                actual = [[row["chisq"], row["df"], row["p"]] for row in actual]
            assert_close(actual, expected)


def test_strata_interaction_curves_require_newdata():
    fit = r.coxph("Surv(time,status) ~ age * strata(sex)", REFERENCE["data"])
    with pytest.raises(ValueError, match="interaction terms require newdata"):
        r.survfit(fit)
    with pytest.raises(ValueError, match="interaction terms require newdata"):
        r.basehaz(fit)


@pytest.mark.parametrize("population", ["data", "sas", "factorial"])
def test_yates_averages_compound_strata_without_reparsing_labels(population):
    fit = r.coxph("Surv(time,status) ~ age * strata(sex, ph.ecog)", REFERENCE["data"])
    means = []
    for age in (50, 70):
        frame = (
            {**REFERENCE["data"], "age": [age] * len(REFERENCE["data"]["age"])}
            if population == "data"
            else {"age": [age] * 6, "sex": [1, 1, 1, 2, 2, 2], "ph.ecog": [0, 1, 2, 0, 1, 2]}
        )
        means.append(np.mean(r.model_matrix(fit, frame)["data"], axis=0))
    cmat = np.asarray(means)
    variance = cmat @ fit.var @ cmat.T
    expected = cmat @ fit.coefficients - np.dot(fit.means, fit.coefficients)
    result = r.yates(fit, "age", levels=[50, 70], population=population)
    assert_close(result.cmat, cmat)
    assert_close(result.estimate["pmm"], expected)
    assert_close(result.estimate["std"], np.sqrt(np.diag(variance)))


def test_strata_interaction_preserves_declared_factor_order():
    data = REFERENCE["data"]
    numeric = r.coxph("Surv(time,status) ~ age:strata(sex)", data)
    factor = RFactor([str(value) for value in data["sex"]], ["2", "1"])
    fit = r.coxph("Surv(time,status) ~ age:strata(sex)", {**data, "sex": factor})
    assert r.coef_names(fit) == ["age:strata(sex)2", "age:strata(sex)1"]
    assert_close(r.coef(fit), list(reversed(r.coef(numeric))))


def test_survreg_model_matrix_newdata_drops_incomplete_rows():
    fit = r.survreg("Surv(time,status) ~ age * strata(sex)", REFERENCE["data"])
    matrix = r.model_matrix(fit, {"age": [50, None, 70], "sex": [1, 2, 2]})
    assert matrix["columns"] == r.coef_names(fit)
    assert matrix["data"] == [[1, 50, 0], [1, 70, 70]]


def test_survreg_term_selection_skips_all_scale_strata():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "aft_two_strata_calls")
    fit = r.survreg(case["formula"], REFERENCE["data"])
    selected = r.predict(fit, REFERENCE["newdata"], type="terms", terms="age:strata(sex)")
    assert_close(selected, np.asarray(case["terms"])[:, 1:2])
    missing = r.predict(
        fit,
        {"age": [None, None], "sex": [1, 2], "ph.ecog": [0, 1]},
        type="terms",
        na_action="na.pass",
    )
    assert np.asarray(missing).shape == (2, 2)
    assert np.isnan(missing).all()


def test_aliased_strata_factor_zph_has_the_same_global_test():
    # R's cox.zph errors for the redundant full indicators, though its
    # treatment-coded equivalent has the same four estimable effects.
    fit = r.coxph("Surv(time,status) ~ strata(sex):factor(ph.ecog)", REFERENCE["data"])
    table = r.cox_zph(fit).table[-1]
    equivalent = next(case for case in REFERENCE["cases"] if case["name"] == "factor_star")
    assert_close([table["chisq"], table["df"], table["p"]], equivalent["zph"]["table"][-1])
