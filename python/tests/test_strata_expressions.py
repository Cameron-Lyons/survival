"""Evaluated strata formulas against R survival, including missing-value groups."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/strata_expression_reference.json").read_text()
)
pytestmark = pytest.mark.filterwarnings(r"ignore:NaNs produced in sqrt\(z\):UserWarning")


def frame(name):
    data = dict(REFERENCE[name])
    data["label"] = RFactor(data["label"], REFERENCE["label_levels"])
    return data


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-7, atol=2e-9
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_strata_expressions_match_r_fits_and_predictions(case):
    fit = getattr(r, case["kind"])(case["formula"], frame("data"), model=True)
    assert r.coef_names(fit) == case["beta_names"]
    assert_close(r.coef(fit), case["beta"])
    assert_close(r.vcov(fit), case["variance"])
    assert list(fit.na_action.rows if fit.na_action else ()) == case["omitted"]
    matrix = r.model_matrix(fit)
    assert len(matrix["data"]) == case["n"]
    assert matrix["assign"] == case["assign"]
    assert_close(matrix["data"][:8], case["x_head"])
    assert_close(r.predict(fit, type="lp"), case["lp"])
    assert list(fit.strata_levels) == case["used_strata_levels"]
    predicted = r.predict(fit, frame("newdata"), type="lp", se_fit=True)
    assert_close(predicted.fit, case["prediction"])
    assert_close(predicted.se_fit, case["prediction_se"])
    assert_close(r.predict(fit, frame("newdata"), type="terms"), case["terms"])
    if case["kind"] == "survreg":
        assert_close(fit.scale, case["scale"])
        assert_close(
            r.predict(fit, frame("newdata"), type="quantile", p=[0.25, 0.5, 0.75]),
            case["quantiles"],
        )
    else:
        expected = r.predict(fit, frame("newdata"), type="expected", se_fit=True)
        assert_close(expected.fit, case["expected"])
        assert_close(expected.se_fit, case["expected_se"])
    for prediction in case.get("missing_predictions", []):
        if prediction["error"]:
            with pytest.raises(ValueError, match="missing values"):
                r.predict(fit, frame("missing_newdata"), na_action=prediction["action"])
        else:
            result = r.predict(
                fit,
                frame("missing_newdata"),
                type="lp",
                se_fit=True,
                na_action=prediction["action"],
            )
            assert_close(result.fit, prediction["fit"])
            assert_close(result.se_fit, prediction["se_fit"])


@pytest.mark.parametrize("case", REFERENCE["grouping"], ids=lambda case: case["formula"])
def test_grouping_methods_use_evaluated_strata(case):
    fit = r.survfit(case["formula"], frame("data"))
    summary = r.summary_survfit(fit, times=[100, 300, 600], extend=True)
    assert_close(summary.surv, case["surv"])
    assert_close(summary.time, case["time"])
    assert_close(summary.n_risk, case["n_risk"])
    assert list(summary.strata) == case["strata"]
    test = r.survdiff(case["formula"].replace("~", "~ sex +"), frame("data"))
    assert_close(test.chisq, case["test_chisq"])
    assert_close(test.obs, case["test_observed"])
    assert_close(test.exp, case["test_expected"])


@pytest.mark.parametrize("case", REFERENCE["subsets"], ids=lambda case: case["formula"])
def test_strata_evaluation_precedes_subset(case):
    fit = r.coxph(case["formula"], frame("data"), subset=case["subset"])
    assert_close(r.coef(fit), case["beta"])
    assert_close(r.vcov(fit), case["variance"])
    assert list(fit.strata_levels) == case["strata_levels"]
    assert_close(r.predict(fit, type="lp"), case["lp"])


@pytest.mark.parametrize(
    "call",
    [
        "strata()",
        "strata(sex, sep = 2)",
        "strata(sex, shortlabel = 'yes')",
        "strata(sex, na.group = TRUE, na.group = FALSE)",
        "strata(offset(age))",
        "strata(ridge(age))",
        "strata(tt(age))",
    ],
)
def test_invalid_strata_options_and_specials_are_rejected(call):
    with pytest.raises(ValueError, match="strata"):
        r.coxph(f"Surv(time, status) ~ age + {call}", frame("data"))
