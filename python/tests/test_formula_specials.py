"""Formula offsets, cluster expansions, and retained variables against R."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
pytestmark = pytest.mark.filterwarnings(r"ignore:NaNs produced in log\(transformed\):UserWarning")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/formula_special_reference.json").read_text()
)


def frame(name):
    data = dict(REFERENCE[name])
    data["label"] = RFactor(data["label"], REFERENCE["label_levels"])
    return data


def assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-7, atol=2e-9
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_formula_specials_match_r(case):
    fit = getattr(r, case["kind"])(case["formula"], frame("data"), model=True)
    assert r.coef_names(fit) == case["beta_names"]
    assert_close(r.coef(fit), case["beta"])
    assert_close(r.vcov(fit), case["variance"])
    matrix = r.model_matrix(fit)
    assert matrix["assign"] == case["assign"]
    assert len(matrix["data"]) == case["n"]
    assert_close(matrix["data"][:8], case["x_head"])
    assert list(fit.na_action.rows if fit.na_action else ()) == case["omitted"]
    assert_close(r.predict(fit, type="lp"), case["train_prediction"])
    predicted = r.predict(fit, frame("newdata"), type="lp", se_fit=True)
    assert_close(predicted.fit, case["intended_prediction"])
    assert_close(predicted.se_fit, case["prediction_se"])
    terms = r.predict(fit, frame("newdata"), type="terms")
    if isinstance(case["terms"], dict):
        # R indexes nonexistent term columns in intercept/offset-only models.
        assert not any(matrix["assign"])
        assert terms == [[] for _ in REFERENCE["newdata"]["age"]]
    else:
        assert_close(terms, case["terms"])
    if case["kind"] == "survreg":
        assert_close(fit.scale, case["scale"])
    # The Python model frame keeps source columns rather than R call labels.
    for column in ("unused", "transformed"):
        expected = any(column in name for name in case["retained_columns"])
        assert (column in r.model_frame(fit)) == expected


@pytest.mark.parametrize("case", REFERENCE["errors"], ids=lambda case: case["formula"])
@pytest.mark.parametrize("fitter", [r.coxph, r.survreg])
def test_cluster_interactions_reject_the_same_invalid_formulas(case, fitter):
    with pytest.raises(ValueError, match=re.escape(case["error"])):
        fitter(case["formula"], frame("data"))


@pytest.mark.parametrize("fitter", [r.coxph, r.survreg])
@pytest.mark.parametrize("removed", ["unused", "log(transformed)"])
def test_unused_prediction_variables_obey_na_action(fitter, removed):
    fit = fitter(f"Surv(time,status) ~ age + {removed} - {removed}", frame("data"))
    complete = frame("newdata")
    complete["unused"] = [1, 2, 3, 4]
    expected = r.predict(fit, complete, type="lp", se_fit=True)
    incomplete = {**complete, "unused": [1, None, 3, 4], "transformed": [2, -1, 2, 2]}
    passed = r.predict(fit, incomplete, type="lp", se_fit=True)
    assert_close(passed.fit, expected.fit)
    assert_close(passed.se_fit, expected.se_fit)
    omitted = r.predict(fit, incomplete, type="lp", na_action="na.omit")
    assert_close(omitted, np.asarray(expected.fit)[[0, 2, 3]])
    excluded = r.predict(fit, incomplete, type="lp", na_action="na.exclude")
    assert_close(excluded, [expected.fit[0], None, *expected.fit[2:]])
    with pytest.raises(ValueError, match="missing values"):
        r.predict(fit, incomplete, type="lp", na_action="na.fail")
    missing_column = dict(complete)
    del missing_column["unused" if removed == "unused" else "transformed"]
    with pytest.raises(KeyError, match="column"):
        r.predict(fit, missing_column, type="lp")
