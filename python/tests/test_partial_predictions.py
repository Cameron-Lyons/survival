"""Missing covariates propagate to dependent predictions and standard errors."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/partial_prediction_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_partial_predictions_match_r(case):
    fit = getattr(r, case["kind"])(case["formula"], REFERENCE["data"])
    for prediction in case["results"]:
        if "r_error" in prediction["raw"]:
            assert prediction["type"] == "survival"
            assert case["name"] in {"coxph_ridge", "coxph_pspline"}
        actual = r.predict(fit, REFERENCE["newdata"], type=prediction["type"], se_fit=True)
        for attr in ("fit", "se_fit"):
            np.testing.assert_allclose(
                np.asarray(getattr(actual, attr), dtype=float),
                np.asarray(prediction[attr], dtype=float),
                rtol=3e-7,
                atol=3e-9,
                err_msg=f"{case['name']} {prediction['type']} {attr}",
            )


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_selected_terms_keep_their_own_missing_values(kind):
    case = next(case for case in REFERENCE["cases"] if case["name"] == f"{kind}_additive")
    expected = next(result for result in case["results"] if result["type"] == "terms")
    fit = getattr(r, kind)(case["formula"], REFERENCE["data"])
    result = r.predict(fit, REFERENCE["newdata"], type="terms", terms="sex", se_fit=True)
    np.testing.assert_allclose(result.fit, np.asarray(expected["fit"], dtype=float)[:, 1:2])
    np.testing.assert_allclose(result.se_fit, np.asarray(expected["se_fit"], dtype=float)[:, 1:2])


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("penalty", ["ridge", "pspline"])
def test_all_missing_penalty_variable_preserves_other_terms(kind, penalty):
    case = next(case for case in REFERENCE["cases"] if case["name"] == f"{kind}_{penalty}")
    expected = next(result for result in case["results"] if result["type"] == "terms")
    fit = getattr(r, kind)(case["formula"], REFERENCE["data"])
    newdata = {**REFERENCE["newdata"], "age": [None] * 4}
    result = r.predict(fit, newdata, type="terms", se_fit=True)
    for field in ("fit", "se_fit"):
        actual = np.asarray(getattr(result, field))
        assert np.isnan(actual[:, 0]).all()
        np.testing.assert_allclose(actual[:, 1], np.asarray(expected[field], dtype=float)[:, 1])
