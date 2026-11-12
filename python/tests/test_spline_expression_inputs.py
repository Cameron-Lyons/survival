"""Composed basis arguments agree with stock R fits and retained predictions."""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/spline_expression_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("layout", ["list", "array", "iterator"])
def test_composed_basis_inputs_match_stock_r(case, layout):
    source = REFERENCE["missing_data" if case["missing"] else "data"]
    data = {
        key: iter(values)
        if layout == "iterator"
        else np.asarray(values, dtype=float)
        if layout == "array"
        else copy.copy(values)
        for key, values in source.items()
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fit = getattr(r, case["model"])(
            case["formula"], data, subset=case["subset"], na_action="exclude", x=True, model=True
        )
        prediction = r.predict(fit, REFERENCE["newdata"], type="lp", se_fit=True, na_action="pass")
        newmatrix = r.model_matrix(fit, REFERENCE["newdata"])
    np.testing.assert_allclose(r.coef(fit), case["coefficients"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(r.vcov(fit), case["variance"], rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(fit.loglik, case["loglik"], rtol=1e-10, atol=1e-10)
    for actual, expected in [(r.model_matrix(fit), case["matrix"]), (newmatrix, case["newmatrix"])]:
        np.testing.assert_allclose(actual["data"], expected["data"], rtol=1e-11, atol=1e-12)
        assert actual["columns"] == expected["columns"]
        assert actual["assign"] == expected["assign"]
    np.testing.assert_allclose(
        prediction.fit, np.asarray(case["predict"], dtype=float), rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        prediction.se_fit, np.asarray(case["se"], dtype=float), rtol=1e-7, atol=1e-10
    )
    assert ([] if fit.na_action is None else list(fit.na_action.rows)) == case["omitted"]
