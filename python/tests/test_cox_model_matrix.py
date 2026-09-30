"""Full matrices against R calls and explicitly marked independent references."""

import json
import pickle
import re
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_model_matrix_reference.json").read_text()
)


def frame(data, levels):
    return {**data, "small": RFactor(data["small"], levels)}


@cache
def fitted(model, formula):
    data = frame(REFERENCE["data"], REFERENCE["levels"])
    if model == "coxph" and "cluster(" in formula:
        with pytest.warns(RuntimeWarning, match="cluster specified with robust=FALSE"):
            return r.coxph(formula, data, robust=False)
    return r.coxph(formula, data, robust=False) if model == "coxph" else r.survreg(formula, data)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_complete_model_matrices_match_stock_r(case):
    fit = fitted(case["model"], case["formula"])
    data = case["newdata"]
    if data is not None:
        data = frame(data, case["new_levels"])
    expected = case["expected"]
    if "error" in expected:
        # Ordinary factor validation uses a Python-oriented message; the error
        # still precedes matrix construction, as stock R does.
        match = (
            "unknown level|new levels"
            if "new level" in expected["error"]
            else re.escape(expected["error"])
        )
        with pytest.raises(ValueError, match=match):
            r.model_matrix(fit, data)
        return
    result = r.model_matrix(fit, data)
    np.testing.assert_allclose(result["data"], expected["data"], rtol=1e-12, atol=2e-14)
    assert result["columns"] == expected["columns"]
    assert result["assign"] == expected["assign"]
    if data is not None and case["model"] == "coxph":
        assert result["strata"] == expected["strata"]


@pytest.mark.parametrize("position", ["first", "middle", "last", "only"])
def test_sparse_snapshot_is_owned_and_pickle_retains_formula_order(position):
    terms = {
        "first": "frailty(group, sparse=TRUE, theta=.4) + age + small",
        "middle": "age + frailty(group, sparse=TRUE, theta=.4) + small",
        "last": "age + small + frailty(group, sparse=TRUE, theta=.4)",
        "only": "frailty(group, sparse=TRUE, theta=.4)",
    }
    fit = r.coxph(
        "Surv(time,status) ~ " + terms[position], frame(REFERENCE["data"], REFERENCE["levels"])
    )
    coefficients, dense, lp = fit.coefficients, fit.x, r.predict(fit)
    matrix = r.model_matrix(fit)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 -- locally created bytes
    assert r.model_matrix(restored) == matrix
    col = next(i for i, name in enumerate(matrix["columns"]) if name.startswith("frailty("))
    assert [row[col] for row in matrix["data"]] == [
        float(i + 1) for i in range(12) for _ in range(2)
    ]
    matrix["data"][0][col] = -999
    assert fit.x == dense
    assert fit.coefficients == coefficients
    assert r.predict(fit) == lp
    assert r.model_matrix(fit)["data"][0][col] == 1
    # Old fitted objects that did not retain the raw sparse column still work.
    restored = replace(restored, _sparse_values=None)
    assert r.model_matrix(restored)["data"] == r.model_matrix(fit)["data"]


def test_newdata_group_codes_precede_omission_and_prediction_remains_dense():
    fit = r.coxph(
        "Surv(time,status) ~ age + frailty(group,sparse=TRUE,theta=.4)", REFERENCE["data"]
    )
    new = {"age": [40, None, 60], "group": [112, 103, 112]}
    assert r.model_matrix(fit, new)["data"] == [[40.0, 2.0], [60.0, 2.0]]
    plain = r.model_matrix(fit, {"age": [40, 60], "group": [112, 112]})
    assert plain["data"] == [[40.0, 1.0], [60.0, 1.0]]
    before = r.predict(fit, {"age": [40, 60], "group": [112, 112]}, reference="zero")
    r.model_matrix(fit, {"age": [40, 60], "group": [999, 888]})
    assert r.predict(fit, {"age": [40, 60], "group": [112, 112]}, reference="zero") == before
    with pytest.raises((KeyError, ValueError), match="group"):
        r.model_matrix(fit, {"age": [40, 60]})
