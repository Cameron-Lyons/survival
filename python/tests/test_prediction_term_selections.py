"""Term subscripts, values, errors and prediction labels match independent R calls."""

import importlib
import json
import re
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
bridge = importlib.import_module("survival.r_api")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/prediction_term_selection_reference.json").read_text()
)


@cache
def fitted(name):
    data = {**REFERENCE["data"], "cl": RFactor(REFERENCE["data"]["cl"], REFERENCE["levels"])}
    kind, rhs = REFERENCE["specs"][name]
    return getattr(r, kind)("Surv(futime,fustat)~" + rhs, data, na_action="na.exclude")


def compare(actual, expected, *, vector=False):
    values = np.asarray(expected["values"], dtype=float).reshape(expected["dim"])
    if vector:
        values = values[:, 0]
    np.testing.assert_allclose(actual, values, rtol=3e-7, atol=3e-8, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_term_prediction_selectors_values_errors_and_labels_match_r(case):
    fit = fitted(case["model"])
    newdata = {
        **REFERENCE["newdata"],
        "cl": RFactor(REFERENCE["newdata"]["cl"], REFERENCE["levels"]),
    }
    options = {"type": case.get("type", "terms"), "se_fit": case["se_fit"]}
    if case["selection"] != "default":
        options["terms"] = bridge._r_term_subscript(case["terms"], case["kind"], case["levels"])
    expected = case["expected"]["result"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises((ValueError, TypeError), match=re.escape(expected["error"])):
                r.predict(fit, newdata, **options)
        else:
            actual = r.predict(fit, newdata, **options)
            vector = options["type"] in {"lp", "risk", "expected", "survival"}
            if case["se_fit"]:
                compare(actual.fit, expected["fit"], vector=vector)
                compare(actual.se_fit, expected["se_fit"], vector=vector)
                columns = expected["fit"]["columns"]
            else:
                compare(actual, expected, vector=vector)
                columns = expected["columns"]
            if options["type"] == "terms":
                assert bridge._prediction_term_names(fit, options.get("terms")) == columns
    assert [str(w.message) for w in caught] == case["expected"]["warnings"]


@pytest.mark.parametrize(
    "selection", [np.float64(1), np.array(1.0), np.array([True, False]), np.array([-1, 0])]
)
def test_numpy_term_selectors_match_r_vector_semantics(selection):
    fit = fitted("aft_plain")
    actual = r.predict(fit, type="terms", terms=selection, se_fit=True)
    full = r.predict(fit, type="terms", se_fit=True)
    chosen = [0] if np.asarray(selection).dtype.kind in "fb" else [1]
    for field in ("fit", "se_fit"):
        np.testing.assert_array_equal(
            getattr(actual, field), np.asarray(getattr(full, field))[:, chosen]
        )


def test_full_formula_labels_remain_distinct_from_prediction_columns():
    for name in ("aft_strata_first", "cox_strata_first"):
        fit = fitted(name)
        assert r.model_term_names(fit) == ["strata(cl)", "age", "rx"]
        assert bridge._prediction_term_names(fit) == ["age", "rx"]
