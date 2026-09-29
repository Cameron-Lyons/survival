"""Standalone P-spline prediction against R survival 3.8-12."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/pspline_prediction_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_pspline_prediction_keeps_training_basis_and_matches_r(case):
    options = dict(case["options"])
    if "Boundary.knots" in options:
        options["boundary_knots"] = options.pop("Boundary.knots")
    original = r.pspline(REFERENCE["x"], **options)
    assert original.dmat == case["dmat"]
    assert case["omitted_identical"]
    for function in (r.predict_pspline, r.predict, survival.predict):
        assert function(original) is original
        predicted = function(original, REFERENCE["newx"])
        assert isinstance(predicted, r.PsplineResult)
        assert predicted.penalty is False
        assert predicted.knots == original.knots
        assert predicted.n_cols == case["ncol"]
        assert predicted.nterm == case["nterm"]
        assert predicted.degree == case["degree"]
        assert predicted.intercept == case["intercept"]
        assert list(predicted.boundary_knots) == case["boundary_knots"]
        assert predicted.combine == case["combine"]
        assert predicted.dmat == case["dmat"]
        np.testing.assert_allclose(
            predicted.basis,
            np.asarray(case["basis"], dtype=float),
            atol=2e-14,
            rtol=2e-13,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            function(original, 0).basis, case["scalar"], atol=2e-14, rtol=2e-13
        )
    assert original.penalty == options.get("penalty", True)


def test_pspline_prediction_keyword_aliases_and_ignored_r_options():
    original = r.pspline(REFERENCE["x"])
    values = [-4, 0, 5]
    expected = r.predict_pspline(original, values)
    assert r.predict(original, newx=values) == expected
    assert r.predict(original, newdata=np.asarray(values)) == expected
    assert r.predict_pspline(original, values, degree=1, penalty=True) == expected
    assert r.predict(original, values, degree=1, penalty=True) == expected
    with pytest.raises(ValueError, match="use only one of newdata or newx"):
        r.predict(original, values, newx=values)


@pytest.mark.parametrize(
    ("newx", "message"),
    [
        (None, "x is required"),
        ([], "at least one non-missing"),
        ([math.nan, math.nan], "at least one non-missing"),
    ],
)
def test_pspline_prediction_refuses_empty_or_entirely_missing_inputs(newx, message):
    original = r.pspline(REFERENCE["x"])
    with pytest.raises(ValueError, match=message):
        r.predict_pspline(original, newx)
    with pytest.raises(ValueError, match=message):
        r.predict(original, newx=newx)


def test_pspline_prediction_preserves_r_restrictions():
    small = r.pspline(REFERENCE["x"], df=2, nterm=3)
    assert r.predict(small) is small
    with pytest.raises(ValueError, match="nterm' too small for df=4"):
        r.predict(small, [0])
    constant = r.pspline([2] * 5)
    with pytest.raises(ValueError, match=REFERENCE["errors"]["constant_boundary"]):
        r.predict(constant, [2])
    with pytest.raises(ValueError, match="infinite"):
        r.predict(r.pspline(REFERENCE["x"]), [math.inf])
    with pytest.raises(TypeError, match="requires a PsplineResult"):
        r.predict_pspline([1, 2, 3])
