"""Grouped predictions agree with independent R calls before crossing Python."""

import json
import math
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/grouped_cox_prediction_reference.json").read_text()
)


@cache
def fitted(name):
    data = dict(REFERENCE["data"])
    data["cl"] = list(data["cl"])
    data["x"] = list(data["x"])
    for row in (1, 8):
        data["cl" if name == "sparse_only" else "x"][row] = None
    data["cl"] = RFactor(data["cl"], REFERENCE["levels"])
    return r.coxph(
        "Surv(futime, fustat) ~ " + REFERENCE["specs"][name],
        data,
        robust=False,
        na_action="na.exclude",
    )


def close(actual, expected, case):
    def numeric(value):
        return (
            [numeric(v) for v in value]
            if isinstance(value, list)
            else math.nan
            if value is None
            else value
        )

    reference = np.asarray(numeric(expected["values"]))
    if case["type"] != "terms" or case["model"] == "sparse_only":
        reference = reference.reshape(-1)
    np.testing.assert_allclose(actual, reference, atol=3e-8, rtol=2e-7, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("as_arrays", [False, True], ids=["lists", "arrays"])
def test_complete_grouped_predictions_match_r(case, as_arrays):
    group = [int(g) for g in case["collapse"]] if case["numeric_groups"] else case["collapse"]
    if case["levels"] is not None:
        group = RFactor(group, case["levels"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = r.predict(
            fitted(case["model"]),
            case.get("newdata"),
            type=case["type"],
            se_fit=case["se_fit"],
            terms=case["terms"],
            na_action=case.get("na_action", "na.pass"),
            collapse=group,
            _with_group_names=True,
            _as_arrays=as_arrays,
        )
    expected = case["expected"]["result"]
    assert "error" not in expected
    if case["se_fit"]:
        close(result["values"].fit, expected["fit"], case)
        close(result["values"].se_fit, expected["se_fit"], case)
        names = expected["fit"]["names"]
    else:
        assert isinstance(result["values"], np.ndarray if as_arrays else list)
        close(result["values"], expected, case)
        names = expected["names"]
    assert result["group_names"] == names


@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("kind", ["lp", "risk", "expected", "survival", "terms"])
@pytest.mark.parametrize("se_fit", [False, True])
def test_native_grouping_returns_only_observed_group_rows_and_owned_values(order, kind, se_fit):
    fit = fitted("plain").fit
    original = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 1.0], [0.4, 2.0], [0.5, 1.0], [0.6, 2.0]])
    values = (
        np.asfortranarray(original)
        if order == "F"
        else original[::2]
        if order == "strided"
        else original
    )
    groups = np.array([-900 if i % 2 else 100 for i in range(len(values))], dtype=np.int32)
    args = {"newdata": values, "se_fit": se_fit, "reference": "zero"}
    method = fit.predict_terms if kind == "terms" else fit.predict
    if kind != "terms":
        args["type"] = kind
    if kind in {"expected", "survival"}:
        args["new_time"] = np.arange(1, len(values) + 1, dtype=float) * 50
    ungrouped = method(**args)
    before = values.copy()
    result = method(**args, collapse=groups)
    full = np.asarray(ungrouped.fit)
    expected = np.stack([full[groups == g].sum(axis=0) for g in sorted(set(groups))])
    np.testing.assert_array_equal(result.fit, expected)
    assert len(result.fit) == 2
    if se_fit:
        full = np.asarray(ungrouped.se_fit)
        expected = np.stack(
            [np.sqrt((full[groups == g] ** 2).sum(axis=0)) for g in sorted(set(groups))]
        )
        np.testing.assert_array_equal(result.se_fit, expected)
    else:
        assert result.se_fit is None
    public = result.fit
    public[0] = -999 if kind != "terms" else [-999] * 2
    assert result.fit[0] != public[0]
    np.testing.assert_array_equal(values, before)


def test_native_grouped_predictions_validate_lengths_and_preserve_empty_term_widths():
    fit = fitted("plain").fit
    groups = [i % 3 for i in range(24)]
    result = fit.predict_terms(assign=[], se_fit=True, collapse=groups)
    assert result.fit == [[], [], []]
    assert result.se_fit == [[], [], []]
    with pytest.raises(ValueError, match="one value per prediction row"):
        fit.predict_terms(collapse=[1, 2])
    with pytest.raises(ValueError, match="one value per matrix row"):
        fit.predict(collapse=[1, 2])
