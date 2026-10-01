"""Grouped model outputs against complete independent R calls."""

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
    (Path(__file__).parent / "fixtures/model_collapse_reference.json").read_text()
)


@cache
def fitted(name):
    spec = REFERENCE["specs"][name]
    data = dict(REFERENCE["data"])
    data["cl"] = RFactor(data["cl"], REFERENCE["levels"])
    if spec.get("excluded"):
        data["x"] = list(data["x"])
        for row in (1, 8):
            data["x"][row] = None
    args = {"na_action": "na.exclude" if spec.get("excluded") else "na.omit"}
    if spec.get("weights"):
        args["weights"] = "w"
    if spec.get("id"):
        args["id"] = "cl"
    if spec.get("subset"):
        args["subset"] = [i for i in range(26) if (i + 1) % 3]
    formula = "Surv(futime, fustat) ~ " + spec["rhs"]
    return (
        r.survreg(formula, data, **args)
        if spec.get("model") == "survreg"
        else r.coxph(formula, data, robust=name == "cluster", **args)
    )


def close(actual, expected):
    def numeric(value):
        return (
            [numeric(v) for v in value]
            if isinstance(value, list)
            else math.nan
            if value is None
            else value
        )

    np.testing.assert_allclose(actual, numeric(expected), atol=3e-8, rtol=2e-7, equal_nan=True)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_grouped_predictions_and_residuals_match_r(case):
    fit = fitted(case["fit"])
    group = [math.nan if isinstance(value, dict) else value for value in case["collapse"]]
    if case["levels"] is not None:
        group = RFactor(group, case["levels"])
    args = {"collapse": group, "type": case["type"]}
    if case["kind"] == "predict":
        args["se_fit"] = True
        args["newdata"] = case.get("newdata")
        args["na_action"] = case.get("na_action", "na.pass")
    method = r.predict if case["kind"] == "predict" else r.residuals
    expected = case["expected"]["result"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(ValueError, match="missing"):
                method(fit, **args)
            return
        result = method(fit, **args, _with_group_names=True)
    values = result["values"]
    if case["kind"] == "predict":
        close(values.fit, expected["fit"]["values"])
        close(values.se_fit, expected["se_fit"]["values"])
        names = expected["fit"]["names"]
    else:
        reference_values = expected["values"]
        if (
            case["type"] == "partial"
            and fit.nvar == 1
            and not isinstance(reference_values[0], list)
        ):
            # The Python partial-residual API retains its matrix shape; R drops
            # a single column after rowsum. Compare that same column explicitly.
            reference_values = [[value] for value in reference_values]
        close(values, reference_values)
        names = expected["names"]
    if names is not None:
        assert result["group_names"] == names
    missing_group = any(value is None or isinstance(value, dict) for value in case["collapse"])
    if missing_group and case.get("na_action", "na.pass") != "na.omit":
        assert any("missing values for 'group'" in str(w.message) for w in caught)


def test_cluster_and_id_factor_order_survive_subset_and_serialization():
    import pickle
    from dataclasses import replace

    data = dict(REFERENCE["data"])
    groups = RFactor(data["cl"], REFERENCE["levels"])
    for key in ("cluster", "id"):
        fit = r.coxph("Surv(futime,fustat)~x+rx", data, **{key: groups}, subset=list(range(20)))
        restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 -- locally created bytes
        explicit = r.residuals(
            restored, type="dfbeta", collapse=RFactor(data["cl"][:20], REFERENCE["levels"])
        )
        close(r.residuals(restored, type="dfbeta", collapse=True), explicit)
        old = (
            replace(restored, id_levels=None)
            if key == "id"
            else replace(restored, cluster_levels=None)
        )
        assert len(r.residuals(old, type="dfbeta", collapse=True)) == 3


@pytest.mark.parametrize("kind", ["schoenfeld", "scaledsch"])
def test_schoenfeld_ignores_collapse_even_with_invalid_lengths(kind):
    fit = fitted("plain")
    plain = r.residuals(fit, type=kind)
    for collapse in (True, [1, 2], [[1]]):
        result = r.residuals(fit, type=kind, collapse=collapse)
        close(result.values, plain.values)
        assert result.time == plain.time
        assert result.colnames == plain.colnames


def test_collapse_without_available_groups_fails_and_bad_labels_are_rejected():
    with pytest.raises(ValueError, match="no cluster or id"):
        r.residuals(fitted("plain"), collapse=True)
    with pytest.raises(ValueError, match="Wrong length"):
        r.residuals(fitted("plain"), collapse=[1, 2])
    with pytest.raises(ValueError, match="one-dimensional"):
        r.predict(fitted("plain"), collapse=[[1]] * 26)


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_native_grouped_sum_is_owned_and_accepts_matrix_layouts(order):
    original = np.arange(80.0, dtype=float).reshape(20, 4)
    values = (
        np.asfortranarray(original)
        if order == "F"
        else original[::2]
        if order == "strided"
        else original
    )
    groups = np.arange(len(values), dtype=np.int32) % 3 - 2
    before = values.copy()
    result = survival.core.grouped_sum(values, groups)
    expected = np.stack([values[groups == g].sum(axis=0) for g in sorted(set(groups))])
    np.testing.assert_array_equal(result, expected)
    result[:] = -999
    np.testing.assert_array_equal(values, before)
    root = survival.core.grouped_sum(values, groups, squares=True)
    np.testing.assert_allclose(
        root,
        np.stack([np.sqrt((values[groups == g] ** 2).sum(axis=0)) for g in sorted(set(groups))]),
    )


def test_native_grouped_sum_handles_empty_dimensions_and_rejects_wrong_group_length():
    assert survival.core.grouped_sum(np.empty((0, 3)), []).shape == (0, 3)
    assert survival.core.grouped_sum(np.empty((3, 0)), [1, 2, 1]).shape == (2, 0)
    with pytest.raises(ValueError, match="one value per matrix row"):
        survival.core.grouped_sum([[1.0, 2.0]], [])


def test_empty_newdata_and_all_omitted_collapse_return_empty_values():
    fit = fitted("plain")
    empty = {key: [] for key in REFERENCE["data"]}
    for kind in ("lp", "terms", "expected"):
        result = r.predict(fit, empty, type=kind, se_fit=True, collapse=[], _with_group_names=True)
        assert result["values"].fit == []
        assert result["values"].se_fit == []
        assert result["group_names"] == []
    newdata = {key: values[:3] for key, values in REFERENCE["data"].items()}
    result = r.predict(
        fit, newdata, collapse=[None] * 3, na_action="na.omit", _with_group_names=True
    )
    assert result == {"values": [], "group_names": []}
