"""Ordinary Cox curve margin selection against independent stock-R fits."""

import json
import pickle
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_survfit_subset_reference.json").read_text()
)
CASES = REFERENCE["cases"]


def _array(spec):
    return (
        None
        if spec is None
        else np.asarray(spec["values"], dtype=float).reshape(spec["shape"], order="F")
    )


@cache
def _source(name):
    frame = pd.DataFrame(REFERENCE["tiny_data" if name == "tiny" else "data"])
    frame["g"] = pd.Categorical(frame.g, categories=["a", "b"])
    one_coefficient = name in {"tiny", "narrow"}
    formula = "Surv(time,status) ~ x" if one_coefficient else "Surv(time,status) ~ x + z"
    if name != "plain":
        formula += " + strata(g)"
    fit = r.coxph(formula, frame, init=[0.25] if one_coefficient else [0.25, -0.125], iter_max=0)
    newdata = pd.DataFrame(REFERENCE["newdata"], index=REFERENCE["newdata_rownames"])
    newdata.category = pd.Categorical(newdata.category, categories=REFERENCE["category_levels"])
    if name == "single":
        newdata = newdata.iloc[[0]]
    elif name == "narrow":
        newdata = newdata[["x"]]
    return r.survfit(fit, newdata=newdata)


def _selected(case):
    source = _source(case["source"])
    if case["curves"] is not None:
        return r._subset_cox_survfit(source, curves=case["curves"], drop=case["drop"])
    return source.subset(strata=case["strata"], data=case["data"], drop=case["drop"])


def _check_fields(result, expected):
    for name, spec in expected.items():
        actual = getattr(result, name)
        if spec is None:
            assert actual is None, name
        else:
            values = np.asarray(actual, dtype=float)
            reference = _array(spec)
            # An empty time margin is encoded by [] and its column count is
            # retained in colnames; nonempty matrices retain their full shape.
            if not values.size and not reference.size:
                continue
            np.testing.assert_allclose(
                values, reference, rtol=2e-12, atol=2e-13, equal_nan=True, err_msg=name
            )


def _check_curve(result, expected):
    _check_fields(result, expected["fields"])
    assert result.n == expected["n"]
    assert result.dim == (expected["dim"] or {})
    assert result.strata_names == (expected["strata_names"] or [])
    assert (None if result.strata is None else list(result.strata.values())) == expected[
        "strata_sizes"
    ]
    assert (result.colnames or None) == expected["colnames"]
    if expected["newdata"] is None:
        assert result.newdata is None
    elif expected["newdata_kind"] == "vector":
        assert isinstance(result.newdata, pd.Series)
        np.testing.assert_array_equal(result.newdata.tolist(), expected["newdata"])
    else:
        assert result.newdata.index.tolist() == expected["newdata_rownames"]
        for column, values in expected["newdata"].items():
            assert result.newdata[column].astype(str).tolist() == values
        assert result.newdata.category.cat.categories.tolist() == REFERENCE["category_levels"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_selected_fields_dimensions_and_metadata_match_stock(case):
    source = _source(case["source"])
    original = np.asarray(source.surv).copy()
    result = _selected(case)
    _check_curve(result, case["expected"])
    np.testing.assert_array_equal(source.surv, original)
    _check_curve(pickle.loads(pickle.dumps(result)), case["expected"])  # noqa: S301 - local data


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["derived"] is not None], ids=lambda case: case["name"]
)
def test_selected_initial_summary_quantile_and_display_labels_match_stock(case):
    result = _selected(case)
    expected = case["derived"]
    _check_curve(r.survfit0(result), expected["initial"])
    for name, options in (
        ("summary", {"censored": True}),
        ("at_times", {"times": [0, 4, 8, 12], "extend": True}),
    ):
        actual = r.summary_survfit(result, **options)
        _check_fields(actual, expected[name]["fields"])
        assert actual.strata == expected[name]["strata"]
    quantile = r.quantile_survfit(result, probs=[0.5])
    for name, values in expected["quantile"].items():
        np.testing.assert_allclose(
            np.asarray(getattr(quantile, name)).ravel(),
            np.asarray(values, dtype=float),
            rtol=1e-12,
            equal_nan=True,
        )
    if result.strata_labels is not None:
        assert (
            r.as_data_frame(result)["strata"]
            == [
                name
                for name, count in zip(result.strata_names, result.strata.values(), strict=True)
                for _ in range(count)
            ]
            * result.ncurve
        )
        labels = (
            result.strata_names
            if not result.has_data_margin
            else [
                f"{group}, {column}" for column in result.colnames for group in result.strata_names
            ]
        )
        assert quantile.strata == labels
        assert r.summary_survfit(result).table.rownames == labels


def test_subsequent_selection_uses_original_repeated_labels_and_one_shot_indices():
    source = _source("stratified")
    selected = source.subset(strata=[1, 0, 1], data=[2, 0], drop=False)
    by_name = selected.subset(strata=["b"], data=iter([1, 0]), drop=False)
    direct = source.subset(strata=[1], data=[0, 2], drop=False)
    np.testing.assert_array_equal(by_name.surv, direct.surv)
    assert by_name.strata_names == direct.strata_names == ["b"]
    assert by_name.newdata.equals(direct.newdata)
    assert source.subset() is source


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"strata": [2]}, "subscript"),
        ({"strata": ["missing"]}, "not matched"),
        ({"data": [-1]}, "subscript"),
        ({"data": [3]}, "subscript"),
        ({"drop": None}, "drop must be"),
    ],
)
def test_invalid_margin_positions_fail_before_changing_source(options, error):
    source = _source("stratified")
    original = np.asarray(source.surv).copy()
    with pytest.raises((ValueError, IndexError), match=error):
        source.subset(**options)
    np.testing.assert_array_equal(source.surv, original)
