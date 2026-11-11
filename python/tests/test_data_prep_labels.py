"""Identifier coercion and factor metadata against independently generated stock R data.

Regenerate with ``Rscript scripts/generate_data_prep_labels_reference.R``. The
fixture records both stock outputs and stock errors for supported nostutter edges.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
Factor = survival.Factor
_as_character = importlib.import_module("survival.r._coerce")._as_character
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "data_prep_labels_reference.json").read_text()
)


def _column(spec, storage):
    values = spec["values"]
    if spec["kind"] == "double":
        values = [None if value is None else float(value) for value in values]
    if "nan" in spec:
        values = [
            float("nan") if nan else value for value, nan in zip(values, spec["nan"], strict=True)
        ]
    if spec["kind"] == "factor":
        if storage == "pandas":
            pd = pytest.importorskip("pandas")
            return pd.Series(
                pd.Categorical(values, categories=spec["levels"], ordered=spec["ordered"])
            )
        return Factor(values, spec["levels"], ordered=spec["ordered"])
    if storage == "numpy":
        return np.asarray(values, dtype=object if None in values else None)
    if storage == "pandas":
        # A float Series would erase the NA/NaN distinction before the call.
        return pytest.importorskip("pandas").Series(
            values, dtype=object if None in values else None
        )
    if storage == "iterator":
        return iter(values)
    return list(values)


@pytest.mark.parametrize("storage", ["list", "numpy", "pandas", "iterator"])
@pytest.mark.parametrize("case", REFERENCE["neardate"], ids=lambda case: case["name"])
def test_neardate_identifier_types_match_stock_r(case, storage):
    args = (_column(case["id1"], storage), _column(case["id2"], storage), case["y1"], case["y2"])
    if isinstance(case["expected"], dict):
        with pytest.raises(ValueError, match=case["expected"]["error"]):
            r.neardate(*args, best=case["best"], nomatch=case["nomatch"])
    else:
        assert r.neardate(*args, best=case["best"], nomatch=case["nomatch"]) == case["expected"]


_MIXED_IDENTIFIER_INPUTS = {
    "logical_integer_character": ([True, 2], ["1", "2"]),
    "character_logical_integer": (["1", "2"], [True, 2]),
    "logical_double_character": ([True, 2.5], ["1", "2.5"]),
    "character_logical_double": (["1", "2.5"], [True, 2.5]),
    "logical_nan_character": ([True, float("nan")], ["1", "NaN"]),
    "character_logical_nan": (["1", "NaN"], [True, float("nan")]),
    "logical_na_character": ([True, None], ["TRUE", None]),
    "character_logical_na": (["TRUE", None], [True, None]),
    "mixed_character_labels": ([True, 1.5, "a"], ["TRUE", "1.5", "a"]),
    "character_mixed_labels": (["TRUE", "1.5", "a"], [True, 1.5, "a"]),
    "integer_double_character": ([1000000000, 0.1], ["1e+09", "0.1"]),
    "character_integer_double": (["1e+09", "0.1"], [1000000000, 0.1]),
}


@pytest.mark.parametrize("storage", ["list", "numpy", "pandas", "iterator"])
@pytest.mark.parametrize(
    "case",
    [
        case
        for case in REFERENCE["neardate"]
        if case["name"].rsplit("_", 2)[0] in _MIXED_IDENTIFIER_INPUTS
    ],
    ids=lambda case: case["name"],
)
def test_neardate_promotes_each_mixed_vector_before_cross_type_matching(case, storage):
    # The R expressions in the generator have already promoted these scalars by
    # the time JSON records them. Preserve the original heterogeneous Python
    # inputs here so the facade, rather than NumPy/pandas, performs promotion.
    inputs = _MIXED_IDENTIFIER_INPUTS[case["name"].rsplit("_", 2)[0]]
    if storage == "numpy":
        id1, id2 = (np.asarray(values, dtype=object) for values in inputs)
    elif storage == "pandas":
        pd = pytest.importorskip("pandas")
        id1, id2 = (pd.Series(values, dtype=object) for values in inputs)
    elif storage == "iterator":
        id1, id2 = (iter(values) for values in inputs)
    else:
        id1, id2 = (list(values) for values in inputs)
    assert (
        r.neardate(id1, id2, case["y1"], case["y2"], best=case["best"], nomatch=case["nomatch"])
        == case["expected"]
    )


def _check_factor_and_response(actual, expected):
    assert isinstance(actual, Factor)
    assert [None if value is None else _as_character(value) for value in actual] == expected[
        "values"
    ]
    assert [_as_character(value) for value in actual.categories] == expected["levels"]
    assert actual.ordered is expected["ordered"]
    response = r.Surv(range(1, len(actual) + 1), actual)
    assert response.type == expected["response"]["type"]
    assert list(response.states) == expected["response"]["states"]
    assert response.as_matrix() == expected["response"]["matrix"]


@pytest.mark.parametrize("storage", ["list", "pandas"])
@pytest.mark.parametrize("case", REFERENCE["lvcf"], ids=lambda case: case["name"])
def test_lvcf_preserves_factor_levels_and_surv_interpretation(case, storage):
    values = _column(case["x"], storage)
    original = list(values)
    actual = r.lvcf(case["id"], values, time=case["time"])
    _check_factor_and_response(actual, case["expected"])
    assert list(values) == original


@pytest.mark.parametrize("storage", ["list", "numpy", "pandas", "iterator"])
@pytest.mark.parametrize("case", REFERENCE["nostutter"], ids=lambda case: case["name"])
def test_nostutter_factor_levels_and_surv_interpretation(case, storage):
    actual = r.nostutter(
        case["id"], _column(case["x"], storage), censor=case["censor"], single=case["single"]
    )
    _check_factor_and_response(actual, case["expected"])
    if case["stock_error"] is not None:
        # These existing successful Python cases are deliberate fixes of stock R
        # errors. The generator constructs the intended factor explicitly in R.
        assert case["name"] in {"no_repeats", "unused_levels", "single", "empty"}


def test_factor_labels_retain_python_numeric_payloads():
    carried = r.lvcf([1, 1, 1], Factor([None, 1, None], [0, 1, 2], ordered=True))
    assert isinstance(carried, Factor)
    assert carried == [None, 1, 1]
    assert carried.levels == [0, 1, 2]
    assert carried.ordered is False
    result = r.nostutter([1] * 4, [0, 1, 1, 2])
    assert result == [0, 1, 0, 2]
    assert result.levels == [0, 1, 2]
    assert all(type(value) is int for value in result)


def test_neardate_integer_arrays_keep_native_fast_path(monkeypatch):
    module = importlib.import_module("survival.r._data_prep")

    def unexpected_materialization(*args):
        raise AssertionError("integer ndarray identifiers should remain arrays")

    monkeypatch.setattr(module, "_materialize_labels", unexpected_materialization)
    assert r.neardate(np.array([1, 2, 1]), np.array([1, 2]), [1, 1, 3], [2, 2]) == [0, 1, None]


@pytest.mark.parametrize("dtype", [float, np.float32, np.float64, np.longdouble])
def test_neardate_numeric_na_nan_distinction_changes_actual_matches(dtype):
    # Stock R: neardate(c(NA_real_, NaN), NA_real_, c(2, 0), 1, nomatch=-1).
    query = [None, dtype("nan")]
    assert r.neardate(query, [None], [2, 0], [1], nomatch=-1) == [-1, -1]
    assert r.neardate(query, [None], [2, 0], [1], best="prior", nomatch=-1) == [None, None]


def test_neardate_masked_and_nullable_missing_markers_remain_na():
    pd = pytest.importorskip("pandas")
    masked = np.ma.array([0.0, np.nan], mask=[True, False])
    assert r.neardate(masked, [None], [2, 0], [1], nomatch=-1) == [-1, -1]
    # Nullable arrays have already converted both input markers to R-style NA.
    nullable = pd.array([pd.NA, np.nan], dtype="Float64")
    assert r.neardate(nullable, [None], [2, 0], [1], nomatch=-1) == [-1, None]


@pytest.mark.parametrize("dtype", ["str", "string"])
def test_neardate_declared_character_missing_is_not_the_literal_nan_label(dtype):
    pd = pytest.importorskip("pandas")
    values = pd.Series([np.nan, "NaN"], dtype=dtype)
    if getattr(values.dtype, "type", None) is not str:
        pytest.skip("this pandas version does not provide a declared string dtype")
    # Stock R character NA stays missing; the literal "NaN" remains a label.
    assert r.neardate(values, ["NaN"], [2, 0], [1], nomatch=-1) == [None, 0]
    assert r.neardate(values, ["NaN"], [2, 0], [1], best="prior", nomatch=-1) == [-1, -1]
