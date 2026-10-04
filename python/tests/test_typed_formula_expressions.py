"""Typed vector evaluation and whole fitted outputs from unmodified stock R."""

import importlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from survival import r

from .r_fixture_support import RFactor

formula = importlib.import_module("survival.r._formula")
expression = importlib.import_module("survival.r._expression")
bridge = importlib.import_module("survival.pybridge")
coerce = importlib.import_module("survival.r._coerce")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/typed_formula_expression_reference.json").read_text(
        encoding="utf-8"
    )
)


def decoded(value):
    if value is None:
        return coerce._NA_REAL
    return (
        {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}.get(value, value)
        if isinstance(value, str)
        else value
    )


def numbers(encoded):
    values = np.asarray([decoded(value) for value in encoded["values"]], dtype=float)
    return values.reshape(encoded["dim"] or [len(values)], order="F")


def messages(caught):
    return [
        re.sub(
            r"\s+",
            "",
            re.sub(r" in (?:log|sqrt)\(.*\)$", "", str(item)).replace("‘", "'").replace("’", "'"),
        )
        for item in caught
    ]


def compare_values(actual, encoded, *, rtol=3e-7, atol=3e-9):
    expected = numbers(encoded)
    values = np.asarray(actual, dtype=float).reshape(expected.shape)
    np.testing.assert_array_equal(
        np.isnan(values), np.asarray(encoded["na"]).reshape(expected.shape, order="F")
    )
    np.testing.assert_allclose(values, expected, rtol=rtol, atol=atol, equal_nan=True)


def compare_matrix(actual, encoded):
    compare_values(actual["data"], encoded, rtol=2e-13, atol=2e-13)
    assert actual["columns"] == (encoded["columns"] or [])
    assert actual["assign"] == encoded["assign"]
    assert actual["row_names"] == (encoded["rows"] or [])
    expected = encoded["contrasts"]
    if expected is None:
        assert actual["contrasts"] is None
    else:
        assert list(actual["contrasts"]) == list(expected)
        for name, value in expected.items():
            assert actual["contrasts"][name] == value["values"][0]


def raw_columns(values, levels, storage):
    data = {name: list(column) for name, column in values.items()}
    data["g"] = RFactor(data["g"], levels)
    for name in ("a", "b"):
        data[name] = (
            np.asarray(data[name], dtype=bool)
            if storage == "numpy" and not any(value is None for value in data[name])
            else expression._ExpressionVector(data[name], "logical")
        )
    for name in data:
        if name in {"a", "b", "g"}:
            continue
        if name == "h":
            data[name] = expression._ExpressionVector(data[name], "character")
            continue
        if name == "i":
            data[name] = (
                np.asarray(data[name], dtype=np.int32)
                if storage == "numpy" and not any(value is None for value in data[name])
                else expression._ExpressionVector(
                    [decoded(value) for value in data[name]], "numeric", storage="integer"
                )
            )
            continue
        numeric = [decoded(value) for value in data[name]]
        data[name] = np.asarray(numeric, dtype=float) if storage == "numpy" else numeric
    return data


def training_data(case, storage):
    columns = raw_columns(REFERENCE["data"], REFERENCE["levels"], storage)
    if case["variant"] == "complete":
        n = len(REFERENCE["row_names"])
        columns["a"] = [i % 2 == 0 for i in range(n)]
        columns["b"] = [i % 4 < 2 for i in range(n)]
    return bridge._r_data_frame(columns, len(REFERENCE["row_names"]), REFERENCE["row_names"])


def new_data(name, storage):
    source = REFERENCE["newdata"][name]
    return bridge._r_data_frame(
        raw_columns(source["data"], REFERENCE["levels"], storage),
        len(source["row_names"]),
        source["row_names"],
    )


@pytest.mark.parametrize("case", REFERENCE["evaluation"], ids=lambda case: case["name"])
@pytest.mark.parametrize("storage", ["lists", "numpy"])
def test_typed_expression_values_types_and_warnings_match_r(case, storage):
    data = raw_columns(REFERENCE["truth"], REFERENCE["truth_levels"], storage)
    if case["variant"] == "empty":
        data = {
            name: RFactor([], values.categories)
            if isinstance(values, RFactor)
            else expression._ExpressionVector(
                [],
                "logical" if name in {"a", "b"} else "character" if name == "h" else "numeric",
                storage="integer" if name == "i" else None,
            )
            for name, values in data.items()
        }
    if case["variant"] == "all_missing":
        for name, values in data.items():
            data[name] = (
                RFactor([None] * len(values), values.categories)
                if isinstance(values, RFactor)
                else expression._ExpressionVector(
                    [None] * len(values),
                    "logical" if name in {"a", "b"} else "character" if name == "h" else "numeric",
                    storage="integer" if name == "i" else None,
                )
            )
    n = len(data["a"])
    expected = case["expected"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = formula._expression_values(data, case["expression"], n)
    assert messages([item.message for item in caught]) == messages(expected["warnings"])
    value = expected["value"]
    assert actual.kind == value["kind"]
    assert actual.categories == (None if value["levels"] is None else tuple(value["levels"]))
    if value["kind"] in {"factor", "character", "logical"}:
        assert list(actual) == value["values"]
    else:
        assert actual.storage == value["type"]
        compare_values(actual, value, rtol=2e-13, atol=2e-13)
        actual_nan = [
            isinstance(value, (float, np.floating))
            and math.isnan(value)
            and np.float64(value).view(np.uint64) != np.float64(coerce._NA_REAL).view(np.uint64)
            for value in actual
        ]
        assert actual_nan == value["nan"]


@pytest.mark.parametrize("case", REFERENCE["literal_evaluation"], ids=lambda case: case["name"])
def test_declared_literals_keep_r_types_when_empty_or_missing(case):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = formula._expression_values({}, case["expression"], case["n"])
    assert messages([item.message for item in caught]) == messages(case["expected"]["warnings"])
    expected = case["expected"]["value"]
    assert actual.kind == expected["kind"]
    assert actual.categories is None
    if actual.kind == "numeric":
        assert actual.storage == expected["type"]
        compare_values(actual, expected, rtol=0, atol=0)
    else:
        assert list(actual) == expected["values"]


@pytest.mark.parametrize("case", REFERENCE["escape_evaluation"], ids=lambda case: case["name"])
def test_r_literal_escape_values_and_explicit_encoding_boundary(case):
    sources = {
        name: expression._ExpressionVector(values[: case["n"]], "character")
        for name, values in REFERENCE["escape_sources"].items()
    }
    reference = case["expected"]
    if case["unsupported"] or "error" in reference["value"]:
        with pytest.raises(ValueError, match="formula|escape|Unicode|UTF"):
            formula._expression_values(sources, case["expression"], case["n"])
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = formula._expression_values(sources, case["expression"], case["n"])
    assert messages([item.message for item in caught]) == messages(reference["warnings"])
    assert actual.kind == "character"
    assert list(actual) == reference["value"]["strings"]
    assert [list(value.encode("utf-8")) for value in actual] == reference["value"]["bytes"]


@pytest.mark.parametrize("n", [0, 4])
@pytest.mark.parametrize("wrapper", ["a", "I(a)", "identity(I(a))"])
def test_declared_empty_and_missing_boolean_array_types(n, wrapper):
    pandas = pytest.importorskip("pandas")
    actual = formula._expression_values(
        {"a": pandas.array([None] * n, dtype="boolean")}, wrapper, n
    )
    assert actual.kind == "logical"
    assert all(coerce._is_missing_value(value) for value in actual)
    assert len(actual) == n
    empty = formula._expression_values({"a": np.asarray([], dtype=bool)}, wrapper, 0)
    assert empty.kind == "logical"
    assert list(empty) == []


@pytest.mark.parametrize(
    "source",
    ["x[0]", "arbitrary(x)", "x && z", "x || z", "I()", "I(x,z)", "'x", "(x", "x < z < x"],
)
def test_restricted_expression_parser_rejects_unsupported_syntax(source):
    with pytest.raises(ValueError, match="unsupported formula expression"):
        formula._expression_values({"x": [1], "z": [2]}, source, 1)


def test_cached_expression_syntax_reads_current_values_and_never_attributes():
    source = "I(x > z & !a)"
    parsed = expression._parse_expression(source)
    assert expression._parse_expression(source) is parsed
    data = {"x": [2, 0], "z": [1, 1], "a": [False, False]}
    assert list(formula._expression_values(data, source, 2)) == [True, False]
    data["x"] = [0, 2]
    assert list(formula._expression_values(data, source, 2)) == [False, True]
    with pytest.raises(KeyError, match="not found in data"):
        formula._expression_values({"x": [1]}, "x.__class__", 1)


@pytest.mark.parametrize(
    "case", REFERENCE.get("offset_overrides", []), ids=lambda case: case["name"]
)
def test_logical_offsets_preserve_contrasts_and_validate_declared_override_type(case):
    data = training_data({"variant": "complete"}, "lists")
    fit = getattr(r, case["kind"])("Surv(futime, fustat) ~ x + offset(a)", data)
    new = new_data("complete", "lists")
    source = new["a"]
    if case["numeric"]:
        source = expression._ExpressionVector([float(value) for value in source], "numeric")
    if case["evaluated"]:
        new = bridge._r_model_frame(
            {"x": new["x"], "offset(a)": source},
            len(source),
            REFERENCE["newdata"]["complete"]["row_names"],
            {"x": {"kind": "numeric"}, "offset(a)": {"kind": source.kind}},
        )
    else:
        new["a"] = source
    expected = case["expected"]["value"]
    if "error" in expected:
        assert expected["error"] == "contrasts apply only to factors"
        with pytest.raises(ValueError, match="contrasts apply only to factors"):
            r.model_matrix(fit, new, _with_metadata=True)
    else:
        compare_matrix(r.model_matrix(fit, new, _with_metadata=True), expected)


@pytest.mark.parametrize(
    "case", REFERENCE.get("training_failures", []), ids=lambda case: case["name"]
)
def test_empty_or_all_missing_training_data_fails_like_stock_r(case):
    data = training_data({"variant": "truth"}, "lists")
    if case["variant"] == "empty":
        data = new_data("empty", "lists")
    else:
        n = len(REFERENCE["row_names"])
        for name in ("a", "b"):
            data[name] = expression._ExpressionVector([None] * n, "logical")
    assert "error" in case["expected"]["value"]
    with pytest.raises(ValueError, match="empty|rows|observations|missing|finite|categories"):
        getattr(r, case["kind"])(
            "Surv(futime, fustat) ~ x + I(a & b)", data, na_action=case["na_action"]
        )


def assert_prediction(fit, data, kind, action, expected):
    options = {"type": kind, "na_action": action, "_with_row_names": True}
    reference = expected.get("intended", expected["raw"]["value"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in reference:
            with pytest.raises((ValueError, TypeError), match="missing|finite|NA|contrasts"):
                r.predict(fit, data, **options)
            return
        result = r.predict(fit, data, **options)
    assert messages([item.message for item in caught]) == messages(expected["raw"]["warnings"])
    compare_values(result["values"], reference)
    assert result["fit_names"] == (reference["rows"] if reference["dim"] else reference["names"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("storage", ["lists", "numpy"])
def test_typed_formula_fits_frames_matrices_and_predictions_match_r(case, storage):
    data = training_data(case, storage)
    options = {"x": True, "model": True, "na_action": case["na_action"], "subset": case["subset"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in case:
            with pytest.raises((ValueError, TypeError), match="missing|finite|initial|NA"):
                getattr(r, case["kind"])("Surv(futime, fustat) ~ " + case["rhs"], data, **options)
            return
        fit = getattr(r, case["kind"])("Surv(futime, fustat) ~ " + case["rhs"], data, **options)
    assert messages([item.message for item in caught]) == messages(case["fit_warnings"])
    assert r.coef_names(fit) == case["coefficients"]["names"]
    compare_values(r.coef(fit), case["coefficients"])
    compare_values(r.vcov(fit), case["variance"])
    if case.get("means") is not None:
        compare_values(fit.means, case["means"])
    if case.get("scale") is not None:
        compare_values(fit.scale, case["scale"])
    assert (list(fit.na_action.rows) if fit.na_action else []) == case["na_rows"]
    compare_matrix(r.model_matrix(fit, _with_metadata=True), case["matrix"])
    frame = formula.model_frame(
        "Surv(futime, fustat) ~ " + case["rhs"],
        data,
        subset=case["subset"],
        na_action=case["na_action"],
    )
    variables = dict(formula._model_variables(frame))
    assert list(variables) == list(case["frame"])
    for name, encoded in case["frame"].items():
        actual = variables[name]
        assert actual.kind == encoded["kind"]
        assert actual.categories == (
            None if encoded["levels"] is None else tuple(encoded["levels"])
        )
        if encoded["kind"] in {"logical", "factor", "character"}:
            assert list(actual) == encoded["values"]
        else:
            assert actual.storage == encoded["type"]
            compare_values(actual, encoded, rtol=2e-13, atol=2e-13)
    if case.get("offset") is not None:
        compare_values(frame.offset, case["offset"], rtol=0, atol=0)
    for kind, encoded in case["training"].items():
        assert_prediction(fit, None, kind, "na.pass", {"raw": encoded})
    for name, outputs in case["newdata"].items():
        new = new_data(name, storage)
        matrix = outputs["matrix"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in matrix["value"]:
                with pytest.raises((ValueError, TypeError), match="missing|contrasts|levels"):
                    r.model_matrix(fit, new, _with_metadata=True)
            else:
                compare_matrix(r.model_matrix(fit, new, _with_metadata=True), matrix["value"])
        assert messages([item.message for item in caught]) == messages(matrix["warnings"])
        for key, expected in outputs.items():
            if key == "matrix":
                continue
            kind, action = key.split("/")
            assert_prediction(fit, new, kind, action, expected)
