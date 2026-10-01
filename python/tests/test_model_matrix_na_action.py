"""Whole stock-R matrices under all four omission rules, including NA kinds."""

import importlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .r_fixture_support import RFactor
from .test_logical_model_matrices import columns_as as _columns_as
from .test_logical_model_matrices import compare_contrasts, formula, frame, r

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/matrix_na_action_reference.json").read_text()
)
NA_REAL = importlib.import_module("survival.r._coerce")._NA_REAL


def columns_as(data, container):
    if container == "pandas":
        import pandas as pd

        # Float Series erase the distinction between source None and NaN.
        return {
            name: values
            if isinstance(values, RFactor)
            else pd.Series(values, dtype="boolean" if name == "flag" else object)
            for name, values in data.items()
        }
    return _columns_as(data, container)


def data_for(case, container):
    data = {name: list(values) for name, values in REFERENCE["data"].items()}
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    n = len(data["age"])
    labels = [f"病人 / {i + 1}" for i in range(n)] if case["named"] else None
    return frame(columns_as(data, container), n, labels)


def newdata_for(data, input_name, container):
    if input_name == "stored":
        return None
    new = formula._data_rows(data, list(data), [6, 2, 6, 0], len(data["age"]))

    def change(name, index, value):
        values = list(new[name])
        values[index] = value
        new[name] = RFactor(values, new[name].categories) if name in {"g", "h"} else values

    if input_name in {"missing_numeric", "nan_numeric"}:
        change("age", 1, None if input_name == "missing_numeric" else float("nan"))
    if input_name == "missing_factor":
        change("g", 1, None)
    if input_name == "missing_logical":
        change("flag", 1, None)
    if input_name == "missing_strata":
        change("g", 1, None)
        change("h", 2, None)
    if input_name == "all_missing":
        for name in ("age", "flag", "g", "h"):
            for i in range(4):
                change(name, i, None)
    if input_name in {"missing_offset", "invalid_offset"}:
        change("off", 1, None if input_name == "missing_offset" else -1)
    if input_name == "invalid_numeric":
        change("z", 1, -1)
        change("z", 2, 0)
        change("w", 2, 0)
    if input_name in {"missing_unused", "invalid_unused"}:
        change("z", 1, None if input_name == "missing_unused" else -1)
    if input_name == "empty":
        new = formula._data_rows(new, list(new), [], 4)
    if input_name == "single":
        new = formula._data_rows(new, list(new), [1], 4)
        change("age", 0, None)
        change("flag", 0, None)
    return frame(columns_as(new, container), new.nrow, new.row_names)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_model_matrix_omission_matches_stock(case, container):
    data = data_for(case, container)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"], data, x=True, model=True
        )
    assert [re.sub(r"\s+", "", str(w.message)) for w in caught] == [
        re.sub(r"\s+", "", message) for message in case["fit_warnings"]
    ]
    for input_name, expected in case["expected"].items():
        new = newdata_for(data, input_name, container)
        value = expected["value"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in value:
                assert "missing values" in value["error"][0]
                with pytest.raises(ValueError, match="missing values"):
                    r.model_matrix(fit, new, na_action=case["action"], _with_metadata=True)
                continue
            result = r.model_matrix(fit, new, na_action=case["action"], _with_metadata=True)
        assert [re.sub(r" in (log|sqrt)\(.*\)$", "", str(w.message)) for w in caught] == expected[
            "warnings"
        ]
        shape = value["dim"]
        actual = np.asarray(result["data"], dtype=float).reshape(shape).ravel(order="F")
        numeric = np.asarray(value["values"], dtype=float)
        for key, number in (("positive_infinity", np.inf), ("negative_infinity", -np.inf)):
            numeric[np.asarray(value[key], dtype=int) - 1] = number
        np.testing.assert_allclose(actual, numeric, rtol=1e-12, atol=1e-12)
        na = actual.view(np.uint64) == np.asarray(NA_REAL).view(np.uint64)
        assert (np.flatnonzero(na) + 1).tolist() == value["na"]
        assert (np.flatnonzero(np.isnan(actual) & ~na) + 1).tolist() == value["nan"]
        assert result["columns"] == (value["columns"] or [])
        assert result["assign"] == value["assign"]
        assert result["row_names"] == (value["rows"] or [])
        compare_contrasts(result["contrasts"], value["contrasts"])
        if case["kind"] == "coxph" and new is not None:
            group = value.get("strata")
            assert result["strata"] == (None if group is None else group["labels"])


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_omission_options_are_validated_only_for_new_frames(kind):
    data = data_for({"named": False}, "lists")
    fit = getattr(r, kind)("Surv(futime,fustat)~age", data)
    assert r.model_matrix(fit, na_action="invalid") == r.model_matrix(fit)
    with pytest.raises(ValueError, match="na_action must"):
        r.model_matrix(fit, data, na_action="invalid")
    with pytest.raises(TypeError, match="string or None"):
        r.model_matrix(fit, data, na_action=[])
    new = newdata_for(data, "missing_numeric", "lists")
    np.testing.assert_equal(
        r.model_matrix(fit, new, na_action=None)["data"],
        r.model_matrix(fit, new, na_action="pass")["data"],
    )
