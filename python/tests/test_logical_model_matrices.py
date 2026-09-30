"""Logical coding and complete matrix metadata from unmodified stock R calls."""

import importlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
frame = importlib.import_module("survival.pybridge")._r_data_frame
formula = importlib.import_module("survival.r._formula")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/logical_matrix_reference.json").read_text()
)


def columns_as(data, container):
    if container == "numpy":
        return {
            name: values
            if isinstance(values, RFactor)
            else np.asarray(values, dtype=object if None in values else None)
            for name, values in data.items()
        }
    if container == "pandas":
        import pandas as pd

        return {
            name: values
            if isinstance(values, RFactor)
            else pd.Series(values, dtype="boolean" if name in {"flag", "other"} else None)
            for name, values in data.items()
        }
    return data


def data_for(case, container):
    data = {name: list(values) for name, values in REFERENCE["data"].items()}
    n = len(data["age"])
    if case["variant"] in {"false", "true"}:
        data["flag"] = [case["variant"] == "true"] * n
    if case["variant"] == "missing":
        data["flag"][1] = data["flag"][6] = None
        data["age"][4] = None
    data["g"] = RFactor(data["g"], REFERENCE["levels"])
    labels = [f"病人 / {i + 1}" for i in range(n)] if case["named"] else None
    return frame(columns_as(data, container), n, labels)


def compare_contrasts(actual, expected):
    assert (actual is None) == (expected is None)
    if expected is None:
        return
    assert list(actual) == list(expected)
    for name, value in expected.items():
        if isinstance(value, list):
            assert actual[name] == value[0]
        else:
            compare_values(actual[name]["data"], value)
            assert actual[name]["rows"] == value["rows"]
            assert actual[name]["columns"] == value["columns"]


def compare_values(actual, expected):
    shape = expected["dim"] or [len(expected["values"])]
    values = np.asarray(actual, dtype=float).reshape(shape).ravel(order="F")
    np.testing.assert_allclose(
        values, np.asarray(expected["values"], dtype=float), rtol=3e-7, atol=3e-9
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_logical_coding_and_matrix_metadata_match_stock(case, container):
    data = data_for(case, container)
    options = {"x": True, "model": True, "na_action": "na.exclude"}
    if case["variant"] == "subset":
        options["subset"] = [6, 2, 6, 0, 4, 1, 9, 7, 3, 10, 5, 8]
    expected = case["expected"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(ValueError, match="at least two levels|" + expected["error"]):
                getattr(r, case["kind"])("Surv(futime,fustat)~" + case["rhs"], data, **options)
            return
        fit = getattr(r, case["kind"])("Surv(futime,fustat)~" + case["rhs"], data, **options)
    # R's paste() adds spaces around the variable list in convergence warnings.
    assert [re.sub(r"\s+", "", str(w.message)) for w in caught] == [
        re.sub(r"\s+", "", message) for message in case["fit_warnings"]
    ]
    coefficients = expected.get("coefficients")
    if coefficients is None:
        assert r.coef_names(fit) == []
        assert r.coef(fit) == []
    else:
        assert r.coef_names(fit) == (coefficients["names"] or [])
        compare_values(r.coef(fit), coefficients)
    for input_name in ("stored", "complete", "partial", "all_missing", "empty", "single"):
        new = None
        if input_name != "stored":
            new = formula._data_rows(data, list(data), [6, 2, 6, 0], len(data["age"]))
            if input_name == "partial":
                new["flag"] = list(new["flag"])
                new["age"] = list(new["age"])
                new["flag"][1] = None
                new["age"][3] = None
            if input_name == "all_missing":
                new["flag"] = [None] * 4
            if input_name in {"empty", "single"}:
                new = formula._data_rows(new, list(new), [] if input_name == "empty" else [0], 4)
            new = frame(columns_as(new, container), new.nrow, new.row_names)
        result = expected[input_name]
        value = result["value"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in value:
                with pytest.raises(ValueError, match=re.escape(value["error"][0])):
                    r.model_matrix(fit, new, _with_metadata=True)
                continue
            actual = r.model_matrix(fit, new, _with_metadata=True)
        assert [str(w.message) for w in caught] == result["warnings"]
        compare_values(actual["data"], value)
        assert actual["columns"] == (value["columns"] or [])
        assert actual["assign"] == value["assign"]
        assert actual["row_names"] == (value["rows"] or [])
        compare_contrasts(actual["contrasts"], value["contrasts"])
        assert "row_names" not in r.model_matrix(fit, new)
