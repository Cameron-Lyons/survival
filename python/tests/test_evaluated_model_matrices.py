"""Supplied R model-frame columns against unmodified stock matrix methods."""

import importlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .r_fixture_support import RFactor
from .test_formula_evaluation_warnings import compare_matrix
from .test_logical_model_matrices import frame, r
from .test_model_matrix_na_action import NA_REAL, columns_as

model_frame = importlib.import_module("survival.pybridge")._r_model_frame
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/evaluated_matrix_reference.json").read_text()
)


def numbers(value):
    values = np.asarray(value["values"], dtype=float)
    for name, number in (
        ("na", NA_REAL),
        ("nan", np.nan),
        ("positive_infinity", np.inf),
        ("negative_infinity", -np.inf),
    ):
        values[np.asarray(value[name], dtype=int) - 1] = number
    return values.reshape(value["dim"] or (len(values),), order="F")


def training_data(case, container):
    columns = {name: list(values) for name, values in REFERENCE["data"].items()}
    for name, levels in REFERENCE["levels"].items():
        columns[name] = RFactor(columns[name], levels)
    n = len(columns["age"])
    labels = [f"病人 / {i + 1}" for i in range(n)] if case["named"] else None
    return frame(columns_as(columns, container), n, labels)


def evaluated_frame(value, container):
    columns, metadata = {}, {}
    for name, column in value["columns"].items():
        kind = column["kind"]
        info = {"kind": kind}
        if kind == "factor":
            source = RFactor(column["value"], column["levels"])
        elif kind in {"character", "logical"}:
            source = column["value"]
        else:
            source = numbers(column["value"])
            if kind == "matrix":
                info["matrix_names"] = column["value"]["columns"]
            elif container == "lists":
                source = source.tolist()
        if kind != "factor" and kind != "matrix" and container == "numpy":
            source = np.asarray(source, dtype=object if kind in {"logical", "character"} else None)
        if kind != "factor" and kind != "matrix" and container == "pandas":
            import pandas as pd

            source = pd.Series(source, dtype="boolean" if kind == "logical" else None)
        if "contrast" in column:
            contrast = column["contrast"]
            info["contrast"] = {
                "data": numbers(contrast["value"]),
                "columns": contrast["value"]["columns"],
                "rows": contrast["value"]["rows"],
                "label": contrast["label"],
            }
        columns[name], metadata[name] = source, info
    constructor = model_frame if value["tagged"] else frame
    arguments = (columns, value["nrow"], value["rows"])
    return constructor(*arguments, metadata) if value["tagged"] else constructor(*arguments)


def warning_messages(values):
    return [re.sub(r"\s+", "", str(value)) for value in values]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("action", ["na.omit", "na.exclude", "na.pass", "na.fail"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_evaluated_frames_match_stock_values_metadata_and_warnings(case, action, container):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"],
            training_data(case, container),
            x=case["cached"],
            model=True,
        )
    assert warning_messages([str(w.message) for w in caught]) == warning_messages(
        case["fit_warnings"]
    )
    for input_name, value in case["inputs"].items():
        data = evaluated_frame(value, container)
        expected = case["references"][case["actions"][action] - 1][input_name]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in expected["value"]:
                message = expected["value"]["error"][0]
                if message == "number of variables != number of variable names":
                    # Stock AFT's reduced terms/predvars mismatch fails before
                    # resolving this untagged frame's absent raw stratum.
                    assert case["kind"] == "survreg"
                    assert case["rhs"] == "age + strata(g) + strata(g):flag"
                    assert input_name == "untagged"
                    exception, match = KeyError, "column 'g' not found"
                elif "not found" in message:
                    name = re.search("object '([^']+)'", message).group(1)
                    if (
                        case["kind"] == "coxph"
                        and "strata(" in case["rhs"]
                        and not re.search(r"strata\([^)]*\):", case["rhs"])
                        and name in {"g", "h"}
                    ):
                        exception, match = ValueError, "must contain the strata"
                    else:
                        exception, match = KeyError, f"column '{name}' not found"
                else:
                    exception, match = ValueError, re.escape(message)
                with pytest.raises(exception, match=match):
                    r.model_matrix(fit, data, na_action=action, _with_metadata=True)
            else:
                actual = r.model_matrix(fit, data, na_action=action, _with_metadata=True)
                compare_matrix(actual, expected["value"])
                group = expected["value"].get("strata")
                assert actual.get("strata") == (None if group is None else group["labels"])
                if value["tagged"] and group is not None:
                    assert actual["strata_levels"] == group["levels"]
        assert warning_messages([str(w.message) for w in caught]) == warning_messages(
            expected["warnings"]
        ), (case["name"], action, input_name)


@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_evaluated_inputs_and_results_do_not_change_fitted_or_later_matrices(container):
    for case in REFERENCE["cases"]:
        if (
            case["named"]
            or not case["cached"]
            or case["rhs"]
            not in {
                "age + log(z)",
                "age * g",
                "strata(g) + flag - 1",
                "ridge(age, rx, theta = 2)",
                "pspline(age, df = 3)",
                "age + frailty(id, sparse = FALSE, theta = 0.4)",
            }
        ):
            continue
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"], training_data(case, container), x=True
        )
        stored = r.model_matrix(fit, _with_metadata=True)
        data = evaluated_frame(case["inputs"]["complete"], container)
        result = r.model_matrix(fit, data, _with_metadata=True)
        if result["data"] and result["data"][0]:
            result["data"][0][0] = -999
        result["columns"].clear()
        compare_matrix(
            r.model_matrix(fit, data, _with_metadata=True),
            case["references"][0]["complete"]["value"],
        )
        changed = evaluated_frame(case["inputs"]["altered_values"], container)
        data.clear()
        data.update(changed)
        data.column_metadata = changed.column_metadata
        compare_matrix(
            r.model_matrix(fit, data, _with_metadata=True),
            case["references"][0]["altered_values"]["value"],
        )
        assert r.model_matrix(fit, _with_metadata=True) == stored


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE["cases"] if case["named"] and not case["cached"]],
    ids=lambda case: case["name"],
)
@pytest.mark.parametrize("layout", ["strided", "big_endian", "readonly", "integer"])
def test_evaluated_numeric_array_layouts_match_stock_without_kind_metadata(case, layout):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"], training_data(case, "numpy"), model=True
        )
    for input_name, value in case["inputs"].items():
        if not value["tagged"]:
            continue
        data = evaluated_frame(value, "numpy")
        for name, source in data.items():
            if not isinstance(source, np.ndarray) or source.ndim != 1:
                continue
            if (
                data.column_metadata[name].get("kind") == "logical"
                and len(source)
                and all(isinstance(item, bool | np.bool_) for item in source)
            ):
                data[name] = np.asarray(source, dtype=bool)
                data.column_metadata[name].pop("kind")
            if source.dtype.kind not in "iuf":
                continue
            data.column_metadata[name].pop("kind", None)
            if layout == "strided":
                storage = np.empty(len(source) * 2, dtype=source.dtype)
                storage[::2] = source
                data[name] = storage[::2]
            elif layout == "big_endian":
                data[name] = source.astype(source.dtype.newbyteorder(">"))
            elif layout == "readonly":
                source.flags.writeable = False
            elif np.all(np.isfinite(source)) and np.all(source == np.floor(source)):
                data[name] = source.astype(np.int64)
        expected = case["references"][0][input_name]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in expected["value"]:
                with pytest.raises(ValueError, match=re.escape(expected["value"]["error"][0])):
                    r.model_matrix(fit, data, na_action="na.fail", _with_metadata=True)
            else:
                actual = r.model_matrix(fit, data, na_action="na.fail", _with_metadata=True)
                compare_matrix(actual, expected["value"])
                group = expected["value"].get("strata")
                assert actual.get("strata") == (None if group is None else group["labels"])
                if group is not None:
                    assert actual["strata_levels"] == group["levels"]
        assert warning_messages([str(w.message) for w in caught]) == warning_messages(
            expected["warnings"]
        ), (case["name"], layout, input_name)
