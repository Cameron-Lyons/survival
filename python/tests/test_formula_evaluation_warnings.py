"""Whole stock-R matrices and fits when transforms precede row selection."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .r_fixture_support import RFactor
from .test_logical_model_matrices import compare_contrasts, compare_values, frame, r
from .test_model_matrix_na_action import NA_REAL, columns_as

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/formula_warning_reference.json").read_text()
)


def messages(values):
    return [re.sub(r"\s+", "", re.sub(r" in (log|sqrt)\(.*\)$", "", value)) for value in values]


def data_for(container, named=False):
    data = {name: list(values) for name, values in REFERENCE["data"].items()}
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    n = len(data["age"])
    labels = [f"病人 / {i + 1}" for i in range(n)] if named else None
    return frame(columns_as(data, container), n, labels)


def new_data(input_name, container, named):
    indices = [6, 2, 6, 0]
    data = {name: [values[i] for i in indices] for name, values in REFERENCE["data"].items()}
    labels = ["病人 / 7", "病人 / 3", "病人 / 7.1", "病人 / 1"] if named else ["7", "3", "7.1", "1"]
    if input_name in {
        "domain",
        "overlap_covariate",
        "overlap_response",
        "overlap_offset",
        "overlap_strata",
    }:
        data["z"][1] = -1
        data["w"][1] = -4
        column = {
            "overlap_covariate": "age",
            "overlap_response": "fustat",
            "overlap_offset": "off",
            "overlap_strata": "g",
        }.get(input_name)
        if column:
            data[column][1] = None
    if input_name in {"source_na", "source_nan", "zero", "overflow_omitted"}:
        data["z"][1] = {
            "source_na": None,
            "source_nan": float("nan"),
            "zero": 0,
            "overflow_omitted": 1000,
        }[input_name]
        if input_name == "overflow_omitted":
            data["age"][1] = None
    if input_name == "all_omitted":
        data["age"], data["z"], data["w"] = [None] * 4, [-1] * 4, [-4] * 4
    if input_name == "empty":
        data = {name: [] for name in data}
        labels = []
    if input_name == "single_omitted":
        data = {name: [values[1]] for name, values in data.items()}
        data["age"], data["z"], data["w"] = [None], [-1], [-4]
        labels = [labels[1]]
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    return frame(columns_as(data, container), len(labels), labels or None)


def compare_matrix(actual, expected):
    values = np.asarray(actual["data"], dtype=float).reshape(expected["dim"]).ravel(order="F")
    numbers = np.asarray(expected["values"], dtype=float)
    for key, number in (("positive_infinity", np.inf), ("negative_infinity", -np.inf)):
        numbers[np.asarray(expected[key], dtype=int) - 1] = number
    np.testing.assert_allclose(values, numbers, rtol=1e-12, atol=1e-12)
    na = values.view(np.uint64) == np.asarray(NA_REAL).view(np.uint64)
    assert (np.flatnonzero(na) + 1).tolist() == expected["na"]
    assert (np.flatnonzero(np.isnan(values) & ~na) + 1).tolist() == expected["nan"]
    assert actual["columns"] == (expected["columns"] or [])
    assert actual["assign"] == expected["assign"]
    assert actual["row_names"] == (expected["rows"] or [])
    compare_contrasts(actual["contrasts"], expected["contrasts"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_transform_matrices_match_stock_values_and_warnings(case, container):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"],
            data_for(container, case["named"]),
            x=True,
            model=True,
        )
    assert messages([str(w.message) for w in caught]) == messages(case["fit_warnings"])
    for input_name, expected in case["expected"].items():
        new = new_data(input_name, container, case["named"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in expected["value"]:
                assert "missing values" in expected["value"]["error"][0]
                with pytest.raises(ValueError, match="missing values"):
                    r.model_matrix(fit, new, na_action=case["action"], _with_metadata=True)
            else:
                actual = r.model_matrix(fit, new, na_action=case["action"], _with_metadata=True)
                compare_matrix(actual, expected["value"])
                group = expected["value"].get("strata")
                assert actual.get("strata") == (None if group is None else group["labels"])
        assert messages([str(w.message) for w in caught]) == messages(expected["warnings"])


@pytest.mark.parametrize("case", REFERENCE["fit_cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_training_transforms_precede_subset_and_omission(case, container):
    data = data_for("lists")
    data["z"][1] = -1
    options = {"x": True, "model": True, "na_action": case["action"]}
    if case["variant"] == "covariate":
        data["age"][1] = None
    if case["variant"] == "weights":
        options["weights"] = [None if i == 1 else 1 for i in range(data.nrow)]
    if case["variant"] == "subset":
        options["subset"] = [6, 2, 6, 0, *[i for i in range(data.nrow) if i not in {0, 1, 2, 6}]]
    data = frame(columns_as(data, container), data.nrow, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in case["value"]:
            assert "missing values" in case["value"]["error"][0]
            with pytest.raises(ValueError, match="missing values"):
                getattr(r, case["kind"])("Surv(futime,fustat)~" + case["rhs"], data, **options)
        else:
            fit = getattr(r, case["kind"])("Surv(futime,fustat)~" + case["rhs"], data, **options)
            compare_values(r.coef(fit), case["value"]["coef"])
            compare_values(r.vcov(fit), case["value"]["var"])
            compare_matrix(r.model_matrix(fit, _with_metadata=True), case["value"]["matrix"])
    assert messages([str(w.message) for w in caught]) == messages(case["warnings"])
