"""AFT matrix input evaluation, compared with unmodified stock R."""

import json
import re
import warnings
from pathlib import Path

import pytest

from .r_fixture_support import RFactor
from .test_logical_model_matrices import (
    columns_as,
    compare_contrasts,
    compare_values,
    formula,
    frame,
    r,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aft_matrix_input_reference.json").read_text()
)


def data_for(case, container):
    data = {name: list(values) for name, values in REFERENCE["data"].items()}
    n = len(data["age"])
    if case["variant"] == "missing":
        data["g"][1] = data["g"][6] = None
        data["age"][4] = data["off"][8] = None
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    labels = [f"病人 / {i + 1}" for i in range(n)] if case["named"] else None
    return frame(columns_as(data, container), n, labels)


def newdata_for(data, input_name, container):
    if input_name == "stored":
        return None
    new = formula._data_rows(data, list(data), [6, 2, 6, 0], len(data["age"]))
    changes = {
        "missing_g": "g",
        "missing_h": "h",
        "missing_age": "age",
        "missing_flag": "flag",
        "missing_offset": "off",
        "missing_cluster": "id",
        "invalid_offset": "off",
    }
    if input_name in changes:
        name = changes[input_name]
        values = list(new[name])
        values[1] = -1 if input_name == "invalid_offset" else None
        new[name] = RFactor(values, new[name].categories) if name in {"g", "h"} else values
    if input_name == "all_missing_g":
        new["g"] = RFactor([None] * 4, REFERENCE["levels"]["g"])
    if input_name in {"absent_g", "absent_groups"}:
        del new["g"]
    if input_name in {"absent_h", "absent_groups"}:
        del new["h"]
    if input_name == "unknown_g":
        new["g"] = RFactor(["unseen"] * 4, ["unseen"])
    if input_name == "numeric_g":
        new["g"] = [1, 2, 3, 4]
    if input_name == "absent_cluster":
        del new["id"]
    if input_name == "missing_response":
        new["futime"] = [None] * 4
    if input_name == "absent_response":
        del new["futime"], new["fustat"]
    if input_name in {"empty", "single"}:
        new = formula._data_rows(new, list(new), [] if input_name == "empty" else [0], 4)
    if input_name == "zero_columns":
        new.clear()
    return frame(columns_as(new, container), new.nrow, new.row_names)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_aft_matrix_inputs_match_stock(case, container):
    data = data_for(case, container)
    options = {"x": case["cached"], "model": True, "na_action": "na.exclude"}
    if case["variant"] == "subset":
        options["subset"] = [6, 2, 6, 0, 4, 1, 9, 7, 3, 10, 5, 8]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.survreg("Surv(futime,fustat)~" + case["rhs"], data, **options)
    assert [re.sub(r"\s+", "", str(w.message)) for w in caught] == [
        re.sub(r"\s+", "", message) for message in case["fit_warnings"]
    ]
    for input_name, expected in case["expected"].items():
        new = newdata_for(data, input_name, container)
        value = expected["value"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if "error" in value:
                message = value["error"][0]
                if "not found" in message:
                    column = re.search("object '([^']+)'", message).group(1)
                    match = f"column '{column}' not found"
                    error = KeyError
                else:
                    match = "unknown level|new level|contrasts apply only"
                    error = ValueError
                with pytest.raises(error, match=match):
                    r.model_matrix(fit, new, _with_metadata=True)
                continue
            actual = r.model_matrix(fit, new, _with_metadata=True)
        assert [str(w.message).removesuffix(" in log(off)") for w in caught] == expected["warnings"]
        compare_values(actual["data"], value)
        assert actual["columns"] == (value["columns"] or [])
        assert actual["assign"] == value["assign"]
        assert actual["row_names"] == (value["rows"] or [])
        compare_contrasts(actual["contrasts"], value["contrasts"])
    assert "row_names" not in r.model_matrix(fit)


def test_aft_prediction_requires_scale_strata_after_matrix_evaluation():
    case = {"variant": "complete", "named": True}
    data = data_for(case, "lists")
    fit = r.survreg("Surv(futime,fustat)~age+strata(g)+offset(off)", data)
    new = newdata_for(data, "missing_g", "lists")
    predictions = r.predict(fit, new, type="uquantile", p=[0.2, 0.8], na_action="omit")
    assert len(predictions) == 3
    matrix = r.model_matrix(fit, new, _with_metadata=True)
    assert len(matrix["data"]) == 4
    del new["g"]
    assert r.model_matrix(fit, new, _with_metadata=True) == matrix
    with pytest.raises(KeyError, match="column 'g' not found"):
        r.predict(fit, new)
