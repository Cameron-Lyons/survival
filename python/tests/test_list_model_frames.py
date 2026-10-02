"""Stock-R list versus data-frame row inference for variable-free designs."""

import json
import re
import warnings
from pathlib import Path

import pytest

from .r_fixture_support import RFactor
from .test_logical_model_matrices import compare_contrasts, compare_values, frame, r
from .test_model_matrix_na_action import columns_as

REFERENCE = json.loads((Path(__file__).parent / "fixtures/list_frame_reference.json").read_text())


def input_data(container):
    data = {name: list(values) for name, values in REFERENCE["data"].items()}
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    return columns_as(data, container)


def new_data(input_name, container):
    data = {name: [values[i] for i in [6, 2, 6, 0]] for name, values in REFERENCE["data"].items()}
    if input_name.endswith("missing"):
        data["z"][1] = data["g"][2] = data["off"][3] = None
    for name, levels in REFERENCE["levels"].items():
        data[name] = RFactor(data[name], levels)
    n = 4
    if input_name == "frame_empty":
        data = {
            name: RFactor([], REFERENCE["levels"][name]) if name in {"g", "h"} else []
            for name in data
        }
        n = 0
    if input_name in {"list_empty", "frame_zero_columns"}:
        data = {}
    data = columns_as(data, container)
    return data if input_name.startswith("list") else frame(data, n, None)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("container", ["lists", "numpy", "pandas"])
def test_list_and_frame_matrices_match_stock(case, container):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(r, case["kind"])(
            "Surv(futime,fustat)~" + case["rhs"], input_data(container), x=True
        )
    assert [re.sub(r"\s+", "", str(w.message)) for w in caught] == [
        re.sub(r"\s+", "", message) for message in case["fit_warnings"]
    ]
    for input_name, expected in case["expected"].items():
        new = new_data(input_name, container)
        value = expected["value"]
        if "error" in value:
            message = value["error"][0]
            if "not found" in message:
                column = re.search("object '([^']+)'", message).group(1)
                error = ValueError if case["kind"] == "coxph" and column == "g" else KeyError
                cause = (
                    "must contain the strata"
                    if error is ValueError
                    else f"column '{column}' not found"
                )
                with pytest.raises(error, match=cause):
                    r.model_matrix(fit, new, na_action=case["action"])
            else:
                assert "missing values" in message
                with pytest.raises(ValueError, match="missing values"):
                    r.model_matrix(fit, new, na_action=case["action"])
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = r.model_matrix(fit, new, na_action=case["action"], _with_metadata=True)
        assert [str(w.message) for w in caught] == expected["warnings"]
        compare_values(result["data"], value)
        assert result["columns"] == (value["columns"] or [])
        assert result["assign"] == value["assign"]
        assert result["row_names"] == (value["rows"] or [])
        compare_contrasts(result["contrasts"], value["contrasts"])
        group = value.get("strata")
        assert result.get("strata") == (None if group is None else group["labels"])
