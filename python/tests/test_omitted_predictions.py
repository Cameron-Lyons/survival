"""Full missing-value kinds and omission shapes from independent stock R calls."""

import importlib
import json
import math
import re
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
frame = importlib.import_module("survival.pybridge")._r_data_frame
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/omitted_prediction_reference.json").read_text()
)


@cache
def fitted(name):
    data = {**REFERENCE["data"], "cl": RFactor(REFERENCE["data"]["cl"], REFERENCE["levels"])}
    kind, rhs = REFERENCE["specs"][name]
    options = {"scale": 1} if name == "aft_fixed" else {}
    return getattr(r, kind)(
        "Surv(futime,fustat)~" + rhs,
        frame(data, len(REFERENCE["row_names"]), REFERENCE["row_names"]),
        x=True,
        **options,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("as_arrays", [False, True], ids=["lists", "arrays"])
def test_omitted_predictions_match_complete_r_outputs(case, as_arrays):
    data = {name: values[2:6] for name, values in REFERENCE["data"].items()}
    column = "cl" if case["model"] in {"cox_sparse_only", "cox_factor", "aft_factor"} else "age"
    retained = [2, 0, 1, 3][: case["keep"]]
    for row in range(4):
        if row not in retained:
            data[column][row] = None if column == "cl" or case["kind"] == "NA" else math.nan
    data["cl"] = RFactor(data["cl"], REFERENCE["levels"])
    options = {
        "type": case["type"],
        "se_fit": case["se_fit"],
        "na_action": case["na_action"],
        "_as_arrays": as_arrays,
        "_with_group_names" if case["grouped"] else "_with_row_names": True,
    }
    if case["p"] is not None:
        options["p"] = case["p"]
    if case["grouped"]:
        options["collapse"] = ["b", "a", "b", "a"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in case["result"]:
            with pytest.raises(ValueError, match=re.escape(case["result"]["error"])):
                r.predict(fitted(case["model"]), frame(data, 4, REFERENCE["new_names"]), **options)
            return
        output = r.predict(fitted(case["model"]), frame(data, 4, REFERENCE["new_names"]), **options)
    assert [str(w.message) for w in caught] == case["warnings"]
    if case["se_fit"]:
        pairs = [
            (output["values"].fit, case["result"]["fit"], "fit_names"),
            (output["values"].se_fit, case["result"]["se.fit"], "se_names"),
        ]
    else:
        pairs = [(output["values"], case["result"], "fit_names")]
    for actual, expected, name in pairs:
        assert isinstance(actual, np.ndarray if as_arrays else list)
        shape = expected["dim"] or [len(expected["values"])]
        values = np.asarray(actual, dtype=float).reshape(shape).ravel(order="F")
        np.testing.assert_allclose(
            values, np.asarray(expected["values"], dtype=float), rtol=3e-7, atol=3e-9
        )
        bits = np.ascontiguousarray(values).view(np.uint64)
        genuine_nan = np.isnan(values) & (
            (bits & 0xFFFFFFFF) != REFERENCE["metadata"]["na_payload"]
        )
        np.testing.assert_array_equal(genuine_nan, expected["nan"])
        if case["grouped"]:
            labels = output["group_names"]
        else:
            labels = (
                output["fit_names"]
                if name == "se_names" and output["shared_names"]
                else output[name]
            )
        assert labels == (expected["rows"] if expected["dim"] else expected["names"])
