"""Complete row metadata and values are checked against independent R predictions."""

import importlib
import json
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
_r_data_frame = importlib.import_module("survival.pybridge")._r_data_frame

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/prediction_row_reference.json").read_text()
)


@cache
def fitted(name, action):
    data = {**REFERENCE["data"], "cl": RFactor(REFERENCE["data"]["cl"], REFERENCE["levels"])}
    kind, rhs = REFERENCE["specs"][name]
    options = {
        "subset": [value - 1 for value in REFERENCE["subset"]],
        "na_action": action,
        "x": True,
    }
    if name == "aft_fixed":
        options["scale"] = 1
    return getattr(r, kind)(
        "Surv(futime,fustat)~" + rhs,
        _r_data_frame(data, len(REFERENCE["row_names"]), REFERENCE["row_names"]),
        **options,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_prediction_rows_and_values_match_independent_r(case):
    options = {
        "type": case["type"],
        "se_fit": case["se_fit"],
        "na_action": case["na_action"],
        "_with_row_names": True,
    }
    if case["p"] is not None:
        options["p"] = case["p"]
    newdata = None
    if case["source"] != "stored":
        data = dict(REFERENCE["newdata"])
        names = REFERENCE["new_names"]
        if case["source"] == "single":
            data = {name: values[:1] for name, values in data.items()}
            names = names[:1]
        data["cl"] = RFactor(data["cl"], REFERENCE["levels"])
        newdata = _r_data_frame(data, len(names), names)
    expected = case["expected"]["result"]
    assert "error" not in expected, case["name"]
    fit = fitted(case["model"], case["fit_action"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = r.predict(fit, newdata, **options)
    assert [str(w.message) for w in caught] == case["expected"]["warnings"]
    value = result["values"]
    pairs = (
        [
            (value.fit, expected["fit"], result["fit_names"]),
            (
                value.se_fit,
                expected["se.fit"],
                result["fit_names"] if result["shared_names"] else result["se_names"],
            ),
        ]
        if case["se_fit"]
        else [(value, expected, result["fit_names"])]
    )
    for actual, reference, labels in pairs:
        shape = reference["dim"] or [len(reference["values"])]
        values = np.asarray(reference["values"], dtype=float).reshape(shape, order="F")
        np.testing.assert_allclose(actual, values, rtol=3e-7, atol=3e-8, equal_nan=True)
        assert labels == (reference["rows"] if reference["dim"] else reference["names"])


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
def test_python_dataframe_indices_and_default_names_survive_row_selection(kind):
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(
        {"time": [1, 2, 3, 4, 5, 6], "status": [1, 1, 0, 1, 1, 1], "x": [2, 1, 4, 3, np.nan, 2]},
        index=["a", "a.1", "c", "d", "e", "f"],
    )
    for indexed, labels in [
        (data, ["a", "a.1", "a.2", "e", "c", "d", "f"]),
        (data.reset_index(drop=True), ["1", "2", "1.1", "5", "3", "4", "6"]),
    ]:
        fit = getattr(r, kind)(
            "Surv(time,status)~x", indexed, subset=[0, 1, 0, 4, 2, 3, 5], na_action="na.exclude"
        )
        prediction = r.predict(fit, type="terms", _with_row_names=True)
        assert prediction["fit_names"] == labels
