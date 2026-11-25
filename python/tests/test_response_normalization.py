"""Normalized survival responses govern formula missingness before any row removal."""

import copy
import importlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
_r_factor = importlib.import_module("survival.r._coerce")._r_factor
model_frame = importlib.import_module("survival.r._formula").model_frame

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/response_normalization_reference.json").read_text()
)


@pytest.mark.parametrize("storage", ["list", "numpy", "pandas"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_formula_response_normalization_matches_stock_r(case, storage):
    data = copy.deepcopy(case["data"])
    levels = case["status_levels"]
    if storage == "numpy":
        data = {
            name: np.asarray([np.nan if x is None else x for x in values])
            if name != "status" or levels is None and "logical/" not in case["name"]
            else values
            for name, values in data.items()
        }
    elif storage == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(data)
        if case["name"].startswith("logical/"):
            data["status"] = pd.array(case["data"]["status"], dtype="boolean")
    if levels:
        if storage == "pandas":
            data["status"] = pd.Categorical(data["status"], categories=levels)
        else:
            data["status"] = _r_factor(data["status"], levels)
    expected = case["expected"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in expected:
            with pytest.raises(ValueError, match="missing values"):
                model_frame(
                    case["formula"],
                    data,
                    subset=case["subset"],
                    na_action=case["action"],
                    weights="w",
                )
        else:
            frame = model_frame(
                case["formula"], data, subset=case["subset"], na_action=case["action"], weights="w"
            )
    assert [str(w.message) for w in caught] == case["warnings"]
    assert all(w.filename == __file__ for w in caught)
    if "error" in expected:
        return
    assert frame.response.type == expected["type"]
    np.testing.assert_allclose(
        np.asarray(frame.response.as_matrix(), dtype=float),
        np.asarray(expected["response"], dtype=float),
        equal_nan=True,
    )
    assert list(frame.response.states) == expected["states"]
    assert ([] if frame.na_action is None else list(frame.na_action.rows)) == expected["omitted"]
    np.testing.assert_allclose(
        np.asarray(frame.weights, dtype=float),
        np.asarray(expected["weights"], dtype=float),
        equal_nan=True,
    )


@pytest.mark.parametrize("action", ["na.omit", "na.exclude"])
def test_omitting_every_event_keeps_censored_observations(action):
    data = {
        "time": [1, 2, 3, 4],
        "status": [1, 2, 1, 2],
        "w": [1, None, 2, None],
        "id": [1, 2, 3, 4],
    }
    curve = r.survfit("Surv(time, status) ~ 1", data, weights="w", na_action=action)
    assert list(curve.n_event) == [0, 0]
    assert list(curve.surv) == [1, 1]
    check = r.survcheck("Surv(time, status) ~ w", data, id="id", na_action=action)
    assert check.y.event == (0, 0)
    assert check.na_action == [2, 4]
    assert check.n == {"id": 2, "observations": 2, "transitions": 0}


@pytest.mark.parametrize("kind", ["right", "left", "interval"])
@pytest.mark.parametrize("formula_input", [True, False])
def test_aft_na_pass_never_turns_unknown_statuses_into_censoring(kind, formula_input):
    data = {
        "time": list(range(1, 9)),
        "end": list(range(2, 10)),
        "status": [1, None, 0, 1, 0, 1, 0, 1],
        "x": [0, 1] * 4,
    }
    args = (
        (data["time"], data["end"], data["status"])
        if kind == "interval"
        else (data["time"], data["status"])
    )
    response = (
        f"Surv(time, end, status, type='{kind}') ~ x"
        if kind == "interval"
        else f"Surv(time, status, type='{kind}') ~ x"
    )
    arguments = {"data": data} if formula_input else {"x": [[1.0, x] for x in data["x"]]}
    response = response if formula_input else r.Surv(*args, type=kind)
    with pytest.raises(ValueError, match="missing values in the response"):
        r.survreg(response, na_action="na.pass", **arguments)
