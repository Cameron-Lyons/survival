"""Shared Fine–Gray preparation against the stock-R numerical oracle."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival.regression
CASES = json.loads((Path(__file__).parent / "fixtures/finegray_prepared.json").read_text())["cases"]


def _arguments(case):
    return {key: value for key, value in case.items() if key not in {"name", "expected"}}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_prepared_finegray_matches_stock_r(case):
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=expected["error"]):
            core.finegray_expand(**_arguments(case))
        return
    result = core.finegray_expand(**_arguments(case)).to_arrays()
    assert set(result) == set(expected)
    for key, values in expected.items():
        np.testing.assert_allclose(result[key], values, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("layout", ["list", "f32", "strided", "readonly"])
def test_prepared_arrays_are_independent_and_layouts_are_accepted(layout):
    case = CASES[0]
    args = _arguments(case)
    for key, value in args.items():
        if not isinstance(value, list):
            continue
        integer = key in {"id", "status", "strata"}
        if layout == "f32":
            args[key] = np.array(value, dtype=np.int32 if integer else np.float32)
        elif layout in {"strided", "readonly"}:
            array = np.repeat(np.array(value), 2)[::2]
            if layout == "readonly":
                array.flags.writeable = False
            args[key] = array
    result = core.finegray_expand(**args)
    original = result.row
    arrays = result.to_arrays()
    assert arrays["row"].dtype == np.int64
    assert arrays["wt"].dtype == np.float64
    arrays["row"][0] = -7
    arrays["wt"][0] = -9
    assert result.row == original
    assert result.wt[0] >= 0
    del result
    assert arrays["row"][0] == -7


def test_prepared_calls_are_independent():
    cases = [case for case in CASES[:12] if "error" not in case["expected"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda case: core.finegray_expand(**_arguments(case)).row, cases))
    assert values == [case["expected"]["row"] for case in cases]


def test_missing_strata_levels_do_not_drop_events():
    # A subset can retain unused factor levels; native codes need not be consecutive.
    result = core.finegray_expand([1, 3, 2, 4], [1, 2, 2, 1], strata=[10, 10, 90, 90])
    assert result.row == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": [1]}, "status length"),
        ({"time": [float("nan"), 2]}, "time"),
        ({"status": [1, -1]}, "non-negative"),
        ({"event_type": 0}, "positive"),
        ({"strata": [1]}, "strata length"),
        ({"id": [1]}, "id length"),
        ({"weights": [1]}, "weights length"),
        ({"weights": [1, float("inf")]}, "weights"),
        ({"start": [0, 0]}, "subject id"),
        ({"start": [1, 0], "id": [1, 2]}, "start"),
    ],
)
def test_prepared_rejects_malformed_inputs(updates, message):
    with pytest.raises(ValueError, match=message):
        core.finegray_expand(**{"time": [1, 2], "status": [1, 2], **updates})


def test_formula_rejects_missing_subject_ids():
    with pytest.raises(ValueError, match="id must not contain missing"):
        survival.r_api.finegray(
            "Surv(start, time, status, type='mstate') ~ x",
            {"start": [0, 0, 0], "time": [1, 2, 3], "status": [0, 1, 2], "x": [1, 2, 3]},
            id=[1, None, 3],
        )


def test_formula_output_names_replace_matching_covariates():
    result = survival.r_api.finegray(
        "Surv(time, status, type='mstate') ~ fgstart",
        {"time": [1, 2, 3], "status": [0, 1, 2], "fgstart": [8, 9, 10]},
    )
    assert result["fgstart"] == [0, 0, 0]
