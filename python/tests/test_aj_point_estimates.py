"""Point-only AJ results retain every estimate, count and metadata field."""

import dataclasses
import json
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/aj_entry_reference.json").read_text())
UNCERTAINTY = {"std_err", "std_chaz", "std_auc", "se0", "lower", "upper", "influence_pstate"}


def snapshot(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {name: snapshot(item) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [snapshot(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    descriptors = {
        name
        for name, descriptor in vars(type(value)).items()
        if isinstance(descriptor, types.GetSetDescriptorType) and not name.startswith("_")
    }
    if descriptors:
        return {name: snapshot(getattr(value, name)) for name in descriptors}
    return value


def fit_case(case, api, se_fit):
    data = {
        name: [values[row - 1] for row in case["rows"]]
        for name, values in REFERENCE["data"].items()
    }
    options = {
        "entry": case["entry"],
        "time0": case["time0"],
        "se_fit": se_fit,
        # Point-only calls must leave influence absent even when it was requested.
        "influence": True,
    }
    if case["start_time"] is not None:
        options["start_time"] = case["start_time"]
    if api == "formula":
        data["event"] = r._r_factor(data["event"], REFERENCE["event_levels"])
        return r.survfit(
            "Surv(start, stop, event) ~ " + ("group" if case["grouped"] else "1"),
            data,
            id="id",
            weights="weight" if case["weighted"] else None,
            **options,
        )
    return survival.surv_analysis.survfitaj(
        data["stop"],
        [REFERENCE["event_levels"].index(event) for event in data["event"]],
        REFERENCE["event_levels"][1:],
        start=data["start"],
        id=data["id"],
        weights=data["weight"] if case["weighted"] else None,
        strata=[0 if label == "one" else 1 for label in data["group"]] if case["grouped"] else None,
        **options,
    )


@pytest.mark.parametrize("api", ["formula", "native"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_point_only_complete_outputs_preserve_stock_r_estimates(case, api):
    actual = fit_case(case, api, False)
    reference = fit_case(case, api, True)
    for name in UNCERTAINTY:
        assert getattr(actual, name) is None, name
    for name, expected in case["expected"].items():
        if name in UNCERTAINTY:
            continue
        observed = getattr(actual, name)
        if name == "strata":
            observed = list(observed.values()) if isinstance(observed, dict) else observed
        if expected is None or name == "states":
            assert observed == expected, name
        elif name == "counts":
            assert snapshot(observed) == expected
        else:
            np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-14, err_msg=name)
    # Enumerate every exposed Rust field and every facade dataclass component,
    # covering metadata absent from the stock numerical fixture as well.
    payload, expected = snapshot(actual), snapshot(reference)
    for name in UNCERTAINTY:
        expected[name] = None
    if api == "formula":
        for name in UNCERTAINTY:
            assert payload["engine"][name] is None, name
            expected["engine"][name] = None
        for name in ("conf_int", "conf_type", "logse"):
            assert getattr(actual, name) is None, name
            expected[name] = None
    assert payload == expected
