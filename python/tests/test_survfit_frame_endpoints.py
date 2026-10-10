"""Public survival frames preserve stock terminal values and fitter missingness."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
core = survival._survival
survfit_boundary = importlib.import_module("survival.r._survfit")
types = importlib.import_module("survival.r._types")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/survfit_frame_endpoint_reference.json").read_text()
)


def make_fit(case, interface):
    time = np.asarray(case["time"], dtype=float)
    status = np.asarray(case["status"], dtype=float)
    options = {
        "weights": case["weights"],
        "id": list(range(1, len(time) + 1)),
        "robust": case["robust"],
        "stype": case["stype"],
        "ctype": case["stype"],
        "timefix": False,
    }
    if case["robust"]:
        options["influence"] = 3
    if interface == "native":
        strata = None if case["group"] is None else [int(value == "b") for value in case["group"]]
        engine = core.survfitkm(time, status.astype(np.int32), strata=strata, **options)
        return survfit_boundary._km_result(
            engine, ["a", "b"], types.SurvfitCall(), None, True, None
        )
    if interface == "formula":
        data = {"time": time, "status": status}
        if case["group"] is not None:
            data["group"] = case["group"]
        group = "1" if case["group"] is None else "group"
        return r.survfit(f"Surv(time,status) ~ {group}", data, **options)
    return r.survfit(r.Surv(time, status), group=case["group"], **options)


@pytest.mark.parametrize("interface", ["facade", "formula", "native"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_public_survival_frame_preserves_complete_stock_endpoint_values(case, interface):
    fit = make_fit(case, interface)
    actual = r.as_data_frame(fit)
    expected = case["frame"]
    assert set(actual) == set(expected)
    assert fit.logse is case["logse"]
    for field, values in expected.items():
        if field == "strata":
            assert actual[field] == (
                values
                if interface == "formula"
                else [value.removeprefix("group=") for value in values]
            )
        else:
            np.testing.assert_allclose(
                np.asarray(actual[field], dtype=float),
                np.asarray(values, dtype=float),
                rtol=2e-12,
                atol=1e-14,
                equal_nan=True,
                err_msg=field,
            )
    zero_rows = np.asarray(actual["surv"]) == 0
    if case["stype"] == 1:
        assert zero_rows.any()
        for field in ("std.err", "lower", "upper"):
            if case["robust"]:
                np.testing.assert_array_equal(np.asarray(actual[field])[zero_rows], 0.0)
            else:
                assert np.isnan(np.asarray(actual[field])[zero_rows]).all()
