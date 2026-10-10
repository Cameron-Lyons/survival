"""Large finite near ties and full curves against independent stock-R calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
CASES = json.loads((Path(__file__).parent / "fixtures/aeq_large_time_reference.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_large_time_normalization_matches_stock_r(case, interface):
    def normalize():
        if interface == "native":
            return survival.data_prep.aeq_surv(
                case["time"] if case["start"] is None else case["start"],
                None if case["start"] is None else case["time"],
                case["tolerance"],
            )
        response = (
            r.Surv(case["time"], case["status"])
            if case["start"] is None
            else r.Surv(case["start"], case["time"], case["status"])
        )
        return r.aeqSurv(response, tolerance=case["tolerance"])

    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match="effective length 0"):
            normalize()
        return
    actual = normalize()
    time = actual.time if interface == "facade" or case["start"] is None else actual.time2
    np.testing.assert_array_equal(time, expected["time"])
    if case["start"] is not None:
        start = actual.start if interface == "facade" else actual.time
        np.testing.assert_array_equal(start, expected["start"])


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["fit"] is not None], ids=lambda case: case["name"]
)
@pytest.mark.parametrize("interface", ["native", "formula", "facade"])
def test_large_time_complete_curves_match_stock_r(case, interface):
    if interface == "native":
        actual = survival.surv_analysis.survfitkm(case["time"], case["status"], start=case["start"])
    elif interface == "formula":
        response = "Surv(time,status)" if case["start"] is None else "Surv(start,time,status)"
        actual = r.survfit(f"{response} ~ 1", case)
    else:
        response = (
            r.Surv(case["time"], case["status"])
            if case["start"] is None
            else r.Surv(case["start"], case["time"], case["status"])
        )
        actual = r.survfit(response)
    for field, expected in case["fit"].items():
        expected = np.asarray(expected, dtype=float)
        values = np.asarray(getattr(actual, field), dtype=float)
        if field == "time" or field.startswith("n_"):
            np.testing.assert_array_equal(values, expected, err_msg=field)
        else:
            np.testing.assert_allclose(
                values, expected, rtol=1e-12, atol=1e-14, equal_nan=True, err_msg=field
            )


@pytest.mark.parametrize("interface", ["native", "facade"])
def test_large_time_normalization_retains_nonfinite_endpoints(interface):
    time = [-np.inf, 1e308, 1e308 * (1 + 1e-12), 1.2e308, np.inf, np.nan]
    if interface == "native":
        actual = survival.data_prep.aeq_surv(time).time
    else:
        actual = r.aeqSurv(r.Surv(time, [1, 0, 1, 0, 1, 0])).time
    # The finite rows use the independent huge_near_tie reference; nonfinite
    # rows retain the already documented correction of stock aeqSurv indexing.
    np.testing.assert_array_equal(actual, [-np.inf, 1e308, 1e308, 1.2e308, np.inf, np.nan])
