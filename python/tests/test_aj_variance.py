"""Aalen-Johansen uncertainty against stock R, including independent and fallback paths."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
sa = survival.surv_analysis
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/aj_variance_reference.json").read_text())
MATRIX_FIELDS = [
    "n_risk",
    "n_event",
    "n_censor",
    "n_transition",
    "pstate",
    "cumhaz",
    "std_err",
    "std_chaz",
    "std_auc",
    "lower",
    "upper",
    "p0",
]


def fit_case(case, api, influence):
    source = case["input"]
    options = {**case["options"], "influence": influence}
    if api == "native":
        return sa.survfitaj(**source, **options)
    data = {
        "time": source["time"],
        "event": r._r_factor(
            [["censor", *source["states"]][code] for code in source["state"]],
            ["censor", *source["states"]],
        ),
        "weight": source["weights"],
    }
    if source["istate"] is not None:
        data["initial"] = source["istate"]
        options["istate"] = "initial"
    if source["strata"] is not None:
        data["g"] = source["strata"]
    formula = "Surv(time, event) ~ " + ("g" if source["strata"] is not None else "1")
    return r.survfit(formula, data, weights="weight", **options)


@pytest.mark.parametrize("api", ["native", "formula"])
@pytest.mark.parametrize("influence", [False, True], ids=["standard_errors", "full_influences"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_aj_point_values_and_all_uncertainty_match_stock_r(case, api, influence):
    fit = fit_case(case, api, influence)
    expected = case["expected"]
    assert fit.states == expected["states"]
    assert fit.n == expected["n"]
    assert fit.n_id == expected["n_id"]
    assert fit.t0 == expected["t0"]
    np.testing.assert_array_equal(fit.time, expected["time"])
    if api == "native":
        hazards = [
            f"{origin + 1}:{target + 1}"
            for origin, target in zip(fit.hazard_from, fit.hazard_to, strict=True)
        ]
    else:
        hazards = fit.hazard_names
    assert hazards == expected["hazard_names"]
    for name in MATRIX_FIELDS:
        actual = getattr(fit, name)
        reference = expected[name]
        if reference is None:
            assert actual is None, name
        else:
            np.testing.assert_allclose(
                np.asarray(actual, dtype=float),
                np.asarray(reference, dtype=float),
                rtol=3e-12,
                atol=2e-14,
                err_msg=name,
            )


@pytest.mark.parametrize("api", ["native", "formula"])
def test_empty_weighted_risk_set_retains_r_undefined_uncertainty(api):
    cases = {case["name"]: case for case in REFERENCE["cases"]}
    events = fit_case(cases["zero_weight_event_tail"], api, False)
    censors = fit_case(cases["zero_weight_censor_tail"], api, False)
    np.testing.assert_array_equal(events.pstate, censors.pstate)
    assert np.isfinite(events.pstate).all()
    assert np.isnan(events.std_err[-1]).all()
    assert np.isnan(events.std_chaz[-1]).all()
    assert np.isfinite(censors.std_err).all()
    assert np.isfinite(censors.std_chaz).all()
