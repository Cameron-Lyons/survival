"""Expected cumulative hazards retain R's logarithm and exhausted-group behavior."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/expected_cumhaz_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["synthetic"], ids=lambda case: case["name"])
def test_expected_cumulative_hazards_match_r_logarithm(case):
    values = np.asarray(case["surv"], dtype=float)
    fit = r.SurvExpResult(
        list(range(len(values))), values.tolist(), np.ones_like(values).tolist(), "cohort", 1
    )
    for result in (fit, pickle.loads(pickle.dumps(fit))):  # noqa: S301
        np.testing.assert_allclose(result.cumhaz, np.asarray(case["cumhaz"], dtype=float))
        # Cumulative hazard snapshots do not mutate the survival curve or each other.
        hazard = result.cumhaz
        if values.ndim == 1:
            hazard[0] = 999
        else:
            hazard[0][0] = 999
        np.testing.assert_allclose(result.surv, values)
        np.testing.assert_allclose(result.cumhaz, np.asarray(case["cumhaz"], dtype=float))


@pytest.mark.parametrize("case", REFERENCE["curves"], ids=lambda case: case["method"])
def test_exhausted_cox_groups_retain_missing_expected_hazards(case):
    source = survival.datasets.load_lung()
    keep = [i for i, value in enumerate(source["ph.ecog"]) if value is not None and value == value]
    data = {
        name: [source[name][i] for i in keep]
        for name in ("time", "status", "ph.ecog", "age", "sex")
    }
    model = r.coxph("Surv(time, status) ~ age + sex", data)
    fit = r.survexp(
        "Surv(time, status) ~ ph.ecog",
        data,
        ratetable=model,
        method=case["method"],
        times=case["time"],
    )
    expected_surv = np.asarray(case["surv"], dtype=float)
    expected_hazard = np.asarray(case["cumhaz"], dtype=float)
    assert fit.strata == case["labels"]
    np.testing.assert_allclose(fit.surv, expected_surv, rtol=2e-12)
    np.testing.assert_allclose(fit.cumhaz, expected_hazard, rtol=2e-12)
    # ph.ecog=3 has no subject at risk after time 118; missing values persist
    # through time queries and full-precision report tables.
    summary = r.summary_survexp(fit, times=case["time"])
    np.testing.assert_allclose(summary.surv, expected_surv, rtol=2e-12)
    for report in (r.print_survexp(fit), r.print_summary_survexp(summary)):
        np.testing.assert_allclose(
            np.asarray(report.tables[0].values)[:, -4:], expected_surv, rtol=2e-12
        )
