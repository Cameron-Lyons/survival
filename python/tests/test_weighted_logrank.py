"""G-rho output against stock R and independent delayed-entry risk-set moments."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "weighted_logrank_reference.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_complete_weighted_logrank_output(case):
    data = {key: list(value) for key, value in REFERENCE["data"].items()}
    if case["near"]:
        data["time"][1] += 5e-10
    actual = survival.surv_analysis.survdiff(
        np.array(data["time"]),
        np.array(data["status"]),
        np.array(data["group"]),
        start=np.array(data["start"]) if case["counting"] else None,
        strata=np.array(data["stratum"]) if case["grouped"] else None,
        rho=case["rho"],
        timefix=case["timefix"],
    )
    for field, values in case["expected"].items():
        result = getattr(actual, field)
        if values is None:
            assert result is None, field
        else:
            np.testing.assert_allclose(result, values, rtol=2e-12, atol=2e-13, err_msg=field)
    if not case["counting"]:
        formula = "Surv(time,status) ~ group" + (" + strata(stratum)" if case["grouped"] else "")
        fit = survival.r.survdiff(formula, data, rho=case["rho"], timefix=case["timefix"])
        for field in ("n", "obs", "exp", "var", "chisq", "pvalue", "df"):
            values = case["expected"][field]
            if not case["grouped"] and field in {"obs", "exp"}:
                values = np.array(values)[:, 0]
            np.testing.assert_allclose(
                getattr(fit, field),
                values,
                rtol=2e-12,
                atol=2e-13,
                err_msg=field,
            )
