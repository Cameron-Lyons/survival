"""All summary-table fields agree with independently computed stock-R tables."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

sa = setup_survival_import().surv_analysis
REFERENCE = json.loads((Path(__file__).parent / "fixtures/survmean_reference.json").read_text())


@pytest.fixture(scope="module")
def fits():
    return {
        name: sa.SurvfitKMResult.from_stacked(
            source["time"],
            source["n_risk"],
            source["n_event"],
            [np.nan if value is None else value for value in source["surv"]],
            source["n"],
            strata=source["strata"],
            n_id=source["n_id"],
            t0=source["t0"],
            lower=None
            if source["lower"] is None
            else [np.nan if value is None else value for value in source["lower"]],
            upper=None
            if source["upper"] is None
            else [np.nan if value is None else value for value in source["upper"]],
        )
        for name, source in REFERENCE["sources"].items()
    }


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survmean_table_matches_current_stock_r(case, fits):
    table = sa.survmean(fits[case["source"]], scale=case["scale"], rmean=str(case["rmean"]))
    for field, values in case["expected"].items():
        actual = getattr(table, field)
        if values is None:
            assert actual is None, field
        else:
            np.testing.assert_allclose(
                actual,
                np.asarray(values, dtype=float),
                rtol=2e-12,
                atol=1e-14,
                equal_nan=True,
                err_msg=field,
            )
