"""Cox individual curves with interleaved subjects and changing strata, from R."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "cox_individual_reference.json").read_text()
)


@pytest.fixture(scope="module")
def fit():
    return r.coxph(
        "Surv(start, stop, event) ~ x + z + strata(g) + offset(offset)",
        REFERENCE["data"],
        weights="weight",
        ties="efron",
        robust=False,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_interleaved_subjects_and_changing_strata_match_r(fit, case):
    curve = r.survfit(
        fit,
        REFERENCE["newdata"],
        id="id",
        stype=case["stype"],
        ctype=case["ctype"],
        se_fit=case["se_fit"],
    )
    for field, expected in case["expected"].items():
        actual = getattr(curve, field.replace(".", "_"))
        if field == "strata":
            # First appearance, including nonconsecutive rows for each subject.
            assert list(actual.items()) == list(expected.items())
        else:
            np.testing.assert_allclose(
                actual, np.asarray(expected, dtype=float), rtol=2e-10, atol=2e-12
            )
    if not case["se_fit"]:
        assert curve.std_err is None
        assert curve.lower is None
        assert curve.upper is None
