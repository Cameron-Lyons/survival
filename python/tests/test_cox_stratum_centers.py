"""Stratum-centered predictions retain finite means with large covariates."""

import numpy as np
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r


def _model(constants, weighted, counting):
    data = {
        "start": [0.0] * 8,
        "time": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        "status": [1, 1, 1, 1, 1, 1, 0, 0],
        "x": list(constants) * 4,
        "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "s": [-4, 19] * 4,
        "w": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
    }
    response = "Surv(start, time, status)" if counting else "Surv(time, status)"
    return r.coxph(
        f"{response} ~ x + z + strata(s)",
        data,
        init=[0.0, 0.25],
        nocenter=constants,
        weights="w" if weighted else None,
        robust=False,
        control=r.coxph_control(iter_max=0),
    )


def _newdata(constants):
    return {"x": list(reversed(constants)) * 2, "z": [2.5, -1.0, 5.0, 0.25], "s": [19, -4] * 2}


@pytest.mark.parametrize("constants", [(5e307, -5e307), (-1e308, 1e308), (1e308, -1.0)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("counting", [False, True])
def test_stratum_predictions_ignore_large_constant_covariates(constants, weighted, counting):
    ordinary = (1.0, -1.0)
    fit = _model(constants, weighted, counting)
    reference = _model(ordinary, weighted, counting)
    for newdata, expected_newdata in [(None, None), (_newdata(constants), _newdata(ordinary))]:
        for prediction_type in ("lp", "risk"):
            actual = r.predict(
                fit, newdata=newdata, type=prediction_type, reference="strata", se_fit=True
            )
            expected = r.predict(
                reference,
                newdata=expected_newdata,
                type=prediction_type,
                reference="strata",
                se_fit=True,
            )
            assert np.isfinite(actual.fit).all()
            assert np.isfinite(actual.se_fit).all()
            np.testing.assert_allclose(actual.fit, expected.fit, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(actual.se_fit, expected.se_fit, rtol=2e-12, atol=2e-12)
