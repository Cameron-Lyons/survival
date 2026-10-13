"""Cox summaries and baselines at the edge of the floating-point range, against R.

The reference values come from R survival 3.8-11::

    d <- data.frame(t = 1:10, st = 1, x = rep(1:0, each = 5),
                    z = c(0.3, 1.2, -0.5, 0.8, 0.1, -1.1, 0.4, 0.9, -0.2, 0.6))
    summary(coxph(Surv(t, st) ~ x + z, d))$conf.int
    h <- lung; h$xs <- h$age - 1e5
    basehaz(coxph(Surv(time, status) ~ xs + sex, h), centered = FALSE)$hazard
"""

import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r

SEPARATED = {
    "t": list(range(1, 11)),
    "st": [1] * 10,
    "x": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "z": [0.3, 1.2, -0.5, 0.8, 0.1, -1.1, 0.4, 0.9, -0.2, 0.6],
}


@pytest.fixture(scope="module")
def separated_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return r.coxph("Surv(t, st) ~ x + z", SEPARATED)


def test_summary_of_a_monotone_likelihood_has_an_infinite_limit(separated_fit):
    summary = r.model_summary(separated_fit)
    x, z = summary["conf_int"]
    assert x["exp(coef)"] == pytest.approx(4786492362.04304, rel=1e-4)
    assert x["lower .95"] == 0.0
    assert math.isinf(x["upper .95"])
    assert [z["exp(coef)"], z["lower .95"], z["upper .95"]] == pytest.approx(
        [0.719655829906486, 0.196483911211995, 2.63586219514738], rel=1e-5
    )
    assert "Inf" in str(r.print_summary_coxph(summary))


def test_uncentered_baseline_overflows_to_infinity():
    lung = survival.datasets.load_lung()
    data = dict(lung)
    data["xs"] = [age - 1e5 for age in lung["age"]]
    fit = r.coxph("Surv(time, status) ~ xs + sex", data)
    hazard = r.basehaz(fit, centered=False).hazard
    assert all(math.isinf(value) for value in hazard)
