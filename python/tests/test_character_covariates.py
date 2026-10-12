"""Character columns are factors, as in R survival 3.8-11.

The reference values come from::

    library(survival)
    d <- lung
    d$e <- as.character(d$ph.ecog)
    d$e2 <- c("10", "2", "9", "9")[d$ph.ecog + 1]
    fit <- coxph(Surv(time, status) ~ e + age, d)
    coef(fit); fit$loglik
    predict(fit, newdata = data.frame(e = c("0", "3", "1"), age = c(50, 60, 70)))
    coef(survreg(Surv(time, status) ~ e, d))
    coef(survreg(Surv(time, status) ~ factor(e2), d))
    coef(coxph(Surv(time, status) ~ e2, d))
    names(survfit(Surv(time, status) ~ e2, d)$strata)
"""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


def _label(value, labels=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(int(value)) if labels is None else labels[int(value)]


@pytest.fixture(scope="module")
def data():
    lung = survival.datasets.load_lung()
    frame = dict(lung)
    frame["e"] = [_label(value) for value in lung["ph.ecog"]]
    frame["e2"] = [_label(value, ["10", "2", "9", "9"]) for value in lung["ph.ecog"]]
    return frame


def assert_close(actual, expected):
    np.testing.assert_allclose(np.ravel(actual), expected, rtol=1e-6)


def test_coxph_fits_a_digit_string_column_as_a_factor(data):
    fit = r.coxph("Surv(time, status) ~ e + age", data)
    assert r.coef_names(fit) == ["e1", "e2", "e3", "age"]
    assert_close(
        r.coef(fit), [0.358722465165305, 0.856800881209799, 2.10733112782226, 0.0107905056580584]
    )
    assert_close(fit.loglik, [-744.48045576144, -734.589491162309])
    predicted = r.predict(fit, newdata={"e": ["0", "3", "1"], "age": [50.0, 60.0, 70.0]})
    assert_close(predicted, [-0.134429735687177, 2.08080644871567, 0.440102842639296])


def test_survreg_fits_a_digit_string_column_as_a_factor(data):
    fit = r.survreg("Surv(time, status) ~ e", data)
    assert r.coef_names(fit) == ["(Intercept)", "e1", "e2", "e3"]
    assert_close(
        r.coef(fit),
        [6.31972136939675, -0.264045795267124, -0.669147899224745, -1.54903674493109],
    )


def test_string_levels_sort_as_strings(data):
    fit = r.survreg("Surv(time, status) ~ factor(e2)", data)
    assert r.coef_names(fit) == ["(Intercept)", "factor(e2)2", "factor(e2)9"]
    assert_close(r.coef(fit), [6.32039049183404, -0.264772784701399, -0.682385193519863])
    cox = r.coxph("Surv(time, status) ~ e2", data)
    assert r.coef_names(cox) == ["e22", "e29"]
    assert_close(r.coef(cox), [0.368186229286836, 0.930939309628372])
    curves = r.survfit("Surv(time, status) ~ e2", data)
    assert list(curves.strata) == ["e2=10", "e2=2", "e2=9"]


def test_newdata_numbers_match_string_levels_as_r_characters():
    heart = survival.datasets.load_heart()
    fit = r.coxph("Surv(start, stop, event) ~ age + transplant", heart)
    assert r.coef_names(fit) == ["age", "transplant1"]
    numeric = r.predict(fit, newdata={"age": [-5.0, 3.0], "transplant": [0, 1]})
    character = r.predict(fit, newdata={"age": [-5.0, 3.0], "transplant": ["0", "1"]})
    assert_close(numeric, np.ravel(character))
