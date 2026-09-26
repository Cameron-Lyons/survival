"""R's S3 methods reached through the ``survival.r`` generics: dispatch on concordance,
survfit, pyears and aareg objects, ``model_frame`` on a formula or a fit, ``as_data_frame``
on the data-frame results, and the survreg object without a fall-through to the Rust fit.

Reference values come from R 4.5.3 with survival 3.8-12.
"""

from __future__ import annotations

import copy

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


@pytest.fixture(scope="module")
def lung():
    data = survival.datasets.load_lung()
    return {name: values for name, values in data.items() if not name.startswith("_")}


def _small():
    return {
        "id": [1, 2, 3, 4],
        "time": [5.0, 8.0, 12.0, 20.0],
        "status": [1, 0, 1, 1],
        "grp": ["a", "b", "a", "b"],
        "age": [50.0, 61.0, 70.0, 45.0],
    }


def test_coef_and_vcov_of_a_concordance(lung):
    # cf <- concordance(Surv(time, status) ~ age, lung); coef(cf); vcov(cf)
    fit = r.concordance("Surv(time, status) ~ age", lung)
    assert r.coef(fit) == pytest.approx(0.4497601678825, rel=1e-10)
    assert r.vcov(fit) == pytest.approx(0.0006321257754, rel=1e-9)


def test_residuals_and_summary_of_a_survfit(lung):
    km = r.survfit("Surv(time, status) ~ sex", lung)
    # the first row of R's residuals(km, times = c(100, 300))
    resid = r.residuals(km, times=[100, 300])
    assert len(resid.resid) == 228
    assert resid.resid[0] == pytest.approx([0.001260239445, 0.004376894225], rel=1e-9)
    assert resid == r.survfit_residuals(km, times=[100, 300])
    # summary(km)$table[, "median"]
    table = r.model_summary(km).table
    assert [row[table.colnames.index("median")] for row in table.values] == [270.0, 426.0]
    # survfit.coxph curves have no residuals method in R either
    with pytest.raises(TypeError, match="residuals method for coxph survival curve not found"):
        r.residuals(r.survfit(r.coxph("Surv(time, status) ~ age", lung)))


def test_summary_of_a_pyears_table():
    # summary(pyears(Surv(time, status) ~ grp + cut(age, c(40, 60, 80)), d, scale = 1),
    #         rate = TRUE): event rate a (0.2, 1/12), b (0.05, 0)
    table = r.pyears("Surv(time, status) ~ grp + cut(age, c(40, 60, 80))", _small(), scale=1)
    summary = r.model_summary(table, rate=True)
    assert summary == r.summary_pyears(table, rate=True)
    assert summary.rate[0] == pytest.approx([0.2, 1 / 12])
    assert summary.rate[1] == pytest.approx([0.05, 0.0])


def test_term_labels_of_an_aareg_fit(lung):
    fit = r.aareg("Surv(time, status) ~ age + sex", lung)
    assert r.model_term_names(fit) == ["age", "sex"]
    assert r.model_term_names(fit, [2]) == ["sex"]


def test_model_frame_of_a_formula_and_of_a_fit_without_model(lung):
    # dim(model.frame(Surv(time, status) ~ age, lung)): 228 x 2
    frame = r.model_frame("Surv(time, status) ~ age", lung)
    assert list(frame) == ["time", "status", "age"]
    assert len(frame["time"]) == 228
    subset = r.model_frame("Surv(time, status) ~ age", lung, subset=list(range(10)))
    assert len(subset["age"]) == 10
    # model.frame(Surv(time, status) ~ ph.ecog, lung): 227 rows, na.omit drops one
    assert len(r.model_frame("Surv(time, status) ~ ph.ecog", lung)["ph.ecog"]) == 227
    # model.frame(coxph(Surv(time, status) ~ age + sex, lung)): 228 rows of the response,
    # age and sex, rebuilt although the fit kept no model
    fit = r.coxph("Surv(time, status) ~ age + sex", lung)
    assert fit.model is None
    rebuilt = r.model_frame(fit)
    assert list(rebuilt) == ["time", "status", "age", "sex"]
    assert len(rebuilt["time"]) == 228
    assert rebuilt["age"][:3] == lung["age"][:3]
    with pytest.raises(TypeError, match="model=TRUE"):
        r.model_frame(r.survreg("Surv(time, status) ~ age", lung))
    with pytest.raises(TypeError, match="requires a formula or a fitted model"):
        r.model_frame(object())


def test_as_data_frame_of_tmerge_survsplit_and_pyears():
    d = _small()
    base = {name: d[name] for name in ("id", "grp")}
    # R: tm <- tmerge(d[, c("id", "grp")], d, id = id, death = event(time, status)),
    # then tmerge(tm, data.frame(id = c(1, 3), t = c(2, 6)), id = id, trt = tdc(t))
    tm = r.tmerge(base, d, id="id", death=r.event("time", "status"))
    tm = r.tmerge(tm, {"id": [1, 3], "t": [2.0, 6.0]}, id="id", trt=r.tdc("t"))
    frame = r.as_data_frame(tm)
    assert list(frame) == ["id", "grp", "tstart", "tstop", "death", "trt"]
    assert frame["tstop"] == [2.0, 5.0, 8.0, 6.0, 12.0, 20.0]
    assert frame["death"] == [0, 1, 0, 0, 1, 1]
    assert frame["trt"] == [0, 1, 0, 0, 1, 0]
    # survSplit(Surv(time, status) ~ ., d, cut = c(6, 10), episode = "ep")
    split = r.as_data_frame(r.survSplit("Surv(time, status) ~ .", d, cut=[6, 10], episode="ep"))
    assert split["time"] == [5.0, 6.0, 8.0, 6.0, 10.0, 12.0, 6.0, 10.0, 20.0]
    assert split["ep"] == [1, 1, 2, 1, 2, 3, 1, 2, 3]
    # pyears(Surv(time, status) ~ grp + cut(age, c(40, 60, 80)), d, scale = 1,
    #        data.frame = TRUE)$data; as_data_frame of the table gives the same layout
    formula = "Surv(time, status) ~ grp + cut(age, c(40, 60, 80))"
    expected = {
        "grp": ["a", "b", "a", "b"],
        "cut(age, c(40, 60, 80))": ["(40,60]", "(40,60]", "(60,80]", "(60,80]"],
        "pyears": [5.0, 20.0, 12.0, 8.0],
        "n": [1.0, 1.0, 1.0, 1.0],
        "event": [1.0, 1.0, 1.0, 0.0],
    }
    assert r.as_data_frame(r.pyears(formula, d, scale=1)) == expected
    assert r.as_data_frame(r.pyears(formula, d, scale=1, data_frame=True)) == expected


def test_as_data_frame_keeps_only_cells_with_follow_up():
    # the table has an empty cell: b x (60,80] once subject 2 is 45
    d = _small()
    d["age"] = [50.0, 45.0, 70.0, 45.0]
    frame = r.as_data_frame(r.pyears("Surv(time, status) ~ grp + cut(age, c(40, 60, 80))", d))
    assert frame["grp"] == ["a", "b", "a"]
    assert frame["cut(age, c(40, 60, 80))"] == ["(40,60]", "(40,60]", "(60,80]"]


def test_survreg_object_copies_and_reads_r_components(lung):
    fit = r.survreg("Surv(time, status) ~ age + sex", lung)
    clone = copy.copy(fit)
    assert clone.coefficients == fit.coefficients
    # survreg(Surv(time, status) ~ age + sex, lung): scale, iter, df.residual
    assert fit.scale == pytest.approx([0.754050947641], rel=1e-9)
    assert (fit.iter, fit.df, fit.df_residual, fit.n) == (5, 4, 224, 228)
    assert (fit.weights, fit.score, fit.model) == (None, None, None)
    with pytest.raises(AttributeError):
        _ = fit.status  # the Rust fit's attributes are not R components


def test_as_data_frame_rejects_mappings_that_are_not_columns(lung):
    km = r.survfit("Surv(time, status) ~ 1", lung)
    for mapping in ({"x": "abc"}, {"x": 1.0}, {"curve": km, "x": [1.0]}):
        with pytest.raises(TypeError, match="requires a survival result object"):
            r.as_data_frame(mapping)


def test_multistate_curve_residuals_and_summary():
    aj = r.survfit("Surv(time, status, type = 'mstate') ~ 1", _multistate())
    assert r.model_summary(aj) == r.summary_survfit(aj)
    assert r.residuals(aj, times=[2.5, 5.5]) == r.survfit_residuals(aj, times=[2.5, 5.5])


def test_reprs_stay_short(lung):
    km = r.survfit("Surv(time, status) ~ sex", lung)
    cox = r.coxph("Surv(time, status) ~ age + sex", lung)
    for result in (
        km,
        r.survfit("Surv(time, status, type = 'mstate') ~ 1", _multistate()),
        r.survfit(cox),
        r.cox_zph(cox),
        r.survreg("Surv(time, status) ~ age + sex", lung),
    ):
        assert len(repr(result)) < 600


def _multistate():
    return {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": ["censor", "a", "b", "a", "censor", "b"],
    }
