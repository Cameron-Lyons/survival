"""Model frames on the bundled data sets, with ``subset``/``na.action`` applied to the formula
variables only, against R 4.5.3 / survival 3.8-12.

The data sets are plain column mappings, so every formula function accepts them directly;
``subset`` and ``na.action`` index only the columns the formula uses (R's ``model.frame``),
whatever the container; counting-process rows with ``start >= stop`` are dropped as missing.
"""

from __future__ import annotations

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
datasets = survival.datasets

LUNG_COLUMNS = [
    "inst",
    "time",
    "status",
    "age",
    "sex",
    "ph.ecog",
    "ph.karno",
    "pat.karno",
    "meal.cal",
    "wt.loss",
]


@pytest.fixture
def lung():
    return datasets.load_lung()


def _older(lung):
    return [age > 60 for age in lung["age"]]


def test_datasets_hold_only_their_columns(lung):
    assert list(lung) == LUNG_COLUMNS
    assert all(len(values) == 228 for values in lung.values())


def test_survfit_and_survdiff_drop_the_missing_ph_ecog_row(lung):
    # survfit(Surv(time, status) ~ ph.ecog, lung)$n
    assert r.survfit("Surv(time, status) ~ ph.ecog", lung).n == [63, 113, 50, 1]
    # survdiff(Surv(time, status) ~ ph.ecog, lung)
    test = r.survdiff("Surv(time, status) ~ ph.ecog", lung)
    assert test.n == [63, 113, 50, 1]
    assert test.obs == [37, 82, 44, 1]
    assert test.exp == pytest.approx(
        [54.152697018922929, 83.527564575081882, 26.147353065330211, 0.172385340664962], rel=1e-12
    )
    assert test.chisq == pytest.approx(21.962131682476, rel=1e-12)


def test_survreg_omits_the_missing_ph_ecog_row(lung):
    # survreg(Surv(time, status) ~ age + ph.ecog, lung, na.action = na.omit)
    fit = r.survreg("Surv(time, status) ~ age + ph.ecog", lung, na_action="omit")
    assert fit.coefficients == pytest.approx(
        [6.8305035534691747, -0.0077197275613747, -0.3263819844626905], rel=1e-8
    )
    assert fit.scale == pytest.approx([0.738110535588488], rel=1e-8)


def test_subset_on_the_bundled_data(lung):
    # survfit(Surv(time, status) ~ sex, lung, subset = age > 60)$n
    assert r.survfit("Surv(time, status) ~ sex", lung, subset=_older(lung)).n == [89, 45]
    # pyears(Surv(time, status) ~ sex, lung, subset = age > 60, scale = 1)
    table = r.pyears("Surv(time, status) ~ sex", lung, subset=_older(lung), scale=1)
    assert table.pyears == [22988.0, 16075.0]
    assert table.n == [89.0, 45.0]
    assert table.event == [73.0, 28.0]
    # survreg(Surv(futime, fustat) ~ ecog.ps + rx, ovarian, subset = 1:20)
    fit = r.survreg(
        "Surv(futime, fustat) ~ ecog.ps + rx", datasets.load_ovarian(), subset=range(20)
    )
    assert fit.coefficients == pytest.approx(
        [7.770793695010096, -0.698536084713902, 0.429722284712473], rel=1e-7
    )
    assert fit.scale == pytest.approx([0.946123455677063], rel=1e-7)


def test_dot_formulas_expand_to_the_data_columns():
    aml = datasets.load_aml()
    # survreg(Surv(time, status) ~ ., aml)
    fit = r.survreg("Surv(time, status) ~ .", aml)
    assert list(fit.coefficient_names) == ["(Intercept)", "xNonmaintained"]
    assert fit.coefficients == pytest.approx([4.109055051262132, -0.929341611365523], rel=1e-8)
    assert fit.scale == pytest.approx([0.790954436951233], rel=1e-8)
    # survfit(Surv(time, status) ~ ., aml)
    curves = r.survfit("Surv(time, status) ~ .", aml)
    assert curves.n == [11, 12]
    assert curves.strata_names == ["x=Maintained", "x=Nonmaintained"]


def test_survsplit_dot_keeps_r_column_order(lung):
    # survSplit(Surv(time, status) ~ ., lung, cut = 100)
    split = r.survSplit("Surv(time, status) ~ .", lung, cut=[100])
    assert list(split) == [*LUNG_COLUMNS, "tstart"]
    assert len(split["time"]) == 424
    # ... na.action = na.omit: the model frame's columns, then the new time columns
    split = r.survSplit("Surv(time, status) ~ .", lung, cut=[100], na_action="omit")
    assert list(split) == [
        "inst",
        "age",
        "sex",
        "ph.ecog",
        "ph.karno",
        "pat.karno",
        "meal.cal",
        "wt.loss",
        "tstart",
        "time",
        "status",
    ]
    assert len(split["time"]) == 310


def test_survsplit_character_id_follows_the_subset():
    # R cannot add the id column to a subset (its 1:nrow(data) has the wrong length); the
    # port numbers the rows of data and keeps those of the subset: rows 3 and 4 of aml
    # (times 13+ and 18) split at 10
    aml = datasets.load_aml()
    split = r.survSplit("Surv(time, status) ~ x", aml, cut=[10], id="id", subset=[2, 3])
    assert split["id"] == [3, 3, 4, 4]
    assert split["time"] == [10.0, 13.0, 10.0, 18.0]


def test_documented_survexp_example(lung):
    # docs/r-compatibility.md's example, against R's
    # survexp(~ sex, lung, ratetable = cox, times = c(0, 100, 365))$surv
    cox = r.coxph("Surv(time, status) ~ age + sex", lung, model=True)
    expected = r.survexp("~ sex", lung, ratetable=cox, times=[0, 100, 365])
    assert expected.strata == ["sex=1", "sex=2"]
    assert [row[0] for row in expected.surv] == pytest.approx(
        [1.0, 0.837068860129983, 0.334645175117436], rel=1e-9
    )
    assert [row[1] for row in expected.surv] == pytest.approx(
        [1.0, 0.902619962241841, 0.530666050168176], rel=1e-9
    )
    assert expected.n_risk == [[138.0, 90.0]] * 3


def test_polars_frames_with_missing_values_and_subset(lung):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(lung)
    # coxph(Surv(time, status) ~ ph.ecog, lung)
    fit = r.coxph("Surv(time, status) ~ ph.ecog", frame, na_action="omit")
    assert fit.n == 227
    assert fit.coefficients == pytest.approx([0.47594344950965], rel=1e-9)
    assert r.survfit("Surv(time, status) ~ sex", frame, subset=_older(lung)).n == [89, 45]


def test_pandas_factor_levels_survive_row_removal():
    pd = pytest.importorskip("pandas")
    # aml$x <- factor(aml$x, levels = c("Nonmaintained", "Maintained")); aml$x[3] <- NA
    frame = pd.DataFrame(datasets.load_aml())
    frame["x"] = pd.Categorical(frame["x"], categories=["Nonmaintained", "Maintained"])
    frame.loc[2, "x"] = None
    curves = r.survfit("Surv(time, status) ~ x", frame)
    assert curves.n == [12, 10]
    assert curves.strata_names == ["x=Nonmaintained", "x=Maintained"]
    fit = r.coxph("Surv(time, status) ~ x", frame, na_action="omit")
    assert list(fit.coef_names) == ["xMaintained"]
    assert fit.coefficients == pytest.approx([-0.876828103972665], rel=1e-9)


def test_columns_outside_the_formula_are_not_touched(lung):
    data = {**lung, "meta": [1, 2]}
    assert r.survfit("Surv(time, status) ~ ph.ecog", data).n == [63, 113, 50, 1]
    fit = r.coxph("Surv(time, status) ~ ph.ecog", data, na_action="omit")
    assert fit.coefficients == pytest.approx([0.47594344950965], rel=1e-9)


def test_formula_columns_must_have_the_response_length(lung):
    data = {**lung, "short": lung["age"][:-1]}
    with pytest.raises(ValueError, match="variable lengths differ \\(found for 'short'\\)"):
        r.survfit("Surv(time, status) ~ short", data, subset=_older(lung))


def test_underscore_columns_are_ordinary_columns(lung):
    # d$`_x` <- d$age; coxph(Surv(time, status) ~ `_x`, d)
    fit = r.coxph("Surv(time, status) ~ `_x`", {**lung, "_x": lung["age"]})
    assert fit.coefficients == pytest.approx([0.0187201792045515], rel=1e-9)


# start = c(0, 2, 5, 1), stop = c(10, 12, 6, 1): R's Surv makes the last start NA, and
# model.frame's na.omit drops that row (na.action = 4).
BACKWARDS = {
    "start": [0, 2, 5, 1],
    "stop": [10, 12, 6, 1],
    "status": [1, 0, 1, 1],
    "x": [0.5, 1.2, 2.0, 0.3],
    "sex": [1, 2, 1, 2],
}


def test_start_not_before_stop_rows_are_missing():
    with pytest.warns(UserWarning, match="Stop time must be > start time, NA created"):
        fit = r.coxph("Surv(start, stop, status) ~ x", BACKWARDS, na_action="omit")
    # coxph(Surv(start, stop, status) ~ x, d)
    assert fit.n == 3
    assert fit.coefficients == pytest.approx([0.874234772123199], rel=1e-9)
    assert fit.loglik == pytest.approx([-1.79175946922805, -1.61414151920695], rel=1e-12)

    with pytest.warns(UserWarning, match="Stop time must be > start time, NA created"):
        curve = r.survfit("Surv(start, stop, status) ~ 1", BACKWARDS)
    # survfit(Surv(start, stop, status) ~ 1, d)
    assert curve.n == [3]
    assert curve.time == [6.0, 10.0, 12.0]
    assert curve.n_risk == [3.0, 2.0, 1.0]
    assert curve.surv == pytest.approx([2 / 3, 1 / 3, 1 / 3], rel=1e-12)

    with pytest.warns(UserWarning, match="Stop time must be > start time, NA created"):
        table = r.pyears("Surv(start, stop, status) ~ sex", BACKWARDS, scale=1)
    # pyears(Surv(start, stop, status) ~ sex, d, scale = 1)
    assert table.pyears == [11.0, 10.0]
    assert table.n == [2.0, 1.0]
    assert table.event == [2.0, 0.0]

    with (
        pytest.warns(UserWarning, match="Stop time must be > start time"),
        pytest.raises(ValueError, match="missing values"),
    ):
        r.coxph("Surv(start, stop, status) ~ x", BACKWARDS, na_action="fail")
