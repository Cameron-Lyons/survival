"""``survival.r.cox_zph`` regressions against R survival 3.8-12.

Reference values were computed with R 4.5.3 and survival 3.8-12 on the bundled
datasets; the R calls are quoted above each test.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def _complete(data, *columns):
    """The rows of a bundled dataset with no missing value in ``columns``."""

    keep = [i for i in range(data["_nrow"]) if all(data[name][i] is not None for name in columns)]
    return {
        name: [values[i] for i in keep] for name, values in data.items() if isinstance(values, list)
    }


def _column(zph, key):
    return [row[key] for row in zph.table]


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


# fit <- coxph(Surv(time, status) ~ pspline(age, df = 4) + sex, lung)
# z <- cox.zph(fit); z$table; z$var; head(z$y, 3); tail(z$y, 2)
# cox.zph(fit, terms = FALSE)$table; cox.zph(fit, singledf = TRUE)$table
def test_pspline_fit_adds_the_penalty_and_takes_df_from_the_fit(lung):
    fit = r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", lung)
    zph = r.cox_zph(fit)
    assert _column(zph, "chisq") == pytest.approx(
        [0.862967843078756, 2.641625114940909, 3.395525657145105], rel=1e-9
    )
    assert _column(zph, "df") == pytest.approx(
        [4.092186074265781, 0.996457053487749, 5.088643127753530], rel=1e-12
    )
    assert _column(zph, "p") == pytest.approx(
        [0.935465336477770, 0.103604575277078, 0.650950737581564], rel=1e-9
    )
    assert zph.var[0] == pytest.approx([25.616912564562714, 0.467336747008603], rel=1e-9)
    assert zph.var[1] == pytest.approx([0.467336747008603, 4.623762279674438], rel=1e-9)
    assert zph.y[:3] == [
        pytest.approx([-1.65441589278317, 2.79853575089749], rel=1e-9),
        pytest.approx([6.02806810670399, -1.67647850302293], rel=1e-9),
        pytest.approx([19.58367777628894, -1.42917959411906], rel=1e-9),
    ]
    assert zph.y[-2:] == [
        pytest.approx([-0.744274780436742, -1.39163374044771], rel=1e-9),
        pytest.approx([-2.800366600041335, -1.29186292089168], rel=1e-9),
    ]
    assert r.as_data_frame(zph)["df"] == _column(zph, "df")

    # R indexes fit$df by position, so past its two entries the df are NA
    by_coefficient = r.cox_zph(fit, terms=False)
    assert _column(by_coefficient, "chisq")[:3] == pytest.approx(
        [0.000534234624748221, 0.004658672921705095, 0.003635278625629485], rel=1e-8
    )
    assert _column(by_coefficient, "df")[:2] == pytest.approx(
        [4.092186074265781, 0.996457053487749], rel=1e-12
    )
    assert all(math.isnan(df) for df in _column(by_coefficient, "df")[2:-1])
    assert all(math.isnan(p) for p in _column(by_coefficient, "p")[2:-1])
    assert by_coefficient.table[-1]["chisq"] == pytest.approx(3.395525657145105, rel=1e-9)

    single = r.cox_zph(fit, singledf=True)
    assert _column(single, "chisq")[0] == pytest.approx(12.64108508721849, rel=1e-9)
    assert _column(single, "df") == pytest.approx(
        [1.0, 0.996457053487749, 5.088643127753530], rel=1e-12
    )
    assert _column(single, "p")[0] == pytest.approx(0.000377360883008123, rel=1e-8)


# R reads a diagonal-only coxlist2$second with matrix(second, nvar), which
# recycles it into pmat[i, j] = second[i] (R prints 3.548 for the first fit);
# these references come from cox.zph with that line replaced by
#   tmat <- if (length(second) == nc) diag(second, nc) else matrix(second, nc)
# fit <- coxph(Surv(time, status) ~ ridge(age, sex, theta = 1), lung)
# fit2 <- coxph(Surv(time, status) ~ ridge(age, ph.ecog, theta = 1) + sex,
#               lung[!is.na(lung$ph.ecog), ])
def test_ridge_penalty_enters_as_its_diagonal(lung):
    zph = r.cox_zph(r.coxph("Surv(time, status) ~ ridge(age, sex, theta=1)", lung))
    assert _column(zph, "chisq") == pytest.approx([2.51379658992081, 2.51379658992081], rel=1e-9)
    assert _column(zph, "df") == pytest.approx([1.98637239443218, 1.98637239443218], rel=1e-12)
    assert _column(zph, "p") == pytest.approx([0.28198738697614, 0.28198738697614], rel=1e-9)
    assert zph.var == [pytest.approx([12.3127009972655], rel=1e-9)]
    assert [row[0] for row in zph.y[:3]] == pytest.approx(
        [-3.36327356943977, 4.80259598173648, 6.26239090873465], rel=1e-9
    )

    complete = _complete(lung, "ph.ecog")
    zph = r.cox_zph(r.coxph("Surv(time, status) ~ ridge(age, ph.ecog, theta=1) + sex", complete))
    assert _column(zph, "chisq") == pytest.approx(
        [1.91496358769609, 2.30561880556803, 4.28765358647820], rel=1e-9
    )
    assert _column(zph, "df") == pytest.approx(
        [1.986351277488045, 0.999984198713933, 2.986335476201978], rel=1e-12
    )
    assert zph.var[0] == pytest.approx([8.089068811324143, -0.283527210244434], rel=1e-9)
    assert zph.var[1] == pytest.approx([-0.283527210244434, 4.613112064330112], rel=1e-9)


# cox.zph(coxph(Surv(time, status) ~ age + sex + frailty(id), kidney))$table
# lung2 <- lung[!is.na(lung$inst), ]
# cox.zph(coxph(Surv(time, status) ~ pspline(age, df = 3) + sex + frailty(inst), lung2))$table
# R reads fit$df by position among the tested terms and accepts only fits
# whose frailty is the last term ("subscript out of bounds" otherwise); any
# other order must give the same table.
@pytest.mark.parametrize(
    ("kidney_terms", "lung_terms"),
    [
        ("age + sex + frailty(id)", "pspline(age, df=3) + sex + frailty(inst)"),
        ("age + frailty(id) + sex", "pspline(age, df=3) + frailty(inst) + sex"),
        ("frailty(id) + age + sex", "frailty(inst) + pspline(age, df=3) + sex"),
    ],
)
def test_sparse_frailty_fits_take_df_from_the_fit(lung, kidney_terms, lung_terms):
    zph = r.cox_zph(r.coxph(f"Surv(time, status) ~ {kidney_terms}", datasets.load_kidney()))
    assert [row["name"] for row in zph.table] == ["age", "sex", "GLOBAL"]
    assert _column(zph, "chisq") == pytest.approx(
        [0.0596639502331169, 2.6451729058459819, 2.8523338766384652], rel=1e-9
    )
    assert _column(zph, "df") == pytest.approx(
        [0.547122575511723, 0.584133011049957, 14.144580723483331], rel=1e-9
    )
    assert _column(zph, "p") == pytest.approx(
        [0.5784717733961282, 0.0517397893344987, 0.9993869328697095], rel=1e-9
    )

    # With a sparse term coxpenal.fit keeps no coxlist2: the dense pspline
    # penalty is not added to the information.
    zph = r.cox_zph(r.coxph(f"Surv(time, status) ~ {lung_terms}", _complete(lung, "inst")))
    assert _column(zph, "chisq") == pytest.approx(
        [91.63862907179184, 2.91898671262468, 91.81326620248043], rel=1e-9
    )
    assert _column(zph, "df") == pytest.approx(
        [3.000071304587040, 0.997107306576604, 3.997178850816106], rel=1e-9
    )


# lung$big <- lung$age + 1e6; heart$big <- heart$age + 1e6
# cox.zph(coxph(Surv(time, status) ~ big + sex, lung))$table
# cox.zph(coxph(Surv(start, stop, event) ~ big + surgery, heart))$table
def test_a_large_covariate_mean_does_not_lose_precision(lung):
    data = dict(lung, big=[age + 1e6 for age in lung["age"]])
    zph = r.cox_zph(r.coxph("Surv(time, status) ~ big + sex", data))
    assert _column(zph, "chisq") == pytest.approx(
        [0.209202826364685, 2.607671703535354, 2.770785375936977], rel=1e-10
    )
    heart = datasets.load_heart()
    data = dict(heart, big=[age + 1e6 for age in heart["age"]])
    zph = r.cox_zph(r.coxph("Surv(start, stop, event) ~ big + surgery", data))
    assert _column(zph, "chisq") == pytest.approx(
        [0.895379349131291, 0.098297108901689, 0.993221043716714], rel=1e-10
    )


def sqrt_times(times):
    return [math.sqrt(t) for t in times]


# sqrt_times <- function(t) sqrt(t)
# z <- cox.zph(coxph(Surv(time, status) ~ age + sex, lung), transform = sqrt_times)
# z2 <- cox.zph(coxph(Surv(start, stop, event) ~ age + surgery, heart), transform = sqrt_times)
def test_a_function_transform_is_applied_to_the_stop_times(lung):
    fit = r.coxph("Surv(time, status) ~ age + sex", lung)
    zph = r.cox_zph(fit, transform=sqrt_times)
    assert zph.transform == "sqrt_times"
    assert _column(zph, "chisq") == pytest.approx(
        [0.956743463637213, 2.750195636772541, 3.587111850001905], rel=1e-10
    )
    assert zph.x[:3] == pytest.approx([2.23606797749979, 3.31662479035540, 3.31662479035540])

    heart_fit = r.coxph("Surv(start, stop, event) ~ age + surgery", datasets.load_heart())
    zph = r.cox_zph(heart_fit, transform=sqrt_times)
    assert _column(zph, "chisq") == pytest.approx(
        [1.276528694391162, 0.808121874639506, 2.017484834024878], rel=1e-10
    )
    assert zph.x[:3] == pytest.approx([1.0, math.sqrt(2.0), math.sqrt(2.0)])
    assert zph.time[:3] == [1.0, 2.0, 2.0]

    anonymous = r.cox_zph(fit, transform=lambda times: times)
    assert anonymous.transform == "user"
    assert anonymous.table == r.cox_zph(fit, transform="identity").table
    with pytest.raises(ValueError, match="one finite value per observation"):
        r.cox_zph(fit, transform=lambda times: times[1:])
    with pytest.raises(ValueError, match="one finite value per observation"):
        r.cox_zph(fit, transform=lambda times: [math.inf for _ in times])
    with pytest.raises(TypeError, match="or a function"):
        r.cox_zph(fit, transform=2)


# d <- lung[!is.na(lung$ph.karno), ]; d$wt <- d$ph.karno / 100; d$age2 <- d$age * s
# cox.zph(coxph(Surv(time, status) ~ age2 + wt + sex, d))$table["GLOBAL", ]
# gives chisq 10.3589845408429 on 3 df (p 0.0157486420131667) for every s up to
# 1e5; at s = 1e6 R's solve() stops: the system is computationally singular
# (reciprocal condition number 1.85494e-17).
@pytest.mark.parametrize("scale", [1.0, 1e2, 1e3, 1e4, 1e5])
def test_badly_scaled_covariates_follow_r_solve_rule(lung, scale):
    data = _complete(lung, "ph.karno")
    data["wt"] = [karno / 100 for karno in data["ph.karno"]]
    data["age2"] = [age * scale for age in data["age"]]
    zph = r.cox_zph(r.coxph("Surv(time, status) ~ age2 + wt + sex", data))
    assert zph.table[-1]["chisq"] == pytest.approx(10.3589845408429, rel=1e-10)
    assert zph.table[-1]["p"] == pytest.approx(0.0157486420131667, rel=1e-10)


def test_a_computationally_singular_information_matrix_is_an_error(lung):
    data = _complete(lung, "ph.karno")
    data["wt"] = [karno / 100 for karno in data["ph.karno"]]
    data["age2"] = [age * 1e6 for age in data["age"]]
    fit = r.coxph("Surv(time, status) ~ age2 + wt + sex", data)
    with pytest.raises(RuntimeError, match=r"condition number = 1\.85494e-17\): .* singular"):
        r.cox_zph(fit)
