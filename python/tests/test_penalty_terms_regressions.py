"""The ``ridge()``, ``pspline()`` and ``frailty()`` formula terms, against R 4.5.3 with
survival 3.8-12.

R evaluates these functions in ``model.frame``, before ``subset`` and ``na.action``: the
pspline knots, the ridge variances and the frailty groups, ``sparse`` default and df
search come from all the rows.  The coefficients are named as R names them, pspline takes
``combine`` and ``penalty = FALSE``, and survreg refuses penalized terms instead of fitting
them unpenalized.
"""

import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.fixture(scope="module")
def lung_na(lung):
    # lung with wt.loss missing and age 30 in the first row
    data = {key: list(values) for key, values in lung.items()}
    data["wt.loss"][0] = math.nan
    data["age"][0] = 30
    return data


def _kidney(max_id):
    kidney = datasets.load_kidney()
    keep = [value <= max_id for value in kidney["id"]]
    return {
        key: [v for v, k in zip(values, keep, strict=True) if k] for key, values in kidney.items()
    }


def _coxph(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return r.coxph(*args, **kwargs)


# --- the terms see the data before na.action and subset ------------------------------


def test_pspline_knots_span_the_rows_na_action_drops(lung_na):
    # coxph(Surv(time, status) ~ pspline(age, df = 3) + wt.loss, d, na.action = na.omit)
    fit = _coxph(
        "Surv(time, status) ~ pspline(age, df = 3) + wt.loss", lung_na, na_action="na.omit"
    )
    assert fit.coefficients == approx(
        [
            0.37412977756873,
            0.74825955513746,
            1.12137164575791,
            1.44586040322671,
            1.49236029485729,
            1.43174743101105,
            1.6073427231022,
            1.9379854423548,
            2.48489438255688,
            3.0999154427597,
            0.000995435420671675,
        ]
    )
    assert fit.df == approx([3.08118502605601, 0.997942601264998])


def test_ridge_scales_by_the_variances_before_na_action(lung):
    # coxph(Surv(time, status) ~ ridge(age, wt.loss, theta = 1) + ph.karno, lung,
    #       na.action = na.omit)
    fit = _coxph(
        "Surv(time, status) ~ ridge(age, wt.loss, theta = 1) + ph.karno", lung, na_action="na.omit"
    )
    assert fit.coefficients == approx(
        [0.015959299162833, -0.00159480316427902, -0.0137096643239935]
    )
    assert fit.loglik == approx([-680.390349333354, -675.53366627964])


def test_frailty_df_search_starts_from_the_length_before_na_action(lung):
    # coxph(Surv(time, status) ~ age + frailty(inst, df = 4), lung, na.action = na.omit):
    # guess = 3 * 4 / 228 with the row whose inst is missing
    fit = _coxph("Surv(time, status) ~ age + frailty(inst, df = 4)", lung, na_action="na.omit")
    assert fit.coefficients == approx([0.0193680390769959])
    assert fit.df == approx([0.984302099151188, 3.98713808220157])


def test_penalty_terms_see_the_rows_subset_drops(lung):
    # coxph(Surv(time, status) ~ pspline(age, df = 3), lung, subset = age > 50)
    fit = _coxph(
        "Surv(time, status) ~ pspline(age, df = 3)", lung, subset=[a > 50 for a in lung["age"]]
    )
    assert fit.coefficients == approx(
        [
            0.0331257300002079,
            0.0662514600004136,
            0.099377190000615,
            0.0945029259359176,
            0.0250860365800171,
            0.0574315093946005,
            0.237216986909628,
            0.526022566044918,
            1.12146345678529,
            1.80237840172981,
        ]
    )
    assert fit.df == approx([3.0332541355986])

    # coxph(Surv(time, status) ~ age + frailty(inst, df = 4), lung, subset = sex == 1,
    #       na.action = na.omit)
    fit = _coxph(
        "Surv(time, status) ~ age + frailty(inst, df = 4)",
        lung,
        subset=[s == 1 for s in lung["sex"]],
        na_action="na.omit",
    )
    assert fit.coefficients == approx([0.0187834225842315])
    assert fit.df == approx([0.97312154451152, 4.00665501491095])


def test_frailty_groups_are_the_levels_before_subset():
    kidney = _kidney(6)
    subset = [value != 3 for value in kidney["id"]]
    # coxph(Surv(time, status) ~ age + frailty(id, theta = 0.5), k6, subset = id != 3):
    # six levels make the term sparse although five remain
    fit = _coxph("Surv(time, status) ~ age + frailty(id, theta = 0.5)", kidney, subset=subset)
    assert fit.coef_names == ("age",)
    assert fit.coefficients == approx([-0.00597873632701662])
    assert fit.frail == approx(
        [
            0.47998280391531,
            0.0991827550239132,
            -1.03709282568035,
            0.105335665778507,
            -0.205649881530227,
        ]
    )
    assert fit.df == approx([0.551917423496589, 1.32034859935297])

    # the dense term keeps a column for the level subset removed
    fit = _coxph(
        "Surv(time, status) ~ age + frailty(id, theta = 0.5, sparse = FALSE)", kidney, subset=subset
    )
    assert fit.coef_names == tuple(["age"] + [f"gamma:{level}" for level in range(1, 7)])
    assert fit.coefficients == approx(
        [
            -0.00597794726709674,
            0.479982006013189,
            0.0991755389100812,
            3.87064914288259e-17,
            -1.03711717303118,
            0.105339784680742,
            -0.205633530076152,
        ]
    )


# --- coefficient names -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "prefix", "coefficients"),
    [
        (
            "frailty.gaussian(id, theta = 0.5)",
            "gauss",
            [
                -0.00977298926917718,
                0.598032336252194,
                0.0628617838576808,
                0.11056026610385,
                -0.736682660044589,
                -0.0347717261691358,
            ],
        ),
        ('frailty(id, dist = "gau", theta = 0.5)', "gauss", None),
        (
            'frailty(id, dist = "t", theta = 0.5)',
            "t",
            [
                -0.0109055199585559,
                0.36384573926614,
                0.0403271686600787,
                0.077120152357031,
                -0.51163246254321,
                -0.0158272865385325,
            ],
        ),
    ],
)
def test_dense_frailty_names_carry_the_distribution(term, prefix, coefficients):
    # coxph(Surv(time, status) ~ age + <term>, kidney[kidney$id <= 5, ])
    fit = _coxph(f"Surv(time, status) ~ age + {term}", _kidney(5))
    assert fit.coef_names == tuple(["age"] + [f"{prefix}:{level}" for level in range(1, 6)])
    if coefficients is not None:
        assert fit.coefficients == approx(coefficients)


@pytest.mark.parametrize("dist", ["g", "ga", "Gamma"])
def test_frailty_distribution_is_matched_as_pmatch_does(dist):
    with pytest.raises(ValueError, match=f"Function 'frailty.{dist}' not found"):
        _coxph(f'Surv(time, status) ~ age + frailty(id, dist = "{dist}", theta = 0.5)', _kidney(5))


def test_pspline_with_intercept_numbers_the_columns_from_one(lung):
    # coxph(Surv(time, status) ~ pspline(age, df = 3, intercept = TRUE), lung)
    fit = _coxph("Surv(time, status) ~ pspline(age, df = 3, intercept = TRUE)", lung)
    assert fit.coef_names == tuple(f"ps(age){j}" for j in range(1, 12))
    assert fit.coefficients[:10] == approx(
        [
            -2.36381855487005,
            -2.02248734831008,
            -1.68236581429351,
            -1.4009987449212,
            -1.32924155938545,
            -1.37793263965236,
            -1.3535977885166,
            -1.18431261125672,
            -0.896477843553963,
            -0.465139556771537,
        ]
    )
    assert math.isnan(fit.coefficients[10])


# --- pspline(combine =) and pspline(penalty = FALSE) ----------------------------------


def test_pspline_combine_sums_basis_columns(lung):
    # coxph(Surv(time, status) ~ pspline(age, df = 4, combine = c(1,1,2,2,...,6,6)), lung)
    fit = _coxph(
        "Surv(time, status) ~ pspline(age, df = 4, combine = c(1,1,2,2,3,3,4,4,5,5,6,6))", lung
    )
    assert fit.coef_names == tuple(f"ps(age){j}" for j in range(3, 9))
    assert fit.coefficients == approx(
        [
            0.742931340577409,
            1.48453722762625,
            1.42354521795693,
            1.45559659311473,
            1.79825562239195,
            2.88241541473311,
        ]
    )
    assert fit.df == approx([3.93598442356371])
    # R's type = "lp" predictions at ages 40, 55.5, 70 and 82
    assert r.predict(fit, newdata={"age": [40, 55.5, 70, 82]}, type="lp") == approx(
        [-0.80252747513847, -0.051650689817932, 0.0639781587297657, 1.21067594676777]
    )


@pytest.mark.parametrize(
    ("combine", "message"),
    [
        ("c(1, 1, 2)", "wrong length for combine"),
        ("c(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)", "increasing vector of positive integers"),
        ("c(1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)", "increasing vector of positive integers"),
    ],
)
def test_pspline_combine_is_checked_as_in_r(lung, combine, message):
    with pytest.raises(ValueError, match=message):
        _coxph(f"Surv(time, status) ~ pspline(age, df = 4, combine = {combine})", lung)


@pytest.mark.parametrize("knots", ["c(40)", "40", "c(40, 60, 80)", "c(50, 50)", "c(80, 40)"])
def test_pspline_boundary_knots_are_checked_as_in_r(lung, knots):
    # R: "Invalid values for Boundary.knots" for each of these
    with pytest.raises(ValueError, match="Invalid values for Boundary.knots"):
        _coxph(f"Surv(time, status) ~ pspline(age, Boundary.knots = {knots})", lung)


def test_pspline_scalar_boundary_knots_are_one_value(lung):
    # pspline(lung$age, Boundary.knots = 40): "Invalid values for Boundary.knots"
    with pytest.raises(ValueError, match="Invalid values for Boundary.knots"):
        r.pspline(lung["age"], Boundary_knots=40)


def test_pspline_without_penalty_is_an_ordinary_matrix_term(lung):
    # coxph(Surv(time, status) ~ pspline(age, df = 4, penalty = FALSE) + sex, lung)
    fit = _coxph("Surv(time, status) ~ pspline(age, df = 4, penalty = FALSE) + sex", lung)
    assert fit.penalized is None
    assert fit.coef_names == (
        *(f"pspline(age, df = 4, penalty = FALSE){j}" for j in range(1, 13)),
        "sex",
    )
    assert fit.loglik == approx([-749.909801390395, -738.127026495647])
    newdata = {"age": [40, 55.5, 70, 82], "sex": [1, 2, 1, 2]}
    assert r.predict(fit, newdata=newdata, type="lp") == approx(
        [0.0374816281782562, -0.36826562549665, 0.34009139936594, 2.91800895292195], rel=1e-7
    )
    assert _coxph(
        "Surv(time, status) ~ pspline(age, df = 4, penalty = TRUE)", lung
    ).coef_names == tuple(f"ps(age){j}" for j in range(3, 15))


def test_pspline_without_penalty_can_be_in_an_interaction(lung):
    # coxph(Surv(time, status) ~ sex + pspline(age, df = 2, nterm = 3, penalty = FALSE):sex,
    #       lung)
    fit = _coxph(
        "Surv(time, status) ~ sex + pspline(age, df = 2, nterm = 3, penalty = FALSE):sex", lung
    )
    assert fit.coef_names == (
        "sex",
        *(f"sex:pspline(age, df = 2, nterm = 3, penalty = FALSE){j}" for j in range(1, 6)),
    )
    assert fit.loglik == approx([-749.909801390395, -741.632072504261])
    assert fit.linear_predictors[:4] == approx(
        [0.339485082526231, 0.21754701812551, 0.196967163512483, 0.193641875064991], rel=1e-7
    )


@pytest.mark.parametrize(
    "rhs", ["ridge(age, theta = 1):sex", "sex:pspline(age, df = 2)", "sex + frailty(inst):sex"]
)
def test_penalized_terms_are_refused_in_interactions(lung, rhs):
    # coxph.R's message; R 3.8-12 stops before it with "missing value where TRUE/FALSE
    # needed", because the penalty column's name matches no term label
    with pytest.raises(ValueError, match="^Penalty terms cannot be in an interaction$"):
        _coxph(f"Surv(time, status) ~ {rhs}", lung, na_action="na.omit")


# --- survreg ---------------------------------------------------------------------------


def test_survreg_fits_an_unpenalized_pspline_with_the_knots_before_subset(lung):
    # survreg(Surv(time, status) ~ pspline(age, df = 2, nterm = 3, penalty = FALSE), lung,
    #         subset = age > 50)
    fit = r.survreg(
        "Surv(time, status) ~ pspline(age, df = 2, nterm = 3, penalty = FALSE)",
        lung,
        subset=[a > 50 for a in lung["age"]],
    )
    assert fit.loglik == approx([-1038.48188919565, -1034.86179744496])
    assert fit.linear_predictors[:4] == approx(
        [5.93924092183784, 5.97545580058567, 6.08705914445437, 6.12655335880692], rel=1e-7
    )


@pytest.mark.parametrize(
    ("rhs", "error", "message"),
    [
        ("ridge(age, sex, theta = 1)", NotImplementedError, "survpenal.fit"),
        ("pspline(age, df = 3) + sex", NotImplementedError, "survpenal.fit"),
        ("age + frailty(inst, df = 2)", ValueError, "survreg does not support frailty terms"),
        ("age + frailty.gaussian(inst)", ValueError, "survreg does not support frailty terms"),
        ("ridge(age, theta = 1) + frailty(inst)", ValueError, "does not support frailty"),
        # R reports that survreg does not support frailty terms
        ("sex + frailty(inst):sex", ValueError, "Penalty terms cannot be in an interaction"),
    ],
)
def test_survreg_refuses_penalized_terms(lung, rhs, error, message):
    with pytest.raises(error, match=message):
        r.survreg(f"Surv(time, status) ~ {rhs}", lung, na_action="na.omit")
