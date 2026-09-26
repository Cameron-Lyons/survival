"""survreg's vcov, confint and summary methods against R 4.5.3 / survival 3.8-12.

vcov.survreg keeps the ``Log(scale)`` rows and, without ``complete``, drops only the
aliased coefficients, naming a stratum's scale ``Log(scale[<stratum>])``;
summary.survreg returns the (intercept-only, full) log-likelihood pair, ``var`` and, on
request, the correlation matrix.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
_survreg = survival.r._survreg


def approx(values, rel=1e-9):
    return pytest.approx(values, rel=rel, abs=1e-15, nan_ok=True)


def assert_matrix(actual, expected, rel=1e-9):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == approx(expected_row, rel=rel)


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


@pytest.fixture(scope="module")
def lung_age2(lung):
    """``lung$age2 <- 2 * lung$age``: an aliased copy of ``age``."""

    return dict(lung, age2=[2 * age for age in lung["age"]])


@pytest.fixture(scope="module")
def weibull(lung):
    return r.survreg("Surv(time, status) ~ age + sex", data=lung)


@pytest.fixture(scope="module")
def aliased(lung_age2):
    return r.survreg("Surv(time, status) ~ age + age2 + sex", data=lung_age2)


# R: vcov(survreg(Surv(time, status) ~ age + sex, data = lung))
WEIBULL_VAR = [
    [
        0.23171414330272782633,
        -3.1097617536932310e-03,
        -2.3843522025335913e-02,
        7.2336678389959034e-04,
    ],
    [
        -0.00310976175369323147,
        4.8406420317608284e-05,
        3.6562441424825735e-05,
        -4.0023830227527392e-05,
    ],
    [
        -0.02384352202533591286,
        3.6562441424825735e-05,
        1.6250344864536283e-02,
        1.2292561430270054e-03,
    ],
    [
        0.00072336678389959034,
        -4.0023830227527392e-05,
        1.2292561430270054e-03,
        3.8295393687759597e-03,
    ],
]

# R: vcov(survreg(Surv(time, status) ~ age + strata(sex), data = lung), complete = FALSE);
# the same matrix with age2 = 2 * age added to the model
STRATA_VAR = [
    [
        0.1906255971225058188,
        -2.9756831508658188e-03,
        1.0702187496288568e-03,
        0.00587230398801269655,
    ],
    [
        -0.0029756831508658188,
        4.7356487239585828e-05,
        -5.1067626770767945e-06,
        -0.00011843710959046602,
    ],
    [
        0.0010702187496288568,
        -5.1067626770767945e-06,
        6.2202639582025964e-03,
        -0.00030984064235154926,
    ],
    [
        0.0058723039880126966,
        -1.1843710959046602e-04,
        -3.0984064235154926e-04,
        0.01140437957131366399,
    ],
]


# --- vcov ------------------------------------------------------------------------------------


def test_vcov_without_complete_keeps_the_scale_when_nothing_is_aliased(weibull):
    # R: vcov(fit, complete = FALSE) is fit$var, Log(scale) included
    assert_matrix(r.vcov(weibull, complete=False), WEIBULL_VAR)
    assert r.vcov(weibull, complete=False) == r.vcov(weibull)
    names = ["(Intercept)", "age", "sex", "Log(scale)"]
    assert _survreg.survreg_vcov_names(weibull, False) == names
    assert r.coef_names(weibull, complete=True) == names


def test_vcov_without_complete_drops_only_the_aliased_coefficient(aliased):
    # R: vcov(survreg(Surv(time, status) ~ age + age2 + sex, lung), complete = FALSE)
    assert math.isnan(r.coef(aliased)[2])
    assert_matrix(
        r.vcov(aliased, complete=False),
        [
            [
                0.23171414330272782633,
                -3.1097617536932310e-03,
                -2.3843522025335906e-02,
                7.2336678389959045e-04,
            ],
            [
                -0.00310976175369323147,
                4.8406420317608284e-05,
                3.6562441424825681e-05,
                -4.0023830227527439e-05,
            ],
            [
                -0.02384352202533590592,
                3.6562441424825681e-05,
                1.6250344864536279e-02,
                1.2292561430270073e-03,
            ],
            [
                0.00072336678389959045,
                -4.0023830227527439e-05,
                1.2292561430270073e-03,
                3.8295393687759606e-03,
            ],
        ],
    )
    assert _survreg.survreg_vcov_names(aliased, False) == [
        "(Intercept)",
        "age",
        "sex",
        "Log(scale)",
    ]
    # complete = TRUE is fit$var, with the aliased coefficient's zero row and column
    full = r.vcov(aliased)
    assert len(full) == 5
    assert full[2] == [0.0] * 5
    assert [row[2] for row in full] == [0.0] * 5
    assert r.coef_names(aliased, complete=True) == [
        "(Intercept)",
        "age",
        "age2",
        "sex",
        "Log(scale)",
    ]
    # R keeps the aliased name in names(coef(fit)) and drops it with complete = FALSE
    assert r.coef_names(aliased) == ["(Intercept)", "age", "age2", "sex"]
    assert r.coef_names(aliased, complete=False) == ["(Intercept)", "age", "sex"]


def test_vcov_names_a_stratum_scale_by_its_level(lung, lung_age2):
    # R: vcov(survreg(Surv(time, status) ~ age + strata(sex), lung), complete = FALSE)
    fit = r.survreg("Surv(time, status) ~ age + strata(sex)", data=lung)
    names = ["(Intercept)", "age", "Log(scale[sex=1])", "Log(scale[sex=2])"]
    assert_matrix(r.vcov(fit, complete=False), STRATA_VAR)
    assert _survreg.survreg_vcov_names(fit, False) == names
    assert r.coef_names(fit, complete=True) == names

    with_alias = r.survreg("Surv(time, status) ~ age + age2 + strata(sex)", data=lung_age2)
    assert_matrix(r.vcov(with_alias, complete=False), STRATA_VAR)
    assert _survreg.survreg_vcov_names(with_alias, False) == names


def test_vcov_keeps_every_stratum_scale_where_r_recycles_the_alias_pattern(lung_age2):
    # R's var[keep, keep] recycles the 3-long location pattern (TRUE, TRUE, FALSE) over
    # the six rows, drops Log(scale[grp=3]) and then stops in dimnames<-; the port keeps
    # every scale row, R's fit$var[-3, -3]
    data = dict(lung_age2, grp=[(row + 1) % 3 + 1 for row in range(len(lung_age2["age"]))])
    fit = r.survreg("Surv(time, status) ~ age + age2 + strata(grp)", data=data)
    assert_matrix(
        r.vcov(fit, complete=False),
        [
            [
                0.2027174537142933941,
                -3.1467418415722639e-03,
                0.00729668356085087181,
                5.3815510953109640e-03,
                -4.3943724741659336e-03,
            ],
            [
                -0.0031467418415722643,
                4.9700064136632666e-05,
                -0.00011881094856798365,
                -8.1051863643532212e-05,
                6.2948644682657851e-05,
            ],
            [
                0.0072966835608508718,
                -1.1881094856798365e-04,
                0.01125163884965351725,
                1.7756515460479226e-04,
                -1.2397853919319377e-04,
            ],
            [
                0.0053815510953109640,
                -8.1051863643532212e-05,
                0.00017756515460479226,
                1.1663325386312571e-02,
                -1.3197810012317683e-04,
            ],
            [
                -0.0043943724741659336,
                6.2948644682657851e-05,
                -0.00012397853919319377,
                -1.3197810012317683e-04,
                1.2365449732417835e-02,
            ],
        ],
    )
    assert _survreg.survreg_vcov_names(fit, False) == [
        "(Intercept)",
        "age",
        "Log(scale[grp=1])",
        "Log(scale[grp=2])",
        "Log(scale[grp=3])",
    ]


def test_vcov_of_a_fixed_scale_fit_has_no_scale_rows(lung_age2):
    # R: vcov(survreg(Surv(time, status) ~ age + age2 + sex, lung, dist = "exponential"),
    #         complete = FALSE)
    fit = r.survreg("Surv(time, status) ~ age + age2 + sex", data=lung_age2, dist="exponential")
    assert_matrix(
        r.vcov(fit, complete=False),
        [
            [0.4038209548322661546, -5.3811314597289423e-03, -0.04330588460926872163],
            [-0.0053811314597289423, 8.2913411628748784e-05, 0.00010139837902576305],
            [-0.0433058846092687286, 1.0139837902576305e-04, 0.02792050039790095611],
        ],
    )
    assert _survreg.survreg_vcov_names(fit, False) == ["(Intercept)", "age", "sex"]
    assert r.coef_names(fit, complete=True) == ["(Intercept)", "age", "age2", "sex"]


# --- confint ---------------------------------------------------------------------------------


def test_confint_reads_the_location_block_of_the_complete_variance(aliased, lung_age2):
    # R: confint(survreg(Surv(time, status) ~ age + age2 + sex, lung)); NA for age2
    intervals = r.confint(aliased)
    assert [row["name"] for row in intervals] == ["(Intercept)", "age", "age2", "sex"]
    assert [row["lower"] for row in intervals] == approx(
        [5.331391167471476678, -0.025893420651695333, math.nan, 0.132235123410748140]
    )
    assert [row["upper"] for row in intervals] == approx(
        [7.218314949366066102, 0.001379369473798138, math.nan, 0.631935155907022206]
    )

    # R: confint(survreg(Surv(time, status) ~ age + age2 + strata(sex), lung))
    strata = r.survreg("Surv(time, status) ~ age + age2 + strata(sex)", data=lung_age2)
    intervals = r.confint(strata)
    assert [row["lower"] for row in intervals] == approx(
        [5.983059656358339140, -0.025956873589986031, math.nan]
    )
    assert [row["upper"] for row in intervals] == approx(
        [7.6945273090077659361, 0.0010185222828159642, math.nan]
    )


# --- summary ---------------------------------------------------------------------------------


def test_summary_has_the_loglik_pair_and_var(weibull):
    # R: s <- summary(survreg(Surv(time, status) ~ age + sex, lung)); s$loglik, s$var, s$chi
    summary = r.model_summary(weibull)
    assert summary["loglik"] == approx([-1153.8511880894062, -1147.0544314319495], rel=1e-12)
    assert summary["chi"] == approx(13.593513314913253)
    assert_matrix(summary["var"], WEIBULL_VAR)
    assert summary["correlation"] is None
    assert r.model_summary(weibull, correlation=False)["correlation"] is None
    with pytest.raises(TypeError, match="correlation must be True or False"):
        r.model_summary(weibull, correlation="yes")


def test_summary_correlation(weibull, lung):
    # R: summary(survreg(Surv(time, status) ~ age + sex, lung), correlation = TRUE)$correlation
    assert_matrix(
        r.model_summary(weibull, correlation=True)["correlation"],
        [
            [
                1.00000000000000000,
                -0.928537317944768348,
                -0.388564253663504666,
                0.024283373618960550,
            ],
            [
                -0.92853731794476846,
                0.999999999999999889,
                0.041224216750194165,
                -0.092959524732490098,
            ],
            [
                -0.38856425366350472,
                0.041224216750194165,
                1.000000000000000222,
                0.155825248091623986,
            ],
            [
                0.02428337361896055,
                -0.092959524732490084,
                0.155825248091623986,
                0.999999999999999778,
            ],
        ],
    )

    # R: summary(survreg(Surv(time, status) ~ age + strata(sex), lung), correlation = TRUE)
    stratified = r.model_summary(
        r.survreg("Surv(time, status) ~ age + strata(sex)", data=lung), correlation=True
    )
    assert stratified["coefficient_names"] == ["(Intercept)", "age", "sex=1", "sex=2"]
    assert_matrix(
        stratified["correlation"],
        [
            [
                0.99999999999999989,
                -0.9903902066161403006,
                0.0310797516293830900,
                0.125945335913083267,
            ],
            [-0.99039020661614030, 1.0, -0.0094091746168164202, -0.161161844108043656],
            [0.03107975162938309, -0.0094091746168164202, 1.0, -0.036787319598684594],
            [
                0.12594533591308327,
                -0.1611618441080436559,
                -0.0367873195986845938,
                0.999999999999999778,
            ],
        ],
    )

    # R: s <- summary(survreg(Surv(time, status) ~ age + sex, lung, robust = TRUE),
    #                 correlation = TRUE); s$var, s$correlation
    robust = r.model_summary(
        r.survreg("Surv(time, status) ~ age + sex", data=lung, robust=True), correlation=True
    )
    assert_matrix(
        robust["var"],
        [
            [
                0.2406829272221544247,
                -3.3653030210303988e-03,
                -1.8319750567392677e-02,
                0.00481233306116271055,
            ],
            [
                -0.0033653030210303988,
                5.4268084783706290e-05,
                -3.5013789574117078e-05,
                -0.00010058278721841293,
            ],
            [
                -0.0183197505673926772,
                -3.5013789574117078e-05,
                1.4740279784649299e-02,
                0.00114209283215141043,
            ],
            [
                0.0048123330611627105,
                -1.0058278721841293e-04,
                1.1420928321514104e-03,
                0.00436435450334235562,
            ],
        ],
    )
    assert_matrix(
        robust["correlation"],
        [
            [
                1.00000000000000022,
                -0.931170664503100598,
                -0.307570055043681068,
                0.14848173202225945,
            ],
            [
                -0.93117066450310060,
                1.000000000000000222,
                -0.039148399046889465,
                -0.20667664773867064,
            ],
            [-0.30757005504368107, -0.039148399046889465, 1.0, 0.14239296475336255],
            [0.14848173202225948, -0.206676647738670644, 0.142392964753362550, 1.00000000000000022],
        ],
    )


def test_summary_correlation_leaves_out_an_aliased_coefficient(aliased):
    # R's summary.survreg(correlation = TRUE) stops in dimnames<- here (it labels the
    # reduced matrix with every name); the port returns its intent, R's
    # cov2cor(fit$var[-3, -3])
    summary = r.model_summary(aliased, correlation=True)
    assert summary["coefficient_names"] == ["(Intercept)", "age", "age2", "sex", "Log(scale)"]
    assert len(summary["var"]) == 5
    assert_matrix(
        summary["correlation"],
        [
            [1.0, -0.928537317944768348, -0.388564253663504555, 0.024283373618960553],
            [-0.928537317944768459, 1.0, 0.041224216750194102, -0.092959524732490209],
            [-0.388564253663504611, 0.041224216750194102, 1.0, 0.155825248091624236],
            [0.024283373618960557, -0.092959524732490195, 0.155825248091624236, 1.0],
        ],
    )
