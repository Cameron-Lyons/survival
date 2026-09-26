"""The methods of a penalized survreg fit (R's ``survreg.penal``): print.survreg.penal's
table, summary.survreg, anova with penalized refits, logLik and friends, and predict,
residuals, vcov, confint, concordance and model.matrix, against R 4.5.3 / survival
3.8-12.  The reference values were computed with R and are hard-coded; the printed lines
are R's ``print(fit)`` output after the Call block, with trailing blanks stripped.
"""

import math
from typing import Any

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets

NAN = math.nan
S = "Surv(time, status) ~ "


def approx(values: Any, rel: float = 1e-6) -> Any:
    return pytest.approx(values, rel=rel, abs=1e-12, nan_ok=True)


@pytest.fixture(scope="module")
def lung() -> dict[str, list[Any]]:
    lung = {k: list(v) for k, v in datasets.load_lung().items() if not k.startswith("_")}
    lung["off"] = [s / 10 for s in lung["sex"]]
    lung["w"] = [2.0 if a > 60 else 1.0 for a in lung["age"]]
    lung["agec2"] = [(a - 62) ** 2 / 100 for a in lung["age"]]
    return lung


@pytest.fixture(scope="module")
def kidney() -> Any:
    pl = pytest.importorskip("polars")
    data = datasets.load_kidney()
    frame = pl.DataFrame({k: list(v) for k, v in data.items() if not k.startswith("_")})
    return frame.with_columns(pl.col("disease").cast(pl.Enum(["Other", "GN", "AN", "PKD"])))


@pytest.fixture(scope="module")
def f_ps(lung):
    return r.survreg(S + "pspline(age, df = 3) + sex", lung)


@pytest.fixture(scope="module")
def f_ridge(lung):
    return r.survreg(S + "ridge(age, sex, theta = 1)", lung)


@pytest.fixture(scope="module")
def f_kid(kidney):
    return r.survreg(S + "pspline(age, df = 3) + sex + disease", kidney)


@pytest.fixture(scope="module")
def f_strata(lung):
    return r.survreg(S + "pspline(age, df = 3) + strata(sex)", lung)


@pytest.fixture(scope="module")
def f_fixed(lung):
    return r.survreg(S + "ridge(age, sex, theta = 1)", lung, scale=1)


# --- print.survreg.penal -----------------------------------------------------------------


def test_print_pspline(f_ps):
    out = r.print_survreg_penal(f_ps)
    assert type(out) is r.SurvregPenalPrint
    assert out.rownames == [
        "(Intercept)",
        "pspline(age, df = 3), lin",
        "pspline(age, df = 3), non",
        "sex",
    ]
    assert out.columns == ("coef", "se(coef)", "se2", "Chisq", "DF", "p")
    assert out.rows == [
        approx([6.24585597390229, 0.774295468910945, 0.542072019950927, 65.0684550440064, 1,
                7.23408783883134e-16]),
        approx([-0.0122845071132055, 0.00678832327074835, 0.00678794746085093,
                3.27484266952672, 1, 0.0703496715686645]),
        approx([NAN, NAN, NAN, 1.83094970775341, 2.06393485506937, 0.41463880579785]),
        approx([0.38107546373868, 0.12718861465403, 0.127035140001751, 8.97688165194598, 1,
                0.00273416868047694]),
    ]  # fmt: skip
    assert out.history == ["Theta= 0.924"]
    assert (out.logtest, out.logtest_df, out.logtest_p) == approx(
        (16.1136210525574, 3.55034143263268, 0.00188889308490712)
    )
    assert (out.n, out.iter, out.digits, out.fixed_scale) == (228, [5, 16], 3, False)
    assert out.lines == [
        "                          coef    se(coef) se2     Chisq DF   p",
        "(Intercept)                6.2459 0.77430  0.54207 65.07 1.00 7.2e-16",
        "pspline(age, df = 3), lin -0.0123 0.00679  0.00679  3.27 1.00 7.0e-02",
        "pspline(age, df = 3), non                           1.83 2.06 4.1e-01",
        "sex                        0.3811 0.12719  0.12704  8.98 1.00 2.7e-03",
        "",
        "Scale= 0.75",
        "",
        "Iterations: 5 outer, 16 Newton-Raphson",
        "     Theta= 0.924",
        "Degrees of freedom for terms= 0.5 3.1 1.0 1.0",
        "Likelihood ratio test=16.1  on 3.6 df, p=0.002  n= 228",
    ]
    text = str(out)
    assert text.startswith(
        "Call:\nsurvreg(formula = Surv(time, status) ~ pspline(age, df = 3) + sex)\n\n"
    )
    assert text.endswith("n= 228\n")


def test_print_ridge(f_ridge):
    out = r.print_survreg_penal(f_ridge)
    assert out.rows == [
        approx([6.27390503073021, 0.480249954868952, 0.479287126382474, 170.663722971442, 1,
                5.29912768643092e-39]),
        approx([-0.0122117247372104, 0.00694129170709535, 0.00692748808151663,
                3.09509098849476, 1, 0.0785287619538989]),
        approx([0.380638679432984, 0.127140477869816, 0.12689321283388, 8.96309825142868, 1,
                0.00275487309610932]),
    ]  # fmt: skip
    assert out.history == []
    assert (out.logtest, out.logtest_df, out.logtest_p) == approx(
        (13.5933356904902, 1.98802260978291, 0.00110003907510153)
    )
    assert out.lines == [
        "            coef    se(coef) se2     Chisq  DF p",
        "(Intercept)  6.2739 0.48025  0.47929 170.66 1  5.3e-39",
        "ridge(age)  -0.0122 0.00694  0.00693   3.10 1  7.9e-02",
        "ridge(sex)   0.3806 0.12714  0.12689   8.96 1  2.8e-03",
        "",
        "Scale= 0.754",
        "",
        "Iterations: 1 outer, 5 Newton-Raphson",
        "Degrees of freedom for terms= 1 2 1",
        "Likelihood ratio test=13.6  on 2 df, p=0.001  n= 228",
    ]


def test_print_terms_makes_a_multicolumn_ridge_one_wald_row(f_ridge):
    out = r.print_survreg_penal(f_ridge, terms=True)
    assert out.rownames == ["(Intercept)", "ridge(age, sex, theta = 1"]
    # the p-value is on 1 df, not the DF column's 1.99 (as R)
    assert out.rows == [
        approx([6.27390503073021, 0.480249954868952, 0.479287126382474, 170.663722971442, 1,
                5.29912768643092e-39]),
        approx([NAN, NAN, NAN, 12.5148891620972, 1.99215425102188, 0.000403721737726793]),
    ]  # fmt: skip
    assert out.lines[:3] == [
        "                          coef se(coef) se2   Chisq DF   p",
        "(Intercept)               6.27 0.48     0.479 170.7 1.00 5.3e-39",
        "ridge(age, sex, theta = 1                      12.5 1.99 4.0e-04",
    ]


def test_print_kidney_terms(f_kid):
    out = r.print_survreg_penal(f_kid, terms=True)
    assert out.rows[1:4] == [
        approx([-0.00177481342176895, 0.00999652483657338, 0.00994487143196454,
                0.0315215315088221, 1, 0.85908186013957]),
        approx([NAN, NAN, NAN, 1.88051184783359, 2.09669387411263, 0.411916563593353]),
        approx([1.63849733939882, 0.319828993739394, 0.318467368560539, 26.2455584256732, 1,
                3.00645489161996e-07]),
    ]  # fmt: skip
    assert out.rownames[4] == "disease"
    assert out.rows[4] == approx(
        [NAN, NAN, NAN, 12.6245175637379, 2.81019582338287, 0.000380720125350966]
    )
    assert out.history == ["Theta= 0.773"]
    assert (out.logtest, out.logtest_df, out.logtest_p, out.n) == approx(
        (23.4605035615029, 6.53574696452374, 0.0010018447322212, 76)
    )
    assert out.lines == [
        "                          coef     se(coef) se2     Chisq DF   p",
        "(Intercept)                1.47134 1.11     0.89147  1.76 1.00 1.8e-01",
        "pspline(age, df = 3), lin -0.00177 0.01     0.00994  0.03 1.00 8.6e-01",
        "pspline(age, df = 3), non                            1.88 2.10 4.1e-01",
        "sex                        1.63850 0.32     0.31847 26.25 1.00 3.0e-07",
        "disease                                             12.62 2.81 3.8e-04",
        "",
        "Scale= 0.951",
        "",
        "Iterations: 3 outer, 12 Newton-Raphson",
        "     Theta= 0.773",
        "Degrees of freedom for terms= 0.6 3.1 1.0 2.8 1.0",
        "Likelihood ratio test=23.5  on 6.5 df, p=0.001  n= 76",
    ]

    out = r.print_survreg_penal(f_kid)
    assert out.rownames[4:] == ["diseaseGN", "diseaseAN", "diseasePKD"]
    assert [row[0] for row in out.rows[4:]] == approx(
        [0.0375630322774717, -0.54106253073379, 1.29467449421004]
    )
    assert [row[1] for row in out.rows[4:]] == approx(
        [0.444735909244706, 0.403645933059445, 0.578755098298541]
    )
    assert out.lines[:8] == [
        "                          coef     se(coef) se2     Chisq DF  p",
        "(Intercept)                1.47134 1.109    0.89147  1.76 1.0 1.8e-01",
        "pspline(age, df = 3), lin -0.00177 0.010    0.00994  0.03 1.0 8.6e-01",
        "pspline(age, df = 3), non                            1.88 2.1 4.1e-01",
        "sex                        1.63850 0.320    0.31847 26.25 1.0 3.0e-07",
        "diseaseGN                  0.03756 0.445    0.41639  0.01 1.0 9.3e-01",
        "diseaseAN                 -0.54106 0.404    0.38941  1.80 1.0 1.8e-01",
        "diseasePKD                 1.29467 0.579    0.56074  5.00 1.0 2.5e-02",
    ]


def test_print_two_penalties(lung):
    fit = r.survreg(S + "pspline(age, df = 3) + ridge(sex, agec2, theta = 1)", lung)
    out = r.print_survreg_penal(fit)
    assert out.rows == [
        approx([6.71517334646903, 2.66448191361222, 0.564516236121696, 6.35168428373779, 1,
                0.0117269908968652]),
        approx([-0.0122130506656044, 0.00677582073043233, 0.00677497367473356,
                3.24881141975497, 1, 0.0714752705380747]),
        approx([NAN, NAN, NAN, 2.465838804334, 2.09339521947694, 0.309224058005302]),
        approx([0.381221278439617, 0.127009462826432, 0.126515891527678, 9.00911453902082, 1,
                0.00268636534316447]),
        approx([-0.0209728592672551, 0.366960357063472, 0.106540062600304,
                0.00326645730392958, 1, 0.954423373972043]),
    ]  # fmt: skip
    assert out.history == ["Theta= 0.782"]
    assert (out.logtest, out.logtest_df) == approx((17.0188579485948, 3.21274702574432))
    assert out.lines == [
        "                          coef    se(coef) se2     Chisq DF   p",
        "(Intercept)                6.7152 2.66448  0.56452 6.35  1.00 0.0120",
        "pspline(age, df = 3), lin -0.0122 0.00678  0.00677 3.25  1.00 0.0710",
        "pspline(age, df = 3), non                          2.47  2.09 0.3100",
        "ridge(sex)                 0.3812 0.12701  0.12652 9.01  1.00 0.0027",
        "ridge(agec2)              -0.0210 0.36696  0.10654 0.00  1.00 0.9500",
        "",
        "Scale= 0.749",
        "",
        "Iterations: 3 outer, 11 Newton-Raphson",
        "     Theta= 0.782",
        "Degrees of freedom for terms= 0.0 3.1 1.1 1.0",
        "Likelihood ratio test=17  on 3.2 df, p=9e-04  n= 228",
    ]


def test_print_strata(f_strata):
    out = r.print_survreg_penal(f_strata)
    assert out.rows == [
        approx([6.82903491801213, 0.774731747541877, 0.513766301526636, 77.6990949906332, 1,
                1.1999631372355e-18]),
        approx([-0.0124855135229911, 0.00670851214226097, 0.00670829869273367,
                3.46386054011834, 1, 0.0627237093565249]),
        approx([NAN, NAN, NAN, 2.0301528056425, 2.06217204105684, 0.375680675199099]),
    ]  # fmt: skip
    assert out.scale_names == ["sex=1", "sex=2"]
    assert (out.logtest, out.logtest_df) == approx((6.14446166823427, 2.49352777012269))
    assert out.lines[4:] == [
        "",
        "Scale:",
        "sex=1 sex=2",
        "0.807 0.654",
        "",
        "Iterations: 5 outer, 16 Newton-Raphson",
        "     Theta= 0.921",
        "Degrees of freedom for terms= 0.4 3.1 2.0",
        "Likelihood ratio test=6.14  on 2.5 df, p=0.07  n= 228",
    ]


def test_print_fixed_scale(f_fixed):
    out = r.print_survreg_penal(f_fixed)
    assert out.fixed_scale
    assert out.rows[0] == approx(
        [6.35776040444341, 0.633084281121129, 0.630876043731395, 100.852171453125, 1,
         9.91110954389704e-24]
    )  # fmt: skip
    assert (out.logtest, out.logtest_df) == approx((12.4778047330378, 1.97961128683534))
    assert out.lines == [
        "            coef    se(coef) se2     Chisq  DF p",
        "(Intercept)  6.3578 0.63308  0.63088 100.85 1  9.9e-24",
        "ridge(age)  -0.0155 0.00907  0.00904   2.93 1  8.7e-02",
        "ridge(sex)   0.4779 0.16644  0.16588   8.24 1  4.1e-03",
        "",
        "Scale fixed at 1",
        "",
        "Iterations: 1 outer, 4 Newton-Raphson",
        "Degrees of freedom for terms= 1 2",
        "Likelihood ratio test=12.5  on 2 df, p=0.002  n= 228",
    ]


def test_print_digits_wrap_the_table(f_ps, f_kid):
    # print(f_ps, digits = 7): a 80-character line wraps p into a block of its own
    assert r.print_survreg_penal(f_ps, digits=7).lines == [
        "                          coef        se(coef)    se2         Chisq DF",
        "(Intercept)                6.24585597 0.774295469 0.542072020 65.07 1.00",
        "pspline(age, df = 3), lin -0.01228451 0.006788323 0.006787947  3.27 1.00",
        "pspline(age, df = 3), non                                      1.83 2.06",
        "sex                        0.38107546 0.127188615 0.127035140  8.98 1.00",
        "                          p",
        "(Intercept)               7.2e-16",
        "pspline(age, df = 3), lin 7.0e-02",
        "pspline(age, df = 3), non 4.1e-01",
        "sex                       2.7e-03",
        "",
        "Scale= 0.7501515",
        "",
        "Iterations: 5 outer, 16 Newton-Raphson",
        "     Theta= 0.9238247",
        "Degrees of freedom for terms= 0.5 3.1 1.0 1.0",
        "Likelihood ratio test=16.11  on 3.6 df, p=0.00189  n= 228",
    ]
    # print(f_ps, maxlabel = 40, digits = 5): a 78-character line does not wrap
    assert r.print_survreg_penal(f_ps, maxlabel=40, digits=5).lines == [
        "                             coef      se(coef)  se2       Chisq DF   p",
        "(Intercept)                   6.245856 0.7742955 0.5420720 65.07 1.00 7.2e-16",
        "pspline(age, df = 3), linear -0.012285 0.0067883 0.0067879  3.27 1.00 7.0e-02",
        "pspline(age, df = 3), nonlin                                1.83 2.06 4.1e-01",
        "sex                           0.381075 0.1271886 0.1270351  8.98 1.00 2.7e-03",
        "",
        "Scale= 0.75015",
        "",
        "Iterations: 5 outer, 16 Newton-Raphson",
        "     Theta= 0.92382",
        "Degrees of freedom for terms= 0.5 3.1 1.0 1.0",
        "Likelihood ratio test=16.11  on 3.6 df, p=0.002  n= 228",
    ]
    lines = r.print_survreg_penal(f_kid, digits=8).lines
    assert lines[0] == "                          coef          se(coef)     se2          Chisq DF"
    assert lines[1] == "(Intercept)                1.4713422918 1.1093110679 0.8914721910  1.76 1.0"
    assert lines[8] == "                          p"
    assert lines[16:] == [
        "",
        "Scale= 0.95146921",
        "",
        "Iterations: 3 outer, 12 Newton-Raphson",
        "     Theta= 0.77291194",
        "Degrees of freedom for terms= 0.6 3.1 1.0 2.8 1.0",
        "Likelihood ratio test=23.46  on 6.5 df, p=0.001002  n= 76",
    ]


def test_print_reports_the_deleted_rows(lung):
    fit = r.survreg(S + "pspline(age, df = 3) + ph.karno", lung)
    assert r.print_survreg_penal(fit).lines == [
        "                          coef     se(coef) se2     Chisq DF   p",
        "(Intercept)                5.92622 0.88755  0.65477 44.58 1.00 2.4e-11",
        "pspline(age, df = 3), lin -0.00975 0.00706  0.00706  1.90 1.00 1.7e-01",
        "pspline(age, df = 3), non                            1.87 2.06 4.1e-01",
        "ph.karno                   0.00989 0.00462  0.00461  4.57 1.00 3.2e-02",
        "",
        "Scale= 0.754",
        "",
        "Iterations: 5 outer, 16 Newton-Raphson",
        "     Theta= 0.919",
        "Degrees of freedom for terms= 0.5 3.1 1.0 1.0",
        "Likelihood ratio test=11.2  on 3.6 df, p=0.02",
        "  n=227 (1 observation deleted due to missingness)",
    ]


def test_print_guards_and_surface(lung, f_ps):
    with pytest.raises(TypeError, match="Invalid object"):
        r.print_survreg_penal(r.survreg(S + "age", lung))
    for name in ("print_survreg_penal", "SurvregPenalPrint"):
        assert name in r.__all__
        assert getattr(survival.r_api, name) is getattr(r, name)


# --- summary.survreg -----------------------------------------------------------------------


def _summary_row(summary: dict[str, Any], name: str) -> list[float]:
    row = next(row for row in summary["coefficients"] if row["name"] == name)
    return [row["value"], row["se"], row["z"], row["p"]]


def test_summary(f_ps, f_strata, f_fixed, lung):
    summary = r.model_summary(f_ps)
    assert summary["coefficient_names"] == [
        "(Intercept)",
        *[f"ps(age){k}" for k in range(3, 13)],
        "sex",
        "Log(scale)",
    ]
    assert _summary_row(summary, "(Intercept)") == approx(
        [6.24585597390229, 0.774295468910945, 8.06650203272809, 7.23408783883129e-16]
    )
    assert _summary_row(summary, "ps(age)3") == approx(
        [-0.252583945887303, 0.423080803826475, -0.597011123177547, 0.550499954277177]
    )
    assert _summary_row(summary, "Log(scale)") == approx(
        [-0.287480069996392, 0.0619003017054583, -4.6442434378481, 3.4132491214903e-06]
    )
    assert (summary["chi"], summary["chi_df"]) == approx((16.1136210525574, 3.55034143263268))
    assert summary["loglik"] == approx([-1153.85118808941, -1145.79437756313])
    assert summary["iter"] == [5, 16]
    assert summary["df"] == f_ps.df

    summary = r.model_summary(f_strata)
    assert _summary_row(summary, "sex=1")[:2] == approx([-0.214776421924717, 0.0788082347448615])
    assert _summary_row(summary, "sex=2")[:2] == approx([-0.423894943495099, 0.10775808052333])
    assert summary["chi_df"] == approx(2.49352777012269)

    summary = r.model_summary(f_fixed)
    assert summary["coefficient_names"] == ["(Intercept)", "ridge(age)", "ridge(sex)"]
    assert summary["chi_df"] == approx(1.97961128683534)

    robust = r.model_summary(r.survreg(S + "ridge(age, sex, theta = 1)", lung, robust=True))
    row = robust["coefficients"][1]
    assert row["name"] == "ridge(age)"
    assert [row["value"], row["se"], row["naive_se"], row["z"], row["p"]] == approx(
        [-0.0122117247372104, 0.00733358364618504, 0.00694129170709535, -1.66517835295477,
         0.0958771783686467]
    )  # fmt: skip


def test_summary_chi_df_of_an_unpenalized_fit_is_an_integer(lung):
    summary = r.model_summary(r.survreg(S + "age + sex", lung))
    assert summary["chi_df"] == 2
    assert isinstance(summary["chi_df"], int)


# --- anova -----------------------------------------------------------------------------------


def test_anova_refits_with_the_penalties(f_ps, f_ridge):
    table = r.anova(f_ps)
    assert table.terms == ["NULL", "pspline(age, df = 3)", "sex"]
    assert table.df == approx([NAN, 2.50673672260825, 1.04360471002443])
    assert table.deviance == approx([NAN, 6.44467099754411, 9.66895005501328])
    assert table.resid_df == approx([226, 223.493263277392, 222.449658567367])
    assert table.loglik == approx([2307.70237617881, 2301.25770518127, 2291.58875512625])
    assert table.p == approx([NAN, 0.0631136167765497, 0.00202937617340942])

    table = r.anova(f_ridge)
    assert table.df == approx([NAN, 1.98802260978292])
    assert table.resid_df == approx([226, 224.011977390217])
    assert table.loglik == approx([2307.70237617881, 2294.10904048832])
    assert table.p == approx([NAN, 0.00110003907510154])


def test_anova_refits_drop_strata_and_reindex_the_penalties(f_strata, lung):
    table = r.anova(f_strata)
    assert table.terms == ["NULL", "pspline(age, df = 3)", "strata(sex)"]
    assert table.df == approx([NAN, 2.50673672260825, 0.98679104751443])
    assert table.resid_df == approx([226, 223.493263277392, 222.506472229877])
    assert table.loglik == approx([2307.70237617881, 2301.25770518127, 2298.91364432996])
    assert table.p == approx([NAN, 0.0631136167765497, 0.123617122185947])

    fit = r.survreg(S + "sex + pspline(age, df = 3) + ridge(agec2, w, theta = 1)", lung, scale=1)
    table = r.anova(fit)
    assert table.heading[2] == "Scale fixed at 1"
    assert table.df == approx([NAN, 1, 2.59867774832577, 0.260198191649209])
    assert table.resid_df == approx([227, 226, 223.401322251674, 223.141124060025])
    assert table.loglik == approx(
        [2324.67635157493, 2315.19911923822, 2310.02207362889, 2308.98276734856]
    )
    assert table.p == approx([NAN, 0.00208037581710402, 0.121876528676235, 0.0741683048985101])


def test_anova_list(lung, f_ps):
    table = r.anova(r.survreg(S + "sex", lung), f_ps)
    assert table.resid_df == approx([225, 222.449658567367])
    assert table.loglik == approx([2297.30312936469, 2291.58875512625])
    assert table.test_labels == ["", "+pspline(age, df = 3)"]
    assert table.df == approx([NAN, 2.55034143263268])
    assert table.deviance == approx([NAN, 5.71437423843645])
    assert table.p == approx([NAN, 0.0918407374421038])


# --- logLik and friends ------------------------------------------------------------------------


def test_loglik_uses_the_fractional_df(f_ps, lung):
    assert r.loglik(f_ps) == approx(-1145.79437756313)
    assert r.degrees_freedom(f_ps) == approx(5.55034143263268)
    assert r.aic(f_ps) == approx(2302.68943799152)
    assert r.aic(f_ps, k=3) == approx(2308.23977942415)
    assert r.bic(f_ps) == approx(2321.72347712272)
    assert r.extract_aic(f_ps) == approx([5.55034143263268, 2302.68943799152])
    assert r.nobs(f_ps) == 228
    assert r.df_residual(f_ps) == approx(222.449658567367)
    plain = r.survreg(S + "age + sex", lung)
    assert (r.degrees_freedom(plain), r.df_residual(plain)) == (4, 224)
    assert isinstance(r.degrees_freedom(plain), int)
    assert isinstance(r.df_residual(plain), int)


# --- predict, residuals and the rest ---------------------------------------------------------


def test_predict(f_ps):
    lp = r.predict(f_ps, type="lp", se_fit=True)
    assert lp.fit[:5] == approx(
        [5.71345267741673, 5.87720770178742, 5.92388651523252, 5.92835354437333,
         5.94207633434705]
    )  # fmt: skip
    assert lp.se_fit[:2] == approx([0.117559537131727, 0.0976583708205022])
    response = r.predict(f_ps, type="response", se_fit=True)
    assert response.fit[:2] == approx([302.915133075289, 356.811525212041])
    assert response.se_fit[:2] == approx([35.6105628345263, 34.8456322421865])
    quantile = r.predict(f_ps, type="quantile", p=[0.1, 0.5, 0.9], se_fit=True)
    assert quantile.fit[0] == approx([55.999133236796, 230.099644944015, 566.288339123827])
    assert quantile.se_fit[0] == approx([9.37443861755558, 27.845684304765, 67.2497492754164])
    uquantile = r.predict(f_ps, type="uquantile", p=0.5, se_fit=True)
    assert uquantile.fit[:2] == approx([5.43851245398935, 5.60226747836005])
    assert uquantile.se_fit[:2] == approx([0.121015763894551, 0.101163002892076])
    terms = r.predict(f_ps, type="terms", se_fit=True)
    assert terms.fit[0] == approx([-0.177810293506352, -0.150424525160005])
    assert terms.se_fit[0] == approx([0.160749461504986, 0.0191322869650913])
    assert r.model_term_names(f_ps) == ["pspline(age, df = 3)", "sex"]


def test_predict_newdata(f_ps, f_strata, f_ridge):
    # ages 30 and 90 lie outside the boundary knots: the basis extrapolates linearly
    new = {"age": [30, 40, 55, 70, 85, 90], "sex": [1, 2, 1, 2, 1, 2]}
    lp = r.predict(f_ps, new, type="lp", se_fit=True)
    assert lp.fit == approx(
        [6.7944994952506, 6.70963441784024, 5.92102634425018, 6.21375439906028,
         5.18299218080553, 5.29441746602627]
    )  # fmt: skip
    assert lp.se_fit == approx(
        [0.944393555619267, 0.381025690811457, 0.114424110248264, 0.124723788348632,
         0.520373063881881, 0.815262155228203]
    )  # fmt: skip
    assert r.predict(f_ps, new, type="response") == approx(
        [892.922235922185, 820.270708292105, 372.794132710534, 499.573332467857,
         178.215266293895, 199.221538842833]
    )  # fmt: skip
    assert r.predict(f_ps, new, type="quantile", p=0.5) == approx(
        [678.279382619171, 623.092008708187, 283.180957989925, 379.484660463568,
         135.375440247933, 151.332173099264]
    )  # fmt: skip
    assert r.predict(f_ps, new, type="terms")[0] == approx([0.90323652432752, -0.150424525160005])

    quantile = r.predict(
        f_strata, {"age": [50, 70], "sex": [1, 2]}, type="quantile", p=0.5, se_fit=True
    )
    assert quantile.fit == approx([348.143562802868, 321.047216184426])
    assert quantile.se_fit == approx([45.8254891885977, 32.9370617225873])

    lp = r.predict(f_ridge, new, type="lp", se_fit=True)
    assert lp.fit == approx(
        [6.28819196804688, 6.54671340010777, 5.98289884961662, 6.18036165799145,
         5.61654710750031, 5.93612716324724]
    )  # fmt: skip
    assert lp.se_fit == approx(
        [0.243273644929016, 0.188905383260956, 0.0928969522428197, 0.115529066928607,
         0.165730812064884, 0.215954416923269]
    )  # fmt: skip


def test_residuals(f_ps):
    assert r.residuals(f_ps, type="response")[:5] == approx(
        [3.08486692471115, 98.1884747879585, 636.138086038147, -165.535701672121,
         502.275378969938]
    )  # fmt: skip
    deviance = r.residuals(f_ps, type="deviance")
    assert deviance[:3] == approx([0.0135376464761823, 0.342543521323455, 2.74281792290041])
    assert sum(value * value for value in deviance) == approx(308.000411195986)
    assert r.residuals(f_ps, type="working")[:3] == approx(
        [0.0100643014039875, 0.207633632233864, 0.750151517144443]
    )
    assert r.residuals(f_ps, type="ldcase")[:3] == approx(
        [0.00387004537401937, 0.00608448721179569, 0.380657071049035]
    )
    assert r.residuals(f_ps, type="ldresp")[:3] == approx(
        [0.0251630103186225, 0.0327803411355592, 0.565227956658018]
    )
    assert r.residuals(f_ps, type="ldshape")[:3] == approx(
        [1.82644387583634e-05, 0.0115342092384076, 2.05737871360994]
    )
    dfbeta = r.residuals(f_ps, type="dfbeta")
    assert len(dfbeta[0]) == 13
    assert dfbeta[0][:3] == approx(
        [0.000464457065837899, 0.000349214450732697, 0.000653510668682336]
    )
    assert r.residuals(f_ps, type="dfbetas")[0][:3] == approx(
        [0.00059984474207393, 0.000825408403251323, 0.000962139430662173]
    )
    assert r.residuals(f_ps, type="matrix")[0] == approx(
        [-0.712611563939665, 0.0181280771269601, -1.8012255793311, -0.999816318626535,
         -0.000368606047373462, -0.0363788593812241]
    )  # fmt: skip


def test_offset_is_in_the_linear_predictors(lung):
    # R's survpenal.fit leaves the offset out of linear.predictors (lp 5.70208670271706,
    # response residual 6.50830047908886); these are R's values with it restored
    fit = r.survreg(S + "pspline(age, df = 3) + offset(off)", lung, weights="w")
    assert fit.linear_predictors[:3] == approx(
        [5.80208670271706, 5.97823046555435, 6.03039093198702]
    )
    assert r.residuals(fit, type="response")[:3] == approx(
        [-24.9895165155612, 60.2587584505925, 594.122422544991]
    )
    assert r.residuals(fit, type="deviance")[:3] == approx(
        [-0.0991387323999992, 0.18814831780872, 2.49996284310001]
    )
    assert r.predict(fit, type="response")[:3] == approx(
        [330.989516515561, 394.741241549407, 415.877577455008]
    )


def test_vcov_confint_concordance_model_matrix(f_ps):
    vcov = r.vcov(f_ps)
    assert len(vcov) == 13
    assert r.coef_names(f_ps, complete=True)[-2:] == ["sex", "Log(scale)"]
    assert [vcov[i][i] for i in range(3)] == approx(
        [0.599533473176021, 0.178997366566456, 0.461348738098326]
    )
    assert vcov[12][12] == approx(0.00383164735122676)
    assert len(r.vcov(f_ps, complete=False)) == 13
    intervals = r.confint(f_ps)
    assert [[row["lower"], row["upper"]] for row in intervals[:3]] == [
        approx([4.72826474144428, 7.7634472063603]),
        approx([-1.08180708393745, 0.576639192162843]),
        approx([-1.83228272858962, 0.830236514070547]),
    ]
    concordance = r.concordance(f_ps)
    assert concordance.concordance == approx(0.598655940841411)
    assert concordance.var == approx(0.000650419157417025)
    assert r.model_matrix(f_ps)["data"][0] == approx(
        [1, 0, 0, 0, 0, 0, 0.0194133849849699, 0.471866208845343, 0.486399520377661,
         0.0223208857920267, 0, 1]
    )  # fmt: skip
