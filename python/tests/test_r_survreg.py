"""R-style ``survreg`` facade: fitting, methods and the d/p/q/rsurvreg helpers.

Reference numbers come from R survival 3.8.11 (``survreg``, ``predict.survreg``,
``residuals.survreg``, ``anova.survreg``, ``summary.survreg``, ``dsurvreg``) on the bundled
``lung`` and ``tobin`` data with ``na.action = na.omit``.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
_survreg = survival.r._survreg
SurvregDistribution = survival._survival.SurvregDistribution
SurvregFamily = survival._survival.SurvregFamily
SurvregTransform = survival._survival.SurvregTransform

NEWDATA = {"age": [50, 70], "sex": [1, 2]}


def _approx_matrix(actual, expected, **tolerance):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == pytest.approx(expected_row, **tolerance)


def _frame(loader):
    return {key: list(values) for key, values in loader().items() if not key.startswith("_")}


@pytest.fixture(scope="module")
def lung():
    return _frame(survival.datasets.load_lung)


@pytest.fixture(scope="module")
def tobin():
    return _frame(survival.datasets.load_tobin)


@pytest.fixture(scope="module")
def lung_weibull(lung):
    return r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit")


# --- fitting ---------------------------------------------------------------------------------


def test_survreg_weibull_matches_r_object(lung_weibull):
    fit = lung_weibull
    assert isinstance(fit, _survreg.SurvregModelResult)
    assert r.coef(fit) == pytest.approx([6.27485305842, -0.0122570255889, 0.382085139659])
    assert fit.coefficients == r.coef(fit)
    assert fit.scale == pytest.approx([0.754050947641])
    assert fit.loglik == pytest.approx([-1153.85118809, -1147.05443143])
    assert r.loglik(fit) == pytest.approx(-1147.05443143)
    assert (fit.iter, fit.df, fit.df_residual, fit.n, fit.idf) == (5, 4, 224, 228, 2)
    assert (r.degrees_freedom(fit), r.df_residual(fit), r.nobs(fit)) == (4, 224, 228)
    assert fit.var[0][0] == pytest.approx(0.231714143303)
    assert r.vcov(fit) == fit.var
    assert len(r.vcov(fit, complete=False)) == 3
    assert r.coef_names(fit) == ["(Intercept)", "age", "sex"]
    assert r.coef_names(fit, complete=True) == ["(Intercept)", "age", "sex", "Log(scale)"]
    assert fit.icoef == pytest.approx([6.0349039102, math.log(0.759393601108)])
    assert fit.means == pytest.approx([1.0, 62.4473684211, 1.39473684211])
    assert fit.dist == "weibull"
    assert fit.parms is None
    assert not fit.robust
    assert fit.distribution.name == "Weibull"
    assert fit.term_labels == ("age", "sex")
    assert fit.assign == (0, 1, 2)
    assert fit.formula == "Surv(time, status) ~ age + sex"
    assert len(fit.linear_predictors) == 228
    assert len(fit.y) == 228
    assert r.aic(fit) == pytest.approx(-2 * -1147.05443143 + 8)
    assert repr(fit).startswith("SurvregModelResult(formula='Surv(time, status) ~ age + sex', ")


def test_survreg_fixed_scale(lung):
    fit = r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", scale=1)
    assert r.coef(fit) == pytest.approx([6.3596715418, -0.0156187110404, 0.48093492396])
    assert fit.scale == [1.0]
    assert (fit.df, fit.df_residual, fit.idf) == (3, 225, 1)
    assert len(r.vcov(fit)) == 3
    assert r.coef_names(fit, complete=True) == ["(Intercept)", "age", "sex"]
    assert fit.loglik == pytest.approx([-1162.33817579, -1156.09903714])
    assert len(r.residuals(fit, type="dfbeta")[0]) == 3


def test_survreg_exponential_and_rayleigh_fix_the_scale(lung):
    exponential = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="exponential"
    )
    assert r.coef(exponential) == pytest.approx([6.3596715418, -0.0156187110404, 0.48093492396])
    assert exponential.scale == [1.0]
    assert exponential.df == 3
    assert exponential.loglik[1] == pytest.approx(-1156.09903714)

    with pytest.warns(RuntimeWarning, match="Exponential has a fixed scale"):
        ignored = r.survreg(
            "Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="exp", scale=2
        )
    assert r.coef(ignored) == pytest.approx(r.coef(exponential))

    rayleigh = r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", dist="ray")
    assert rayleigh.scale == [0.5]
    assert rayleigh.distribution.name == "Rayleigh"


def test_survreg_strata_scales_and_labels(lung):
    fit = r.survreg("Surv(time, status) ~ age + strata(sex) + sex", data=lung, na_action="omit")
    assert fit.scale == pytest.approx([0.803442799135, 0.642536589059])
    assert fit.strata_levels == ("sex=1", "sex=2")
    assert fit.term_labels == ("age", "strata(sex)", "sex")
    assert fit.strata_term == 2
    assert fit.assign == (0, 1, 3)
    assert r.coef_names(fit, complete=True) == [
        "(Intercept)",
        "age",
        "sex",
        "Log(scale)",
        "Log(scale)",
    ]
    assert fit.df == 5
    assert fit.df_residual == 223

    summary = r.model_summary(fit)
    assert summary["coefficient_names"] == ["(Intercept)", "age", "sex", "sex=1", "sex=2"]
    assert [row["coef"] for row in summary["coefficients"]] == pytest.approx(
        [6.21020507, -0.01120008, 0.37043715, -0.21884929, -0.44233152]
    )
    assert [row["se"] for row in summary["coefficients"]] == pytest.approx(
        [0.471152566, 0.006825942, 0.117360605, 0.074958411, 0.109408127]
    )
    assert summary["scales"] == pytest.approx(fit.scale)

    data = dict(lung, sexf=["m" if value == 1 else "f" for value in lung["sex"]])
    character = r.survreg("Surv(time, status) ~ age + strata(sexf)", data=data, na_action="omit")
    assert character.strata_levels == ("f", "m")
    mixed = r.survreg("Surv(time, status) ~ age + strata(sexf, sex)", data=data, na_action="omit")
    assert mixed.strata_levels == ("sexf=f, sex=2", "sexf=m, sex=1")

    with pytest.raises(ValueError, match="not valid with multiple strata"):
        r.survreg("Surv(time, status) ~ age + strata(sex)", data=lung, na_action="omit", scale=1)


def test_survreg_cluster_and_robust_variance(lung, lung_weibull):
    clustered = r.survreg(
        "Surv(time, status) ~ age + sex + cluster(inst)", data=lung, na_action="omit"
    )
    assert clustered.n == 227  # one institution is missing
    assert clustered.robust
    assert clustered.var[0][0] == pytest.approx(0.172296116856)
    assert clustered.naive_var[0][0] == pytest.approx(0.233558196788)
    assert r.vcov(clustered) == clustered.var
    assert len(clustered.cluster) == 227

    by_argument = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", cluster=lung["inst"]
    )
    _approx_matrix(by_argument.var, clustered.var)

    robust = r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", robust=True)
    assert robust.var[0][0] == pytest.approx(0.240682927222)
    assert robust.naive_var[0][0] == pytest.approx(0.231714143303)
    assert robust.cluster is None
    summary = r.model_summary(robust)
    assert summary["robust"] is True
    first = summary["coefficients"][0]
    assert first["se"] == pytest.approx(0.490594463098)
    assert first["naive_se"] == pytest.approx(0.481366952857)
    assert first["robust_se"] == first["se"]
    assert first["z"] == pytest.approx(12.7903054975)
    assert first["p"] == pytest.approx(1.85742323572e-37)

    not_robust = r.survreg(
        "Surv(time, status) ~ age + sex + cluster(inst)",
        data=lung,
        na_action="omit",
        robust=False,
    )
    assert not not_robust.robust
    assert not_robust.naive_var is None
    assert not_robust.cluster is None
    assert not lung_weibull.robust


def test_survreg_custom_distributions(lung, lung_weibull):
    as_list = {"name": "Mine", "dist": "extreme", "trans": "log"}
    custom = r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist=as_list)
    assert r.coef(custom) == pytest.approx(r.coef(lung_weibull))
    assert custom.loglik == pytest.approx(lung_weibull.loglik)
    assert custom.dist.name == "Mine"
    assert custom.distribution.transform == SurvregTransform.Log

    as_object = SurvregDistribution.custom(
        "Mine", SurvregFamily.ExtremeValue, SurvregTransform.Log, scale=1.0
    )
    fixed = r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist=as_object)
    assert fixed.scale == [1.0]
    assert fixed.df == 3

    assert set(_survreg.survreg_distributions) == {
        "extreme",
        "logistic",
        "gaussian",
        "weibull",
        "exponential",
        "rayleigh",
        "loggaussian",
        "lognormal",
        "loglogistic",
        "t",
    }
    assert _survreg.survregDtest(as_list) is True
    assert _survreg.survregDtest(_survreg.survreg_distributions["lognormal"]) is True
    assert _survreg.survregDtest({"name": "x"}) is False
    assert _survreg.survregDtest({"name": "x"}, verbose=True) == [
        "custom densities are not supported; give 'dist' (a built-in name)"
    ]
    with pytest.raises(ValueError, match="trans must be 'log' or 'identity'"):
        r.survreg(
            "Surv(time, status) ~ age", data=lung, na_action="omit", dist={**as_list, "trans": 1}
        )
    with pytest.raises(TypeError, match="Invalid distribution object"):
        r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", dist=3)


def test_survreg_t_distribution_parms(lung):
    default = r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="t")
    assert default.parms == [4.0]
    fit = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="t", parms=6
    )
    assert r.coef(fit) == pytest.approx([335.145836644, -2.70112809541, 125.475192919])
    assert fit.scale == pytest.approx([207.854020941])
    assert fit.loglik[1] == pytest.approx(-1178.82739437)
    assert fit.parms == [6.0]
    named = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="t", parms={"df": 6}
    )
    assert r.coef(named) == pytest.approx(r.coef(fit))
    assert r.model_summary(fit)["parms"] == "Student-t distribution: parmameters= 6.0"

    with pytest.raises(ValueError, match="Degrees of freedom must be >=3"):
        r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", dist="t", parms=2)
    with pytest.raises(ValueError, match="has no optional parameters"):
        r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", parms=3)


def test_survreg_control_and_init(lung, lung_weibull):
    control = _survreg.survreg_control(iter_max=0)
    assert (control.iter_max, control.rel_tolerance, control.toler_chol) == (0, 1e-9, 1e-10)
    assert _survreg.survreg_control(maxiter=7).iter_max == 7
    dotted = _survreg.survreg_control(**{"iter.max": 3, "rel.tolerance": 1e-5})
    assert (dotted.iter_max, dotted.rel_tolerance) == (3, 1e-5)
    assert _survreg.survreg_control(max_iter=2, eps=1e-6, tol_chol=1e-8).toler_chol == 1e-8
    with pytest.raises(TypeError, match="unused argument"):
        _survreg.survreg_control(bogus=1)
    with pytest.raises(ValueError, match="use only one of"):
        _survreg.survreg_control(iter_max=1, **{"iter.max": 2})

    start = [6.5, 0.0, -0.5, -0.2]
    at_start = r.survreg(
        "Surv(time, status) ~ age + sex",
        data=lung,
        na_action="omit",
        init=start,
        control={"iter.max": 0},
    )
    assert r.coef(at_start) == pytest.approx(start[:3])
    assert at_start.scale == pytest.approx([math.exp(-0.2)])
    assert at_start.loglik == pytest.approx([-1153.85118809, -1186.43678027])
    assert at_start.iter == 0

    location_only = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", init=start[:3], scale=0.9
    )
    assert location_only.scale == [0.9]
    with pytest.raises(ValueError, match="Wrong length for initial parameters"):
        r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", init=[1.0])

    with pytest.warns(RuntimeWarning, match="Ran out of iterations and did not converge"):
        short = r.survreg(
            "Surv(time, status) ~ age + sex", data=lung, na_action="omit", **{"iter.max": 2}
        )
    assert short.iter == 2
    assert not short.converged
    via_options = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", maxiter=30, eps=1e-9
    )
    assert r.coef(via_options) == pytest.approx(r.coef(lung_weibull))
    with pytest.raises(TypeError, match="unused argument"):
        r.survreg(
            "Surv(time, status) ~ age", data=lung, na_action="omit", control=control, maxiter=3
        )


def test_survreg_weights_offset_and_collapsed_residuals(lung):
    weights = [1, 2, 0.5, 1.5] * 57
    weighted = r.survreg(
        "Surv(time, status) ~ age + sex", data=lung, na_action="omit", weights=weights
    )
    assert r.model_weights(weighted) == weights
    collapsed = r.residuals(weighted, type="dfbeta", weighted=True, collapse=lung["sex"])
    assert collapsed[0] == pytest.approx(
        [-0.0722227043513, 0.0007899802488, 0.0154302664141, 0.0394388601417]
    )
    assert collapsed[1] == pytest.approx([-value for value in collapsed[0]], abs=1e-9)
    assert r.residuals(weighted, type="response", collapse=lung["sex"]) == pytest.approx(
        [-7414.24580574, -16691.5026995]
    )
    labelled = r.residuals(
        weighted, type="response", collapse=["b" if value == 1 else "a" for value in lung["sex"]]
    )
    assert labelled == pytest.approx([-16691.5026995, -7414.24580574])  # rowsum orders groups
    with pytest.raises(ValueError, match="Wrong length for 'collapse'"):
        r.residuals(weighted, type="response", collapse=[1, 2, 3])

    in_formula = r.survreg("Surv(time, status) ~ age + offset(sex)", data=lung, na_action="omit")
    as_argument = r.survreg(
        "Surv(time, status) ~ age", data=lung, na_action="omit", offset=lung["sex"]
    )
    assert r.coef(in_formula) == pytest.approx(r.coef(as_argument))
    assert in_formula.term_labels == ("age",)
    with pytest.raises(ValueError, match="only one of formula offset"):
        r.survreg(
            "Surv(time, status) ~ age + offset(sex)",
            data=lung,
            na_action="omit",
            offset=lung["sex"],
        )


# --- predict ---------------------------------------------------------------------------------


def test_predict_survreg_types_and_shapes(lung_weibull):
    fit = lung_weibull
    assert r.predict(fit)[:2] == pytest.approx([314.164993370, 338.140151189])
    assert r.predict(fit, type="lp") == pytest.approx(fit.linear_predictors)
    assert r.predict(fit, type="link") == r.predict(fit, type="linear")
    assert r.fitted(fit) == r.predict(fit)
    linear = r.predict(fit, NEWDATA, type="lp")
    assert linear == pytest.approx([6.04408691863, 6.18103154651])
    response = r.predict(fit, NEWDATA, se_fit=True)
    assert response.se_fit == pytest.approx([49.8387551502, 55.9758588568])
    assert response.fit == pytest.approx([math.exp(value) for value in linear])

    median = r.predict(fit, NEWDATA, type="quantile", p=0.5, se_fit=True)
    assert median.fit == pytest.approx([319.806940741, 366.743293817])
    assert median.se_fit == pytest.approx([38.2338658641, 42.5897383801])
    quantiles = r.predict(fit, NEWDATA, type="quantile", p=[0.1, 0.5, 0.9])
    assert len(quantiles) == 2
    assert [row[1] for row in quantiles] == pytest.approx(median.fit)
    assert r.predict(fit, {"age": [50], "sex": [1]}, type="uquantile", p=[0.1, 0.9]) == (
        pytest.approx([4.34719530293, 6.67298987433])
    )
    training = r.predict(fit, type="quantile")  # p defaults to c(.1, .9)
    assert len(training) == 228
    assert len(training[0]) == 2

    terms = r.predict(fit, NEWDATA, type="terms", se_fit=True)
    _approx_matrix(
        terms.fit, [[0.152567713252, -0.150823081444], [-0.0925727985271, 0.231262058215]]
    )
    assert len(terms.se_fit) == 2
    assert len(terms.se_fit[0]) == 2
    _approx_matrix(
        r.predict(fit, NEWDATA, type="terms", terms="sex"), [[-0.150823081444], [0.231262058215]]
    )
    _approx_matrix(
        r.predict(fit, NEWDATA, type="terms", terms=1), [[0.152567713252], [-0.0925727985271]]
    )
    assert r.model_term_names(fit) == ["age", "sex"]

    with pytest.raises(ValueError, match="'type' should be one of"):
        r.predict(fit, type="risk")
    with pytest.raises(ValueError, match="unknown model term"):
        r.predict(fit, NEWDATA, type="terms", terms="ph.ecog")
    with pytest.raises(ValueError, match="not an argument of predict.survreg"):
        r.predict(fit, NEWDATA, reference="strata")
    with pytest.raises(TypeError, match="newdata must be a data frame"):
        r.predict(fit, [[1.0, 50.0, 1.0]])


def test_predict_survreg_newdata_with_strata(lung):
    fit = r.survreg("Surv(time, status) ~ age + strata(sex) + sex", data=lung, na_action="omit")
    quantiles = r.predict(fit, NEWDATA, type="quantile", p=[0.1, 0.5, 0.9], se_fit=True)
    _approx_matrix(
        quantiles.fit,
        [
            [67.5316693858938, 306.7907507794, 804.914009209731],
            [112.29510099832, 376.747658066852, 814.822212857738],
        ],
    )
    _approx_matrix(
        quantiles.se_fit,
        [
            [12.6178741633356, 37.7558457656515, 101.751823333111],
            [20.7205137157321, 38.955867363891, 96.7801439948295],
        ],
    )
    assert r.predict(fit, NEWDATA, type="lp") == pytest.approx([6.02063808847565, 6.16707358317737])
    with pytest.raises(ValueError, match="unknown strata level"):
        r.predict(fit, {"age": [50], "sex": [3]}, type="quantile")


# --- residuals -------------------------------------------------------------------------------


def test_residuals_survreg_all_types(lung_weibull):
    fit = lung_weibull
    response = r.residuals(fit, type="response")
    assert response[:2] == pytest.approx([-8.16499336963, 116.859848811])
    assert len(r.residuals(fit, type="deviance")) == 228
    assert len(r.residuals(fit, type="working")) == 228
    assert r.residuals(fit, type="ldcase")[:3] == pytest.approx(
        [0.00375040307336, 0.00582868986098, 0.233591514655]
    )
    assert len(r.residuals(fit, type="ldresp")) == 228
    assert len(r.residuals(fit, type="ldshape")) == 228
    matrix = r.residuals(fit, type="matrix")
    assert matrix[0] == pytest.approx(
        [
            -0.718307403688,
            -0.045513589091,
            -1.69836899321,
            -0.99880148144,
            -0.00237623140413,
            0.090237083782,
        ]
    )
    dfbeta = r.residuals(fit, type="dfbeta")
    assert len(dfbeta[0]) == 4
    dfbetas = r.residuals(fit, type="dfbetas")
    standard_errors = [math.sqrt(fit.var[idx][idx]) for idx in range(4)]
    assert dfbetas[0] == pytest.approx(
        [value / se for value, se in zip(dfbeta[0], standard_errors, strict=True)]
    )
    assert len(r.residuals(fit, type="dfbeta", rsigma=False)[0]) == 3
    assert r.residuals(fit, type="dev") == r.residuals(fit, type="deviance")
    assert r.residuals(fit, type="response", weighted=True) == response  # no weights: no-op
    with pytest.raises(ValueError, match="'type' should be one of"):
        r.residuals(fit, type="martingale")
    with pytest.raises(ValueError, match="terms is only supported for Cox"):
        r.residuals(fit, type="response", terms="age")


# --- anova -----------------------------------------------------------------------------------


def test_anova_survreg_sequential_terms(lung, lung_weibull):
    table = _survreg.anova_survreg(lung_weibull)
    assert table.terms == ["NULL", "age", "sex"]
    assert table.loglik == pytest.approx([2307.70237618, 2303.78770423, 2294.10886286])
    assert table.resid_df == [226, 225, 224]
    assert math.isnan(table.df[0])
    assert table.df[1:] == [1.0, 1.0]
    assert table.deviance[1:] == pytest.approx([3.91467195, 9.67884137])
    assert table.p[1:] == pytest.approx([0.0478663565064, 0.00186402139024])
    frame = r.as_data_frame(table)
    assert list(frame) == ["Df", "Deviance", "Resid. Df", "-2*LL", "Pr(>Chi)"]
    assert "Scale estimated" in table.heading
    assert _survreg.anova_survreg(lung_weibull, test="none").p is None

    stratified = r.survreg(
        "Surv(time, status) ~ age + strata(sex) + sex", data=lung, na_action="omit"
    )
    table = _survreg.anova_survreg(stratified)
    assert table.terms == ["NULL", "age", "strata(sex)", "sex"]
    assert table.deviance[1:] == pytest.approx(
        [3.9146719483806, 2.15017143626028, 10.2579304565488]
    )
    assert table.resid_df == [226, 225, 224, 223]
    assert table.p[1:] == pytest.approx(
        [0.0478663565063668, 0.142553971171007, 0.00136098270268315]
    )

    intercept_only = r.survreg("Surv(time, status) ~ 1", data=lung, na_action="omit")
    table = _survreg.anova_survreg(intercept_only)
    assert table.terms == ["NULL"]
    assert table.resid_df == [226]


def test_anova_survreg_model_list(lung, lung_weibull):
    small = r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit")
    large = r.survreg("Surv(time, status) ~ age + strata(sex) + sex", data=lung, na_action="omit")
    table = _survreg.anova_survreg(small, large)
    assert table.terms == ["age", "age + strata(sex) + sex"]
    assert table.test_labels == ["", "+strata(sex)+sex"]
    assert table.resid_df == [225, 223]
    assert table.loglik == pytest.approx([2303.788, 2291.380], abs=1e-3)
    assert table.df[1] == 2.0
    assert table.deviance[1] == pytest.approx(12.4081, abs=1e-4)
    assert table.p[1] == pytest.approx(0.002021226, rel=1e-6)
    reversed_table = _survreg.anova_survreg([large, small])
    assert reversed_table.test_labels == ["", "-strata(sex)-sex"]
    assert reversed_table.df[1] == -2.0
    assert reversed_table.p[1] == pytest.approx(0.002021226, rel=1e-6)
    assert list(r.as_data_frame(table)) == [
        "Terms",
        "Resid. Df",
        "-2*LL",
        "Test",
        "Df",
        "Deviance",
        "Pr(>Chi)",
    ]
    with pytest.raises(TypeError, match="requires survreg model fits"):
        _survreg.anova_survreg(lung_weibull, object())
    with pytest.raises(ValueError, match="'test' should be one of"):
        _survreg.anova_survreg(lung_weibull, test="F")


# --- censoring types and the model frame -------------------------------------------------------


def test_survreg_left_censored_gaussian_tobin(tobin):
    fit = r.survreg(
        'Surv(durable, durable > 0, type = "left") ~ age + quant', data=tobin, dist="gaussian"
    )
    assert r.coef(fit) == pytest.approx([15.1448663607, -0.129059284097, -0.0455416629543])
    assert fit.scale == pytest.approx([5.57253976309])
    assert fit.loglik[1] == pytest.approx(-28.9401331997)
    assert fit.y.type == "left"
    assert r.predict(fit, type="response") == pytest.approx(fit.linear_predictors)


def test_survreg_interval2_missing_endpoints_are_censoring():
    data = {
        "left": [1, 2, None, 4, 5, 3, 6, None, 2, 7],
        "right": [3, 4, 2, 6, 5, None, 8, 5, 3, None],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    }
    fit = r.survreg('Surv(left, right, type = "interval2") ~ g', data=data, na_action="omit")
    assert fit.n == 10  # a missing endpoint is a censoring code, not a missing response
    events = list(fit.y.event)
    assert (events.count(3), events.count(2), events.count(0), events.count(1)) == (5, 2, 2, 1)

    with_missing = {key: [*values, None] for key, values in data.items()}
    with_missing["g"][-1] = "a"
    dropped = r.survreg(
        'Surv(left, right, type = "interval2") ~ g', data=with_missing, na_action="omit"
    )
    assert dropped.n == 10
    assert r.coef(dropped) == pytest.approx(r.coef(fit))
    with pytest.raises(ValueError, match="missing values"):
        r.survreg('Surv(left, right, type = "interval2") ~ g', data=with_missing, na_action="fail")


def test_survreg_intercept_only_and_model_pieces(lung):
    fit = r.survreg("Surv(time, status) ~ 1", data=lung, na_action="omit", model=True, x=True)
    assert r.coef(fit) == pytest.approx([6.0349039102])
    assert fit.scale == pytest.approx([0.759393601108])
    assert fit.loglik == pytest.approx([-1153.85118809, -1153.85118809])
    assert fit.df_residual == 226
    assert fit.term_labels == ()
    assert fit.x[:2] == [[1.0], [1.0]]
    matrix = r.model_matrix(fit)
    assert matrix["columns"] == ["(Intercept)"]
    assert matrix["assign"] == [0]
    frame = r.model_frame(fit)
    assert set(frame) >= {"time", "status"}
    assert r.model_formula(fit) == "Surv(time, status) ~ 1"
    assert r.extract_aic(fit) == pytest.approx([2.0, r.aic(fit)])
    assert r.bic(fit) == pytest.approx(-2 * fit.loglik[1] + 2 * math.log(228))
    intervals = r.confint(fit)
    assert intervals[0]["lower"] < 6.0349039102 < intervals[0]["upper"]
    without_y = r.survreg("Surv(time, status) ~ 1", data=lung, na_action="omit", y=False)
    assert without_y.y_response is None
    scored = r.survreg("Surv(time, status) ~ 1", data=lung, na_action="omit", score=True)
    assert len(scored.score) == 2


def test_survreg_subset_and_na_action(lung):
    men = [value == 1 for value in lung["sex"]]
    subset = r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", subset=men)
    assert subset.n == sum(men)
    with pytest.raises(ValueError, match="missing values"):
        r.survreg("Surv(time, status) ~ age + ph.ecog", data=lung)
    omitted = r.survreg("Surv(time, status) ~ age + ph.ecog", data=lung, **{"na.action": "omit"})
    assert omitted.n == 227
    named_weights = r.survreg(
        "Surv(time, status) ~ age", data=dict(lung, w=[1.0] * 228), na_action="omit", weights="w"
    )
    assert r.model_weights(named_weights) == [1.0] * 228


def test_survreg_matrix_input_uses_the_design_as_given(lung):
    response = survival.Surv(lung["time"], [value - 1 for value in lung["status"]])
    fit = r.survreg(response, x={"age": lung["age"], "sex": lung["sex"]})
    assert r.coef_names(fit) == ["age", "sex"]  # no intercept unless the caller adds one
    assert fit.formula is None
    assert fit.term_labels == ("age", "sex")
    prediction = r.predict(fit, {"age": [50, 70], "sex": [1, 2]}, type="lp")
    coefficients = r.coef(fit)
    assert prediction[0] == pytest.approx(50 * coefficients[0] + coefficients[1])
    assert len(r.predict(fit, {"age": [50], "sex": [1]}, type="terms", terms="sex")[0]) == 1
    assert r.predict(fit, [[50.0, 1.0]], type="lp") == pytest.approx(prediction[:1])
    with_intercept = r.survreg(
        response, x=[[1.0, age, sex] for age, sex in zip(lung["age"], lung["sex"], strict=True)]
    )
    assert r.coef_names(with_intercept) == ["x1", "x2", "x3"]
    with pytest.raises(ValueError, match="subset and na_action require a formula"):
        r.survreg(response, x=[[1.0]] * 228, subset=[0, 1])


def test_survreg_argument_errors(lung):
    with pytest.raises(ValueError, match="'dist' should be one of"):
        r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", dist="xx")
    with pytest.raises(ValueError, match="Invalid scale value"):
        r.survreg("Surv(time, status) ~ age", data=lung, na_action="omit", scale=-1)
    with pytest.raises(ValueError, match="start-stop type Surv objects are not supported"):
        r.survreg(
            "Surv(start, stop, event) ~ x",
            data={
                "start": [0] * 5,
                "stop": [1, 2, 3, 4, 5],
                "event": [1] * 5,
                "x": [1, 2, 3, 4, 5],
            },
        )
    with pytest.raises(ValueError, match="multi-state survival is not supported"):
        r.survreg(
            'Surv(time, state, type = "mstate") ~ x',
            data={"time": [1, 2, 3, 4], "state": ["a", "b", "censor", "a"], "x": [1, 2, 3, 4]},
        )
    with pytest.raises(TypeError, match="a formula argument is required"):
        r.survreg(None, data=lung)
    with pytest.raises(ValueError, match="Invalid survival times for this distribution"):
        r.survreg(
            "Surv(time, status) ~ x",
            data={"time": [0, 1, 2, 3], "status": [1, 1, 0, 1], "x": [1, 2, 3, 4]},
        )
    with pytest.raises(ValueError, match="a formula cannot have multiple cluster terms"):
        r.survreg(
            "Surv(time, status) ~ age + cluster(inst) + cluster(sex)", data=lung, na_action="omit"
        )
    with pytest.warns(RuntimeWarning, match="cluster appears both"):
        r.survreg(
            "Surv(time, status) ~ age + cluster(sex)",
            data=lung,
            na_action="omit",
            cluster=lung["sex"],
        )


# --- summary -----------------------------------------------------------------------------------


def test_model_summary_survreg_structure(lung_weibull):
    summary = r.model_summary(lung_weibull)
    assert summary["model_type"] == "survreg"
    assert summary["coefficient_names"] == ["(Intercept)", "age", "sex", "Log(scale)"]
    assert summary["location_coefficient_names"] == ["(Intercept)", "age", "sex"]
    assert summary["location_coefficients"] == pytest.approx(r.coef(lung_weibull))
    assert summary["scale"] == pytest.approx(0.754050947641)
    assert summary["scales"] == pytest.approx([0.754050947641])
    assert summary["distribution"] == "Weibull"
    assert summary["parms"] == "Weibull distribution"
    assert (summary["df"], summary["n"], summary["iter"], summary["idf"]) == (4, 228, 5, 2)
    assert summary["loglik"] == pytest.approx(-1147.05443143)
    assert summary["chi"] == pytest.approx(2 * (-1147.05443143 + 1153.85118809))
    assert summary["robust"] is False
    rows = summary["coefficients"]
    assert [row["name"] for row in rows] == summary["coefficient_names"]
    assert rows[3]["coef"] == pytest.approx(math.log(0.754050947641))
    assert rows[3]["se"] == pytest.approx(math.sqrt(lung_weibull.var[3][3]))
    assert rows[1]["p"] == pytest.approx(0.0781188632, rel=1e-6)


# --- dsurvreg, psurvreg, qsurvreg, rsurvreg ---------------------------------------------------


def test_survreg_distribution_functions_match_r():
    assert r.dsurvreg([1.0, 2.0], mean=0.5, scale=1.2) == pytest.approx([0.2841569, 0.1512009])
    assert r.psurvreg([1.0, 2.0], mean=0.5, scale=1.2) == pytest.approx([0.4827560, 0.6910677])
    assert r.qsurvreg([0.25, 0.5, 0.75], mean=0.5, scale=1.2) == pytest.approx(
        [0.3696942, 1.0620325, 2.4399099]
    )
    assert r.dsurvreg([1.0, 2.0], 0.5, 1.2, "lognormal") == pytest.approx([0.3048103, 0.1640866])
    assert r.psurvreg([1.0, 2.0], 0.5, 1.2, "lognormal") == pytest.approx([0.3384611, 0.5639360])
    assert r.qsurvreg([0.25, 0.5, 0.75], 0.5, 1.2, "lognormal") == pytest.approx(
        [0.7338962, 1.6487213, 3.7039051]
    )
    assert r.dsurvreg([-1.0, 0.0, 1.0], mean=0.0, distribution="gaussian") == pytest.approx(
        [0.2419707, 0.3989423, 0.2419707]
    )
    assert r.psurvreg([-1.0, 0.0, 1.0], mean=0.0, distribution="gaussian") == pytest.approx(
        [0.1586553, 0.5, 0.8413447]
    )
    assert r.qsurvreg([0.25, 0.5, 0.75], mean=0.0, distribution="gaussian") == pytest.approx(
        [-0.6744898, 0.0, 0.6744898]
    )
    assert r.dsurvreg([1.0, 2.0], 0.0, 1.0, "t", parms=5) == pytest.approx([0.2196798, 0.06509031])
    assert r.psurvreg([1.0, 2.0], 0.0, 1.0, "t", parms=5) == pytest.approx([0.8183913, 0.9490303])
    assert r.qsurvreg([0.25, 0.5], 0.0, 1.0, "t", parms={"df": 5}) == pytest.approx(
        [-0.7266868, 0.0]
    )
    assert r.dsurvreg([1.0], 0.5, 1.2, "loggaussian") == r.dsurvreg([1.0], 0.5, 1.2, "lognormal")
    assert r.dsurvreg([1.0], 0.5, 1.2, "rayleigh") == r.dsurvreg([1.0], 0.5, 1.2, "weibull")
    assert r.qsurvreg([0.5, 0.5], mean=[0.0, 1.0], scale=1.0, distribution="gaussian") == [
        0.0,
        1.0,
    ]
    assert all(math.isnan(value) for value in r.dsurvreg([0.0, -1.0], mean=0.0))
    assert r.psurvreg([0.0], mean=0.0) == [0.0]
    assert math.isnan(r.psurvreg([-1.0], mean=0.0)[0])  # R: log(-1) is NaN
    assert r.qsurvreg([0.0, 1.0], mean=0.0) == [0.0, math.inf]

    draws = r.rsurvreg(5, mean=0.5, scale=1.2, seed=7)
    assert len(draws) == 5
    assert all(value > 0.0 for value in draws)
    assert draws == r.rsurvreg(5, mean=0.5, scale=1.2, seed=7)
    assert r.rsurvreg(0, mean=0.5) == []
    with pytest.raises(ValueError, match="n must be non-negative"):
        r.rsurvreg(-1, mean=0.5)
    with pytest.raises(ValueError, match="length"):
        r.dsurvreg([1.0, 2.0, 3.0], mean=[0.0, 1.0])
    with pytest.raises(ValueError, match="nope"):
        r.dsurvreg([1.0], mean=0.0, distribution="nope")
