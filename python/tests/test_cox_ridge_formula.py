"""Formula boundaries and naming for native fixed-theta Cox ridge fitting."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


@pytest.fixture
def ridge_data():
    return {
        "time": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8],
        "event": [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "x": [0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.9, 0.6, 1.2],
        "z": [0.3, 1.2, 0.4, 0.8, 1.1, 0.6, 0.2, 0.7, 1.4, 0.9],
        "w": [1, 2, 1, 0.5, 1.5, 1, 2, 1, 1, 1],
        "group": ["a", "b"] * 5,
        "id": list(range(10)),
    }


def _fit(rhs, data, **options):
    return survival.coxph("Surv(time, event) ~ " + rhs, data=data, robust=False, **options)


@pytest.mark.parametrize(
    ("rhs", "label", "coefficients", "suffixes"),
    [
        (
            "ridge(x,z,theta=2,scale=False)",
            "ridge(x, z, theta = 2, scale = FALSE)",
            ["ridge(x)", "ridge(z)"],
            ["x", "z"],
        ),
        ("ridge(`a b`,theta=2)", "ridge(`a b`, theta = 2)", ["ridge(`a b`)"], [""]),
        (
            "ridge(`a b`,z,theta=2)",
            "ridge(`a b`, z, theta = 2)",
            ["ridge(`a b`)", "ridge(z)"],
            ["a b", "z"],
        ),
        (
            "ridge(log(`a b`),z,theta=2)",
            "ridge(log(`a b`), z, theta = 2)",
            ["ridge(log(`a b`))", "ridge(z)"],
            ["", "z"],
        ),
        (
            "ridge(log(x),I(x+z),theta=2)",
            "ridge(log(x), I(x + z), theta = 2)",
            ["ridge(log(x))", "ridge(I(x + z))"],
            ["1", "2"],
        ),
        (
            "ridge(I((x+z)^2),theta=2)",
            "ridge(I((x + z)^2), theta = 2)",
            ["ridge(I((x + z)^2))"],
            [""],
        ),
        ("ridge(I(-x+z/2),theta=2)", "ridge(I(-x + z/2), theta = 2)", ["ridge(I(-x + z/2))"], [""]),
        ("ridge(x,theta=1e-4)", "ridge(x, theta = 1e-04)", ["ridge(x)"], [""]),
        ("ridge(x,theta=2,eps=.0001)", "ridge(x, theta = 2, eps = 1e-04)", ["ridge(x)"], [""]),
    ],
)
def test_ridge_names_follow_r_formula_and_matrix_conventions(
    ridge_data, rhs, label, coefficients, suffixes
):
    ridge_data["a b"] = ridge_data["x"]
    fit = _fit(rhs, ridge_data)
    assert survival.coef_names(fit) == coefficients
    assert survival.model_term_names(fit) == [label]
    matrix = survival.model_matrix(fit)
    assert matrix["columns"] == [label + suffix for suffix in suffixes]
    assert matrix["assign"] == [1] * len(coefficients)


@pytest.mark.parametrize(
    ("rhs", "width", "assignments"),
    [
        ("ridge(x,x,theta=2)", 2, [1, 1]),
        ("ridge(x,theta=2)+ridge(x,theta=3)", 2, [1, 2]),
        ("ridge(x,theta=2)+ridge(x,theta=2.0)", 1, [1]),
        ("ridge(x,theta=2)+ridge(x,theta=2,scale=TRUE)", 2, [1, 2]),
        ("ridge(I(x+z),theta=2)+ridge(I(x + z),theta=2)", 1, [1]),
    ],
)
def test_ridge_term_identity_preserves_duplicate_arguments_and_explicit_options(
    ridge_data, rhs, width, assignments
):
    fit = _fit(rhs, ridge_data)
    assert len(survival.coef(fit)) == width
    assert survival.model_matrix(fit)["assign"] == assignments
    assert len(survival.model_term_names(fit)) == len(set(assignments))


def test_ridge_matrix_term_has_degree_one_alongside_factor_and_interaction(ridge_data):
    fit = _fit("z:x + ridge(x,z,theta=2) + factor(group)", ridge_data)
    assert survival.model_term_names(fit) == ["ridge(x, z, theta = 2)", "factor(group)", "z:x"]
    assert survival.model_matrix(fit)["assign"] == [1, 1, 2, 3]
    assert survival.coef_names(fit) == ["ridge(x)", "ridge(z)", "factor(group)b", "z:x"]


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        ("ridge(x,theta=2,df=.5)", "only one of df or theta"),
        ("ridge(x,theta=2,theta=3)", "duplicate theta"),
        ("ridge(x,theta=nan)", "finite"),
        ("ridge(x,theta=inf)", "finite"),
        ("ridge(x,theta=-1)", "nonnegative"),
        ("ridge(x,theta=2,scale=1)", "scale must"),
        ("ridge(x,theta=2,eps=0)", "eps positive"),
        ("ridge(factor(group),theta=2)", "numeric columns"),
        ("ridge(tt(x),theta=2)", "numeric columns"),
        ("ridge(ridge(x,theta=1),theta=2)", "numeric columns"),
        ("offset(ridge(x,theta=2))", "offset"),
        ("ridge(x,theta=2,unknown=1)", "unsupported ridge"),
    ],
)
def test_ridge_rejects_invalid_or_unsupported_arguments(ridge_data, rhs, message):
    with pytest.raises(ValueError, match=message):
        _fit(rhs, ridge_data)


@pytest.mark.parametrize("operator", [":", "*", "/"])
def test_ridge_interactions_have_an_explicit_boundary(ridge_data, operator):
    with pytest.raises(NotImplementedError, match="interactions involving ridge"):
        _fit(f"z {operator} ridge(x,theta=2)", ridge_data)


@pytest.mark.parametrize(
    "function",
    ["survfit", "survdiff", "survreg", "aareg", "pyears", "concordance", "survobrien", "rttright"],
)
def test_ridge_is_not_silently_unpenalized_by_other_formula_models(ridge_data, function):
    with pytest.raises(NotImplementedError, match="ridge.*not yet supported"):
        getattr(survival, function)("Surv(time,event) ~ ridge(x,theta=2)", data=ridge_data)


def test_ridge_case_cohort_and_condensed_histories_have_explicit_boundaries(ridge_data):
    with pytest.raises(NotImplementedError, match="ridge.*not yet supported"):
        survival.cch(
            "Surv(time,event) ~ ridge(x,theta=2)",
            ridge_data,
            subcoh=[True] * 10,
            id=ridge_data["id"],
            cohort_size=10,
        )
    ridge_data["start"] = [0.0] * 10
    with pytest.raises(NotImplementedError, match="ridge.*not yet supported"):
        survival.survcondense("Surv(start,time,event) ~ ridge(x,theta=2)", data=ridge_data, id="id")


@pytest.mark.parametrize(("missing_column", "variance"), [("z", 0.11566666666666665), ("x", 0.075)])
def test_ridge_scaling_precedes_other_column_na_omission(ridge_data, missing_column, variance):
    ridge_data[missing_column][-1] = None
    fit = _fit("z + ridge(x,theta=2)", ridge_data, na_action="omit")
    assert len(fit.event_times) == 9
    assert fit.penalty_diagnostics.penalty_diagonal == pytest.approx([0, 2 * variance])


def test_ridge_transforms_and_subsets_reuse_original_sample_variance(ridge_data):
    fit = _fit("ridge(log(x),I(x+z),theta=2)", ridge_data, subset=list(range(8)))
    first = [math.log(value) for value in ridge_data["x"]]
    second = [x + z for x, z in zip(ridge_data["x"], ridge_data["z"], strict=True)]
    variances = [
        math.fsum((value - math.fsum(column) / len(column)) ** 2 for value in column)
        / (len(column) - 1)
        for column in (first, second)
    ]
    assert fit.penalty_diagnostics.penalty_diagonal == pytest.approx([2 * v for v in variances])
    assert survival.model_matrix(fit)["data"] == [
        list(row) for row in zip(first[:8], second[:8], strict=True)
    ]
    newdata = {"x": [2.0], "z": [3.0]}
    prediction = survival.predict(fit, newdata, type="lp", reference="zero")
    beta = survival.coef(fit)
    assert prediction == pytest.approx([math.log(2) * beta[0] + 5 * beta[1]])


def test_ridge_display_rounding_does_not_change_numeric_arithmetic(ridge_data):
    fit = _fit("ridge(I(x + 1000000000000001),theta=2,scale=False)", ridge_data, max_iter=0)
    assert survival.model_matrix(fit)["data"] == [
        [value + 1000000000000001.0] for value in ridge_data["x"]
    ]


def test_ridge_coexists_with_tt_without_rescaling_expanded_risk_rows(ridge_data):
    fit = _fit(
        "ridge(x,theta=2)+tt(z)",
        ridge_data,
        tt=lambda x, time, riskset, weights: [
            value * math.log(stop + 1) for value, stop in zip(x, time, strict=True)
        ],
    )
    assert survival.coef(fit) == pytest.approx([-1.3043852613785663, -0.7657049304995345])
    assert fit.means == pytest.approx([0.6757575757575758, 1.0522089975381268])
    assert fit.df == pytest.approx([0.6619232541590048, 0.9866198486029956])
    assert fit.penalty_diagnostics.penalty_diagonal == pytest.approx([0.2313333333333333, 0])


def test_ridge_initialized_summary_retains_initial_penalized_loglik(ridge_data):
    fit = _fit("z+ridge(x,theta=2)", ridge_data, weights="w", init=[0.4, -0.2], max_iter=0)
    summary = survival.model_summary(fit)
    assert summary["null_loglik"] == pytest.approx(-16.932030305120637, abs=1e-10)
    assert summary["loglik"] == pytest.approx(-16.927403638453971, abs=1e-10)
    assert 2 * (summary["loglik"] - summary["null_loglik"]) == pytest.approx(0.009253333333333558)


@pytest.mark.parametrize("scale", [True, False])
def test_ridge_constant_argument_preserves_alias_and_df_semantics(ridge_data, scale):
    ridge_data["x"] = [1.0] * 10
    fit = _fit(f"ridge(x,theta=2,scale={scale})+z", ridge_data)
    coefficients = survival.coef(fit)
    assert coefficients[1] == pytest.approx(-1.4181573772746472)
    if scale:
        assert math.isnan(coefficients[0])
        assert math.isnan(fit.df[0])
        assert math.isnan(survival.degrees_freedom(fit))
        assert math.isnan(survival.aic(fit))
    else:
        assert coefficients[0] == pytest.approx(0, abs=1e-12)
        assert fit.df == pytest.approx([0, 1], abs=1e-10)


@pytest.mark.parametrize("kind", ["score", "schoenfeld", "dfbeta", "dfbetas", "scaledsch"])
def test_ridge_exact_fit_exposes_r_residual_boundary(ridge_data, kind):
    fit = _fit("z+ridge(x,theta=2)", ridge_data, ties="exact")
    with pytest.raises(ValueError, match="exact method"):
        survival.r_api.residuals(fit, type=kind)


def test_ridge_exact_metadata_preserves_requested_method_and_detail_boundary(ridge_data):
    fit = _fit("z+ridge(x,theta=2)", ridge_data, ties="exact")
    breslow = _fit("z+ridge(x,theta=2)", ridge_data, ties="breslow")
    assert survival.coef(fit) == pytest.approx(survival.coef(breslow))
    assert survival.model_summary(fit)["method"] == "exact"
    with pytest.raises(ValueError, match="exact method"):
        survival.coxph_detail(fit)


def test_ridge_no_events_returns_ordinary_null_inference(ridge_data):
    ridge_data["event"] = [0] * 10
    fit = _fit("ridge(x,theta=2)+z", ridge_data)
    assert fit.penalty_diagnostics is None
    assert survival.coef_names(fit) == ["ridge(x, theta = 2)", "z"]
    assert all(math.isnan(value) for value in survival.coef(fit))
    assert survival.vcov(fit) == [[0, 0], [0, 0]]
    assert survival.degrees_freedom(fit) == 0
    assert survival.loglik(fit) == 0


def test_ridge_does_not_run_unpenalized_anova_or_zph(ridge_data):
    fit = _fit("z+ridge(x,theta=2)", ridge_data)
    with pytest.raises(NotImplementedError, match="penalized Cox"):
        survival.anova(fit)
    with pytest.raises(NotImplementedError, match="penalized"):
        survival.cox_zph(fit)
