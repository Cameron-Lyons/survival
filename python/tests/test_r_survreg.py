import importlib
import math
from statistics import NormalDist

import pytest

from .helpers import setup_survival_import
from .r_api_support import (
    _backtick_data,
    _factor_data,
    _interaction_contrast_data,
    _interaction_contrast_rows,
    _manual_survreg_robust_variance,
    _numeric_data,
    _numeric_data_with_id,
    _survreg_deviance_from_matrix,
    _toy_data,
    _weibull_saturated_center_loglik,
    _with_intercept,
)

survival = setup_survival_import()
r_survreg = importlib.import_module("survival.r._survreg")


def test_model_summary_survreg_scale_rows_and_robust_standard_errors():
    data = _toy_data()
    data["group_num"] = [1 if group == "A" else 2 for group in data["group"]]
    fixed_scale = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        scale=1.0,
        max_iter=10,
        eps=1e-5,
    )
    stratified_scale = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    numeric_strata = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group_num)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    multi_strata = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group_num, group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    robust = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    fixed_summary = survival.model_summary(fixed_scale)
    stratified_summary = survival.model_summary(stratified_scale)
    numeric_summary = survival.model_summary(numeric_strata)
    multi_summary = survival.model_summary(multi_strata)
    robust_summary = survival.model_summary(robust)

    assert fixed_summary["coefficient_names"] == ["(Intercept)", "x1", "x2"]
    assert stratified_summary["coefficient_names"] == [
        "(Intercept)",
        "x1",
        "x2",
        "A",
        "B",
    ]
    assert [row["coef"] for row in stratified_summary["coefficients"]] == pytest.approx(
        stratified_scale.coefficients
    )
    assert survival.coef_names(stratified_scale, complete=True) == [
        "(Intercept)",
        "x1",
        "x2",
        "Log(scale:A)",
        "Log(scale:B)",
    ]
    assert numeric_summary["coefficient_names"][-2:] == ["group_num=1", "group_num=2"]
    assert multi_summary["coefficient_names"][-2:] == [
        "group_num=1, group=A",
        "group_num=2, group=B",
    ]

    assert robust_summary["robust"] is True
    for idx, (row, robust_variance_row, naive_variance_row) in enumerate(
        zip(
            robust_summary["coefficients"],
            robust.variance_matrix,
            robust.naive_variance,
            strict=True,
        )
    ):
        assert row["se"] == pytest.approx(math.sqrt(max(robust_variance_row[idx], 0.0)))
        assert row["robust_se"] == pytest.approx(row["se"])
        assert row["naive_se"] == pytest.approx(math.sqrt(max(naive_variance_row[idx], 0.0)))
        assert row["z"] == pytest.approx(row["coef"] / row["robust_se"])
        assert row["p"] == pytest.approx(2.0 * NormalDist().cdf(-abs(row["z"])))


@pytest.mark.parametrize(
    ("rhs", "intercept_columns", "no_intercept_columns"),
    [
        (
            "g:x",
            ["(Intercept)", "gA:x", "gB:x", "gC:x"],
            ["gA:x", "gB:x", "gC:x"],
        ),
        (
            "x + g:x",
            ["(Intercept)", "x", "x:gB", "x:gC"],
            ["x", "x:gA", "x:gB", "x:gC"],
        ),
        (
            "g + g:x",
            ["(Intercept)", "gB", "gC", "gA:x", "gB:x", "gC:x"],
            ["gA", "gB", "gC", "gA:x", "gB:x", "gC:x"],
        ),
        (
            "g * x",
            ["(Intercept)", "gB", "gC", "x", "gB:x", "gC:x"],
            ["gA", "gB", "gC", "x", "gB:x", "gC:x"],
        ),
        (
            "g * h",
            ["(Intercept)", "gB", "gC", "hH", "gB:hH", "gC:hH"],
            ["gA", "gB", "gC", "hH", "gB:hH", "gC:hH"],
        ),
        (
            "g:h",
            [
                "(Intercept)",
                "gA:hL",
                "gB:hL",
                "gC:hL",
                "gA:hH",
                "gB:hH",
                "gC:hH",
            ],
            ["gA:hL", "gB:hL", "gC:hL", "gA:hH", "gB:hH", "gC:hH"],
        ),
    ],
)
def test_survreg_interaction_contrasts_match_r_intercept_rules(
    rhs,
    intercept_columns,
    no_intercept_columns,
):
    data = _interaction_contrast_data()

    for suffix, columns in (
        ("", intercept_columns),
        (" + 0", no_intercept_columns),
        (" - 1", no_intercept_columns),
    ):
        fit = survival.survreg(
            f"Surv(time, status) ~ {rhs}{suffix}",
            data=data,
            max_iter=0,
        )
        matrix = survival.model_matrix(fit)
        expected_rows = _interaction_contrast_rows(data, columns)

        assert survival.coef_names(fit) == columns
        assert matrix["columns"] == columns
        for actual, expected in zip(matrix["data"], expected_rows, strict=True):
            assert actual == pytest.approx(expected)


def test_survreg_config_defaults_to_weibull():
    config = survival.SurvregConfig()

    assert config.distribution == survival.DistributionType.weibull


def test_low_level_survreg_omitted_distribution_matches_explicit_weibull():
    data = _toy_data()
    kwargs = {
        "time": data["time"],
        "status": [float(value) for value in data["status"]],
        "covariates": _with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        "max_iter": 5,
        "eps": 1e-5,
    }

    default = survival.regression.survreg(**kwargs)
    explicit = survival.regression.survreg(**kwargs, distribution="weibull")
    extreme = survival.regression.survreg(**kwargs, distribution="extreme_value")

    assert default.distribution == "weibull"
    assert default.coefficients == pytest.approx(explicit.coefficients)
    assert default.log_likelihood == pytest.approx(explicit.log_likelihood)
    assert default.iterations == explicit.iterations
    assert extreme.distribution == "extreme_value"
    assert extreme.log_likelihood != pytest.approx(default.log_likelihood)


def test_survreg_formula_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_fixed_scale_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        scale=1.25,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        fixed_scale=1.25,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.location_coefficients == pytest.approx(low_level.location_coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.scale == pytest.approx(1.25)
    assert fit.scales == pytest.approx([1.25])
    assert len(fit.variance_matrix) == len(fit.coefficients)
    assert all(len(row) == len(fit.coefficients) for row in fit.variance_matrix)
    assert len(fit.score_vector) == len(fit.coefficients)


def test_survreg_exponential_ignores_user_scale_like_r():
    data = _toy_data()
    default = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="exponential",
        max_iter=50,
        eps=1e-8,
    )
    scale_zero = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="exponential",
        scale=0,
        max_iter=50,
        eps=1e-8,
    )
    with pytest.warns(RuntimeWarning, match="fixed scale"):
        scale_ignored = survival.survreg(
            "Surv(time, status) ~ x1",
            data=data,
            dist="exponential",
            scale=2,
            max_iter=50,
            eps=1e-8,
        )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="exponential",
        max_iter=50,
        eps=1e-8,
    )

    assert default.scale == pytest.approx(1.0)
    assert default.scales == pytest.approx([1.0])
    assert default.coefficients == pytest.approx(default.location_coefficients)
    assert default.coefficients == pytest.approx([0.90128018, 1.3997076], abs=5e-4)
    assert scale_zero.coefficients == pytest.approx(default.coefficients)
    assert scale_ignored.coefficients == pytest.approx(default.coefficients)
    assert scale_ignored.scale == pytest.approx(1.0)
    assert default.coefficients == pytest.approx(low_level.coefficients)
    assert default.log_likelihood == pytest.approx(low_level.log_likelihood)

    with pytest.raises(ValueError, match="fixed scale and strata"):
        survival.survreg(
            "Surv(time, status) ~ x1 + strata(group)",
            data=data,
            dist="exponential",
            max_iter=50,
            eps=1e-8,
        )


def test_survreg_rayleigh_matches_weibull_fixed_scale_like_r():
    data = _toy_data()
    rayleigh = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="rayleigh",
        max_iter=200,
        eps=1e-8,
    )
    abbreviated = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="ray",
        max_iter=200,
        eps=1e-8,
    )
    weibull_fixed = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="weibull",
        scale=0.5,
        max_iter=200,
        eps=1e-8,
    )
    with pytest.warns(RuntimeWarning, match="fixed scale"):
        scale_ignored = survival.survreg(
            "Surv(time, status) ~ x1",
            data=data,
            dist="rayleigh",
            scale=2,
            max_iter=200,
            eps=1e-8,
        )
    matrix_rayleigh = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="rayleigh",
        max_iter=200,
        eps=1e-8,
    )

    assert rayleigh.distribution == "rayleigh"
    assert rayleigh.scale == pytest.approx(0.5)
    assert rayleigh.scales == pytest.approx([0.5])
    assert rayleigh.coefficients == pytest.approx([0.95288161, 1.10060316], abs=5e-4)
    assert rayleigh.coefficients == pytest.approx(weibull_fixed.coefficients)
    assert rayleigh.log_likelihood == pytest.approx(weibull_fixed.log_likelihood)
    assert survival.loglik(rayleigh) == pytest.approx(survival.loglik(weibull_fixed))
    assert abbreviated.distribution == "rayleigh"
    assert abbreviated.coefficients == pytest.approx(rayleigh.coefficients)
    assert scale_ignored.distribution == "rayleigh"
    assert scale_ignored.coefficients == pytest.approx(rayleigh.coefficients)
    assert scale_ignored.scale == pytest.approx(0.5)
    assert matrix_rayleigh.distribution == "rayleigh"
    assert matrix_rayleigh.coefficients == pytest.approx(rayleigh.coefficients)
    assert survival.model_summary(rayleigh)["distribution"] == "rayleigh"
    rayleigh_residuals = survival.r_api.residuals(rayleigh, type="matrix")
    weibull_residuals = survival.r_api.residuals(weibull_fixed, type="matrix")
    for actual, expected in zip(rayleigh_residuals, weibull_residuals, strict=True):
        assert actual == pytest.approx(expected)

    with pytest.raises(ValueError, match="fixed scale and strata"):
        survival.survreg(
            "Surv(time, status) ~ x1 + strata(group)",
            data=data,
            dist="rayleigh",
            max_iter=200,
            eps=1e-8,
        )


def test_survreg_score_true_exposes_score_vector_alias():
    data = _toy_data()
    formula_fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        score=True,
        max_iter=10,
        eps=1e-5,
    )
    formula_low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert formula_fit.score == pytest.approx(formula_low_level.score_vector)
    assert formula_fit.score_vector == pytest.approx(formula_low_level.score_vector)

    no_score = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        score=False,
        max_iter=10,
        eps=1e-5,
    )
    assert not hasattr(no_score, "score")

    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    matrix_fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        score=True,
        max_iter=10,
        eps=1e-5,
    )
    matrix_low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=rows,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert matrix_fit.score == pytest.approx(matrix_low_level.score_vector)
    assert matrix_fit.score_vector == pytest.approx(matrix_low_level.score_vector)


def test_survreg_cluster_computes_robust_variance():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    plain = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    formula_clustered = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        dist="weibull",
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )
    explicit_cluster = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    matrix_cluster = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=_with_intercept(rows),
        distribution="weibull",
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    singleton_robust = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        robust=True,
        max_iter=10,
        eps=1e-5,
    )
    nonrobust_cluster = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        cluster=data["subject"],
        robust=False,
        max_iter=10,
        eps=1e-5,
    )

    expected_robust = _manual_survreg_robust_variance(plain, data["subject"])
    singleton_expected = _manual_survreg_robust_variance(plain, list(range(len(data["time"]))))

    assert formula_clustered.robust is True
    assert formula_clustered.cluster == data["subject"]
    assert formula_clustered.coefficients == pytest.approx(plain.coefficients)
    assert explicit_cluster.coefficients == pytest.approx(plain.coefficients)
    assert matrix_cluster.coefficients == pytest.approx(plain.coefficients)
    assert formula_clustered.model["subject"] == data["subject"]
    assert formula_clustered.model["(cluster)"] == data["subject"]
    for actual, expected in zip(formula_clustered.x, _with_intercept(rows), strict=True):
        assert actual == pytest.approx(expected)

    for actual, expected in zip(
        formula_clustered.naive_variance,
        plain.variance_matrix,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(formula_clustered.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(explicit_cluster.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(matrix_cluster.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(singleton_robust.variance_matrix, singleton_expected, strict=True):
        assert actual == pytest.approx(expected)

    assert nonrobust_cluster.robust is False
    for actual, expected in zip(
        nonrobust_cluster.variance_matrix,
        plain.variance_matrix,
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_survreg_accepts_r_style_control_mapping():
    data = _toy_data()
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
        tol_chol=1e-8,
    )
    controlled = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        scale=0,
        parms=None,
        control={
            "maxiter": 10,
            "rel.tolerance": 1e-5,
            "toler.chol": 1e-8,
            "debug": 0,
            "outer.max": 10,
        },
    )

    assert controlled.coefficients == pytest.approx(explicit.coefficients)
    assert controlled.log_likelihood == pytest.approx(explicit.log_likelihood)

    nondefault_ignored = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        scale=0,
        control={
            "maxiter": 10,
            "rel.tolerance": 1e-5,
            "toler.chol": 1e-8,
            "debug": 1,
            "outer.max": 2,
        },
    )

    assert nondefault_ignored.coefficients == pytest.approx(explicit.coefficients)
    assert nondefault_ignored.log_likelihood == pytest.approx(explicit.log_likelihood)

    matrix_controlled = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        distribution="weibull",
        scale=0.0,
        control={"maxiter": 10, "rel.tolerance": 1e-5, "toler.chol": 1e-8},
    )
    matrix_explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
        tol_chol=1e-8,
    )

    assert matrix_controlled.coefficients == pytest.approx(matrix_explicit.coefficients)
    assert matrix_controlled.log_likelihood == pytest.approx(matrix_explicit.log_likelihood)


def test_survreg_accepts_r_style_formula_storage_flags():
    data = _toy_data()
    default = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit_defaults = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        model=False,
        x=False,
        y=True,
        robust=False,
        cluster=None,
        score=False,
        max_iter=10,
        eps=1e-5,
    )
    with_model = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + offset(offset)",
        data=data,
        dist="weibull",
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )

    assert explicit_defaults.coefficients == pytest.approx(default.coefficients)
    assert explicit_defaults.log_likelihood == pytest.approx(default.log_likelihood)
    assert explicit_defaults.y.time == pytest.approx(data["time"])
    assert explicit_defaults.y.event == tuple(data["status"])
    assert not hasattr(explicit_defaults, "x")
    assert not hasattr(explicit_defaults, "model")
    assert not hasattr(explicit_defaults, "score")

    model_frame = with_model.model
    assert model_frame["Surv(time, status)"].time == pytest.approx(data["time"])
    assert model_frame["Surv(time, status)"].event == tuple(data["status"])
    assert model_frame["time"] == pytest.approx(data["time"])
    assert model_frame["status"] == data["status"]
    assert model_frame["x1"] == pytest.approx(data["x1"])
    assert model_frame["x2"] == pytest.approx(data["x2"])
    assert model_frame["offset"] == pytest.approx(data["offset"])
    assert model_frame["(offset)"] == pytest.approx(data["offset"])

    with_x = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        x=True,
        y=False,
        max_iter=10,
        eps=1e-5,
    )
    expected_x = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    for actual, expected in zip(with_x.x, expected_x, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(with_x, "y")


def test_survreg_model_true_stores_matrix_inputs():
    data = _toy_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]

    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        model=True,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.model["time"] == pytest.approx(data["time"])
    assert fit.model["status"] == pytest.approx([float(value) for value in data["status"]])
    for actual, expected in zip(fit.model["x"], rows, strict=True):
        assert actual == pytest.approx(expected)


def test_survreg_accepts_r_style_init_alias():
    data = _toy_data()
    initial = [0.15, -0.1, 0.05, 0.0]
    alias = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        init=initial,
        max_iter=0,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        initial_beta=initial,
        max_iter=0,
    )

    assert alias.coefficients == pytest.approx(explicit.coefficients)
    assert alias.log_likelihood == pytest.approx(explicit.log_likelihood)

    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    matrix_alias = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        init=initial[1:],
        max_iter=0,
    )
    matrix_explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        initial=initial[1:],
        max_iter=0,
    )

    assert matrix_alias.coefficients == pytest.approx(matrix_explicit.coefficients)
    assert matrix_alias.log_likelihood == pytest.approx(matrix_explicit.log_likelihood)


def test_survreg_distribution_accepts_r_style_prefixes_and_aliases():
    data = _toy_data()
    full = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    abbreviated = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="wei",
        max_iter=10,
        eps=1e-5,
    )

    assert abbreviated.distribution == "weibull"
    assert abbreviated.coefficients == pytest.approx(full.coefficients)
    assert abbreviated.log_likelihood == pytest.approx(full.log_likelihood)

    expected = {
        "exp": "exponential",
        "ext": "extreme_value",
        "extreme": "extreme_value",
        "extreme value": "extreme_value",
        "extreme_value": "extreme_value",
        "gauss": "gaussian",
        "normal": "gaussian",
        "logi": "logistic",
        "logg": "lognormal",
        "loggaussian": "lognormal",
        "logn": "lognormal",
        "logl": "loglogistic",
        "log-normal": "lognormal",
        "log-logistic": "loglogistic",
    }
    for dist, distribution in expected.items():
        fit = survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist=dist,
            max_iter=10,
            eps=1e-5,
        )
        assert fit.distribution == distribution

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist="log",
            max_iter=10,
            eps=1e-5,
        )

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist="ex",
            max_iter=10,
            eps=1e-5,
        )


def test_survreg_distribution_helpers_match_r_reference_values(monkeypatch):
    weibull_density = survival.dsurvreg(
        [1.0, 2.0],
        mean=0.5,
        scale=1.2,
        distribution="weibull",
    )
    weibull_cdf = survival.psurvreg(
        [1.0, 2.0],
        mean=0.5,
        scale=1.2,
        distribution="weibull",
    )
    weibull_quantiles = survival.qsurvreg(
        [0.25, 0.5, 0.75],
        mean=0.5,
        scale=1.2,
        distribution="weibull",
    )

    assert weibull_density == pytest.approx([0.2841569, 0.1512009])
    assert weibull_cdf == pytest.approx([0.4827560, 0.6910677])
    assert weibull_quantiles == pytest.approx([0.3696942, 1.0620325, 2.4399099])

    assert survival.dsurvreg(
        [1.0, 2.0],
        mean=0.5,
        scale=1.2,
        distribution="lognormal",
    ) == pytest.approx([0.3048103, 0.1640866])
    assert survival.psurvreg(
        [1.0, 2.0],
        mean=0.5,
        scale=1.2,
        distribution="lognormal",
    ) == pytest.approx([0.3384611, 0.5639360])
    assert survival.qsurvreg(
        [0.25, 0.5, 0.75],
        mean=0.5,
        scale=1.2,
        distribution="lognormal",
    ) == pytest.approx([0.7338962, 1.6487213, 3.7039051])

    assert survival.dsurvreg([-1.0, 0.0, 1.0], mean=0.0, distribution="gaussian") == (
        pytest.approx([0.2419707, 0.3989423, 0.2419707])
    )
    assert survival.psurvreg([-1.0, 0.0, 1.0], mean=0.0, distribution="gaussian") == (
        pytest.approx([0.1586553, 0.5, 0.8413447])
    )
    assert survival.qsurvreg([0.25, 0.5, 0.75], mean=0.0, distribution="gaussian") == (
        pytest.approx([-0.6744898, 0.0, 0.6744898])
    )
    assert survival.dsurvreg(
        [1.0, 2.0],
        mean=0.0,
        scale=1.0,
        distribution="t",
        parms=5,
    ) == pytest.approx([0.2196798, 0.06509031])
    assert survival.psurvreg(
        [1.0, 2.0],
        mean=0.0,
        scale=1.0,
        distribution="t",
        parms=5,
    ) == pytest.approx([0.8183913, 0.9490303])
    assert survival.qsurvreg(
        [0.25, 0.5],
        mean=0.0,
        scale=1.0,
        distribution="t",
        parms=5,
    ) == pytest.approx([-0.7266868, 0.0])
    assert survival.dsurvreg(
        [1.0, math.inf, -math.inf],
        mean=[math.inf, 0.0, 0.0],
        distribution="t",
        parms=5,
    ) == [0.0, 0.0, 0.0]
    assert survival.psurvreg(
        [1.0, math.inf, -math.inf],
        mean=[math.inf, 0.0, 0.0],
        distribution="t",
        parms=5,
    ) == [0.0, 1.0, 0.0]

    assert survival.dsurvreg([1.0], mean=0.5, scale=1.2, distribution="loggaussian") == (
        pytest.approx(survival.dsurvreg([1.0], mean=0.5, scale=1.2, distribution="lognormal"))
    )
    assert survival.dsurvreg([1.0], mean=0.5, scale=1.2, distribution="rayleigh") == (
        pytest.approx(survival.dsurvreg([1.0], mean=0.5, scale=1.2, distribution="weibull"))
    )
    assert survival.dsurvreg([1.0], mean=0.0, distribution="gaussian", parms=5) == pytest.approx(
        survival.dsurvreg([1.0], mean=0.0, distribution="gaussian")
    )

    nonpositive_density = survival.dsurvreg([0.0, -1.0], mean=0.0, distribution="weibull")
    assert all(math.isnan(value) for value in nonpositive_density)
    assert survival.psurvreg([0.0, -1.0], mean=0.0, distribution="weibull") == [0.0, 0.0]
    boundary_quantiles = survival.qsurvreg([0.0, 1.0], mean=0.0, distribution="weibull")
    assert boundary_quantiles[0] == pytest.approx(0.0)
    assert math.isinf(boundary_quantiles[1])
    assert boundary_quantiles[1] > 0.0

    draws = iter([0.25, 0.5])
    monkeypatch.setattr(r_survreg.random, "random", lambda: next(draws))
    assert survival.rsurvreg(2, mean=0.5, scale=1.2, distribution="weibull") == pytest.approx(
        [0.3696942, 1.0620325]
    )
    draws = iter([0.25, 0.5])
    monkeypatch.setattr(r_survreg.random, "random", lambda: next(draws))
    assert survival.rsurvreg(2, mean=0.0, scale=1.0, distribution="t", parms=5) == pytest.approx(
        [-0.7266868, 0.0]
    )

    with pytest.raises(ValueError, match="length 1 or 2"):
        survival.dsurvreg([1.0, 2.0], mean=[0.0, 1.0, 2.0], distribution="weibull")
    with pytest.raises(TypeError, match="parms"):
        survival.dsurvreg([1.0], mean=0.0, distribution="t")


def test_survreg_loglik_and_response_transform_follow_r_distribution_scale():
    data = _toy_data()

    weibull = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=80,
        eps=1e-8,
    )
    expected_loglik = weibull.log_likelihood - sum(
        math.log(time)
        for time, event in zip(data["time"], data["status"], strict=True)
        if event == 1
    )
    assert survival.loglik(weibull) == pytest.approx(expected_loglik)

    lognormal = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="lognormal",
        max_iter=80,
        eps=1e-8,
    )
    lognormal_lp = survival.predict(lognormal, type="lp")
    assert survival.predict(lognormal, type="response") == pytest.approx(
        [math.exp(value) for value in lognormal_lp]
    )

    gaussian = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=80,
        eps=1e-8,
    )
    assert survival.loglik(gaussian) == pytest.approx(gaussian.log_likelihood)
    assert survival.predict(gaussian, type="response") == pytest.approx(
        survival.predict(gaussian, type="lp")
    )


def test_low_level_survreg_rejects_invalid_numeric_inputs():
    kwargs = {
        "time": [1.0, 2.0, 3.0],
        "status": [1.0, 0.0, 1.0],
        "covariates": [[0.2, 1.0], [0.4, 0.9], [0.1, 1.1]],
        "distribution": "weibull",
        "max_iter": 1,
    }

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.survreg(**{**kwargs, "time": [1.0, float("nan"), 3.0]})

    with pytest.raises(ValueError, match="must be positive"):
        survival.regression.survreg(**{**kwargs, "time": [1.0, 0.0, 3.0]})

    with pytest.raises(ValueError, match="status must contain only 0/1/2/3"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 4.0, 0.0]})

    with pytest.raises(ValueError, match=r"covariates\[1\]\[0\] contains non-finite"):
        survival.regression.survreg(
            **{**kwargs, "covariates": [[0.2, 1.0], [float("inf"), 0.9], [0.1, 1.1]]}
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.regression.survreg(**{**kwargs, "weights": [1.0, -1.0, 1.0]})

    with pytest.raises(ValueError, match="at least one positive"):
        survival.regression.survreg(**{**kwargs, "weights": [0.0, 0.0, 0.0]})

    with pytest.raises(ValueError, match="offsets contains non-finite"):
        survival.regression.survreg(**{**kwargs, "offsets": [0.0, float("nan"), 0.0]})

    with pytest.raises(ValueError, match="initial_beta contains non-finite"):
        survival.regression.survreg(**{**kwargs, "initial_beta": [0.0, 0.0, float("nan")]})

    with pytest.raises(ValueError, match="fixed_scale must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "fixed_scale": 0.0})

    with pytest.raises(ValueError, match="fixed_scale must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "fixed_scale": float("nan")})

    with pytest.raises(ValueError, match="cannot have both a fixed scale and strata"):
        survival.regression.survreg(**{**kwargs, "strata": [0, 1, 1], "fixed_scale": 1.0})

    with pytest.raises(ValueError, match="initial_beta has 3 values but model expects 2"):
        survival.regression.survreg(
            **{**kwargs, "initial_beta": [0.0, 0.0, 0.0], "fixed_scale": 1.0}
        )

    with pytest.raises(ValueError, match="eps must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "eps": 0.0})

    with pytest.raises(ValueError, match="tol_chol must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "tol_chol": float("nan")})

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.regression.survreg(**{**kwargs, "distribution": "mystery"})

    with pytest.raises(ValueError, match="time2 is required"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 3.0, 0.0]})

    with pytest.raises(ValueError, match="time2 has 2"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 3.0, 0.0], "time2": [1.0, 2.5]})

    with pytest.raises(ValueError, match="non-finite interval endpoint"):
        survival.regression.survreg(
            **{
                **kwargs,
                "status": [1.0, 3.0, 0.0],
                "time2": [1.0, float("inf"), 3.0],
            }
        )

    with pytest.raises(ValueError, match="greater than time"):
        survival.regression.survreg(
            **{**kwargs, "status": [1.0, 3.0, 0.0], "time2": [1.0, 2.0, 3.0]}
        )


def test_low_level_survreg_accepts_left_and_interval_censoring():
    fit = survival.regression.survreg(
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        time2=[1.0, 2.0, 3.0, 4.5, 5.0],
        status=[1.0, 2.0, 0.0, 3.0, 1.0],
        covariates=[[0.2], [0.4], [0.1], [0.8], [1.0]],
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.status == [1, 2, 0, 3, 1]
    assert fit.time2 == pytest.approx([1.0, 2.0, 3.0, 4.5, 5.0])
    assert math.isfinite(fit.log_likelihood)
    assert len(fit.coefficients) == 2


def test_survreg_left_censored_formula_matches_low_level_binding():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [0, 1, 0, 1, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(time, status, type='left') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[2.0, 1.0, 2.0, 1.0, 1.0],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.status == [2, 1, 2, 1, 1]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval_formula_matches_low_level_binding():
    data = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0],
        "right": [1.0, 2.0, 3.0, 4.5, 5.0],
        "status": [1, 2, 0, 3, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(left, right, status, type='interval') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["left"],
        time2=data["right"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.time2 == pytest.approx(low_level.time2)
    assert fit.status == data["status"]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval2_formula_derives_low_level_status_codes():
    data = {
        "left": [float("-inf"), 2.0, 3.0, 4.0],
        "right": [1.0, 5.0, 3.0, float("inf")],
        "x1": [0.2, 0.4, 0.1, 0.8],
    }
    fit = survival.survreg(
        "Surv(left, right, type='interval2') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=[1.0, 2.0, 3.0, 4.0],
        time2=[1.0, 5.0, 3.0, float("inf")],
        status=[2.0, 3.0, 1.0, 0.0],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert fit.time2 == pytest.approx(low_level.time2)
    assert fit.status == [2, 3, 1, 0]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval_residuals_support_scalar_and_influence_types():
    data = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0],
        "right": [1.0, 2.0, 3.0, 4.5, 5.0],
        "status": [1, 2, 0, 3, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(left, right, status, type='interval') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="ldcase",
        time2=fit.time2,
    )
    low_response = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="response",
        time2=fit.time2,
    )
    low_deviance = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="deviance",
        time2=fit.time2,
    )
    low_working = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="working",
        time2=fit.time2,
    )
    matrix = survival.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=fit.time2,
    )
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    expected_ldcase = survival.survreg_influence_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldcase",
        True,
    )
    expected_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        False,
    )
    saturated = _weibull_saturated_center_loglik(fit.time, fit.time2, fit.status, fit.scale)
    expected_response = [
        math.exp(center) - math.exp(linear_predictor)
        for (center, _), linear_predictor in zip(saturated, fit.linear_predictors, strict=True)
    ]
    expected_deviance = _survreg_deviance_from_matrix(matrix, saturated)
    expected_working = [0.0 if abs(row[2]) <= 1e-12 else -row[1] / row[2] for row in matrix]
    expected_location_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        False,
        False,
    )
    low_location_dfbeta = survival.dfbeta_survreg(
        fit.time,
        fit.status,
        fit.covariates,
        fit.linear_predictors,
        fit.scale,
        location_vcov,
        fit.distribution,
        time2=fit.time2,
    )

    assert survival.r_api.residuals(fit, type="ldcase") == pytest.approx(expected_ldcase)
    assert survival.r_api.residuals(fit, type="ldc") == pytest.approx(expected_ldcase)
    assert low_response.residuals == pytest.approx(expected_response)
    assert fit.residuals("response").residuals == pytest.approx(expected_response)
    assert survival.r_api.residuals(fit, type="response") == pytest.approx(expected_response)
    assert low_deviance.residuals == pytest.approx(expected_deviance)
    assert fit.residuals("deviance").residuals == pytest.approx(expected_deviance)
    assert survival.r_api.residuals(fit, type="deviance") == pytest.approx(expected_deviance)
    assert low_working.residuals == pytest.approx(expected_working)
    assert fit.residuals("working").residuals == pytest.approx(expected_working)
    assert survival.r_api.residuals(fit, type="working") == pytest.approx(expected_working)
    for actual, expected in zip(
        survival.r_api.residuals(fit, type="dfbeta"),
        expected_dfbeta,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.dfbeta(), expected_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(low_location_dfbeta, expected_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    assert all(math.isfinite(value) for value in low_level.residuals)
    assert all(math.isfinite(value) for value in expected_ldcase)

    with pytest.raises(ValueError, match="ambiguous"):
        survival.r_api.residuals(fit, type="ld")


def test_low_level_survreg_residuals_handle_interval_ldcase_and_working():
    time = [1.0, 1.0, 1.0, 1.0]
    time2 = [1.0, 1.0, 2.0, 1.0]
    status = [1, 2, 3, 0]
    linear_pred = [0.0, 0.0, 0.0, 0.0]

    ldcase = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="ldcase",
        time2=time2,
    )

    assert ldcase.residuals[0] == pytest.approx(-1.0)
    assert ldcase.residuals[1] == pytest.approx(math.log(1.0 - math.exp(-1.0)))
    assert ldcase.residuals[2] == pytest.approx(math.log(math.exp(-1.0) - math.exp(-2.0)))
    assert ldcase.residuals[3] == pytest.approx(-1.0)

    covariates = [[1.0], [1.0], [1.0], [1.0]]
    matrix = survival.survreg_residual_matrix(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        time2=time2,
    )
    dfbeta = survival.dfbeta_survreg(
        time,
        status,
        covariates,
        linear_pred,
        1.0,
        [[1.0]],
        "weibull",
        time2=time2,
    )
    expected_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        covariates,
        [1.0],
        [0, 0, 0, 0],
        [[1.0]],
        False,
        False,
    )
    for actual, expected in zip(dfbeta, expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)

    saturated = _weibull_saturated_center_loglik(time, time2, status, 1.0)
    response = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="response",
        time2=time2,
    )
    deviance = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="deviance",
        time2=time2,
    )
    assert response.residuals == pytest.approx(
        [
            math.exp(center) - math.exp(lp)
            for (center, _), lp in zip(saturated, linear_pred, strict=True)
        ]
    )
    assert deviance.residuals == pytest.approx(_survreg_deviance_from_matrix(matrix, saturated))

    working = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="working",
        time2=time2,
    )
    expected_working = [0.0 if abs(row[2]) <= 1e-12 else -row[1] / row[2] for row in matrix]
    assert working.residuals == pytest.approx(expected_working)

    with pytest.raises(ValueError, match="time2 is required"):
        survival.residuals_survreg(
            time,
            status,
            linear_pred,
            1.0,
            "weibull",
            residual_type="ldcase",
        )


def test_survreg_fit_exposes_prediction_metadata_and_methods():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)
    expected_lp = [
        sum(
            value * coefficient
            for value, coefficient in zip(row, fit.location_coefficients, strict=True)
        )
        for row in design_rows
    ]
    training_rows = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    training_lp = [
        sum(
            value * coefficient
            for value, coefficient in zip(row, fit.location_coefficients, strict=True)
        )
        for row in training_rows
    ]

    assert fit.n_covariates == 3
    assert fit.n_strata == 1
    assert fit.distribution == "weibull"
    assert fit.scale > 0.0
    assert fit.scales == pytest.approx([fit.scale])
    assert fit.location_coefficients == pytest.approx(fit.coefficients[:3])
    assert fit.linear_predictors == pytest.approx(training_lp)
    for actual, expected in zip(fit.information_matrix, fit.fit.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.variance_matrix, fit.fit.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)

    lp = fit.predict(design_rows, "lp")
    response = fit.predict(design_rows)
    quantiles = fit.predict_quantile(design_rows, [0.25, 0.5])

    assert lp.predictions == pytest.approx(expected_lp)
    assert response.predictions == pytest.approx([math.exp(value) for value in expected_lp])
    assert quantiles.quantiles == pytest.approx([0.25, 0.5])
    assert len(quantiles.predictions) == 2
    assert all(len(row) == 2 for row in quantiles.predictions)


def test_predict_survreg_r_style_generic_types():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)

    lp = survival.predict(fit, rows, type="lp")
    prefix_lp = survival.predict(fit, rows, type="l")
    response = survival.predict(fit, rows)
    prefix_response = survival.predict(fit, rows, type="r")
    response_with_se = survival.predict(fit, rows, se_fit=True)
    dotted_response_with_se = survival.predict(fit, rows, **{"se.fit": True})
    terms = survival.predict(fit, rows, type="terms")
    prefix_terms = survival.predict(fit, rows, type="t")
    terms_with_se = survival.predict(fit, rows, type="terms", se_fit=True)
    x2_with_se = survival.predict(fit, rows, type="terms", terms="x2", se_fit=True)
    training_terms = survival.predict(fit, type="terms")
    training_terms_with_se = survival.predict(fit, type="terms", se_fit=True)
    training_x1_with_se = survival.predict(fit, type="terms", terms="x1", se_fit=True)
    median = survival.predict(fit, rows, type="quantile", p=0.5)
    prefix_median = survival.predict(fit, rows, type="q", p=0.5)
    median_with_se = survival.predict(fit, rows, type="quantile", p=0.5, se_fit=True)
    uquantile_with_se = survival.predict(fit, rows, type="uquantile", p=0.5, se_fit=True)
    prefix_uquantile_with_se = survival.predict(fit, rows, type="u", p=0.5, se_fit=True)
    default_bands = survival.predict(fit, rows, type="quantile")
    bands = survival.predict(fit, rows, type="quantile", quantiles=[0.25, 0.75])
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]
    full_vcov = [
        row[: fit.n_covariates + len(fit.scales)]
        for row in fit.variance_matrix[: fit.n_covariates + len(fit.scales)]
    ]
    training_rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    training_design_rows = _with_intercept(training_rows)
    means = [
        sum(row[col_idx] for row in training_design_rows) / len(training_design_rows)
        for col_idx in range(fit.n_covariates)
    ]

    assert lp == pytest.approx(fit.predict(design_rows, "lp").predictions)
    assert prefix_lp == pytest.approx(lp)
    assert response == pytest.approx(fit.predict(design_rows).predictions)
    assert prefix_response == pytest.approx(response)
    expected_linear_se = [
        math.sqrt(
            max(
                sum(
                    design_row[left] * location_vcov[left][right] * design_row[right]
                    for left in range(fit.n_covariates)
                    for right in range(fit.n_covariates)
                ),
                0.0,
            )
        )
        for design_row in design_rows
    ]
    assert response_with_se.fit == pytest.approx(response)
    assert response_with_se.se_fit == pytest.approx(
        [se * prediction for se, prediction in zip(expected_linear_se, response, strict=True)]
    )
    assert isinstance(dotted_response_with_se, survival.r_api.PredictResult)
    assert dotted_response_with_se.fit == pytest.approx(response_with_se.fit)
    assert dotted_response_with_se.se_fit == pytest.approx(response_with_se.se_fit)
    expected_terms = [
        [
            (row[col_idx] - means[col_idx]) * fit.location_coefficients[col_idx]
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in design_rows
    ]
    for actual, expected in zip(
        terms,
        expected_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(terms_with_se.fit, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    expected_terms_se = [
        [
            abs(row[col_idx] - means[col_idx])
            * math.sqrt(max(location_vcov[col_idx][col_idx], 0.0))
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in design_rows
    ]
    for actual, expected in zip(terms_with_se.se_fit, expected_terms_se, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        x2_with_se.fit,
        [[row[1]] for row in expected_terms],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        x2_with_se.se_fit,
        [[row[1]] for row in expected_terms_se],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_training_terms = [
        [
            (row[col_idx] - means[col_idx]) * fit.location_coefficients[col_idx]
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in training_design_rows
    ]
    expected_training_terms_se = [
        [
            abs(row[col_idx] - means[col_idx])
            * math.sqrt(max(location_vcov[col_idx][col_idx], 0.0))
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in training_design_rows
    ]
    for actual, expected in zip(
        training_terms,
        expected_training_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_terms_with_se.fit,
        expected_training_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_terms_with_se.se_fit,
        expected_training_terms_se,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_x1_with_se.fit,
        [[row[0]] for row in expected_training_terms],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_x1_with_se.se_fit,
        [[row[0]] for row in expected_training_terms_se],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_median = [row[0] for row in fit.predict_quantile(design_rows, [0.5]).predictions]
    median_score = math.log(-math.log1p(-0.5))
    expected_uquantile = [lp_value + median_score * fit.scale for lp_value in lp]
    expected_quantile_se = []
    expected_uquantile_se = []
    for row, prediction in zip(design_rows, expected_median, strict=True):
        design = [*row, median_score * fit.scale]
        variance = sum(
            design[left] * full_vcov[left][right] * design[right]
            for left in range(len(design))
            for right in range(len(design))
        )
        linear_se = math.sqrt(max(variance, 0.0))
        expected_uquantile_se.append(linear_se)
        expected_quantile_se.append(linear_se * prediction)
    assert median == pytest.approx(expected_median)
    assert prefix_median == pytest.approx(expected_median)
    assert median_with_se.fit == pytest.approx(expected_median)
    assert median_with_se.se_fit == pytest.approx(expected_quantile_se)
    assert uquantile_with_se.fit == pytest.approx(expected_uquantile)
    assert uquantile_with_se.se_fit == pytest.approx(expected_uquantile_se)
    assert prefix_uquantile_with_se.fit == pytest.approx(expected_uquantile)
    assert prefix_uquantile_with_se.se_fit == pytest.approx(expected_uquantile_se)
    assert len(default_bands) == 2
    assert all(len(row) == 2 for row in default_bands)
    assert len(bands) == 2
    assert all(len(row) == 2 for row in bands)
    with pytest.raises(ValueError, match="collapse"):
        survival.predict(fit, rows, collapse=["A", "B"])


def test_predict_survreg_quantile_rejects_nonfinite_probability():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        survival.predict(fit, [[0.5, 0.8]], type="quantile", p=math.nan)


def test_predict_survreg_formula_accepts_newdata_mapping():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)
    newdata = {"x1": [0.5, 1.1], "x2": [0.8, 0.3]}

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp").predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(fit.predict(design_rows).predictions)
    assert survival.predict(fit, newdata, type="quantile", p=0.5) == pytest.approx(
        [row[0] for row in fit.predict_quantile(design_rows, [0.5]).predictions]
    )


def test_predict_survreg_gaussian_response_uses_identity_transform():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]

    lp_with_se = survival.predict(fit, rows, type="lp", se_fit=True)
    response_with_se = survival.predict(fit, rows, se_fit=True)
    quantile_with_se = survival.predict(fit, rows, type="quantile", p=0.9, se_fit=True)
    uquantile_with_se = survival.predict(fit, rows, type="uquantile", p=0.9, se_fit=True)

    assert response_with_se.fit == pytest.approx(lp_with_se.fit)
    assert response_with_se.se_fit == pytest.approx(lp_with_se.se_fit)
    assert quantile_with_se.fit == pytest.approx(uquantile_with_se.fit)
    assert quantile_with_se.se_fit == pytest.approx(uquantile_with_se.se_fit)


def test_survreg_gaussian_residuals_use_identity_response_scale():
    normal = NormalDist()
    low_level_response = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="response",
    )
    low_level_deviance = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="deviance",
    )
    low_level_working = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="working",
    )
    z = 0.5
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    survivor = 1.0 - normal.cdf(z)

    assert low_level_response.residuals == pytest.approx([0.5, 0.5])
    assert low_level_deviance.residuals == pytest.approx(
        [
            z,
            math.sqrt(-2.0 * math.log(survivor)),
        ]
    )
    assert low_level_working.residuals == pytest.approx([z, density / survivor])

    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=10,
        eps=1e-5,
    )
    residuals = fit.residuals("response")
    linear_predictors = survival.predict(fit, type="lp")

    assert residuals.residual_type == "response"
    assert residuals.residuals == pytest.approx(
        [
            time - linear_predictor
            for time, linear_predictor in zip(data["time"], linear_predictors, strict=True)
        ]
    )


def test_predict_survreg_formula_newdata_mapping_rebuilds_transforms_and_interactions():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ sqrt(x1) + group:x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.25, 1.0], "x2": [0.25, 0.8], "group": ["B", "A"]}
    rows = [
        [math.sqrt(0.25), 0.0 * 0.25, 1.0 * 0.25],
        [math.sqrt(1.0), 1.0 * 0.8, 0.0 * 0.8],
    ]
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp").predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(fit.predict(design_rows).predictions)


def test_predict_survreg_uses_training_rows_and_offsets():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    expected_lp = [
        fit.location_coefficients[0]
        + data["x1"][idx] * fit.location_coefficients[1]
        + data["offset"][idx]
        for idx in range(len(data["time"]))
    ]
    expected_se = [
        math.sqrt(
            max(
                fit.variance_matrix[0][0]
                + 2.0 * data["x1"][idx] * fit.variance_matrix[0][1]
                + data["x1"][idx] ** 2 * fit.variance_matrix[1][1],
                0.0,
            )
        )
        for idx in range(len(data["time"]))
    ]
    lp_with_se = survival.predict(fit, type="lp", se_fit=True)
    response_with_se = survival.predict(fit, se_fit=True)

    assert survival.predict(fit, type="lp") == pytest.approx(expected_lp)
    assert survival.predict(fit) == pytest.approx([math.exp(value) for value in expected_lp])
    assert lp_with_se.fit == pytest.approx(expected_lp)
    assert lp_with_se.se_fit == pytest.approx(expected_se)
    assert response_with_se.fit == pytest.approx([math.exp(value) for value in expected_lp])
    assert response_with_se.se_fit == pytest.approx(
        [
            se * math.exp(linear_predictor)
            for se, linear_predictor in zip(expected_se, expected_lp, strict=True)
        ]
    )
    assert fit.predict([[1.0, 0.5]], "lp", [0.2]).predictions == pytest.approx(
        [fit.location_coefficients[0] + 0.5 * fit.location_coefficients[1] + 0.2]
    )


def test_predict_survreg_formula_newdata_mapping_uses_offsets():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "offset": offsets}
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(
        fit.predict(design_rows, "response", offsets).predictions
    )


def test_predict_survreg_formula_rebuilds_transformed_offsets_from_newdata():
    data = _toy_data()
    data["exposure"] = [math.exp(value) for value in data["offset"]]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "exposure": [math.exp(value) for value in offsets]}
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(
        fit.predict(design_rows, "response", offsets).predictions
    )


def test_predict_survreg_formula_rebuilds_identity_arithmetic_offsets_from_newdata():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(I(offset + x2))",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    bare_fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset + x2)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        offsets=[offset + x2 for offset, x2 in zip(data["offset"], data["x2"], strict=True)],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.5, 0.3]
    newdata = {"x1": [0.5, 1.0], "offset": [0.2, -0.1], "x2": [0.3, 0.4]}
    design_rows = _with_intercept(rows)

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert bare_fit.coefficients == pytest.approx(low_level.coefficients)
    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(bare_fit, newdata, type="lp") == pytest.approx(
        bare_fit.predict(design_rows, "lp", offsets).predictions
    )


def test_survreg_fit_residuals_match_low_level_apis():
    data = _toy_data()
    weights = [1.0, 1.5, 0.75, 2.0, 1.25, 0.5, 1.75, 1.0]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]

    response = fit.residuals("response")
    deviance = fit.residuals()
    working = survival.r_api.residuals(fit, type="working")
    dfbeta = survival.r_api.residuals(fit, type="dfbeta")
    dfbetas = survival.r_api.residuals(fit, type="dfbetas")
    dfbeta_without_scale = survival.r_api.residuals(fit, type="dfbeta", rsigma=False)
    matrix_residuals = survival.r_api.residuals(fit, type="matrix")
    ldcase = survival.r_api.residuals(fit, type="ldcase")
    ldresp = survival.r_api.residuals(fit, type="ldresp")
    ldshape = survival.r_api.residuals(fit, type="ldshape")
    ldcase_without_scale = survival.r_api.residuals(fit, type="ldcase", rsigma=False)
    prefix_response = survival.r_api.residuals(fit, type="r")
    prefix_working = survival.r_api.residuals(fit, type="w")
    prefix_dfbeta = survival.r_api.residuals(fit, type="dfb")
    prefix_matrix = survival.r_api.residuals(fit, type="mat")
    prefix_ldcase = survival.r_api.residuals(fit, type="ldc")
    low_response = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="response",
    )
    low_working = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="working",
    )
    low_location_dfbeta = survival.dfbeta_survreg(
        fit.time,
        fit.status,
        fit.covariates,
        fit.linear_predictors,
        fit.scale,
        location_vcov,
        fit.distribution,
    )
    low_matrix = survival.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=fit.time2,
    )
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    low_dfbeta = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        False,
    )
    low_dfbetas = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        True,
    )
    low_dfbeta_without_scale = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        False,
        False,
    )
    low_ldcase = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldcase",
        True,
    )
    low_ldresp = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldresp",
        True,
    )
    low_ldshape = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldshape",
        True,
    )
    low_ldcase_without_scale = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        "ldcase",
        False,
    )

    assert fit.time == pytest.approx(data["time"])
    assert fit.status == data["status"]
    assert fit.weights == pytest.approx(weights)
    expected_covariates = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    for actual, expected in zip(fit.covariates, expected_covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert response.residual_type == "response"
    assert response.residuals == pytest.approx(low_response.residuals)
    assert prefix_response == pytest.approx(low_response.residuals)
    assert response.residuals == pytest.approx(
        [
            time - math.exp(linear_predictor)
            for time, linear_predictor in zip(fit.time, fit.linear_predictors, strict=True)
        ]
    )
    assert deviance.residual_type == "deviance"
    assert len(deviance.residuals) == len(data["time"])
    assert working == pytest.approx(low_working.residuals)
    assert prefix_working == pytest.approx(low_working.residuals)
    assert len(dfbeta) == len(data["time"])
    for actual, expected in zip(fit.dfbeta(), low_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for dfbeta_matrix in (dfbeta, prefix_dfbeta):
        for actual, expected in zip(dfbeta_matrix, low_dfbeta, strict=True):
            assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("dfbeta")
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("dfbetas")
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("matrix")
    assert len(dfbetas) == len(data["time"])
    for actual, expected in zip(dfbeta_without_scale, low_dfbeta_without_scale, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(dfbetas, low_dfbetas, strict=True):
        assert actual == pytest.approx(expected)
    for dfbeta_row, dfbetas_row in zip(dfbeta, dfbetas, strict=True):
        for col_idx, value in enumerate(dfbeta_row):
            scale = max(math.sqrt(abs(full_vcov[col_idx][col_idx])), 1e-12)
            assert dfbetas_row[col_idx] == pytest.approx(value / scale)
    assert len(matrix_residuals) == len(data["time"])
    assert all(len(row) == 6 for row in matrix_residuals)
    for actual, expected in zip(matrix_residuals, low_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_matrix, low_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert ldcase == pytest.approx(low_ldcase)
    assert prefix_ldcase == pytest.approx(low_ldcase)
    assert ldresp == pytest.approx(low_ldresp)
    assert ldshape == pytest.approx(low_ldshape)
    assert ldcase_without_scale == pytest.approx(low_ldcase_without_scale)

    collapse = ["A", "A", "B", "B", "C", "C", "D", "D"]
    collapsed_response = survival.r_api.residuals(fit, type="response", collapse=collapse)
    collapsed_matrix = survival.r_api.residuals(fit, type="matrix", collapse=collapse)
    collapsed_ldcase = survival.r_api.residuals(fit, type="ldcase", collapse=collapse)
    expected_response = [
        sum(
            residual
            for residual, label in zip(low_response.residuals, collapse, strict=True)
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    collapsed_dfbeta = survival.r_api.residuals(fit, type="dfbeta", collapse=collapse)
    collapsed_dfbetas = survival.r_api.residuals(fit, type="dfbetas", collapse=collapse)
    collapsed_weighted_response = survival.r_api.residuals(
        fit,
        type="response",
        collapse=collapse,
        weighted=True,
    )
    collapsed_weighted_matrix = survival.r_api.residuals(
        fit,
        type="matrix",
        collapse=collapse,
        weighted=True,
    )
    collapsed_weighted_ldcase = survival.r_api.residuals(
        fit,
        type="ldcase",
        collapse=collapse,
        weighted=True,
    )
    expected_dfbeta = [
        [
            sum(
                row[col_idx]
                for row, label in zip(low_dfbeta, collapse, strict=True)
                if label == group
            )
            for col_idx in range(len(low_dfbeta[0]))
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_response = [
        sum(
            residual * weights[idx]
            for idx, (residual, label) in enumerate(
                zip(low_response.residuals, collapse, strict=True)
            )
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_matrix = [
        [
            sum(
                row[col_idx]
                for row, label in zip(low_matrix, collapse, strict=True)
                if label == group
            )
            for col_idx in range(6)
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_matrix = [
        [
            sum(
                row[col_idx] * weights[idx]
                for idx, (row, label) in enumerate(zip(low_matrix, collapse, strict=True))
                if label == group
            )
            for col_idx in range(6)
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_ldcase = [
        sum(
            residual for residual, label in zip(low_ldcase, collapse, strict=True) if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_ldcase = [
        sum(
            residual * weights[idx]
            for idx, (residual, label) in enumerate(zip(low_ldcase, collapse, strict=True))
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_dfbetas = [
        [
            sum(
                row[col_idx] for row, label in zip(dfbetas, collapse, strict=True) if label == group
            )
            for col_idx in range(len(dfbetas[0]))
        ]
        for group in ("A", "B", "C", "D")
    ]

    assert survival.r_api.residuals(fit, type="working", weighted=False) == pytest.approx(
        low_working.residuals
    )
    assert survival.r_api.residuals(fit, type="working", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(low_working.residuals)]
    )
    weighted_matrix = survival.r_api.residuals(fit, type="matrix", weighted=True)
    for row_idx, actual in enumerate(weighted_matrix):
        assert actual == pytest.approx([value * weights[row_idx] for value in low_matrix[row_idx]])
    assert survival.r_api.residuals(fit, type="ldcase", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(low_ldcase)]
    )
    weighted_dfbeta = survival.r_api.residuals(fit, type="dfbeta", weighted=True)
    weighted_dfbetas = survival.r_api.residuals(fit, type="dfbetas", weighted=True)
    for row_idx, (actual_dfbeta, actual_dfbetas) in enumerate(
        zip(weighted_dfbeta, weighted_dfbetas, strict=True)
    ):
        assert actual_dfbeta == pytest.approx(
            [value * weights[row_idx] for value in dfbeta[row_idx]]
        )
        assert actual_dfbetas == pytest.approx(
            [value * weights[row_idx] for value in dfbetas[row_idx]]
        )
    assert collapsed_response == pytest.approx(expected_response)
    assert collapsed_weighted_response == pytest.approx(expected_weighted_response)
    for actual, expected in zip(collapsed_matrix, expected_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(collapsed_weighted_matrix, expected_weighted_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert collapsed_ldcase == pytest.approx(expected_ldcase)
    assert collapsed_weighted_ldcase == pytest.approx(expected_weighted_ldcase)
    for actual, expected in zip(collapsed_dfbeta, expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(collapsed_dfbetas, expected_dfbetas, strict=True):
        assert actual == pytest.approx(expected)


def test_survreg_formula_accepts_intercept_only_rhs():
    data = _toy_data()
    fit = survival.survreg("Surv(time, status) ~ 1", data=data, max_iter=10, eps=1e-5)
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=[[1.0] for _ in data["time"]],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.n_covariates == 1
    assert fit.covariates == [[1.0] for _ in data["time"]]


def test_survreg_formula_accepts_numeric_interactions():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [data["x1"][idx], data["x2"][idx], data["x1"][idx] * data["x2"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_dot_expands_remaining_covariates():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ .",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(explicit.coefficients)
    assert fit.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_formula_dot_can_exclude_identifier_columns():
    data = _numeric_data_with_id()
    fit = survival.survreg(
        "Surv(time, status) ~ . - id",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(explicit.coefficients)
    assert fit.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_formula_accepts_backtick_column_names():
    data = _backtick_data()
    fit = survival.survreg(
        "Surv(`follow-up`, `event status`) ~ `age-years` + `marker/value`",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["follow-up"],
        status=[float(value) for value in data["event status"]],
        covariates=_with_intercept(
            [
                [data["age-years"][idx], data["marker/value"][idx]]
                for idx in range(len(data["follow-up"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_accepts_backtick_numeric_transforms():
    data = _backtick_data()
    fit = survival.survreg(
        "Surv(`follow-up`, `event status`) ~ sqrt(`marker/value`) + `age-years`",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["follow-up"],
        status=[float(value) for value in data["event status"]],
        covariates=_with_intercept(
            [
                [math.sqrt(data["marker/value"][idx]), data["age-years"][idx]]
                for idx in range(len(data["follow-up"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_accepts_identity_wrappers_for_numeric_terms():
    data = _toy_data()
    direct = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for wrapper in ("I", "identity", "as.numeric"):
        fit = survival.survreg(
            f"Surv(time, status) ~ {wrapper}(x1) + x2",
            data=data,
            dist="weibull",
            max_iter=10,
            eps=1e-5,
        )

        assert fit.coefficients == pytest.approx(direct.coefficients)
        assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_formula_filters_external_weights_with_subset_and_na_action():
    data = _numeric_data()
    indices = [0, 2, 3, 4, 5]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        subset=[0, 1, 2, 3, 4, 5],
        na_action="omit",
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    dotted = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        subset=[0, 1, 2, 3, 4, 5],
        dist="weibull",
        max_iter=10,
        eps=1e-5,
        **{"na.action": "omit"},
    )
    low_level = survival.regression.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[float(data["status"][idx]) for idx in indices],
        covariates=_with_intercept([[data["x1"][idx], data["x2"][idx]] for idx in indices]),
        weights=[1.0, 1.0, 0.8, 1.2, 1.1],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert dotted.coefficients == pytest.approx(low_level.coefficients)
    assert dotted.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_as_factor_treatment_codes_numeric_covariates():
    data = _factor_data()
    fit = survival.survreg(
        "Surv(time, status) ~ as.factor(dose) + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [
                    1.0 if data["dose"][idx] == 1 else 0.0,
                    1.0 if data["dose"][idx] == 2 else 0.0,
                    data["x2"][idx],
                ]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert survival.coef_names(fit) == [
        "(Intercept)",
        "as.factor(dose)1",
        "as.factor(dose)2",
        "x2",
    ]


def test_survreg_matrix_input_applies_subset_to_row_aligned_arrays():
    data = _numeric_data()
    indices = [0, 1, 3, 5, 6]
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        subset=indices,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_matrix_input_defaults_to_weibull_distribution():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    default = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert default.distribution == "weibull"
    assert default.coefficients == pytest.approx(explicit.coefficients)
    assert default.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_matrix_na_action_omit_filters_covariate_rows():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    rows[2][0] = float("nan")
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        na_action="omit",
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_matrix_input_rejects_fractional_status_codes():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    status = list(data["status"])
    status[2] = 1.5

    with pytest.raises(ValueError, match="0/1/2/3 censoring codes"):
        survival.survreg(
            time=data["time"],
            status=status,
            covariates=rows,
            distribution="weibull",
            max_iter=10,
            eps=1e-5,
        )


def test_survreg_matrix_na_action_omit_filters_status_before_code_validation():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    status = list(data["status"])
    status[2] = float("nan")
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.survreg(
        time=data["time"],
        status=status,
        covariates=rows,
        na_action="omit",
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_formula_treatment_codes_categorical_covariates():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ group + x1",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [1.0 if data["group"][idx] == "B" else 0.0, data["x1"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    newdata = {"group": ["B", "A"], "x1": [0.5, 0.8]}
    term_se = survival.predict(fit, newdata, type="terms", terms="group", se_fit=True)
    group_var = fit.variance_matrix[1][1]
    group_mean = sum(1.0 if value == "B" else 0.0 for value in data["group"]) / len(data["group"])
    for actual, expected in zip(
        term_se.fit,
        [
            [(1.0 - group_mean) * fit.location_coefficients[1]],
            [(0.0 - group_mean) * fit.location_coefficients[1]],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        term_se.se_fit,
        [
            [abs(1.0 - group_mean) * math.sqrt(max(group_var, 0.0))],
            [abs(0.0 - group_mean) * math.sqrt(max(group_var, 0.0))],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_survreg_formula_passes_strata_to_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        strata=[0, 0, 0, 0, 1, 1, 1, 1],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.strata == [0, 0, 0, 0, 1, 1, 1, 1]

    score = math.log(-math.log1p(-0.5))
    quantile_with_se = survival.predict(fit, type="uquantile", p=0.5, se_fit=True)
    newdata = {"x1": [0.5, 0.8], "x2": [0.4, 0.6], "group": ["B", "A"]}
    newdata_quantile = survival.predict(fit, newdata, type="uquantile", p=0.5)
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    expected_fit = []
    expected_se = []
    for idx, (x1, x2) in enumerate(zip(data["x1"], data["x2"], strict=True)):
        stratum = fit.strata[idx]
        expected_fit.append(fit.linear_predictors[idx] + score * fit.scales[stratum])
        design = [1.0, x1, x2, *([0.0] * len(fit.scales))]
        design[fit.n_covariates + stratum] = score * fit.scales[stratum]
        variance = sum(
            design[left] * full_vcov[left][right] * design[right]
            for left in range(full_width)
            for right in range(full_width)
        )
        expected_se.append(math.sqrt(max(variance, 0.0)))

    assert quantile_with_se.fit == pytest.approx(expected_fit)
    assert quantile_with_se.se_fit == pytest.approx(expected_se)
    assert newdata_quantile == pytest.approx(
        [
            (
                fit.location_coefficients[0]
                + 0.5 * fit.location_coefficients[1]
                + 0.4 * fit.location_coefficients[2]
                + score * fit.scales[1]
            ),
            (
                fit.location_coefficients[0]
                + 0.8 * fit.location_coefficients[1]
                + 0.6 * fit.location_coefficients[2]
                + score * fit.scales[0]
            ),
        ]
    )


def test_survreg_formula_offset_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        offsets=data["offset"],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)

    transformed = {**data, "exposure": [math.exp(value) for value in data["offset"]]}
    transformed_fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=transformed,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert transformed_fit.coefficients == pytest.approx(low_level.coefficients)
    assert transformed_fit.log_likelihood == pytest.approx(low_level.log_likelihood)
