import importlib
import math
from bisect import bisect_right

import pytest

from .helpers import setup_survival_import
from .r_api_support import (
    _backtick_data,
    _clogit_data,
    _counting_cox_data,
    _factor_data,
    _fit_event_times,
    _interaction_contrast_data,
    _interaction_contrast_rows,
    _manual_cox_loglik,
    _manual_cox_loglik_at_zero,
    _manual_cox_robust_variance,
    _numeric_data,
    _numeric_data_with_id,
    _take,
    _tied_cox_data,
    _toy_data,
)

survival = setup_survival_import()
r_coerce = importlib.import_module("survival.r._coerce")
r_coxph = importlib.import_module("survival.r._coxph")


def test_coxph_wtest_matches_r_wald_helper_shapes():
    identity = survival.coxph_wtest([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0])
    correlated = survival.coxph_wtest([[2.0, 0.5], [0.5, 1.0]], [1.0, 2.0])
    singular = survival.coxph_wtest([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    trailing = survival.coxph_wtest([[0.0, 0.0], [0.0, 2.0]], [1.0, 2.0])
    zero = survival.coxph_wtest([[0.0, 0.0], [0.0, 0.0]], [1.0, 2.0])
    indefinite = survival.coxph_wtest([[1.0, 2.0], [2.0, 1.0]], [1.0, 2.0])
    matrix_rhs = survival.coxph_wtest(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 3.0], [2.0, 4.0]],
    )
    missing_rhs = survival.coxph_wtest([[1.0, 0.0], [0.0, 1.0]], [None, 2.0])
    scalar = survival.coxph_wtest([2.0], [4.0])

    assert identity.test == pytest.approx([5.0])
    assert identity.df == 2
    assert identity.solve == pytest.approx([1.0, 2.0])
    assert correlated.test == pytest.approx([4.0])
    assert correlated.solve == pytest.approx([0.0, 2.0])
    assert singular.df == 1
    assert singular.test == pytest.approx([1.0])
    assert singular.solve == pytest.approx([1.0, 0.0])
    assert trailing.df == 1
    assert trailing.test == pytest.approx([2.0])
    assert trailing.solve == pytest.approx([0.0, 1.0])
    assert zero.test == pytest.approx([0.0])
    assert zero.df == 0
    assert zero.solve == pytest.approx([0.0, 0.0])
    assert indefinite.test == pytest.approx([1.0])
    assert indefinite.df == 1
    assert indefinite.solve == pytest.approx([1.0, 0.0])
    assert matrix_rhs.test == pytest.approx([5.0, 25.0])
    assert len(matrix_rhs.solve) == 2
    for actual_row, expected_row in zip(
        matrix_rhs.solve,
        [[1.0, 3.0], [2.0, 4.0]],
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert missing_rhs.test == []
    assert missing_rhs.df == 0
    assert missing_rhs.solve == 0.0
    assert scalar.test == pytest.approx([8.0])
    assert scalar.solve == pytest.approx([2.0])

    with pytest.raises(ValueError, match="Argument lengths"):
        survival.coxph_wtest([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="square matrix"):
        survival.coxph_wtest([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="infinite"):
        survival.coxph_wtest([[1.0, 0.0], [0.0, float("inf")]], [1.0, 2.0])


def test_coxph_formula_returns_fitted_cox_model():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        max_iter=10,
    )

    assert sum(len(row) for row in fit.coefficients) == 2
    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert len(fit.risk_scores) == 8
    assert len(fit.predict([[0.5, 0.8]])) == 1


def test_coxph_formula_accepts_rep_constant_response_argument():
    data = {
        "case": [1, 0, 1, 0, 0, 1, 0, 1],
        "set": [1, 1, 2, 2, 3, 3, 4, 4],
        "x": [0.2, 0.4, 0.3, 0.1, 0.5, 0.2, 0.3, 0.7],
    }
    formula = survival.coxph(
        "Surv(rep(1, 8L), case) ~ x + strata(set)",
        data=data,
        method="breslow",
        eps=1e-9,
    )
    inferred_length = survival.coxph(
        "Surv(rep(1, nrow(data)), case) ~ x + strata(set)",
        data=data,
        method="breslow",
        eps=1e-9,
    )
    direct = survival.coxph(
        survival.Surv([1.0] * len(data["case"]), data["case"]),
        x=[[value] for value in data["x"]],
        strata=data["set"],
        method="breslow",
        eps=1e-9,
    )

    assert survival.coef(formula) == pytest.approx(survival.coef(direct))
    assert survival.coef(inferred_length) == pytest.approx(survival.coef(direct))
    assert survival.loglik(formula) == pytest.approx(survival.loglik(direct))
    assert survival.nobs(formula) == sum(data["case"])


def test_clogit_exact_matches_r_reference_for_multiple_cases_per_set():
    data = _clogit_data()
    fit = survival.clogit(
        "case ~ x + z + strata(set)",
        data=data,
        control={"iter.max": 50, "eps": 1e-9},
    )
    equivalent_cox = survival.coxph(
        "Surv(rep(1, nrow(data)), case) ~ x + z + strata(set)",
        data=data,
        method="exact",
        control={"iter.max": 50, "eps": 1e-9},
    )
    expected_variance = [
        [1.6714828664413921, -0.20436032396075088],
        [-0.20436032396075085, 1.5517480346884],
    ]

    assert survival.coef(fit) == pytest.approx(
        [-0.8457808761181879, 1.4766531964610223], rel=0, abs=1e-10
    )
    for actual, expected in zip(survival.vcov(fit), expected_variance, strict=True):
        assert actual == pytest.approx(expected, rel=0, abs=1e-10)
    assert fit.log_likelihood == pytest.approx(
        [-6.579251212010101, -5.679859460220504], rel=0, abs=1e-10
    )
    assert fit.score_test == pytest.approx(1.804139493304423, rel=0, abs=1e-10)
    assert fit.iterations == 3
    assert fit.method == "exact"
    assert survival.model_formula(fit) == "Surv(rep(1, n), case) ~ x + z + strata(set)"
    assert survival.coef(fit) == pytest.approx(survival.coef(equivalent_cox))
    for actual, expected in zip(survival.vcov(fit), survival.vcov(equivalent_cox), strict=True):
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="survival curves are not defined for a clogit model"):
        survival.basehaz(fit)
    with pytest.raises(ValueError, match="survival curves are not defined for a clogit model"):
        survival.survfit(fit)
    with pytest.raises(ValueError, match="survival curves are not defined for a clogit model"):
        fit.basehaz()
    with pytest.raises(ValueError, match="survival curves are not defined for a clogit model"):
        fit.survival_curve()
    for residual_type in ("score", "schoenfeld", "dfbeta", "dfbetas", "scaledsch"):
        with pytest.raises(ValueError, match=f"{residual_type} residuals are not available"):
            survival.r_api.residuals(fit, type=residual_type)
    with pytest.raises(ValueError, match="score residuals are not available"):
        fit.score_residuals()
    assert len(survival.r_api.residuals(fit, type="martingale")) == len(data["case"])
    with pytest.raises(ValueError, match="schoenfeld residuals are not available"):
        survival.cox_zph(fit)


def test_clogit_method_and_exact_inference_rules_match_r():
    data = {**_clogit_data(), "id": list(range(16))}
    formula = "case ~ x + z + strata(set)"
    exact = survival.clogit(formula, data=data)
    explicit_exact = survival.clogit(formula, data=data, method="exact")
    approximate = survival.clogit(formula, data=data, method="ap")
    breslow = survival.clogit(formula, data=data, method="breslow")

    assert survival.coef(exact) == pytest.approx(survival.coef(explicit_exact))
    assert survival.coef(approximate) == pytest.approx(survival.coef(breslow))
    assert approximate.method == breslow.method == "breslow"

    with pytest.warns(RuntimeWarning, match="weights ignored"):
        weighted = survival.clogit(formula, data=data, weights=[2.0] * len(data["case"]))
    assert survival.coef(weighted) == pytest.approx(survival.coef(exact))
    assert survival.model_weights(weighted) is None

    breslow_weights = [0.5 + 0.1 * (idx % 5) for idx in range(len(data["case"]))]
    weighted_breslow = survival.clogit(
        formula,
        data=data,
        method="breslow",
        weights=breslow_weights,
    )
    assert survival.model_weights(weighted_breslow) == pytest.approx(breslow_weights)
    assert weighted_breslow.robust is True
    expected_weighted_variance = [
        [0.720954464895698, 0.00931434307520126],
        [0.00931434307520126, 0.7781943393567565],
    ]
    expected_weighted_naive = [
        [1.6761284190128223, -0.19215580825836115],
        [-0.19215580825836115, 1.8422617494368836],
    ]
    for actual, expected in zip(
        survival.vcov(weighted_breslow), expected_weighted_variance, strict=True
    ):
        assert actual == pytest.approx(expected, rel=0, abs=1e-10)
    for actual, expected in zip(weighted_breslow.naive_var, expected_weighted_naive, strict=True):
        assert actual == pytest.approx(expected, rel=0, abs=1e-10)

    integer_weighted = survival.clogit(
        formula,
        data=data,
        method="breslow",
        weights=[2.0] * len(data["case"]),
    )
    assert integer_weighted.robust is False

    id_exact = survival.clogit(formula, data=data, id=data["id"])
    id_breslow = survival.clogit(formula, data=data, method="breslow", id=data["id"])
    robust_id_breslow = survival.clogit(
        formula,
        data=data,
        method="breslow",
        id=data["id"],
        robust=True,
    )
    assert survival.coef(id_exact) == pytest.approx(survival.coef(exact))
    assert id_exact.robust is False
    for actual, expected in zip(survival.vcov(id_breslow), survival.vcov(breslow), strict=True):
        assert actual == pytest.approx(expected)
    assert id_breslow.robust is False
    assert robust_id_breslow.robust is True

    repeated_event_ids = [
        "a",
        "b",
        "a",
        "c",
        "d",
        "e",
        "f",
        "a",
        "g",
        "a",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
    ]
    repeated_id_breslow = survival.clogit(
        formula,
        data=data,
        method="breslow",
        id=repeated_event_ids,
    )
    expected_id_variance = [
        [0.5065837992703215, 0.1113609349577885],
        [0.1113609349577885, 0.5874222540961469],
    ]
    assert repeated_id_breslow.robust is True
    for actual, expected in zip(
        survival.vcov(repeated_id_breslow), expected_id_variance, strict=True
    ):
        assert actual == pytest.approx(expected, rel=0, abs=1e-10)

    with pytest.raises(ValueError, match="dfbeta residuals are not available"):
        survival.clogit(formula, data=data, id=repeated_event_ids)
    for kwargs in ({"robust": True}, {"cluster": data["id"]}):
        with pytest.raises(ValueError, match="dfbeta residuals are not available"):
            survival.clogit(formula, data=data, **kwargs)
    with pytest.warns(RuntimeWarning, match="cluster specified with robust=FALSE"):
        ignored_cluster = survival.clogit(
            formula,
            data=data,
            cluster=data["id"],
            robust=False,
        )
    assert survival.coef(ignored_cluster) == pytest.approx(survival.coef(exact))
    with pytest.warns(RuntimeWarning, match="cluster specified with robust=FALSE"):
        ignored_breslow_cluster = survival.clogit(
            formula,
            data=data,
            method="breslow",
            cluster=data["id"],
            robust=False,
        )
    assert ignored_breslow_cluster.robust is False
    for actual, expected in zip(
        survival.vcov(ignored_breslow_cluster), survival.vcov(breslow), strict=True
    ):
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="robust variance plus the exact method"):
        survival.clogit(f"{formula} + cluster(id)", data=data)
    with pytest.raises(ValueError, match="survival curves are not defined for a clogit model"):
        survival.survfit(breslow)


def test_clogit_delegates_subset_and_missing_value_alignment():
    data = _clogit_data()
    filtered = _take(data, list(range(1, len(data["case"]))))
    subset_fit = survival.clogit(
        "case ~ x + z + strata(set)",
        data=data,
        subset=list(range(1, len(data["case"]))),
    )
    filtered_fit = survival.clogit("case ~ x + z + strata(set)", data=filtered)

    with_missing = {key: list(values) for key, values in data.items()}
    with_missing["z"][0] = None
    omitted_fit = survival.clogit(
        "case ~ x + z + strata(set)",
        data=with_missing,
        na_action="omit",
    )

    assert survival.coef(subset_fit) == pytest.approx(survival.coef(filtered_fit))
    assert survival.loglik(subset_fit) == pytest.approx(survival.loglik(filtered_fit))
    assert survival.coef(omitted_fit) == pytest.approx(survival.coef(filtered_fit))
    assert survival.loglik(omitted_fit) == pytest.approx(survival.loglik(filtered_fit))


def test_coxph_formula_cluster_computes_robust_variance():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    clustered = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    plain = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    explicit_cluster = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    id_cluster = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        id=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    id_model = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        id=data["subject"],
        model=True,
        max_iter=10,
        eps=1e-5,
    )
    matrix_id = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        id=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    expected_robust = _manual_cox_robust_variance(plain, data["subject"])

    assert clustered.robust is True
    assert clustered.cluster == data["subject"]
    assert id_cluster.robust is True
    assert id_cluster.id == data["subject"]
    assert id_cluster.cluster == data["subject"]
    assert id_model.model["(id)"] == data["subject"]
    assert clustered.coefficients[0] == pytest.approx(plain.coefficients[0])
    assert id_cluster.coefficients[0] == pytest.approx(plain.coefficients[0])
    assert len(clustered.covariates[0]) == 2
    for actual, expected in zip(
        clustered.naive_information_matrix, plain.information_matrix, strict=True
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.naive_var, plain.information_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(explicit_cluster.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(id_cluster.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(matrix_id.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)

    score = clustered.score_residuals()
    expected_dfbeta = [
        [
            sum(
                clustered.naive_information_matrix[col_idx][inner_idx] * row[inner_idx]
                for inner_idx in range(2)
            )
            for col_idx in range(2)
        ]
        for row in score
    ]
    expected_dfbetas = [
        [
            row[col_idx]
            / max(
                math.sqrt(abs(clustered.naive_information_matrix[col_idx][col_idx])),
                r_coerce._COX_DFBETAS_SCALE_FLOOR,
            )
            for col_idx in range(2)
        ]
        for row in expected_dfbeta
    ]
    for actual, expected in zip(clustered.dfbeta(), expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.dfbetas(), expected_dfbetas, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.r_api.residuals(clustered, type="dfbeta", weighted=False),
        expected_dfbeta,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.r_api.residuals(clustered, type="dfbetas", weighted=False),
        expected_dfbetas,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert clustered.dfbeta()[3] == pytest.approx(clustered.fit.dfbeta()[3])

    row = [0.5, 0.8]
    robust_prediction = survival.predict(clustered, [row], reference="zero", se_fit=True)
    expected_se = math.sqrt(
        max(
            sum(
                row[left] * expected_robust[left][right] * row[right]
                for left in range(2)
                for right in range(2)
            ),
            0.0,
        )
    )

    assert robust_prediction.fit == pytest.approx(plain.predict([row]))
    assert robust_prediction.se_fit == pytest.approx([expected_se])


def test_coxph_generics_mask_converged_aliased_coefficients():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
    }
    data["x2"] = [2.0 * value for value in data["x1"]]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.coefficients[0][0] == pytest.approx(0.43940480983777153)
    assert fit.coefficients[0][1] == 0.0
    prediction_rows = [[0.25, 0.5], [1.0, 2.0]]
    raw_predictions = fit.predict(prediction_rows)
    assert all(math.isfinite(value) for value in raw_predictions)
    assert survival.predict(fit, prediction_rows, reference="zero") == pytest.approx(
        raw_predictions
    )

    coefficients = survival.coef(fit)
    assert coefficients[0] == pytest.approx(fit.coefficients[0][0])
    assert math.isnan(coefficients[1])
    assert survival.coef_names(fit) == ["x1", "x2"]
    assert survival.coef_names(fit, complete=True) == ["x1", "x2"]
    assert survival.coef_names(fit, complete=False) == ["x1"]
    assert survival.degrees_freedom(fit) == 1
    assert survival.aic(fit) == pytest.approx(-2.0 * survival.loglik(fit) + 2.0)
    assert survival.extract_aic(fit) == pytest.approx([1.0, survival.aic(fit)])
    for actual, expected in zip(
        survival.vcov(fit),
        [[1.3555601527463446, 0.0], [0.0, 0.0]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert survival.vcov(fit, complete=False)[0] == pytest.approx([1.3555601527463446])

    summary = survival.model_summary(fit)
    assert summary["df"] == 1
    aliased_row = summary["coefficients"][1]
    assert aliased_row["name"] == "x2"
    assert math.isnan(aliased_row["coef"])
    assert aliased_row["se"] == 0.0
    assert math.isnan(aliased_row["statistic"])
    assert math.isnan(aliased_row["p"])

    aliased_interval = survival.confint(fit, parm="x2")[0]
    assert math.isnan(aliased_interval["lower"])
    assert math.isnan(aliased_interval["upper"])

    anova_frame = survival.as_data_frame(survival.anova(fit))
    assert anova_frame["df"] == [0, 1, 1]
    assert anova_frame["chisq"][2] == 0.0
    assert anova_frame["p"][2] == 1.0

    reduced_fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )
    nested_frame = survival.as_data_frame(survival.anova(reduced_fit, fit))
    assert nested_frame["df"] == [1, 1]
    assert nested_frame["chisq"][1] == 0.0
    assert nested_frame["p"][1] == 1.0


@pytest.mark.parametrize("max_iter", [0, 1, 2])
def test_coxph_generics_do_not_mask_aliases_before_convergence(max_iter):
    data = {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
    }
    data["x2"] = [2.0 * value for value in data["x1"]]

    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=max_iter,
        eps=1e-9,
        toler=1e-9,
        singular_ok=False,
    )

    assert survival.coef(fit) == pytest.approx(fit.coefficients[0])
    assert survival.degrees_freedom(fit) == 2
    assert len(survival.vcov(fit, complete=False)) == 2
    unmasked_row = survival.model_summary(fit)["coefficients"][1]
    assert unmasked_row["coef"] == 0.0
    assert unmasked_row["se"] == 0.0
    assert math.isnan(unmasked_row["statistic"])
    assert math.isnan(unmasked_row["p"])


def test_coxph_generics_mask_aliases_after_step_halving_convergence():
    values = [
        -2.6291240340330893,
        4.591206129787794,
        4.46532600950345,
        0.5254034794341393,
        3.8258202073353136,
        -3.519487461823151,
    ]
    fit = survival.coxph(
        survival.Surv([1, 2, 3, 4, 8, 8], [0, 1, 0, 0, 1, 0]),
        x=[[value, 2.0 * value] for value in values],
        method="breslow",
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.convergence_flag == -2
    assert fit.coefficients[0][1] == 0.0
    assert math.isnan(survival.coef(fit)[1])
    assert survival.degrees_freedom(fit) == 1


def test_coxph_alias_mask_uses_naive_rank_with_robust_variance():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
        "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
    }
    data["x2"] = [2.0 * value for value in data["x1"]]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.robust is True
    assert math.isnan(survival.coef(fit)[1])
    assert survival.vcov(fit)[1] == [0.0, 0.0]
    assert len(survival.vcov(fit, complete=False)) == 1


def test_anova_coxph_single_formula_model_refits_terms_sequentially():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)
    first_term = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
    )

    result = survival.anova(fit)

    assert result.test_type == "Chisq"
    assert [row.model_name for row in result.rows] == ["NULL", "x1", "x2"]
    assert [row.df for row in result.rows] == [0, 1, 2]
    assert result.rows[0].loglik == pytest.approx(fit.log_likelihood[0])
    assert result.rows[1].loglik == pytest.approx(first_term.log_likelihood[-1])
    assert result.rows[2].loglik == pytest.approx(fit.log_likelihood[-1])
    assert result.rows[1].chisq == pytest.approx(
        2.0 * (result.rows[1].loglik - result.rows[0].loglik)
    )
    assert result.rows[2].chisq == pytest.approx(
        2.0 * (result.rows[2].loglik - result.rows[1].loglik)
    )
    assert 0.0 <= result.rows[1].p_value <= 1.0


def test_anova_coxph_single_formula_model_preserves_offsets():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2 + offset(offset)", data=data)
    first_term = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=data["offset"],
    )

    result = survival.anova(fit)

    assert [row.model_name for row in result.rows] == ["NULL", "x1", "x2"]
    assert result.rows[1].loglik == pytest.approx(first_term.log_likelihood[-1])
    assert result.rows[2].loglik == pytest.approx(fit.log_likelihood[-1])


def test_anova_coxph_compares_nested_models():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)

    result = survival.anova(fit_x1, fit_full)

    assert [row.model_name for row in result.rows] == ["Model 1", "Model 2"]
    assert [row.df for row in result.rows] == [1, 2]
    assert result.rows[0].loglik == pytest.approx(fit_x1.log_likelihood[-1])
    assert result.rows[1].loglik == pytest.approx(fit_full.log_likelihood[-1])
    assert result.rows[1].chisq == pytest.approx(
        2.0 * (fit_full.log_likelihood[-1] - fit_x1.log_likelihood[-1])
    )


def test_anova_coxph_accepts_r_style_test_aliases_and_prefixes():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)

    chisq = survival.anova(fit_x1, fit_full, test="ch")
    lrt = survival.anova(fit_x1, fit_full, test="likelihood-ratio")
    none = survival.anova(fit_x1, fit_full, test="n")

    assert chisq.test_type == "Chisq"
    assert chisq.rows[1].chisq == pytest.approx(
        2.0 * (fit_full.log_likelihood[-1] - fit_x1.log_likelihood[-1])
    )
    assert lrt.test_type == "LRT"
    assert lrt.rows[1].chisq == pytest.approx(chisq.rows[1].chisq)
    assert none.test_type == "none"
    assert none.rows[1].chisq is None
    assert none.rows[1].p_value is None

    with pytest.raises(ValueError, match="anova test"):
        survival.anova(fit_x1, fit_full, test="wald")


def test_anova_coxph_can_omit_tests_and_rejects_non_cox_models():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)
    without_tests = survival.anova(fit_x1, fit_full, test=None)

    assert without_tests.test_type == "none"
    assert without_tests.rows[0].chisq is None
    assert without_tests.rows[1].p_value is None

    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
    )
    with pytest.raises(TypeError, match="anova requires fitted Cox model objects"):
        survival.anova(aft)


def test_coxph_detail_exposes_r_style_event_contributions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        init=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )

    detail = survival.coxph_detail(fit)
    baseline_times, baseline_cumhaz = fit.basehaz(True)
    event_indices = [idx for idx, event in enumerate(data["status"]) if event == 1]

    assert isinstance(detail, survival.r_api.CoxPHDetailResult)
    assert detail.time == pytest.approx([data["time"][idx] for idx in event_indices])
    assert detail.times() == pytest.approx(detail.time)
    assert detail.nevent == [1] * len(event_indices)
    assert detail.n_event == detail.nevent
    assert detail.nrisk == [len(data["time"]) - idx for idx in event_indices]
    assert detail.n_risk_at_times() == detail.nrisk
    assert detail.cumulative_hazard == pytest.approx(baseline_cumhaz)
    assert detail.cumulative_hazards() == pytest.approx(baseline_cumhaz)
    assert detail.hazards() == pytest.approx(detail.hazard)
    expected_x = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]
    expected_y = [
        [time, float(status)] for time, status in zip(data["time"], data["status"], strict=True)
    ]
    assert detail.x == expected_x
    assert detail.y == expected_y
    for actual, expected in zip(detail.score, fit.schoenfeld_residuals(), strict=True):
        assert actual == pytest.approx(expected)
    assert [sum(row[col_idx] for row in detail.score) for col_idx in range(2)] == pytest.approx(
        fit.score_vector
    )
    assert baseline_times == pytest.approx(detail.time)
    assert all(value > 0.0 for value in detail.varhaz)


def test_coxph_detail_efron_tie_averages_step_risk_means():
    data = {
        "time": [1.0, 1.0, 2.0],
        "status": [1, 1, 0],
        "x": [0.0, 1.0, 2.0],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x",
        data=data,
        init=[0.0],
        max_iter=0,
        method="efron",
    )

    detail = survival.coxph_detail(fit)

    assert detail.means[0] == pytest.approx([9.0 / 8.0])
    assert detail.score[0] == pytest.approx([-5.0 / 4.0])
    assert detail.imat[0][0] == pytest.approx([65.0 / 48.0])
    assert detail.hazard == pytest.approx([5.0 / 6.0])
    assert detail.varhaz == pytest.approx([13.0 / 36.0])


def test_coxph_detail_weighted_tied_event_moments_match_native_values():
    data = {
        "time": [1.0, 1.0, 2.0],
        "status": [1, 1, 0],
        "x": [0.0, 1.0, 2.0],
        "weight": [1.0, 2.0, 0.5],
    }
    expected = {
        "breslow": {
            "mean": 6.0 / 7.0,
            "score": -4.0 / 7.0,
            "imat": 60.0 / 49.0,
            "hazard": 6.0 / 7.0,
            "varhaz": 18.0 / 49.0,
        },
        "efron": {
            "mean": 13.0 / 14.0,
            "score": -11.0 / 14.0,
            "imat": 267.0 / 196.0,
            "hazard": 33.0 / 28.0,
            "varhaz": 585.0 / 784.0,
        },
    }

    for method, values in expected.items():
        fit = survival.coxph(
            "Surv(time, status) ~ x",
            data=data,
            weights=data["weight"],
            init=[0.0],
            max_iter=0,
            method=method,
        )

        detail = survival.coxph_detail(fit)

        assert detail.nevent == [2]
        assert detail.nrisk == [3]
        assert detail.means[0] == pytest.approx([values["mean"]])
        assert detail.score[0] == pytest.approx([values["score"]])
        assert detail.imat[0][0] == pytest.approx([values["imat"]])
        assert detail.hazard == pytest.approx([values["hazard"]])
        assert detail.varhaz == pytest.approx([values["varhaz"]])
        assert detail.wtrisk == pytest.approx([3.5])
        assert detail.weights == pytest.approx(data["weight"])
        assert detail.nevent_wt == pytest.approx([3.0])
        assert detail.nrisk_wt == pytest.approx([3.5])

        scaled_weights = [10.0 * weight for weight in data["weight"]]
        scaled_fit = survival.coxph(
            "Surv(time, status) ~ x",
            data=data,
            weights=scaled_weights,
            init=[0.0],
            max_iter=0,
            method=method,
        )
        scaled_detail = survival.coxph_detail(scaled_fit)
        assert scaled_detail.means[0] == pytest.approx(detail.means[0])
        assert scaled_detail.hazard == pytest.approx(detail.hazard)
        assert scaled_detail.varhaz == pytest.approx(detail.varhaz)
        assert scaled_detail.nevent_wt == pytest.approx([30.0])
        assert scaled_detail.nrisk_wt == pytest.approx([35.0])


def test_coxph_detail_riskmat_honors_counting_entry_and_strata():
    data = _counting_cox_data() | {"group": ["A", "A", "A", "A", "B", "B"]}
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1 + strata(group)",
        data=data,
        init=[0.0],
        max_iter=0,
        method="breslow",
    )

    detail = survival.coxph_detail(fit, riskmat=True)

    assert detail.time == pytest.approx([2.0, 4.0, 5.0])
    assert detail.strata == {0: 2, 1: 1}
    assert detail.y[0] == pytest.approx([0.0, 2.0, 1.0])
    assert detail.riskmat is not None
    assert [row[0] for row in detail.riskmat] == [1, 1, 1, 0, 0, 0]
    assert [row[1] for row in detail.riskmat] == [0, 0, 1, 1, 0, 0]
    assert [row[2] for row in detail.riskmat] == [0, 0, 0, 0, 1, 1]

    time_order = survival.coxph_detail(fit, riskmat=True, rorder="time")
    time_prefix = survival.coxph_detail(fit, riskmat=True, rorder="t")
    data_prefix = survival.coxph_detail(fit, riskmat=True, rorder="d")
    assert time_order.sortorder == [0, 1, 2, 3, 4, 5]
    assert time_order.riskmat == detail.riskmat
    assert time_prefix.sortorder == time_order.sortorder
    assert time_prefix.riskmat == time_order.riskmat
    assert data_prefix.sortorder is None
    assert data_prefix.riskmat == detail.riskmat

    with pytest.raises(ValueError, match="rorder"):
        survival.coxph_detail(fit, rorder="event")


def test_coxph_detail_rejects_exact_ties_like_r():
    data = _tied_cox_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        init=[0.0],
        max_iter=0,
        method="exact",
    )

    with pytest.raises(ValueError, match="exact method"):
        survival.coxph_detail(fit)


def test_coxph_formula_accepts_counting_process_response():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    low_level = survival.regression.coxph_fit(
        data["stop"],
        data["status"],
        [[value] for value in data["x1"]],
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
        entry_times=data["start"],
    )

    assert fit.entry_times == pytest.approx(data["start"])
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.score_vector == pytest.approx(low_level.score_vector)

    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    raw_times, raw_hazard = survival.basehaz(
        data["stop"],
        data["status"],
        fit.linear_predictors,
        False,
        entry_times=data["start"],
    )
    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(raw_times, time)) == 0 else raw_hazard[pos - 1]
        for time in sorted(set(data["stop"]))
    ]
    assert fitted_times == pytest.approx(sorted(set(data["stop"])))
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)


def test_predict_coxph_r_style_generic_types():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[0.5, 0.8], [1.1, 0.3], [0.7, 0.9]]
    collapse = ["A", "A", "B"]

    lp = survival.predict(fit, rows)
    centered_lp = survival.predict(fit, rows, centered=True)
    risk = survival.predict(fit, rows, type="risk")
    prefix_lp = survival.predict(fit, rows, type="l")
    prefix_risk = survival.predict(fit, rows, type="r")
    uncentered_lp = survival.predict(fit, rows, reference="zero")
    terms = survival.predict(fit, rows, type="terms")
    prefix_terms = survival.predict(fit, rows, type="t")
    uncentered_terms = survival.predict(fit, rows, type="terms", reference="zero")
    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )

    assert lp == pytest.approx([value - center for value in direct_lp])
    assert prefix_lp == pytest.approx(lp)
    assert centered_lp == pytest.approx([value - center for value in direct_lp])
    assert uncentered_lp == pytest.approx(direct_lp)
    assert risk == pytest.approx([math.exp(value - center) for value in direct_lp])
    assert prefix_risk == pytest.approx(risk)
    expected_terms = [
        [
            (rows[row_idx][col_idx] - fit.means[col_idx]) * fit.coefficients[0][col_idx]
            for col_idx in range(2)
        ]
        for row_idx in range(len(rows))
    ]
    expected_uncentered_terms = [
        [rows[row_idx][col_idx] * fit.coefficients[0][col_idx] for col_idx in range(2)]
        for row_idx in range(len(rows))
    ]
    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(uncentered_terms, expected_uncentered_terms, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.predict(fit, rows, reference="zero", collapse=collapse) == pytest.approx(
        [direct_lp[0] + direct_lp[1], direct_lp[2]]
    )
    variance = fit.information_matrix
    centered_rows = [[row[col_idx] - fit.means[col_idx] for col_idx in range(2)] for row in rows]
    expected_lp_se = [
        math.sqrt(
            max(
                sum(
                    centered_row[row_idx] * variance[row_idx][col_idx] * centered_row[col_idx]
                    for row_idx in range(2)
                    for col_idx in range(2)
                ),
                0.0,
            )
        )
        for centered_row in centered_rows
    ]
    lp_with_se = survival.predict(fit, rows, se_fit=True)
    dotted_lp_with_se = survival.predict(fit, rows, **{"se.fit": True})
    unpacked_lp, unpacked_se = lp_with_se
    assert isinstance(lp_with_se, survival.r_api.PredictResult)
    assert lp_with_se.fit == pytest.approx(lp)
    assert lp_with_se.predictions == pytest.approx(lp)
    assert lp_with_se.se_fit == pytest.approx(expected_lp_se)
    assert lp_with_se.se == pytest.approx(expected_lp_se)
    assert isinstance(dotted_lp_with_se, survival.r_api.PredictResult)
    assert dotted_lp_with_se.fit == pytest.approx(lp_with_se.fit)
    assert dotted_lp_with_se.se_fit == pytest.approx(lp_with_se.se_fit)
    assert unpacked_lp == pytest.approx(lp)
    assert unpacked_se == pytest.approx(expected_lp_se)

    risk_with_se = survival.predict(fit, rows, type="risk", se_fit=True)
    assert risk_with_se.fit == pytest.approx(risk)
    assert risk_with_se.se_fit == pytest.approx(
        [se * value for se, value in zip(expected_lp_se, risk, strict=True)]
    )
    zero_se = [
        math.sqrt(
            max(
                sum(
                    row[row_idx] * variance[row_idx][col_idx] * row[col_idx]
                    for row_idx in range(2)
                    for col_idx in range(2)
                ),
                0.0,
            )
        )
        for row in rows
    ]
    collapsed_lp_with_se = survival.predict(
        fit,
        rows,
        reference="zero",
        collapse=collapse,
        se_fit=True,
    )
    assert collapsed_lp_with_se.fit == pytest.approx([direct_lp[0] + direct_lp[1], direct_lp[2]])
    assert collapsed_lp_with_se.se_fit == pytest.approx(
        [math.sqrt(zero_se[0] ** 2 + zero_se[1] ** 2), zero_se[2]]
    )
    assert survival.predict(
        fit,
        rows,
        type="risk",
        reference="zero",
        collapse=collapse,
    ) == pytest.approx([math.exp(direct_lp[0]) + math.exp(direct_lp[1]), math.exp(direct_lp[2])])
    collapsed_terms = survival.predict(
        fit,
        rows,
        type="terms",
        reference="zero",
        collapse=collapse,
    )
    expected_collapsed_terms = [
        [
            expected_uncentered_terms[0][col_idx] + expected_uncentered_terms[1][col_idx]
            for col_idx in range(2)
        ],
        expected_uncentered_terms[2],
    ]
    for actual, expected in zip(collapsed_terms, expected_collapsed_terms, strict=True):
        assert actual == pytest.approx(expected)
    terms_with_se = survival.predict(
        fit,
        rows,
        type="terms",
        reference="zero",
        collapse=collapse,
        se_fit=True,
    )
    expected_term_se = [
        [
            abs(row[col_idx]) * math.sqrt(max(variance[col_idx][col_idx], 0.0))
            for col_idx in range(2)
        ]
        for row in rows
    ]
    expected_collapsed_term_se = [
        [
            math.sqrt(expected_term_se[0][col_idx] ** 2 + expected_term_se[1][col_idx] ** 2)
            for col_idx in range(2)
        ],
        expected_term_se[2],
    ]
    for actual, expected in zip(terms_with_se.fit, expected_collapsed_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(terms_with_se.se_fit, expected_collapsed_term_se, strict=True):
        assert actual == pytest.approx(expected)

    full_times, full_curves = fit.survival_curve(rows, True)
    full_survfit = survival.survfit(fit, newdata=rows, censor=False)
    requested_times = [0.5, full_times[0], full_times[-1]]
    times, curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    assert curves[0][0] == pytest.approx(1.0)
    assert curves[0][1] == pytest.approx(full_curves[0][0])
    assert curves[0][-1] == pytest.approx(full_curves[0][-1])
    collapsed_times, collapsed_curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        collapse=collapse,
    )
    expected_collapsed_curves = [
        [curves[0][idx] + curves[1][idx] for idx in range(len(requested_times))],
        curves[2],
    ]
    assert collapsed_times == pytest.approx(requested_times)
    for actual, expected in zip(collapsed_curves, expected_collapsed_curves, strict=True):
        assert actual == pytest.approx(expected)
    all_times, all_curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        collapse=collapse,
    )
    expected_all_curves = [
        [full_curves[0][idx] + full_curves[1][idx] for idx in range(len(full_times))],
        full_curves[2],
    ]
    assert all_times == pytest.approx(full_times)
    for actual, expected in zip(all_curves, expected_all_curves, strict=True):
        assert actual == pytest.approx(expected)
    survival_with_se = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        se_fit=True,
    )
    (fit_times, fit_curves), (se_times, se_curves) = survival_with_se
    expected_se_curves = []
    for std_err_curve in full_survfit.std_err:
        expected_se = []
        for time in requested_times:
            pos = bisect_right(full_survfit.time, time)
            expected_se.append(0.0 if pos == 0 else std_err_curve[pos - 1])
        expected_se_curves.append(expected_se)
    assert fit_times == pytest.approx(requested_times)
    assert se_times == pytest.approx(requested_times)
    for actual, expected in zip(fit_curves, curves, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(se_curves, expected_se_curves, strict=True):
        assert actual == pytest.approx(expected)
    collapsed_with_se = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        collapse=collapse,
        se_fit=True,
    )
    collapsed_fit, collapsed_se = collapsed_with_se
    assert collapsed_fit[0] == pytest.approx(requested_times)
    assert collapsed_se[0] == pytest.approx(requested_times)
    for actual, expected in zip(collapsed_fit[1], expected_collapsed_curves, strict=True):
        assert actual == pytest.approx(expected)
    expected_collapsed_se = [
        [
            math.sqrt(expected_se_curves[0][idx] ** 2 + expected_se_curves[1][idx] ** 2)
            for idx in range(len(requested_times))
        ],
        expected_se_curves[2],
    ]
    for actual, expected in zip(collapsed_se[1], expected_collapsed_se, strict=True):
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="same length as predictions"):
        survival.predict(fit, rows, collapse=["A"])


def test_predict_coxph_formula_accepts_newdata_mapping():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[0.5, 0.8], [1.1, 0.3]]
    newdata = {"x1": [0.5, 1.1], "x2": [0.8, 0.3]}

    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )
    assert survival.predict(fit, newdata) == pytest.approx([value - center for value in direct_lp])
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(direct_lp)
    assert survival.predict(fit, newdata, type="risk") == pytest.approx(
        [math.exp(value - center) for value in direct_lp]
    )
    terms = survival.predict(fit, newdata, type="terms")
    expected_terms = [
        [(row[col_idx] - fit.means[col_idx]) * fit.coefficients[0][col_idx] for col_idx in range(2)]
        for row in rows
    ]
    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)


def test_predict_coxph_formula_newdata_mapping_uses_training_factor_levels():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ group + x1", data=data, max_iter=10, eps=1e-5)
    newdata = {"group": ["B", "A"], "x1": [0.5, 0.8]}
    rows = [[1.0, 0.5], [0.0, 0.8]]
    beta = fit.coefficients[0]
    center = fit.means[1] * beta[1]

    assert survival.predict(fit, newdata) == pytest.approx(
        [value - center for value in fit.predict(rows)]
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(fit.predict(rows))
    with pytest.raises(ValueError, match="unknown level"):
        survival.predict(fit, {"group": ["C"], "x1": [0.5]})


def test_predict_coxph_terms_groups_formula_terms_and_selects_by_name_or_index():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"dose": [0, 1, 2], "x1": [0.5, 0.8, 1.1]}
    beta = fit.coefficients[0]
    expected_terms = [
        [0.0, (0.5 - fit.means[2]) * beta[2]],
        [beta[0], (0.8 - fit.means[2]) * beta[2]],
        [beta[1], (1.1 - fit.means[2]) * beta[2]],
    ]

    terms = survival.predict(fit, newdata, type="terms")
    factor_terms = survival.predict(fit, newdata, type="terms", terms="factor(dose)")
    x1_terms = survival.predict(fit, newdata, type="terms", terms=[2])

    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(factor_terms, [[row[0]] for row in expected_terms], strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(x1_terms, [[row[1]] for row in expected_terms], strict=True):
        assert actual == pytest.approx(expected)

    with pytest.raises(ValueError, match="unknown model term"):
        survival.predict(fit, newdata, type="terms", terms=["missing"])


def test_predict_coxph_terms_selects_matrix_fit_columns_with_one_based_indices():
    data = _toy_data()
    fit = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    beta = fit.coefficients[0]

    selected_x2 = survival.predict(fit, rows, type="terms", terms=[2])
    selected_x1 = survival.predict(fit, rows, type="terms", terms="x1")
    for actual, expected in zip(
        selected_x2,
        [[(0.8 - fit.means[1]) * beta[1]], [(0.3 - fit.means[1]) * beta[1]]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        selected_x1,
        [[(0.5 - fit.means[0]) * beta[0]], [(1.1 - fit.means[0]) * beta[0]]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_predict_coxph_formula_newdata_mapping_rebuilds_transforms_and_interactions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ log(x1) + x1:x2 + group:x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.5, 1.0], "x2": [0.25, 0.8], "group": ["B", "A"]}
    rows = [
        [math.log(0.5), 0.5 * 0.25, 0.0 * 0.25, 1.0 * 0.25],
        [math.log(1.0), 1.0 * 0.8, 1.0 * 0.8, 0.0 * 0.8],
    ]

    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )
    assert survival.predict(fit, newdata) == pytest.approx([value - center for value in direct_lp])
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(direct_lp)


def test_predict_coxph_formula_offset_newdata_mapping_survival():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "offset": offsets}
    linear_predictors = [
        value + offset for value, offset in zip(fit.predict(rows), offsets, strict=True)
    ]
    center = sum(fit.linear_predictors) / len(fit.linear_predictors)
    baseline_times, hazards = fit.basehaz(True)
    full_curves = [
        [math.exp(-hazard * math.exp(lp - center)) for hazard in hazards]
        for lp in linear_predictors
    ]
    requested_times = [0.5, baseline_times[0], baseline_times[-1]]

    times, curves = survival.predict(
        fit,
        newdata,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    for actual, expected_curve in zip(curves, full_curves, strict=True):
        assert actual == pytest.approx([1.0, expected_curve[0], expected_curve[-1]])


def test_predict_coxph_formula_rebuilds_transformed_offsets_from_newdata():
    data = _toy_data()
    data["exposure"] = [math.exp(value) for value in data["offset"]]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "exposure": [math.exp(value) for value in offsets]}
    expected = [value + offset for value, offset in zip(fit.predict(rows), offsets, strict=True)]

    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(expected)


def test_cox_zph_rank_transform_matches_low_level_ph_test():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10, eps=1e-5)
    raw = fit.schoenfeld_residuals()
    scaled = fit.scaled_schoenfeld_residuals()
    ranks = list(range(1, len(raw) + 1))
    low_level = survival.ph_test(scaled, ranks, None)
    raw_level = survival.ph_test(raw, ranks, None)

    result = survival.cox_zph(fit, transform="rank", terms=False)

    assert isinstance(result, survival.r_api.CoxZPHResult)
    assert result.variable_names == ["x1", "x2"]
    assert result.x == pytest.approx(ranks)
    assert result.time == pytest.approx(_fit_event_times(fit))
    assert result.chi2_values == pytest.approx(low_level.chi2_values)
    assert result.p_values == pytest.approx(low_level.p_values)
    assert result.global_chi2 == pytest.approx(low_level.global_chi2)
    assert result.global_df == low_level.global_df
    assert result.global_p_value == pytest.approx(low_level.global_p_value)
    assert result.global_chi2 != pytest.approx(raw_level.global_chi2)
    for actual, expected in zip(result.y, scaled, strict=True):
        assert actual == pytest.approx(expected)
    assert result.table[-1]["name"] == "GLOBAL"


def test_cox_zph_scales_variance_preserves_strata_and_subsets_variables():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    result = survival.cox_zph(fit, transform="rank", terms=False)
    event_indices = r_coxph._cox_event_indices(fit)
    event_count = len(event_indices)

    assert result.strata == [data["group"][idx] for idx in event_indices]
    for actual, expected in zip(result.var, fit.information_matrix, strict=True):
        assert actual == pytest.approx([event_count * value for value in expected])

    reversed_result = result.subset([1, 0])
    assert reversed_result.variable_names == ["x2", "x1"]
    assert reversed_result.chi2_values == pytest.approx(
        [result.chi2_values[1], result.chi2_values[0]]
    )
    assert reversed_result.strata == result.strata
    assert reversed_result.global_chi2 is None
    for actual, expected in zip(reversed_result.y, result.y, strict=True):
        assert actual == pytest.approx([expected[1], expected[0]])
    assert reversed_result.var[0] == pytest.approx([result.var[1][1], result.var[1][0]])
    assert reversed_result.var[1] == pytest.approx([result.var[0][1], result.var[0][0]])

    with_global = result.subset([0], include_global=True)
    assert [row["name"] for row in with_global.table] == ["x1", "GLOBAL"]

    empty = result.subset([])
    assert empty.variable_names == []
    assert empty.x == []
    assert empty.time == []
    assert empty.y == []
    assert empty.var == []
    assert empty.strata == []

    with pytest.raises(IndexError, match="out of range"):
        result.subset([2])


def test_cox_zph_clustered_fit_uses_robust_scaled_schoenfeld_residuals():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    raw = fit.schoenfeld_residuals()
    robust_scaled = survival.r_api.residuals(fit, type="scaledsch")
    naive_scaled = fit.fit.scaled_schoenfeld_residuals()
    ranks = list(range(1, len(raw) + 1))
    low_level = survival.ph_test(robust_scaled, ranks, None)
    naive_level = survival.ph_test(naive_scaled, ranks, None)

    result = survival.cox_zph(fit, transform="rank", terms=False)

    assert result.variable_names == ["x1", "x2"]
    assert result.x == pytest.approx(ranks)
    assert result.chi2_values == pytest.approx(low_level.chi2_values)
    assert result.p_values == pytest.approx(low_level.p_values)
    assert result.global_chi2 == pytest.approx(low_level.global_chi2)
    assert result.global_df == low_level.global_df
    assert result.global_p_value == pytest.approx(low_level.global_p_value)
    assert result.global_chi2 != pytest.approx(naive_level.global_chi2)
    for actual, expected in zip(fit.scaled_schoenfeld_residuals(), robust_scaled, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.scaled_schoenfeld_residuals()[0] != pytest.approx(naive_scaled[0])
    for actual, expected in zip(result.y, robust_scaled, strict=True):
        assert actual == pytest.approx(expected)
    event_count = len(r_coxph._cox_event_indices(fit))
    for actual, expected in zip(result.var, fit.naive_information_matrix, strict=True):
        assert actual == pytest.approx([event_count * value for value in expected])


def test_cox_zph_identity_and_km_transforms_expose_r_style_time_axes():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=10, eps=1e-5)
    event_times = _fit_event_times(fit)

    identity = survival.cox_zph(fit, transform="identity")
    ranked = survival.cox_zph(fit, transform="rank")
    km = survival.cox_zph(fit)
    identity_prefix = survival.cox_zph(fit, transform="i")
    ranked_prefix = survival.cox_zph(fit, transform="r")
    km_prefix = survival.cox_zph(fit, transform="k")
    km_alias = survival.cox_zph(fit, transform="kaplan_meier")
    logged_prefix = survival.cox_zph(fit, transform="l")
    low_level = survival.survfitkm(data["time"], data["status"], conf_type="none")
    expected_km = []
    cursor = 0
    for event_time in event_times:
        while cursor < len(low_level.time) and low_level.time[cursor] < event_time - 1e-9:
            cursor += 1
        previous_survival = low_level.estimate[cursor - 1] if cursor else 1.0
        expected_km.append(1.0 - previous_survival)

    assert identity.transform == "identity"
    assert identity.x == pytest.approx(event_times)
    assert identity_prefix.transform == "identity"
    assert identity_prefix.x == pytest.approx(identity.x)
    assert ranked.transform == "rank"
    assert ranked.x == pytest.approx(list(range(1, len(event_times) + 1)))
    assert ranked_prefix.transform == "rank"
    assert ranked_prefix.x == pytest.approx(ranked.x)
    assert km.transform == "km"
    assert km.x == pytest.approx(expected_km)
    assert km_prefix.transform == "km"
    assert km_prefix.x == pytest.approx(km.x)
    assert km_alias.transform == "km"
    assert km_alias.x == pytest.approx(km.x)
    assert logged_prefix.transform == "log"
    assert logged_prefix.x == pytest.approx([math.log(time) for time in event_times])
    assert km.x != pytest.approx(identity.x)


def test_cox_zph_rejects_unknown_transform():
    fit = survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=10, eps=1e-5)

    with pytest.raises(ValueError, match="transform"):
        survival.cox_zph(fit, transform="weird")


def test_cox_zph_formula_terms_group_multi_column_factors():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    by_term = survival.cox_zph(fit, transform="rank")
    by_column = survival.cox_zph(fit, transform="rank", terms=False)
    single_df = survival.cox_zph(fit, transform="rank", singledf=True)

    assert by_term.variable_names == ["factor(dose)", "x1"]
    assert by_term.df == [2, 1]
    assert by_column.variable_names == ["factor(dose)1", "factor(dose)2", "x1"]
    assert by_column.df == [1, 1, 1]
    assert single_df.df == [1, 1]
    assert len(by_term.y[0]) == 2
    assert len(by_column.y[0]) == 3
    assert by_term.table[-1]["name"] == "GLOBAL"
    assert survival.cox_zph(fit, global_test=False).table[-1]["name"] != "GLOBAL"
    assert survival.cox_zph(fit, **{"global": False}).table[-1]["name"] != "GLOBAL"


def test_cox_zph_drops_aliased_columns_like_explicitly_reduced_fit():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
    }
    data["x2"] = [2.0 * value for value in data["x1"]]
    aliased_fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )
    reduced_fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert math.isnan(survival.coef(aliased_fit)[1])
    for terms in (True, False):
        result = survival.cox_zph(aliased_fit, transform="rank", terms=terms)
        reduced = survival.cox_zph(reduced_fit, transform="rank", terms=terms)

        assert result.variable_names == reduced.variable_names == ["x1"]
        assert result.df == reduced.df == [1]
        assert result.chi2_values == pytest.approx(reduced.chi2_values)
        assert result.p_values == pytest.approx(reduced.p_values)
        assert result.x == pytest.approx(reduced.x)
        assert result.time == pytest.approx(reduced.time)
        for actual, expected in zip(result.y, reduced.y, strict=True):
            assert actual == pytest.approx(expected)
        for actual, expected in zip(result.var, reduced.var, strict=True):
            assert actual == pytest.approx(expected)
        assert result.global_chi2 == pytest.approx(reduced.global_chi2)
        assert result.global_df == reduced.global_df == 1
        assert result.global_p_value == pytest.approx(reduced.global_p_value)
        assert [row["name"] for row in result.table] == ["x1", "GLOBAL"]


def test_cox_zph_filters_aliases_with_robust_scaling_and_variance():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
        "subject": ["a", "a", "b", "b", "c", "c", "d", "d"],
    }
    data["x2"] = [2.0 * value for value in data["x1"]]
    aliased_fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )
    reduced_fit = survival.coxph(
        "Surv(time, status) ~ x1 + cluster(subject)",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert aliased_fit.robust is True
    assert math.isnan(survival.coef(aliased_fit)[1])
    for terms in (True, False):
        result = survival.cox_zph(aliased_fit, transform="rank", terms=terms)
        reduced = survival.cox_zph(reduced_fit, transform="rank", terms=terms)

        assert result.variable_names == reduced.variable_names == ["x1"]
        assert result.df == reduced.df == [1]
        assert result.chi2_values == pytest.approx(reduced.chi2_values)
        assert result.p_values == pytest.approx(reduced.p_values)
        assert result.global_chi2 == pytest.approx(reduced.global_chi2)
        assert result.global_df == reduced.global_df == 1
        for actual, expected in zip(result.y, reduced.y, strict=True):
            assert actual == pytest.approx(expected)
        for actual, expected in zip(result.var, reduced.var, strict=True):
            assert actual == pytest.approx(expected)


def test_cox_zph_preserves_survivor_names_when_the_first_column_is_aliased():
    data = {
        "time": list(range(1, 9)),
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "constant": [1.0] * 8,
        "x": [1.0, 0.9, 1.1, 0.7, 0.4, 0.3, 0.6, 0.2],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ constant + x",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )
    scaled = survival.r_api.residuals(fit, type="scaledsch")

    assert math.isnan(survival.coef(fit)[0])
    for terms in (True, False):
        result = survival.cox_zph(fit, transform="rank", terms=terms)

        assert result.variable_names == ["x"]
        assert result.df == [1]
        assert result.global_df == 1
        for actual, expected in zip(result.y, scaled, strict=True):
            assert actual == pytest.approx([expected[1]])
        assert len(result.var) == 1
        assert len(result.var[0]) == 1


def test_cox_zph_remaps_partially_aliased_multi_column_terms():
    data = {
        "time": list(range(1, 9)),
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "group": ["a", "a", "b", "b", "c", "c", "a", "b"],
        "is_b": [0, 0, 1, 1, 0, 0, 0, 1],
        "x": [1.0, 0.9, 1.1, 0.7, 0.4, 0.3, 0.6, 0.2],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ is_b + factor(group) + x",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert math.isnan(survival.coef(fit)[1])
    by_term = survival.cox_zph(fit, transform="rank", terms=True)
    by_column = survival.cox_zph(fit, transform="rank", terms=False)

    assert by_term.variable_names == ["is_b", "factor(group)", "x"]
    assert by_term.df == [1, 1, 1]
    assert by_term.global_df == 3
    assert by_column.variable_names == ["is_b", "factor(group)c", "x"]
    assert by_column.df == [1, 1, 1]
    assert by_column.global_df == 3
    assert all(len(row) == 3 for row in by_term.y)
    assert all(len(row) == 3 for row in by_column.y)
    assert len(by_term.var) == len(by_column.var) == 3
    assert all(len(row) == 3 for row in by_term.var)
    assert all(len(row) == 3 for row in by_column.var)


@pytest.mark.parametrize(
    "formula",
    ["Surv(time, status) ~ 1", "Surv(time, status) ~ constant"],
)
def test_cox_zph_rejects_fits_without_estimable_coefficients(formula):
    data = {
        "time": list(range(1, 9)),
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "constant": [1.0] * 8,
    }
    fit = survival.coxph(
        formula,
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    with pytest.raises(ValueError, match="at least one estimable coefficient"):
        survival.cox_zph(fit)


def test_cox_zph_rejects_fits_without_events():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [0, 0, 0, 0],
        "x": [0.2, 0.8, 0.4, 1.1],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x",
        data=data,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    with pytest.raises(ValueError, match="at least one event"):
        survival.cox_zph(fit)


def test_coxph_partial_residuals_add_terms_to_martingales():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10, eps=1e-5)
    martingale = fit.martingale_residuals()
    term_predictions = survival.predict(fit, type="terms")
    expected = [
        [martingale[row_idx] + term for term in row] for row_idx, row in enumerate(term_predictions)
    ]

    for alias in ("partial", "partials", "p"):
        for actual, expected_row in zip(
            survival.r_api.residuals(fit, type=alias),
            expected,
            strict=True,
        ):
            assert actual == pytest.approx(expected_row)

    selected_x2 = survival.r_api.residuals(fit, type="partial", terms="x2")
    selected_by_index = survival.r_api.residuals(fit, type="partial", terms=[2, 1])
    for row_idx, actual in enumerate(selected_x2):
        assert actual == pytest.approx([expected[row_idx][1]])
    for row_idx, actual in enumerate(selected_by_index):
        assert actual == pytest.approx([expected[row_idx][1], expected[row_idx][0]])

    with pytest.raises(ValueError, match="unknown model term"):
        survival.r_api.residuals(fit, type="partial", terms="missing")


def test_coxph_partial_residuals_group_factor_coefficients_by_formula_term():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ as.factor(dose) + x1",
        data=data,
        initial_beta=[0.2, -0.1, 0.3],
        max_iter=0,
    )
    martingale = fit.martingale_residuals()
    term_predictions = survival.predict(fit, type="terms")

    partial = survival.r_api.residuals(fit, type="partial")
    selected = survival.r_api.residuals(
        fit,
        type="partial",
        terms="as.factor(dose)",
    )

    assert survival.model_term_names(fit) == ["as.factor(dose)", "x1"]
    assert all(len(row) == 2 for row in partial)
    for row_idx, actual in enumerate(partial):
        assert actual == pytest.approx(
            [martingale[row_idx] + value for value in term_predictions[row_idx]]
        )
        assert selected[row_idx] == pytest.approx([actual[0]])


def test_coxph_counting_process_partial_residuals_keep_training_row_order():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        method="breslow",
        initial_beta=[0.25],
        max_iter=0,
    )
    martingale = fit.martingale_residuals()
    term_predictions = survival.predict(fit, type="terms")
    expected = [
        [martingale[row_idx] + term_predictions[row_idx][0]] for row_idx in range(len(martingale))
    ]

    partial = survival.r_api.residuals(fit, type="partial")
    assert len(partial) == len(data["stop"])
    for actual, expected_row in zip(partial, expected, strict=True):
        assert actual == pytest.approx(expected_row)


def test_coxph_residuals_honor_case_weighting_rules():
    data = _toy_data()
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.75, 2.5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.1, -0.2],
        max_iter=0,
    )

    martingale = fit.martingale_residuals()
    assert survival.r_api.residuals(fit, type="martingale", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(martingale)]
    )
    assert survival.r_api.residuals(fit, type="m") == pytest.approx(martingale)

    score = fit.score_residuals()
    weighted_score = survival.r_api.residuals(fit, type="score", weighted=True)
    for row_idx, (actual, expected_row) in enumerate(zip(weighted_score, score, strict=True)):
        assert actual == pytest.approx([value * weights[row_idx] for value in expected_row])

    dfbeta = fit.dfbeta()
    default_dfbeta = survival.r_api.residuals(fit, type="dfbeta")
    unweighted_dfbeta = survival.r_api.residuals(fit, type="dfbeta", weighted=False)
    for row_idx, (default_row, raw_row, unweighted_row) in enumerate(
        zip(default_dfbeta, dfbeta, unweighted_dfbeta, strict=True)
    ):
        assert unweighted_row == pytest.approx(raw_row)
        assert default_row == pytest.approx([value * weights[row_idx] for value in raw_row])

    terms = survival.predict(fit, type="terms")
    partial = survival.r_api.residuals(fit, type="partial", weighted=True)
    for row_idx, actual in enumerate(partial):
        assert actual == pytest.approx(
            [term + martingale[row_idx] * weights[row_idx] for term in terms[row_idx]]
        )


def test_coxph_residuals_collapse_training_rows_by_label():
    data = _counting_cox_data()
    collapse = ["A", "A", "B", "B", "C", "C"]
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        method="breslow",
        initial_beta=[0.25],
        max_iter=0,
    )

    martingale = fit.martingale_residuals()
    expected_martingale = [
        sum(martingale[idx] for idx, label in enumerate(collapse) if label == group)
        for group in ("A", "B", "C")
    ]
    assert survival.r_api.residuals(
        fit,
        type="martingale",
        collapse=collapse,
    ) == pytest.approx(expected_martingale)

    score = fit.score_residuals()
    collapsed_score = survival.r_api.residuals(fit, type="score", collapse=collapse)
    expected_score = [
        [sum(score[idx][0] for idx, label in enumerate(collapse) if label == group)]
        for group in ("A", "B", "C")
    ]
    for actual, expected_row in zip(collapsed_score, expected_score, strict=True):
        assert actual == pytest.approx(expected_row)

    partial = survival.r_api.residuals(fit, type="partial")
    collapsed_partial = survival.r_api.residuals(fit, type="partial", collapse=collapse)
    expected_partial = [
        [sum(partial[idx][0] for idx, label in enumerate(collapse) if label == group)]
        for group in ("A", "B", "C")
    ]
    for actual, expected_row in zip(collapsed_partial, expected_partial, strict=True):
        assert actual == pytest.approx(expected_row)

    collapsed_status = [
        sum(fit.status[idx] for idx, label in enumerate(collapse) if label == group)
        for group in ("A", "B", "C")
    ]
    expected_deviance = []
    for residual, status in zip(expected_martingale, collapsed_status, strict=True):
        log_term = status * math.log(max(status - residual, 1e-12)) if status > 0 else 0.0
        magnitude = math.sqrt(max(-2.0 * (residual + log_term), 0.0))
        expected_deviance.append(magnitude if residual >= 0.0 else -magnitude)
    assert survival.r_api.residuals(
        fit,
        type="deviance",
        collapse=collapse,
    ) == pytest.approx(expected_deviance)


def test_coxph_event_residuals_support_weighted_schoenfeld_output():
    data = _toy_data()
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.75, 2.5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.1, -0.2],
        max_iter=0,
    )
    raw = fit.schoenfeld_residuals()
    event_weights = [weights[idx] for idx, status in enumerate(data["status"]) if status == 1]
    weighted_raw = [
        [value * event_weights[row_idx] for value in row] for row_idx, row in enumerate(raw)
    ]

    for actual, expected_row in zip(
        survival.r_api.residuals(fit, type="schoenfeld", weighted=True),
        weighted_raw,
        strict=True,
    ):
        assert actual == pytest.approx(expected_row)

    beta = fit.coefficients[0]
    variance = fit.information_matrix
    expected_scaled = [
        [
            beta[col_idx]
            + len(weighted_raw)
            * sum(row[inner_idx] * variance[inner_idx][col_idx] for inner_idx in range(2))
            for col_idx in range(2)
        ]
        for row in weighted_raw
    ]
    for actual, expected_row in zip(
        survival.r_api.residuals(fit, type="scaledsch", weighted=True),
        expected_scaled,
        strict=True,
    ):
        assert actual == pytest.approx(expected_row)


def test_coxph_schoenfeld_residuals_match_event_risk_set_means():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]
    risks = [math.exp(value) for value in fit.linear_predictors]
    expected = []
    for event_idx, status in enumerate(data["status"]):
        if status != 1:
            continue
        at_risk = [idx for idx, time in enumerate(data["time"]) if time >= data["time"][event_idx]]
        denom = sum(risks[idx] for idx in at_risk)
        means = [
            sum(risks[idx] * rows[idx][col_idx] for idx in at_risk) / denom for col_idx in range(2)
        ]
        expected.append([rows[event_idx][col_idx] - means[col_idx] for col_idx in range(2)])

    assert fit.method == "efron"
    for actual, expected_row in zip(fit.covariates, rows, strict=True):
        assert actual == pytest.approx(expected_row)
    for residuals in (
        fit.schoenfeld_residuals(),
        survival.r_api.residuals(fit, type="schoenfeld"),
        survival.r_api.residuals(fit, type="sch"),
        survival.r_api.residuals(fit, type="scho"),
    ):
        assert len(residuals) == len(expected)
        for actual, expected_row in zip(residuals, expected, strict=True):
            assert actual == pytest.approx(expected_row)


def test_coxph_scaled_schoenfeld_residuals_use_r_scaling():
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_toy_data(),
        max_iter=10,
        eps=1e-5,
    )
    raw = fit.schoenfeld_residuals()
    beta = fit.coefficients[0]
    variance = fit.information_matrix
    event_count = len(raw)
    expected = [
        [
            beta[col_idx]
            + event_count
            * sum(row[inner_idx] * variance[inner_idx][col_idx] for inner_idx in range(2))
            for col_idx in range(2)
        ]
        for row in raw
    ]

    scaled = fit.scaled_schoenfeld_residuals()
    for actual, expected_row in zip(scaled, expected, strict=True):
        assert actual == pytest.approx(expected_row)
    for alias in ("scaledsch", "scaledschoenfeld", "scaled_schoenfeld", "sca"):
        for actual, expected_row in zip(
            survival.r_api.residuals(fit, type=alias),
            expected,
            strict=True,
        ):
            assert actual == pytest.approx(expected_row)


def test_coxph_score_and_dfbeta_residuals_use_fitted_information():
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_toy_data(),
        ties="efron",
        max_iter=10,
        eps=1e-5,
    )
    score = fit.score_residuals()
    dfbeta = fit.dfbeta()
    dfbetas = fit.dfbetas()

    for actual_matrix, expected_matrix in (
        (survival.r_api.residuals(fit, type="score"), score),
        (survival.r_api.residuals(fit, type="sco"), score),
        (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
        (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
    ):
        for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
            assert actual == pytest.approx(expected)
    assert len(score) == len(fit.event_times)
    assert len(dfbeta) == len(fit.event_times)
    assert len(dfbetas) == len(fit.event_times)
    for row_idx, (score_row, dfbeta_row, dfbetas_row) in enumerate(
        zip(score, dfbeta, dfbetas, strict=True)
    ):
        assert len(score_row) == 2
        assert len(dfbeta_row) == 2
        assert len(dfbetas_row) == 2
        for col_idx in range(2):
            expected_dfbeta = sum(
                fit.information_matrix[col_idx][inner_idx] * score_row[inner_idx]
                for inner_idx in range(2)
            )
            scale = math.sqrt(abs(fit.information_matrix[col_idx][col_idx]))
            assert dfbeta_row[col_idx] == pytest.approx(expected_dfbeta)
            assert dfbetas_row[col_idx] == pytest.approx(dfbeta[row_idx][col_idx] / scale)


@pytest.mark.parametrize(
    ("method", "expected_score", "expected_dfbeta", "expected_robust_variance"),
    [
        (
            "breslow",
            [
                -0.566205435493211,
                0.118975338134561,
                -0.247269050248814,
                -0.385057268840082,
                -0.244685141396534,
                1.32424155784408,
            ],
            [
                -0.145720034481586,
                0.0306197879579248,
                -0.0636377757431793,
                -0.0990992932519008,
                -0.0629727745555388,
                0.34081009007428,
            ],
            0.15615942412013,
        ),
        (
            "efron",
            [
                -0.629825947484865,
                0.116681966051309,
                -0.218899195194712,
                -0.400556408828364,
                -0.25072229332147,
                1.3833218787781,
            ],
            [
                -0.158565652518724,
                0.0293759762645186,
                -0.0551102949322474,
                -0.10084450885211,
                -0.0631221120696446,
                0.348266592108208,
            ],
            0.164486793924122,
        ),
    ],
)
def test_coxph_mixed_event_censor_ties_match_r_score_inference(
    method,
    expected_score,
    expected_dfbeta,
    expected_robust_variance,
):
    data = {
        "time": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0, 1, 0],
        "x": [0.0, 1.0, 0.5, 1.5, 2.0, -0.5],
        "id": list(range(6)),
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x",
        data=data,
        ties=method,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )
    robust_fit = survival.coxph(
        "Surv(time, status) ~ x",
        data=data,
        ties=method,
        cluster=data["id"],
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    score = [row[0] for row in fit.score_residuals()]
    dfbeta = [row[0] for row in fit.dfbeta()]

    assert score == pytest.approx(expected_score, abs=1e-12)
    assert dfbeta == pytest.approx(expected_dfbeta, abs=1e-12)
    assert sum(score) == pytest.approx(fit.score_vector[0], abs=1e-12)
    assert robust_fit.variance_matrix[0][0] == pytest.approx(
        expected_robust_variance,
        abs=1e-12,
    )


def test_coxph_counting_process_score_and_dfbeta_residuals_sum_to_score_vector():
    data = _counting_cox_data()
    for method in ("breslow", "efron", "exact"):
        fit = survival.coxph(
            "Surv(start, stop, status) ~ x1",
            data=data,
            method=method,
            initial_beta=[0.0],
            max_iter=0,
        )
        score = fit.score_residuals()
        dfbeta = fit.dfbeta()
        dfbetas = fit.dfbetas()

        assert len(score) == len(data["stop"])
        assert [sum(row[col_idx] for row in score) for col_idx in range(1)] == pytest.approx(
            fit.score_vector
        )
        for actual_matrix, expected_matrix in (
            (survival.r_api.residuals(fit, type="score"), score),
            (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
            (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
        ):
            for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
                assert actual == pytest.approx(expected)
        for row_idx, score_row in enumerate(score):
            expected_dfbeta = fit.information_matrix[0][0] * score_row[0]
            scale = math.sqrt(abs(fit.information_matrix[0][0]))
            assert dfbeta[row_idx][0] == pytest.approx(expected_dfbeta)
            assert dfbetas[row_idx][0] == pytest.approx(expected_dfbeta / scale)


@pytest.mark.parametrize(
    ("method", "expected_coef", "expected_score", "expected_dfbeta", "expected_variance"),
    [
        (
            "breslow",
            [0.0459536309248242, 0.353370968238964],
            [-0.477002515097761, 0.0460464768418501],
            [-0.143794564182129, -0.0193305596279085],
            [[0.108272854521586, 0.0296127805336108], [0.0296127805336108, 0.227116707749646]],
        ),
        (
            "efron",
            [-0.0071109501556483, 0.280700887836646],
            [-0.581824262049244, 0.0154020097871799],
            [-0.182250992353341, -0.0380955483800846],
            [[0.149167505896732, 0.0546676978998634], [0.0546676978998634, 0.278398896247739]],
        ),
    ],
)
def test_coxph_counting_process_score_inference_matches_r(
    method,
    expected_coef,
    expected_score,
    expected_dfbeta,
    expected_variance,
):
    data = {
        "start": [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        "stop": [2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "x": [-1.2, 0.4, 1.1, -0.3, 0.8, 1.7, -0.9, 0.2],
        "z": [0.5, -1.0, 0.3, 1.2, -0.7, 0.9, 0.1, -1.3],
        "group": [0, 0, 0, 0, 0, 1, 1, 1],
        "id": list(range(8)),
    }
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x + z + strata(group)",
        data=data,
        weights=[1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3],
        ties=method,
        cluster=data["id"],
        max_iter=50,
        eps=1e-9,
        toler=1e-10,
    )

    assert fit.coefficients[0] == pytest.approx(expected_coef, abs=1e-12)
    assert fit.score_residuals()[0] == pytest.approx(expected_score, abs=1e-12)
    assert fit.dfbeta()[0] == pytest.approx(expected_dfbeta, abs=1e-12)
    for actual, expected in zip(fit.variance_matrix, expected_variance, strict=True):
        assert actual == pytest.approx(expected, abs=1e-12)


def test_coxph_exact_tie_score_and_dfbeta_residuals_sum_to_score_vector():
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=_tied_cox_data(),
        ties="exact",
        initial_beta=[0.0],
        max_iter=0,
    )
    score = fit.score_residuals()
    dfbeta = fit.dfbeta()
    dfbetas = fit.dfbetas()

    assert [sum(row[col_idx] for row in score) for col_idx in range(1)] == pytest.approx(
        fit.score_vector
    )
    for actual_matrix, expected_matrix in (
        (survival.r_api.residuals(fit, type="score"), score),
        (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
        (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
    ):
        for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
            assert actual == pytest.approx(expected)
    for row_idx, score_row in enumerate(score):
        expected_dfbeta = fit.information_matrix[0][0] * score_row[0]
        scale = math.sqrt(abs(fit.information_matrix[0][0]))
        assert dfbeta[row_idx][0] == pytest.approx(expected_dfbeta)
        assert dfbetas[row_idx][0] == pytest.approx(expected_dfbeta / scale)


def test_coxph_formula_treatment_codes_categorical_covariates():
    fit = survival.coxph("Surv(time, status) ~ group + x1", data=_toy_data(), max_iter=10)

    assert sum(len(row) for row in fit.coefficients) == 2
    assert len(fit.predict([[1.0, 0.5]])) == 1


def test_coxph_formula_factor_treatment_codes_numeric_covariates():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [
                1.0 if data["dose"][idx] == 1 else 0.0,
                1.0 if data["dose"][idx] == 2 else 0.0,
                data["x1"][idx],
            ]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_numeric_transforms():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ log(x2) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [math.log(data["x2"][idx]), data["x1"][idx]] for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_identity_wrappers_for_numeric_terms():
    data = _toy_data()
    direct = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)

    for wrapper in ("I", "identity", "as.numeric"):
        fit = survival.coxph(
            f"Surv(time, status) ~ {wrapper}(x1) + x2",
            data=data,
            max_iter=10,
            eps=1e-5,
        )

        assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
        assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_accepts_identity_arithmetic_terms():
    data = _numeric_data()
    fit = survival.coxph(
        "Surv(time, status) ~ I(-x1) + I(x1 + x2) + I((x1 + x2)^2)",
        data=data,
        max_iter=0,
    )
    expected_rows = [
        [
            -data["x1"][idx],
            data["x1"][idx] + data["x2"][idx],
            (data["x1"][idx] + data["x2"][idx]) ** 2,
        ]
        for idx in range(len(data["time"]))
    ]
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=expected_rows,
        max_iter=0,
    )

    for actual, expected in zip(fit.covariates, expected_rows, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)

    data_with_missing = _numeric_data()
    data_with_missing["x2"][1] = None
    filtered = survival.coxph(
        "Surv(time, status) ~ I(x1 + x2)",
        data=data_with_missing,
        na_action="omit",
        max_iter=0,
    )
    expected_indices = [
        idx for idx, value in enumerate(data_with_missing["x2"]) if value is not None
    ]
    assert filtered.event_times == pytest.approx(
        [data_with_missing["time"][idx] for idx in expected_indices]
    )
    expected_filtered_rows = [
        [data_with_missing["x1"][idx] + data_with_missing["x2"][idx]] for idx in expected_indices
    ]
    for actual, expected in zip(filtered.covariates, expected_filtered_rows, strict=True):
        assert actual == pytest.approx(expected)


def test_coxph_formula_accepts_numeric_interactions():
    data = _numeric_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [data["x1"][idx], data["x2"][idx], data["x1"][idx] * data["x2"][idx]]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_categorical_interactions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ group * x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [
                1.0 if data["group"][idx] == "B" else 0.0,
                data["x1"][idx],
                data["x1"][idx] if data["group"][idx] == "B" else 0.0,
            ]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


@pytest.mark.parametrize(
    ("rhs", "columns"),
    [
        ("g:x", ["gA:x", "gB:x", "gC:x"]),
        ("x + g:x", ["x", "x:gB", "x:gC"]),
        ("g + g:x", ["gB", "gC", "gA:x", "gB:x", "gC:x"]),
        ("g * x", ["gB", "gC", "x", "gB:x", "gC:x"]),
        ("g * h", ["gB", "gC", "hH", "gB:hH", "gC:hH"]),
        (
            "g:h",
            ["gA:hL", "gB:hL", "gC:hL", "gA:hH", "gB:hH", "gC:hH"],
        ),
    ],
)
def test_coxph_interaction_contrasts_match_r_with_virtual_intercept(rhs, columns):
    data = _interaction_contrast_data()
    expected_rows = _interaction_contrast_rows(data, columns)

    for intercept_suffix in ("", " + 0", " - 1"):
        fit = survival.coxph(
            f"Surv(time, status) ~ {rhs}{intercept_suffix}",
            data=data,
            max_iter=0,
        )
        matrix = survival.model_matrix(fit)

        assert survival.coef_names(fit) == columns
        assert matrix["columns"] == columns
        for actual, expected in zip(matrix["data"], expected_rows, strict=True):
            assert actual == pytest.approx(expected)


def test_coxph_formula_defaults_to_efron_ties():
    data = _tied_cox_data()
    default = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=15, eps=1e-5)
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    breslow = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert default.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert efron.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert efron.risk_scores == pytest.approx(low_level.risk_scores)
    assert efron.coefficients[0][0] != pytest.approx(breslow.coefficients[0][0])


def test_coxph_accepts_r_style_ties_alias():
    data = _tied_cox_data()
    by_ties = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="breslow",
        max_iter=15,
        eps=1e-5,
    )
    by_method = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
    )
    partial_ties = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="br",
        max_iter=15,
        eps=1e-5,
    )
    matching_aliases = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="br",
        ties="breslow",
        max_iter=15,
        eps=1e-5,
    )

    assert by_ties.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert by_ties.log_likelihood == pytest.approx(by_method.log_likelihood)
    assert partial_ties.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert partial_ties.log_likelihood == pytest.approx(by_method.log_likelihood)
    assert matching_aliases.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert matching_aliases.log_likelihood == pytest.approx(by_method.log_likelihood)


def test_coxph_accepts_unused_tt_argument_without_time_transform_terms():
    data = _toy_data()
    baseline = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    unused_tt = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        tt=lambda value, *_args: value,
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(unused_tt.coefficients, baseline.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert unused_tt.log_likelihood == pytest.approx(baseline.log_likelihood)


def test_coxph_right_censored_tt_transform_matches_r():
    data = {
        "time": [5, 1, 9, 3, 12, 7, 2, 10, 4, 11, 6, 8],
        "status": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "x1": [-0.4, 0.2, 1.1, -0.8, 0.5, 1.4, -1.2, 0.7, 0.0, -0.3, 0.9, -0.6],
        "x2": [1.2, -0.5, 0.3, 1.1, -0.9, 0.8, -0.2, 1.5, -1.1, 0.4, -0.7, 0.6],
    }

    def transform(values, times, riskset, weights):
        assert len(values) == len(times) == len(riskset) == 58
        assert weights is None
        return [value * math.log(time) for value, time in zip(values, times, strict=True)]

    fit = survival.coxph(
        "Surv(time, status) ~ x1 + tt(x2)",
        data=data,
        tt=transform,
        eps=1e-10,
        max_iter=50,
    )

    assert survival.coef_names(fit) == ["x1", "tt(x2)"]
    assert fit.coefficients[0] == pytest.approx(
        [-2.02447677883511, -0.0849208564716238],
        abs=1e-11,
    )
    for actual, expected in zip(
        fit.variance_matrix,
        [
            [0.972801236030821, 0.204657367018178],
            [0.204657367018178, 0.242009052405649],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected, abs=1e-11)
    assert fit.log_likelihood == pytest.approx(
        [-13.7646382275905, -9.83384786326646],
        abs=1e-10,
    )
    assert fit.n == len(data["time"])
    assert len(fit.y) == 58


def test_coxph_counting_process_tt_transform_matches_r():
    data = {
        "start": [0, 0, 1, 2, 0, 3, 1, 4, 2, 5],
        "stop": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "status": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        "x1": [-0.4, 0.2, 1.1, -0.8, 0.5, 1.4, -1.2, 0.7, 0.0, -0.3],
        "x2": [1.2, -0.5, 0.3, 1.1, -0.9, 0.8, -0.2, 1.5, -1.1, 0.4],
    }

    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1 + tt(x2)",
        data=data,
        tt=lambda values, times, _riskset, _weights: [
            value * math.sqrt(time) for value, time in zip(values, times, strict=True)
        ],
        eps=1e-10,
        max_iter=50,
    )

    assert fit.coefficients[0] == pytest.approx(
        [0.338462892860216, 0.243057037178683],
        abs=1e-11,
    )
    for actual, expected in zip(
        fit.variance_matrix,
        [
            [0.348891892139636, -0.0281799632994735],
            [-0.0281799632994735, 0.0518588099168937],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected, abs=1e-11)
    assert fit.log_likelihood == pytest.approx(
        [-8.59415423255237, -7.52770636448589],
        abs=1e-10,
    )
    assert fit.n == len(data["start"])
    assert len(fit.y) == 28


def test_coxph_default_tt_uses_obrien_risk_set_ranks():
    data = {
        "time": [5, 1, 9, 3, 12, 7, 2, 10, 4, 11, 6, 8],
        "status": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "x1": [-0.4, 0.2, 1.1, -0.8, 0.5, 1.4, -1.2, 0.7, 0.0, -0.3, 0.9, -0.6],
        "x2": [1.2, -0.5, 0.3, 1.1, -0.9, 0.8, -0.2, 1.5, -1.1, 0.4, -0.7, 0.6],
    }

    fit = survival.coxph(
        "Surv(time, status) ~ x1 + tt(x2)",
        data=data,
        eps=1e-10,
        max_iter=50,
    )

    assert fit.coefficients[0] == pytest.approx(
        [-1.92744172321465, -0.0775171744618016],
        abs=1e-11,
    )
    for actual, expected in zip(
        fit.variance_matrix,
        [
            [0.78059048651903, 0.00570342140879421],
            [0.00570342140879421, 0.0415301874370861],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected, abs=1e-11)
    assert fit.log_likelihood == pytest.approx(
        [-13.7646382275905, -9.76126051837732],
        abs=1e-11,
    )


def test_coxph_tt_validates_transform_configuration_and_results():
    data = _toy_data()
    formula = "Surv(time, status) ~ x1 + tt(x2)"

    with pytest.raises(ValueError, match="one function per"):
        survival.coxph(
            formula,
            data=data,
            tt=[lambda *args: args[0], lambda *args: args[0]],
        )
    with pytest.raises(TypeError, match="tt must be a function"):
        survival.coxph(formula, data=data, tt=1)
    with pytest.raises(ValueError, match="expanded risk-set rows"):
        survival.coxph(formula, data=data, tt=lambda *_args: [1.0])
    with pytest.raises(ValueError, match="numeric values"):
        survival.coxph(formula, data=data, tt=lambda values, *_args: ["x"] * len(values))
    with pytest.raises(ValueError, match="finite values"):
        survival.coxph(
            formula,
            data=data,
            tt=lambda values, *_args: [math.nan] * len(values),
        )
    with pytest.raises(ValueError, match="model=True"):
        survival.coxph(formula, data=data, model=True)


def test_coxph_accepts_r_style_control_mapping():
    data = _tied_cox_data()
    explicit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
        toler=1e-8,
    )
    controlled = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        control={
            "iter.max": 15,
            "eps": 1e-5,
            "toler.chol": 1e-8,
            "toler.inf": math.sqrt(1e-5),
            "outer.max": 10,
            "timefix": True,
        },
    )

    assert controlled.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert controlled.log_likelihood == pytest.approx(explicit.log_likelihood)
    assert controlled.risk_scores == pytest.approx(explicit.risk_scores)

    nondefault_ignored = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        control={
            "iter.max": 15,
            "eps": 1e-5,
            "toler.chol": 1e-8,
            "toler.inf": 0.25,
            "outer.max": 2,
        },
    )

    assert nondefault_ignored.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert nondefault_ignored.log_likelihood == pytest.approx(explicit.log_likelihood)
    assert nondefault_ignored.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_accepts_r_style_formula_storage_flags():
    data = _toy_data()
    default = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    explicit_defaults = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        x=False,
        y=True,
        model=False,
        tt=None,
        id=None,
        istate=None,
        statedata=None,
        singular_ok=True,
        nocenter=[-1, 0, 1],
        max_iter=10,
        eps=1e-5,
    )
    strict_singular = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        singular_ok=False,
        max_iter=10,
        eps=1e-5,
    )
    with_model = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + offset(offset) + strata(group)",
        data=data,
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(explicit_defaults.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(strict_singular.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert explicit_defaults.log_likelihood == pytest.approx(default.log_likelihood)
    assert strict_singular.log_likelihood == pytest.approx(default.log_likelihood)
    assert explicit_defaults.y.time == pytest.approx(data["time"])
    assert explicit_defaults.y.event == tuple(data["status"])
    assert not hasattr(explicit_defaults, "x")
    assert not hasattr(explicit_defaults, "model")

    model_frame = with_model.model
    assert model_frame["Surv(time, status)"].time == pytest.approx(data["time"])
    assert model_frame["Surv(time, status)"].event == tuple(data["status"])
    assert model_frame["time"] == pytest.approx(data["time"])
    assert model_frame["status"] == data["status"]
    assert model_frame["x1"] == pytest.approx(data["x1"])
    assert model_frame["x2"] == pytest.approx(data["x2"])
    assert model_frame["offset"] == pytest.approx(data["offset"])
    assert model_frame["group"] == data["group"]
    assert model_frame["(offset)"] == pytest.approx(data["offset"])
    assert model_frame["(strata)"] == data["group"]

    with_x = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        x=True,
        y=False,
        max_iter=10,
        eps=1e-5,
    )
    expected_x = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    for actual, expected in zip(with_x.x, expected_x, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(with_x, "y")

    dotted_singular = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
        **{"singular.ok": True},
    )
    for actual, expected in zip(dotted_singular.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert dotted_singular.log_likelihood == pytest.approx(default.log_likelihood)


def test_coxph_nocenter_controls_r_style_column_centering():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 0, 1],
        "dummy": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "x": [0.2, 1.3, 0.7, 1.8, 1.1, 2.2],
    }

    default = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        max_iter=20,
        eps=1e-9,
    )
    center_all = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        nocenter=[],
        max_iter=20,
        eps=1e-9,
    )
    scalar = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        nocenter=0,
        max_iter=20,
        eps=1e-9,
    )

    assert default.nocenter == pytest.approx([-1.0, 0.0, 1.0])
    assert center_all.nocenter == []
    assert scalar.nocenter == pytest.approx([0.0])
    assert default.coefficients[0] == pytest.approx(center_all.coefficients[0])
    assert default.log_likelihood == pytest.approx(center_all.log_likelihood)
    assert default.means[0] == pytest.approx(0.0)
    assert center_all.means[0] == pytest.approx(sum(data["dummy"]) / len(data["dummy"]))
    assert scalar.means[0] == pytest.approx(center_all.means[0])

    default_lp = survival.predict(default, reference="sample")
    center_all_lp = survival.predict(center_all, reference="sample")
    shift = center_all.means[0] * center_all.coefficients[0][0]
    assert center_all_lp == pytest.approx([value - shift for value in default_lp])
    assert survival.predict(default, reference="zero") == pytest.approx(
        survival.predict(center_all, reference="zero")
    )


def test_coxph_model_true_stores_matrix_inputs():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]

    fit = survival.coxph(response, x=rows, model=True, y=False, max_iter=10, eps=1e-5)

    assert fit.model["response"].time == pytest.approx(data["time"])
    assert fit.model["response"].event == tuple(data["status"])
    for actual, expected in zip(fit.model["x"], rows, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(fit, "y")


def test_coxph_singular_ok_false_rejects_dependent_designs():
    data = _toy_data()
    singular_data = {
        **data,
        "constant": [1.0] * len(data["time"]),
        "x_duplicate": [2.0 * value + 1.0 for value in data["x1"]],
    }
    response = survival.Surv(data["time"], data["status"])

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x_duplicate",
            data=singular_data,
            singular_ok=False,
        )

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            "Surv(time, status) ~ constant",
            data=singular_data,
            singular_ok=False,
        )

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            response,
            x=[[value, 2.0 * value + 1.0] for value in data["x1"]],
            singular_ok=False,
        )

    intercept_only = survival.coxph(
        "Surv(time, status) ~ 1",
        data=data,
        singular_ok=False,
        max_iter=0,
    )
    assert intercept_only.coefficients == [[]]


def test_coxph_singular_ok_false_uses_fitted_information_rank():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [0, 0, 1, 0, 1, 0],
        "x1": [-0.2, 0.3, 0.5, -1.0, 1.0, 0.0],
        "x2": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        singular_ok=True,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.coefficients[0][0] == pytest.approx(1.50746704439023)
    assert fit.coefficients[0][1] == 0.0
    assert math.isnan(survival.coef(fit)[1])

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            singular_ok=False,
            max_iter=50,
            eps=1e-9,
            toler=1e-9,
        )


def test_coxph_control_timefix_matches_r_near_tie_behavior():
    data = {
        "time": [1.0, 1.0 + 5e-10, 2.0, 3.0],
        "status": [1, 1, 0, 1],
        "x1": [0.0, 1.0, 0.5, 1.5],
    }
    fixed_times = [1.0, 1.0, 2.0, 3.0]
    rows = [[value] for value in data["x1"]]
    fixed_low_level = survival.regression.coxph_fit(
        fixed_times,
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    exact_low_level = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )

    default = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    explicit_true = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": True},
    )
    exact = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": False},
    )
    exact_dotted = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"time.fix": False},
    )

    assert default.coefficients[0] == pytest.approx(fixed_low_level.coefficients[0])
    assert default.log_likelihood == pytest.approx(fixed_low_level.log_likelihood)
    assert explicit_true.coefficients[0] == pytest.approx(default.coefficients[0])
    assert exact.coefficients[0] == pytest.approx(exact_low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(exact_low_level.log_likelihood)
    assert exact_dotted.coefficients[0] == pytest.approx(exact.coefficients[0])
    assert exact.coefficients[0] != pytest.approx(default.coefficients[0])

    with pytest.raises(ValueError, match=r"control\.timefix or control\.time\.fix"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=data,
            control={"timefix": False, "time.fix": True},
        )


def test_coxph_control_timefix_applies_to_counting_process_endpoints():
    start = [0.0, 0.0, 0.5, 1.0 + 5e-10, 0.0]
    stop = [1.0, 1.0 + 5e-10, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1, 0]
    rows = [[value] for value in [0.0, 1.0, 0.5, 1.5, 0.2]]
    fixed_start, fixed_stop = r_coerce._timefix_vectors(
        [float(value) for value in start],
        [float(value) for value in stop],
    )
    fixed_low_level = survival.regression.coxph_fit(
        fixed_stop,
        status,
        rows,
        entry_times=fixed_start,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    exact_low_level = survival.regression.coxph_fit(
        stop,
        status,
        rows,
        entry_times=start,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    response = survival.Surv(start, stop, status)

    default = survival.coxph(response, x=rows, max_iter=20, eps=1e-9, method="efron")
    exact = survival.coxph(
        response,
        x=rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": False},
    )

    assert default.coefficients[0] == pytest.approx(fixed_low_level.coefficients[0])
    assert default.log_likelihood == pytest.approx(fixed_low_level.log_likelihood)
    assert exact.coefficients[0] == pytest.approx(exact_low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(exact_low_level.log_likelihood)
    assert exact.coefficients[0] != pytest.approx(default.coefficients[0])


def test_coxph_exact_ties_alias_accepts_data_without_tied_events():
    data = _toy_data()
    exact = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="exact",
        max_iter=10,
        eps=1e-5,
    )
    partial_exact = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="ex",
        max_iter=10,
        eps=1e-5,
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="efron",
        max_iter=10,
        eps=1e-5,
    )

    assert exact.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert exact.log_likelihood == pytest.approx(efron.log_likelihood)
    assert exact.risk_scores == pytest.approx(efron.risk_scores)
    assert partial_exact.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert partial_exact.log_likelihood == pytest.approx(efron.log_likelihood)
    assert partial_exact.risk_scores == pytest.approx(efron.risk_scores)


def test_coxph_exact_ties_handles_tied_events():
    data = _tied_cox_data()
    exact = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="exact",
        max_iter=15,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="exact",
        max_iter=15,
        eps=1e-5,
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert exact.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert exact.risk_scores == pytest.approx(low_level.risk_scores)
    assert exact.coefficients[0][0] != pytest.approx(efron.coefficients[0][0])


def test_coxph_exact_rejects_case_weights_across_interfaces():
    data = _tied_cox_data()
    rows = [[value] for value in data["x1"]]
    weights = [1.0, 2.0, *([1.0] * (len(data["time"]) - 2))]
    error = "Case weights are not supported for the exact method"

    with pytest.raises(ValueError, match=error):
        survival.regression.coxph_fit(
            time=data["time"],
            status=data["status"],
            covariates=rows,
            weights=weights,
            method="exact",
        )
    with pytest.raises(ValueError, match=error):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=data,
            weights=weights,
            ties="exact",
        )
    with pytest.raises(ValueError, match=error):
        survival.coxph(
            survival.Surv(data["time"], data["status"]),
            x=rows,
            weights=weights,
            ties="exact",
        )

    omitted = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        method="exact",
        max_iter=0,
    )
    explicit_units = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        weights=[1.0] * len(data["time"]),
        method="exact",
        max_iter=0,
    )
    assert explicit_units.log_likelihood == pytest.approx(omitted.log_likelihood)
    assert explicit_units.score_vector == pytest.approx(omitted.score_vector)


def test_low_level_coxph_tie_methods_match_hand_likelihood_at_initial_beta():
    data = _tied_cox_data()
    for method in ("efron", "breslow", "exact"):
        fit = survival.regression.coxph_fit(
            time=data["time"],
            status=data["status"],
            covariates=[[value] for value in data["x1"]],
            initial_beta=[0.0],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik_at_zero(data["time"], data["status"], method)

        assert fit.log_likelihood == pytest.approx([expected, expected])


@pytest.mark.parametrize(
    ("method", "expected_beta", "expected_variance", "expected_loglik", "expected_iterations"),
    [
        (
            "breslow",
            [0.17569299458865062, -0.8920800877103963],
            [
                [1.7665625849926303, 0.7884637138499169],
                [0.7884637138499169, 2.3311009321070157],
            ],
            [-8.070906088787817, -7.818812517162524],
            3,
        ),
        (
            "efron",
            [0.10305623522446912, -1.021973929290916],
            [
                [1.8044465510612007, 0.9722159657351814],
                [0.9722159657351814, 2.6559563822056127],
            ],
            [-7.7142311448490855, -7.430873243936032],
            4,
        ),
        (
            "exact",
            [0.18990918067302853, -1.1018604705682347],
            [
                [2.1702188669318634, 1.0117674000364538],
                [1.0117674000364538, 2.9114455509256167],
            ],
            [-6.327936783729195, -6.021664785573822],
            3,
        ),
    ],
)
def test_low_level_coxph_rank_aware_information_matches_reference(
    method,
    expected_beta,
    expected_variance,
    expected_loglik,
    expected_iterations,
):
    time = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    status = [1, 1, 0, 1, 1, 0, 1, 0]
    x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5]
    x2 = [1.0, 0.2, 0.7, 1.3, 0.4, 1.1, 0.5, 0.9]
    fit = survival.regression.coxph_fit(
        time,
        status,
        [[left, right] for left, right in zip(x1, x2, strict=True)],
        method=method,
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.coefficients[0] == pytest.approx(expected_beta, rel=0, abs=1e-12)
    for actual, expected in zip(fit.information_matrix, expected_variance, strict=True):
        assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    assert fit.log_likelihood == pytest.approx(expected_loglik, rel=0, abs=1e-12)
    assert fit.convergence_flag == 2
    assert fit.iterations == expected_iterations


def test_low_level_coxph_defaults_match_efron_reference():
    time = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    status = [1, 1, 0, 1, 1, 0, 1, 0]
    x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5]
    x2 = [1.0, 0.2, 0.7, 1.3, 0.4, 1.1, 0.5, 0.9]
    fit = survival.regression.coxph_fit(
        time,
        status,
        [[left, right] for left, right in zip(x1, x2, strict=True)],
    )

    assert fit.method == "efron"
    assert fit.coefficients[0] == pytest.approx(
        [0.10305623522446912, -1.021973929290916], rel=0, abs=1e-12
    )
    assert fit.log_likelihood == pytest.approx(
        [-7.7142311448490855, -7.430873243936032], rel=0, abs=1e-12
    )
    assert fit.convergence_flag == 2
    assert fit.iterations == 4


def test_low_level_coxph_default_rank_tolerance_preserves_near_independent_columns():
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 1, 0, 1, 1, 0, 1, 0]
    x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5]
    z = [0.7, -0.4, 1.1, -0.8, 0.2, 0.9, -0.5, 0.3]
    x2 = [left + 5e-6 * perturbation for left, perturbation in zip(x1, z, strict=True)]
    covariates = [[left, right] for left, right in zip(x1, x2, strict=True)]

    default_fit = survival.regression.coxph_fit(time, status, covariates, max_iter=0)
    rank_reduced_fit = survival.regression.coxph_fit(
        time,
        status,
        covariates,
        max_iter=0,
        toler=1e-10,
    )

    assert default_fit.convergence_flag == 2
    assert rank_reduced_fit.convergence_flag == 1
    assert rank_reduced_fit.information_matrix[0][1] == 0.0
    assert rank_reduced_fit.information_matrix[1] == [0.0, 0.0]


def test_low_level_coxph_reduces_rank_for_dependent_columns():
    time = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0]
    status = [1, 1, 0, 1, 1, 0, 1, 0]
    x1 = [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5]
    fit = survival.regression.coxph_fit(
        time,
        status,
        [[value, 2.0 * value] for value in x1],
        method="efron",
        max_iter=50,
        eps=1e-9,
        toler=1e-9,
    )

    assert fit.coefficients[0][0] == pytest.approx(0.43940480983777153, rel=0, abs=1e-12)
    assert fit.coefficients[0][1] == 0.0
    assert fit.information_matrix[0][0] == pytest.approx(1.3555601527463446, rel=0, abs=1e-12)
    assert fit.information_matrix[0][1] == 0.0
    assert fit.information_matrix[1] == [0.0, 0.0]
    assert fit.log_likelihood[1] == pytest.approx(-7.643272995750461, rel=0, abs=1e-12)
    assert fit.convergence_flag == 1
    assert fit.iterations == 3
    assert all(math.isfinite(value) for value in fit.predict([[0.25, 0.5], [1.0, 2.0]]))


def test_low_level_coxph_accepts_zero_column_null_model():
    data = _toy_data()
    fit = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        max_iter=0,
    )
    expected = _manual_cox_loglik_at_zero(data["time"], data["status"], "efron")

    assert fit.coefficients == [[]]
    assert fit.means == []
    assert fit.score_vector == []
    assert fit.information_matrix == []
    assert fit.linear_predictors == pytest.approx([0.0] * len(data["time"]))
    assert fit.risk_scores == pytest.approx([1.0] * len(data["time"]))
    assert fit.log_likelihood == pytest.approx([expected, expected])
    assert fit.predict([[], []]) == pytest.approx([0.0, 0.0])
    times, curves = fit.survival_curve([[]])
    assert times
    assert len(curves) == 1


def test_low_level_coxph_nocenter_disables_column_scaling():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 0, 1],
        "dummy": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "x": [0.2, 1.3, 0.7, 1.8, 1.1, 2.2],
    }
    rows = [[dummy, x] for dummy, x in zip(data["dummy"], data["x"], strict=True)]

    centered = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
    )
    default = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        nocenter=[-1.0, 0.0, 1.0],
    )
    zero_only = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        nocenter=[0.0],
    )

    assert centered.nocenter == []
    assert default.nocenter == pytest.approx([-1.0, 0.0, 1.0])
    assert centered.coefficients[0] == pytest.approx(default.coefficients[0])
    assert centered.log_likelihood == pytest.approx(default.log_likelihood)
    assert centered.means[0] == pytest.approx(sum(data["dummy"]) / len(data["dummy"]))
    assert default.means[0] == pytest.approx(0.0)
    assert default.means[1] != pytest.approx(0.0)
    assert zero_only.means[0] == pytest.approx(centered.means[0])


def test_low_level_coxph_rejects_invalid_numeric_inputs():
    data = _toy_data()
    kwargs = {
        "time": data["time"],
        "status": data["status"],
        "covariates": [[value] for value in data["x1"]],
        "initial_beta": [0.0],
        "max_iter": 0,
    }

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "time": [1.0, float("nan"), *data["time"][2:]]})

    bad_status = [*data["status"]]
    bad_status[0] = 2
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.regression.coxph_fit(**{**kwargs, "status": bad_status})

    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        survival.regression.coxph_fit(
            **{**kwargs, "covariates": [[float("inf")], *kwargs["covariates"][1:]]}
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.regression.coxph_fit(**{**kwargs, "weights": [1.0, -1.0, *([1.0] * 6)]})

    with pytest.raises(ValueError, match="at least one positive"):
        survival.regression.coxph_fit(**{**kwargs, "weights": [0.0] * len(data["time"])})

    with pytest.raises(ValueError, match="offset contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "offset": [0.0, float("inf"), *([0.0] * 6)]})

    with pytest.raises(ValueError, match="entry_times contains non-finite"):
        survival.regression.coxph_fit(
            **{**kwargs, "entry_times": [0.0, float("nan"), *([0.0] * 6)]}
        )

    with pytest.raises(ValueError, match="initial_beta contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "initial_beta": [float("nan")]})

    with pytest.raises(ValueError, match="nocenter contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "nocenter": [0.0, float("nan")]})

    with pytest.raises(ValueError, match="eps must be a finite positive value"):
        survival.regression.coxph_fit(**{**kwargs, "eps": 0.0})

    with pytest.raises(ValueError, match="toler must be a finite positive value"):
        survival.regression.coxph_fit(**{**kwargs, "toler": 0.0})

    fit = survival.regression.coxph_fit(**kwargs)
    with pytest.raises(ValueError, match="covariates row contains non-finite"):
        fit.predict([[float("nan")]])
    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        fit.survival_curve([[float("inf")]])
    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        fit.survival_curve_with_strata([[float("nan")]], [0])


def test_low_level_coxph_counting_process_uses_entry_times():
    data = _counting_cox_data()
    for method in ("efron", "breslow"):
        fit = survival.regression.coxph_fit(
            time=data["stop"],
            status=data["status"],
            covariates=[[value] for value in data["x1"]],
            entry_times=data["start"],
            initial_beta=[0.0],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik_at_zero(
            data["stop"],
            data["status"],
            method,
            entry_times=data["start"],
        )

        assert fit.entry_times == pytest.approx(data["start"])
        assert fit.log_likelihood == pytest.approx([expected, expected])


def test_low_level_coxph_counting_process_matches_weighted_hand_likelihood():
    data = _counting_cox_data()
    data["strata"] = ["A", "A", "A", "B", "B", "B"]
    weights = [1.5, 0.75, 2.0, 1.0, 1.25, 0.5]
    offset = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2]
    covariates = [[value] for value in data["x1"]]
    strata = [0 if value == "A" else 1 for value in data["strata"]]
    for method in ("efron", "breslow"):
        fit = survival.regression.coxph_fit(
            time=data["stop"],
            status=data["status"],
            covariates=covariates,
            strata=strata,
            weights=weights,
            offset=offset,
            entry_times=data["start"],
            initial_beta=[0.35],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik(
            data["stop"],
            data["status"],
            covariates,
            [0.35],
            method,
            entry_times=data["start"],
            weights=weights,
            offset=offset,
            strata=strata,
        )

        assert fit.log_likelihood == pytest.approx([expected, expected])


def test_low_level_coxph_zero_entry_matches_right_censored_fit():
    data = _tied_cox_data()
    right = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    counting = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        entry_times=[0.0] * len(data["time"]),
        max_iter=15,
        eps=1e-5,
    )

    assert counting.coefficients[0] == pytest.approx(right.coefficients[0])
    assert counting.log_likelihood == pytest.approx(right.log_likelihood)


def test_low_level_coxph_sorts_counting_process_inputs_with_strata():
    data = _counting_cox_data()
    data["strata"] = ["A", "A", "A", "B", "B", "B"]
    sorted_indices = sorted(
        range(len(data["stop"])),
        key=lambda idx: (data["strata"][idx], data["stop"][idx]),
    )
    shuffled_indices = [3, 0, 5, 1, 4, 2]
    sorted_data = _take(data, sorted_indices)
    shuffled_data = _take(data, shuffled_indices)
    sorted_fit = survival.regression.coxph_fit(
        time=sorted_data["stop"],
        status=sorted_data["status"],
        covariates=[[value] for value in sorted_data["x1"]],
        strata=[0 if value == "A" else 1 for value in sorted_data["strata"]],
        entry_times=sorted_data["start"],
        method="breslow",
        initial_beta=[0.0],
        max_iter=0,
    )
    shuffled_fit = survival.regression.coxph_fit(
        time=shuffled_data["stop"],
        status=shuffled_data["status"],
        covariates=[[value] for value in shuffled_data["x1"]],
        strata=[0 if value == "A" else 1 for value in shuffled_data["strata"]],
        entry_times=shuffled_data["start"],
        method="breslow",
        initial_beta=[0.0],
        max_iter=0,
    )

    assert shuffled_fit.log_likelihood == pytest.approx(sorted_fit.log_likelihood)
    assert shuffled_fit.score_vector == pytest.approx(sorted_fit.score_vector)


def test_low_level_coxph_sorts_rows_with_strata_before_fitting():
    data = _tied_cox_data()
    sorted_indices = sorted(
        range(len(data["time"])),
        key=lambda idx: (data["strata"][idx], data["time"][idx]),
    )
    shuffled_indices = [3, 0, 7, 1, 5, 2, 6, 4]
    sorted_data = _take(data, sorted_indices)
    shuffled_data = _take(data, shuffled_indices)
    sorted_fit = survival.regression.coxph_fit(
        time=sorted_data["time"],
        status=sorted_data["status"],
        covariates=[[value] for value in sorted_data["x1"]],
        strata=[0 if value == "A" else 1 for value in sorted_data["strata"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    shuffled_fit = survival.regression.coxph_fit(
        time=shuffled_data["time"],
        status=shuffled_data["status"],
        covariates=[[value] for value in shuffled_data["x1"]],
        strata=[0 if value == "A" else 1 for value in shuffled_data["strata"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert shuffled_fit.coefficients[0] == pytest.approx(sorted_fit.coefficients[0])
    assert shuffled_fit.log_likelihood == pytest.approx(sorted_fit.log_likelihood)


def test_coxph_formula_dot_expands_remaining_covariates():
    data = _numeric_data()
    fit = survival.coxph("Surv(time, status) ~ .", data=data, max_iter=10, eps=1e-5)
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert fit.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_formula_dot_can_exclude_identifier_columns():
    data = _numeric_data_with_id()
    fit = survival.coxph("Surv(time, status) ~ . - id", data=data, max_iter=10, eps=1e-5)
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert fit.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_formula_accepts_backticks_for_covariates_and_offsets():
    data = _backtick_data()
    fit = survival.coxph(
        "Surv(`follow-up`, `event status`) ~ `age-years` + `marker/value` + offset(`log exposure`)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["follow-up"],
        status=data["event status"],
        covariates=[
            [data["age-years"][idx], data["marker/value"][idx]]
            for idx in range(len(data["follow-up"]))
        ],
        offset=data["log exposure"],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_applies_subset_before_design_matrix():
    data = _numeric_data()
    indices = [0, 1, 3, 5, 6]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        subset=[idx in indices for idx in range(len(data["time"]))],
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_take(data, indices),
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_na_action_omit_matches_filtered_data():
    data = _numeric_data()
    data = {**data, "x1": [0.2, 0.4, float("nan"), 0.8, 1.0, 1.2, 0.6, 1.4]}
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        na_action="omit",
        max_iter=10,
        eps=1e-5,
    )
    dotted = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
        **{"na.action": "omit"},
    )
    direct = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_take(data, indices),
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)
    assert dotted.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert dotted.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_na_action_fail_and_omit_detect_primitive_sentinels():
    data = _numeric_data()
    data = {
        **data,
        "x1": [0.2, 0.4, None, 0.8, 1.0, 1.2, 0.6, 1.4],
        "x2": [1.0, 0.9, 1.1, 0.7, 0.4, 0.3, float("nan"), 0.2],
    }
    retained = [0, 1, 3, 4, 5, 7]

    with pytest.raises(ValueError, match="missing values in formula data"):
        survival.coxph("Surv(time, status) ~ x1 + x2", data=data)

    omitted = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        na_action="omit",
        max_iter=10,
        eps=1e-5,
    )
    filtered = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_take(data, retained),
        max_iter=10,
        eps=1e-5,
    )

    assert omitted.coefficients[0] == pytest.approx(filtered.coefficients[0])
    assert omitted.risk_scores == pytest.approx(filtered.risk_scores)


def test_coxph_formula_filters_external_weights_and_offset_with_subset_and_na_action():
    data = _numeric_data()
    indices = [0, 2, 3, 4, 5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        offset=[0.1, 0.0, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1],
        subset=[0, 1, 2, 3, 4, 5],
        na_action="omit",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.regression.coxph_fit(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in indices],
        weights=[1.0, 1.0, 0.8, 1.2, 1.1],
        offset=[0.1, 0.0, -0.1, 0.2, 0.0],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_passes_strata_to_rust_optimizer():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        strata=[0, 0, 0, 0, 1, 1, 1, 1],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert len(fit.risk_scores) == 8
    assert len(fit.predict([[0.5]])) == 1


def test_predict_coxph_reference_strata_uses_training_stratum_means():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    beta = fit.coefficients[0][0]
    strata_means = {"A": 0.375, "B": 1.05}
    sample_mean = sum(data["x1"]) / len(data["x1"])

    default_lp = survival.predict(fit)
    sample_lp = survival.predict(fit, reference="sample")
    sample_prefix_lp = survival.predict(fit, reference="sa")
    zero_lp = survival.predict(fit, reference="zero")
    zero_prefix_lp = survival.predict(fit, reference="z")
    strata_prefix_lp = survival.predict(fit, reference="st")

    assert zero_lp == pytest.approx(fit.linear_predictors)
    assert zero_prefix_lp == pytest.approx(zero_lp)
    assert sample_lp == pytest.approx([beta * (value - sample_mean) for value in data["x1"]])
    assert sample_prefix_lp == pytest.approx(sample_lp)
    assert default_lp == pytest.approx(
        [
            beta * (value - strata_means[group])
            for value, group in zip(data["x1"], data["group"], strict=True)
        ]
    )
    assert strata_prefix_lp == pytest.approx(default_lp)


def test_predict_coxph_reference_strata_uses_formula_newdata_strata():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    beta = fit.coefficients[0][0]
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}

    assert survival.predict(fit, newdata) == pytest.approx(
        [beta * (0.5 - 0.375), beta * (0.5 - 1.05)]
    )
    assert survival.predict(fit, newdata, reference="sample") == pytest.approx(
        [beta * (0.5 - sum(data["x1"]) / len(data["x1"]))] * 2
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx([beta * 0.5] * 2)
    with_se = survival.predict(fit, newdata, se_fit=True)
    var = fit.information_matrix[0][0]
    assert with_se.fit == pytest.approx([beta * (0.5 - 0.375), beta * (0.5 - 1.05)])
    assert with_se.se_fit == pytest.approx(
        [abs(0.5 - 0.375) * math.sqrt(var), abs(0.5 - 1.05) * math.sqrt(var)]
    )
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(fit, {"x1": [0.5], "group": ["C"]})
    with pytest.raises(ValueError, match="newdata strata are required"):
        survival.predict(fit, [[0.5]])


def test_predict_coxph_survival_uses_formula_newdata_strata():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}
    rows = [[0.5], [0.5]]
    strata = [0, 1]
    full_times, full_curves = fit.survival_curve_with_strata(rows, strata, True)
    requested_times = [0.5, *full_times[:2], full_times[-1]]

    times, curves = survival.predict(
        fit,
        newdata,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    for actual, expected_curve in zip(curves, full_curves, strict=True):
        expected = []
        for time in requested_times:
            pos = bisect_right(full_times, time)
            expected.append(1.0 if pos == 0 else expected_curve[pos - 1])
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(fit, {"x1": [0.5], "group": ["C"]}, type="survival")


def test_predict_coxph_expected_accepts_formula_newdata_response():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 4.5, 8.0],
        "status": [0, 0, 1],
        "x1": [0.2, 0.8, 1.4],
        "x2": [1.0, 0.7, 0.2],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = [
        0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1]
        for time in newdata["time"]
    ]
    collapse = ["A", "A", "B"]

    actual = survival.predict(fit, newdata, type="expected")

    assert actual == pytest.approx(expected)
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    assert survival.predict(fit, newdata, type="expected", collapse=collapse) == pytest.approx(
        [expected[0] + expected[1], expected[2]]
    )
    assert survival.predict(fit, newdata, type="survival", collapse=collapse) == pytest.approx(
        [math.exp(-expected[0]) + math.exp(-expected[1]), math.exp(-expected[2])]
    )
    newdata_with_different_status = {**newdata, "status": [1, 1, 0]}
    assert survival.predict(fit, newdata_with_different_status, type="expected") == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="same length as predictions"):
        survival.predict(fit, newdata, type="expected", collapse=["A"])
    expected_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    survival_with_se = survival.predict(fit, newdata, type="survival", se_fit=True)
    assert expected_with_se.fit == pytest.approx(expected)
    assert len(expected_with_se.se_fit) == len(expected)
    assert all(math.isfinite(value) and value >= 0.0 for value in expected_with_se.se_fit)
    assert survival_with_se.fit == pytest.approx([math.exp(-value) for value in expected])
    assert survival_with_se.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(expected_with_se.se_fit, expected, strict=True)]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(fit, {"x1": [0.2], "x2": [1.0]}, type="expected")


def test_predict_coxph_expected_accepts_formula_response_comparison_newdata():
    data = _toy_data()
    data["r_status"] = [2 if status == 1 else 1 for status in data["status"]]
    fit = survival.coxph(
        "Surv(time, r_status == 2) ~ x1 + x2",
        data=data,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 4.5, 8.0],
        "r_status": [1, 1, 2],
        "x1": [0.2, 0.8, 1.4],
        "x2": [1.0, 0.7, 0.2],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = [
        0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1]
        for time in newdata["time"]
    ]

    assert survival.predict(fit, newdata, type="expected") == pytest.approx(expected)
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(fit, {"time": [1.0], "x1": [0.2], "x2": [1.0]}, type="expected")


def test_predict_coxph_expected_and_survival_se_use_baseline_variance():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ 1",
        data=data,
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 2.5, 4.0],
        "status": [0, 0, 1],
    }
    expected = [0.0, 1.0 / 4.0 + 1.0 / 3.0, 1.0 / 4.0 + 1.0 / 3.0 + 1.0]
    expected_var = [0.0, 1.0 / 16.0 + 1.0 / 9.0, 1.0 / 16.0 + 1.0 / 9.0 + 1.0]
    expected_se = [math.sqrt(value) for value in expected_var]

    expected_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    survival_with_se = survival.predict(fit, newdata, type="survival", se_fit=True)
    collapsed = survival.predict(
        fit,
        newdata,
        type="expected",
        collapse=["A", "A", "B"],
        se_fit=True,
    )
    training = survival.predict(fit, type="expected", se_fit=True)
    training_survival = survival.predict(fit, type="survival", se_fit=True)

    assert expected_with_se.fit == pytest.approx(expected)
    assert expected_with_se.se_fit == pytest.approx(expected_se)
    assert survival_with_se.fit == pytest.approx([math.exp(-value) for value in expected])
    assert survival_with_se.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(expected_se, expected, strict=True)]
    )
    assert collapsed.fit == pytest.approx([expected[0] + expected[1], expected[2]])
    assert collapsed.se_fit == pytest.approx(
        [math.sqrt(expected_se[0] ** 2 + expected_se[1] ** 2), expected_se[2]]
    )
    assert training.fit == pytest.approx(
        [expected[1] - 1.0 / 3.0, expected[1], expected[1], expected[2]]
    )
    assert training.se_fit == pytest.approx(
        [
            math.sqrt(expected_var[1] - 1.0 / 9.0),
            expected_se[1],
            expected_se[1],
            expected_se[2],
        ]
    )
    assert training_survival.fit == pytest.approx([math.exp(-value) for value in training.fit])
    assert training_survival.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(training.se_fit, training.fit, strict=True)]
    )


def test_predict_coxph_expected_accepts_counting_newdata_response():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "start": [0.0, 2.5],
        "stop": [4.0, 6.0],
        "status": [0, 1],
        "x1": [0.2, 1.1],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = []
    for start, stop in zip(newdata["start"], newdata["stop"], strict=True):
        start_pos = bisect_right(base_times, start)
        stop_pos = bisect_right(base_times, stop)
        start_hazard = 0.0 if start_pos == 0 else base_hazards[start_pos - 1]
        stop_hazard = 0.0 if stop_pos == 0 else base_hazards[stop_pos - 1]
        expected.append(stop_hazard - start_hazard)

    actual = survival.predict(fit, newdata, type="expected")
    actual_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    baseline = r_coxph._cox_expected_baseline_by_stratum(fit)[0]
    variance = fit.information_matrix
    means = fit.means
    linear_predictors = survival.predict(
        fit,
        [[value] for value in newdata["x1"]],
        reference="zero",
    )
    expected_se = []
    for start, stop, x1, linear_predictor in zip(
        newdata["start"],
        newdata["stop"],
        newdata["x1"],
        linear_predictors,
        strict=True,
    ):
        start_hazard, start_varhaz, start_xbar = r_coxph._cox_expected_baseline_at(
            baseline,
            start,
            1,
        )
        stop_hazard, stop_varhaz, stop_xbar = r_coxph._cox_expected_baseline_at(
            baseline,
            stop,
            1,
        )
        centered_x = x1 - means[0]
        start_delta = start_hazard * centered_x - start_xbar[0]
        stop_delta = stop_hazard * centered_x - stop_xbar[0]
        interval_delta = stop_delta - start_delta
        expected_var = stop_varhaz - start_varhaz + interval_delta * variance[0][0] * interval_delta
        expected_se.append(math.sqrt(max(expected_var, 0.0)) * math.exp(linear_predictor))
    conditioned = survival.survfit(
        fit,
        newdata={"x1": [newdata["x1"][1]]},
        start_time=newdata["start"][1],
        conf_type="none",
    )

    assert actual == pytest.approx(expected)
    assert actual_with_se.fit == pytest.approx(expected)
    assert actual_with_se.se_fit == pytest.approx(expected_se)
    assert conditioned.std_chaz[0][-1] == pytest.approx(expected_se[1])
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(
            fit,
            {"stop": [4.0], "status": [1], "x1": [0.2]},
            type="expected",
        )


def test_predict_coxph_expected_uses_formula_newdata_strata():
    data = {
        "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["A", "A", "A", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 1.0, 1.2, 0.8],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [2.5, 4.0],
        "status": [0, 1],
        "group": ["A", "B"],
        "x1": [0.5, 0.5],
    }

    actual = survival.predict(fit, newdata, type="expected")

    assert actual == pytest.approx([5.0 / 6.0, 1.5])
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-5.0 / 6.0), math.exp(-1.5)]
    )
    with pytest.raises(ValueError, match="partial formula response"):
        survival.predict(
            fit,
            {"time": [2.0], "group": ["A"], "x1": [0.5]},
            type="survival",
        )
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(
            fit,
            {"time": [2.0], "status": [1], "group": ["C"], "x1": [0.5]},
            type="expected",
        )


def test_basehaz_accepts_fitted_coxph_model_and_raw_inputs():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    r_status = [2 if event else 1 for event in data["status"]]
    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    method_times, method_hazard = fit.basehaz(False)
    raw_times, raw_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
    )
    raw_r_times, raw_r_hazard = survival.basehaz(
        data["time"],
        r_status,
        fit.linear_predictors,
        False,
    )
    keyword_times, keyword_hazard = survival.basehaz(
        time=data["time"],
        status=data["status"],
        linear_predictors=fit.linear_predictors,
        centered=False,
    )
    keyword_r_times, keyword_r_hazard = survival.basehaz(
        time=data["time"],
        status=r_status,
        linear_predictors=fit.linear_predictors,
        centered=False,
    )

    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(method_times, time)) == 0 else method_hazard[pos - 1]
        for time in data["time"]
    ]
    assert fitted_times == pytest.approx(data["time"])
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)
    assert raw_times == pytest.approx(method_times)
    assert raw_hazard == pytest.approx(method_hazard)
    assert keyword_times == pytest.approx(method_times)
    assert keyword_hazard == pytest.approx(method_hazard)
    assert raw_r_times == pytest.approx(method_times)
    assert raw_r_hazard == pytest.approx(method_hazard)
    assert keyword_r_times == pytest.approx(method_times)
    assert keyword_r_hazard == pytest.approx(method_hazard)


def test_fitted_basehaz_uses_scaled_risk_scores_for_large_linear_predictors():
    fit = survival.regression.coxph_fit(
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        covariates=[[1.0], [709.0 / 710.0], [708.0 / 710.0]],
        initial_beta=[710.0],
        max_iter=0,
        method="breslow",
    )

    times, hazard = fit.basehaz(False)
    expected_first = math.exp(-710.0) / (1.0 + math.exp(-1.0) + math.exp(-2.0))

    assert fit.linear_predictors == pytest.approx([710.0, 709.0, 708.0])
    assert times == pytest.approx([1.0, 2.0, 3.0])
    assert hazard[0] == pytest.approx(expected_first, rel=1e-12, abs=0.0)
    assert 0.0 < hazard[0] < hazard[1] < hazard[2]


def test_basehaz_accepts_fitted_coxph_newdata():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    newdata = {"x1": [0.5, 1.0], "x2": [0.8, 0.2]}

    result = survival.basehaz(fit, newdata=newdata, centered=False)
    positional = survival.basehaz(fit, newdata, centered=False)
    centered = survival.basehaz(fit, newdata=newdata, centered=True)
    survfit = survival.survfit(fit, newdata=newdata)
    unpacked_times, unpacked_hazards = result
    single = survival.basehaz(fit, newdata={"x1": [0.5], "x2": [0.8]}, centered=False)
    base_times, base_hazards = fit.basehaz(False)
    linear_predictors = survival.predict(fit, newdata, reference="zero")
    expected_uncentered = [
        [
            (0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1])
            * math.exp(linear_predictor)
            for time in result.time
        ]
        for linear_predictor in linear_predictors
    ]

    assert isinstance(result, survival.r_api.CoxBaseHazardResult)
    assert result.centered is True
    assert positional.centered is True
    assert centered.centered is True
    assert result.time == pytest.approx(survfit.time)
    assert positional.time == pytest.approx(result.time)
    assert centered.time == pytest.approx(survfit.time)
    assert unpacked_times == pytest.approx(result.time)
    assert result.hazard == result.cumhaz
    assert result.cumulative_hazard == result.cumhaz
    assert len(result.cumhaz) == 2
    for actual, positional_curve, expected in zip(
        result.cumhaz,
        positional.cumhaz,
        expected_uncentered,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
        assert positional_curve == pytest.approx(expected)
    for actual, expected in zip(centered.cumhaz, survfit.cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    assert result.cumhaz[0] == pytest.approx(centered.cumhaz[0])
    assert unpacked_hazards == result.cumhaz
    assert single.centered is True
    assert single.time == pytest.approx(result.time)
    assert single.cumhaz == pytest.approx(expected_uncentered[0])
    with pytest.raises(ValueError, match="positional newdata"):
        survival.basehaz(fit, newdata, newdata=newdata)


def test_basehaz_uses_fitted_coxph_weights():
    data = _toy_data()
    weights = [2.0, 1.0, 0.5, 1.5, 1.0, 2.0, 0.75, 1.25]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    raw_times, raw_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
        weights=weights,
    )
    unweighted_times, unweighted_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
    )

    assert fit.weights == pytest.approx(weights)
    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(raw_times, time)) == 0 else raw_hazard[pos - 1]
        for time in data["time"]
    ]
    assert fitted_times == pytest.approx(data["time"])
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)
    assert unweighted_times == pytest.approx(raw_times)
    assert fitted_hazard[0] == pytest.approx(weights[0] / sum(weights))
    assert raw_hazard != pytest.approx(unweighted_hazard)


def test_basehaz_uses_counting_process_weights():
    data = _counting_cox_data()
    weights = [2.0, 1.0, 3.0, 1.0, 4.0, 1.0]
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        weights=weights,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    fitted_times, fitted_hazard = fit.basehaz(False)
    raw_times, raw_hazard = survival.basehaz(
        data["stop"],
        data["status"],
        fit.linear_predictors,
        False,
        entry_times=data["start"],
        weights=weights,
    )

    assert fit.weights == pytest.approx(weights)
    assert fitted_times == pytest.approx(raw_times)
    assert fitted_hazard == pytest.approx(raw_hazard)
    assert fitted_hazard[0] == pytest.approx((weights[0] + weights[1]) / 10.0)


def test_basehaz_uses_fitted_coxph_strata():
    data = {
        "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["A", "A", "A", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 1.0, 1.2, 0.8],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )

    times, hazards, strata = fit.basehaz_with_strata(False)
    result = survival.basehaz(fit, centered=False)
    unpacked_times, unpacked_hazards = result
    expected_event_times = [1.0, 2.0, 3.0, 4.0]
    expected_event_hazards = [1.0 / 3.0, 1.0 / 3.0 + 1.0 / 2.0, 1.0 / 2.0, 1.5]
    expected_times = [1.0, 2.0, 4.0, 1.0, 3.0, 4.0]
    expected_hazards = [1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0, 0.0, 0.5, 1.5]

    assert times == pytest.approx(expected_event_times)
    assert hazards == pytest.approx(expected_event_hazards)
    assert strata == [0, 0, 1, 1]
    assert unpacked_times == pytest.approx(expected_times)
    assert unpacked_hazards == pytest.approx(expected_hazards)
    assert result.time == pytest.approx(expected_times)
    assert result.cumhaz == pytest.approx(expected_hazards)
    assert result.hazard == pytest.approx(expected_hazards)
    assert result.cumulative_hazard == pytest.approx(expected_hazards)
    assert result.strata == [0, 0, 0, 1, 1, 1]
    assert result.strata_labels == data["group"]
    frame = survival.as_data_frame(result)
    assert frame["strata"] == data["group"]

    expected = survival.predict(fit, type="expected")
    assert expected == pytest.approx([1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0, 0.0, 0.5, 1.5])


def test_basehaz_newdata_uses_fitted_coxph_strata():
    data = {
        "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["A", "A", "A", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 1.0, 1.2, 0.8],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}

    result = survival.basehaz(fit, newdata=newdata, centered=False)
    survfit = survival.survfit(fit, newdata=newdata)
    single = survival.basehaz(fit, newdata={"x1": [0.5], "group": ["B"]})

    assert result.centered is True
    assert result.curve_strata == [0, 1]
    assert result.curve_strata_labels == ["A", "B"]
    assert result.time == pytest.approx(survfit.time)
    for actual, expected in zip(result.cumhaz, survfit.cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    result_frame = survival.as_data_frame(result)
    assert set(result_frame["strata"]) == {"A", "B"}
    assert single.time == pytest.approx([1.0, 3.0, 4.0])
    assert single.cumhaz == pytest.approx([0.0, 0.5, 1.5])
    assert single.strata == [1, 1, 1]
    assert single.curve_strata == [1]
    assert single.strata_labels == ["B", "B", "B"]
    assert single.curve_strata_labels == ["B"]
    assert survival.as_data_frame(single)["strata"] == ["B", "B", "B"]


def test_basehaz_uses_efron_tie_increments_for_fitted_coxph():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0],
        "status": [1, 1, 0, 1],
        "x1": [0.2, 0.8, 0.4, 1.1],
    }
    breslow = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="efron",
    )

    breslow_times, breslow_hazard = breslow.basehaz(False)
    efron_times, efron_hazard = efron.basehaz(False)
    detail = survival.coxph_detail(efron)

    assert breslow_times == pytest.approx([1.0, 3.0])
    assert efron_times == pytest.approx(breslow_times)
    assert breslow_hazard == pytest.approx([2.0 / 4.0, 2.0 / 4.0 + 1.0 / 1.0])
    assert efron_hazard == pytest.approx([1.0 / 4.0 + 1.0 / 3.0, 1.0 / 4.0 + 1.0 / 3.0 + 1.0])
    assert efron_hazard[0] > breslow_hazard[0]
    assert efron_hazard == pytest.approx(detail.cumulative_hazard)
    surv = survival.survfit(efron)
    event_only = survival.survfit(efron, censor=False)
    assert surv.time == pytest.approx([1.0, 2.0, 3.0])
    assert surv.cumhaz[0] == pytest.approx([efron_hazard[0], efron_hazard[0], efron_hazard[1]])
    assert event_only.cumhaz[0] == pytest.approx(efron_hazard)


def test_coxph_advanced_inputs_use_rust_optimizer():
    data = _toy_data()
    weights = [1.0, 1.5, 1.0, 0.8, 1.2, 1.0, 0.9, 1.1]
    offset = [0.1, 0.1, 0.0, -0.1, 0.0, -0.1, 0.1, 0.0]
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        weights=weights,
        offset=offset,
        init=[0.0],
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        weights=weights,
        offset=offset,
        initial_beta=[0.0],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_offset_uses_rust_optimizer():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=data["offset"],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)

    transformed = {**data, "exposure": [math.exp(value) for value in data["offset"]]}
    transformed_fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=transformed,
        max_iter=10,
        eps=1e-5,
    )
    arithmetic_offsets = [
        offset + x2 for offset, x2 in zip(data["offset"], data["x2"], strict=True)
    ]
    arithmetic_fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset + x2)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    arithmetic_low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=arithmetic_offsets,
        max_iter=10,
        eps=1e-5,
    )

    assert transformed_fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert transformed_fit.risk_scores == pytest.approx(low_level.risk_scores)
    assert arithmetic_fit.coefficients[0] == pytest.approx(arithmetic_low_level.coefficients[0])
    assert arithmetic_fit.risk_scores == pytest.approx(arithmetic_low_level.risk_scores)


def test_coxph_formula_accepts_intercept_only_rhs():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ 1", data=data, max_iter=0)
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        max_iter=0,
    )

    assert fit.coefficients == [[]]
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert survival.predict(fit, {"row": [0, 1]}) == pytest.approx([0.0, 0.0])
    assert survival.predict(fit, {"row": [0, 1]}, type="risk") == pytest.approx([1.0, 1.0])
    times, curves = survival.survfit(fit, newdata={"row": [0]})
    assert times
    assert len(curves) == 1


def test_coxph_formula_accepts_offset_only_rhs():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ offset(offset)",
        data=data,
        max_iter=0,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        offset=data["offset"],
        max_iter=0,
    )

    assert fit.coefficients == [[]]
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.linear_predictors == pytest.approx(data["offset"])
    assert fit.risk_scores == pytest.approx([math.exp(value) for value in data["offset"]])

    newdata = {"offset": [0.2, -0.1]}
    offset_center = sum(data["offset"]) / len(data["offset"])
    assert survival.predict(fit, newdata) == pytest.approx(
        [0.2 - offset_center, -0.1 - offset_center]
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx([0.2, -0.1])
    assert survival.predict(fit, newdata, type="risk") == pytest.approx(
        [math.exp(0.2 - offset_center), math.exp(-0.1 - offset_center)]
    )
