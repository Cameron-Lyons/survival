import importlib
import math
from statistics import NormalDist

import pytest

from .helpers import setup_survival_import
from .r_api_support import _counting_cox_data, _factor_data, _NamedMatrix, _toy_data

survival = setup_survival_import()
r_models = importlib.import_module("survival.r._models")


def test_as_data_frame_flattens_multistate_curves_like_r_summary():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        Factor(
            ["ill", "death", "censor", "death"],
            ["censor", "ill", "death"],
        ),
        type="mstate",
    )
    fit = survival.survfit(response, p0=[0.25, 0.5, 0.25])

    frame = survival.as_data_frame(fit)

    assert list(frame) == [
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "pstate",
        "std.err",
        "lower",
        "upper",
        "state",
    ]
    assert frame["time"] == pytest.approx([1.0, 2.0, 3.0, 4.0] * 3)
    assert frame["state"] == ["(s0)"] * 4 + ["ill"] * 4 + ["death"] * 4
    assert frame["n.risk"] == pytest.approx([4.0, 3.0, 2.0, 1.0] + [0.0] * 8)
    assert frame["n.event"] == pytest.approx(
        [0.0] * 4 + [1.0, 0.0, 0.0, 0.0] + [0.0, 1.0, 0.0, 1.0]
    )
    assert frame["n.censor"] == pytest.approx([0.0, 0.0, 1.0, 0.0] + [0.0] * 8)
    assert frame["pstate"] == pytest.approx(
        [0.1875, 0.125, 0.125, 0.0] + [0.5625] * 4 + [0.25, 0.3125, 0.3125, 0.4375]
    )

    grouped = survival.survfit(
        response,
        group=["a", "a", "b", "b"],
        p0=[0.25, 0.5, 0.25],
    )
    grouped_frame = survival.as_data_frame(grouped)
    assert grouped_frame["state"] == ["(s0)"] * 4 + ["ill"] * 4 + ["death"] * 4
    assert grouped_frame["strata"] == ["a", "a", "b", "b"] * 3
    assert grouped_frame["time"] == pytest.approx([1.0, 2.0, 3.0, 4.0] * 3)

    without_se = survival.as_data_frame(survival.survfit(response, se_fit=False))
    assert list(without_se) == [
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "pstate",
        "state",
    ]


def test_multistate_survfit_structure_preserves_matrix_dimensions_and_metadata():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        Factor(
            ["ill", "death", "censor", "death"],
            ["censor", "ill", "death"],
        ),
    )
    fit = survival.survfit(
        response,
        p0=[0.25, 0.5, 0.25],
        conf_level=0.9,
        conf_type="plain",
    )

    structure = r_models._survfit_multistate_structure(fit)

    assert structure["n"] == 4
    assert structure["n.id"] == 4
    assert structure["n.risk"] == fit.n_risk
    assert structure["n.event"] == fit.n_event
    assert structure["pstate"] == fit.pstate
    assert structure["n.transition"] == fit.n_transition
    assert structure["cumhaz"] == fit.cumhaz
    assert structure["p0"] == pytest.approx([0.25, 0.5, 0.25])
    assert structure["states"] == ["(s0)", "ill", "death"]
    assert structure["type"] == "mright"
    assert structure["conf.type"] == "plain"
    assert structure["conf.int"] == pytest.approx(0.9)
    assert structure["_transition_names"] == ["1:2", "1:3"]
    assert structure["transitions"] == {
        "values": [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "rows": ["(s0)", "ill", "death"],
        "columns": ["ill", "death", "(censored)"],
    }

    grouped = survival.survfit(
        response,
        group=["a", "a", "b", "b"],
        p0=[0.25, 0.5, 0.25],
    )
    grouped_structure = r_models._survfit_multistate_structure(grouped)
    assert grouped_structure["n"] == [2, 2]
    assert grouped_structure["n.id"] == [2, 2]
    assert grouped_structure["strata"] == {"a": 2, "b": 2}
    assert grouped_structure["time"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert grouped_structure["p0"] == [
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
    ]
    assert len(grouped_structure["pstate"]) == 4
    assert all(len(row) == 3 for row in grouped_structure["pstate"])

    subset = r_models._subset_survfit_multistate(fit, [1])
    assert subset.states == ("ill",)
    assert subset.p0 == pytest.approx([0.5])
    assert subset.pstate == [[row[1]] for row in fit.pstate]
    assert subset.n_risk == [[row[1]] for row in fit.n_risk]
    assert subset.transitions == ()
    assert subset.surv_type == fit.surv_type
    assert subset.conf_type == fit.conf_type
    assert subset.conf_level == fit.conf_level
    assert subset.oldstate == fit.states
    subset_structure = r_models._survfit_multistate_structure(subset)
    assert "n.id" not in subset_structure
    assert subset_structure["oldstate"] == ["(s0)", "ill", "death"]


def test_model_generic_helpers_report_core_fit_metadata():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    fixed_scale_aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        scale=1.0,
        max_iter=10,
        eps=1e-5,
    )
    matrix_cox = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[value] for value in data["x1"]],
        max_iter=0,
    )
    weighted_matrix_cox = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[value] for value in data["x1"]],
        weights=[1.0] * len(data["time"]),
        max_iter=0,
    )
    weighted_cox = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        weights=data["x2"],
        max_iter=0,
    )
    weighted_aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        weights=data["x2"],
        max_iter=1,
    )

    assert survival.coef(cox) == pytest.approx(cox.coefficients[0])
    for actual, expected in zip(survival.vcov(cox), cox.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.loglik(cox) == pytest.approx(cox.log_likelihood[-1])
    assert survival.nobs(cox) == sum(data["status"])
    assert survival.degrees_freedom(cox) == len(cox.coefficients[0])
    assert survival.aic(cox) == pytest.approx(
        -2.0 * survival.loglik(cox) + 2.0 * survival.degrees_freedom(cox)
    )
    assert survival.aic(cox, k=4) == pytest.approx(
        -2.0 * survival.loglik(cox) + 4.0 * survival.degrees_freedom(cox)
    )
    assert survival.bic(cox) == pytest.approx(
        -2.0 * survival.loglik(cox) + math.log(sum(data["status"])) * survival.degrees_freedom(cox)
    )
    assert survival.extract_aic(cox) == pytest.approx(
        [survival.degrees_freedom(cox), survival.aic(cox)]
    )
    assert survival.model_formula(cox) == "Surv(time, status) ~ x1 + x2 + cluster(group)"
    assert survival.model_weights(cox) is None
    assert survival.model_weights(weighted_matrix_cox) == pytest.approx([1.0] * len(data["time"]))
    assert survival.model_weights(weighted_cox) == pytest.approx(data["x2"])
    cox_matrix = survival.model_matrix(cox)
    assert cox_matrix["columns"] == ["x1", "x2"]
    for actual, expected in zip(cox_matrix["data"], cox.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.fitted(cox) == pytest.approx(survival.predict(cox))
    assert survival.fitted(cox, type="risk") == pytest.approx(survival.predict(cox, type="risk"))
    cox_fitted_with_se = survival.fitted(cox, **{"se.fit": True})
    assert cox_fitted_with_se.fit == pytest.approx(survival.fitted(cox))
    assert len(cox_fitted_with_se.se_fit) == len(data["time"])

    assert survival.coef(aft) == pytest.approx(aft.location_coefficients)
    for actual, expected in zip(survival.vcov(aft), aft.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.vcov(aft, complete=False),
        [row[: aft.n_covariates] for row in aft.variance_matrix[: aft.n_covariates]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_aft_loglik = aft.log_likelihood - sum(
        math.log(time)
        for time, event in zip(data["time"], data["status"], strict=True)
        if event == 1
    )
    assert survival.loglik(aft) == pytest.approx(expected_aft_loglik)
    assert survival.nobs(aft) == len(data["time"])
    assert survival.degrees_freedom(aft) == len(aft.coefficients)
    assert survival.df_residual(aft) == survival.nobs(aft) - survival.degrees_freedom(aft)
    assert survival.degrees_freedom(fixed_scale_aft) == len(fixed_scale_aft.location_coefficients)
    assert survival.aic(aft) == pytest.approx(
        -2.0 * survival.loglik(aft) + 2.0 * survival.degrees_freedom(aft)
    )
    assert survival.extract_aic(aft, k=3) == pytest.approx(
        [survival.degrees_freedom(aft), survival.aic(aft, k=3)]
    )
    assert survival.model_formula(aft) == "Surv(time, status) ~ x1 + x2"
    assert survival.model_weights(aft) is None
    assert survival.model_weights(weighted_aft) == pytest.approx(data["x2"])
    aft_matrix = survival.model_matrix(aft)
    assert aft_matrix["columns"] == ["(Intercept)", "x1", "x2"]
    for actual, expected in zip(aft_matrix["data"], aft.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.fitted(aft) == pytest.approx(survival.predict(aft))
    assert survival.fitted(aft, type="lp") == pytest.approx(survival.predict(aft, type="lp"))

    with pytest.raises(TypeError, match="fitted coxph or survreg"):
        survival.coef(survival.Surv([1.0, 2.0], [1, 0]))
    with pytest.raises(TypeError, match="unexpected keyword"):
        survival.fitted(cox, newdata={"x1": [0.2], "x2": [0.8]})
    with pytest.raises(ValueError, match="finite"):
        survival.aic(cox, k=float("nan"))
    with pytest.raises(TypeError, match="formula-based"):
        survival.model_formula(matrix_cox)
    with pytest.raises(TypeError, match="fitted coxph or survreg"):
        survival.model_weights(survival.Surv([1.0, 2.0], [1, 0]))
    with pytest.raises(TypeError, match="survreg"):
        survival.df_residual(cox)
    with pytest.raises(TypeError, match="stored model frame"):
        survival.model_frame(cox)


def test_cox_likelihood_metadata_counts_events_but_summary_counts_rows():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        weights=data["x2"],
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    event_count = sum(data["status"])
    assert survival.nobs(cox) == event_count
    assert survival.bic(cox) == pytest.approx(
        -2.0 * survival.loglik(cox) + math.log(event_count) * survival.degrees_freedom(cox)
    )
    cox_summary = survival.model_summary(cox)
    assert cox_summary["n"] == len(data["time"])
    assert cox_summary["n_event"] == event_count

    assert survival.nobs(aft) == len(data["time"])
    assert survival.bic(aft) == pytest.approx(
        -2.0 * survival.loglik(aft) + math.log(len(data["time"])) * survival.degrees_freedom(aft)
    )


def test_zero_event_cox_bic_is_nan():
    fit = survival.coxph(
        survival.Surv([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]),
        x=[[0.2], [0.4], [0.8], [1.0]],
    )

    assert survival.nobs(fit) == 0
    assert survival.degrees_freedom(fit) == 0
    summary = survival.model_summary(fit)
    assert summary["n"] == 4
    assert summary["n_event"] == 0
    assert math.isnan(survival.bic(fit))


def test_counting_process_and_intercept_only_cox_metadata_use_event_rows():
    start = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0]
    stop = [1.0, 3.0, 2.0, 4.0, 5.0, 2.0, 4.0, 6.0]
    status = [0, 1, 1, 0, 1, 1, 0, 1]
    values = [0.2, 0.4, 0.1, 0.8, 0.5, 0.3, 0.9, 0.6]
    response = survival.Surv(start, stop, status)
    counting = survival.coxph(
        response,
        x=[[value] for value in values],
        weights=[0.5, 2.0, 1.5, 0.75, 3.0, 1.25, 0.8, 4.0],
        max_iter=0,
    )

    event_count = sum(status)
    assert survival.nobs(counting) == event_count
    counting_summary = survival.model_summary(counting)
    assert counting_summary["n"] == len(status)
    assert counting_summary["n_event"] == event_count
    assert survival.bic(counting) == pytest.approx(
        -2.0 * survival.loglik(counting)
        + math.log(event_count) * survival.degrees_freedom(counting)
    )

    intercept_only = survival.coxph(
        response,
        x=[[] for _ in status],
        max_iter=0,
    )
    assert survival.nobs(intercept_only) == event_count
    assert survival.degrees_freedom(intercept_only) == 0
    assert survival.bic(intercept_only) == pytest.approx(-2.0 * survival.loglik(intercept_only))


def test_model_frame_returns_stored_formula_columns():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + offset(offset) + strata(group)",
        data=data,
        model=True,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        model=True,
        max_iter=10,
        eps=1e-5,
    )

    cox_frame = survival.model_frame(cox)
    assert {"time", "status", "x1", "x2", "offset", "group", "(offset)", "(strata)"} <= set(
        cox_frame
    )
    assert cox_frame["time"] == pytest.approx(data["time"])
    assert cox_frame["status"] == list(data["status"])
    assert cox_frame["x1"] == pytest.approx(data["x1"])
    assert cox_frame["(strata)"] == data["group"]

    aft_frame = survival.model_frame(aft)
    assert {"time", "status", "x1", "x2"} <= set(aft_frame)
    assert aft_frame["x2"] == pytest.approx(data["x2"])


def test_model_generic_helpers_report_formula_coefficient_names():
    data = _factor_data()
    data["group"] = ["A", "A", "B", "B", "C", "C", "A", "B"]
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + factor(group) + x1:x2",
        data=data,
        max_iter=0,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert survival.coef_names(cox) == [
        "x1",
        "factor(group)B",
        "factor(group)C",
        "x1:x2",
    ]
    assert survival.coef_names(aft) == ["(Intercept)", "x1", "x2"]
    assert survival.coef_names(aft, complete=True) == [
        "(Intercept)",
        "x1",
        "x2",
        "Log(scale:A)",
        "Log(scale:B)",
        "Log(scale:C)",
    ]
    assert len(survival.coef_names(aft, complete=True)) == len(aft.coefficients)
    assert len(survival.coef_names(aft)) == len(survival.coef(aft))


def test_model_matrix_assignments_keep_strata_term_positions():
    data = _factor_data()
    data["group"] = ["A", "A", "B", "B", "A", "B", "A", "B"]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group) + as.factor(dose)",
        data=data,
        max_iter=0,
    )

    matrix = survival.model_matrix(fit)
    assert survival.model_term_names(fit) == ["x1", "as.factor(dose)"]
    assert matrix["columns"] == ["x1", "as.factor(dose)1", "as.factor(dose)2"]
    assert matrix["assign"] == [1, 3, 3]


def test_direct_model_matrix_preserves_tabular_column_names():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    matrix = _NamedMatrix(["marker"], [[value] for value in data["x1"]])
    mapping = {"marker": data["x1"]}

    cox = survival.coxph(response, x=matrix, max_iter=0)
    cox_mapping = survival.coxph(response, x=mapping, max_iter=0)
    aft = survival.survreg(response, x=matrix, max_iter=1)
    aft_mapping = survival.survreg(response, x=mapping, max_iter=1)
    aft_matrix = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=mapping,
        max_iter=1,
    )
    omitted_response = survival.Surv(data["time"][:4], data["status"][:4])
    omitted = survival.coxph(
        omitted_response,
        x={"marker": [0.2, None, 0.8, 1.0]},
        na_action="omit",
        max_iter=0,
    )
    manual_time = [data["time"][0], *data["time"][2:4]]
    manual_status = [data["status"][0], *data["status"][2:4]]
    manual = survival.coxph(
        survival.Surv(manual_time, manual_status),
        x={"marker": [0.2, 0.8, 1.0]},
        max_iter=0,
    )

    assert survival.coef_names(cox) == ["marker"]
    assert survival.coef_names(cox_mapping) == ["marker"]
    assert cox_mapping.coefficients[0] == pytest.approx(cox.coefficients[0])
    assert survival.coef_names(aft) == ["marker"]
    assert survival.coef_names(aft_mapping) == ["marker"]
    assert aft_mapping.location_coefficients == pytest.approx(aft.location_coefficients)
    assert survival.coef_names(aft_matrix) == ["marker"]
    assert survival.coef_names(omitted) == ["marker"]
    assert omitted.coefficients[0] == pytest.approx(manual.coefficients[0])

    newdata = {"marker": [0.5, 0.7]}
    new_rows = [[0.5], [0.7]]
    assert survival.predict(cox_mapping, newdata) == pytest.approx(
        survival.predict(cox_mapping, new_rows)
    )
    assert survival.predict(aft_mapping, newdata) == pytest.approx(
        survival.predict(aft_mapping, new_rows)
    )
    cox_terms_by_name = survival.predict(cox_mapping, newdata, type="terms", terms="marker")
    cox_terms_by_index = survival.predict(cox_mapping, new_rows, type="terms", terms=[1])
    aft_terms_by_name = survival.predict(aft_mapping, newdata, type="terms", terms="marker")
    aft_terms_by_index = survival.predict(aft_mapping, new_rows, type="terms", terms=[1])
    for actual, expected in zip(cox_terms_by_name, cox_terms_by_index, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(aft_terms_by_name, aft_terms_by_index, strict=True):
        assert actual == pytest.approx(expected)
    cox_term_se = survival.predict(
        cox_mapping,
        newdata,
        type="terms",
        terms="marker",
        se_fit=True,
    )
    aft_term_se = survival.predict(
        aft_mapping,
        newdata,
        type="terms",
        terms="marker",
        se_fit=True,
    )
    assert len(cox_term_se.fit) == len(cox_term_se.se_fit) == 2
    assert len(aft_term_se.fit) == len(aft_term_se.se_fit) == 2
    named_curves = survival.survfit(cox_mapping, newdata=newdata, se_fit=False)
    matrix_curves = survival.survfit(cox_mapping, newdata=new_rows, se_fit=False)
    for actual, expected in zip(named_curves.surv, matrix_curves.surv, strict=True):
        assert actual == pytest.approx(expected)

    with pytest.raises(KeyError, match="marker"):
        survival.predict(cox_mapping, {"wrong": [0.5]})


def test_model_summary_reports_named_coefficient_table():
    data = _toy_data()
    plain_cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    plain_cox_summary = survival.model_summary(plain_cox)
    cox_summary = survival.model_summary(cox)
    aft_summary = survival.model_summary(aft)

    assert plain_cox_summary["robust"] is False
    for row in plain_cox_summary["coefficients"]:
        assert row["naive_se"] == pytest.approx(row["se"])
        assert "robust_se" not in row

    assert cox_summary["model_type"] == "coxph"
    assert cox_summary["robust"] is True
    assert cox_summary["n"] == len(data["time"])
    assert cox_summary["n_event"] == sum(data["status"])
    assert cox_summary["df"] == 2
    assert cox_summary["loglik"] == pytest.approx(cox.log_likelihood[-1])
    assert cox_summary["null_loglik"] == pytest.approx(cox.log_likelihood[0])
    assert cox_summary["score_test"] == pytest.approx(cox.score_test)
    assert plain_cox_summary["score_test"] == pytest.approx(plain_cox.score_test)
    assert [row["name"] for row in cox_summary["coefficients"]] == ["x1", "x2"]
    assert cox_summary["coefficient_names"] == ["x1", "x2"]
    for idx, (row, coefficient, variance_row, naive_variance_row) in enumerate(
        zip(
            cox_summary["coefficients"],
            survival.coef(cox),
            survival.vcov(cox),
            cox.naive_variance,
            strict=True,
        )
    ):
        assert row["coef"] == pytest.approx(coefficient)
        assert row["exp_coef"] == pytest.approx(math.exp(coefficient))
        assert row["se"] == pytest.approx(math.sqrt(max(variance_row[idx], 0.0)))
        assert row["naive_se"] == pytest.approx(math.sqrt(max(naive_variance_row[idx], 0.0)))
        assert row["robust_se"] == pytest.approx(row["se"])
        assert row["statistic"] == pytest.approx(row["coef"] / row["se"])
        assert row["z"] == pytest.approx(row["statistic"])
        assert row["p"] == pytest.approx(2.0 * NormalDist().cdf(-abs(row["z"])))
        assert 0.0 <= row["p"] <= 1.0

    assert aft_summary["model_type"] == "survreg"
    assert aft_summary["n"] == len(data["time"])
    assert aft_summary["df"] == len(aft.coefficients)
    assert aft_summary["scale"] == pytest.approx(aft.scale)
    assert aft_summary["scales"] == pytest.approx(aft.scales)
    assert [row["name"] for row in aft_summary["coefficients"]] == [
        "(Intercept)",
        "x1",
        "x2",
        "Log(scale)",
    ]
    assert aft_summary["coefficient_names"] == [
        "(Intercept)",
        "x1",
        "x2",
        "Log(scale)",
    ]
    assert aft_summary["location_coefficient_names"] == ["(Intercept)", "x1", "x2"]
    assert aft_summary["location_coefficients"] == pytest.approx(aft.location_coefficients)
    for idx, (row, coefficient, variance_row) in enumerate(
        zip(
            aft_summary["coefficients"],
            aft.coefficients,
            survival.vcov(aft),
            strict=True,
        )
    ):
        assert row["coef"] == pytest.approx(coefficient)
        assert row["value"] == pytest.approx(coefficient)
        assert row["se"] == pytest.approx(math.sqrt(max(variance_row[idx], 0.0)))
        assert row["naive_se"] == pytest.approx(row["se"])
        assert "robust_se" not in row
        assert row["z"] == pytest.approx(row["statistic"])
        assert row["p"] == pytest.approx(2.0 * NormalDist().cdf(-abs(row["z"])))
        assert 0.0 <= row["p"] <= 1.0


def test_model_summary_reports_zero_coefficient_cox_inference_payload():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ 1", data=data, max_iter=0)

    summary = survival.model_summary(fit)

    assert summary["coefficients"] == []
    assert summary["coefficient_names"] == []
    assert summary["df"] == 0
    assert summary["loglik"] == pytest.approx(fit.log_likelihood[-1])
    assert summary["null_loglik"] == pytest.approx(fit.log_likelihood[0])
    assert summary["score_test"] == pytest.approx(fit.score_test)
    assert summary["score_test"] == pytest.approx(0.0)
    assert summary["n_event"] == sum(data["status"])


def test_model_confint_reports_named_coefficient_intervals():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    z = NormalDist().inv_cdf(0.95)

    intervals = survival.confint(cox, level=0.9)

    assert [row["name"] for row in intervals] == ["x1", "x2"]
    for idx, row in enumerate(intervals):
        coefficient = survival.coef(cox)[idx]
        se = math.sqrt(max(survival.vcov(cox)[idx][idx], 0.0))
        assert row["lower"] == pytest.approx(coefficient - z * se)
        assert row["upper"] == pytest.approx(coefficient + z * se)

    assert survival.confint(cox, parm="x2")[0]["name"] == "x2"
    assert [row["name"] for row in survival.confint(aft, parm=[1, "x2"])] == [
        "(Intercept)",
        "x2",
    ]
    with pytest.raises(ValueError, match="level"):
        survival.confint(cox, level=1.5)
    with pytest.raises(ValueError, match="unknown coefficient"):
        survival.confint(cox, parm="missing")
    with pytest.raises(IndexError, match="parm index"):
        survival.confint(cox, parm=0)
    with pytest.raises(TypeError, match="parm"):
        survival.confint(cox, parm=1.5)


def test_as_data_frame_returns_r_friendly_result_tables():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    surv_table = survival.as_data_frame(survival.Surv(data["time"], data["status"]))
    assert set(surv_table) == {"time", "status", "type"}
    assert surv_table["time"] == pytest.approx(data["time"])
    assert surv_table["status"] == data["status"]
    assert surv_table["type"] == ["right"] * len(data["time"])

    counting_table = survival.as_data_frame(survival.Surv([0.0, 1.0], [2.0, 3.0], [1, 0]))
    assert set(counting_table) == {"start", "stop", "status", "type"}
    assert counting_table["start"] == pytest.approx([0.0, 1.0])
    assert counting_table["stop"] == pytest.approx([2.0, 3.0])
    assert counting_table["type"] == ["counting", "counting"]

    interval_table = survival.as_data_frame(
        survival.Surv([1.0, 2.0], [2.0, 4.0], [3, 0], type="interval")
    )
    assert set(interval_table) == {"time", "status", "time2", "type"}
    assert interval_table["time2"] == pytest.approx([2.0, 4.0])
    assert interval_table["type"] == ["interval", "interval"]

    km_table = survival.as_data_frame(survival.survfit("Surv(time, status) ~ group", data=data))
    assert {"strata", "time", "n.risk", "n.event", "surv", "cumhaz"} <= set(km_table)
    assert len(km_table["strata"]) == len(km_table["time"])
    assert set(km_table["strata"]) == {"A", "B"}

    zero_survival_table = survival.as_data_frame(
        survival.survfit(survival.Surv([1.0, 2.0], [1, 1]))
    )
    assert zero_survival_table["surv"][-1] == pytest.approx(0.0)
    assert math.isnan(zero_survival_table["std.err"][-1])
    assert math.isnan(zero_survival_table["lower"][-1])
    assert math.isnan(zero_survival_table["upper"][-1])

    cox_surv_table = survival.as_data_frame(
        survival.survfit(cox, newdata={"x1": [0.2, 1.0], "x2": [1.0, 0.4]})
    )
    assert {"curve", "time", "surv", "cumhaz", "linear.predictor"} <= set(cox_surv_table)
    assert set(cox_surv_table["curve"]) == {1, 2}
    assert len(cox_surv_table["surv"]) == 2 * len(set(data["time"]))

    basehaz_table = survival.as_data_frame(survival.basehaz(cox))
    assert set(basehaz_table) == {"time", "cumhaz"}
    assert basehaz_table["time"] == sorted(set(data["time"]))

    survdiff_table = survival.as_data_frame(
        survival.survdiff("Surv(time, status) ~ group", data=data)
    )
    assert set(survdiff_table) == {"group", "observed", "expected", "variance"}
    assert survdiff_table["group"] == [1, 2]

    zph_table = survival.as_data_frame(survival.cox_zph(cox))
    assert {"name", "chisq", "df", "p"} <= set(zph_table)
    assert zph_table["name"][-1] == "GLOBAL"

    detail_table = survival.as_data_frame(survival.coxph_detail(cox))
    assert {"time", "n.event", "n.risk", "hazard", "cumhaz"} <= set(detail_table)
    assert len(detail_table["time"]) == sum(data["status"])

    anova_table = survival.as_data_frame(survival.anova(cox))
    assert set(anova_table) == {"model", "loglik", "df", "chisq", "p"}
    assert anova_table["model"][0] == "NULL"


def test_predict_expected_and_residuals_share_martingale_identity():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)

    expected = survival.predict(fit, type="expected")
    martingale = survival.r_api.residuals(fit)
    deviance = survival.r_api.residuals(fit, type="deviance")

    assert expected == pytest.approx(fit.expected_events())
    assert martingale == pytest.approx(fit.martingale_residuals())
    assert martingale == pytest.approx(
        [status - value for status, value in zip(fit.status, expected, strict=True)]
    )
    assert len(deviance) == len(data["time"])
    assert all(math.isfinite(value) for value in deviance)


def test_predict_terms_constant_restores_zero_reference_linear_predictor():
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_toy_data(),
        initial_beta=[0.25, -0.4],
        max_iter=0,
    )
    constant = survival.r_api.predict_terms_constant(fit)
    term_predictions = survival.predict(fit, type="terms")
    zero_reference = survival.predict(fit, reference="zero")

    assert constant == pytest.approx(
        sum(
            mean * coefficient
            for mean, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
        )
    )
    assert [sum(row) + constant for row in term_predictions] == pytest.approx(zero_reference)

    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=_toy_data(),
        max_iter=1,
    )
    with pytest.raises(TypeError, match="coxph"):
        survival.r_api.predict_terms_constant(aft)


def test_predict_expected_uses_counting_process_entry_intervals():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )

    hazard_times, hazards = fit.basehaz(False)
    expected = survival.predict(fit, type="expected")
    martingale = survival.r_api.residuals(fit, type="martingale")

    assert hazard_times == pytest.approx([2.0, 4.0, 5.0])
    assert expected[3] == pytest.approx(hazards[-1] - hazards[0])
    assert martingale[3] == pytest.approx(-expected[3])


def test_ordinary_istate_matches_r_model_frame_semantics():
    ordinary_curve = survival.survfit(
        survival.Surv([1.0, 2.0], [1, 0]),
        istate=["entry", "other"],
        model=True,
    )
    curve_reference = survival.survfit(survival.Surv([1.0, 2.0], [1, 0]))
    assert ordinary_curve.estimate == pytest.approx(curve_reference.estimate)
    assert ordinary_curve.model["(istate)"] == ["entry", "other"]

    data = _toy_data()
    ordinary_fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        istate="group",
        statedata={"states": ["A", "B"]},
        model=True,
        max_iter=0,
    )
    fit_reference = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        model=True,
        max_iter=0,
    )
    assert ordinary_fit.coefficients[0] == pytest.approx(fit_reference.coefficients[0])
    assert ordinary_fit.model["(istate)"] == data["group"]
    assert ordinary_fit.model["group"] == data["group"]

    missing_istate_data = _toy_data()
    missing_istate_data["group"] = [*missing_istate_data["group"][:-1], None]
    omitted_fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=missing_istate_data,
        istate="group",
        na_action="omit",
        model=True,
        max_iter=0,
    )
    assert omitted_fit.model["(istate)"] == missing_istate_data["group"][:-1]
    assert len(omitted_fit.y) == len(missing_istate_data["group"]) - 1
