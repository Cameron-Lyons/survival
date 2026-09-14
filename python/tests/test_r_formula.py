import importlib
import math

import pytest

from .helpers import setup_survival_import
from .r_api_support import (
    _counting_cox_data,
    _factor_data,
    _interaction_contrast_data,
    _interaction_contrast_rows,
    _numeric_data,
    _toy_data,
    _with_intercept,
)

survival = setup_survival_import()
r_types = importlib.import_module("survival.r._types")
r_formula = importlib.import_module("survival.r._formula")


def test_formula_term_cache_returns_independent_terms():
    first = r_formula._split_terms("group + strata(x1) + offset(x2)", None)
    second = r_formula._split_terms("group + strata(x1) + offset(x2)", None)

    assert first is not second
    assert first.covariates == second.covariates
    assert first.strata == second.strata
    assert first.offsets == second.offsets

    first.covariates.append(r_types._CovariateTerm("mutated"))
    first.strata.append("mutated")
    first.offsets.clear()

    third = r_formula._split_terms("group + strata(x1) + offset(x2)", None)
    assert [term.column for term in third.covariates] == ["group"]
    assert third.strata == ["x1"]
    assert [term.column for term in third.offsets] == ["x2"]


def test_formula_metadata_preserves_categorical_spelling_and_assignments():
    data = _factor_data()
    data["group"] = ["A", "A", "B", "B", "C", "C", "A", "B"]
    data["band"] = [0, 1, 2, 0, 1, 2, 0, 1]
    formula = "Surv(time, status) ~ group + factor(dose) + as.factor(band) + x1"
    term_names = ["group", "factor(dose)", "as.factor(band)", "x1"]
    cox_columns = [
        "groupB",
        "groupC",
        "factor(dose)1",
        "factor(dose)2",
        "as.factor(band)1",
        "as.factor(band)2",
        "x1",
    ]

    cox = survival.coxph(formula, data=data, max_iter=0)
    aft = survival.survreg(formula, data=data, max_iter=1)

    assert survival.model_term_names(cox) == term_names
    assert survival.model_term_names(aft) == term_names
    assert survival.r_api.model_term_names(cox, [3, 1]) == [
        "as.factor(band)",
        "group",
    ]
    assert survival.coef_names(cox) == cox_columns
    assert survival.coef_names(aft) == ["(Intercept)", *cox_columns]

    cox_matrix = survival.model_matrix(cox)
    assert cox_matrix["columns"] == cox_columns
    assert cox_matrix["assign"] == [1, 1, 2, 2, 3, 3, 4]

    aft_matrix = survival.model_matrix(aft)
    assert aft_matrix["columns"] == ["(Intercept)", *cox_columns]
    assert aft_matrix["assign"] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_formula_design_orders_main_effects_before_interactions_like_r():
    data = _interaction_contrast_data()
    columns = ["x", "gB", "gC", "gB:x", "gC:x"]
    fit = survival.coxph(
        "Surv(time, status) ~ g:x + x + g",
        data=data,
        max_iter=0,
    )
    matrix = survival.model_matrix(fit)

    assert matrix["columns"] == columns
    for actual, expected in zip(
        matrix["data"],
        _interaction_contrast_rows(data, columns),
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_formula_identity_arithmetic_terms_rebuild_for_newdata():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ I(x1 + x2) + I(x1 * x2) + I(x1^2)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    expected_rows = [
        [
            data["x1"][idx] + data["x2"][idx],
            data["x1"][idx] * data["x2"][idx],
            data["x1"][idx] ** 2,
        ]
        for idx in range(len(data["time"]))
    ]
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(expected_rows),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.25, 1.0], "x2": [0.5, 0.8]}
    new_rows = [[0.75, 0.125, 0.0625], [1.8, 0.8, 1.0]]

    for actual, expected in zip(fit.covariates, low_level.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(_with_intercept(new_rows), "lp").predictions
    )


def test_formula_slash_expands_nested_terms_like_r():
    data = _numeric_data()
    nested = survival.survreg(
        "Surv(time, status) ~ x1 / x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x1:x2",
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
                [data["x1"][idx], data["x1"][idx] * data["x2"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(nested.covariates, low_level.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert nested.coefficients == pytest.approx(explicit.coefficients)
    assert nested.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_formula_in_operator_expands_nested_terms_like_r():
    data = _numeric_data()
    nested = survival.survreg(
        "Surv(time, status) ~ x1 + x2 %in% x1",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x1:x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    slash = survival.survreg(
        "Surv(time, status) ~ x1 / x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(nested.covariates, slash.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert nested.coefficients == pytest.approx(explicit.coefficients)
    assert nested.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_formula_power_expands_crossing_degree_like_r():
    data = _numeric_data()
    power = survival.survreg(
        "Surv(time, status) ~ (x1 + x2)^2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    crossed = survival.survreg(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    for actual, expected in zip(power.covariates, crossed.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert power.coefficients == pytest.approx(crossed.coefficients)
    assert power.log_likelihood == pytest.approx(crossed.log_likelihood)

    grouped_data = {
        **data,
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    grouped = survival.survfit(
        "Surv(time, status) ~ (x1 + x2 + x3)^2",
        data=grouped_data,
    )
    explicit = survival.survfit(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3",
        data=grouped_data,
    )

    assert list(grouped) == list(explicit)
    for key in explicit:
        assert grouped[key].estimate == pytest.approx(explicit[key].estimate)


def test_formula_parenthesized_crossing_expands_like_r():
    data = {
        **_numeric_data(),
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    grouped = survival.coxph(
        "Surv(time, status) ~ (x1 + x2) * x3",
        data=data,
        max_iter=0,
    )
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x3 + x2:x3",
        data=data,
        max_iter=0,
    )
    expected_rows = [
        [
            data["x1"][idx],
            data["x2"][idx],
            data["x3"][idx],
            data["x1"][idx] * data["x3"][idx],
            data["x2"][idx] * data["x3"][idx],
        ]
        for idx in range(len(data["time"]))
    ]

    for actual, expected in zip(grouped.covariates, expected_rows, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(grouped.covariates, explicit.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert grouped.log_likelihood == pytest.approx(explicit.log_likelihood)

    grouped_curves = survival.survfit("Surv(time, status) ~ (x1 + x2) * x3", data=data)
    explicit_curves = survival.survfit(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x3 + x2:x3",
        data=data,
    )
    assert list(grouped_curves) == list(explicit_curves)
    for key in explicit_curves:
        assert grouped_curves[key].estimate == pytest.approx(explicit_curves[key].estimate)


def test_formula_dot_expands_inside_compound_expressions_like_r():
    data = {
        **_numeric_data(),
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    crossed = survival.coxph(
        "Surv(time, status) ~ x1 * .",
        data=data,
        max_iter=0,
    )
    explicit_crossed = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x2 + x1:x3",
        data=data,
        max_iter=0,
    )
    expected_crossed_rows = [
        [
            data["x1"][idx],
            data["x2"][idx],
            data["x3"][idx],
            data["x1"][idx] * data["x2"][idx],
            data["x1"][idx] * data["x3"][idx],
        ]
        for idx in range(len(data["time"]))
    ]

    for actual, expected in zip(crossed.covariates, expected_crossed_rows, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(crossed.covariates, explicit_crossed.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert crossed.log_likelihood == pytest.approx(explicit_crossed.log_likelihood)

    interaction = survival.coxph(
        "Surv(time, status) ~ x1:.",
        data=data,
        max_iter=0,
    )
    explicit_interaction = survival.coxph(
        "Surv(time, status) ~ x1 + x1:x2 + x1:x3",
        data=data,
        max_iter=0,
    )

    for actual, expected in zip(
        interaction.covariates, explicit_interaction.covariates, strict=True
    ):
        assert actual == pytest.approx(expected)
    assert interaction.log_likelihood == pytest.approx(explicit_interaction.log_likelihood)

    powered = survival.survreg(
        "Surv(time, status) ~ (. - x2)^2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit_powered = survival.survreg(
        "Surv(time, status) ~ x1 + x3 + x1:x3",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(powered.covariates, explicit_powered.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert powered.coefficients == pytest.approx(explicit_powered.coefficients)
    assert powered.log_likelihood == pytest.approx(explicit_powered.log_likelihood)


def test_r_api_rejects_unsupported_formula_features():
    with pytest.raises(ValueError, match="unsupported formula"):
        survival.coxph("Surv(time, status) ~ x1(x2)", data=_toy_data())

    with pytest.raises(ValueError, match="unterminated backtick"):
        survival.survfit("Surv(time, status) ~ `group", data=_toy_data())

    with pytest.raises(ValueError, match=r"factor\(\) requires exactly one column"):
        survival.coxph("Surv(time, status) ~ factor(x1, x2)", data=_toy_data())

    with pytest.raises(ValueError, match=r"log\(\) requires exactly one column"):
        survival.coxph("Surv(time, status) ~ log(x1, x2)", data=_toy_data())

    data_with_zero = _numeric_data()
    data_with_zero["x2"][0] = 0.0
    with pytest.raises(ValueError, match="requires positive values"):
        survival.coxph("Surv(time, status) ~ log(x2)", data=data_with_zero)

    with pytest.raises(ValueError, match=r"cluster\(\)"):
        survival.survdiff("Surv(time, status) ~ x1 + cluster(group)", data=_toy_data())

    with pytest.raises(ValueError, match="method or ties"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            method="efron",
            ties="breslow",
        )

    with pytest.raises(ValueError, match="ambiguous"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), ties="e")

    with pytest.raises(TypeError, match="y"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), y=1)

    with pytest.raises(TypeError, match="x"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), x=1)

    with pytest.raises(ValueError, match=r"singular_ok or singular\.ok"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            singular_ok=False,
            **{"singular.ok": True},
        )

    with pytest.raises(ValueError, match="id must have length"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), id=["a"])

    nonrobust_id = survival.coxph(
        "Surv(time, status) ~ x1",
        data=_toy_data(),
        id=_toy_data()["group"],
        robust=False,
    )
    assert nonrobust_id.robust is False

    with pytest.raises(ValueError, match="nocenter"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), nocenter=[0.0, float("nan")])

    with pytest.raises(ValueError, match="fixed scale and strata"):
        survival.survreg("Surv(time, status) ~ x1 + strata(group)", data=_toy_data(), scale=1.0)

    with pytest.raises(ValueError, match="only supported"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), parms=[1.0])

    t_fit = survival.survreg(
        "Surv(time, status) ~ x1",
        data=_toy_data(),
        distribution="t",
        parms=[5.0],
        max_iter=150,
        eps=1e-10,
    )
    assert t_fit.distribution == "t"
    assert t_fit.distribution_parameters == pytest.approx([5.0])
    assert survival.coef(t_fit) == pytest.approx(
        [1.4187299567002853, 4.8815448794131919],
        abs=1e-3,
    )
    assert t_fit.scale == pytest.approx(1.6900377253911552, abs=1e-3)
    assert t_fit.log_likelihood == pytest.approx(-12.250153810177117, abs=5e-4)
    assert survival.predict(t_fit, type="response") == pytest.approx(
        [
            2.3950389325829233,
            3.3713479084655620,
            1.9068844446416042,
            5.3239658602308388,
            6.3002748361134771,
            7.2765838119961153,
            4.3476568843482006,
            8.2528927878787535,
        ],
        abs=1e-3,
    )
    expected_quantiles = [
        [1.1669107520147790, 2.3950389325829233, 3.6231671131510677],
        [2.1432197278974177, 3.3713479084655620, 4.5994760890337059],
        [0.6787562640734599, 1.9068844446416042, 3.1350126252097485],
        [4.0958376796626945, 5.3239658602308388, 6.5520940407989832],
        [5.0721466555453327, 6.3002748361134771, 7.5284030166816214],
        [6.0484556314279709, 7.2765838119961153, 8.5047119925642605],
        [3.1195287037800563, 4.3476568843482006, 5.5757850649163450],
        [7.0247646073106091, 8.2528927878787535, 9.4810209684468987],
    ]
    actual_quantiles = survival.predict(t_fit, type="quantile", p=[0.25, 0.5, 0.75])
    assert len(actual_quantiles) == len(expected_quantiles)
    for actual_row, expected_row in zip(actual_quantiles, expected_quantiles, strict=True):
        assert actual_row == pytest.approx(expected_row, abs=1e-3)
    assert survival.r_api.residuals(t_fit, type="response") == pytest.approx(
        [
            -1.3950389325829233,
            -1.3713479084655620,
            1.0931155553583958,
            -1.3239658602308388,
            -1.3002748361134771,
            -1.2765838119961153,
            2.6523431156517994,
            -0.2528927878787535,
        ],
        abs=1e-3,
    )
    assert survival.r_api.residuals(t_fit, type="deviance") == pytest.approx(
        [
            -2.1542980450204676,
            -2.1486549886724347,
            1.6110718176140864,
            -2.1375495641185078,
            0.7376815970416505,
            -2.1266948158001666,
            2.5055150731602964,
            1.0825873494449951,
        ],
        abs=2e-3,
    )
    assert survival.r_api.residuals(t_fit, type="working") == pytest.approx(
        [
            -1.8352385766643091,
            -1.7872894113558617,
            4.4964403406419073,
            -1.6944502176851766,
            1.4301227030749066,
            -1.6054634294757755,
            7.8023520372610085,
            1.9841875888800469,
        ],
        abs=1e-2,
    )

    concordance_data = _toy_data()
    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            weights=[1.0, -1.0, *([1.0] * (len(concordance_data["time"]) - 2))],
        )

    with pytest.raises(ValueError, match="cluster must have"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            cluster=["a"],
        )

    with pytest.raises(ValueError, match="formula cluster"):
        survival.concordance(
            "Surv(time, status) ~ x1 + cluster(group)",
            data=concordance_data,
            cluster=concordance_data["group"],
        )

    with pytest.raises(TypeError, match="ymin"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            ymin=object(),
        )

    counting_concordance_data = _counting_cox_data()
    counting_concordance_data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    with pytest.raises(ValueError, match="counting-process"):
        survival.concordance(
            "Surv(start, stop, status) ~ score",
            data=counting_concordance_data,
            timewt="S/G",
        )

    with pytest.raises(ValueError, match="influence"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            influence=4,
        )

    with pytest.raises(TypeError, match="ranks"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            ranks=1,
        )

    with pytest.raises(TypeError, match="keepstrata"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            keepstrata=object(),
        )

    with pytest.raises(TypeError, match="y"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), y=1)

    with pytest.raises(TypeError, match="x"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), x=1)

    with pytest.raises(ValueError, match="cluster must have"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), cluster=["a"])

    with pytest.raises(ValueError, match="scale must be non-negative"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), scale=-1.0)

    with pytest.raises(ValueError, match="scale must be finite"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), scale=float("nan"))

    with pytest.raises(ValueError, match="time="):
        survival.basehaz(survival.coxph("Surv(time, status) ~ x1", data=_toy_data()), time=[1.0])

    with pytest.raises(ValueError, match="weights"):
        survival.basehaz(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data()),
            weights=[1.0] * len(_toy_data()["time"]),
        )

    with pytest.raises(ValueError, match="status and linear_predictors"):
        survival.basehaz([1.0, 2.0], status=[1, 0])

    with pytest.raises(ValueError, match="newdata"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            newdata=[[0.0]],
        )

    with pytest.raises(ValueError, match="status must use 0/1 or 1/2"):
        survival.basehaz([1.0, 2.0], status=[0, 2], linear_predictors=[0.0, 0.0], centered=False)

    with pytest.raises(ValueError, match="linear_predictors contains non-finite"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, float("nan")],
            centered=False,
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            weights=[1.0, -1.0],
        )

    with pytest.raises(ValueError, match="at least one positive"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            weights=[0.0, 0.0],
        )

    with pytest.raises(ValueError, match="entry_times contains non-finite"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            entry_times=[0.0, float("nan")],
        )

    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=1)
    with pytest.raises(ValueError, match="predict type"):
        survival.predict(fit, [[0.5, 0.8]], type="score")

    with pytest.raises(ValueError, match="expected"):
        survival.predict(fit, [[0.5, 0.8]], type="expected")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.predict(fit, [[0.5, 0.8]], reference="s")

    matrix_fit = survival.coxph(
        survival.Surv(_toy_data()["time"], _toy_data()["status"]),
        x=[[x1, x2] for x1, x2 in zip(_toy_data()["x1"], _toy_data()["x2"], strict=True)],
        max_iter=1,
    )
    with pytest.raises(TypeError, match="design matrix"):
        survival.predict(matrix_fit, {"x1": [0.5], "x2": [0.8]})

    with pytest.raises(ValueError, match="residuals type"):
        survival.r_api.residuals(fit, type="unknown")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.r_api.residuals(fit, type="d")

    with pytest.raises(ValueError, match="Cox partial residuals"):
        survival.r_api.residuals(fit, type="score", terms="x1")

    data = _numeric_data()
    data["x1"][2] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        survival.coxph("Surv(time, status) ~ x1", data=data)

    with pytest.raises(ValueError, match="conf_level"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_level=1.2)

    with pytest.raises(ValueError, match="conf_int"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_int=1.2)

    with pytest.raises(ValueError, match="conf_level or conf_int"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_level=0.9,
            conf_int=0.8,
        )

    with pytest.raises(ValueError, match=r"conf_int or conf\.int"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_int=0.9,
            **{"conf.int": 0.8},
        )

    with pytest.raises(ValueError, match=r"conf_type or conf\.type"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_type="plain",
            **{"conf.type": "logit"},
        )

    dotted_timefix = survival.survfit(
        survival.Surv([1.0, 1.0 + 5e-10, 2.0], [1, 1, 0]),
        **{"time.fix": False},
    )
    assert dotted_timefix.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])

    with pytest.raises(TypeError, match="timefix"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), timefix=1)

    with pytest.raises(ValueError, match=r"se_fit or se\.fit"):
        survival.predict(fit, [[0.5, 0.8]], se_fit=True, **{"se.fit": False})

    with pytest.raises(TypeError, match="unexpected keyword"):
        survival.predict(fit, [[0.5, 0.8]], **{"na.action": "omit"})

    with pytest.raises(ValueError, match=r"max_iter or control\.iter\.max"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            max_iter=5,
            control={"iter.max": 10},
        )

    with pytest.raises(ValueError, match="outer\\.max"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"outer.max": 0},
        )

    with pytest.raises(ValueError, match="toler\\.inf"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"toler.inf": 0},
        )

    with pytest.raises(ValueError, match=r"eps or control\.rel\.tolerance"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            eps=1e-5,
            control={"rel.tolerance": 1e-6},
        )

    with pytest.raises(ValueError, match="debug"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"debug": math.nan},
        )

    with pytest.raises(ValueError, match="outer\\.max"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"outer.max": 0},
        )

    with pytest.raises(ValueError, match="only one of init"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            init=[0.0, 0.0, 0.0],
            initial=[0.0, 0.0, 0.0],
        )

    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=_toy_data(),
        max_iter=5,
        eps=1e-5,
    )
    with pytest.raises(ValueError, match="Cox partial residuals"):
        survival.r_api.residuals(aft, type="response", terms="x1")

    with pytest.raises(ValueError, match="conf_level"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            conf_level=1.2,
        )

    with pytest.raises(ValueError, match="conf_type"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_type="weird")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_type="l")

    with pytest.raises(ValueError, match="start_time"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), start_time=float("nan"))

    with pytest.raises(TypeError, match="time0"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), time0=1)

    with pytest.raises(TypeError, match="censor"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), censor=1)

    with pytest.raises(ValueError, match="censor"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), censor=False)

    with pytest.raises(ValueError, match="all observations"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), start_time=3.0)

    with pytest.raises(ValueError, match="survfit type"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), type="mystery")

    with pytest.raises(ValueError, match="stype"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), stype=3)

    with pytest.raises(ValueError, match="ctype"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), ctype=0)

    left_model = survival.survfit(survival.Surv([1.0, 2.0], [0, 1], type="left"), model=True)
    assert isinstance(left_model, survival.r_api.TurnbullSurvfitResult)
    assert left_model.model["response"].type == "left"

    with pytest.raises(ValueError, match="entry=TRUE"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), entry=True)

    with pytest.raises(ValueError, match="entry=TRUE"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), id=["a", "b"], entry=True)

    with pytest.raises(ValueError, match=r"se_fit or se\.fit"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            se_fit=False,
            **{"se.fit": True},
        )

    with pytest.raises(ValueError, match="id must have"):
        survival.survfit(
            survival.Surv([0.0, 0.0], [1.0, 2.0], [1, 0]),
            id=["a"],
        )

    with pytest.raises(ValueError, match="cluster must have"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), cluster=["a"])

    with pytest.raises(NotImplementedError, match="requires cluster or id"):
        survival.survfit(
            survival.Surv([0.0, 0.0], [1.0, 2.0], [1, 0]),
            robust=True,
        )

    robust_fh = survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), robust=True, type="fh")
    assert robust_fh.std_err == pytest.approx([0.2144409712, 0.2144409712])

    etype_fit = survival.survfit(
        survival.Surv([1.0, 2.0], [1, 0]),
        etype=[1, 2],
    )
    assert etype_fit.states == ("(s0)", "1")

    with pytest.raises(TypeError, match="entry"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), entry=1)

    with pytest.raises(ValueError, match=r"start\[0\] must be less than stop\[0\]"):
        survival.Surv([1.0, 1.0], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.Surv([1.0, float("inf")], [1, 0])

    with pytest.raises(ValueError, match="start contains non-finite"):
        survival.Surv([0.0, float("-inf")], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="stop contains non-finite"):
        survival.Surv([0.0, 1.0], [1.0, float("inf")], [1, 0])

    with pytest.raises(ValueError, match="time2 contains non-finite"):
        survival.Surv([1.0], [float("inf")], [3], type="interval")

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            weights=[1.0, -1.0],
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            reverse=True,
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            type="fh",
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            conf_type="plain",
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            start_time=1.0,
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            time0=True,
        )

    with pytest.raises(ValueError, match="Surv or formula"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            reverse=True,
        )

    cox_plain = survival.survfit(
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
        conf_type="plain",
    )
    assert cox_plain.conf_lower
    assert cox_plain.conf_upper

    with pytest.raises(ValueError, match="removed all endpoints"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            start_time=99.0,
        )

    with pytest.raises(ValueError, match="Surv or formula"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            type="fh",
        )

    with pytest.raises(NotImplementedError, match="right-censored and counting"):
        survival.survdiff(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            group=["A", "B"],
        )

    with pytest.raises(TypeError, match="timefix"):
        survival.survdiff(
            survival.Surv([1.0, 2.0], [1, 0]),
            group=["A", "B"],
            timefix=1,
        )

    with pytest.raises(ValueError, match=r"timefix or time\.fix"):
        survival.survdiff(
            survival.Surv([1.0, 1.0 + 5e-10, 2.0], [1, 1, 0]),
            group=["A", "A", "B"],
            timefix=False,
            **{"time.fix": True},
        )

    with pytest.raises(ValueError, match="right endpoint"):
        survival.Surv([2.0], [1.0], type="interval2")

    with pytest.raises(ValueError, match="one-dimensional"):
        survival.survfit(survival.Surv([1.0, 2.0, 3.0], [1, 0, 1]), group=["A", ["B"], "C"])

    with pytest.raises(ValueError, match="selects no rows"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), subset=[False, False])

    with pytest.raises(NotImplementedError, match="right, left, interval, and interval2"):
        survival.survreg(survival.Surv([0.0, 1.0], [1.0, 2.0], [1, 0]), x=[[1.0], [2.0]])

    with pytest.raises(ValueError, match="only one of init"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            init=[0.0],
            initial_beta=[0.0],
        )

    with pytest.raises(ValueError, match="formula offset"):
        survival.coxph(
            "Surv(time, status) ~ x1 + offset(offset)",
            data=_toy_data(),
            offset=[0.0] * 8,
        )

    with pytest.raises(ValueError, match="formula offset"):
        survival.survreg(
            "Surv(time, status) ~ x1 + offset(offset)",
            data=_toy_data(),
            offset=[0.0] * 8,
        )
