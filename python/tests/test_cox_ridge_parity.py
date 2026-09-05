"""Fixed-theta ridge Cox parity with survival 3.8.11, without runtime R calls."""

import json
import math
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "cox_ridge_r_reference.json").read_text()
)
_CASES = _REFERENCE["cases"]


def _assert_numeric(actual, expected):
    """Compare matrices as well as vectors, retaining useful failure locations."""
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_numeric(actual_value, expected_value)
    elif expected is None:
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=2e-6, abs=2e-8)


def _fit_case(case):
    options = {
        "method": case["method"],
        "robust": False,
        "max_iter": case["max_iter"],
        "eps": 1e-11,
        "x": True,
        "y": True,
        "model": True,
        "na_action": "omit",
    }
    if case["weighted"]:
        options["weights"] = case["data"]["w"]
    if case["subset"] is not None:
        options["subset"] = case["subset"]
    if case["initial_beta"] is not None:
        options["initial_beta"] = case["initial_beta"]
    return survival.coxph(case["formula"], data=case["data"], **options)


@pytest.fixture(scope="module", params=list(_CASES))
def ridge_case(request):
    case = _CASES[request.param]
    return request.param, case, _fit_case(case)


def test_ridge_joint_fit_and_penalty_metadata_match_r(ridge_case):
    _, reference, fit = ridge_case
    _assert_numeric(survival.coef(fit), reference["coefficients"])
    _assert_numeric(survival.vcov(fit), reference["variance"])
    _assert_numeric(fit.variance2, reference["variance2"])
    _assert_numeric(fit.df, reference["term_df"])
    _assert_numeric(fit.penalty, reference["penalty"])
    _assert_numeric(fit.means, reference["means"])
    _assert_numeric(fit.log_likelihood, reference["log_likelihood"])
    assert fit.penalty_diagnostics is not None
    _assert_numeric(fit.penalty_diagnostics.penalty_diagonal, reference["penalty_diagonal"])
    assert not fit.robust
    assert survival.coef_names(fit) == reference["coefficient_names"]
    assert survival.model_term_names(fit) == reference["term_names"]
    design = survival.model_matrix(fit)
    assert design["columns"] == reference["model_matrix_names"]
    _assert_numeric(design["data"], reference["model_matrix"])
    _assert_numeric(fit.x, reference["model_matrix"])


def test_ridge_likelihood_generics_use_fractional_term_df(ridge_case):
    _, reference, fit = ridge_case
    _assert_numeric(survival.degrees_freedom(fit), reference["df"])
    _assert_numeric(survival.loglik(fit), reference["log_likelihood"][-1])
    _assert_numeric(survival.aic(fit), reference["aic"])
    _assert_numeric(survival.bic(fit), reference["bic"])
    _assert_numeric(survival.extract_aic(fit), reference["extract_aic"])
    summary = survival.model_summary(fit)
    _assert_numeric(summary["df"], reference["df"])
    assert summary["robust"] is False
    assert summary["coefficient_names"] == reference["summary_coefficient_names"]
    for actual, expected in zip(
        summary["coefficients"], reference["summary_coefficients"], strict=True
    ):
        _assert_numeric(actual["coef"], expected[0])
        _assert_numeric(actual["se"], expected[1])
        _assert_numeric(actual["se2"], expected[2])
        _assert_numeric(actual["p"], expected[5])


@pytest.mark.parametrize("prediction_type", ["lp", "risk", "terms", "expected"])
@pytest.mark.parametrize("newdata", [False, True], ids=["training", "newdata"])
def test_ridge_predictions_and_standard_errors_match_reference(
    ridge_case, prediction_type, newdata
):
    _, reference, fit = ridge_case
    reference_key = "new_predictions" if newdata else "predictions"
    expected = reference[reference_key][prediction_type]
    corrections = reference["corrected_predictions"]
    if corrections:
        expected = corrections.get("newdata" if newdata else "training", {}).get(
            prediction_type, expected
        )
    prediction = survival.predict(
        fit,
        reference["newdata"] if newdata else None,
        type=prediction_type,
        se_fit=True,
    )
    _assert_numeric(prediction.fit, expected["fit"])
    _assert_numeric(prediction.se_fit, expected["se_fit"])
    plain_prediction = survival.predict(
        fit, reference["newdata"] if newdata else None, type=prediction_type
    )
    _assert_numeric(plain_prediction, prediction.fit)


def test_ridge_residuals_match_reference_at_penalized_coefficients(ridge_case):
    name, reference, fit = ridge_case
    for residual_type, expected in reference["residuals"].items():
        if reference["corrected_residuals"]:
            expected = reference["corrected_residuals"].get(residual_type, expected)
        options = (
            {"type": "score", "weighted": True}
            if residual_type == "weighted_score"
            else {"type": residual_type}
        )
        actual = survival.r_api.residuals(fit, **options)
        _assert_numeric(actual, expected)

    # Observation scores are scores of the likelihood, so their sum equals
    # the nonzero penalty gradient at the joint penalized optimum.
    if name != "zero_iterations" and reference["method"] != "exact":
        scores = survival.r_api.residuals(fit, type="score", weighted=True)
        score_sum = [sum(column) for column in zip(*scores, strict=True)]
        gradient = [
            diagonal * coefficient
            for diagonal, coefficient in zip(
                reference["penalty_diagonal"], survival.coef(fit), strict=True
            )
        ]
        _assert_numeric(score_sum, gradient)


def test_ridge_baseline_hazard_matches_r(ridge_case):
    _, reference, fit = ridge_case
    actual = survival.basehaz(fit)
    expected = reference["basehaz"]
    _assert_numeric(actual.time, expected["time"])
    _assert_numeric(actual.cumhaz, expected["cumhaz"])
    if expected["strata"] is not None:
        assert actual.strata_labels == expected["strata"]
    zero_reference = survival.basehaz(fit, centered=False)
    _assert_numeric(zero_reference.time, reference["basehaz_zero"]["time"])
    _assert_numeric(zero_reference.cumhaz, reference["basehaz_zero"]["cumhaz"])


def test_ridge_survival_curve_and_uncertainty_match_r(ridge_case):
    _, reference, fit = ridge_case
    newdata = {column: values[:1] for column, values in reference["newdata"].items()}
    actual = survival.survfit(fit, newdata=newdata)
    expected = reference["curve"]
    _assert_numeric(actual.time, expected["time"])
    _assert_numeric(actual.surv[0], expected["surv"])
    _assert_numeric(actual.cumhaz[0], expected["cumhaz"])
    _assert_numeric(actual.std_chaz[0], expected["std_chaz"])
    _assert_numeric(actual.conf_lower[0], expected["lower"])
    _assert_numeric(actual.conf_upper[0], expected["upper"])


def test_ridge_grouped_terms_change_effective_df_without_changing_fit():
    grouped = _fit_case(_CASES["grouped_scaled"])
    separate = _fit_case(_CASES["separate_equal_scaled"])
    _assert_numeric(survival.coef(grouped), survival.coef(separate))
    _assert_numeric(survival.vcov(grouped), survival.vcov(separate))
    _assert_numeric(grouped.variance2, separate.variance2)
    assert len(grouped.df) == 1
    assert len(separate.df) == 2
    assert survival.degrees_freedom(grouped) > survival.degrees_freedom(separate) + 0.5
    grouped_terms = survival.predict(grouped, type="terms")
    separate_terms = survival.predict(separate, type="terms")
    _assert_numeric([row[0] for row in grouped_terms], [sum(row) for row in separate_terms])


@pytest.mark.parametrize("explicit_robust", [False, True])
def test_ridge_suppresses_automatic_and_explicit_robust_covariance(explicit_robust):
    case = _CASES["weighted_scaled"]
    options = {"robust": True} if explicit_robust else {}
    with pytest.warns(RuntimeWarning, match="robust variance is not defined for a penalized model"):
        fit = survival.coxph(
            case["formula"], data=case["data"], weights=case["data"]["w"], **options
        )
    assert not fit.robust
    _assert_numeric(survival.vcov(fit), case["variance"])


def test_ridge_predictions_reuse_training_variance_and_term_groups():
    case = _CASES["mixed_grouped"]
    fit = _fit_case(case)
    newdata = case["newdata"]
    all_rows = survival.predict(fit, newdata, type="terms", se_fit=True)
    before_penalty = list(fit.penalty)
    for index in range(len(newdata["x"])):
        one_row = {column: [values[index]] for column, values in newdata.items()}
        single = survival.predict(fit, one_row, type="terms", se_fit=True)
        _assert_numeric(single.fit[0], all_rows.fit[index])
        _assert_numeric(single.se_fit[0], all_rows.se_fit[index])
    _assert_numeric(fit.penalty, before_penalty)
    selected = survival.predict(fit, newdata, type="terms", terms=[2], se_fit=True)
    _assert_numeric(selected.fit, [[row[1]] for row in all_rows.fit])
    _assert_numeric(selected.se_fit, [[row[1]] for row in all_rows.se_fit])


@pytest.mark.parametrize(
    "formula",
    [
        "Surv(time, event) ~ ridge(x)",
        "Surv(time, event) ~ ridge(x, df=0.5)",
    ],
)
def test_ridge_df_selection_is_explicitly_unsupported(formula):
    with pytest.raises((ValueError, NotImplementedError), match="theta|fixed|df"):
        survival.coxph(formula, data=_CASES["mixed_scaled_efron"]["data"])


def test_reference_preserves_and_documents_known_r_discrepancies():
    stratified = _CASES["weighted_offset_strata"]
    counting = _CASES["counting_weighted_strata"]
    assert set(stratified["known_differences"]) == {"stratum_residual", "expected_offset"}
    assert set(counting["known_differences"]) == {"expected_offset", "interval_uncertainty"}
    assert stratified["residuals"]["martingale"][0] < -1.0
    assert stratified["corrected_residuals"]["martingale"][0] > 0.9
    for case in (stratified, counting):
        original = case["new_predictions"]["expected"]["fit"]
        corrected = case["corrected_predictions"]["newdata"]["expected"]["fit"]
        assert any(abs(old - new) > 1e-3 for old, new in zip(original, corrected, strict=True))


@pytest.mark.parametrize(
    "case_name", ["mixed_grouped", "transformed_argument", "subset_frozen_scale"]
)
def test_ridge_model_frame_retains_plain_columns_and_stored_term_matrix(case_name):
    reference = _CASES[case_name]
    fit = _fit_case(reference)
    frame = survival.model_frame(fit)
    assert "x" in frame
    assert all(not isinstance(values[0], list | tuple) for values in frame.values() if values)
    ridge_names = [name for name in reference["term_names"] if name.startswith("ridge(")]
    for name in ridge_names:
        assert name in fit.model
        assert name not in frame
        assert len(fit.model[name]) == len(reference["model_matrix"])
        assert isinstance(fit.model[name][0], list)


@pytest.mark.parametrize("robust", [None, False, True])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
def test_ridge_no_events_returns_empty_fit_ignoring_initial_values_and_robust(robust, method):
    # R 3.8.11 coxph returns before optimization: means are unweighted even for
    # binary columns, coefficients are aliased, and robust covariance is absent.
    data = {
        "time": [1, 2, 3, 4, 5, 6],
        "event": [0, 0, 0, 0, 0, 0],
        "x": [0, 1, 0, 1, 0, 1],
        "z": [1, 4, 2, 6, 3, 8],
        "offset": [0.1, 0.3, 0.0, 0.2, 0.5, -0.1],
    }
    fit = survival.coxph(
        "Surv(time, event) ~ z + ridge(x, theta = 2) + offset(offset)",
        data=data,
        weights=[0.5, 2, 3, 4, 5, 6],
        initial_beta=[0.3, -0.2],
        robust=robust,
        method=method,
        model=True,
        x=True,
    )
    assert all(math.isnan(value) for value in survival.coef(fit))
    assert survival.coef_names(fit) == ["z", "ridge(x, theta = 2)"]
    assert fit.means == pytest.approx([4.0, 0.5])
    assert fit.iterations == 0
    assert fit.iter == 0
    assert fit.convergence_flag == 0
    assert fit.method == method
    assert fit.requested_method == method
    assert fit.penalty_diagnostics is None
    assert fit.naive_variance is None
    assert not fit.robust
    assert survival.degrees_freedom(fit) == 0
    _assert_numeric(survival.vcov(fit), [[0, 0], [0, 0]])
    _assert_numeric(fit.log_likelihood, [0, 0])
    _assert_numeric(survival.predict(fit), [value - 1 / 6 for value in data["offset"]])
    _assert_numeric(survival.predict(fit, type="expected"), [0] * 6)
    if method == "exact":
        for residual_type in ("score", "schoenfeld", "dfbeta", "dfbetas", "scaledsch"):
            with pytest.raises(ValueError, match="not available for the exact method"):
                survival.r_api.residuals(fit, type=residual_type)
