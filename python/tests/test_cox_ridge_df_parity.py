"""R survival 3.8.11 references for effective-df ridge selection and joint fits."""

import json
import math
import warnings
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "cox_ridge_df_r_reference.json").read_text()
)
_CASES = _REFERENCE["cases"]
_PARITY_CASES = {
    name: case
    for name, case in _CASES.items()
    if not case.get("controller_roundoff_sensitive", False)
}
_SELECTED_CASES = [
    name
    for name, case in _PARITY_CASES.items()
    if any(history["history"] is not None for history in case["history"].values())
]
_STANDALONE_CASES = [
    name
    for name, case in _CASES.items()
    if case["standalone_df"] is not None or case["standalone_theta"] is not None
]


def _assert_numeric(actual, expected):
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
        "control": {
            "iter.max": case["max_iter"],
            "outer.max": case["outer_max"],
            "eps": 1e-11,
        },
        "x": True,
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


@pytest.fixture(scope="module", params=list(_PARITY_CASES))
def ridge_df_case(request):
    case = _CASES[request.param]
    return request.param, case, _fit_case(case)


def test_selected_ridge_joint_fit_matches_r(ridge_df_case):
    _, reference, fit = ridge_df_case
    _assert_numeric(survival.coef(fit), reference["coefficients"])
    _assert_numeric(survival.vcov(fit), reference["variance"])
    _assert_numeric(fit.variance2, reference["variance2"])
    _assert_numeric(fit.df, reference["term_df"])
    _assert_numeric(fit.means, reference["means"])
    _assert_numeric(fit.penalty_diagnostics.penalty_diagonal, reference["penalty_diagonal"])
    assert survival.coef_names(fit) == reference["coefficient_names"]
    assert survival.model_term_names(fit) == reference["term_names"]


def test_selected_ridge_reports_first_and_final_penalties_and_likelihoods(ridge_df_case):
    _, reference, fit = ridge_df_case
    _assert_numeric(fit.penalty, reference["penalty"])
    _assert_numeric(fit.log_likelihood, reference["log_likelihood"])
    _assert_numeric(survival.degrees_freedom(fit), reference["df"])
    _assert_numeric(survival.aic(fit), reference["aic"])
    _assert_numeric(survival.bic(fit), reference["bic"])
    summary = survival.model_summary(fit)
    _assert_numeric(summary["df"], reference["df"])
    _assert_numeric(summary["penalty"], reference["penalty"])


@pytest.mark.parametrize("case_name", _SELECTED_CASES)
def test_ridge_controller_history_distinguishes_evaluated_and_proposed_theta(case_name):
    reference = _CASES[case_name]
    fit = _fit_case(reference)
    assert list(fit.history) == list(reference["history"])
    selection = fit.ridge_selection
    assert selection is not None
    assert fit.iter == reference["iterations"]
    assert selection.outer_iterations == reference["iterations"][0]
    assert selection.inner_iterations == reference["iterations"][1]
    _assert_numeric(selection.penalty, reference["penalty"])
    _assert_numeric(selection.initial_loglik, reference["log_likelihood"][0])
    for label, expected in reference["history"].items():
        actual = fit.history[label]
        assert actual["done"] is expected["done"]
        if expected["half"] is not None:
            assert actual["half"] == expected["half"]
        _assert_numeric(actual["theta"], expected["theta"])
        if expected["history"] is None:
            assert actual["history"] is None
        else:
            _assert_numeric(actual["history"], expected["history"])
        term_index = reference["term_names"].index(label)
        _assert_numeric(selection.fitted_theta[term_index], reference["applied_theta"][label])
        _assert_numeric(selection.proposed_theta[term_index], expected["theta"])
        assert selection.done[term_index] is expected["done"]


@pytest.mark.parametrize("case_name", _STANDALONE_CASES)
def test_standalone_ridge_uses_the_joint_cox_fit_and_model_selected_df(case_name):
    reference = _CASES[case_name]
    rows = reference["model_matrix"]
    n_obs = len(rows)
    n_vars = len(rows[0])
    scale = reference["standalone_scale"]
    target_df = reference["standalone_df"]
    penalty = (
        survival.RidgePenalty.from_df(target_df, n_vars, scale=scale)
        if target_df is not None
        else survival.RidgePenalty(reference["standalone_theta"], scale=scale)
    )
    fit = survival.ridge_fit(
        [value for row in rows for value in row],
        n_obs,
        n_vars,
        reference["data"]["time"],
        reference["data"]["event"],
        penalty,
        weights=reference["data"]["w"] if reference["weighted"] else None,
    )
    _assert_numeric(fit.coefficients, reference["coefficients"])
    _assert_numeric(fit.std_err, reference["std_err"])
    _assert_numeric(fit.df, reference["df"])
    _assert_numeric(fit.theta, next(iter(reference["applied_theta"].values())))
    if scale:
        _assert_numeric(fit.scale_factors, reference["scale_factors"])
    else:
        assert fit.scale_factors is None
    gcv = (-2 * reference["log_likelihood"][-1] / n_obs) / (1 - reference["df"] / n_obs) ** 2
    _assert_numeric(fit.gcv, gcv)


def test_default_selection_tolerance_stops_at_the_evaluated_df():
    reference = _CASES["default_grouped"]
    fit = _fit_case(reference)
    label = "ridge(x, z)"
    history = fit.history[label]
    assert abs(fit.df[0] - 1.0) < 0.1
    assert abs(fit.df[0] - 1.0) > 0.01
    _assert_numeric(history["history"][-1][1], fit.df[0])
    assert history["theta"] != pytest.approx(history["history"][-1][0])
    assert fit.ridge_selection.fitted_theta[0] != pytest.approx(2 / 1 - 1)


def test_outer_iteration_limit_preserves_unfinished_controller_state():
    reference = _CASES["outer_limit"]
    fit = _fit_case(reference)
    assert fit.iter[0] == 4
    assert fit.ridge_selection.done == [False]
    assert fit.df[0] > 0.2
    _assert_numeric(fit.df, reference["term_df"])


def test_full_df_boundary_fits_zero_theta_and_retains_nan_next_proposal():
    reference = _CASES["full_df"]
    fit = _fit_case(reference)
    assert fit.ridge_selection.fitted_theta == [0.0]
    assert math.isnan(fit.ridge_selection.proposed_theta[0])
    assert fit.ridge_selection.done == [True]
    _assert_numeric(fit.df, [2.0])
    _assert_numeric(fit.penalty[-1], 0.0)


def test_roundoff_sensitive_controller_returns_a_consistent_attained_fit():
    reference = _CASES["subset_roundoff_sensitive_controller"]
    fit = _fit_case(reference)
    label = next(iter(fit.history))
    actual_theta = fit.ridge_selection.fitted_theta[1]
    fixed = _fit_case(
        dict(reference, formula=f"Surv(time, event) ~ z + ridge(x, theta={actual_theta!r})")
    )
    # Retain the raw R path; it is unstable at a df rounding difference of 2^-53.
    assert reference["history"][label]["history"][5][1] == 1 - 2**-53
    assert fit.iter[0] <= reference["outer_max"]
    assert all(math.isfinite(value) for value in survival.coef(fit))
    history = fit.history[label]["history"]
    # The singular interpolation at row 8 must take a bracket midpoint and
    # continue improving the attained df, rather than repeating the last theta.
    _assert_numeric(history[8][0], (history[5][0] + history[7][0]) / 2)
    assert history[7][1] < history[8][1] < history[9][1] < history[10][1]
    assert history[-1][1] > 0.3
    assert math.isfinite(fit.history[label]["theta"])
    _assert_numeric(survival.coef(fit), survival.coef(fixed))
    _assert_numeric(survival.vcov(fit), survival.vcov(fixed))
    _assert_numeric(fit.variance2, fixed.variance2)
    _assert_numeric(fit.df, fixed.df)
    _assert_numeric(fit.log_likelihood[-1], fixed.log_likelihood[-1])
    _assert_numeric(fit.penalty[-1], fixed.penalty[-1])
    _assert_numeric(fit.history[label]["history"][-1], [actual_theta, fit.df[1]])
    assert fit.history[label]["done"] is (abs(fit.df[1] - 0.6) < 0.1)


@pytest.mark.parametrize("outer_max", [0, -1, 1.5, math.inf, math.nan])
def test_ridge_outer_iteration_limit_must_be_a_positive_integer(outer_max):
    with pytest.raises(ValueError, match="outer.max"):
        survival.coxph(
            "Surv(time, event) ~ ridge(x, z)",
            data=_CASES["default_grouped"]["data"],
            control={"outer.max": outer_max},
        )


def test_ridge_inner_iteration_failures_warn_with_affected_outer_iterations():
    reference = dict(_CASES["default_grouped"], max_iter=2, outer_max=3)
    with pytest.warns(RuntimeWarning, match="Inner loop failed to converge for iterations 1 2 3"):
        fit = _fit_case(reference)
    assert fit.ridge_selection.inner_failures == [1, 2, 3]


@pytest.mark.parametrize("max_iter", [0, 1])
def test_zero_or_one_inner_iteration_does_not_warn(max_iter):
    reference = dict(_CASES["default_grouped"], max_iter=max_iter, outer_max=3)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fit = _fit_case(reference)
    assert not recorded
    assert fit.ridge_selection.inner_failures == []
    assert fit.iter[1] == 3 * max_iter


@pytest.mark.parametrize("df", [-0.1, 2.1, math.inf, math.nan])
def test_standalone_ridge_rejects_nonconvex_or_nonfinite_df_targets(df):
    with pytest.raises(ValueError, match="df"):
        survival.RidgePenalty.from_df(df, 2)


@pytest.mark.parametrize(
    "formula",
    [
        "Surv(time, event) ~ ridge(x, z, df = -0.1)",
        "Surv(time, event) ~ ridge(x, z, df = 2.1)",
        "Surv(time, event) ~ ridge(x, z, df = 1, eps = 0)",
        "Surv(time, event) ~ ridge(x, z, df = 1, eps = -0.1)",
    ],
)
def test_ridge_formula_validates_df_and_controller_tolerance(formula):
    with pytest.raises(ValueError, match="df|eps"):
        survival.coxph(formula, data=_CASES["default_grouped"]["data"])
