"""Prediction parity with cached R survival 3.8.11 references.

Regenerate the JSON with the adjacent R script. Exact initial parameters and
positive definite information at maxiter=0 isolate prediction from optimization.
R drops new-data offsets; the offset regression below checks their mathematical
effect instead of reproducing that R bug.
"""

import json
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "survreg_prediction_r3811.json").read_text()
)
CASES = REFERENCE["fixtures"]
PROBABILITIES = REFERENCE["probabilities"]
TYPES = ("lp", "response", "quantile", "uquantile")
ALIASES = {
    "normal": "gaussian",
    "GAUSSIAN": "gaussian",
    "log_normal": "lognormal",
    "log-normal": "lognormal",
    "loggaussian": "lognormal",
    "log_gaussian": "lognormal",
    "log-logistic": "loglogistic",
    "extreme_value": "extreme",
    "extreme-value": "extreme",
    "extremevalue": "extreme",
    "student": "t",
    "student_t": "t",
    "student-t": "t",
    "studentt": "t",
}


def _array(value):
    return np.asarray(value, dtype=float)


def _assert_close(actual, expected):
    # The existing native normal inverse CDF has about 1e-9 absolute error at
    # these probabilities. This tolerance also covers the Student-t inversion.
    np.testing.assert_allclose(actual, _array(expected), rtol=2e-8, atol=2e-8, equal_nan=True)


@cache
def _fit(name, distribution=None):
    case = CASES[name]
    formula = "Surv(time, status) ~ x"
    if case["mode"] == "stratified":
        formula += " + strata(group)"
    elif case["mode"] == "offset":
        formula += " + offset(offset_value)"
    kwargs = {"parms": case["parms"]} if case["distribution"] == "t" else {}
    with warnings.catch_warnings():
        # The forced-scale families warn even when given their required scale.
        warnings.filterwarnings("ignore", message=".*has a fixed scale.*", category=RuntimeWarning)
        return survival.survreg(
            formula,
            data=case["data"],
            dist=distribution or case["distribution"],
            scale=case["fixed_scale"],
            init=case["initial"],
            max_iter=0,
            **kwargs,
        )


def _native_inputs(case, where):
    if where == "training":
        return {}
    return {
        "covariates": [[1.0, value] for value in case["newdata"]["x"]],
        "offset": case["newdata"]["offset_value"] if case["mode"] == "offset" else None,
    }


def _quantile_inputs(case, where):
    kwargs = _native_inputs(case, where)
    if where == "new" and case["mode"] == "stratified":
        kwargs["strata"] = [1, 0, 1]
    return kwargs


@pytest.mark.parametrize("name", CASES)
def test_reference_covariance_agrees_before_comparing_prediction_standard_errors(name):
    case = CASES[name]
    fit = _fit(name)
    np.testing.assert_allclose(fit.location_coefficients, case["beta"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(fit.scales, np.atleast_1d(case["scales"]), rtol=0, atol=1e-14)
    # This is intentionally much tighter than the quantile approximation tolerance.
    np.testing.assert_allclose(fit.variance_matrix, case["variance"], rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("where", ["training", "new"])
@pytest.mark.parametrize("kind", ["lp", "response"])
def test_native_location_predictions_and_standard_errors_match_r(name, where, kind):
    case = CASES[name]
    actual = _fit(name).predict(**_native_inputs(case, where), predict_type=kind, se_fit=True)
    expected = case[where][kind]
    _assert_close(actual.predictions, expected["fit"])
    assert actual.se is not None
    _assert_close(actual.se, expected["se.fit"])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("where", ["training", "new"])
@pytest.mark.parametrize("kind", TYPES)
def test_generic_predictions_and_standard_errors_match_r(name, where, kind):
    case = CASES[name]
    actual = survival.predict(
        _fit(name),
        None if where == "training" else case["newdata"],
        type=kind,
        p=PROBABILITIES,
        se_fit=True,
    )
    expected = case[where][kind]
    _assert_close(actual.fit, expected["fit"])
    _assert_close(actual.se_fit, expected["se.fit"])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("where", ["training", "new"])
@pytest.mark.parametrize("transform", [True, False])
def test_native_quantiles_use_each_rows_scale_and_fitted_distribution_parameters(
    name, where, transform
):
    case = CASES[name]
    actual = _fit(name).predict_quantile(
        **_quantile_inputs(case, where), quantiles=PROBABILITIES, transform=transform
    )
    kind = "quantile" if transform else "uquantile"
    _assert_close(actual.predictions, case[where][kind]["fit"])
    assert actual.quantiles == PROBABILITIES


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("where", ["training", "new"])
@pytest.mark.parametrize("transform", [True, False])
def test_probability_endpoints_match_r_and_allow_nonfinite_predictions(name, where, transform):
    case = CASES[name]
    kind = "quantile" if transform else "uquantile"
    expected = case[where][kind + "_endpoints"]
    native = _fit(name).predict_quantile(
        **_quantile_inputs(case, where), quantiles=[0.0, 1.0], transform=transform
    )
    generic = survival.predict(
        _fit(name),
        None if where == "training" else case["newdata"],
        type=kind,
        p=[0.0, 1.0],
        se_fit=True,
    )
    _assert_close(native.predictions, expected["fit"])
    _assert_close(generic.fit, expected["fit"])
    if case["fixed_scale"] > 0:
        _assert_close(generic.se_fit, expected["se.fit"])
    else:
        # At infinite quantiles, estimated-scale delta-method arithmetic in R
        # gives NaN or Inf depending on its accumulation order. Both are undefined.
        assert np.all(~np.isfinite(generic.se_fit))


@pytest.mark.parametrize("name", [name for name in CASES if name.endswith("_stratified")])
def test_new_multiscale_rows_require_explicit_native_strata(name):
    with pytest.raises(ValueError, match="strata"):
        _fit(name).predict_quantile([[1.0, 0.0]], [0.9])


@pytest.mark.parametrize("strata", [[0], [0, 1, 2], [0, 1, 3], [-1, 0, 1]])
def test_native_new_strata_reject_wrong_length_or_invalid_codes(strata):
    with pytest.raises((ValueError, OverflowError), match="strata|range|negative"):
        _fit("gaussian_stratified").predict_quantile(
            [[1.0, 0.25], [1.0, -0.75], [1.0, 0.5]], [0.9], strata=strata
        )


def test_native_training_strata_can_be_supplied_explicitly():
    fit = _fit("gaussian_stratified")
    implicit = fit.predict_quantile(quantiles=PROBABILITIES)
    explicit = fit.predict_quantile(quantiles=PROBABILITIES, strata=fit.strata)
    _assert_close(explicit.predictions, implicit.predictions)
    with pytest.raises(ValueError, match="strata"):
        fit.predict_quantile(quantiles=PROBABILITIES, strata=[0])


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_native_and_generic_quantiles_reject_invalid_probabilities(probability):
    fit = _fit("gaussian_fixed")
    with pytest.raises(ValueError, match="[Qq]uantile|between"):
        fit.predict_quantile(quantiles=[probability])
    with pytest.raises(ValueError, match="between"):
        survival.predict(fit, type="quantile", p=probability)


@pytest.mark.parametrize(("alias", "canonical"), ALIASES.items())
def test_distribution_aliases_share_native_and_generic_quantile_semantics(alias, canonical):
    name = canonical + "_estimated"
    case = CASES[name]
    fit = _fit(name, alias)
    expected = case["new"]["quantile"]["fit"]
    _assert_close(
        fit.predict_quantile(**_native_inputs(case, "new"), quantiles=PROBABILITIES).predictions,
        expected,
    )
    _assert_close(
        survival.predict(fit, case["newdata"], type="quantile", p=PROBABILITIES), expected
    )


@pytest.mark.parametrize("name", [name for name in CASES if name.endswith("_fixed")])
def test_standalone_prediction_functions_match_r_including_default_student_t(name):
    case = CASES[name]
    rows = [[1.0, value] for value in case["newdata"]["x"]]
    quantile = survival.predict_survreg_quantile(
        rows, case["beta"], case["fixed_scale"], case["distribution"], PROBABILITIES
    )
    if case["distribution"] == "t":
        # The six-argument standalone API has no fitted parms: its t default is df=4.
        scores = (_array(REFERENCE["standalone_t_df4"])[1:-1] - 2.0) / 0.8
        expected = np.array(rows) @ case["beta"]
        _assert_close(quantile.predictions, expected[:, None] + 0.8 * scores)
    else:
        _assert_close(quantile.predictions, case["new"]["quantile"]["fit"])
    point = survival.predict_survreg(
        rows,
        case["beta"],
        case["fixed_scale"],
        case["distribution"],
        var_matrix=case["variance"],
        se_fit=True,
    )
    _assert_close(point.predictions, case["new"]["response"]["fit"])
    _assert_close(point.se, case["new"]["response"]["se.fit"])


@pytest.mark.parametrize("name", [name for name in CASES if name.endswith("_offset")])
@pytest.mark.parametrize("kind", TYPES)
def test_newdata_offsets_have_their_mathematical_effect_despite_r_omission(name, kind):
    case = CASES[name]
    shift = np.array([0.2, -0.3, 0.1])
    newdata = {**case["newdata"], "offset_value": shift.tolist()}
    actual = survival.predict(_fit(name), newdata, type=kind, p=PROBABILITIES, se_fit=True)
    expected = _array(case["new"][kind]["fit"])
    expected_se = _array(case["new"][kind]["se.fit"])
    if kind in ("quantile", "uquantile"):
        shift = shift[:, None]
    if kind in ("response", "quantile") and case["log_response"]:
        expected = expected * np.exp(shift)
        expected_se = expected_se * np.exp(shift)
    else:
        expected = expected + shift
    _assert_close(actual.fit, expected)
    _assert_close(actual.se_fit, expected_se)


@pytest.mark.parametrize(
    ("distribution", "canonical"),
    [
        (name.removesuffix("_estimated"), name.removesuffix("_estimated"))
        for name in CASES
        if name.endswith("_estimated")
    ]
    + [("exponential", "exponential"), ("rayleigh", "rayleigh")]
    + list(ALIASES.items()),
)
def test_sklearn_predictions_use_fitted_distribution_transform_and_quantiles(
    distribution, canonical
):
    from survival.sklearn_compat import AFTEstimator

    name = canonical + "_fixed"
    case = CASES[name]
    fitted = _fit(name, distribution)
    # Install a valid deterministic fitted model so optimizer behavior cannot
    # mask a prediction regression, including forwarding fitted Student-t df=7.
    estimator = AFTEstimator(distribution=distribution)
    estimator.model_ = fitted.fit
    estimator.intercept_ = fitted.location_coefficients[0]
    estimator.coef_ = _array(fitted.location_coefficients[1:])
    estimator.scale_ = fitted.scale
    estimator.n_features_in_ = 1
    estimator.is_fitted_ = True
    x = np.array(case["newdata"]["x"])[:, None]
    _assert_close(estimator.predict(x), case["new"]["response"]["fit"])
    expected = _array(case["new"]["quantile"]["fit"])
    _assert_close(estimator.predict_median(x), expected[:, 1])
    for index, probability in enumerate(PROBABILITIES):
        _assert_close(estimator.predict_quantile(x, probability), expected[:, index])
    for probability, column in [(0.0, 0), (1.0, 1)]:
        endpoint = _array(case["new"]["quantile_endpoints"]["fit"])[:, column]
        _assert_close(estimator.predict_quantile(x, probability), endpoint)
