"""Real-valued response parity with R survival 3.8.11."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

TIME = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.4, 0.8, 1.1, 1.5, 2.0, 2.4, 3.0]
X = [-1.0, -0.5, 0.5, 1.0, -1.0, 0.25, -0.75, 0.75, 1.0, 0.1, -0.2, 0.6]
BINARY_STATUS = [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
INTERVAL_STATUS = [1, 2, 3, 0, 1, 3, 1, 0, 2, 3, 1, 1]
UPPER = [
    time + width
    for time, width in zip(
        TIME, [1.0, 1.0, 0.5, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0], strict=True
    )
]
ROWS = [[1.0, value] for value in X]
IDENTITY_FAMILIES = ["gaussian", "logistic", "t", "extreme"]
LOG_FAMILIES = ["weibull", "exponential", "rayleigh", "lognormal", "loglogistic"]

# R: survreg(y ~ x, dist=distribution,
#            control=survreg.control(maxiter=100, rel.tolerance=1e-11))
# The responses use the exact, right, left, and interval Surv constructors
# with the arrays above; Student-t uses R's default four degrees of freedom.
# Each tuple is (intercept, slope, scale, fitted log likelihood).
R_FITS = {
    "gaussian": {
        "exact": (0.4680404916847432, 0.7780187997107737, 1.403986679869987, -21.09905221789439),
        "right": (0.8666750049934697, 0.601550247195041, 1.537444925124804, -18.34872304802794),
        "left": (0.09966852936271958, 0.8938012469369623, 1.598589388570989, -18.90006135274993),
        "interval": (0.689464162391059, 1.299182304134374, 1.565770217311491, -19.4395085719919),
    },
    "logistic": {
        "exact": (0.4719420598379421, 0.7527476739220237, 0.8578645074664432, -21.68436397909667),
        "right": (0.9142435796297704, 0.5824213052083148, 0.9462130161186608, -18.81521658195536),
        "left": (0.05270826651738914, 0.8420186448951206, 0.9811157110753653, -19.3150541154736),
        "interval": (0.7687275035410731, 1.274034221201934, 0.9770397044918034, -19.84299557509099),
    },
    "t": {
        "exact": (0.4757771287092137, 0.7411037712102994, 1.289806245219696, -22.06761286379186),
        "right": (0.9296687629759162, 0.5763614505790255, 1.426038044409265, -19.11523377850178),
        "left": (0.04052602342417064, 0.8154990658869581, 1.474347193897174, -19.58552245411927),
        "interval": (0.8180316433715416, 1.261953890680037, 1.474494764467731, -20.07267042562637),
    },
    "extreme": {
        "exact": (1.164519867114599, 0.8465985776253373, 1.269749588694016, -21.48854966304163),
        "right": (1.549768633063684, 0.6625151641641263, 1.246218904854497, -18.37172906722343),
        "left": (0.8529592925339171, 0.9645921110031856, 1.565190391629287, -19.19639095556385),
        "interval": (1.413398225518165, 1.436767718189621, 1.30342121168541, -18.85354974444285),
    },
}


def _response(censoring):
    data = {"time": TIME, "status": BINARY_STATUS, "x": X}
    if censoring == "exact":
        return "Surv(time) ~ x", data, [1] * len(TIME), None
    if censoring == "right":
        return "Surv(time, status) ~ x", data, BINARY_STATUS, None
    if censoring == "left":
        return "Surv(time, status, type='left') ~ x", data, [2 - s for s in BINARY_STATUS], None
    data = {**data, "upper": UPPER, "status": INTERVAL_STATUS}
    if censoring == "interval":
        return "Surv(time, upper, status, type='interval') ~ x", data, INTERVAL_STATUS, UPPER
    left = [
        -math.inf if status == 2 else time
        for time, status in zip(TIME, INTERVAL_STATUS, strict=True)
    ]
    right = [
        math.inf if status == 0 else upper if status == 3 else time
        for time, upper, status in zip(TIME, UPPER, INTERVAL_STATUS, strict=True)
    ]
    return (
        "Surv(left, right, type='interval2') ~ x",
        {"left": left, "right": right, "x": X},
        INTERVAL_STATUS,
        UPPER,
    )


@pytest.mark.parametrize("distribution", IDENTITY_FAMILIES)
@pytest.mark.parametrize("censoring", ["exact", "right", "left", "interval", "interval2"])
def test_signed_survreg_fits_match_r(distribution, censoring):
    formula, data, status, upper = _response(censoring)
    native = survival.regression.survreg(
        TIME,
        status,
        ROWS,
        time2=upper,
        distribution=distribution,
        max_iter=100,
        eps=1e-11,
    )
    facade = survival.survreg(formula, data=data, dist=distribution, max_iter=100, eps=1e-11)
    expected = R_FITS[distribution]["interval" if censoring == "interval2" else censoring]
    # Existing Gaussian approximation and Student-t fitting differences are
    # ~1e-6 in log likelihood and ~6e-8 in coefficients, respectively.
    tolerance = {"gaussian": 2e-6, "t": 1e-7}.get(distribution, 2e-8)
    for fit in (native, facade):
        assert fit.convergence_flag == 0
        assert fit.location_coefficients == pytest.approx(expected[:2], rel=0, abs=tolerance)
        assert fit.scale == pytest.approx(expected[2], rel=0, abs=tolerance)
        assert fit.log_likelihood == pytest.approx(expected[3], rel=0, abs=tolerance)
        assert fit.time == pytest.approx(TIME)
        assert fit.status == status
        assert fit.predict(ROWS).predictions == pytest.approx(fit.linear_predictors)


@pytest.mark.parametrize(
    ("distribution", "canonical"),
    [("normal", "gaussian"), ("student-t", "t"), ("extreme_value", "extreme")],
)
def test_signed_survreg_distribution_aliases(distribution, canonical):
    fit = survival.regression.survreg(
        TIME, [1] * len(TIME), ROWS, distribution=distribution, max_iter=100, eps=1e-11
    )
    assert fit.location_coefficients == pytest.approx(R_FITS[canonical]["exact"][:2], abs=2e-6)


def test_signed_student_t_quantiles_use_the_fitted_degrees_of_freedom():
    fit = survival.regression.survreg(
        TIME,
        INTERVAL_STATUS,
        ROWS,
        time2=UPPER,
        distribution="t",
        distribution_parameter=7.0,
        max_iter=100,
        eps=1e-11,
    )
    # R's same mixed-censoring fit with parms=7 and predict(type="quantile").
    assert fit.distribution_parameters == [7.0]
    assert fit.location_coefficients == pytest.approx(
        [0.759161693137288, 1.27983122921885], rel=0, abs=2e-8
    )
    assert fit.scale == pytest.approx(1.51450433503582, rel=0, abs=2e-8)
    assert fit.log_likelihood == pytest.approx(-19.8101659937541, rel=0, abs=2e-8)
    predictions = fit.predict_quantile([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], [0.1, 0.5, 0.9])
    expected = [
        [-2.66357795825416588, -0.52066953608156363, 1.6222388860910393],
        [-1.38374672903531470, 0.75916169313728776, 2.9020701153098907],
        [-0.10391549981646309, 2.03899292235613938, 4.1819013445287423],
    ]
    for actual, reference in zip(predictions.predictions, expected, strict=True):
        assert actual == pytest.approx(reference, rel=0, abs=2e-8)


def _assert_values_close(actual, expected):
    if expected and isinstance(expected[0], list | tuple):
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert actual_row == pytest.approx(expected_row, rel=1e-8, abs=1e-8, nan_ok=True)
    else:
        assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8, nan_ok=True)


@pytest.mark.parametrize("distribution", IDENTITY_FAMILIES)
def test_signed_survreg_residual_apis_are_translation_invariant(distribution):
    # Binary fractions and location parameters -1.5/+1.5 keep the finite-
    # difference perturbations identical after translation. Fixed parameters
    # isolate response-domain behavior from optimizer and derivative rounding.
    time = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]
    status = [1, 2, 3, 0, 1, 3, 1, 3, 1]
    rows = [[1.0, -0.5 + index * 0.125] for index in range(len(time))]
    fits = [
        survival.regression.survreg(
            [value + shift for value in time],
            status,
            rows,
            time2=[value + shift + 0.25 for value in time],
            distribution=distribution,
            initial_beta=[shift - 1.5, 0.0],
            fixed_scale=1.0,
            max_iter=0,
        )
        for shift in (0.0, 3.0)
    ]
    original, shifted = fits
    assert shifted.location_coefficients == pytest.approx(
        [original.location_coefficients[0] + 3.0, original.location_coefficients[1]], abs=1e-8
    )
    _assert_values_close(original.variance_matrix, shifted.variance_matrix)
    scalar_types = ("response", "deviance", "working", "ldcase", "ldresp", "ldshape")
    for residual_type in scalar_types:
        low_level = [
            survival.residuals_survreg(
                fit.time,
                fit.status,
                fit.linear_predictors,
                fit.scale,
                fit.distribution,
                residual_type=residual_type,
                time2=fit.time2,
                distribution_parameter=4.0 if distribution == "t" else None,
            ).residuals
            for fit in fits
        ]
        _assert_values_close(*low_level)
        _assert_values_close(
            original.residuals(residual_type).residuals,
            shifted.residuals(residual_type).residuals,
        )
    for residual_type in (*scalar_types, "matrix", "dfbeta", "dfbetas"):
        _assert_values_close(
            survival.r_api.residuals(original, type=residual_type),
            survival.r_api.residuals(shifted, type=residual_type),
        )
    _assert_values_close(original.dfbeta(), shifted.dfbeta())
    direct_dfbeta = [
        survival.dfbeta_survreg(
            fit.time,
            fit.status,
            fit.covariates,
            fit.linear_predictors,
            fit.scale,
            [row[:2] for row in fit.variance_matrix[:2]],
            fit.distribution,
            time2=fit.time2,
            distribution_parameter=4.0 if distribution == "t" else None,
        )
        for fit in fits
    ]
    _assert_values_close(*direct_dfbeta)


@pytest.mark.parametrize("distribution", IDENTITY_FAMILIES)
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_identity_survreg_still_rejects_nonfinite_responses(distribution, invalid):
    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.survreg(
            [invalid, 0.0, 1.0], [1, 1, 1], [[1.0]] * 3, distribution=distribution
        )
    with pytest.raises(ValueError, match="non-finite interval endpoint"):
        survival.regression.survreg(
            [-2.0, 0.0, 1.0],
            [3, 1, 1],
            [[1.0]] * 3,
            time2=[invalid, 0.0, 1.0],
            distribution=distribution,
        )


@pytest.mark.parametrize("distribution", LOG_FAMILIES)
@pytest.mark.parametrize("time", [-1.0, 0.0])
@pytest.mark.parametrize("status", [0, 1, 2])
def test_log_survreg_still_requires_positive_responses(distribution, time, status):
    with pytest.raises(ValueError, match="must be positive"):
        survival.regression.survreg(
            [time, 1.0, 2.0],
            [status, 1, 1],
            [[1.0]] * 3,
            time2=[0.5, 1.0, 2.0],
            distribution=distribution,
        )


@pytest.mark.parametrize("distribution", IDENTITY_FAMILIES)
@pytest.mark.parametrize("upper", [-2.0, -3.0])
def test_signed_survreg_still_requires_ordered_intervals(distribution, upper):
    with pytest.raises(ValueError, match="greater than time"):
        survival.regression.survreg(
            [-2.0, 0.0, 1.0],
            [3, 1, 1],
            [[1.0]] * 3,
            time2=[upper, 0.0, 1.0],
            distribution=distribution,
        )


def _call_residual_api(api, time, upper, distribution, scale=1.0, status=3):
    common = {
        "time": [time, 1.0, 2.0],
        "status": [status, 1, 0],
        "linear_pred": [0.0, 0.0, 0.0],
        "scale": scale,
        "distribution": distribution,
        "time2": [upper, 1.0, 2.0],
    }
    if api == "dfbeta":
        return survival.dfbeta_survreg(**common, covariates=[[1.0]] * 3, var_matrix=[[1.0]])
    if api == "matrix":
        return survival.survreg_residual_matrix(**common)
    return survival.residuals_survreg(**common)


@pytest.mark.parametrize("api", ["scalar", "matrix", "dfbeta"])
@pytest.mark.parametrize(
    ("time", "upper", "error"),
    [
        (math.nan, 1.0, "time contains non-finite"),
        (-1.0, math.inf, "non-finite interval endpoint"),
        (-1.0, -2.0, "greater than time"),
    ],
)
def test_identity_residual_apis_preserve_response_validation(api, time, upper, error):
    with pytest.raises(ValueError, match=error):
        _call_residual_api(api, time, upper, "gaussian")


@pytest.mark.parametrize("api", ["scalar", "matrix", "dfbeta"])
@pytest.mark.parametrize("distribution", LOG_FAMILIES)
@pytest.mark.parametrize("time", [-1.0, 0.0])
def test_log_residual_apis_still_require_positive_responses(api, distribution, time):
    with pytest.raises(ValueError, match="must be positive"):
        _call_residual_api(api, time, 0.5, distribution, status=1)


@pytest.mark.parametrize("api", ["scalar", "matrix", "dfbeta"])
@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_identity_residual_apis_still_require_positive_finite_scale(api, scale):
    with pytest.raises(ValueError, match="scale must be a finite positive value"):
        _call_residual_api(api, -1.0, 0.0, "gaussian", scale=scale)
