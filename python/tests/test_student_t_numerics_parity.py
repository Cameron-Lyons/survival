"""Shared Student-t numerics against R and independent high-precision references.

The generator records R's answers as well as mathematical quantile references;
R qt is inaccurate for several median and extreme-tail cases. No test imports
mpmath or invokes R. Relative comparisons retain representable rare probabilities.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "student_t_numerics_reference.json").read_text()
)
CASES = REFERENCE["cases"]
CASE_IDS = [f"df={case['df']:g}" for case in CASES]


def _call(values, df, kind, api="native", *, mean=0.0, scale=1.0):
    if api == "native":
        return survival._survival.survreg_distribution(
            values, [mean] * len(values), [scale] * len(values), "t", kind, df
        )
    helper = {
        "density": survival.dsurvreg,
        "distribution": survival.psurvreg,
        "quantile": survival.qsurvreg,
    }[kind]
    return helper(values, mean=mean, scale=scale, distribution="t", parms=df)


def _assert_relative(actual, expected, *, rtol, label):
    expected = float(expected)
    if not math.isfinite(expected) or expected == 0:
        assert actual == expected, label
        return
    assert math.isfinite(actual), label
    assert actual != 0, f"Representable nonzero reference was rounded to zero: {label}"
    assert math.copysign(1.0, actual) == math.copysign(1.0, expected), label
    # Subnormal results necessarily have coarse relative spacing. Permit two
    # output ulps, but never permit zero when the reference is representable.
    tolerance = max(rtol * abs(expected), 2 * math.ulp(expected))
    assert abs(actual - expected) <= tolerance, (
        f"{label}: actual={actual!r}, expected={expected!r}, "
        f"relative_error={(actual - expected) / expected!r}"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_density_matches_r_without_erasing_subnormal_values(case, api):
    values = [float(row["x"]) for row in case["points"]]
    actual = _call(values, case["df"], "density", api)
    for row, value in zip(case["points"], actual, strict=True):
        _assert_relative(
            value, row["pdf"], rtol=3e-12, label=f"{api} dt({row['x']}, df={case['df']})"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_cdf_preserves_direct_lower_tails_and_center_resolution(case, api):
    values = [float(row["x"]) for row in case["points"]]
    actual = _call(values, case["df"], "distribution", api)
    assert actual == sorted(actual)
    for row, value in zip(case["points"], actual, strict=True):
        expected = float(row["cdf"])
        label = f"{api} pt({row['x']}, df={case['df']})"
        if expected < 0.25:
            _assert_relative(value, expected, rtol=3e-11, label=label)
        else:
            assert abs(value - expected) <= 2 * math.ulp(expected), label
        x = float(row["x"])
        if 0 < abs(x) <= 1e-9:
            central_mass = case["density_zero_mp"] * x
            rounding = math.ulp(expected)
            assert abs((value - 0.5) - central_mass) <= rounding, label
            if abs(central_mass) > rounding / 2:
                assert value != 0.5, f"Lost representable centered mass: {label}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_quantiles_match_independent_center_and_tail_references(case, api):
    probabilities = [row["p"] for row in case["quantiles"]]
    actual = _call(probabilities, case["df"], "quantile", api)
    assert actual == sorted(actual)
    for row, value in zip(case["quantiles"], actual, strict=True):
        _assert_relative(
            value,
            row["expected"],
            rtol=3e-11,
            label=f"{api} qt({row['p']}, df={case['df']}); {row['source']}",
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_student_t_inverse_roundtrips_rare_tails_and_centered_mass(case):
    # Supplemental to independent references above: roundtrips alone would miss
    # a common error shared by the CDF and inverse.
    probabilities = [row["p"] for row in case["quantiles"] if 0 < row["p"] < 0.5]
    quantiles = _call(probabilities, case["df"], "quantile")
    cdf = _call(quantiles, case["df"], "distribution")
    for p, q, value in zip(probabilities, quantiles, cdf, strict=True):
        if not math.isfinite(q):
            continue  # The independently verified quantile exceeds float64 range.
        if p <= 0.25:
            _assert_relative(value, p, rtol=1e-10, label=f"pt(qt({p}, df={case['df']}))")
        else:
            assert abs((value - 0.5) - (p - 0.5)) <= math.ulp(p)
    center_p = [0.5 - 2.0**-30, 0.5 + 2.0**-30]
    center_q = _call(center_p, case["df"], "quantile")
    for p, q in zip(center_p, center_q, strict=True):
        _assert_relative(
            q * case["density_zero_mp"],
            p - 0.5,
            rtol=3e-11,
            label=f"centered mass df={case['df']}, p={p}",
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_student_t_symmetry_uses_exactly_representable_complements(case):
    probabilities = [0.125, 0.25, 0.5 - 2.0**-30]
    quantiles = _call(probabilities + [1 - p for p in probabilities], case["df"], "quantile")
    for lower, upper in zip(quantiles[:3], quantiles[3:], strict=True):
        _assert_relative(lower, -upper, rtol=3e-13, label=f"quantile symmetry df={case['df']}")
    x = [1e-15, 1e-9, 0.1, 1.0, 20.0, 1e155, 1e200]
    density = _call(x + [-value for value in x], case["df"], "density")
    assert density[: len(x)] == density[len(x) :]


@pytest.mark.parametrize("api", ["native", "facade"])
@pytest.mark.parametrize("kind", ["density", "distribution"])
def test_student_t_nan_inputs_propagate_and_infinite_inputs_have_limits(api, kind):
    actual = _call([float("nan"), -math.inf, math.inf], 7.0, kind, api)
    assert math.isnan(actual[0])
    assert actual[1:] == ([0.0, 0.0] if kind == "density" else [0.0, 1.0])


@pytest.mark.parametrize("api", ["native", "facade"])
@pytest.mark.parametrize("df", [0.0, -1.0, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("kind", ["density", "distribution", "quantile"])
def test_student_t_public_helpers_reject_invalid_degrees_of_freedom(api, df, kind):
    with pytest.raises(ValueError, match="positive finite|non-finite"):
        _call([0.5], df, kind, api)


@pytest.mark.parametrize("api", ["native", "facade"])
@pytest.mark.parametrize("p", [-0.1, 1.1, math.nan, math.inf])
def test_student_t_quantiles_reject_invalid_probabilities(api, p):
    with pytest.raises(ValueError, match="between"):
        _call([p], 7.0, "quantile", api)


def test_fitted_student_t_quantiles_reuse_df7_kernel_at_center_and_extreme_tails():
    case = next(case for case in CASES if case["df"] == 7.0)
    x = [-1.0, 0.0, 1.0] * 4
    residual = [-0.96] * 3 + [0.96] * 3
    times = [10.0 + 0.2 * value + residual[index % 6] for index, value in enumerate(x)]
    fit = survival.regression.survreg(
        time=times,
        status=[1.0] * len(times),
        covariates=[[1.0, value] for value in x],
        distribution="t",
        distribution_parameter=7.0,
        initial_beta=[10.0, 0.2, math.log(0.8)],
        max_iter=0,
    )
    probabilities = [row["p"] for row in case["quantiles"]]
    standardized = np.array([float(row["expected"]) for row in case["quantiles"]])
    # A zero-location design keeps tiny center quantiles visible instead of
    # rounding them away when added to the training intercept.
    native = fit.predict_quantile([[0.0, 0.0]], probabilities).predictions[0]
    facade = survival.predict(fit, [[0.0, 0.0]], type="quantile", p=probabilities)[0]
    for api, actual in [("native", native), ("facade", facade)]:
        for p, value, expected in zip(probabilities, actual, 0.8 * standardized, strict=True):
            _assert_relative(value, expected, rtol=3e-11, label=f"fitted {api} t(df=7), p={p}")
    training = fit.predict_quantile(quantiles=probabilities).predictions
    for lp, actual in zip(fit.linear_predictors, training, strict=True):
        for p, value, expected in zip(probabilities, actual, lp + 0.8 * standardized, strict=True):
            _assert_relative(value, expected, rtol=3e-11, label=f"training t(df=7), p={p}")


@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_standardization_avoids_subtraction_overflow_when_z_is_finite(api):
    case = next(case for case in CASES if case["df"] == 7.0)
    reference = next(row for row in case["points"] if row["x"] == 2.0)
    # (1e308 - (-1e308))/1e308 is exactly two before binary64 rounding.
    for kind, expected in [
        ("distribution", reference["cdf"]),
        ("density", reference["pdf"] / 1e308),
    ]:
        actual = _call([1e308], 7.0, kind, api, mean=-1e308, scale=1e308)[0]
        _assert_relative(actual, expected, rtol=3e-12, label=f"{api} finite z=2 {kind}")


@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_density_rescaling_recovers_representable_cauchy_density(api):
    # f(x;0,s) = s/[pi*(s*s+x*x)]. Its standardized density underflows,
    # while the requested density s/pi remains representable.
    actual = _call([1.0], 1.0, "density", api, scale=1e-200)[0]
    _assert_relative(
        actual, 1e-200 / math.pi, rtol=3e-12, label=f"{api} Cauchy density after rescaling"
    )


@pytest.mark.parametrize("api", ["native", "facade"])
def test_student_t_quantile_affine_transform_avoids_intermediate_product_overflow(api):
    # Standard Cauchy q(.875)=1+sqrt(2), so location cancels the unit term.
    actual = _call([0.875], 1.0, "quantile", api, mean=-1e308, scale=1e308)[0]
    _assert_relative(
        actual, math.sqrt(2.0) * 1e308, rtol=3e-11, label=f"{api} finite affine Cauchy quantile"
    )
