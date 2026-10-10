"""Stock survival DPQR parameter recycling, warning stages and missing values."""

import json
import math
import re
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

r = survival.r
core = survival._survival
FIXTURE = Path(__file__).parent / "fixtures" / "survreg_dpqr_parms_reference.json"
REFERENCE = json.loads(FIXTURE.read_text())
CASES = REFERENCE["cases"]
NA_REAL = struct.unpack(">d", bytes.fromhex("7ff80000000007a2"))[0]
SPECIAL = {"NA": NA_REAL, "NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}


def numbers(values):
    return [SPECIAL[value] if isinstance(value, str) else value for value in values]


def is_na(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return math.isnan(value) and bits & 0x0007FFFFFFFFFFFF == 0x7A2


def inputs(case, interface):
    mode = case["parms_mode"]
    if mode == "omitted" or mode == "null":
        return None
    if mode == "character":
        return case["parms"]
    values = numbers(case["parms"])
    if interface == "facade" and case["parm_names"] is not None:
        return dict(zip(case["parm_names"], values, strict=True))
    return values


def run_case(case, interface):
    parms = inputs(case, interface)
    method = case["method"]
    query = case["query"] if method == "rsurvreg" else numbers(case["query"])
    means = numbers(case["mean"])
    scales = numbers(case["scale"])
    if interface == "query_object":
        obj = core.SurvregDistribution.for_query(case["distribution"], parms)
        if method == "rsurvreg":
            return obj.sample(query, means, scales, seed=123)
        return getattr(
            obj,
            {"dsurvreg": "pdf_values", "psurvreg": "cdf_values", "qsurvreg": "quantile_values"}[
                method
            ],
        )(query, means, scales)
    function = getattr(core if interface == "native" else r, method)
    kwargs = {"seed": 123} if method == "rsurvreg" else {}
    if case["parms_mode"] != "omitted":
        kwargs["parms"] = parms
    return function(query, means, scales, case["distribution"], **kwargs)


def supported(case, interface):
    # R NULL is distinct from the documented Python None meaning omitted parms.
    if case["distribution"] == "t" and case["parms_mode"] == "null":
        return False
    # The numeric native ABI is stricter than Python's ignored arbitrary parms.
    return interface == "facade" or case["parms_mode"] != "character"


PARAMETERS = [
    (case, interface)
    for case in CASES
    for interface in ("native", "facade", "query_object")
    if supported(case, interface)
]


@pytest.mark.parametrize(
    ("case", "interface"),
    PARAMETERS,
    ids=[f"{case['name']}/{interface}" for case, interface in PARAMETERS],
)
def test_parameter_queries_match_stock(case, interface):
    expected = case["expected"]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if expected["error"] is not None:
            with pytest.raises((ValueError, RuntimeError), match=re.escape(expected["error"])):
                run_case(case, interface)
            actual = None
        else:
            actual = run_case(case, interface)
    expected_warnings = [
        event["message"] for event in expected["events"] if event["kind"] == "warning"
    ]
    assert [str(warning.message) for warning in recorded] == expected_warnings
    if actual is None:
        return
    values = numbers(expected["values"])
    assert len(actual) == len(values)
    for actual_value, expected_value in zip(actual, values, strict=True):
        if math.isnan(expected_value):
            assert math.isnan(actual_value)
            assert is_na(actual_value) == is_na(expected_value)
        elif math.isinf(expected_value):
            assert actual_value == expected_value
        else:
            assert actual_value == pytest.approx(expected_value, rel=2e-12, abs=2e-14)


@pytest.mark.parametrize("parms", [[], [0], [-1], [2], [math.inf], [math.nan], [NA_REAL], [3, 4]])
def test_query_parameters_do_not_relax_fitting_definitions(parms):
    with pytest.raises(ValueError, match="Student-t|[Dd]egrees|distribution parameters"):
        core.SurvregDistribution("t", parms)
    with pytest.raises(ValueError, match="Student-t|[Dd]egrees|distribution parameters"):
        r.survreg("Surv(time, status) ~ age", survival.datasets.load_lung(), dist="t", parms=parms)


def test_constructed_t_default_remains_df_four():
    dist = core.SurvregDistribution("t")
    assert dist.parms == [4]
    assert dist.dtest() == []
    np.testing.assert_allclose(
        dist.quantile_values([0.1, 0.5, 0.9], [0], [1]),
        core.qsurvreg([0.1, 0.5, 0.9], [0], [1], "t", [4]),
    )


@pytest.mark.parametrize("method", ["dsurvreg", "psurvreg", "qsurvreg", "rsurvreg"])
def test_omitted_t_parms_error_occurs_after_stock_preceding_warnings(method):
    name = f"omitted/{method}/2/3/5"
    case = next(case for case in CASES if case["name"] == name)
    messages = [
        event["message"] for event in case["expected"]["events"] if event["kind"] == "warning"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if messages:
            with pytest.raises(RuntimeWarning, match=re.escape(messages[0])):
                run_case(case, "facade")
        else:
            with pytest.raises(ValueError, match='argument "parms" is missing'):
                run_case(case, "facade")


@pytest.mark.parametrize("interface", ["native", "facade", "distribution"])
@pytest.mark.parametrize("unequal", [False, True])
def test_multiplication_missing_kinds_follow_stock_operand_lengths(interface, unequal):
    # Stock R 4.5.3 / survival 3.8-12:
    # qsurvreg(c(NA, NaN), 0, c(NaN, NA), "gaussian") -> c(NA, NaN)
    # qsurvreg(c(NA, NaN), 0, c(NaN, NA, 1), "gaussian") -> c(NaN, NA, NA)
    # The unequal case also emits one recycling warning.
    probabilities = [NA_REAL, math.nan]
    scales = [math.nan, NA_REAL, 1.0] if unequal else [math.nan, NA_REAL]

    def query():
        if interface == "distribution":
            return core.SurvregDistribution("gaussian").quantile_values(probabilities, [0], scales)
        api = core if interface == "native" else r
        return api.qsurvreg(probabilities, [0], scales, "gaussian")

    if unequal:
        with pytest.warns(RuntimeWarning, match="longer object length"):
            result = query()
    else:
        result = query()
    assert all(math.isnan(value) for value in result)
    assert [is_na(value) for value in result] == ([False, True, True] if unequal else [True, False])


@pytest.mark.parametrize("interface", ["native", "facade", "query_object"])
@pytest.mark.parametrize(
    ("distribution", "parms"),
    [pytest.param("gaussian", None, id="gaussian"), pytest.param("t", [4, 5], id="t-recycled")],
)
@pytest.mark.parametrize(
    ("probabilities", "means", "expected_na", "warning_count"),
    [
        pytest.param([NA_REAL, math.nan], [math.nan, NA_REAL], [True, False], 0, id="equal"),
        pytest.param(
            [NA_REAL, math.nan], [math.nan, NA_REAL, 0], [False, True, True], 1, id="longer-mean"
        ),
        pytest.param(
            [NA_REAL, math.nan, NA_REAL],
            [math.nan, NA_REAL],
            [False, True, False],
            1,
            id="longer-query",
        ),
        pytest.param([NA_REAL, math.nan], [math.nan], [True, False], 0, id="scalar-mean"),
        pytest.param([NA_REAL], [math.nan, NA_REAL], [False, True], 0, id="scalar-query"),
    ],
)
def test_addition_missing_kinds_follow_stock_operand_lengths(
    interface, distribution, parms, probabilities, means, expected_na, warning_count
):
    # Independent stock survival 3.8-12: equal-length addition and scalar
    # means preserve the scaled quantile's NA/NaN kind; unequal nonscalar
    # means supply their own kind. qt(NA, c(4, 5)) first expands to c(NA, NA),
    # making the scalar-query Student-t addition equal-length too.
    if distribution == "t" and len(probabilities) == 1:
        expected_na = [True, True]

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if interface == "query_object":
            result = core.SurvregDistribution.for_query(distribution, parms).quantile_values(
                probabilities, means, [1]
            )
        else:
            api = core if interface == "native" else r
            result = api.qsurvreg(probabilities, means, [1], distribution, parms)

    assert [str(warning.message) for warning in recorded] == [
        "longer object length is not a multiple of shorter object length"
    ] * warning_count
    assert all(math.isnan(value) for value in result)
    assert [is_na(value) for value in result] == expected_na
