"""Random-draw count coercion against stock R 4.5.3 / survival 3.8-12."""

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "rsurvreg_count_reference.json").read_text()
)
CASES = {case["name"]: case for case in REFERENCE["cases"]}


def count_input(name):
    return {
        "integer0": 0,
        "integer1": 1,
        "integer3": 3,
        "numeric3": 3.0,
        "fractional": 2.5,
        "fractional_small": 0.75,
        "negative_fraction_small": -0.75,
        "negative_integer": -1,
        "negative_fraction": -1.5,
        "length2": [4, 7],
        "ignored_invalid": [-1, None, math.inf],
        "empty": [],
        "null": None,
        "na_integer": None,
        "na_real": math.nan,
        "nan": math.nan,
        "inf": math.inf,
        "negative_inf": -math.inf,
        "true": True,
        "false": False,
        "bool_vector": [True, False, None],
        "string3": "3",
        "stringfraction": "2.5",
        "stringhex": "0x1.8p1",
        "stringwhitespace": " +2.5 ",
        "bad_string": "bad",
        "bad_string_separator": "1_0",
        "strings": ["bad", "also_bad"],
        # Python flat lists are R atomic vectors; a nested singleton is an R list.
        "list1": [[3]],
        "list2": [[-1], [None]],
        "list0": [],
        "matrix": np.array([[1, 3], [2, 4]]),
        "matrix1": np.array([[2.5]]),
        "matrix0": np.empty((0, 3)),
        "factor": r._r_factor(["3"], ["3", "5"]),
        "factor_missing": r._r_factor([None], ["3", "5"]),
        "factor_vector": r._r_factor(["5", "3", None], ["3", "5"]),
        "complex": 3 + 2j,
        "complex_fraction": 2.5 + 0j,
        "complex_vector": [1 + 1j, 2 + 1j],
    }[name]


def callback_distribution(calls):
    def quantile(probabilities):
        calls.append(probabilities.tolist())
        return probabilities

    # Density and init are required constructor callbacks, but sampling only
    # invokes quantile. The R generator registers the same quantile separately.
    return {
        "name": "Count callback oracle",
        "density": lambda z: np.column_stack(
            (np.full(len(z), 0.5), np.full(len(z), 0.5), np.ones(len(z)), z * 0, z * 0)
        ),
        "init": lambda y, weights: [0, 1],
        "deviance": lambda y, scale: (np.zeros(len(y)), np.zeros(len(y))),
        "quantile": quantile,
    }


def assert_stock_result(n, expected, distribution="weibull"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if expected["error"] is not None:
            # Preserve the facade's existing negative-count message.
            with pytest.raises(ValueError, match="invalid arguments|n must be non-negative"):
                r.rsurvreg(n, 0, distribution=distribution, seed=REFERENCE["metadata"]["seed"])
        else:
            actual = r.rsurvreg(n, 0, distribution=distribution, seed=REFERENCE["metadata"]["seed"])
            assert isinstance(actual, list)
            assert len(actual) == expected["length"]
            assert expected["dim"] is None
            assert actual == pytest.approx(expected["values"], rel=1e-14, abs=1e-15)
    assert [str(warning.message) for warning in caught] == expected["warnings"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("custom", [False, True], ids=["builtin", "callback"])
def test_count_semantics_against_stock_r(case, custom):
    calls = []
    distribution = callback_distribution(calls) if custom else "weibull"
    assert_stock_result(
        count_input(case["name"]), case["custom" if custom else "rsurvreg"], distribution
    )
    if custom:
        assert calls == case["quantile_calls"]


@pytest.mark.parametrize("name", ["numeric3", "fractional", "fractional_small", "true", "string3"])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (1, 1, 1)])
def test_numpy_singletons_use_the_value(name, shape):
    n = np.asarray(count_input(name)).reshape(shape)
    assert_stock_result(n, CASES[name]["rsurvreg"])


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1), (1, 1, 3)])
def test_numpy_vectors_ignore_invalid_values_and_return_flat_draws(shape):
    n = np.array([-1, math.nan, math.inf]).reshape(shape)
    assert_stock_result(n, CASES["ignored_invalid"]["rsurvreg"])


def test_numpy_strided_vectors_and_empty_dimensions():
    n = np.array([-1, 100, math.nan, 100, math.inf, 100])[::2]
    assert_stock_result(n, CASES["ignored_invalid"]["rsurvreg"])
    # Zero elements must not require scanning the large remaining dimensions.
    assert_stock_result(np.empty((0, 3, 1 << 28)), CASES["empty"]["rsurvreg"])


@pytest.mark.parametrize(
    ("values", "name"), [([], "empty"), ([2.5], "fractional"), ([4, 7], "length2")]
)
def test_one_shot_iterator_is_materialized_once(values, name):
    visited = []

    def n():
        for value in values:
            visited.append(value)
            yield value

    assert_stock_result(n(), CASES[name]["rsurvreg"])
    assert visited == values


def test_vector_values_are_not_coerced():
    class Uncoercible:
        def __float__(self):
            raise AssertionError("vector elements must not be converted")

    assert_stock_result([Uncoercible(), Uncoercible()], CASES["length2"]["rsurvreg"])


@pytest.mark.parametrize("custom", [False, True], ids=["builtin", "callback"])
@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE["recycling"] if case["n"] == 0],
    ids=lambda case: case["name"],
)
def test_zero_draws_accept_valid_unused_mean_and_scale_vectors(case, custom):
    calls = []
    distribution = callback_distribution(calls) if custom else "weibull"
    actual = r.rsurvreg(
        case["n"],
        case["mean"],
        case["scale"],
        distribution,
        seed=REFERENCE["metadata"]["seed"],
    )
    expected = case["custom" if custom else "rsurvreg"]
    assert actual == expected["values"] == []
    if custom:
        assert calls == case["quantile_calls"] == [[]]


@pytest.mark.parametrize("n", [0, [], np.empty((0, 3)), 0.75])
def test_zero_draws_preserve_distribution_and_seed_validation(n):
    with pytest.raises(ValueError, match="Distribution not found"):
        r.rsurvreg(n, [1, 2, 3], [0.5, 2], distribution="missing", seed=123)
    with pytest.raises(ValueError, match="supplied seed is not a valid integer"):
        r.rsurvreg(n, [1, 2, 3], [0.5, 2], seed=-(2**31))
    with pytest.raises(ValueError, match=re.escape("seed must be an integer")):
        r.rsurvreg(n, [1, 2, 3], [0.5, 2], seed=1.5)
    with pytest.raises(TypeError, match=re.escape("seed must be an integer")):
        r.rsurvreg(n, [1, 2, 3], [0.5, 2], seed=True)


@pytest.mark.parametrize(
    ("mean", "scale"), [(math.inf, 1), (0, -1), ([1, math.inf], 1), (0, [1, -1])]
)
def test_zero_draws_accept_unused_nonfinite_locations_and_nonpositive_scales(mean, scale):
    assert r.rsurvreg(0, mean, scale, seed=123) == []
