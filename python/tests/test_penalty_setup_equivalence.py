"""The vectorized penalty-term setup against reference copies of the per-element code it
replaced: the ridge variances, pspline knots, frailty levels and design columns must be
the same values bit for bit, and fail the same way, on randomized inputs with missing
values, signed zeros, infinities, large integers and mixed types."""

import math
import re

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

from survival.r import _fit as model_frame  # noqa: E402
from survival.r import _penalties as penalties  # noqa: E402
from survival.r._coerce import _is_missing_value, _strata_level_sort_key  # noqa: E402
from survival.r._types import _CovariateTerm  # noqa: E402


def reference_observed(values):
    return [float(value) for value in values if not _is_missing_value(value)]


def reference_r_var(values):
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return math.nan
    if all(value == values[0] for value in values):
        return 0.0
    array = np.asarray(values, dtype=np.longdouble)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean = np.sum(array) / len(values)
        mean += np.sum(array - mean) / len(values)
        return float(np.sum((array - mean) ** 2) / (len(values) - 1))


def reference_levels(values):
    present = {value for value in values if not _is_missing_value(value)}
    return tuple(sorted(present, key=_strata_level_sort_key))


def reference_columns(spec, values):
    x = values[spec.columns[0]]
    if spec.kind == "pspline":
        basis = penalties._core.pspline_basis(
            [float(value) for value in x], spec.nterm, spec.degree, spec.boundary
        ).basis
        if spec.combine is not None:
            basis = penalties._combine_basis(basis, spec.combine)
        first = 0 if spec.intercept else 1
        if not basis:
            return []
        return np.asarray(basis, dtype=float)[:, first:].T.tolist()
    lookup = {level: i + 1 for i, level in enumerate(spec.levels)}
    if any(value not in lookup for value in x):
        raise ValueError(f"newdata column {spec.columns[0]!r} contains an unknown frailty level")
    if spec.penalty.sparse:
        return [[float(lookup[value]) for value in x]]
    return [[float(value == level) for value in x] for level in spec.levels]


def same(actual, expected):
    """Equal as values and as bits: a float's sign of zero, an int is not a float."""
    if isinstance(expected, float):
        return type(actual) is float and (
            math.isnan(actual)
            and math.isnan(expected)
            or np.float64(actual).view(np.uint64) == np.float64(expected).view(np.uint64)
        )
    if isinstance(expected, list | tuple):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(same(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


NUMBER_KINDS = ("float", "int", "mixed", "missing", "zeros", "infinite", "huge", "numpy")


def pick(rng, choices, n):
    """*n* plain Python values drawn from *choices*."""
    return [choices[i] for i in rng.integers(0, len(choices), n).tolist()]


def numbers(rng, kind, n):
    values = rng.normal(0.0, 3.0, n).tolist()
    coin = (rng.random(n) < 0.3).tolist()
    if kind == "int":
        return rng.integers(-50, 50, n).tolist()
    if kind == "mixed":
        return [
            i if c else v
            for i, c, v in zip(rng.integers(-5, 5, n).tolist(), coin, values, strict=True)
        ]
    if kind == "missing":
        return [math.nan if c else v for c, v in zip(coin, values, strict=True)]
    if kind == "zeros":
        return pick(rng, [0.0, -0.0, 1.5, -2.5], n)
    if kind == "infinite":
        return [
            s * math.inf if c else v
            for c, s, v in zip(coin, pick(rng, [1, -1], n), values, strict=True)
        ]
    if kind == "huge":
        return [2**60 + i if c else v for c, i, v in zip(coin, range(n), values, strict=True)]
    if kind == "numpy":
        return [np.float64(value) for value in values] + [None, math.nan]
    return values


@pytest.mark.parametrize("kind", NUMBER_KINDS)
@pytest.mark.parametrize("n", [0, 1, 2, 7, 400])
def test_ridge_variances_and_pspline_ranges_match_the_reference(kind, n):
    rng = np.random.default_rng([NUMBER_KINDS.index(kind), n])
    for _ in range(5):
        values = numbers(rng, kind, n)
        expected = reference_observed(values)
        observed = penalties._observed(values)
        assert same(observed.tolist(), expected)
        assert same(penalties._r_var(observed), reference_r_var(expected))
        if expected:
            assert same(penalties._observed_range(observed), [min(expected), max(expected)])
        else:
            assert penalties._observed_range(observed) == []


def test_signed_zero_ranges_keep_the_first_extreme():
    for values in ([0.0, -0.0, 0.0], [-0.0, 0.0], [0.0, -0.0, -1.0, 1.0, -0.0]):
        observed = penalties._observed(values)
        assert same(penalties._observed_range(observed), [min(values), max(values)])


@pytest.mark.parametrize("values", [["a", "b"], ["1.5", 2], [True, 1.0], [None, 2.0]])
def test_non_numeric_inputs_take_the_per_element_path(values):
    try:
        expected = reference_observed(values)
    except (TypeError, ValueError) as exc:
        with pytest.raises(type(exc), match=re.escape(str(exc))):
            penalties._observed(values)
    else:
        assert same(penalties._observed(values).tolist(), expected)


FRAILTY_KINDS = ("int", "huge", "float", "mixed", "bool", "str")


def frailty_values(rng, kind, n):
    if kind == "int":
        return rng.integers(0, 40, n).tolist()
    if kind == "huge":
        return pick(rng, [2**53, 2**53 + 1, 2**53 + 2, 3, 0, -2], n)
    if kind == "float":
        return pick(rng, [0.5, -1.25, 3.0, 0.0, -0.0, math.inf, math.nan], n)
    if kind == "mixed":
        return pick(rng, [1, 2.5, 3, -1.0, 2**53, None], n)
    if kind == "bool":
        return pick(rng, [True, False, 2], n)
    return pick(rng, ["b", "a10", "a2", "1", None], n)


@pytest.mark.parametrize("kind", FRAILTY_KINDS)
def test_frailty_levels_and_columns_match_the_reference(kind):
    rng = np.random.default_rng(FRAILTY_KINDS.index(kind))
    for sparse in [True, False]:
        values = frailty_values(rng, kind, 300)
        levels = reference_levels(values)
        if not sparse and len(levels) < 2:
            continue
        spec = penalties.fit_penalty(
            _CovariateTerm(column="g", call="frailty(g)"),
            ["g"],
            {"g": values},
            {"sparse": sparse, "theta": 1.0},
        )
        assert same(spec.levels, levels)
        complete = [value for value in values if not _is_missing_value(value)]
        expected = reference_columns(spec, {"g": complete})
        assert same(penalties.penalty_columns(spec, {"g": complete}), expected)
        arrays = penalties.penalty_columns(spec, {"g": complete}, as_arrays=True)
        assert same([np.asarray(column).tolist() for column in arrays], expected)
        unknown = [*complete, "unseen"]
        with pytest.raises(ValueError, match="unknown frailty level"):
            reference_columns(spec, {"g": unknown})
        with pytest.raises(ValueError, match="unknown frailty level"):
            penalties.penalty_columns(spec, {"g": unknown})


@pytest.mark.parametrize(
    "options", [{}, {"intercept": True}, {"combine": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]}]
)
@pytest.mark.parametrize("kind", ["float", "int", "zeros", "missing", "numpy"])
def test_pspline_knots_and_columns_match_the_reference(kind, options):
    rng = np.random.default_rng([NUMBER_KINDS.index(kind), len(options), len(str(options))])
    values = numbers(rng, kind, 250)
    observed = reference_observed(values)
    spec = penalties.fit_penalty(
        _CovariateTerm(column="x", call="pspline(x)"), ["x"], {"x": values}, dict(options)
    )
    assert same(list(spec.boundary), [min(observed), max(observed)])
    complete = {"x": [value for value in values if not _is_missing_value(value)]}
    expected = reference_columns(spec, complete)
    assert same(penalties.penalty_columns(spec, complete), expected)
    arrays = penalties.penalty_columns(spec, complete, as_arrays=True)
    assert same([np.asarray(column).tolist() for column in arrays], expected)


def test_pspline_of_only_missing_values_fails_as_before():
    with pytest.raises(ValueError, match="empty") as reference:
        min(reference_observed([math.nan, None]))
    with pytest.raises(ValueError, match=re.escape(str(reference.value))):
        penalties.fit_penalty(
            _CovariateTerm(column="x", call="pspline(x)"), ["x"], {"x": [math.nan, None]}, {}
        )


@pytest.mark.parametrize(
    "terms",
    [
        "pspline(x) + ridge(z, w)",
        "pspline(x, df=3, intercept=TRUE) + frailty(g)",
        "x + frailty(small)",
        "pspline(x) + frailty.gaussian(g, sparse=TRUE)",
    ],
)
def test_array_designs_match_the_row_lists_bit_for_bit(terms):
    rng = np.random.default_rng(len(terms))
    n = 300
    data = {
        "time": rng.exponential(1.0, n).round(1).tolist(),
        "status": rng.integers(0, 2, n).tolist(),
        "x": rng.normal(0.0, 2.0, n).tolist(),
        "z": rng.integers(-3, 4, n).tolist(),
        "w": [0.0 if i % 7 else -0.0 for i in range(n)],
        "g": rng.integers(0, 40, n).tolist(),
        "small": rng.integers(0, 3, n).tolist(),
    }
    formula = f"Surv(time, status) ~ {terms}"
    arrays = model_frame._model_frame(formula, data, as_array=True)
    rows = model_frame._model_frame(formula, data)
    assert isinstance(arrays.x, np.ndarray)
    expected = np.asarray(rows.x, dtype=float)
    assert arrays.x.shape == expected.shape
    assert np.array_equal(arrays.x.view(np.uint64), expected.view(np.uint64))
