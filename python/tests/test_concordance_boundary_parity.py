"""Concordance boundary references from R survival 3.8.11.

The adjacent R generator uses concordancefit directly to preserve the requested
timefix flag. Its recorded limitations distinguish unstable R implementation
artifacts from the numerical references. NaN and infinity are explicit strings
in JSON, converted to floats for assertions. No R installation is needed here.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "concordance_boundary_r3811.json").read_text()
)
_CASES = _REFERENCE["cases"]
_NATIVE_CASES = [case for case in _CASES if not case["cluster"]]
_ERRORS = _REFERENCE["error_cases"]
_PERMUTATION_CASES = [
    case
    for case in _CASES
    if case["timewt"] == "I"
    or case["dataset"] in {"aeq_adjacent_chain", "simultaneous_events", "ymin_after_timefix"}
]


def _assert_close(actual, expected):
    np.testing.assert_allclose(
        actual, np.asarray(expected, dtype=float), rtol=2e-12, atol=2e-13, equal_nan=True
    )


def _data(case, order=None):
    data = _REFERENCE["datasets"][case["dataset"]]
    if order is None:
        order = list(range(len(data["time"])))
    return {key: [values[index] for index in order] for key, values in data.items()}


def _bound(case, name):
    value = case[name]
    return None if value is None else float(value)


def _facade(case, mode=3, order=None):
    data = _data(case, order)
    response = (
        "Surv(start, time, status)" if case["response"] == "counting" else "Surv(time, status)"
    )
    formula = response + " ~ score"
    if "group" in data:
        formula += " + strata(group)"
    return survival.concordance(
        formula,
        data=data,
        weights="w",
        cluster="cluster" if case["cluster"] else None,
        timewt=case["timewt"],
        timefix=case["timefix"],
        ymin=_bound(case, "ymin"),
        ymax=_bound(case, "ymax"),
        influence=mode,
        ranks=case["check_ranks"],
    )


def _native_arguments(case):
    data = _data(case)
    prefix = "counting_" if case["response"] == "counting" else ""
    args = [data["time"], data["status"], [-score for score in data["score"]]]
    if prefix:
        args.insert(0, data["start"])
    if "group" in data:
        levels = list(dict.fromkeys(data["group"]))
        args.append([levels.index(group) for group in data["group"]])
        prefix = "stratified_" + prefix
    options = {
        "weights": data["w"],
        "timewt": case["timewt"],
        "timefix": case["timefix"],
        "ymin": _bound(case, "ymin"),
        "ymax": _bound(case, "ymax"),
    }
    return prefix, args, options


def _counts(concordant, comparable, tied_x, tied_y, tied_xy):
    # Python's concordant field is the numerator, including half of tied.x.
    return [
        concordant - 0.5 * tied_x,
        comparable - concordant - 0.5 * tied_x,
        tied_x,
        tied_y,
        tied_xy,
    ]


def _assert_ranks(rows, case):
    # At a tied event time R orders rows by predictor, not source row number.
    actual = sorted(tuple(row) for row in rows)
    expected = sorted(tuple(row) for row in case["ranks"])
    _assert_close(np.asarray(actual).reshape(-1, 4), np.asarray(expected).reshape(-1, 4))


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_boundary_facade_matches_r(case, mode):
    result = _facade(case, mode)
    _assert_close(result.concordance, case["concordance"])
    _assert_close(
        _counts(result.concordant, result.comparable, result.tied_x, result.tied_y, result.tied_xy),
        case["count"],
    )
    _assert_close(result.variance, case["variance"])
    _assert_close(result.covariance, [[case["variance"]]])
    _assert_close(survival.vcov(result), [[case["variance"]]])
    _assert_close(result.conditional_variance, case["cvar"])
    if mode in {1, 3}:
        _assert_close(result.dfbeta, case["dfbeta"])
    else:
        assert result.dfbeta is None
    if mode in {2, 3}:
        _assert_close(result.influence, case["influence"])
    else:
        assert result.influence is None
    if case["check_ranks"]:
        _assert_ranks(
            [[row[key] for key in ("time", "rank", "timewt", "casewt")] for row in result.ranks],
            case,
        )
    else:
        assert result.ranks is None


@pytest.mark.parametrize("case", _NATIVE_CASES, ids=lambda case: case["name"])
def test_boundary_native_functions_share_r_semantics(case):
    prefix, args, options = _native_arguments(case)
    core = survival._survival
    summary = getattr(core, prefix + "concordance_summary")(*args, **options)
    _assert_close(summary["concordance"], case["concordance"])
    _assert_close(
        _counts(
            *(summary[key] for key in ("concordant", "comparable", "tied_x", "tied_y", "tied_xy"))
        ),
        case["count"],
    )
    _assert_close(summary["conditional_variance"], case["cvar"])
    influence, dfbeta, variance = getattr(core, prefix + "concordance_influence_rows")(
        *args, **options
    )
    _assert_close(influence, case["influence"])
    _assert_close(dfbeta, case["dfbeta"])
    _assert_close(variance, case["variance"])
    if case["check_ranks"]:
        ranks = getattr(core, prefix + "concordance_rank_rows")(*args, **options)
        _assert_ranks(ranks, case)
    if not prefix.startswith("stratified_"):
        _assert_close(
            getattr(core, prefix + "concordance_index")(*args, **options), case["concordance"]
        )


@pytest.mark.parametrize("case", _PERMUTATION_CASES, ids=lambda case: case["name"])
def test_boundary_results_preserve_observation_alignment(case):
    original = _data(case)
    order = list(range(len(original["time"])))
    random.Random(1103).shuffle(order)  # noqa: S311 - deterministic observation permutation
    result = _facade(case, order=order)
    _assert_close(result.concordance, case["concordance"])
    _assert_close(result.variance, case["variance"])
    _assert_close(result.influence, np.asarray(case["influence"], dtype=float)[order])
    if case["cluster"]:
        reference_levels = sorted(set(original["cluster"]))
        output_levels = list(dict.fromkeys(original["cluster"][index] for index in order))
        dfbeta_order = [reference_levels.index(level) for level in output_levels]
    else:
        dfbeta_order = order
    _assert_close(result.dfbeta, np.asarray(case["dfbeta"], dtype=float)[dfbeta_order])


@pytest.mark.parametrize("case", _ERRORS, ids=lambda case: case["name"])
def test_timefix_rejects_intervals_that_collapse(case):
    with pytest.raises(ValueError, match="effective length 0"):
        _facade(case)
    prefix, args, options = _native_arguments(case)
    for suffix in ("index", "summary", "rank_rows", "influence_rows"):
        function = getattr(survival._survival, prefix + "concordance_" + suffix)
        with pytest.raises(ValueError, match="effective length 0"):
            function(*args, **options)


@pytest.mark.parametrize("response", ["right", "counting"])
@pytest.mark.parametrize("bound", ["ymin", "ymax"])
def test_concordance_rejects_nan_bounds(response, bound):
    case = next(
        item
        for item in _CASES
        if item["dataset"] == "bound_limits"
        and item["response"] == response
        and item["ymin"] is None
        and item["ymax"] is None
    )
    case = {**case, bound: float("nan")}
    with pytest.raises(ValueError, match="(?i)nan"):
        _facade(case)
    prefix, args, options = _native_arguments(case)
    for suffix in ("index", "summary", "rank_rows", "influence_rows"):
        with pytest.raises(ValueError, match="(?i)nan"):
            getattr(survival._survival, prefix + "concordance_" + suffix)(*args, **options)
