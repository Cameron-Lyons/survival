"""Modern concordance references from installed R survival 3.8.11.

The adjacent R generator records counts, rank rows and both influence outputs
from concordancefit, the numeric engine behind R's formula interface. It calls
the engine directly because 3.8.11's formula method omits the timefix argument.
Stratified rank rows come from separate R fits because that version's pooled
rank assembly fails for censored strata; all other stratified references are
from the pooled fit. The tests do not require R to be installed.

Joint-score covariance comes directly from R's matrix-score fit where supported.
R 3.8.11 cannot assemble stratified matrix-score results, so those references use
crossprod of its independently pooled, aligned single-score dfbeta vectors.

Python's concordant numerator includes half of tied.x; R's count[0] does not.
Raw influence columns retain R's five exclusive pair categories. Case weights
are applied to dfbeta after forming these raw derivatives. Rank output order is
not an API contract, so rank comparisons preserve the multiset of event rows.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "concordance_tied_censoring_r3811.json").read_text()
)
_CASES = _REFERENCE["cases"]
_MULTISCORE_CASES = _REFERENCE["multiscore_cases"]
_NATIVE_CASES = [
    case
    for case in _CASES
    if case["ymin"] is None
    and case["ymax"] is None
    and not case["cluster"]
    and not case["dataset"].startswith("near")
]


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def _data(case, permutation=False):
    original = _REFERENCE["datasets"][case["dataset"]]
    order = list(range(len(original["time"])))
    if permutation:
        random.Random(2718).shuffle(order)  # noqa: S311 - deterministic test permutation
    return {key: [values[i] for i in order] for key, values in original.items()}, order


def _facade_result(case, *, reverse=False, permutation=False, direct=False, influence=3):
    data, order = _data(case, permutation)
    options = {
        "timewt": case["timewt"],
        "timefix": case["timefix"],
        "ymin": case["ymin"],
        "ymax": case["ymax"],
        "influence": influence,
        "ranks": True,
        "reverse": reverse,
    }
    if influence is None:
        options.pop("influence")
    if direct:
        response_args = (
            [data["start"], data["time"], data["status"]]
            if case["response"] == "counting"
            else [data["time"], data["status"]]
        )
        # Direct Surv inputs use risk orientation; R's formula uses survival.
        result = survival.concordance(
            survival.Surv(*response_args),
            risk_scores=[-score for score in data["score"]],
            weights=data["w"],
            **options,
        )
    else:
        response = (
            "Surv(start, time, status)" if case["response"] == "counting" else "Surv(time, status)"
        )
        formula = response + " ~ score"
        if "group" in data:
            formula += " + strata(group)"
        result = survival.concordance(
            formula,
            data=data,
            weights="w",
            cluster="cluster" if case["cluster"] else None,
            **options,
        )
    return {
        "count": [
            result.concordant - result.tied_x / 2,
            result.comparable - result.concordant - result.tied_x / 2,
            result.tied_x,
            result.tied_y,
            result.tied_xy,
        ],
        "concordance": result.concordance,
        "cvar": result.conditional_variance,
        "variance": result.variance,
        "covariance": result.covariance,
        "vcov": survival.vcov(result),
        "dfbeta": result.dfbeta,
        "influence": result.influence,
        "ranks": [
            [row[key] for key in ("time", "rank", "timewt", "casewt")] for row in result.ranks
        ],
    }, order


def _native_result(case):
    data, _ = _data(case)
    prefix = "counting_" if case["response"] == "counting" else ""
    args = [data["time"], data["status"], [-score for score in data["score"]]]
    if prefix:
        args.insert(0, data["start"])
    if "group" in data:
        prefix = "stratified_" + prefix
        args.append([0 if group == "A" else 1 for group in data["group"]])
    options = {"weights": data["w"], "timewt": case["timewt"]}
    if case["response"] == "counting":
        options["timefix"] = False
    core = survival._survival
    summary = getattr(core, prefix + "concordance_summary")(*args, **options)
    influence, dfbeta, variance = getattr(core, prefix + "concordance_influence_rows")(
        *args, **options
    )
    ranks = getattr(core, prefix + "concordance_rank_rows")(*args, **options)
    tied_x = summary.get("tied_x", 0.0)
    return {
        "count": [
            summary["concordant"] - tied_x / 2,
            summary["comparable"] - summary["concordant"] - tied_x / 2,
            tied_x,
            summary.get("tied_y", 0.0),
            summary.get("tied_xy", 0.0),
        ],
        "concordance": summary["concordance"],
        "cvar": summary["conditional_variance"],
        "variance": variance,
        "dfbeta": dfbeta,
        "influence": influence,
        "ranks": ranks,
    }


def _check_output(actual, case, output, *, reverse=False, order=None):
    count = np.asarray(case["count"])
    if reverse:
        count = count[[1, 0, 2, 3, 4]]
    if output == "summary":
        _assert_close(actual["count"], count)
        expected_c = 1 - case["concordance"] if reverse else case["concordance"]
        _assert_close(actual["concordance"], expected_c)
        _assert_close(actual["cvar"], case["cvar"])
    elif output == "influence":
        influence = np.asarray(case["influence"])
        dfbeta = np.asarray(case["dfbeta"])
        if order is not None:
            influence = influence[order]
            if not case["cluster"]:
                dfbeta = dfbeta[order]
        if reverse:
            influence = influence[:, [1, 0, 2, 3, 4]]
            dfbeta = -dfbeta
        _assert_close(actual["influence"], influence)
        _assert_close(actual["dfbeta"], dfbeta)
        _assert_close(actual["variance"], case["variance"])
    else:
        ranks = np.asarray(case["ranks"]).copy()
        if reverse:
            ranks[:, 1] *= -1
        _assert_close(sorted(map(tuple, actual["ranks"])), sorted(map(tuple, ranks)))


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("output", ["summary", "influence", "ranks"])
@pytest.mark.parametrize("reverse", [False, True], ids=["survival_scores", "risk_scores"])
def test_formula_concordance_matches_r(case, output, reverse):
    actual, order = _facade_result(case, reverse=reverse)
    _check_output(actual, case, output, reverse=reverse, order=order)


@pytest.mark.parametrize("case", _NATIVE_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("output", ["summary", "influence", "ranks"])
def test_native_concordance_matches_r(case, output):
    _check_output(_native_result(case), case, output)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("influence", [None, 0], ids=["default", "influence_zero"])
def test_variance_matches_r_without_returning_influence(case, influence):
    actual, _ = _facade_result(case, influence=influence)
    _assert_close(actual["variance"], case["variance"])
    _assert_close(actual["covariance"], [[case["variance"]]])
    _assert_close(actual["vcov"], [[case["variance"]]])
    _assert_close(actual["cvar"], case["cvar"])
    assert actual["dfbeta"] is None
    assert actual["influence"] is None


def _multiscore_result(case, influence, *, direct=False, negate_second=False):
    data = {key: list(values) for key, values in _REFERENCE["datasets"][case["dataset"]].items()}
    if negate_second:
        data["score2"] = [-value for value in data["score2"]]
    options = {
        "weights": data["w"],
        "timewt": case["timewt"],
        "timefix": False,
        "influence": influence,
        "cluster": data["cluster"] if case["cluster"] else None,
    }
    if direct:
        response_args = (
            [data["start"], data["time"], data["status"]]
            if case["response"] == "counting"
            else [data["time"], data["status"]]
        )
        return survival.concordance(
            survival.Surv(*response_args),
            risk_scores=[[-a, -b] for a, b in zip(data["score"], data["score2"], strict=True)],
            **options,
        )
    lhs = "Surv(start, time, status)" if case["response"] == "counting" else "Surv(time, status)"
    formula = lhs + " ~ score + score2"
    if "group" in data:
        formula += " + strata(group)"
    return survival.concordance(formula, data=data, **options)


def _check_multiscore_result(result, case, influence):
    expected_covariance = np.asarray(case["covariance"])
    # Require meaningful cross-model coverage: neither score has zero variance,
    # and their covariance is nonzero without perfect linear dependence.
    assert np.all(np.diag(expected_covariance) > 0)
    assert abs(expected_covariance[0, 1]) > 1e-8
    assert np.linalg.det(expected_covariance) > 0
    _assert_close(result.covariance, expected_covariance)
    _assert_close(survival.vcov(result), expected_covariance)
    _assert_close(result.variance, case["variance"])
    _assert_close(result.var, np.diag(expected_covariance))
    _assert_close(result.concordance, case["concordance"])
    _assert_close(result.conditional_variance, case["cvar"])
    expected_count = np.asarray(case["count"])
    _assert_close(result.concordant, expected_count[:, 0] + expected_count[:, 2] / 2)
    _assert_close(result.comparable, expected_count[:, :3].sum(axis=1))
    _assert_close(result.tied_x, expected_count[:, 2])
    _assert_close(result.tied_y, expected_count[:, 3])
    _assert_close(result.tied_xy, expected_count[:, 4])
    if influence in (1, 3):
        _assert_close(result.dfbeta, case["dfbeta"])
    else:
        assert result.dfbeta is None
    if influence in (2, 3):
        _assert_close(result.influence, case["influence"])
    else:
        assert result.influence is None
    assert result.ranks is None


@pytest.mark.parametrize("case", _MULTISCORE_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("influence", [0, 1, 2, 3])
def test_multiscore_formula_covariance_matches_r(case, influence):
    result = _multiscore_result(case, influence)
    assert result.score_names == ["score", "score2"]
    _check_multiscore_result(result, case, influence)


@pytest.mark.parametrize(
    "case",
    [case for case in _MULTISCORE_CASES if case["dataset"] == "multi_entry"],
    ids=lambda case: case["name"],
)
@pytest.mark.parametrize("influence", [0, 1, 2, 3])
def test_direct_matrix_score_covariance_matches_r(case, influence):
    result = _multiscore_result(case, influence, direct=True)
    assert result.score_names == ["score1", "score2"]
    _check_multiscore_result(result, case, influence)


@pytest.mark.parametrize("case", _MULTISCORE_CASES, ids=lambda case: case["name"])
def test_reversing_one_score_changes_cross_covariance_sign(case):
    result = _multiscore_result(case, 0, negate_second=True)
    expected_covariance = np.asarray(case["covariance"]) * [[1, -1], [-1, 1]]
    _assert_close(result.covariance, expected_covariance)
    _assert_close(survival.vcov(result), expected_covariance)
    _assert_close(result.variance, case["variance"])
    _assert_close(result.concordance, [case["concordance"][0], 1 - case["concordance"][1]])
    assert result.dfbeta is None
    assert result.influence is None


@pytest.mark.parametrize(
    "case",
    [case for case in _CASES if case["dataset"] in {"mixed", "weighted_minimal", "entry"}],
    ids=lambda case: case["name"],
)
@pytest.mark.parametrize("direct", [False, True], ids=["formula", "direct_surv"])
def test_permuted_concordance_matches_r(case, direct):
    actual, order = _facade_result(case, permutation=True, direct=direct)
    for output in ("summary", "influence", "ranks"):
        _check_output(actual, case, output, order=order)


@pytest.mark.parametrize("timewt", ["S/G", "n/G2"])
def test_counting_unsupported_time_weights_match_r(timewt):
    # The generator's R concordancefit rejects these modes for counting data.
    data = _REFERENCE["datasets"]["weighted_minimal"]
    with pytest.raises(ValueError, match="not supported for counting-process"):
        survival.concordance("Surv(start, time, status) ~ score", data=data, timewt=timewt)


def _pairwise_n_reference(data, counting):
    """Independent derivative of weighted pair counts; no fitted code reused."""
    n = len(data["time"])
    count = np.zeros(5)
    influence = np.zeros((n, 5))
    for left in range(n):
        for right in range(left + 1, n):
            tl, tr = data["time"][left], data["time"][right]
            el, er = data["status"][left], data["status"][right]
            if tl == tr and el and er:
                column = 4 if data["score"][left] == data["score"][right] else 3
            else:
                if el and (tl < tr or (tl == tr and not er)):
                    event, comparator = left, right
                elif er and (tr < tl or (tr == tl and not el)):
                    event, comparator = right, left
                else:
                    continue
                if counting and data["start"][comparator] >= data["time"][event]:
                    continue
                a, b = data["score"][event], data["score"][comparator]
                column = 0 if a < b else (1 if a > b else 2)
            count[column] += data["w"][left] * data["w"][right]
            influence[left, column] += data["w"][right]
            influence[right, column] += data["w"][left]
    denominator = count[:3].sum()
    c = (count[0] + count[2] / 2) / denominator if denominator else 0.5
    dfbeta = (
        (influence[:, 0] + influence[:, 2] / 2 - c * influence[:, :3].sum(axis=1))
        * data["w"]
        / denominator
        if denominator
        else np.zeros(n)
    )
    return count, influence, dfbeta, c


@pytest.mark.parametrize("counting", [False, True], ids=["right", "counting"])
def test_random_tied_counts_and_derivatives_match_independent_pairwise_oracle(counting):
    rng = random.Random(314159)  # noqa: S311 - reproducible statistical fixture
    for _ in range(30):
        time = [float(rng.randrange(1, 7)) for _ in range(13)]
        data = {
            "time": time,
            "start": [float(rng.randrange(int(t))) for t in time],
            "status": [rng.randrange(2) for _ in time],
            "score": [float(rng.randrange(4)) for _ in time],
            "w": [rng.choice([0.0, 0.5, 1.0, 1.5, 2.0]) for _ in time],
        }
        count, influence, dfbeta, c = _pairwise_n_reference(data, counting)
        response = "Surv(start, time, status)" if counting else "Surv(time, status)"
        result = survival.concordance(
            response + " ~ score", data=data, weights="w", influence=3, timefix=False
        )
        _assert_close(result.concordant, count[0] + count[2] / 2)
        _assert_close(result.comparable, count[:3].sum())
        _assert_close([result.tied_x, result.tied_y, result.tied_xy], count[2:])
        _assert_close(result.concordance, c)
        _assert_close(result.influence, influence)
        _assert_close(result.dfbeta, dfbeta)
        _assert_close(result.variance, dfbeta @ dfbeta)
