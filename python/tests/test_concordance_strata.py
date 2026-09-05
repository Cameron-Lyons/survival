"""Explicit strata dispatch and retained counts, using R 3.8.11 fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "concordance_tied_censoring_r3811.json").read_text()
)
_CASES = [case for case in _REFERENCE["multiscore_cases"] if case["dataset"] == "multi_stratified"]


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def _data():
    return {key: list(value) for key, value in _REFERENCE["datasets"]["multi_stratified"].items()}


def _fit(data, case, *, formula=False, matrix=True, **kwargs):
    options = {
        "weights": data["w"],
        "strata": data["group"],
        "cluster": data["cluster"] if case["cluster"] else None,
        "timewt": case["timewt"],
        "timefix": False,
        **kwargs,
    }
    if formula:
        lhs = (
            "Surv(start, time, status)" if case["response"] == "counting" else "Surv(time, status)"
        )
        return survival.concordance(
            lhs + (" ~ score + score2" if matrix else " ~ score"), data=data, **options
        )
    response = (
        survival.Surv(data["start"], data["time"], data["status"])
        if case["response"] == "counting"
        else survival.Surv(data["time"], data["status"])
    )
    scores = list(zip(data["score"], data["score2"], strict=True)) if matrix else data["score"]
    return survival.concordance(response, scores=scores, reverse=True, **options)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("formula", [False, True], ids=["direct", "formula_vector"])
@pytest.mark.parametrize("influence", [0, 1, 2, 3])
def test_explicit_strata_multiscore_matches_r(case, formula, influence):
    result = _fit(_data(), case, formula=formula, influence=influence)
    counts = np.asarray(case["count"])
    _close(result.concordance, case["concordance"])
    _close(result.concordant, counts[:, 0] + counts[:, 2] / 2)
    _close(result.comparable, counts[:, :3].sum(axis=1))
    _close(result.covariance, case["covariance"])
    _close(survival.vcov(result), case["covariance"])
    _close(result.variance, case["variance"])
    _close(result.conditional_variance, case["cvar"])
    if influence in (1, 3):
        _close(result.dfbeta, case["dfbeta"])
    else:
        assert result.dfbeta is None
    if influence in (2, 3):
        _close(result.influence, case["influence"])
    else:
        assert result.influence is None
    # R suppresses per-stratum count retention for multiple score columns.
    assert result.stratum_labels is None
    assert result.stratum_counts is None


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("keepstrata", [True, 2, 10, False, 0, 1])
@pytest.mark.parametrize("formula", [False, True], ids=["direct", "formula_vector"])
def test_keepstrata_retains_exclusive_counts_without_changing_pooled_result(
    case, keepstrata, formula
):
    data = _data()
    # Reorder strata to ensure metadata are associated with labels, not sorted
    # positions. Counts below derive from R's raw per-case weighted derivatives.
    order = list(range(12, 24)) + list(range(12))
    data = {key: [value[i] for i in order] for key, value in data.items()}
    result = _fit(data, case, formula=formula, matrix=False, influence=0, keepstrata=keepstrata)
    expected_count = np.asarray(case["count"])[0]
    _close(result.concordance, case["concordance"][0])
    _close(result.concordant, expected_count[0] + expected_count[2] / 2)
    _close(result.comparable, expected_count[:3].sum())
    _close(result.covariance, [[case["variance"][0]]])
    assert result.dfbeta is None
    assert result.influence is None
    retained = keepstrata is True or (not isinstance(keepstrata, bool) and keepstrata >= 2)
    if retained:
        original = _data()
        raw = np.asarray(case["influence"])[0]
        expected_by_group = {
            group: sum(
                (
                    original["w"][i] * raw[i] / 2
                    for i, label in enumerate(original["group"])
                    if label == group
                ),
                np.zeros(5),
            )
            for group in ("A", "B")
        }
        assert set(result.stratum_labels) == {"A", "B"}
        for label, count in zip(result.stratum_labels, result.stratum_counts, strict=True):
            _close(count, expected_by_group[label])
        _close(np.asarray(result.stratum_counts).sum(axis=0), expected_count)
    else:
        assert result.stratum_labels is None
        assert result.stratum_counts is None
    frame = survival.as_data_frame(result)
    _close(frame["concordance"], [case["concordance"][0]])
    _close(frame["comparable"], [expected_count[:3].sum()])


@pytest.mark.parametrize("response", ["right", "counting"])
@pytest.mark.parametrize("formula", [False, True], ids=["direct", "formula_vector"])
def test_explicit_strata_follow_subset_and_omitted_rows(response, formula):
    data = _data()
    case = {"response": response, "cluster": True, "timewt": "S"}
    subset = [23, 2, 14, 0, 8, 19, 4, 16, 10, 21, 6, 12, 1, 17, 7, 20, 11, 15]
    data["group"][14] = None
    data["w"][4] = float("nan")
    data["score2"][19] = float("nan")
    data["cluster"][23] = None
    result = _fit(data, case, formula=formula, subset=subset, na_action="omit", influence=3)
    expected_rows = [i for i in subset if i not in {14, 4, 19, 23}]
    clean = {key: [value[i] for i in expected_rows] for key, value in data.items()}
    lhs = "Surv(start,time,status)" if response == "counting" else "Surv(time,status)"
    expected = survival.concordance(
        lhs + " ~ score + score2 + strata(group)",
        data=clean,
        weights="w",
        cluster="cluster",
        influence=3,
        timewt="S",
        timefix=False,
    )
    assert result.n == len(expected_rows)
    for field in ("concordance", "concordant", "comparable", "covariance", "dfbeta", "influence"):
        _close(getattr(result, field), getattr(expected, field))


def test_formula_and_explicit_strata_are_mutually_exclusive():
    data = _data()
    with pytest.raises(ValueError, match="only one.*strata"):
        survival.concordance(
            "Surv(time,status) ~ score + strata(group)", data=data, strata=data["group"]
        )


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        (["A", "B"], ValueError, "(length|same length)"),
        ([["A"]] * 24, ValueError, "one-dimensional"),
        ([{"A": 1}] * 24, TypeError, "unhashable"),
        ([None] + ["A"] * 23, ValueError, "missing"),
    ],
)
def test_explicit_strata_validate_labels(labels, error, message):
    data = _data()
    with pytest.raises(error, match=message):
        survival.concordance(
            survival.Surv(data["time"], data["status"]), scores=data["score"], strata=labels
        )


def test_none_and_one_stratum_do_not_retain_stratum_counts():
    data = _data()
    response = survival.Surv(data["time"], data["status"])
    results = [
        survival.concordance(response, scores=data["score"], strata=labels, keepstrata=True)
        for labels in (None, ["A"] * len(data["time"]))
    ]
    _close(results[0].concordance, results[1].concordance)
    _close(results[0].covariance, results[1].covariance)
    for result in results:
        assert result.stratum_labels is None
        assert result.stratum_counts is None


@pytest.mark.parametrize("timewt", ["n", "I"])
@pytest.mark.parametrize("keepstrata", [True, False, 0, 10, float("inf")])
def test_many_sparse_strata_count_retention_does_not_change_estimates(timewt, keepstrata):
    # R survival3.8.11 with keepstrata=TRUE gives these pooled references.
    # Its >10-stratum packing shortcut with FALSE bypasses the sparse-event
    # fallback for I and changes C; retaining counts must only affect display.
    data = {"time": [], "status": [], "score": [], "weight": [], "group": []}
    for group in range(1, 12):
        data["time"].extend([1.0, 2.0])
        data["status"].extend([1, 0])
        data["score"].extend([0.0, 2.0 if group <= 5 else -2.0])
        data["weight"].extend([1.0, float(group)])
        data["group"].extend([group, group])
    result = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        weights="weight",
        timewt=timewt,
        keepstrata=keepstrata,
    )
    assert result.concordance == pytest.approx(15.0 / 66.0)
    assert result.variance == pytest.approx(0.025774271642040235, rel=1e-12)
    retained = keepstrata is True or keepstrata == float("inf")
    assert (result.stratum_counts is not None) == retained
