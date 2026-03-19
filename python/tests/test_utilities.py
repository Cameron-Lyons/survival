import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def test_collapse():
    y = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 0.0]
    x = [1, 1, 1, 1]
    istate = [0, 0, 0, 0]
    subject_id = [1, 1, 2, 2]
    wt = [1.0, 1.0, 1.0, 1.0]
    order = [0, 1, 2, 3]

    result = survival.collapse(y, x, istate, subject_id, wt, order)
    assert isinstance(result, dict)
    assert "matrix" in result
    assert "dimnames" in result


def test_tmerge_family_public_apis():
    merged = survival.tmerge(
        [1, 1, 2],
        [1.0, 2.0, 1.0],
        [0.0, float("nan"), 0.0],
        [1, 1, 2],
        [0.5, 1.5, 0.5],
        [2.0, 3.0, 4.0],
    )
    indices = survival.tmerge2(
        [1, 1, 2],
        [1.0, 2.0, 1.0],
        [1, 1, 2],
        [0.5, 1.5, 0.5],
    )
    carry = survival.tmerge3([1, 1, 1, 2, 2], [False, True, False, True, False])

    assert merged == pytest.approx([2.0, 5.0, 4.0])
    assert indices == [1, 2, 3]
    assert carry == [1, 1, 3, 0, 5]


def test_surv2data_and_survcondense_public_apis():
    surv2data = survival.surv2data(
        [2, 1, 1, 2],
        [3.0, 0.0, 2.0, 0.0],
        [5.0, 4.0, 4.0, 5.0],
        [0, 1, 1, 0],
    )
    condensed = survival.survcondense(
        [2, 1, 1, 2],
        [0.0, 0.0, 5.0, 3.0],
        [3.0, 5.0, 8.0, 5.0],
        [1, 0, 0, 0],
    )

    assert surv2data.id == [1, 1, 2, 2]
    assert surv2data.time1 == pytest.approx([0.0, 2.0, 0.0, 3.0])
    assert surv2data.time2 == pytest.approx([2.0, 4.0, 3.0, 5.0])
    assert surv2data.status == [0, 1, 0, 0]
    assert surv2data.row_index == [2, 3, 4, 1]

    assert condensed.id == [1, 2, 2]
    assert condensed.time1 == pytest.approx([0.0, 0.0, 3.0])
    assert condensed.time2 == pytest.approx([8.0, 3.0, 5.0])
    assert condensed.status == [0, 1, 0]
    assert condensed.row_map == [[2, 3], [1], [4]]


def test_rttright_public_apis_and_validation():
    result = survival.rttright([3.0, 1.0, 2.0], [1, 0, 1])
    stratified = survival.rttright_stratified(
        [3.0, 1.0, 2.0, 1.5],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
    )

    assert result.time == pytest.approx([1.0, 2.0, 3.0])
    assert result.status == [0, 1, 1]
    assert result.weights == pytest.approx([0.0, 1.0, 2.0])
    assert result.order == [1, 2, 0]

    assert stratified.time == pytest.approx([3.0, 1.0, 2.0, 1.5])
    assert stratified.status == [1, 0, 1, 1]
    assert all(weight >= 0.0 for weight in stratified.weights)
    assert sum(weight > 0.0 for weight in stratified.weights) == 3
    assert sorted(stratified.order) == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="time and status must have same length"):
        survival.rttright([1.0], [1, 0])


def test_agexact_public_api_and_validation():
    result = survival.agexact(
        0,
        1,
        1,
        [0.0],
        [1.0],
        [1],
        [1.0],
        [0.0],
        [0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0, 0.0],
        [0.0] * 5,
        [0, 0],
        1e-9,
        1e-9,
        [1],
    )

    assert sorted(result) == ["beta", "covar", "flag", "imat", "loglik", "maxiter", "means", "sctest", "u"]
    assert result["beta"] == pytest.approx([0.0])
    assert result["loglik"] == pytest.approx([0.0, 0.0])
    assert result["flag"] == 0

    with pytest.raises(ValueError, match="work must have length at least 5"):
        survival.agexact(
            1,
            1,
            1,
            [0.0],
            [1.0],
            [1],
            [1.0],
            [0.0],
            [0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0, 0.0],
            [0.0],
            [0, 0],
            1e-9,
            1e-9,
            [1],
        )
