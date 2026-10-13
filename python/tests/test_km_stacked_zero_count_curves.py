"""Accepted stacked KM curves retain their own zero counts during selection."""

from __future__ import annotations

import pickle

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival

FIELDS = (
    "n",
    "time",
    "n_risk",
    "n_event",
    "n_censor",
    "n_enter",
    "counts",
    "surv",
    "std_err",
    "cumhaz",
    "std_chaz",
    "lower",
    "upper",
    "strata",
    "strata_codes",
    "n_id",
    "logse",
    "conf_int",
    "conf_type",
    "conf_lower",
    "type",
    "t0",
    "influence_surv",
    "influence_chaz",
)


def _encoded(value):
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _payload(result):
    return {field: getattr(result, field) for field in FIELDS}


def _stacked(time, n_risk, n_event, surv, n, *, strata=None, n_id=None, **options):
    rows = len(time)
    defaults = {
        "n_censor": [0.0] * rows,
        "std_err": [0.0] * rows,
        "cumhaz": [0.0] * rows,
        "std_chaz": [0.0] * rows,
        "lower": list(surv),
        "upper": list(surv),
        "logse": False,
        "conf_int": 0.8,
        "conf_type": "plain",
        "type": "counting",
        "t0": -0.5,
    }
    defaults.update(options)
    return core.SurvfitKMResult.from_stacked(
        time, n_risk, n_event, surv, n, strata=strata, n_id=n_id, **defaults
    )


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("with_id", [False, True])
@pytest.mark.parametrize("explicit_stratum", [False, True])
def test_single_curve_keeps_complete_zero_or_positive_metadata(
    count, empty, with_id, explicit_stratum
):
    rows = 0 if empty else 1
    arguments = ([1.0] * rows, [0.0] * rows, [0.0] * rows, [1.0] * rows, [count])
    n_id = [count] if with_id else None
    parent = _stacked(*arguments, n_id=n_id, strata=[rows] if explicit_stratum else None)
    before = _encoded(parent)
    actual = parent.select_curves([0])
    expected = _stacked(*arguments, n_id=n_id)
    assert _encoded(actual) == _encoded(expected)
    assert _encoded(parent) == before
    assert actual.n == [count]
    assert actual.n_id == n_id
    assert _encoded(actual.select_curves([0])) == _encoded(actual)


@pytest.mark.parametrize("with_id", [False, True])
def test_mixed_zero_counts_preserve_every_field_and_repeated_curve_order(with_id):
    parent = _stacked(
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 1.0],
        [0, 2, 0, 0],
        strata=[0, 2, 0, 1],
        n_id=[0, 1, 0, 0] if with_id else None,
        n_censor=[0.0, 1.0, 0.0],
        std_err=[0.125, 0.25, 0.0],
        cumhaz=[0.5, 0.75, 0.0],
        std_chaz=[0.0625, 0.125, 0.0],
        lower=[0.25, 0.25, 1.0],
        upper=[0.75, 0.75, 1.0],
    )
    before = _encoded(parent)
    actual = parent.select_curves([3, 0, 1, 2, 3])
    expected = _stacked(
        [3.0, 1.0, 2.0, 3.0],
        [0.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.5, 0.5, 1.0],
        [0, 0, 2, 0, 0],
        strata=[1, 0, 2, 0, 1],
        n_id=[0, 0, 1, 0, 0] if with_id else None,
        n_censor=[0.0, 0.0, 1.0, 0.0],
        std_err=[0.0, 0.125, 0.25, 0.0],
        cumhaz=[0.0, 0.5, 0.75, 0.0],
        std_chaz=[0.0, 0.0625, 0.125, 0.0],
        lower=[1.0, 0.25, 0.25, 1.0],
        upper=[1.0, 0.75, 0.75, 1.0],
    )
    expected_fields = _payload(expected)
    expected_fields["strata_codes"] = [3, 0, 1, 2, 3]
    assert _encoded(_payload(actual)) == _encoded(expected_fields)
    assert _encoded(parent) == before
    for curve in (0, 2):
        selected = parent.select_curves([curve])
        expected_empty = _stacked([], [], [], [], [0], n_id=[0] if with_id else None)
        assert _encoded(selected) == _encoded(expected_empty)


def test_all_zero_visible_curves_remain_independently_selectable():
    parent = _stacked([], [], [], [], [0, 0, 0], strata=[0, 0, 0], n_id=[0, 0, 0])
    expected = _stacked([], [], [], [], [0], n_id=[0])
    for curve in range(3):
        assert _encoded(parent.select_curves([curve])) == _encoded(expected)
    selected = parent.select_curves([2, 0, 2])
    assert selected.n == [0, 0, 0]
    assert selected.n_id == [0, 0, 0]
    assert selected.strata == [0, 0, 0]
    assert selected.strata_codes == [2, 0, 2]


def test_conditional_original_level_counts_match_independent_fits_in_full():
    arguments = {
        "start_time": 5,
        "conf_type": "plain",
        "influence": 3,
    }
    parent = core.survfitkm(
        [1, 2, 5, 6, 1, 2, 7, 8],
        [1, 0, 1, 0, 1, 0, 1, 0],
        strata=[1, 1, 2, 2, 3, 3, 4, 4],
        id=list(range(8)),
        **arguments,
    )
    before = _encoded(parent)
    assert parent.n == [0, 2, 0, 2]
    assert parent.n_id == [0, 2, 0, 2]
    for curve, (times, ids) in enumerate((([5, 6], [2, 3]), ([7, 8], [6, 7]))):
        expected = core.survfitkm(times, [1, 0], id=ids, **arguments)
        assert _encoded(parent.select_curves([curve])) == _encoded(expected)
    assert _encoded(parent) == before
    repeated = parent.select_curves([1, 0, 1])
    assert repeated.n == [2, 2, 2]
    assert repeated.n_id == [2, 2, 2]


@pytest.mark.parametrize(
    ("indices", "message"), [([], "select at least one"), ([1], "out of bounds")]
)
def test_zero_curve_invalid_indices_keep_the_existing_validation(indices, message):
    parent = _stacked([], [], [], [], [0])
    with pytest.raises(ValueError, match=message):
        parent.select_curves(indices)


@pytest.mark.parametrize("field", ["n", "n_id"])
def test_constructor_rejects_metadata_lengths_that_do_not_match_visible_curves(field):
    options = {"n": [0, 1], "n_id": [0, 1]}
    options[field] = [0]
    with pytest.raises(ValueError, match=field):
        _stacked([1.0], [1.0], [0.0], [1.0], strata=[0, 1], **options)
