"""RMST comparison rejects inconsistent vectors before grouping observations."""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

TIME = [1.0, 2.0, 3.0, 4.0]
STATUS = [1, 1, 0, 1]
GROUP = [-3, 7, -3, 7]


@pytest.mark.parametrize("field", ["status", "weights"])
@pytest.mark.parametrize("length", [0, 1, 3, 5])
def test_rmst_comparison_rejects_misaligned_status_and_weights(field, length):
    status = [1] * length if field == "status" else STATUS
    weights = [1.0] * length if field == "weights" else None
    if length > len(TIME):
        if field == "status":
            status[-1] = 2
        else:
            weights[-1] = -1.0
    with pytest.raises(ValueError, match=f"{field} length mismatch"):
        survival.validation.rmst_comparison(TIME, status, GROUP, 4.0, weights=weights)


def test_weighted_rmst_comparison_preserves_group_means():
    result = survival.validation.rmst_comparison(
        TIME, STATUS, GROUP, 4.0, weights=[0.5, 1.5, 2.0, 1.0]
    )
    assert [group.group for group in result.groups] == [-3, 7]
    assert [group.n for group in result.groups] == [2, 2]
    assert [group.rmean for group in result.groups] == pytest.approx([3.4, 2.8])
    assert result.difference == pytest.approx([-0.6])


def _array(values, layout):
    values = np.asarray(values)
    if layout == "reversed":
        return values[::-1].copy()[::-1]
    if layout == "strided":
        return np.repeat(values, 2)[::2]
    return values


@pytest.mark.parametrize("layout", ["contiguous", "reversed", "strided"])
def test_rmst_comparison_numpy_layouts_match_lists(layout):
    weights = [0.5, 1.5, 2.0, 1.0]
    expected = survival.validation.rmst_comparison(TIME, STATUS, GROUP, 4.0, weights=weights)
    actual = survival.validation.rmst_comparison(
        _array(TIME, layout),
        _array(STATUS, layout),
        _array(GROUP, layout),
        4.0,
        weights=_array(weights, layout),
    )
    for got, want in zip(actual.groups, expected.groups, strict=True):
        for field in ("group", "n", "events", "rmean", "se_rmean", "lower", "upper"):
            assert getattr(got, field) == getattr(want, field)
    for field in (
        "difference",
        "difference_se",
        "difference_lower",
        "difference_upper",
        "difference_p_value",
        "chisq",
        "df",
        "p_value",
    ):
        assert getattr(actual, field) == getattr(expected, field)


@pytest.mark.parametrize("layout", ["contiguous", "reversed", "strided"])
def test_survmean_curves_numpy_layouts_match_lists(layout):
    vectors = [TIME, [0.75, 0.5, 0.5, 0.0], [4.0, 3.0, 2.0, 1.0], STATUS, [4.0]]
    expected = survival.validation.survmean_curves(*vectors)
    actual = survival.validation.survmean_curves(*[_array(values, layout) for values in vectors])
    for got, want in zip(actual, expected, strict=True):
        for field in (
            "records",
            "n_max",
            "n_start",
            "events",
            "rmean",
            "se_rmean",
            "end_time",
            "median",
            "lower",
            "upper",
        ):
            assert getattr(got, field) == getattr(want, field)
