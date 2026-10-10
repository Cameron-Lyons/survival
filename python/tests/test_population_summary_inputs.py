"""Public low-level person-years summaries validate shapes before arithmetic."""

from __future__ import annotations

import math
import struct
import sys

import pytest

from .helpers import setup_survival_import

population = setup_survival_import().population
MAXIMUM = sys.maxsize // struct.calcsize("d")
USIZE_MAX = 2 * sys.maxsize + 1
OVERFLOWING = 1 << (struct.calcsize("P") * 4)


def empty_summary(dims, **options):
    return population.summary_pyears([], [], [], [], dims=dims, **options)


@pytest.mark.parametrize("totals", [False, True])
@pytest.mark.parametrize("tcut", [False, True])
@pytest.mark.parametrize(
    "dims",
    [
        [USIZE_MAX],
        [USIZE_MAX, 0],
        [0, USIZE_MAX],
        [0, 2, USIZE_MAX],
        [USIZE_MAX, 0, 1],
        [MAXIMUM + 1, 0],
        [OVERFLOWING, OVERFLOWING],
        [MAXIMUM, 2],
    ],
)
def test_unaddressable_dimensions_raise_value_error(dims, totals, tcut):
    with pytest.raises(ValueError, match="table dimensions exceed addressable memory"):
        empty_summary(dims, totals=totals, tcut=tcut)


@pytest.mark.parametrize("tcut", [False, True])
@pytest.mark.parametrize("dims", [[0, MAXIMUM], [MAXIMUM, 0], [0, 2, MAXIMUM]])
def test_unaddressable_margins_are_rejected_before_allocation(dims, tcut):
    with pytest.raises(ValueError, match="table dimensions exceed addressable memory"):
        empty_summary(dims, totals=True, tcut=tcut)


@pytest.mark.parametrize("totals", [False, True])
@pytest.mark.parametrize("tcut", [False, True])
@pytest.mark.parametrize("field", ["pyears", "n", "event", "expected"])
def test_mutated_cell_lengths_are_rejected(field, totals, tcut):
    inputs = {
        "pyears": [10.0, 20.0],
        "n": [2.0, 3.0],
        "event": [1.0, 2.0],
        "expected": [0.5, 1.0],
    }
    inputs[field].pop()
    with pytest.raises(ValueError, match=rf"{field} must have prod\(dims\) = 2 cells"):
        population.summary_pyears(**inputs, dims=[2], totals=totals, tcut=tcut)


@pytest.mark.parametrize("totals", [False, True])
@pytest.mark.parametrize("tcut", [False, True])
@pytest.mark.parametrize(
    ("dims", "expanded", "cells"),
    [
        ([0], [1], 1),
        ([0, 2], [1, 3], 3),
        ([2, 0], [3, 1], 3),
        ([0, 0], [1, 1], 1),
        ([2, 2, 0], [3, 3, 0], 0),
        ([0, 2, 2], [1, 3, 2], 6),
    ],
)
def test_zero_cell_tables_preserve_empty_and_margin_conventions(
    dims, expanded, cells, totals, tcut
):
    # Stock survival 3.8-12 accepts the first three shapes with totals=True.
    # The higher-dimensional cases preserve existing native data summaries.
    result = empty_summary(dims, totals=totals, tcut=tcut, rate=True, ci_r=True, ci_rr=True)
    count = cells if totals else 0
    assert result.dims == (expanded if totals else dims)
    for field in ("pyears", "event", "expected"):
        assert getattr(result, field) == [0.0] * count
    assert len(result.n) == count
    assert all(math.isnan(n) if tcut else n == 0.0 for n in result.n)
    for field in ("rate", "rr", "ci_r_lower", "ci_r_upper", "ci_rr_lower", "ci_rr_upper"):
        values = getattr(result, field)
        assert len(values) == count
        assert all(math.isnan(value) for value in values)
    assert result.total_events == result.total_pyears == result.offtable == 0.0
    assert result.observations == 0


@pytest.mark.parametrize("totals", [False, True])
def test_zero_product_does_not_multiply_unused_addressable_extents(totals):
    result = empty_summary([MAXIMUM - 1, MAXIMUM - 1, 0], totals=totals)
    width = MAXIMUM if totals else MAXIMUM - 1
    assert result.dims == [width, width, 0]
    assert result.n == result.pyears == result.event == result.expected == result.rr == []
    assert result.rate is result.ci_r_lower is result.ci_rr_lower is None
    assert result.total_events == result.total_pyears == 0.0


@pytest.mark.parametrize("totals", [False, True])
def test_empty_dimensions_keep_single_cell_convention(totals):
    result = population.summary_pyears(
        [5.0],
        [2.0],
        [1.0],
        [0.5],
        dims=[],
        totals=totals,
        rate=True,
        offtable=0.25,
        observations=2,
    )
    cells = 2 if totals else 1
    assert result.dims == [cells]
    assert result.pyears == [5.0] * cells
    assert result.n == [2.0] * cells
    assert result.event == [1.0] * cells
    assert result.expected == [0.5] * cells
    assert result.rate == [0.2] * cells
    assert result.rr == [2.0] * cells
    assert result.total_events == 1.0
    assert result.total_pyears == 5.0
    assert result.offtable == 0.25
    assert result.observations == 2


@pytest.mark.parametrize("tcut", [False, True])
@pytest.mark.parametrize(
    ("dims", "cells", "expanded", "out_cells"),
    [([2, 3], 6, [3, 4], 12), ([2, 3, 2], 12, [3, 4, 2], 24)],
)
def test_margins_match_hand_sums_and_closed_form_zero_event_limits(
    dims, cells, expanded, out_cells, tcut
):
    expected_n = [
        1,
        2,
        3,
        3,
        4,
        7,
        5,
        6,
        11,
        9,
        12,
        21,
        7,
        8,
        15,
        9,
        10,
        19,
        11,
        12,
        23,
        27,
        30,
        57,
    ]
    expected_events = [
        0,
        1,
        1,
        2,
        3,
        5,
        4,
        5,
        9,
        6,
        9,
        15,
        6,
        7,
        13,
        8,
        9,
        17,
        10,
        11,
        21,
        24,
        27,
        51,
    ]
    margins = [False, False, True, False, False, True, False, False, True, True, True, True]
    result = population.summary_pyears(
        [10.0 * n for n in range(1, cells + 1)],
        list(range(1, cells + 1)),
        list(range(cells)),
        [0.5 * n for n in range(1, cells + 1)],
        dims=dims,
        totals=True,
        tcut=tcut,
        rate=True,
        ci_r=True,
        ci_rr=True,
        scale=1000.0,
        offtable=0.25,
        observations=cells,
    )
    assert result.dims == expanded
    assert result.event == expected_events[:out_cells]
    for i, n in enumerate(expected_n[:out_cells]):
        if tcut and margins[i % 12]:
            assert math.isnan(result.n[i])
        else:
            assert result.n[i] == n
        assert result.pyears[i] == 10.0 * n
        assert result.expected[i] == 0.5 * n
    assert result.rate[11] == pytest.approx(500.0 / 7.0, rel=1e-14)
    assert result.rr[11] == pytest.approx(10.0 / 7.0, rel=1e-14)
    assert result.ci_r_lower[0] == 0.0
    assert result.ci_r_upper[0] == pytest.approx(-math.log(0.025) * 100.0, rel=1e-12)
    assert result.total_events == (15.0 if cells == 6 else 66.0)
    assert result.total_pyears == (210.0 if cells == 6 else 780.0)
    assert result.offtable == 0.25
    assert result.observations == cells
    if cells == 12:
        assert result.rate[23] == pytest.approx(1700.0 / 19.0, rel=1e-14)
        assert result.rr[23] == pytest.approx(34.0 / 19.0, rel=1e-14)
