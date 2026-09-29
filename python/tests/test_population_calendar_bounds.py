"""Checked calendar conversion and rate-table validation at native boundaries."""

import math
import sys

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
population = survival.population


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e30, -1e30, 1e12])
def test_days_to_date_rejects_unrepresentable_values(value):
    with pytest.raises(ValueError, match="supported calendar range"):
        population.days_to_date(value)


@pytest.mark.parametrize("year", [-(2**31), -400, 0, 2000, 2**31 - 1])
@pytest.mark.parametrize(("month", "day"), [(1, 1), (12, 31)])
def test_calendar_date_roundtrip_at_year_boundaries(year, month, day):
    days = population.ratetable_date(year, month, day)
    result = population.days_to_date(days + 0.5)
    assert (result.year, result.month, result.day) == (year, month, day)


@pytest.mark.parametrize("kind", [3, 4])
def test_calendar_cutpoints_and_positions_are_checked(kind):
    args = ([1], ["year"], [["year"]], [[1e30]], [kind], [0.1])
    with pytest.raises(ValueError, match="supported calendar range"):
        population.RateTable(*args)
    check = population.is_ratetable(*args[:-1], n_rates=1)
    assert not check.valid
    table = population.RateTable([1], ["year"], [["year"]], [[0]], [kind], [0.1])
    with pytest.raises(ValueError, match="supported calendar range"):
        population.match_ratetable(table, ["year"], [[1e30]])


def test_rate_table_shape_checks_do_not_overflow():
    check = population.is_ratetable([sys.maxsize, 3], [], [], [], [], 0)
    assert not check.valid
    assert "ratetable dimensions are too large" in check.messages
    with pytest.raises(ValueError, match="must be positive"):
        population.RateTable([0], ["age"], [[]], [[]], [2], [])


@pytest.mark.parametrize("age", [1e30, -1e30])
def test_expected_survival_checks_derived_birth_dates(age):
    with pytest.raises(ValueError, match="supported calendar range"):
        population.survexp(population.survexp_us(), [[age, 1, 0]], times=[1])
