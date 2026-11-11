"""Every count and axis is frozen from stock R; aggregates use its help-page example."""

import copy
import json
from pathlib import Path

import numpy as np

from .helpers import setup_survival_import

survival = setup_survival_import()
datasets = survival.datasets
REFERENCE = json.loads((Path(__file__).parent / "fixtures/uspop2_reference.json").read_text())


def test_uspop2_preserves_every_count_axis_and_integer_storage():
    population = datasets.load_uspop2()
    assert isinstance(population, datasets.Uspop2Data)
    assert population.shape == (101, 2, 21)
    assert population.dimnames == REFERENCE["dimnames"]
    counts = population.counts
    assert counts.dtype == np.dtype(np.int32)
    expected = np.asarray(REFERENCE["counts"], dtype=np.int32).reshape(REFERENCE["dims"], order="F")
    np.testing.assert_array_equal(counts, expected)


def test_uspop2_standard_population_matches_r_age_group_example():
    counts = datasets.load_uspop2().counts
    groups = np.stack(
        [counts[age : min(age + 5, 101), :, 0].sum(axis=0) for age in range(50, 101, 5)]
    )
    np.testing.assert_array_equal(groups, REFERENCE["age50_groups_2000"]["counts"])
    np.testing.assert_array_equal(counts.sum(axis=(0, 1)), REFERENCE["annual_totals"])
    assert counts.sum() == 6_526_228_928


def test_uspop2_loads_and_snapshots_are_independent_and_copyable():
    population = datasets.load_uspop2()
    original = population.counts
    for clone in (datasets.load_uspop2(), copy.copy(population), copy.deepcopy(population)):
        np.testing.assert_array_equal(clone.counts, original)
        assert clone.dimnames == population.dimnames
        counts = clone.counts
        assert counts.flags.writeable
        counts[0, 0, 0] = -1
        labels = clone.dimnames
        labels[0][0] = "changed"
        np.testing.assert_array_equal(population.counts, original)
        assert clone.counts[0, 0, 0] == 1_965_147
        assert clone.dimnames[0][0] == "0"
    assert datasets.load_uspop2().counts[0, 0, 0] == 1_965_147


def test_every_stock_r_data_object_has_a_loader_or_rate_table():
    loaders = {
        name.removeprefix("load_").casefold()
        for name in datasets.__all__
        if name.startswith("load_")
    }
    tables = {"survexp.mn", "survexp.us", "survexp.usr"}
    assert loaders | tables == {name.casefold() for name in REFERENCE["data_items"]}
    for name in tables:
        table = getattr(survival.r, name.replace(".", "_"))()
        assert survival.r.is_ratetable(table)
