"""Independent risk-set/rank checks and bulk expansion boundaries."""

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from survival import r, validation


def _reference(time, status, columns, start=None, strata=None):
    if strata is None:
        events = [(at, None) for at in sorted(set(time[status == 1]))]
    else:
        events = list(dict.fromkeys(zip(time[status == 1], strata[status == 1], strict=True)))
    rows, blocks, outcomes, transformed, offsets = [], [], [], [[] for _ in columns], [0]
    for block, (at, group) in enumerate(events, 1):
        selected = time >= at
        if start is not None:
            selected &= start < at
        if strata is not None:
            selected &= strata == group
        index = np.flatnonzero(selected)
        rows.extend(index)
        outcomes.extend((time[index] == at) & (status[index] == 1))
        blocks.extend([block] * len(index))
        offsets.append(len(rows))
        for destination, values in zip(transformed, columns, strict=True):
            values = values[index]
            valid = values[~np.isnan(values)]
            for value in values:
                if np.isnan(value):
                    destination.append(np.nan)
                else:
                    rank = (
                        np.count_nonzero(valid < value) + (np.count_nonzero(valid == value) + 1) / 2
                    )
                    p = (rank - 0.5) / len(valid)
                    destination.append(np.log(p / (1 - p)))
    return rows, blocks, outcomes, transformed, offsets, [at for at, _ in events]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("stratified", [False, True])
def test_sweeps_match_independent_risk_sets_and_pairwise_ranks(seed, counting, stratified):
    rng = np.random.default_rng(seed)
    n = 31
    time = rng.integers(-2, 8, n).astype(float)
    start = time - rng.integers(1, 6, n) if counting else None
    status = rng.integers(0, 2, n, dtype=np.int32)
    strata = rng.choice([-7, 2, 19], n).astype(np.int32) if stratified else None
    columns = rng.choice([np.nan, -np.inf, -2, -0.0, 0.0, 1.0, 3.0, np.inf], (3, n))
    expected = _reference(time, status, columns, start, strata)
    result = validation.survobrien(time, status, columns, start=start, strata=strata)
    assert result.row == expected[0]
    assert result.strata == expected[1]
    assert result.status == expected[2]
    assert_allclose(result.transformed, expected[3], atol=1e-14)
    assert result.block_offsets == expected[4]
    assert result.event_times == expected[5]
    assert_array_equal(result.time, time[result.row])
    if counting:
        assert_array_equal(result.start, start[result.row])


@pytest.mark.parametrize("layout", ["list", "strided", "fortran", "float32"])
def test_numpy_inputs_and_owned_output_snapshots(layout):
    time = np.array([3, 1, 4, 2, 3.0])
    status = np.array([1, 1, 0, 1, 1])
    values = np.array([[1, 5, 3, 2, 2.0], [-2, 7, 1, 3, 4.0]])
    if layout == "list":
        values = values.tolist()
    elif layout == "strided":
        values = np.repeat(values, 2, axis=1)[:, ::2]
    elif layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "float32":
        values = values.astype(np.float32)
    original = np.array(values, copy=True)
    result = validation.survobrien(time, status, values)
    arrays = result.to_arrays()
    assert_array_equal(values, original)
    for name in ("row", "time", "status", "strata", "event_times", "block_offsets"):
        assert isinstance(arrays[name], np.ndarray)
        assert_array_equal(arrays[name], getattr(result, name))
    assert arrays["start"] is None
    assert_allclose(arrays["transformed"], result.transformed)
    arrays["row"][:] = -1
    arrays["transformed"][0][:] = 123
    assert result.row[0] == 0
    assert result.transformed[0][0] != 123
    del result
    gc.collect()
    assert_array_equal(arrays["row"], -1)


def test_geometry_only_needs_no_continuous_columns():
    result = validation.survobrien([2, 1, 3], [1, 1, 0], [], transform=False)
    assert result.row == [0, 1, 2, 0, 2]
    assert result.block_offsets == [0, 3, 5]
    assert result.transformed == []
    empty = validation.survobrien([2, 1, 3], [0, 0, 0], [[1, 2, 3]])
    assert empty.row == []
    assert empty.event_times == []
    assert empty.block_offsets == [0]
    assert empty.transformed == [[]]


def test_signed_zero_is_one_event_and_one_rank_tie():
    result = validation.survobrien(
        [-0.0, 0.0, 1.0], [1, 1, 0], [[-0.0, 0.0, 1.0]], strata=[1, 1, 1]
    )
    assert result.event_times == [-0.0]
    assert result.status == [1, 1, 0]
    assert result.transformed[0][0] == result.transformed[0][1]


def test_custom_transform_materializes_source_rows_once(monkeypatch):
    from survival.r import _misc

    original = _misc._core.survobrien
    reads = []

    class Counted:
        def __init__(self, result):
            self.result = result

        def __getattr__(self, name):
            if name == "row":
                reads.append(name)
            return getattr(self.result, name)

    def expand(*args, **kwargs):
        assert kwargs["transform"] is False
        assert args[2] == []
        return Counted(original(*args, **kwargs))

    monkeypatch.setattr(_misc._core, "survobrien", expand)
    data = {"time": [3, 1, 4, 2], "status": [1, 1, 0, 1], "x": [5, 4, 3, 2]}
    batches = []

    def transform(values):
        batches.append(list(values))
        return np.array(values) * 2

    result = r.survobrien("Surv(time, status) ~ x", data=data, transform=transform)
    assert reads == ["row"]
    assert batches == [list(range(1, 11)), [5, 4, 3, 2], [5, 3, 2], [5, 3]]
    assert result["x"] == [10, 8, 6, 4, 10, 6, 4, 10, 6]


def test_concurrent_expansions_keep_independent_active_sets():
    time = np.arange(300.0)
    status = np.ones(300, dtype=np.int32)
    values = np.sin(time)[None, :]

    def run(seed):
        return validation.survobrien(time, status, values + seed).transformed

    expected = run(0)
    with ThreadPoolExecutor(4) as pool:
        for result in pool.map(run, range(4)):
            assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([], [], []), r"No \(non-missing\) observations"),
        (([1], [2], [[3]]), "status"),
        (([1], [1], [[3, 4]]), "continuous"),
        (([np.nan], [1], [[3]]), "time"),
    ],
)
def test_invalid_geometry_is_rejected(arguments, message):
    with pytest.raises(ValueError, match=message):
        validation.survobrien(*arguments)


def test_formula_na_pass_keeps_missing_continuous_values():
    data = {"time": [1, 2, 3], "status": [1, 1, 1], "x": [None, 2, 1]}
    result = r.survobrien("Surv(time, status) ~ x", data, na_action="na.pass")
    assert result[".id."] == [1, 2, 3, 2, 3, 3]
    assert_allclose(result["x"], [np.nan, np.log(3), -np.log(3), np.log(3), -np.log(3), 0])
