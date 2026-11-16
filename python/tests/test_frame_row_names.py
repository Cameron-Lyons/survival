"""Row names of formula data: the index fast paths and the selected-row names.

``_data_row_labels`` reads integer and plain-string pandas indexes without a call per
label, ``data[rows, ]`` names increasing rows without ``make.unique``, and
``make.unique`` returns distinct names unchanged.  Each must give exactly what the
per-value algorithm (kept here as the reference) gives.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_coerce = importlib.import_module("survival.r._coerce")
r_formula = importlib.import_module("survival.r._formula")
r_names = importlib.import_module("survival.r._names")


def reference_data_row_labels(data, n):
    labels = getattr(data, "row_names", None)
    if labels is not None:
        return tuple(labels)
    index = None if isinstance(data, Mapping) else getattr(data, "index", None)
    if index is None or (
        type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1
    ):
        return None
    labels = tuple(r_coerce._as_character(value) for value in index)
    if (
        len(labels) == n
        and len(set(labels)) == n
        and not any(map(r_coerce._is_missing_value, index))
    ):
        return labels
    return None


def reference_make_unique(names):
    taken = set(names)
    seen = set()
    counts = {}
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
            continue
        count = counts.get(name, 1)
        while f"{name}.{count}" in taken:
            count += 1
        unique = f"{name}.{count}"
        taken.add(unique)
        counts[name] = count + 1
        out.append(unique)
    return out


def reference_selected_names(labels, rows):
    return tuple(
        reference_make_unique(
            ["NA" if row < 0 else str(row + 1) if labels is None else labels[row] for row in rows]
        )
    )


def _indexes(pd, n):
    rng = np.random.default_rng(5)
    shuffled = rng.permutation(n) * 3 + 7
    words = [f"r{i}" for i in range(n)]
    yield "int64", pd.Index(shuffled)
    yield "int32", pd.Index(shuffled.astype(np.int32))
    yield "int8", pd.Index(np.arange(n, dtype=np.int8) - 5)
    yield "uint64", pd.Index(np.arange(n, dtype=np.uint64) + np.uint64(2**63))
    yield "duplicate_ints", pd.Index(np.arange(n) // 2)
    yield "range_start", pd.RangeIndex(1, n + 1)
    yield "range_step", pd.RangeIndex(0, 2 * n, 2)
    yield "range_reverse", pd.RangeIndex(n, 0, -1)
    yield "nullable_int", pd.Index(pd.array(list(range(n)), dtype="Int64"))
    yield "nullable_int_na", pd.Index(pd.array([None, *range(1, n)], dtype="Int64"))
    yield "str", pd.Index(words)
    yield "str_duplicates", pd.Index([f"r{i // 2}" for i in range(n)])
    yield "str_nan", pd.Index([*words[:-1], None])
    yield "object_str", pd.Index(words, dtype=object)
    yield "object_none", pd.Index([*words[:-1], None], dtype=object)
    yield "na_label", pd.Index(["NA", *words[1:]], dtype=object)
    yield "mixed_int_str", pd.Index([1, "1", *words[2:]], dtype=object)
    yield "numpy_str", pd.Index([np.str_(word) for word in words], dtype=object)
    yield "unicode", pd.Index([f"é{i}ß" for i in range(n)])
    yield "empty_string", pd.Index(["", *words[1:]])
    yield "object_ints", pd.Index(list(range(n)), dtype=object)
    yield "object_bools", pd.Index([i % 2 == 0 for i in range(n)], dtype=object)
    yield "bool", pd.Index(np.arange(n) % 2 == 0)
    yield "float", pd.Index(np.arange(n) + 0.5)
    yield "integral_float", pd.Index(np.arange(n, dtype=float))
    yield "float_nan", pd.Index([math.nan, *np.arange(1.0, n)])
    yield "categorical", pd.CategoricalIndex(words)
    yield "datetime", pd.date_range("2020-01-01", periods=n)
    yield "multi", pd.MultiIndex.from_arrays([np.arange(n), words])
    yield "string_python", pd.Index(words, dtype="string[python]")
    yield "string_python_na", pd.Index([*words[:-1], None], dtype="string[python]")


def test_index_labels_match_the_per_value_labels():
    pd = pytest.importorskip("pandas")
    n = 12
    for name, index in _indexes(pd, n):
        frame = pd.DataFrame({"x": np.arange(n)}, index=index)
        for rows in (n, n - 1):
            expected = reference_data_row_labels(frame, rows)
            actual = r_formula._data_row_labels(frame, rows)
            assert actual == expected, name
            assert actual is None or all(type(label) is str for label in actual), name
    for index in (pd.Index([], dtype=np.int64), pd.Index([], dtype=object)):
        frame = pd.DataFrame({"x": []}, index=index)
        assert r_formula._data_row_labels(frame, 0) == reference_data_row_labels(frame, 0)


def test_fits_on_filtered_sliced_and_string_indexed_frames_keep_their_row_names():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(11)
    n = 300
    base = pd.DataFrame(
        {
            "time": rng.exponential(size=n).round(2) + 0.01,
            "status": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
        }
    )
    base.loc[[3, 40], "x"] = math.nan
    named = base.set_axis([f"id{i}" for i in range(n)])
    frames = {
        "filtered": base[base.x.fillna(0) > -1.5],
        "sliced": base.iloc[7:],
        "strided": base.iloc[::2],
        "sorted": base.sort_values("time"),
        "named": named,
        "named_filtered": named[named.x.fillna(0) < 1.5],
    }
    for name, frame in frames.items():
        labels = reference_data_row_labels(frame, len(frame))
        assert labels is not None, name
        cox = r.coxph("Surv(time, status) ~ x", frame, na_action="na.exclude")
        keep = [i for i, value in enumerate(frame.x) if not math.isnan(value)]
        assert cox._prediction_rows == labels, name
        matrix_names = r.model_matrix(cox, _with_metadata=True)["row_names"]
        assert matrix_names == [labels[i] for i in keep], name
        aft = r.survreg("Surv(time, status) ~ x", frame)
        assert aft._prediction_rows == labels, name


def test_selected_row_names_match_make_unique_of_the_selected_names():
    rng = np.random.default_rng(3)
    n = 40
    unique_labels = tuple(f"r{i}" for i in range(n))
    clashing_labels = tuple(f"r{i % 7}" if i % 3 else f"r{i % 7}.1" for i in range(n))
    for labels in (None, unique_labels, clashing_labels):
        cases = [
            [],
            [5],
            sorted(rng.choice(n, 25, replace=False).tolist()),
            list(range(n)),
            rng.choice(n, 25, replace=False).tolist(),
            rng.integers(n, size=30).tolist(),
            sorted(rng.integers(n, size=30).tolist()),
            [-1, 3, -1, 4],
            [2, 2, 3],
        ]
        for rows in cases:
            index = np.asarray(rows, dtype=np.intp)
            assert r_formula._selected_row_names(labels, rows, index) == reference_selected_names(
                labels, rows
            ), (labels and labels[:2], rows)


def test_make_unique_matches_r_make_unique():
    rng = np.random.default_rng(7)
    pool = ["a", "b", "a.1", "a.2", "b.1", "c", "NA", ""]
    for size in range(0, 30):
        names = [pool[i] for i in rng.integers(len(pool), size=size).tolist()]
        assert r_names._make_unique(names) == reference_make_unique(names)
    distinct = [f"n{i}" for i in range(50)]
    assert r_names._make_unique(distinct) == distinct
    assert r_names._make_unique(("x", "x", "x.1")) == ["x", "x.2", "x.1"]
