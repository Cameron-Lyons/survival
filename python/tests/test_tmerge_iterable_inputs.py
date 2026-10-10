"""Complete time-dependent frames against independently generated stock R outputs."""

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
RFactor = importlib.import_module("survival.r._coerce")._r_factor
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "tmerge_iterable_reference.json").read_text()
)
CONTAINERS = ["list", "tuple", "iterator", "generator", "array", "numpy_iterator", "pandas"]


class CountedIterator:
    def __init__(self, values, levels=None):
        self.source = iter(values)
        self.observed = []
        if levels is not None:
            self.categories = tuple(levels)

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.source)
        self.observed.append(value)
        return value


class UnreadIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("unused data2 column was consumed")


def column(spec, container):
    values, levels = spec["values"], spec["levels"]
    if container in {"iterator", "generator"}:
        return CountedIterator(values if container == "iterator" else (v for v in values), levels)
    if levels is not None:
        if container == "pandas":
            pd = pytest.importorskip("pandas")
            return pd.Categorical(values, categories=levels)
        return RFactor(values, levels)
    if container == "pandas":
        pd = pytest.importorskip("pandas")
        return pd.Series(values)
    if container in {"array", "numpy_iterator"}:
        dtype = (
            object
            if spec["kind"] == "character" or None in values
            else {"integer": np.int32, "logical": bool}.get(spec["kind"], float)
        )
        array = np.asarray(values, dtype=dtype)
        return CountedIterator(array) if container == "numpy_iterator" else array
    return tuple(values) if container == "tuple" else list(values)


def restored_initial():
    spec = REFERENCE["initial"]
    return survival.r._types.TMergeFrame(
        columns={name: value["values"] for name, value in spec["columns"].items()},
        tname=spec["tname"],
        tevent=spec["tevent"],
        tdcvar=tuple(spec["tdcvar"]),
        tcount=spec["tcount"],
    )


def assert_values(actual, expected):
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None or math.isnan(value)
        else:
            assert value == reference


def assert_frame(actual, expected):
    assert list(actual) == list(expected["columns"])
    for name, spec in expected["columns"].items():
        assert_values(actual[name], spec["values"])
    assert actual.tname == expected["tname"]
    assert actual.tevent == expected["tevent"]
    assert list(actual.tdcvar) == expected["tdcvar"]
    assert actual.tcount == expected["tcount"]
    report = r.summary_tmerge(actual)
    assert report["term"] == list(expected["tcount"])
    for name in report:
        if name != "term":
            assert report[name] == [row[name] for row in expected["tcount"].values()]


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_complete_tmerge_matches_stock_r_with_one_shot_sources(case, container):
    data2 = {name: column(spec, container) for name, spec in case["data2"].items()}
    data1 = (
        restored_initial()
        if case["update"]
        else data2
        if case["same"]
        else {name: column(spec, container) for name, spec in case["data1"].items()}
    )
    original_sources = dict(data2)
    if not case["same"]:
        data2["unused"] = UnreadIterator()
    used = {case["id"]}

    def source(name):
        used.add(name)
        return data2[name] if case["direct"] else name

    operations = {}
    for name, operation in case["operations"].items():
        arguments = [source(operation["time"])]
        if "value" in operation:
            arguments.append(source(operation["value"]))
        options = {"init": operation["init"]} if "init" in operation else {}
        operations[name] = getattr(r, operation["kind"])(*arguments, **options)
    arguments = {"id": source(case["id"]), "operations": operations, "options": case["options"]}
    for name in ("tstart", "tstop"):
        value = case[name]
        if value is not None:
            arguments[name] = source(value) if isinstance(value, str) else value
    # Every retained data1 column must survive even when it aliases an operation vector.
    assert_frame(r.tmerge(data1, data2, **arguments), case["result"])
    assert all(data2[name] is value for name, value in original_sources.items())
    for name, value in original_sources.items():
        if isinstance(value, CountedIterator):
            expected = case["data2"][name]["values"] if name in used or case["same"] else []
            assert_values(value.observed, expected)


@pytest.mark.parametrize("container", ["iterator", "generator", "numpy_iterator"])
def test_direct_first_event_reuses_its_range_and_both_frame_aliases(container):
    case = REFERENCE["cases"][0]
    data = {name: column(spec, container) for name, spec in case["data1"].items()}
    actual = r.tmerge(data, data, id="id", death=r.event(data["time"], data["status"]))
    assert_frame(actual, case["result"])
    for name, value in data.items():
        assert_values(value.observed, case["data1"][name]["values"])


@pytest.mark.parametrize("container", ["list", "iterator", "array"])
@pytest.mark.parametrize("argument", ["time", "value"])
def test_used_update_vectors_still_require_the_full_id_length(container, argument):
    data = {"id": [1, 2], "time": [4, 5], "value": [1, 2]}
    data[argument] = column({"values": [1], "levels": None, "kind": "integer"}, container)
    with pytest.raises(ValueError, match="argument lab is not the same length as id"):
        r.tmerge(restored_initial(), data, id="id", lab=r.tdc("time", "value"))


def test_used_columns_are_reusable_without_requiring_unused_vector_alignment():
    data = {"id": iter([1, 2]), "time": iter([4, 5]), "unused": UnreadIterator()}
    actual = r.tmerge(restored_initial(), data, id=data["id"], visit=r.tdc(data["time"]))
    assert actual["visit"] == [0, 0, 1, 0, 1]


def test_direct_ids_must_match_explicit_dataframe_rows():
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame({"id": [1, 2], "time": [4, 5]})
    with pytest.raises(ValueError, match="id variable not found in data2"):
        r.tmerge(restored_initial(), data, id=iter([1]), visit=r.tdc("time"))


@pytest.mark.parametrize("argument", ["data1", "data2"])
def test_invalid_data_inputs_keep_the_frame_type_error(argument):
    arguments = {"data1": restored_initial(), "data2": {"id": [1]}}
    arguments[argument] = 1
    with pytest.raises(TypeError, match=f"{argument} must be a mapping or data frame"):
        r.tmerge(**arguments, id="id")
