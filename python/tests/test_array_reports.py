"""Raw response and rate-array reports against captured R output."""

import json
import math
import pickle
from pathlib import Path

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "array_report_reference.json").read_text()
)


@pytest.fixture(scope="module")
def objects():
    from survival.r._coerce import _RFactorVector

    def decode(value):
        if isinstance(value, dict):
            return _RFactorVector(value["values"], value["levels"])
        if isinstance(value, list):
            return [decode(item) for item in value]
        return {"NA": None, "Inf": math.inf, "-Inf": -math.inf}.get(value, value)

    result = {}
    for name, spec in REFERENCE["objects"].items():
        kind = spec["kind"]
        if kind == "builtin":
            result[name] = getattr(r, spec["name"].replace(".", "_"))()
        elif kind == "ratetable":
            result[name] = r.RateTable(
                spec["dims"],
                spec["dimid"],
                spec["dimnames"],
                [None] * len(spec["dims"]),
                [1] * len(spec["dims"]),
                spec["rates"],
            )
        else:
            result[name] = getattr(r, kind)(
                **{key: decode(value) for key, value in spec["arguments"].items()}
            )
    return result


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_array_reports_match_r(case, objects, capsys):
    value = objects[case["object"]]
    kwargs = dict(case["arguments"])
    kwargs["width"] = case["width"]
    if case["row_names"] is not None:
        kwargs["names"] = case["row_names"]
    function = (
        r.print_surv
        if isinstance(value, r.Surv)
        else (r.print_surv2 if isinstance(value, r.Surv2) else r.print_ratetable)
    )
    report = function(value, **kwargs)
    assert report.lines == case["lines"]
    assert str(report) == "\n".join(case["lines"]) + "\n"
    if case["labels"] is not None:
        assert report.labels == case["labels"]
    assert capsys.readouterr().out == ""


def test_full_precision_frames_and_independent_ownership(objects):
    source = objects["array"]
    report = r.print_ratetable(source, digits=2, max_print=3)
    frame = r.as_data_frame(report)
    assert report.displayed == 3
    assert frame == {
        "age": ["r1", "r2"] * 6,
        "sex": ["one", "one", "two", "two", "three", "three"] * 2,
        "year": ["a"] * 6 + ["b"] * 6,
        "rate": source.rates,
    }
    frame["rate"][0] = 999
    frame["age"][0] = "changed"
    assert report.rates == source.rates
    assert report.dimnames == source.dimnames
    restored = pickle.loads(pickle.dumps(report))  # noqa: S301 - own test data
    assert restored == report
    report.rates[0] = 1000
    report.dimnames[0][0] = "different"
    assert source.rates[0] != 1000
    assert source.dimnames[0][0] == "r1"


@pytest.mark.parametrize("name", ["right", "counting", "interval", "timeline_states"])
def test_response_frames_preserve_hidden_data(name, objects):
    value = objects[name]
    function = r.print_surv2 if isinstance(value, r.Surv2) else r.print_surv
    report = function(value, max_print=1)
    frame = r.as_data_frame(report)
    assert len(frame["status"]) == len(value.time)
    assert frame.get("time", frame.get("stop")) == list(value.time)
    if name == "right":
        assert frame["time"][0] == 1.111111111
    frame["status"][0] = 999
    assert report.data["status"][0] != 999


def test_empty_response_and_named_frame():
    report = r.print_surv(r.Surv([], []))
    assert str(report) == "character(0)\n"
    assert report.displayed == 0
    assert r.as_data_frame(report) == {"time": [], "status": [], "type": []}
    names = ["a", "b", "c"]
    report = r.print_surv2(r.Surv2([0, 1, 2], [0, 1, 0]), names=names, max_print=1)
    names[0] = "changed"
    assert r.as_data_frame(report)["name"] == ["a", "b", "c"]


@pytest.mark.parametrize("ids", [["rate", "rate.1"], ["rate", "rate"]])
def test_rate_column_name_collision(ids):
    value = r.RateTable([1, 1], ids, [["a"], ["b"]], [None, None], [1, 1], [0.1])
    assert r.as_data_frame(r.print_ratetable(value)) == {
        "rate": ["a"],
        "rate.1": ["b"],
        "rate.2": [0.1],
    }


@pytest.mark.parametrize("function", [r.print_surv, r.print_surv2, r.print_ratetable])
def test_invalid_object(function):
    with pytest.raises(TypeError):
        function(None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_print": -1},
        {"max_print": 1.5},
        {"max_print": math.inf},
        {"max_print": 2147483647},
        {"width": 9},
        {"names": ["a"]},
        {"names": [1, 2]},
        {"quote": "yes"},
        {"right": []},
    ],
)
def test_invalid_response_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        r.print_surv(r.Surv([1, 2], [1, 0]), **kwargs)


def test_duplicate_and_unknown_options(objects):
    with pytest.raises(ValueError, match="use only one"):
        r.print_ratetable(objects["array"], max_print=2, **{"max": 3})
    with pytest.raises(TypeError):
        r.print_surv(objects["right"], unknown=True)


def test_rate_rendering_only_formats_displayed_slices(objects, monkeypatch):
    from survival.r import _array_print

    original = _array_print._r_format_numbers
    lengths = []

    def counted(values, digits):
        lengths.append(len(values))
        return original(values, digits)

    monkeypatch.setattr(_array_print, "_r_format_numbers", counted)
    report = r.print_ratetable(objects["survexp.us"], max_print=6)
    assert report.displayed == 6
    # R uses every row of each visible column to choose precision, but no hidden slices.
    assert lengths == [report.dims[0]] * report.dims[1]
    assert len(report.rates) > sum(lengths)
