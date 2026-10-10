"""Aggregate callback order, scalar summaries and sum against stock survival."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r_api
sa = survival.surv_analysis
from survival.r._types import (  # noqa: E402
    CoxSurvfitMultiStateResult,
    CoxSurvfitResult,
    NamedMatrix,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aggregate_fun_reference.json").read_text()
)
CASES = REFERENCE["cases"]


def _array(spec):
    return (
        None
        if spec is None
        else np.asarray(spec["values"], dtype=float).reshape(spec["shape"], order="F")
    )


def _by(spec, native=False):
    if spec is None:
        return [] if native else None
    if spec["kind"] == "mapping":
        columns = list(spec["columns"].items())
    else:
        columns = [(None, spec["values"])]
    if native:
        factors = []
        for name, values in columns:
            levels = spec.get("levels") or sorted(set(values))
            labels = [str(value) for value in levels]
            factors.append(
                sa.GroupingFactor([levels.index(value) for value in values], labels, name)
            )
        return factors
    if spec["kind"] == "mapping":
        # Each input must be consumed exactly once by the public grouping boundary.
        return {name: iter(values) for name, values in columns}
    if spec["kind"] == "factor":
        return RFactor(spec["values"], spec["levels"])
    return iter(spec["values"])


def _groups(value):
    if value is None or isinstance(value, dict):
        return value
    return {name: [row[column] for row in value.labels] for column, name in enumerate(value.names)}


def _callback(kind, calls):
    def callback(values, **kwargs):
        vector = np.asarray(values, dtype=float)
        calls.append((vector.copy(), kwargs))
        if kind == "sum":
            return np.sum(vector)
        if kind == "affine":
            return vector[0] + 2 * vector[-1] + len(vector) + np.sum(vector) / 8
        if kind == "span":
            return np.max(vector) - np.min(vector)
        if kind == "last":
            return vector[-1]
        if kind == "nanmean":
            finite = vector[~np.isnan(vector)]
            return np.mean(finite) if finite.size else np.nan
        if kind == "boolean":
            return True
        if kind == "string":
            return "1"
        if kind == "vector":
            return np.array([1.0, 2.0])
        if kind == "list":
            return [1.0]
        if kind == "required":

            def needs_scale(vector, scale):
                return scale * np.sum(vector)

            return needs_scale(vector, **kwargs)
        raise AssertionError(kind)

    return callback


def _assert_result(actual, expected):
    for field in ("surv", "pstate"):
        values = getattr(actual, field, None)
        if expected[field] is None:
            assert values is None
        else:
            np.testing.assert_allclose(
                values, _array(expected[field]), rtol=3e-14, atol=0, equal_nan=True, err_msg=field
            )
    expected_groups = expected["newdata"]
    if expected_groups is not None:
        expected_groups = {
            name: [str(value) for value in values] for name, values in expected_groups.items()
        }
    assert _groups(actual.newdata) == expected_groups


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("interface", ["native", "facade"])
@pytest.mark.parametrize("container", ["list", "numpy"])
def test_callback_and_named_summary_match_stock_r(case, interface, container):
    fields = {name: _array(spec) for name, spec in REFERENCE["inputs"][case["input"]].items()}
    if container == "list":
        fields = {
            name: None if values is None else values.tolist() for name, values in fields.items()
        }
    calls = []
    summary = (
        None if case["default"] else case["fun"] if case["named"] else _callback(case["fun"], calls)
    )

    def aggregate():
        if interface == "native":
            return sa.aggregate_survfit(
                **fields, by=_by(case["by"], True), fun=summary, **(case["dots"] or {})
            )
        return r.aggregate_survfit(
            SimpleNamespace(**fields), by=_by(case["by"]), FUN=summary, **(case["dots"] or {})
        )

    expected = case["expected"]
    if "error" in expected:
        if case["fun"] == "required":
            with pytest.raises(TypeError, match="scale"):
                aggregate()
        else:
            with pytest.raises(ValueError, match="FUN must return a single value summary"):
                aggregate()
    else:
        _assert_result(aggregate(), expected)
    assert len(calls) == len(case["calls"])
    for (values, kwargs), expected_call in zip(calls, case["calls"], strict=True):
        np.testing.assert_array_equal(values, np.asarray(expected_call["values"], dtype=float))
        # Stock aggregate.survfit accepts ... but never passes it to FUN.
        assert not kwargs
        assert not expected_call["dots"]


def _ordinary_curve():
    common = REFERENCE["common"]
    return CoxSurvfitResult(
        **common,
        surv=_array(REFERENCE["inputs"]["ordinary"]["surv"]).tolist(),
        cumhaz=[[0.3] * 4 for _ in range(3)],
        type="right",
        std_err=[[0.1] * 4 for _ in range(3)],
        std_chaz=[[0.2] * 4 for _ in range(3)],
        lower=[[0.01] * 4 for _ in range(3)],
        upper=[[0.99] * 4 for _ in range(3)],
        logse=True,
        conf_int=0.95,
        conf_type="log",
        newdata={"x": [1, 2, 3, 4]},
        colnames=["one", "two", "three", "four"],
    )


def _coxms_curve():
    common = REFERENCE["common"]
    common = {
        **common,
        **{
            name: [[value, 0.0] for value in common[name]]
            for name in ("n_risk", "n_event", "n_censor")
        },
    }
    return CoxSurvfitMultiStateResult(
        **common,
        n_transition=[[1.0], [2.0], [3.0]],
        n_id=[8],
        pstate=_array(REFERENCE["inputs"]["multistate"]["pstate"]),
        cumhaz=np.full((3, 4, 1), 0.3),
        cumhaz_names=["(s0):absorbed"],
        p0=[[0.8, 0.2]],
        states=["(s0)", "absorbed"],
        transitions=NamedMatrix(["(s0)", "absorbed"], ["(s0)", "absorbed"], [[0, 1], [0, 0]]),
        type="mright",
        t0=0,
        newdata={"x": [1, 2, 3, 4]},
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if case["input"] in {"ordinary", "multistate"} and "error" not in case["expected"]
    ],
    ids=lambda case: case["name"],
)
def test_public_copy_preserves_curve_metadata(case):
    source = _ordinary_curve() if case["input"] == "ordinary" else _coxms_curve()
    original = source.surv if isinstance(source, CoxSurvfitResult) else source.pstate.copy()
    calls = []
    actual = r.aggregate_survfit(
        source,
        by=_by(case["by"]),
        FUN=None
        if case["default"]
        else case["fun"]
        if case["named"]
        else _callback(case["fun"], calls),
    )
    assert type(actual) is type(source)
    _assert_result(actual, case["expected"])
    dropped = {
        "surv",
        "pstate",
        "cumhaz",
        "newdata",
        "colnames",
        "std_err",
        "std_chaz",
        "lower",
        "upper",
        "logse",
        "conf_int",
        "conf_type",
    }
    for field in dataclasses.fields(source):
        if field.name not in dropped:
            assert getattr(actual, field.name) is getattr(source, field.name), field.name
    if isinstance(source, CoxSurvfitResult):
        assert actual.cumhaz == []
        assert all(
            getattr(actual, field) is None
            for field in ("std_err", "std_chaz", "lower", "upper", "logse", "conf_int", "conf_type")
        )
        np.testing.assert_array_equal(source.surv, original)
        assert source.std_err is not None
    else:
        assert actual.cumhaz is None
        assert actual.pstate.shape == tuple(case["expected"]["pstate"]["shape"])
        np.testing.assert_array_equal(source.pstate, original)
        assert source.cumhaz is not None
    assert source.newdata == {"x": [1, 2, 3, 4]}


@pytest.mark.parametrize(
    "scalar",
    [
        2,
        2.0,
        np.int64(2),
        np.float64(2),
        np.array(2.0),
        np.array([2.0]),
        np.array([[2.0]]),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_callback_accepts_one_real_numeric_value(scalar, interface):
    fields = REFERENCE["inputs"]["ordinary"]
    surv = _array(fields["surv"])
    if interface == "native":
        actual = sa.aggregate_survfit(surv=surv, fun=lambda values: scalar)
    else:
        actual = r.aggregate_survfit(SimpleNamespace(surv=surv), FUN=lambda values: scalar)
    np.testing.assert_array_equal(actual.surv, np.full((3, 1), np.asarray(scalar).item()))


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(True),
        np.array([True]),
        "2",
        None,
        [2],
        (2,),
        np.array([1.0, 2.0]),
        1 + 2j,
        np.array([1 + 2j]),
    ],
)
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_callback_rejects_non_real_or_non_scalar_values(invalid, interface):
    # Stock R's is.numeric gate rejects logical, complex and list summaries.
    surv = _array(REFERENCE["inputs"]["ordinary"]["surv"])

    def aggregate():
        if interface == "native":
            return sa.aggregate_survfit(surv=surv, fun=lambda values: invalid)
        return r.aggregate_survfit(SimpleNamespace(surv=surv), FUN=lambda values: invalid)

    with pytest.raises(ValueError, match="FUN must return a single value summary"):
        aggregate()


@pytest.mark.parametrize("interface", ["native", "facade"])
def test_callback_exceptions_propagate_and_stop_at_the_failing_group(interface):
    failure = RuntimeError("summary intentionally failed")
    calls = []

    def summary(values):
        calls.append(list(values))
        raise failure

    surv = _array(REFERENCE["inputs"]["ordinary"]["surv"])

    def aggregate():
        if interface == "native":
            return sa.aggregate_survfit(
                surv=surv, by=[sa.GroupingFactor([1, 0, 1, 0], ["a", "b"])], fun=summary
            )
        return r.aggregate_survfit(SimpleNamespace(surv=surv), by=["b", "a", "b", "a"], FUN=summary)

    with pytest.raises(RuntimeError) as caught:
        aggregate()
    assert caught.value is failure
    assert calls == [[2, 4]]


@pytest.mark.parametrize("interface", ["native", "facade"])
def test_callback_vectors_are_independent_of_source_and_previous_calls(interface):
    surv = _array(REFERENCE["inputs"]["ordinary"]["surv"])
    original = surv.copy()
    retained = []

    def summary(values):
        retained.append(values)
        result = sum(values)
        values[0] = -999
        return result

    if interface == "native":
        actual = sa.aggregate_survfit(surv=surv, fun=summary)
    else:
        actual = r.aggregate_survfit(SimpleNamespace(surv=surv), FUN=summary)
    np.testing.assert_array_equal(surv, original)
    np.testing.assert_allclose(actual.surv, np.sum(original, axis=1)[:, None], rtol=1e-14)
    assert len(retained) == 4
    assert all(values[0] == -999 for values in retained)
    for values, row in zip(retained[1:], original, strict=True):
        np.testing.assert_array_equal(values[1:], row[1:])


@pytest.mark.parametrize("interface", ["native", "facade"])
def test_callback_keeps_the_scalar_contract_after_validation(interface):
    # R only checks the initial index-vector calls and allows apply to change
    # shape afterward. The port consistently enforces its real scalar output.
    calls = []

    def summary(values):
        calls.append(list(values))
        return 1.0 if len(calls) == 1 else np.array([1.0, 2.0])

    surv = _array(REFERENCE["inputs"]["ordinary"]["surv"])

    def aggregate():
        if interface == "native":
            return sa.aggregate_survfit(surv=surv, fun=summary)
        return r.aggregate_survfit(SimpleNamespace(surv=surv), FUN=summary)

    with pytest.raises(ValueError, match="FUN must return a single value summary"):
        aggregate()
    assert len(calls) == 2


def test_empty_time_numpy_curve_keeps_group_metadata_with_callback():
    source = dataclasses.replace(
        _ordinary_curve(),
        time=[],
        n_risk=[],
        n_event=[],
        n_censor=[],
        surv=np.empty((0, 4)),
        cumhaz=np.empty((0, 4)),
        std_err=None,
        std_chaz=None,
        lower=None,
        upper=None,
        strata=None,
    )
    calls = []
    result = r.aggregate_survfit(source, by=["b", "a", "b", "a"], FUN=_callback("sum", calls))
    assert result.surv == []
    assert result.colnames == ["1", "2"]
    assert result.newdata == {"aggregate": ["a", "b"]}
    assert result.ncurve == 2
    # Stock apply also probes a zero vector for this empty axis. The port's
    # well-shaped empty results avoid these additional value callbacks.
    assert [values.tolist() for values, _ in calls] == [[2, 4], [1, 3]]
    assert source.surv.shape == (0, 4)


@pytest.mark.parametrize(("margin", "shape"), [("surv", (3, 0)), ("pstate", (3, 0, 2))])
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_empty_data_axis_fails_before_invoking_callback(margin, shape, interface):
    values = np.empty(shape)
    calls = []

    def aggregate():
        if interface == "native":
            return sa.aggregate_survfit(**{margin: values}, fun=_callback("sum", calls))
        return r.aggregate_survfit(SimpleNamespace(**{margin: values}), FUN=_callback("sum", calls))

    with pytest.raises(ValueError, match="data.*margin"):
        aggregate()
    assert not calls


@pytest.mark.parametrize("summary", [sum, np.sum, np.mean, np.median, np.min, np.max])
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_builtin_and_numpy_callable_summaries(summary, interface):
    surv = _array(REFERENCE["inputs"]["ordinary"]["surv"])
    if interface == "native":
        result = sa.aggregate_survfit(surv=surv, fun=summary)
    else:
        result = r.aggregate_survfit(SimpleNamespace(surv=surv), FUN=summary)
    np.testing.assert_allclose(result.surv, [[summary(row)] for row in surv], rtol=1e-14)


@pytest.mark.parametrize("axis", ["time", "state"])
def test_empty_multistate_curve_keeps_the_probability_array_shape(axis):
    source = _coxms_curve()
    if axis == "time":
        source = dataclasses.replace(
            source,
            time=[],
            n_risk=[],
            n_event=[],
            n_censor=[],
            n_transition=[],
            pstate=np.empty((0, 4, 2)),
            cumhaz=np.empty((0, 4, 1)),
            strata=None,
        )
        expected_shape = (0, 2, 2)
    else:
        source = dataclasses.replace(
            source,
            states=[],
            pstate=np.empty((3, 4, 0)),
            cumhaz=None,
            n_transition=None,
            p0=[[]],
            transitions=None,
        )
        expected_shape = (3, 2, 0)
    calls = []
    result = r.aggregate_survfit(source, by=["b", "a", "b", "a"], FUN=_callback("sum", calls))
    assert result.pstate.shape == expected_shape
    assert result.newdata == {"aggregate": ["a", "b"]}
    assert result.cumhaz is None
    assert [values.tolist() for values, _ in calls] == [[2, 4], [1, 3]]


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["default"]], ids=lambda case: case["name"]
)
@pytest.mark.parametrize("interface", ["native", "facade"])
def test_omitted_summary_uses_stock_default_row_means(case, interface):
    fields = {name: _array(spec) for name, spec in REFERENCE["inputs"][case["input"]].items()}
    if interface == "native":
        result = sa.aggregate_survfit(**fields, by=_by(case["by"], True))
    else:
        result = r.aggregate_survfit(SimpleNamespace(**fields), by=_by(case["by"]))
    _assert_result(result, case["expected"])


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if case["input"] in {"ordinary", "multistate", "both"}
        and case["by"] is not None
        and case["by"]["kind"] == "mapping"
    ],
    ids=lambda case: case["name"],
)
def test_list_of_numpy_grouping_vectors_matches_stock_compound_groups(case):
    fields = {name: _array(spec) for name, spec in REFERENCE["inputs"][case["input"]].items()}
    by = [np.asarray(values) for values in case["by"]["columns"].values()]
    calls = []
    result = r.aggregate_survfit(
        SimpleNamespace(**fields),
        by=by,
        FUN=case["fun"] if case["named"] else _callback(case["fun"], calls),
    )
    expected = {
        **case["expected"],
        "newdata": {
            f"Group.{index}": values
            for index, values in enumerate(case["expected"]["newdata"].values(), 1)
        },
    }
    _assert_result(result, expected)


@pytest.mark.parametrize("case", REFERENCE["default_mean_evidence"], ids=lambda case: case["name"])
@pytest.mark.parametrize("interface", ["native", "facade"])
@pytest.mark.parametrize("default_argument", ["omitted", "none"])
def test_default_survival_mean_differs_from_explicit_mean_and_pstate_mean(
    case, interface, default_argument
):
    fields = {name: _array(case[name]) for name in ("surv", "pstate")}
    options = (
        {}
        if case["default"] and default_argument == "omitted"
        else {"fun" if interface == "native" else "FUN": None if case["default"] else "mean"}
    )
    if interface == "native":
        result = sa.aggregate_survfit(**fields, by=_by(case["by"], True), **options)
    else:
        result = r.aggregate_survfit(SimpleNamespace(**fields), by=_by(case["by"]), **options)
    assert REFERENCE["metadata"]["long_double_mantissa"] == 64
    _assert_result(result, case["expected"])
