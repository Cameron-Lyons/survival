"""R's missing selections enter the shared model frame as missing rows."""

import copy
import importlib
import pickle

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
formula = importlib.import_module("survival.r._formula")
coerce = importlib.import_module("survival.r._coerce")
bridge = importlib.import_module("survival.pybridge")


@pytest.mark.parametrize("storage", ["list", "numpy", "pandas"])
@pytest.mark.parametrize("action", ["na.omit", "na.exclude", "na.pass"])
def test_missing_selections_preserve_response_coding_and_factor_metadata(storage, action):
    data = {"time": [1, 2, 3, 4], "status": [1, 2, 1, 2], "x": [5, 6, 7, 8]}
    if storage == "numpy":
        data = {name: np.asarray(values) for name, values in data.items()}
    elif storage == "pandas":
        data = pytest.importorskip("pandas").DataFrame(data)
    data["g"] = coerce._r_factor(["a", "b", "a", "c"], ["c", "b", "a", "unused"])
    before = copy.deepcopy(data)
    frame = formula.model_frame(
        "Surv(time, status) ~ x + strata(g)",
        data,
        subset=bridge._r_subset([2, -1, 0, 2]),
        weights=[1, 2, 3, 4],
        na_action=action,
    )
    if action == "na.pass":
        assert frame.response.event == (0, None, 0, 0)
        assert np.isnan(frame.response.time[1])
        assert frame.weights == [3, None, 1, 3]
        groups = formula._strata_keep(frame.data, formula._strata_specs(frame.terms))
        assert groups.codes == [0, None, 0, 0]
        assert groups.labels == ["a", None, "a", "a"]
    else:
        assert frame.response.event == (0, 0, 0)
        assert frame.response.time == (3.0, 1.0, 3.0)
        assert frame.weights == [3, 1, 3]
        assert frame.na_action.rows == (2,)
    restored = pickle.loads(pickle.dumps(frame.data))  # noqa: S301 - round-trip our own frame
    assert formula._parse_formula(frame.formula, restored)[0].event == frame.response.event
    for name in data:
        np.testing.assert_array_equal(data[name], before[name])


@pytest.mark.parametrize(
    ("expression", "data", "expected_type", "expected_events"),
    [
        ("Surv(time, status)", {"time": [1, 2, 3], "status": [1, 2, 1]}, "right", (0, None, 0)),
        (
            "Surv(start, time, status)",
            {"start": [0, 0, 1], "time": [1, 2, 3], "status": [1, 2, 1]},
            "counting",
            (0, None, 0),
        ),
        (
            "Surv(time, end, type='interval2')",
            {"time": [1, 2, 3], "end": [np.inf, 2, 4]},
            "interval",
            (3, None, 0),
        ),
        (
            "Surv(time, status)",
            {"time": [1, 2, 3], "status": coerce._r_factor(["c", "event", "c"], ["c", "event"])},
            "mright",
            (0, None, 0),
        ),
    ],
)
def test_missing_selection_keeps_survival_response_type(
    expression, data, expected_type, expected_events
):
    frame = formula.model_frame(
        expression + " ~ 1", data, subset=bridge._r_subset([2, -1, 0]), na_action="na.pass"
    )
    assert frame.response.type == expected_type
    assert frame.response.event == expected_events
    assert np.isnan(frame.response.time[1])
    if frame.response.start is not None:
        assert np.isnan(frame.response.start[1])
    if frame.response.time2 is not None:
        assert np.isnan(frame.response.time2[1])
    if expected_type == "mright":
        assert frame.response.states == ("event",)
        assert frame.response.clabel == "c"


def test_missing_selection_of_matrix_covariate_keeps_its_shape():
    data = {"time": np.arange(1, 5), "status": [1, 1, 0, 1], "x": np.arange(8).reshape(4, 2)}
    frame = formula.model_frame(
        "Surv(time, status) ~ x", data, subset=bridge._r_subset([3, -1, 0]), na_action="na.pass"
    )
    np.testing.assert_array_equal(frame.data["x"], [[6, 7], [np.nan, np.nan], [0, 1]])
    with pytest.raises(ValueError, match="missing"):
        formula.model_frame(
            "Surv(time, status) ~ x", data, subset=bridge._r_subset([3, -1, 0]), na_action="na.fail"
        )


@pytest.mark.parametrize("subset", [[-1], [0, -2], [0.0, 1.0], [True]])
def test_python_subset_contract_remains_zero_based_and_strict(subset):
    with pytest.raises((TypeError, ValueError), match="subset"):
        r.survfit("Surv(time, status) ~ 1", {"time": [1, 2], "status": [1, 0]}, subset=subset)


def test_python_subset_keeps_censoring_when_it_selects_no_events():
    fit = r.survfit(
        "Surv(time, status) ~ 1", {"time": [1, 2, 3, 4], "status": [1, 2, 1, 2]}, subset=[0, 2]
    )
    assert list(fit.n_event) == [0, 0]
    assert list(fit.surv) == [1, 1]


def test_dataframe_column_named_response_cache_is_ordinary_data():
    pd = pytest.importorskip("pandas")
    frame = formula.model_frame(
        "Surv(time, status) ~ response_cache",
        pd.DataFrame({"time": [1, 2], "status": [1, 0], "response_cache": [3, 4]}),
    )
    assert frame.response.event == (1, 0)
