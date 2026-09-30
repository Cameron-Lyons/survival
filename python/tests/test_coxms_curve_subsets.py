"""Selected multistate Cox curves retain aligned counts for every derived result."""

import json
import pickle
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

r = setup_survival_import().r
from survival.r._models import _subset_coxms_curves  # noqa: E402

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "coxms_subset_reference.json").read_text()
)
CASES = REFERENCE["cases"]
MODES = {
    "events": {},
    "all": {"censored": True},
    "times": {"times": [3, 4, 6]},
    "extended": {"times": [0, 4, 8], "extend": True},
    "scaled": {"times": [3, 4, 6], "scale": 2, "rmean": 5},
    "none": {"rmean": "none"},
}


@cache
def _source(stratified, conditional=False, time0=False):
    data = pd.DataFrame(REFERENCE["data"])
    data["event"] = pd.Categorical(data.event, categories=["censor", "ill", "dead"])
    formula = "Surv(start, stop, event) ~ x" + (" + strata(g)" if stratified else "")
    fit = r.coxph(formula, data, id="id", weights="weight")
    options = {"start_time": 3, "p0": [0.6, 0.4, 0]} if conditional else {}
    return r.survfit(fit, newdata=REFERENCE["newdata"], time0=time0, **options)


def _selected(case):
    source = _source(case["stratified"], case["conditional"], case["time0"])
    return _subset_coxms_curves(
        source,
        strata=case["groups"] if case["stratified"] else None,
        data=[2, 0],
        states=case["states"],
    )


def _array(expected):
    return np.asarray(expected["values"]).reshape(expected["shape"], order="F")


def _check_arrays(result, expected, names):
    for name in names:
        np.testing.assert_allclose(
            getattr(result, name), _array(expected[name]), rtol=2e-9, atol=2e-12, err_msg=name
        )


def _check_curve(result, expected):
    for name in ("time", "n", "n_id", "states", "strata"):
        assert getattr(result, name) == expected[name], name
    _check_arrays(result, expected, ("n_risk", "n_event", "n_censor", "pstate", "p0"))
    assert result.cumhaz is None
    assert result.n_transition is None
    assert result.engine.states == result.states


@pytest.mark.parametrize("case", CASES)
def test_selected_curves_and_initial_rows_match_reference(case):
    selected = _selected(case)
    _check_curve(selected, case["curve"])
    initial = r.survfit0(selected)
    _check_curve(initial, case["initial"])
    assert r.survfit0(initial) is initial


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES)
def test_selected_summaries_match_reference(case, mode):
    options = dict(MODES[mode])
    if mode == "extended" and case["conditional"]:
        options["times"] = [3, 4, 8]
    result = r.summary_survfit(_selected(case), **options)
    expected = case["summaries"][mode]["expected"]
    for name in ("time", "states", "strata", "rmean_endtime"):
        assert getattr(result, name) == expected[name], name
    _check_arrays(result, expected, ("n_risk", "n_event", "n_censor", "pstate"))
    np.testing.assert_allclose(result.table.values, _array(expected["table"]), rtol=2e-9)
    assert result.table.rownames == expected["table_rows"]
    assert result.table.colnames == expected["table_columns"]
    assert result.cumhaz is None
    assert result.n_transition is None


@pytest.mark.parametrize("stratified", [False, True])
def test_public_subset_composes_and_preserves_original_state_names(stratified):
    source = _source(stratified)
    selected = source.subset(states=["dead", "(s0)"]).subset(states=[1], data=[2, 0])
    direct = source.subset(states=["(s0)"], data=[2, 0])
    assert selected.oldstate == tuple(source.states)
    assert selected.states == direct.states == ["(s0)"]
    assert selected.n_censor == direct.n_censor
    assert selected.p0 == direct.p0
    np.testing.assert_array_equal(selected.pstate, direct.pstate)
    # Selected probabilities retain their mass instead of being renormalized.
    assert np.any(selected.pstate < 1)
    before = r.survfit0(source).subset(states=[0], data=[2, 0])
    after = r.survfit0(selected)
    assert before.time == after.time
    assert before.n_censor == after.n_censor
    np.testing.assert_array_equal(before.pstate, after.pstate)
    restored = pickle.loads(pickle.dumps(selected))  # noqa: S301 - locally created result
    np.testing.assert_array_equal(
        r.summary_survfit(restored, censored=True).pstate,
        r.summary_survfit(selected, censored=True).pstate,
    )


@pytest.mark.parametrize("stratified", [False, True])
def test_selected_curves_support_frames_reports_and_aggregation(stratified):
    selected = _source(stratified).subset(states=["dead", "ill"])
    frame = r.as_data_frame(selected)
    nt, nd, ns = selected.pstate.shape
    assert len(frame["time"]) == nt * nd * ns
    np.testing.assert_array_equal(frame["pstate"], selected.pstate.ravel(order="F"))
    expected_censor = np.tile(np.asarray(selected.n_censor).T[:, None, :], (1, nd, 1))
    np.testing.assert_array_equal(frame["n.censor"], expected_censor.ravel())
    report = r.print_survfit(selected)
    assert "dead" in str(report)
    assert "ill" in str(report)
    aggregate = r.aggregate_survfit(selected)
    np.testing.assert_allclose(aggregate.pstate, selected.pstate.mean(axis=1, keepdims=True))
    summary = r.summary_survfit(aggregate, censored=True)
    np.testing.assert_array_equal(summary.pstate, aggregate.pstate)


@pytest.mark.parametrize("states", [None, [2], [2, 1, 2], [2, 1, 0]])
def test_subsets_own_mutable_arrays_and_count_rows(states):
    source = _source(True)
    selected = source.subset(strata=[1], states=states)
    original = pickle.dumps(source)
    selected.pstate[:] = -1
    if selected.cumhaz is not None:
        selected.cumhaz[:] = -1
    for name in ("n_risk", "n_event", "n_censor", "n_transition", "p0"):
        values = getattr(selected, name)
        if values:
            values[0][0] = -1
    assert pickle.dumps(source) == original


@pytest.mark.parametrize("selection", [[], [-1], [3], ["absent"], [0.5], [True]])
def test_public_state_subset_validates_indices(selection):
    with pytest.raises((ValueError, IndexError, TypeError)):
        _source(False).subset(states=selection)


def test_repeated_strata_keep_separate_blocks_and_selectable_labels():
    source = _source(True)
    selected = source.subset(strata=[1, 1], states=[2])
    assert list(selected.strata) == ["b", "b.1"]
    one = source.subset(strata=[1], states=[2])
    assert selected.time == one.time * 2
    assert selected.n_id == one.n_id * 2
    np.testing.assert_array_equal(selected.pstate, np.concatenate([one.pstate, one.pstate]))
    summary = r.summary_survfit(selected, times=[3, 6])
    assert summary.strata == ["b", "b", "b.1", "b.1"]
    np.testing.assert_array_equal(selected.subset(strata=["b.1"]).pstate, one.pstate)
