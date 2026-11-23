"""Lazily listed curves of ``survfit.coxph`` and ``CoxSurvfitResult.to_arrays``.

``survfit`` stores a Cox fit's curve matrices as arrays and builds the nested lists
of ``surv``, ``cumhaz``, ``std_err``, ``std_chaz``, ``lower`` and ``upper`` on their
first read. The fields keep reading as the lists they always were, and the package's
own summaries, prints, frames, subsets and plots read the arrays, so they never pay
for the lists.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import math
import pickle

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
r_types = importlib.import_module("survival.r._types")
plot_data = importlib.import_module("survival._plot_data")

CURVES = ("surv", "cumhaz", "std_err", "std_chaz", "lower", "upper")
COUNTS = ("time", "n_risk", "n_event", "n_censor")


def _lung():
    lung = survival.datasets.load_lung()
    keep = [
        i
        for i in range(len(lung["time"]))
        if not any(
            value is None or (isinstance(value, float) and math.isnan(value))
            for value in (lung["ph.ecog"][i], lung["age"][i])
        )
    ]
    return {
        name: [lung[name][i] for i in keep] for name in ("time", "status", "age", "sex", "ph.ecog")
    }


@pytest.fixture(scope="module")
def lung():
    return _lung()


@pytest.fixture(scope="module")
def cox(lung):
    return r.coxph("Surv(time, status) ~ age + sex", lung)


@pytest.fixture(scope="module")
def stratified(lung):
    return r.coxph("Surv(time, status) ~ age + strata(sex)", lung)


NEWDATA = {"age": [50.0, 60.0, 70.0], "sex": [1, 2, 1]}


def _results(cox, stratified):
    return {
        "one curve": r.survfit(cox),
        "newdata": r.survfit(cox, newdata=NEWDATA),
        "one newdata row": r.survfit(cox, newdata={"age": [62.0], "sex": [2]}),
        "strata": r.survfit(stratified),
        "strata newdata": r.survfit(stratified, newdata={"age": [55.0, 65.0]}),
        "no se": r.survfit(cox, newdata=NEWDATA, se_fit=False),
        "no confint": r.survfit(cox, newdata=NEWDATA, conf_type="none"),
        "hazard se": r.survfit(cox, newdata=NEWDATA, stype=2, ctype=1),
    }


def _lazy(result, name):
    return isinstance(result.__dict__.get(name), r_types._CurveArray)


def _listed(result, name):
    stored = result.__dict__.get(name)
    return not isinstance(stored, r_types._CurveArray) or stored._lists is not None


def test_curves_are_stored_as_arrays_and_read_as_their_lists(cox, stratified):
    for label, result in _results(cox, stratified).items():
        stored = {name: result.__dict__.get(name) for name in CURVES}
        arrays = result.to_arrays()
        # ``to_arrays`` builds nothing; check every field before any is read.
        unlisted = {
            name: isinstance(value, r_types._CurveArray) and value._lists is None
            for name, value in stored.items()
        }
        for name in CURVES:
            value = stored[name]
            if value is None:
                assert getattr(result, name) is None, (label, name)
                assert name not in arrays, (label, name)
                continue
            assert isinstance(value, r_types._CurveArray), (label, name)
            assert unlisted[name], (label, name)
            listed = getattr(result, name)
            assert isinstance(listed, list), (label, name)
            assert listed == value.values.tolist(), (label, name)
            assert getattr(result, name) is listed, (label, name)
            np.testing.assert_array_equal(
                arrays[name], np.asarray(listed), err_msg=f"{label} {name}"
            )
        for name in COUNTS:
            np.testing.assert_array_equal(
                arrays[name], np.asarray(getattr(result, name), dtype=float)
            )


def test_one_curve_reads_as_a_vector_and_several_as_rows(cox):
    single = r.survfit(cox)
    assert single.__dict__["surv"].values.ndim == 1
    assert not single.has_data_margin
    assert single.ncurve == 1
    assert all(isinstance(value, float) for value in single.surv)
    one_row = r.survfit(cox, newdata={"age": [62.0], "sex": [2]})
    assert one_row.__dict__["surv"].values.ndim == 1
    assert one_row.colnames is None
    several = r.survfit(cox, newdata=NEWDATA)
    assert several.__dict__["surv"].values.shape == (len(several.time), 3)
    assert several.has_data_margin
    assert several.ncurve == 3
    assert several.dim == {"data": 3}
    assert all(isinstance(row, list) and len(row) == 3 for row in several.surv)
    assert several.colnames == ["1", "2", "3"]


def test_cumulative_hazard_errors_share_one_list(cox):
    result = r.survfit(cox, newdata=NEWDATA, stype=2, ctype=1)
    assert result.__dict__["std_chaz"] is result.__dict__["std_err"]
    assert result.std_chaz is result.std_err


def test_to_arrays_returns_independent_writable_snapshots(cox):
    result = r.survfit(cox, newdata=NEWDATA)
    first = result.to_arrays()
    for values in first.values():
        assert values.flags.writeable
        assert values.dtype == np.float64
        values[...] = -1.0
    second = result.to_arrays()
    assert not any(np.any(values == -1.0) for values in second.values())
    assert all(value >= 0 for row in result.surv for value in row)
    assert not np.shares_memory(second["surv"], result.__dict__["surv"].values)


def test_package_readers_never_build_the_lists(cox, stratified):
    calls = {
        "print_survfit": lambda x: r.print_survfit(x),
        "summary_survfit": lambda x: r.summary_survfit(x),
        "summary times": lambda x: r.summary_survfit(x, times=[100.0, 300.0]),
        "summary frame": lambda x: r.summary_survfit(x, data_frame=True),
        "quantile_survfit": lambda x: r.quantile_survfit(x, probs=[0.25, 0.5]),
        "median_survfit": lambda x: r.median_survfit(x),
        "survfit0": lambda x: r.survfit0(x),
        "as_data_frame": lambda x: r.as_data_frame(x),
        "plot data": lambda x: plot_data.survfit_plot_data(x),
    }
    for label, result in _results(cox, stratified).items():
        for name, call in calls.items():
            call(result)
            built = [field for field in CURVES if _lazy(result, field) and _listed(result, field)]
            assert not built, (label, name, built)


def test_subsets_and_aggregates_stay_arrays_and_match_the_lists(cox, stratified):
    curves = r.survfit(cox, newdata=NEWDATA)
    subset = curves.subset(data=[0, 2])
    assert all(_lazy(subset, name) for name in ("surv", "cumhaz", "std_err", "lower", "upper"))
    assert not any(_listed(curves, name) for name in CURVES if _lazy(curves, name))
    assert subset.surv == [[row[0], row[2]] for row in curves.surv]
    assert subset.colnames == ["1", "3"]
    single = curves.subset(data=1)
    assert single.surv == [row[1] for row in curves.surv]
    blocks = r.survfit(stratified)
    stratum = blocks.subset(strata=0)
    assert stratum.surv == blocks.surv[: next(iter(blocks.strata.values()))]

    fresh = r.survfit(cox, newdata=NEWDATA)
    aggregated = r.aggregate_survfit(fresh, by=[1, 1, 2])
    assert aggregated.cumhaz == []
    assert not any(_listed(fresh, name) for name in CURVES if _lazy(fresh, name))
    expected = r.aggregate_survfit(
        dataclasses.replace(fresh, **{name: getattr(fresh, name) for name in CURVES}), by=[1, 1, 2]
    )
    assert aggregated == expected


def test_lazy_results_compare_pickle_copy_and_replace_like_listed_ones(cox):
    lazy = r.survfit(cox, newdata=NEWDATA)
    listed = dataclasses.replace(r.survfit(cox, newdata=NEWDATA), n=list(lazy.n))
    for name in CURVES:
        getattr(listed, name)
    assert lazy == listed

    # An unread result pickles its arrays and loads unread; one whose lists were
    # read (``==`` above reads them) pickles the lists.
    unread = r.survfit(cox, newdata=NEWDATA)
    restored_unread = pickle.loads(pickle.dumps(unread))  # noqa: S301
    assert _lazy(restored_unread, "surv")
    assert restored_unread.__dict__["surv"]._lists is None
    assert restored_unread == lazy
    restored = pickle.loads(pickle.dumps(lazy))  # noqa: S301
    assert isinstance(restored.__dict__["surv"], list)
    assert restored == lazy
    assert restored.dim == lazy.dim
    assert restored.colnames == lazy.colnames

    listed_state = pickle.loads(pickle.dumps(listed))  # noqa: S301
    assert listed_state == lazy

    assert copy.copy(lazy) == lazy
    assert copy.deepcopy(lazy) == lazy
    replaced = dataclasses.replace(lazy, conf_int=lazy.conf_int)
    assert replaced == lazy
    with pytest.raises(dataclasses.FrozenInstanceError):
        lazy.surv = []  # type: ignore[misc]


def test_constructor_takes_lists_and_arrays_as_given():
    fields = {
        "n": [2],
        "time": [1.0, 2.0],
        "n_risk": [2.0, 1.0],
        "n_event": [1.0, 1.0],
        "n_censor": [0.0, 0.0],
        "type": "right",
    }
    listed = r_types.CoxSurvfitResult(surv=[0.5, 0.25], cumhaz=[0.7, 1.4], **fields)
    assert listed.surv == [0.5, 0.25]
    assert listed.std_err is None
    assert listed.to_arrays()["surv"].tolist() == [0.5, 0.25]
    given = np.empty((0, 4))
    raw = r_types.CoxSurvfitResult(surv=given, cumhaz=[], **fields)
    assert raw.surv is given
    assert r_types.CoxSurvfitResult.std_err is None
    with pytest.raises(AttributeError):
        r_types.CoxSurvfitResult.surv  # noqa: B018
    with pytest.raises(TypeError):
        r_types.CoxSurvfitResult(**fields)  # type: ignore[call-arg]


ONE_CENSORED = {
    "time": [1.0, 2.0, 3.0, 4.0, 5.0],
    "status": [1, 1, 0, 0, 1],
    "x": [0.0, 1.0, 0.5, 0.2, 0.7],
    "g": [1, 1, 1, 2, 1],
}


def test_every_pickle_protocol_round_trips(cox):
    reference = r.survfit(r.coxph("Surv(time, status) ~ x + strata(g)", ONE_CENSORED))
    assert reference.subset(strata=1).cumhaz == [0.0]
    lone = r.survfit(r.coxph("Surv(time, status) ~ x + strata(g)", ONE_CENSORED)).subset(strata=1)
    fits = {
        "one curve": r.survfit(cox),
        "newdata": r.survfit(cox, newdata=NEWDATA),
        "one zero time": lone,
    }
    for label, fit in fits.items():
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(fit, protocol=protocol))  # noqa: S301
            assert restored == fit, (label, protocol)
            assert restored.dim == fit.dim, (label, protocol)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(lone, protocol=protocol))  # noqa: S301
        assert restored.cumhaz == [0.0]
        assert restored.std_err == [0.0]


def test_edits_to_the_lists_reach_every_reader(cox):
    single = r.survfit(cox)
    single.surv[0] = 0.123
    assert r.summary_survfit(single).surv[0] == 0.123
    assert r.as_data_frame(single)["surv"][0] == 0.123
    assert r.survfit0(single).surv[1] == 0.123
    assert single.to_arrays()["surv"][0] == 0.123
    assert pickle.loads(pickle.dumps(single)).surv[0] == 0.123  # noqa: S301
    copied = copy.deepcopy(single)
    assert copied.surv[0] == 0.123
    assert copied == single

    curves = r.survfit(cox, newdata=NEWDATA)
    curves.surv[0][0] = 0.5
    curves.cumhaz[0][0] = 9.0
    assert curves.subset(data=0).surv[0] == 0.5
    assert r.as_data_frame(curves)["surv"][0] == 0.5
    assert r.summary_survfit(curves).surv[0][0] == 0.5
    estimate = plot_data.survfit_plot_data(curves, fun="cumhaz").curves[0].estimate
    assert 9.0 in estimate
    aggregated = r.aggregate_survfit(curves, by=[1, 1, 2])
    assert aggregated.surv[0][0] == pytest.approx((0.5 + curves.surv[0][1]) / 2)


def test_a_deep_copy_compares_equal_when_a_curve_holds_nan():
    data = {"time": [1, 1, 1, 2, 2], "status": [1, 1, 1, 1, 1], "x": [0.1, 0.5, 0.2, 0.9, 0.3]}
    fit = r.survfit(r.coxph("Surv(time, status) ~ x", data), stype=1)
    assert any(math.isnan(value) for value in fit.lower)
    assert copy.deepcopy(fit) == fit
    assert dataclasses.asdict(copy.deepcopy(fit)) == dataclasses.asdict(fit)
    unread = r.survfit(r.coxph("Surv(time, status) ~ x", data), stype=1)
    assert copy.deepcopy(unread) == unread


def test_concurrent_first_reads_share_one_list(cox):
    import threading

    fit = r.survfit(cox, newdata={"age": [50.0 + k / 10 for k in range(400)], "sex": [1] * 400})
    barrier = threading.Barrier(4)
    seen = []

    def read():
        barrier.wait()
        seen.append(fit.surv)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(item is seen[0] for item in seen)
    assert seen[0] is fit.surv


def test_dataclass_subclasses_read_their_own_fields(cox):
    from typing import Any

    fit = r.survfit(cox, newdata=NEWDATA)
    values = {item.name: getattr(fit, item.name) for item in dataclasses.fields(fit)}

    @dataclasses.dataclass(frozen=True)
    class Narrowed(r_types.CoxSurvfitResult):
        lower: Any = None
        upper: Any = None

    narrowed = Narrowed(**values)
    subset = narrowed.subset(data=[0, 1])
    assert subset.lower == [row[:2] for row in fit.lower]
    assert r.summary_survfit(narrowed).surv == r.summary_survfit(fit).surv

    @dataclasses.dataclass(frozen=True, slots=True)
    class Slotted(r_types.CoxSurvfitResult):
        tag: str = "x"

    slotted = Slotted(**values)
    assert r.summary_survfit(slotted).surv == r.summary_survfit(fit).surv
    assert slotted.subset(data=[0, 2]).surv == [[row[0], row[2]] for row in fit.surv]


def test_subsets_of_given_lists_keep_their_element_types():
    curves = r_types.CoxSurvfitResult(
        n=[2, 1],
        time=[1.0, 2.0, 3.0],
        n_risk=[2.0, 1.0, 1.0],
        n_event=[1.0, 1.0, 0.0],
        n_censor=[0.0, 0.0, 1.0],
        type="right",
        strata={"a": 2, "b": 1},
        surv=[[1, 1], [1, 0], [1, 1]],
        cumhaz=[[0, 0], [0, 1], [0, 0]],
        std_err=[[0.1, None], [0.2, 0.3], [None, 0.0]],
        colnames=["p", "q"],
        newdata={"z": [1, 2]},
    )
    assert curves.subset(data=1).surv == [1, 0, 1]
    assert curves.subset(data=1).std_err == [None, 0.3, 0.0]
    assert curves.subset(strata=0).std_err == [[0.1, None], [0.2, 0.3]]


def test_asdict_reads_the_lists(cox):
    fit = r.survfit(cox, newdata=NEWDATA)
    as_dict = dataclasses.asdict(fit)
    assert as_dict["surv"] == fit.surv
    assert isinstance(as_dict["surv"], list)


def test_the_r_bridge_reads_the_arrays_without_lists(cox):
    fit = r.survfit(cox, newdata=NEWDATA)
    arrays = survival.pybridge._cox_curve_arrays(fit)
    assert set(arrays) == set(fit.to_arrays())
    assert not any(_listed(fit, name) for name in CURVES if _lazy(fit, name))
    assert arrays["surv"].shape == (len(fit.time), 3)
    np.testing.assert_array_equal(arrays["surv"], np.asarray(fit.surv))
