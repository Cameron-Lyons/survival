"""The Python wrappers of the Cox diagnostics pass the engine's fresh lists through.

``coxph_detail``, ``cox_zph``, ``residuals`` and the model's ``x``/``var`` read
each engine getter once instead of copying its nested lists again, and
``coxph_detail`` sorts with numpy.  These tests pin the results to reference
copies of the previous list-based code, element types and float bits included.
"""

import struct

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets

from survival.r._coxph import CoxphModel  # noqa: E402
from survival.r._coxphms import CoxphmsModel, _stacked_strata_labels  # noqa: E402


def same(a, b, path="value"):
    """Equal element types and float bits all the way down."""

    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def previous_detail(fit, riskmat=False, rorder="data"):
    """``coxph_detail`` as the list-based wrapper computed it."""

    def response(start, time, status):
        if start is not None:
            return [[s, t, float(e)] for s, t, e in zip(start, time, status, strict=True)]
        mintime = min(time) if time else 0.0
        begin = 2 * mintime - 1 if mintime < 0 else -1.0
        return [[begin, t, float(e)] for t, e in zip(time, status, strict=True)]

    multistate = isinstance(fit, CoxphmsModel)
    detail = survival._survival.coxph_detail(fit.fit, riskmat=riskmat)
    if multistate:
        engine = fit.fit
        y = response(engine.entry, engine.time, engine.status)
        x = [list(row) for row in engine.x]
    else:
        y = response(fit.y.start, fit.y.time, fit.y.event)
        x = [list(row) for row in fit.fit.x]
    n = len(y)
    strata_codes = fit.fit.strata or [0] * n
    order = sorted(range(n), key=lambda idx: (strata_codes[idx], y[idx][1], -y[idx][2]))
    weights = list(fit.fit.weights)
    weighted = any(value != 1.0 for value in weights)
    strata_table = None
    if detail.strata is not None and (multistate or fit.strata_levels):
        labels = (
            _stacked_strata_labels(detail.strata)
            if multistate
            else [fit.strata_levels[int(code)] for code in detail.strata]
        )
        strata_table = {}
        for label in labels:
            strata_table[label] = strata_table.get(label, 0) + 1
    risk_rows = None if detail.riskmat is None else [list(row) for row in detail.riskmat]
    if rorder == "time":
        x = [x[idx] for idx in order]
        y = [y[idx] for idx in order]
        if risk_rows is not None:
            risk_rows = [risk_rows[idx] for idx in order]
    return {
        "time": list(detail.time),
        "nevent": [int(value) for value in detail.nevent],
        "nrisk": [int(value) for value in detail.nrisk],
        "hazard": list(detail.hazard),
        "varhaz": list(detail.varhaz),
        "wtrisk": list(detail.wtrisk),
        "means": [list(row) for row in detail.means],
        "score": [list(row) for row in detail.score],
        "imat": [[list(row) for row in layer] for layer in detail.imat],
        "x": x,
        "y": y,
        "strata": strata_table,
        "riskmat": risk_rows,
        "sortorder": order if rorder == "time" and riskmat else None,
        "weights": [weights[idx] for idx in order] if weighted else None,
        "nevent_wt": list(detail.nevent_wt) if weighted else None,
        "nrisk_wt": list(detail.wtrisk) if weighted else None,
    }


def tied_data(n=120, seed=4, shift=0.0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    t = rng.exponential(np.exp(-0.4 * x1), n)
    c = rng.exponential(1.5, n)
    time = np.round(np.minimum(t, c), 1) + 0.1 + shift
    return pd.DataFrame(
        {
            "time": time,
            "status": (t <= c).astype(int),
            "start": time - np.round(rng.uniform(0.05, 0.8, n), 1),
            "x1": x1,
            "x2": x2,
            "s": rng.integers(0, 3, n),
            "w": rng.choice([0.5, 1.0, 2.0], n),
            "cl": rng.integers(0, 30, n),
        }
    )


def _mgus_multistate():
    mg = pd.DataFrame(datasets.load_mgus2())
    mg["etime"] = np.where(mg.pstat == 1, mg.ptime, mg.futime)
    event = np.where(mg.pstat == 1, "pcm", np.where(mg.death == 1, "death", "censor"))
    mg["event"] = pd.Categorical(event, categories=["censor", "pcm", "death"])
    return r.coxph("Surv(etime, event) ~ age + mspike + strata(sex)", mg, id="id")


@pytest.fixture(scope="module")
def fits():
    d = tied_data()
    negative = tied_data(seed=5, shift=-1.0)
    return {
        "right": r.coxph("Surv(time, status) ~ x1 + x2", data=d),
        "breslow_strata": r.coxph("Surv(time, status) ~ x1 + strata(s)", data=d, ties="breslow"),
        "weighted": r.coxph("Surv(time, status) ~ x1 + x2 + strata(s)", data=d, weights="w"),
        "counting": r.coxph("Surv(start, time, status) ~ x1 + x2", data=d, weights="w"),
        "negative": r.coxph("Surv(time, status) ~ x1", data=negative),
        "cluster": r.coxph("Surv(time, status) ~ x1 + x2 + cluster(cl)", data=d),
        "multistate": _mgus_multistate(),
    }


@pytest.mark.parametrize(
    "name", ["right", "breslow_strata", "weighted", "counting", "negative", "multistate"]
)
@pytest.mark.parametrize("riskmat", [False, True])
@pytest.mark.parametrize("rorder", ["data", "time"])
def test_coxph_detail_matches_the_list_based_wrapper(fits, name, riskmat, rorder):
    fit = fits[name]
    detail = r.coxph_detail(fit, riskmat=riskmat, rorder=rorder)
    expected = previous_detail(fit, riskmat=riskmat, rorder=rorder)
    for field, value in expected.items():
        same(getattr(detail, field), value, field)
    if detail.wtrisk is not None and detail.nrisk_wt is not None:
        assert detail.wtrisk is not detail.nrisk_wt


@pytest.mark.parametrize("name", ["right", "weighted", "counting", "cluster", "multistate"])
def test_cox_zph_passes_fresh_nested_lists(fits, name, monkeypatch):
    engine_zph = survival._survival.cox_zph
    native = []

    def recording(*args, **kwargs):
        native.append(engine_zph(*args, **kwargs))
        return native[-1]

    monkeypatch.setattr(survival._survival, "cox_zph", recording)
    fit = fits[name]
    for transform in ["km", "rank", "identity"]:
        zph = r.cox_zph(fit, transform=transform)
        engine = native[-1]
        same(zph.y, [list(row) for row in engine.y], "y")
        same(zph.var, [list(row) for row in engine.var], "var")
        same(zph.x, list(engine.x), "x")
        same(zph.time, list(engine.time), "time")
        assert zph.y is not r.cox_zph(fit, transform=transform).y


def test_model_matrices_are_independent_copies(fits):
    fit = fits["cluster"]
    assert isinstance(fit, CoxphModel)
    x = fit.x
    same(x, [list(row) for row in fit.fit.x], "x")
    x[0][0] = 1e300
    x[1].insert(0, 0.0)
    assert fit.x[0][0] != 1e300
    assert len(fit.x[1]) == len(fit.coefficient_names)
    same(fit.var, [list(row) for row in fit.fit.var], "var")
    same(fit.naive_var, [list(row) for row in fit.fit.naive_var], "naive_var")
    assert fits["right"].naive_var is None


@pytest.mark.parametrize("name", ["right", "counting", "cluster"])
def test_martingale_and_deviance_residuals_are_lists(fits, name):
    fit = fits[name]
    for kind, method in [
        ("martingale", fit.fit.martingale_residuals),
        ("deviance", fit.fit.deviance_residuals),
    ]:
        values = r.residuals(fit, type=kind)
        same(values, list(method(weighted=False, collapse=None)), kind)


def test_collapse_true_uses_the_cluster_codes(fits):
    fit = fits["cluster"]
    collapsed = r.residuals(fit, type="dfbeta", collapse=True, _with_group_names=True)
    by_vector = r.residuals(fit, type="dfbeta", collapse=list(fit.cluster), _with_group_names=True)
    same(collapsed, by_vector, "dfbeta")
    assert collapsed["group_names"] == [str(value) for value in sorted(set(fit.cluster))]
