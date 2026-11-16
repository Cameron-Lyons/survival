"""survfit's faster paths against the code they replaced.

Each test takes the previous path, or keeps a copy of the previous implementation, and
checks the new one against it.  The clustered Kaplan-Meier variance from running sums
agrees with the dense loop over every cluster to rounding; the rest give exactly the same
numbers: one native call building every Cox curve column's engine, integer id and cluster
labels numbered by the engine instead of in Python, the levels of ``factor()`` of a
numeric column built as a set in C, and the case-weight checks done by NumPy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
core = survival._survival
_survfit = r._survfit


def _identical(a, b):
    """Equal values, NaN where the other is NaN, and the same sign of every zero."""

    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return type(a) is type(b) and a == b


def _clustered_km_data(last_status, n=3000, seed=6):
    """Hundreds of clusters of mixed sizes, ties, zero weights and a last time at which
    everyone still at risk has ``last_status``, so that the curve (with ``reverse``, the
    censoring curve for a status of 0) ends at zero.  Times on a grid of 0.01 give each
    curve enough distinct times for the running sums to replace the dense loop."""

    rng = np.random.default_rng(seed)
    time = np.round(rng.exponential(1.0, n), 2) + 0.01
    status = rng.integers(0, 2, n)
    last = rng.random(n) < 0.02
    time[last], status[last] = time.max() + 1.0, last_status
    cluster = np.where(rng.random(n) < 0.1, 0, rng.integers(1, n // 4, n))
    weights = np.where(rng.random(n) < 0.05, 0.0, rng.uniform(0.5, 2.0, n))
    return pd.DataFrame(
        {
            "time": time,
            "status": status,
            "cl": cluster,
            "w": weights,
            "g": np.array(["a", "b"])[rng.integers(0, 2, n)],
        }
    )


@pytest.mark.parametrize(
    "options",
    [{"weights": "w"}, {}, {"weights": "w", "reverse": True}, {"weights": "w", "stype": 2}],
)
def test_clustered_robust_variance_matches_the_dense_cluster_loop(options):
    """Many clusters take the running-sum variance; ``influence=True`` keeps the dense loop
    that updates every cluster at every time.  The two agree to rounding, and exactly where
    the curve reaches zero, so the confidence limits there are the same zero or NaN."""

    data = _clustered_km_data(last_status=0 if options.get("reverse") else 1)
    fast = r.survfit("Surv(time, status) ~ g", data=data, cluster="cl", **options)
    dense = r.survfit("Surv(time, status) ~ g", data=data, cluster="cl", influence=True, **options)
    assert _identical(fast.surv, dense.surv)
    # The running sums round differently from the dense loop: equal standard errors
    # everywhere would mean that the fit never took them.
    assert (np.array(fast.std_err) != np.array(dense.std_err)).any()
    for field in ("std_err", "std_chaz", "lower", "upper"):
        actual, expected = np.array(getattr(fast, field)), np.array(getattr(dense, field))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14, err_msg=field)
    zero = np.array(fast.surv) == 0.0
    assert zero.any() == ("stype" not in options)
    assert _identical(np.array(fast.std_err)[zero].tolist(), np.array(dense.std_err)[zero].tolist())


ENGINE_FIELDS = (
    "n",
    "time",
    "n_risk",
    "n_event",
    "n_censor",
    "n_enter",
    "surv",
    "std_err",
    "cumhaz",
    "std_chaz",
    "lower",
    "upper",
    "strata",
    "strata_codes",
    "n_id",
    "logse",
    "conf_int",
    "conf_type",
    "conf_lower",
    "type",
    "t0",
)


def _reference_cox_engines(x):
    """``_cox_engines`` as it was: a Python transpose and one ``from_stacked`` per column."""

    def columns(values):
        if values and isinstance(values[0], list):
            return [list(column) for column in zip(*values, strict=True)]
        return [list(values)]

    named = {
        name: columns(values) if (values := getattr(x, name)) else [None] * x.ncurve
        for name in ("cumhaz", "std_err", "std_chaz", "lower", "upper")
    }
    return [
        core.SurvfitKMResult.from_stacked(
            x.time,
            x.n_risk,
            x.n_event,
            surv,
            x.n,
            strata=list(x.strata.values()) if x.strata else None,
            n_censor=x.n_censor,
            std_err=named["std_err"][curve],
            cumhaz=named["cumhaz"][curve],
            std_chaz=named["std_chaz"][curve],
            lower=named["lower"][curve],
            upper=named["upper"][curve],
            logse=x.logse is not False,
            conf_int=0.95 if x.conf_int is None else x.conf_int,
            conf_type=x.conf_type or "none",
            type=x.type,
            t0=min([0.0, *x.time]),
        )
        for curve, surv in enumerate(columns(x.surv))
    ]


def _cox_data(n=240, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    t = rng.exponential(np.exp(-(0.4 * x1 - 0.3 * x2)))
    c = rng.exponential(1.5, n)
    time = np.round(np.minimum(t, c), 1) + 0.1
    return pd.DataFrame(
        {
            "time": time,
            "status": (t <= c).astype(int),
            "start": np.round(time * rng.uniform(0, 0.5, n), 2),
            "x1": x1,
            "x2": x2,
            "g": np.array(["a", "b", "c"])[rng.integers(0, 3, n)],
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


def _cox_curves():
    data = _cox_data()
    newdata = data.iloc[:7].copy()
    extreme = newdata.assign(x1=[60.0, -60.0, 0.0, 30.0, 1.0, 2.0, -30.0])
    plain = r.coxph("Surv(time, status) ~ x1 + x2", data=data)
    yield "plain", r.survfit(plain, newdata=newdata)
    yield "single row", r.survfit(plain, newdata=newdata.iloc[:1])
    yield "no newdata", r.survfit(plain)
    yield "no se", r.survfit(plain, newdata=newdata, se_fit=False)
    yield "stype 1", r.survfit(plain, newdata=newdata, stype=1)
    yield "log-log", r.survfit(plain, newdata=newdata, conf_type="log-log")
    yield "plain limits", r.survfit(plain, newdata=newdata, conf_type="plain")
    yield "extreme", r.survfit(plain, newdata=extreme)
    strata = r.coxph("Surv(time, status) ~ x1 + strata(g)", data=data)
    yield "strata", r.survfit(strata, newdata=newdata)
    weighted = r.coxph("Surv(time, status) ~ x1 + x2", data=data, weights="w", ties="breslow")
    yield "weights", r.survfit(weighted, newdata=newdata)
    counting = r.coxph("Surv(start, time, status) ~ x1 + x2", data=data)
    yield "counting", r.survfit(counting, newdata=newdata)


@pytest.mark.parametrize(("name", "curves"), list(_cox_curves()))
def test_cox_engines_match_one_from_stacked_per_column(name, curves):
    engines = _survfit._cox_engines(curves)
    reference = _reference_cox_engines(curves)
    assert len(engines) == len(reference) == curves.ncurve
    for engine, expected in zip(engines, reference, strict=True):
        for field in ENGINE_FIELDS:
            assert _identical(getattr(engine, field), getattr(expected, field)), (name, field)


def test_cox_engines_keep_one_curve_for_no_rows_and_reject_ragged_columns():
    engines = core.SurvfitKMResult._from_stacked_columns([], [], [], [], [0])
    assert len(engines) == 1
    assert engines[0].surv == []
    surv = [[0.9, 0.8], [0.7, 0.6]]
    engines = core.SurvfitKMResult._from_stacked_columns(
        [1.0, 2.0], [2.0, 1.0], [1.0, 1.0], surv, [2], std_err=[[0.1, 0.2], [0.3, 0.4]]
    )
    assert [e.surv for e in engines] == [[0.9, 0.7], [0.8, 0.6]]
    assert [e.std_err for e in engines] == [[0.1, 0.3], [0.2, 0.4]]
    with pytest.raises(ValueError, match="same shape"):
        core.SurvfitKMResult._from_stacked_columns(
            [1.0, 2.0], [2.0, 1.0], [1.0, 1.0], surv, [2], lower=[0.1, 0.2]
        )


def test_engine_labels_pass_plain_integers_and_encode_everything_else():
    assert _survfit._engine_labels([7, -3, 7, 2**63 - 1], "id") == [7, -3, 7, 2**63 - 1]
    assert _survfit._engine_labels([2**63, 1, 2**63], "id") == [0, 1, 0]
    assert _survfit._engine_labels([True, 1, 2], "id") == [0, 0, 1]
    assert _survfit._engine_labels([1.0, 2.5, 1], "id") == [0, 1, 0]
    assert _survfit._engine_labels(["b", "a", "b"], "id") == [0, 1, 0]


def _ms_data(m=60, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(m):
        ill = float(np.round(rng.exponential(1.0), 2) + 0.01)
        end = float(np.round(rng.exponential(1.5), 2) + 0.01)
        label = int(rng.integers(0, 3)) * 1000 + subject
        if ill < end:
            rows.append((label, 0.0, ill, 1))
            rows.append((label, ill, ill + end, int(rng.integers(0, 2)) * 2))
        else:
            rows.append((label, 0.0, end, 2 * int(rng.integers(0, 2))))
    data = pd.DataFrame(rows, columns=["id", "start", "stop", "st"])
    data["g"] = np.where(data["id"] % 2 == 0, "even", "odd")
    data["cl"] = data["id"] // 7 - 40
    return data


def _strings(values):
    return [f"label {value}" for value in values]


def test_integer_labels_fit_exactly_as_their_codes():
    data = _cox_data(seed=5)
    data["id"] = np.arange(len(data)) * 3 - 100
    data["cl"] = np.random.default_rng(4).integers(-5, 90, len(data)) * 1_000_003
    for formula, options in [
        ("Surv(time, status) ~ g", {"cluster": "cl", "weights": "w"}),
        ("Surv(time, status) ~ 1", {"cluster": "cl", "reverse": True}),
        ("Surv(time, status) ~ g", {"id": "id", "weights": "w"}),
        ("Surv(start, time, status) ~ 1", {"id": "id", "cluster": "cl"}),
    ]:
        labelled = r.survfit(formula, data=data, **options)
        named = data.assign(id=_strings(data["id"]), cl=_strings(data["cl"]))
        coded = r.survfit(formula, data=named, **options)
        for field in ("time", "surv", "std_err", "std_chaz", "lower", "upper", "n_risk"):
            assert _identical(getattr(labelled, field), getattr(coded, field)), (formula, field)
    with_influence = r.survfit(
        "Surv(time, status) ~ g", data=data, cluster="cl", weights="w", influence=True
    )
    assert (
        with_influence.influence_surv[0].cluster[:3]
        == list(dict.fromkeys(data["cl"][data["g"] == "a"]))[:3]
    )

    ms = _ms_data()
    for formula, options in [
        ("Surv(start, stop, factor(st)) ~ 1", {"id": "id"}),
        ("Surv(start, stop, factor(st)) ~ g", {"id": "id", "cluster": "cl", "influence": True}),
    ]:
        labelled = r.survfit(formula, data=ms, **options)
        named = ms.assign(id=_strings(ms["id"]), cl=_strings(ms["cl"]))
        coded = r.survfit(formula, data=named, **options)
        for field in ("time", "pstate", "std_err", "std_auc", "lower", "upper", "n_risk"):
            assert _identical(getattr(labelled, field), getattr(coded, field)), (formula, field)
        if options.get("influence"):
            for left, right in zip(labelled.influence_pstate, coded.influence_pstate, strict=True):
                assert left.cluster == right.cluster
                np.testing.assert_array_equal(left.values, right.values)


@pytest.mark.parametrize(
    "status",
    [
        np.array([0, 2, 1, 2, 0, 1, 2, 2]),
        np.array([0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, -0.0]),
        np.array([0, 2**53 + 1, 2**53, 2**53 + 1, 0, 2**53, 2**53 + 1, 0], dtype=np.int64),
    ],
)
def test_factor_of_a_numeric_column_keeps_its_levels(status):
    """A list takes the per-value missing test the NumPy column skips."""

    data = {"time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "st": status}
    fit = r.survfit("Surv(time, factor(st)) ~ 1", data=data)
    listed = r.survfit("Surv(time, factor(st)) ~ 1", data={**data, "st": status.tolist()})
    assert len(fit.states) == 3
    assert fit.states == listed.states
    assert _identical(fit.pstate, listed.pstate)


def test_factor_of_a_numeric_column_with_nan_drops_the_missing_rows():
    status = np.array([0.0, 2.0, 1.0, np.nan, 0.0, 1.0, 2.0, 2.0])
    data = {"time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "st": status}
    fit = r.survfit("Surv(time, factor(st)) ~ 1", data=data)
    listed = r.survfit("Surv(time, factor(st)) ~ 1", data={**data, "st": status.tolist()})
    assert fit.states == listed.states
    assert fit.n == listed.n == [7]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, math.inf, 1.0], "weights must be finite"),
        ([1.0, math.nan, -1.0], "weights must be finite"),
        ([1.0, -0.5, 1.0], "weights must be non-negative"),
    ],
)
def test_case_weights_reject_non_finite_then_negative_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        _survfit._case_weights(weights, 3)
    assert _survfit._case_weights([0.0, -0.0, 2], 3) == [0.0, -0.0, 2.0]
