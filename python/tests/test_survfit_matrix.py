"""Transition-curve products, checked against R survival and closed forms."""

import dataclasses
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from survival import r, surv_analysis

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "survfit_matrix_reference.json").read_text()
)


def _matrix(curves):
    return [[None, curves[0], curves[1]], [None, None, curves[2]], [None, None, None]]


@pytest.fixture(scope="module")
def fitted_curves():
    return {
        "km": [r.survfit("Surv(time, status) ~ g", d) for d in REFERENCE["data"]],
        "cox": [
            r.survfit(
                r.coxph("Surv(time, status) ~ x + strata(g)", d),
                newdata={"x": [-0.5, 1.0]},
            )
            for d in REFERENCE["data"]
        ],
    }


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_matrix_curves_match_r(case, fitted_curves):
    expected = case["expected"]
    p0 = (
        {"healthy": 0.8, "ill": 0.2, "dead": 0.0} if case["initial"] == "vector" else expected["p0"]
    )
    actual = r.survfit(
        _matrix(fitted_curves[case["kind"]]),
        p0=p0,
        method=case["method"],
        start_time=case["start"],
    )
    assert actual.states == expected["states"]
    assert actual.strata == expected["strata"]
    for name in ("time", "n_risk", "n_event", "pstate", "p0"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=2e-12, atol=2e-14)
    # R leaves n unrepeated for Cox prediction columns, even though the
    # result now has one curve per (stratum, newdata row). Keep it aligned.
    assert actual.n == expected["n"] * (2 if case["kind"] == "cox" else 1)
    assert actual.std_err is None
    assert actual.n_id is None


def _chain():
    curves = []
    for hazard in (0.2, 0.3):
        engine = surv_analysis.SurvfitKMResult.from_stacked(
            [1.0, 2.0],
            [10.0, 8.0],
            [1.0, 1.0],
            [math.exp(-hazard), math.exp(-2 * hazard)],
            [10],
            cumhaz=[hazard, 2 * hazard],
        )
        curve = r.survfit(r.Surv([1.0, 2.0], [1, 1]))
        curves.append(
            dataclasses.replace(
                curve,
                n=[10],
                n_risk=[10.0, 8.0],
                surv=engine.surv,
                cumhaz=engine.cumhaz,
                engine=engine,
            )
        )
    return [[None, curves[0], None], [None, None, curves[1]], [None, None, None]]


def test_unstratified_default_p0_and_exact_chain():
    discrete = r.survfit(_chain())
    np.testing.assert_allclose(discrete.pstate, [[0.8, 0.2, 0.0], [0.64, 0.3, 0.06]])
    assert discrete.states == ["1", "2", "3"]
    assert discrete.strata == {"new1": 2}
    continuous = r.survfit(np.array(_chain(), dtype=object), method="matexp")
    for t, prob in zip(continuous.time, continuous.pstate, strict=True):
        healthy = math.exp(-0.2 * t)
        ill = 2 * (math.exp(-0.2 * t) - math.exp(-0.3 * t))
        np.testing.assert_allclose(prob, [healthy, ill, 1 - healthy - ill], atol=1e-14)


def test_start_time_excludes_earlier_hazards():
    conditional = r.survfit(_chain(), [1, 0, 0], **{"start.time": 1})
    assert conditional.time == [2.0]
    np.testing.assert_allclose(conditional.pstate, [[0.8, 0.2, 0.0]])
    assert r.survfit(_chain(), start_time=3).time == []


def test_matrix_result_works_with_curve_methods():
    fit = r.survfit(_chain())
    summary = r.summary_survfit(fit, times=[0, 1, 2], extend=True)
    np.testing.assert_allclose(summary.pstate, [[1, 0, 0], *fit.pstate])
    initial = r.survfit0(fit)
    assert initial.time == [0, 1, 2]
    np.testing.assert_allclose(initial.pstate, [[1, 0, 0], *fit.pstate])
    with_time0 = r.survfit(_chain(), time0=True)
    np.testing.assert_allclose(with_time0.pstate, initial.pstate)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - bytes created in this test
    np.testing.assert_array_equal(restored.pstate, fit.pstate)
    assert r.as_data_frame(restored) == r.as_data_frame(fit)


@pytest.mark.parametrize("p0", [[1, 0], [-1, 1, 1], [1, 1, 1], [math.nan, 0, 0]])
def test_invalid_initial_distribution(p0):
    with pytest.raises(ValueError, match="p0"):
        r.survfit(_chain(), p0=p0)


def test_matrix_shape_type_and_dimensions_are_checked(fitted_curves):
    with pytest.raises(ValueError, match="square matrix"):
        r.survfit([[None, fitted_curves["km"][0]]])
    with pytest.raises(ValueError, match="2 transitions"):
        r.survfit([[None, fitted_curves["km"][0]], [None, None]])
    with pytest.raises(ValueError, match="same type"):
        r.survfit(_matrix([fitted_curves["km"][0], *fitted_curves["cox"][1:]]))
    matrix = _chain()
    matrix[0][1] = fitted_curves["km"][0]
    with pytest.raises(ValueError, match="same dimension"):
        r.survfit(matrix)
    matrix = _chain()
    matrix[0][1] = r.survfit(_chain())
    with pytest.raises(ValueError, match="multi-state"):
        r.survfit(matrix)
    with pytest.raises(ValueError, match="method"):
        r.survfit(_chain(), method="unknown")
    with pytest.raises(ValueError, match="rows for p0"):
        r.survfit(_matrix(fitted_curves["km"]), p0=[[1, 0, 0]])


def test_common_curve_start_time_is_honored(fitted_curves):
    curves = [dataclasses.replace(c, start_time=2.0) for c in fitted_curves["cox"]]
    with pytest.warns(RuntimeWarning, match="larger start.time"):
        fit = r.survfit(_matrix(curves), start_time=1.0)
    assert fit.start_time == 2.0
    assert min(fit.time) > 2.0
    curves[0] = dataclasses.replace(curves[0], start_time=None)
    with pytest.raises(ValueError, match="consistent start.time"):
        r.survfit(_matrix(curves))


def test_no_events_returns_empty_curves():
    curve = r.survfit(r.Surv([1.0, 2.0], [0, 0]))
    fit = r.survfit([[None, curve], [curve, None]])
    assert fit.time == []
    assert fit.pstate == []
    assert fit.strata == {"new1": 0}
    assert fit.p0 == [[1, 0]]


def test_matrix_exponential_preserves_tiny_transition_rates():
    matrix = _chain()
    for i, j in [(0, 1), (1, 2)]:
        c = matrix[i][j]
        hazards = np.array(c.cumhaz) * 1e-20
        engine = surv_analysis.SurvfitKMResult.from_stacked(
            c.time, c.n_risk, c.n_event, c.surv, c.n, cumhaz=hazards
        )
        matrix[i][j] = dataclasses.replace(c, engine=engine)
    matrix[0][2], matrix[1][2] = matrix[1][2], None
    fit = r.survfit(matrix, method="matexp")
    np.testing.assert_allclose(fit.pstate[0][1:], [2e-21, 3e-21], rtol=1e-14, atol=0)
