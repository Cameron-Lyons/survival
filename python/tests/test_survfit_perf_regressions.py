"""survfit at scale, against R 4.5.3 with survival 3.8-12.

The influence matrices are read-only NumPy views of the engine's column-major arrays, so a
read copies nothing, and the influence of an estimated ``p0`` is computed in one pass over
the rows, with R's row offsets.
"""

from __future__ import annotations

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
_survfit_strata_curves = r._survfit_strata_curves


def _weighted():
    return {
        "time": [1, 2, 2, 3, 4, 5, 6, 7],
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "cl": [1, 1, 2, 2, 3, 3, 4, 4],
        "g": [1, 2, 1, 2, 1, 2, 1, 2],
        "w": [1, 2, 1, 0.5, 1, 1.5, 1, 2],
    }


# R: f <- survfit(Surv(time, status) ~ 1, d, cluster = cl, weights = w, influence = TRUE)
R_INFLUENCE_SURV = [
    [-0.070000000000000007, -0.21000000000000002, -0.1925, -0.1925]
    + [-0.12833333333333335, -0.085555555555555579, -0.085555555555555579],
    [0.015000000000000003, 0.044999999999999998, -0.012222222222222238, -0.012222222222222238]
    + [-0.0081481481481481596, -0.0054320987654321072, -0.0054320987654321072],
    [0.025000000000000001, 0.074999999999999997, 0.093055555555555544, 0.093055555555555544]
    + [-0.080555555555555575, -0.053703703703703726, -0.053703703703703726],
    [0.030000000000000006, 0.089999999999999997, 0.11166666666666665, 0.11166666666666665]
    + [0.21703703703703703, 0.14469135802469138, 0.14469135802469138],
]
R_INFLUENCE_CHAZ = [
    [0.070000000000000007] + [0.2428395061728395] * 6,
    [-0.015000000000000003, -0.052037037037037034] + [0.024351851851851847] * 5,
    [-0.025000000000000001, -0.0867283950617284, -0.12145061728395062, -0.12145061728395062]
    + [0.10077160493827159] * 3,
    [-0.030000000000000006, -0.10407407407407407, -0.14574074074074073, -0.14574074074074073]
    + [-0.36796296296296294, -0.36796296296296299, -0.36796296296296299],
]
# R: the same with stype = 2 and influence = 1, which is -influence.chaz * surv
R_STYPE2_INFLUENCE_SURV = [
    [-0.063338619262517173, -0.17594624715335638, -0.161878361968436, -0.161878361968436]
    + [-0.11599091485478169, -0.083111122235549534, -0.083111122235549534],
    [0.013572561270539395, 0.037702767247147793, -0.016233099592411791, -0.016233099592411791]
    + [-0.011631524125625667, -0.0083343512257052065, -0.0083343512257052065],
    [0.02262093545089899, 0.062837945411912996, 0.080959755254930821, 0.080959755254930821]
    + [-0.048132986400719648, -0.034488791827538022, -0.034488791827538022],
    [0.02714512254107879, 0.075405534494295587, 0.097151706305916977, 0.097151706305916977]
    + [0.17575542538112701, 0.12593426528879278, 0.12593426528879278],
]


def test_km_influence_is_a_read_only_view_with_r_values():
    fit = r.survfit(
        "Surv(time, status) ~ 1", _weighted(), cluster="cl", weights="w", influence=True
    )
    surv = fit.influence_surv[0].values
    chaz = fit.influence_chaz[0].values
    assert isinstance(surv, np.ndarray)
    np.testing.assert_allclose(surv, R_INFLUENCE_SURV, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(chaz, R_INFLUENCE_CHAZ, rtol=1e-12, atol=1e-15)
    # R's layout, and no copy: every read views the engine's matrix
    assert surv.flags.f_contiguous
    assert np.shares_memory(surv, fit.influence_surv[0].values)
    assert np.shares_memory(surv, fit.engine.influence_surv[0].values)
    assert not surv.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        surv[0, 0] = 1.0
    # the view keeps the matrix alive after the fit is gone
    del fit
    np.testing.assert_allclose(surv, R_INFLUENCE_SURV, rtol=1e-12, atol=1e-15)


def test_km_stype2_survival_influence_matches_r_alone_and_with_the_hazard():
    data = _weighted()
    alone = r.survfit(
        "Surv(time, status) ~ 1", data, cluster="cl", weights="w", stype=2, influence=1
    )
    both = r.survfit(
        "Surv(time, status) ~ 1", data, cluster="cl", weights="w", stype=2, influence=3
    )
    assert alone.influence_chaz is None
    np.testing.assert_allclose(
        alone.influence_surv[0].values, R_STYPE2_INFLUENCE_SURV, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_array_equal(both.influence_surv[0].values, alone.influence_surv[0].values)
    np.testing.assert_allclose(
        both.influence_chaz[0].values, R_INFLUENCE_CHAZ, rtol=1e-12, atol=1e-15
    )


def test_survfit0_prepends_a_zero_column_to_the_influence():
    # R: survfit0(f)$influence.surv
    fit = r.survfit(
        "Surv(time, status) ~ 1", _weighted(), cluster="cl", weights="w", influence=True
    )
    values = r.survfit0(fit).influence_surv[0].values
    assert values.shape == (4, 8)
    assert values.flags.f_contiguous
    np.testing.assert_array_equal(values[:, 0], 0.0)
    np.testing.assert_allclose(values[:, 1:], R_INFLUENCE_SURV, rtol=1e-12, atol=1e-15)


def test_strata_influence_matches_r_and_split_curves_share_it():
    # R: survfit(Surv(time, status) ~ g, d, influence = TRUE)$influence.chaz
    fit = r.survfit("Surv(time, status) ~ g", _weighted(), influence=True)
    first, second = (matrix.values for matrix in fit.influence_chaz)
    np.testing.assert_allclose(first, [[0.1875] * 4] + [[-0.0625] * 4] * 3, rtol=1e-12)
    np.testing.assert_allclose(
        second,
        [
            [0.1875] * 4,
            [-0.0625] + [0.15972222222222221] * 3,
            [-0.0625, -0.1736111111111111] + [0.076388888888888895] * 2,
            [-0.0625, -0.1736111111111111] + [-0.4236111111111111] * 2,
        ],
        rtol=1e-12,
    )
    split = _survfit_strata_curves(fit)["2"].influence_chaz[0].values
    assert np.shares_memory(split, second)


# ---------------------------------------------------------------------------
# Aalen-Johansen: the influence of an estimated p0
# ---------------------------------------------------------------------------


def _mixed_istate():
    events = ["b", "censor", "c", "b", "censor", "c", "c", "censor", "b", "c", "censor", "b"]
    istate = ["a", "b", "a", "a", "b", "b", "a", "b", "a", "b", "a", "a"]
    return {
        "id": list(range(1, 13)),
        "start": [0] * 12,
        "stop": [5, 3, 7, 2, 4, 6, 8, 1, 9, 3, 6, 5],
        "ev": r._r_factor(events, ["censor", "b", "c"]),
        "istate": r._r_factor(istate, ["a", "b", "c"]),
        "g": [1, 2] * 6,
    }


def test_estimated_p0_influence_matches_r_across_curves():
    # R: f <- survfit(Surv(start, stop, ev) ~ g, d, id = id, istate = istate, influence = TRUE)
    # p0 is estimated per curve at t0 = 2 from rows interleaved between the curves, where
    # survfitAJ.R reads the rows of curve k at the curve's offset in the sorted data
    fit = r.survfit(
        "Surv(start, stop, ev) ~ g", _mixed_istate(), id="id", istate="istate", influence=True
    )
    assert fit.t0 == 2
    np.testing.assert_allclose(fit.p0, [[5 / 6, 1 / 6, 0], [0.4, 0.6, 0]], rtol=1e-12)
    np.testing.assert_allclose(
        fit.se0,
        [[0.144337567297406, 0.144337567297406, 0], [0.14422205101856, 0.14422205101856, 0]],
        rtol=1e-12,
    )
    first, second = fit.influence_pstate
    np.testing.assert_allclose(
        first.i0,
        [[1 / 36, -1 / 36, 0], [0, 0, 0], [1 / 36, -1 / 36, 0], [0, 0, 0], [-5 / 36, 5 / 36, 0]]
        + [[0, 0, 0]],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        second.i0,
        [[0, 0, 0]] * 3 + [[-0.08, 0.08, 0], [0, 0, 0], [0.12, -0.12, 0]],
        rtol=1e-12,
        atol=1e-15,
    )
    assert not second.i0.flags.writeable
    # f$influence.pstate[[2]][, , 1]
    np.testing.assert_allclose(
        second.values[:, :, 0],
        [
            [0, 0, 0, 0],
            [-0.1, -0.1, 0, 0],
            [0] * 4,
            [-0.04, -0.04, 0, 0],
            [0] * 4,
            [0.16, 0.16, 0, 0],
        ],
        rtol=1e-12,
        atol=1e-15,
    )
    assert second.values.flags.f_contiguous
    np.testing.assert_allclose(
        fit.std_err,
        [
            [0.144337567297406, 0.144337567297406, 0],
            [0.149071198499986, 0.149071198499986, 0],
            [0.149071198499986, 0.149071198499986, 0],
            [0.178393993759886, 0.149071198499986, 0.202182875021573],
            [0.156259430442437, 0.149071198499986, 0.254452800552854],
            [0, 0.254452800552854, 0.254452800552854],
            [0.192873015219859, 0.192873015219859, 0],
            [0.192873015219859, 0.252865064294656, 0.227025859189522],
            [0, 0.227025859189522, 0.227025859189522],
            [0, 0, 0],
        ],
        rtol=1e-12,
        atol=1e-15,
    )
