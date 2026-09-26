"""survfit at scale, against R 4.5.3 with survival 3.8-12.

The influence of an estimated ``p0`` is computed in one pass over the rows, with R's row
offsets.
"""

from __future__ import annotations

import numpy as np

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api


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
