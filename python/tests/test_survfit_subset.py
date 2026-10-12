"""R's ``[.survfit`` and ``[.survfitms`` on formula curves; values from R survival 3.8-11::

sf <- survfit(Surv(time, status) ~ ph.ecog, lung)
dim(sf); sf[c(3, 1)]$strata; summary(sf["ph.ecog=2"], times = 100)$surv
m <- survfit(Surv(etime, event) ~ sex, data = mgus2)   # event = censor/pcm/death
dim(m); tail(m[2, "death"]$pstate, 2)
"""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


@pytest.fixture(scope="module")
def ecog_curves():
    return r.survfit("Surv(time, status) ~ ph.ecog", survival.datasets.load_lung())


def test_curves_are_selected_by_index_or_label_in_order(ecog_curves):
    assert ecog_curves.dim == {"strata": 4}
    reordered = ecog_curves.subset(strata=[2, 0])
    assert reordered.strata == {"ph.ecog=2": 49, "ph.ecog=0": 62}
    assert reordered.n == [50, 63]
    single = ecog_curves.subset(strata="ph.ecog=2")
    assert single.strata is None
    assert r.summary_survfit(single, times=[100]).surv == pytest.approx([0.758974358974359])
    assert ecog_curves.subset(strata=1, drop=False).strata == {"ph.ecog=1": 104}
    with pytest.raises(ValueError, match="not one of"):
        ecog_curves.subset(strata="ph.ecog=9")


def test_multistate_curves_select_strata_and_states():
    mgus2 = survival.datasets.load_mgus2()
    mgus2["etime"] = [min(p, f) for p, f in zip(mgus2["ptime"], mgus2["futime"], strict=True)]
    codes = [1 if p == 1 else 2 * d for p, d in zip(mgus2["pstat"], mgus2["death"], strict=True)]
    labels = ["censor", "pcm", "death"]
    mgus2["event"] = r.Factor([labels[int(code)] for code in codes], labels)
    curves = r.survfit("Surv(etime, event) ~ sex", mgus2)
    assert curves.dim == {"strata": 2, "states": 3}
    selected = curves.subset(strata=1, states="death")
    assert selected.dim == {"strata": 1, "states": 1}
    assert len(selected.time) == 227
    assert np.asarray(selected.pstate)[-2:].ravel() == pytest.approx(
        [0.799436406979631, 0.895539770023348]
    )
