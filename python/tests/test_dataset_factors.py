"""Bundled data sets keep R's factor levels; values from R survival 3.8-11::

coef(coxph(Surv(time, status) ~ celltype, veteran))
coef(coxph(Surv(time, status) ~ rx, colon, subset = etype == 2))
coef(coxph(Surv(time, status == 2) ~ sex, pbc))
summary(survfit(Surv(days, outcome) ~ 1, hoel), times = 600)$pstate
summary(survfit(Surv(start, stop, event) ~ 1, mgus1, id = id), times = 3650)$pstate
"""

import copy
import pickle

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def test_factor_columns_carry_r_levels_and_stay_lists():
    celltype = datasets.load_veteran()["celltype"]
    assert isinstance(celltype, list)
    assert isinstance(celltype, r.Factor)
    assert celltype.levels == ["squamous", "smallcell", "adeno", "large"]
    assert celltype[:2] == ["squamous", "squamous"]
    assert celltype[:2].categories == celltype.categories
    assert datasets.load_heart()["transplant"].levels == ["0", "1"]


def test_models_use_r_reference_levels():
    fit = r.coxph("Surv(time, status) ~ celltype", datasets.load_veteran())
    assert r.coef_names(fit) == ["celltypesmallcell", "celltypeadeno", "celltypelarge"]
    assert r.coef(fit) == pytest.approx(
        [1.0012531827862339, 1.1477130366174952, 0.23014551673309885], rel=1e-9
    )
    colon = datasets.load_colon()
    fit = r.coxph("Surv(time, status) ~ rx", colon, subset=[etype == 2 for etype in colon["etype"]])
    assert r.coef_names(fit) == ["rxLev", "rxLev+5FU"]
    assert r.coef(fit) == pytest.approx([-0.026637463391311861, -0.37171027574353133], rel=1e-9)
    fit = r.coxph("Surv(time, status == 2) ~ sex", datasets.load_pbc())
    assert r.coef_names(fit) == ["sexf"]
    assert r.coef(fit) == pytest.approx([-0.38085155041667917], rel=1e-9)


def test_factor_status_data_sets_fit_multistate_curves():
    hoel = r.survfit("Surv(days, outcome) ~ 1", datasets.load_hoel())
    assert list(hoel.states) == ["(s0)", "thymic lymphoma", "reticulum cell sarcoma", "other"]
    assert r.summary_survfit(hoel, times=[600]).pstate[0] == pytest.approx(
        [0.44198895027624419, 0.26519337016574612, 0.093922651933701862, 0.19889502762430963]
    )
    mgus1 = r.survfit("Surv(start, stop, event) ~ 1", datasets.load_mgus1(), id="id")
    assert list(mgus1.states) == ["(s0)", "pcm", "death"]
    assert r.summary_survfit(mgus1, times=[3650]).pstate[0] == pytest.approx(
        [0.55601659751037502, 0.062240663900415057, 0.38174273858921154]
    )


def test_factor_follows_r_factor_semantics():
    assert r.Factor(["b", "a", None, "b"]).levels == ["a", "b"]
    assert r.Factor(["10", "9", "2"]).levels == ["10", "2", "9"]
    declared = r.Factor(["low", "high", "mid"], levels=["low", "mid"], ordered=True)
    assert list(declared) == ["low", None, "mid"]
    assert declared.ordered
    restored = pickle.loads(pickle.dumps(declared))
    for clone in (copy.copy(declared), declared.copy(), restored):
        assert list(clone) == list(declared)
        assert (clone.categories, clone.ordered) == (("low", "mid"), True)
