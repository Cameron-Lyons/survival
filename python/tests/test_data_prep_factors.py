"""Data-preparation functions keep factor levels, as R survival 3.8-11 does::

sp <- survSplit(Surv(time, status) ~ ., veteran, cut = 100, episode = "ep")
coef(coxph(Surv(tstart, time, status) ~ celltype + karno, sp))
d1 <- data.frame(id = 1:3, futime = c(10, 12, 15))
d2 <- data.frame(id = 1:3, t = c(4, 6, 9),
                 s = factor(c("b", "a", "b"), levels = c("cens", "b", "a")))
t2 <- tmerge(tmerge(d1, d1, id = id, tstop = futime), d2, id = id, s = event(t, s))
summary(survfit(Surv(tstart, tstop, s) ~ 1, t2, id = id), times = 10)$pstate
d3 <- data.frame(id = 1:2, t = c(3, 5), g = factor(c("x", "y")))
tmerge(t2, d3, id = id, g = tdc(t, g, "zzz"))$g
survSplit(Surv(time, st) ~ ., ms, cut = 4)$st
fg <- finegray(Surv(time, ev) ~ sex + age, data = m2, etype = "pcm")
coef(coxph(Surv(fgstart, fgstop, fgstatus) ~ sex + age, data = fg, weights = fgwt))
"""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
Factor = r.Factor


def test_refit_after_survsplit_keeps_the_reference_level():
    split = r.survSplit("Surv(time, status) ~ .", datasets.load_veteran(), cut=[100], episode="ep")
    assert split["celltype"].levels == ["squamous", "smallcell", "adeno", "large"]
    fit = r.coxph("Surv(tstart, time, status) ~ celltype + karno", split)
    assert r.coef_names(fit) == ["celltypesmallcell", "celltypeadeno", "celltypelarge", "karno"]
    assert r.coef(fit) == pytest.approx(
        [0.71533440237888479, 1.1577332667856486, 0.3256449016810426, -0.031056631730318585],
        rel=1e-9,
    )


@pytest.fixture
def event_frame():
    base = {"id": [1, 2, 3], "futime": [10, 12, 15]}
    updates = {
        "id": [1, 2, 3],
        "t": [4, 6, 9],
        "s": Factor(["b", "a", "b"], levels=["cens", "b", "a"]),
    }
    frame = r.tmerge(base, base, id="id", tstop="futime")
    return r.tmerge(frame, updates, id="id", s=r.event("t", "s"))


def test_tmerge_factor_event_keeps_its_levels_and_fits_multistate_curves(event_frame):
    assert event_frame["s"].levels == ["cens", "b", "a"]
    assert list(event_frame["s"]) == ["b", "cens", "a", "cens", "b", "cens"]
    curves = r.survfit("Surv(tstart, tstop, s) ~ 1", event_frame, id="id")
    assert list(curves.states) == ["(s0)", "b", "a"]
    assert r.summary_survfit(curves, times=[10]).pstate[0] == pytest.approx(
        [0.0, 0.66666666666666674, 0.33333333333333337]
    )


def test_tmerge_factor_tdc_appends_its_default_level(event_frame):
    updates = {"id": [1, 2], "t": [3, 5], "g": Factor(["x", "y"])}
    frame = r.tmerge(event_frame, updates, id="id", g=r.tdc("t", "g", "zzz"))
    assert frame["g"].levels == ["x", "y", "zzz"]
    assert list(frame["g"]) == ["zzz", "x", "x", "zzz", "y", "y", "zzz", "zzz"]
    assert frame["s"].levels == ["cens", "b", "a"]


def test_survsplit_relabels_a_multistate_status_as_r_does():
    data = {
        "id": [1, 2, 3, 4],
        "time": [5, 8, 3, 9],
        "st": Factor(["cens", "a", "b", "a"], levels=["cens", "a", "b"]),
    }
    split = r.survSplit("Surv(time, st) ~ .", data, cut=[4])
    assert split["st"].levels == ["censor", "a", "b"]
    assert list(split["st"]) == ["censor", "censor", "censor", "a", "b", "censor", "a"]


def test_finegray_keeps_covariate_levels():
    mgus2 = datasets.load_mgus2()
    mgus2["time"] = [
        ptime if pstat == 1 else futime
        for ptime, pstat, futime in zip(
            mgus2["ptime"], mgus2["pstat"], mgus2["futime"], strict=True
        )
    ]
    codes = [
        1 if pstat == 1 else 2 * death
        for pstat, death in zip(mgus2["pstat"], mgus2["death"], strict=True)
    ]
    mgus2["ev"] = Factor(
        [["c", "pcm", "death"][int(code)] for code in codes], ["c", "pcm", "death"]
    )
    expanded = r.finegray("Surv(time, ev) ~ sex + age", data=mgus2, etype="pcm")
    assert expanded["sex"].levels == ["F", "M"]
    fit = r.coxph("Surv(fgstart, fgstop, fgstatus) ~ sex + age", data=expanded, weights="fgwt")
    assert r.coef_names(fit) == ["sexM", "age"]
    assert r.coef(fit) == pytest.approx([-0.25975679140012597, -0.017301705075068171], rel=1e-9)
