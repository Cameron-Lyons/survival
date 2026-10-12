"""R's ``summary(fit, times = c(100, 200), data.frame = TRUE)`` on survival 3.8-11 curves."""

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


def test_kaplan_meier_summary_frame_has_r_columns():
    curves = r.survfit("Surv(time, status) ~ sex", survival.datasets.load_lung())
    frame = r.summary_survfit(curves, times=[100, 200], data_frame=True)
    assert list(frame) == [
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "surv",
        "cumhaz",
        "std.err",
        "std.chaz",
        "lower",
        "upper",
        "strata",
    ]
    assert frame["n.risk"] == [114.0, 78.0, 82.0, 66.0]
    assert frame["n.censor"] == [0.0, 6.0, 1.0, 5.0]
    assert frame["strata"] == ["sex=1", "sex=1", "sex=2", "sex=2"]
    assert r.as_data_frame(r.summary_survfit(curves, times=[100, 200])) == frame


def test_multistate_summary_frame_is_long_by_state():
    mgus2 = survival.datasets.load_mgus2()
    mgus2["etime"] = [min(p, f) for p, f in zip(mgus2["ptime"], mgus2["futime"], strict=True)]
    codes = [1 if p == 1 else 2 * d for p, d in zip(mgus2["pstat"], mgus2["death"], strict=True)]
    labels = ["censor", "pcm", "death"]
    mgus2["event"] = r.Factor([labels[int(code)] for code in codes], labels)
    curves = r.survfit("Surv(etime, event) ~ sex", mgus2)
    frame = r.summary_survfit(curves, times=[100, 200], data_frame=True)
    assert list(frame) == [
        "time",
        "n.risk",
        "n.event",
        "n.censor",
        "pstate",
        "std.err",
        "lower",
        "upper",
        "strata",
        "state",
    ]
    assert frame["state"] == ["(s0)"] * 4 + ["pcm"] * 4 + ["death"] * 4
    assert frame["strata"] == ["sex=F", "sex=F", "sex=M", "sex=M"] * 3
    assert frame["pstate"] == pytest.approx(
        [0.52133660, 0.25041822, 0.43077900, 0.20314811, 0.06196166, 0.09535198]
        + [0.04865458, 0.08914153, 0.41670174, 0.65422980, 0.52056642, 0.70771036],
        abs=1e-8,
    )
