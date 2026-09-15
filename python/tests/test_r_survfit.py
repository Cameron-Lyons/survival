"""``survfit`` and its methods against R's ``survival`` (3.8.11) reference values.

The R-fixture suite (``test_r_fixtures.py``) covers the numerics of every engine on the
bundled data sets; these tests pin the argument handling, the result structure and a few
hand-checked numbers on small data (R: ``survfit(Surv(time, status) ~ group, toy)`` and
friends on the eight-row toy frame of ``r_api_support``).
"""

from __future__ import annotations

import math
import warnings

import pytest

from .helpers import setup_survival_import
from .r_api_support import _toy_data

survival = setup_survival_import()
r = survival.r_api


def _close(actual, expected, rel=1e-8):
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected, strict=True):
        if e is None or (isinstance(e, float) and math.isnan(e)):
            assert a is None or math.isnan(a), (actual, expected)
        else:
            assert a == pytest.approx(e, rel=rel, abs=1e-12), (actual, expected)


def _counting_data():
    return {
        "id": [1, 1, 2, 3, 3, 4],
        "start": [0, 2, 0, 0, 3, 0],
        "stop": [2, 5, 3, 3, 6, 4],
        "status": [0, 1, 1, 0, 1, 0],
    }


def _mstate_data():
    return {
        "id": [1, 2, 3, 4, 5, 6],
        "time": [1, 2, 3, 4, 5, 6],
        "ev": r._r_factor(["a", "b", "censor", "a", "censor", "b"], ["censor", "a", "b"]),
    }


# ---------------------------------------------------------------------------
# survfit for right-censored data: Kaplan-Meier
# ---------------------------------------------------------------------------


def test_survfit_formula_matches_r_kaplan_meier_by_group():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data())

    assert isinstance(fit, r.SurvfitResult)
    assert fit.strata == {"group=A": 4, "group=B": 4}
    assert fit.strata_names == ["group=A", "group=B"]
    assert fit.n == [4, 4]
    assert fit.type == "right"
    assert fit.t0 == 0.0
    assert fit.logse is True
    assert (fit.conf_int, fit.conf_type, fit.conf_lower) == (0.95, "log", None)
    _close(fit.time, [1, 2, 3, 4, 5, 6, 7, 8])
    _close(fit.n_risk, [4, 3, 2, 1, 4, 3, 2, 1])
    _close(fit.surv, [0.75, 0.5, 0.5, 0.0, 1.0, 2 / 3, 1 / 3, 1 / 3])
    _close(
        fit.std_err,
        [
            0.288675134594813,
            0.5,
            0.5,
            math.inf,
            0.0,
            0.408248290463863,
            0.816496580927726,
            0.816496580927726,
        ],
    )
    _close(
        fit.lower,
        [
            0.425932268497982,
            0.187658928706588,
            0.187658928706588,
            math.nan,
            1.0,
            0.299507130359022,
            0.0672783908519223,
            0.0672783908519223,
        ],
    )
    _close(fit.upper, [1, 1, 1, math.nan, 1, 1, 1, 1])
    assert fit.call == r.SurvfitCall(terms=("group",), stype=1, ctype=1, timefix=True)
    assert set(fit.model) == {"Surv(time, status)", "group"}
    assert isinstance(fit.model["Surv(time, status)"], r.Surv)


def test_survfit_intercept_only_has_no_strata():
    fit = r.survfit("Surv(time, status) ~ 1", _toy_data())

    assert fit.strata is None
    assert fit.strata_names == []
    assert fit.n == [8]
    assert fit.call.terms == ()


def test_survfit_surv_object_with_group_labels_curves_by_level():
    data = _toy_data()
    fit = r.survfit(r.Surv(data["time"], data["status"]), group=data["group"])
    formula_fit = r.survfit("Surv(time, status) ~ group", data)

    assert fit.strata == {"A": 4, "B": 4}
    assert fit.surv == formula_fit.surv
    assert set(fit.model) == {"response", "group"}
    assert fit.call.terms == ("group",)


def test_survfit_factor_levels_order_the_curves():
    data = _toy_data()
    data["group"] = r._r_factor(data["group"], ["B", "A"])
    fit = r.survfit("Surv(time, status) ~ group", data)

    assert list(fit.strata) == ["group=B", "group=A"]
    assert fit.n == [4, 4]
    _close(fit.time, [5, 6, 7, 8, 1, 2, 3, 4])


def test_survfit_two_terms_use_r_strata_labels():
    data = _toy_data()
    data["z"] = [0, 1, 0, 1, 0, 1, 0, 1]
    fit = r.survfit("Surv(time, status) ~ group + z", data)

    assert list(fit.strata) == ["group=A, z=0", "group=A, z=1", "group=B, z=0", "group=B, z=1"]


def test_survfit_strata_term_is_labelled_like_r():
    fit = r.survfit("Surv(time, status) ~ strata(group)", _toy_data())

    assert list(fit.strata) == ["strata(group)=A", "strata(group)=B"]


def test_survfit_rejects_interaction_terms():
    with pytest.raises(ValueError, match="Interaction terms are not valid for this function"):
        r.survfit("Surv(time, status) ~ group * x1", _toy_data())


def test_survfit_warns_about_offset_terms_and_ignores_them():
    with pytest.warns(UserWarning, match="Offset term ignored"):
        fit = r.survfit("Surv(time, status) ~ group + offset(x1)", _toy_data())
    assert fit.strata == {"group=A": 4, "group=B": 4}


def test_survfit_argument_errors_match_r():
    data = _toy_data()
    cases = [
        ({"stype": 3}, "stype must be 1 or 2"),
        ({"ctype": 0}, "ctype must be 1 or 2"),
        ({"type": "bogus"}, "invalid value for 'type'"),
        ({"type": 1}, "type argument must be character"),
        ({"conf_type": "bogus"}, "'conf.type' should be one of"),
        ({"conf_lower": "x"}, "'conf.lower' should be one of"),
        ({"time0": "yes"}, "time0 must be TRUE/FALSE"),
        ({"entry": "yes"}, "entry argument must be TRUE/FALSE"),
        ({"timefix": "yes"}, "invalid value for timefix option"),
        ({"etype": [1]}, "the etype argument is no longer supported"),
        ({"influence": 7}, "influence argument must be 0, 1, 2, or 3"),
        ({"influence": "all"}, "influence argument must be numeric or logical"),
        ({"start_time": "a"}, "start.time must be a single numeric value"),
        ({"robust": "yes"}, "robust must be TRUE/FALSE"),
        ({"weights": ["a"] * 8}, "weights must be numeric"),
        ({"weights": [-1.0] * 8}, "weights must be non-negative"),
        ({"weights": [math.inf] * 8}, "weights must be finite"),
        ({"conf_int": 1.5}, "confidence intervals must be between 0 and 1"),
    ]
    for kwargs, message in cases:
        with pytest.raises((ValueError, TypeError), match=message):
            r.survfit("Surv(time, status) ~ group", data, **kwargs)
    with pytest.raises(ValueError, match="a formula argument is required"):
        r.survfit(None)
    with pytest.raises(TypeError, match="response must be a survival object"):
        r.survfit([1, 2, 3])
    with pytest.raises(TypeError, match="unexpected keyword"):
        r.survfit("Surv(time, status) ~ group", data, bogus=1)
    with pytest.raises(ValueError, match="newdata is only used with a fitted Cox model"):
        r.survfit("Surv(time, status) ~ group", data, newdata=data)
    with pytest.raises(ValueError, match="all observations removed by start.time"):
        r.survfit("Surv(time, status) ~ 1", data, start_time=100)


def test_survfit_accepts_r_dotted_keywords():
    data = _toy_data()
    fit = r.survfit(
        "Surv(time, status) ~ 1", data, **{"se.fit": True, "conf.type": "plain", "conf.int": 0.9}
    )
    assert (fit.conf_type, fit.conf_int) == ("plain", 0.9)
    with pytest.raises(ValueError, match="use only one of conf_type or conf.type"):
        r.survfit("Surv(time, status) ~ 1", data, conf_type="plain", **{"conf.type": "log"})


def test_survfit_type_argument_maps_to_stype_ctype():
    data = _toy_data()
    fh = r.survfit("Surv(time, status) ~ 1", data, type="fleming")
    fh2 = r.survfit("Surv(time, status) ~ 1", data, type="fh")
    assert (fh.call.stype, fh.call.ctype) == (2, 1)
    assert (fh2.call.stype, fh2.call.ctype) == (2, 2)
    assert fh.surv == r.survfit("Surv(time, status) ~ 1", data, stype=2, ctype=1).surv


def test_survfit_se_fit_false_drops_the_standard_errors():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data(), se_fit=False)

    assert fit.std_err is None
    assert fit.std_chaz is None
    assert fit.lower is None
    assert fit.upper is None
    assert fit.logse is None
    assert fit.conf_type is None
    assert fit.conf_int is None
    assert fit.surv[:2] == [0.75, 0.5]


def test_survfit_conf_int_false_means_no_interval():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data(), conf_int=False)

    assert fit.conf_type == "none"
    assert fit.lower is None
    assert fit.std_err is not None
    none = r.survfit("Surv(time, status) ~ group", _toy_data(), conf_type="none")
    assert none.lower is None
    assert none.conf_int == 0.95


def test_survfit_conf_lower_peto_matches_r():
    fit = r.survfit("Surv(time, status) ~ 1", _toy_data(), conf_lower="peto")

    assert fit.conf_lower == "peto"
    _close(
        fit.lower,
        [
            0.684869554191194,
            0.517844409439955,
            0.502701841294047,
            0.34466334377755,
            0.322832826076254,
            0.166491125305491,
            0.0579005742179893,
            0.0346491185068608,
        ],
    )


def test_survfit_weights_report_unweighted_counts_and_robust_se():
    fit = r.survfit("Surv(time, status) ~ 1", _toy_data(), weights=[1, 2, 1, 2, 1, 2, 1, 2])

    _close(fit.n_risk, [12, 11, 9, 8, 6, 5, 3, 2])
    _close(fit.counts.n_risk, [8, 7, 6, 5, 4, 3, 2, 1])
    _close(fit.counts.n_event, [1, 1, 0, 1, 0, 1, 1, 0])
    assert fit.logse is True
    _close(
        fit.std_err,
        [
            0.0870388279778489,
            0.166666666666667,
            0.166666666666667,
            0.263523138347365,
            0.263523138347365,
            0.450308536203543,
            0.607819417627016,
            0.607819417627016,
        ],
        rel=1e-6,
    )


def test_survfit_start_time_conditions_the_curves():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data(), start_time=2)

    assert fit.t0 == 2.0
    assert fit.start_time == 2.0
    assert fit.call.start_time == 2.0
    assert fit.n == [3, 4]
    assert fit.strata == {"group=A": 3, "group=B": 4}
    assert fit.time[:3] == [2.0, 3.0, 4.0]


def test_survfit_start_time_removing_a_whole_curve_matches_r():
    # survfit(Surv(time, status) ~ group, start.time = 5): every group A row ends before 5,
    # so R keeps n = c(0, 4) but strata = c("group=B" = 4) (temp$strata[temp$strata > 0])
    fit = r.survfit("Surv(time, status) ~ group", _toy_data(), start_time=5)

    assert fit.n == [0, 4]
    assert fit.strata == {"group=B": 4}
    assert fit.t0 == 5.0
    _close(fit.time, [5, 6, 7, 8])
    _close(fit.n_risk, [4, 3, 2, 1])
    _close(fit.n_event, [0, 1, 1, 0])
    _close(fit.surv, [1, 2 / 3, 1 / 3, 1 / 3])
    summary = r.summary_survfit(fit)
    assert summary.strata == ["group=B", "group=B"]
    _close(summary.time, [6, 7])
    assert summary.table.rownames == ["group=B"]
    _close(summary.table.values[0], [0, 4, 4, 2, 2, 0.471404520791032, 7, 6, math.nan], 1e-6)
    at_times = r.summary_survfit(fit, times=[5, 6.5])
    assert at_times.strata == ["group=B", "group=B"]
    _close(at_times.surv, [1, 2 / 3])
    quantile = r.quantile_survfit(fit, probs=0.5)
    assert quantile.strata == ["group=B"]
    _close(quantile.quantile[0], [7])
    _close(quantile.lower[0], [6])
    assert r.survfit0(fit).strata == {"group=B": 4}

    lung = survival.datasets.load_lung()
    late = r.survfit("Surv(time, status) ~ sex", lung, start_time=1000)
    assert late.n == [2, 0]
    assert late.strata == {"sex=1": 2}
    _close(late.time, [1010, 1022])
    _close(late.n_risk, [2, 1])


def test_survfit_counting_process_with_id_and_entry_matches_r():
    fit = r.survfit("Surv(start, stop, status) ~ 1", _counting_data(), id="id", entry=True)

    assert fit.type == "counting"
    assert fit.n_id == [4]
    assert fit.logse is True
    _close(fit.time, [0, 3, 4, 5, 6])
    _close(fit.n_risk, [0, 4, 3, 2, 1])
    _close(fit.n_event, [0, 1, 0, 1, 1])
    _close(fit.n_enter, [4, 0, 0, 0, 0])
    _close(fit.surv, [1, 0.75, 0.75, 0.375, 0])
    _close(fit.std_err, [0, 0.288675134594813, 0.288675134594813, 0.763762615825973, math.inf])
    assert fit.call.id == "id"
    assert list(fit.model["(id)"]) == [1, 1, 2, 3, 3, 4]


def test_survfit_influence_returns_one_matrix_per_curve():
    data = _toy_data()
    fit = r.survfit("Surv(time, status) ~ group", data, influence=True)

    assert fit.logse is False
    assert len(fit.influence_surv) == 2
    assert len(fit.influence_chaz) == 2
    assert len(fit.influence_surv[0].values) == 4
    assert len(fit.influence_surv[0].values[0]) == 4
    only_chaz = r.survfit("Surv(time, status) ~ 1", data, influence=2)
    assert only_chaz.influence_surv is None
    assert only_chaz.influence_chaz is not None


def test_survfit_warns_when_cluster_is_ignored():
    data = _toy_data()
    data["id"] = [1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.warns(UserWarning, match="cluster specified with robust=FALSE"):
        r.survfit("Surv(time, status) ~ 1", data, cluster="id", robust=False)
    with pytest.warns(UserWarning, match="robust=FALSE implies influence=FALSE"):
        r.survfit("Surv(time, status) ~ 1", data, robust=False, influence=True)


def test_survfit_subset_and_na_action_follow_r_model_frame():
    data = _toy_data()
    data["status"] = [1, 1, 0, 1, 0, 1, None, 0]
    omitted = r.survfit("Surv(time, status) ~ group", data)
    assert omitted.n == [4, 3]
    with pytest.raises(ValueError, match="missing values"):
        r.survfit("Surv(time, status) ~ group", data, na_action="na.fail")
    subset = r.survfit("Surv(time, status) ~ group", _toy_data(), subset=[0, 1, 2, 4, 5])
    assert subset.n == [3, 2]
    assert len(subset.model["group"]) == 5


def test_survfit_time0_is_accepted_and_recorded():
    fit = r.survfit("Surv(time, status) ~ 1", _toy_data(), time0=True)
    assert fit.time0 is True
    assert r.survfit0(fit) is fit


# ---------------------------------------------------------------------------
# survfitAJ and survfitTurnbull
# ---------------------------------------------------------------------------


def test_survfit_multistate_matches_r_aalen_johansen():
    fit = r.survfit("Surv(time, ev) ~ 1", _mstate_data())

    assert isinstance(fit, r.SurvfitMultiStateResult)
    assert fit.states == ["(s0)", "a", "b"]
    assert fit.type == "mright"
    assert fit.p0 == [[1.0, 0.0, 0.0]]
    assert fit.n_id == [6]
    assert fit.n == [6]
    assert fit.hazard_names == ["1:2", "1:3"]
    assert fit.logse is False
    assert fit.conf_type == "log"
    assert fit.transitions == r.NamedMatrix(
        rownames=["(s0)", "a", "b"],
        colnames=["a", "b", "(censored)"],
        values=[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    _close([row[0] for row in fit.pstate], [5 / 6, 2 / 3, 2 / 3, 4 / 9, 4 / 9, 0])
    _close([row[1] for row in fit.pstate], [1 / 6, 1 / 6, 1 / 6, 7 / 18, 7 / 18, 7 / 18])
    _close([row[0] for row in fit.std_err][:2], [0.152145154862546, 0.192450089729875], rel=1e-6)
    fit0 = r.survfit0(fit)
    assert fit0.time[0] == 0.0
    assert fit0.pstate[0] == [1.0, 0.0, 0.0]
    assert fit0.time0


def test_survfit_multistate_argument_rules():
    data = _mstate_data()
    with pytest.raises(ValueError, match="p0 must be a numeric vector that adds to 1"):
        r.survfit("Surv(time, ev) ~ 1", data, p0=[0.5, 0.2, 0.2])
    with pytest.raises(ValueError, match="multi-state survfit supports only a robust variance"):
        r.survfit("Surv(time, ev) ~ 1", data, robust=False)
    with pytest.warns(UserWarning, match="conf.lower is ignored for multi-state data"):
        r.survfit("Surv(time, ev) ~ 1", data, conf_lower="peto")
    with pytest.warns(UserWarning, match="only stype=1, ctype=1"):
        r.survfit("Surv(time, ev) ~ 1", data, stype=2)
    counting = {
        "id": [1, 1],
        "start": [0, 1],
        "stop": [1, 2],
        "ev": r._r_factor(["a", "b"], ["censor", "a", "b"]),
    }
    with pytest.raises(ValueError, match="id statement is required"):
        r.survfit("Surv(start, stop, ev) ~ 1", counting)
    fit = r.survfit("Surv(time, ev) ~ 1", data, p0=[0.8, 0.1, 0.1], influence=True)
    assert fit.p0 == [[0.8, 0.1, 0.1]]
    assert fit.call.p0 == [0.8, 0.1, 0.1]
    assert len(fit.influence_pstate) == 1


def test_survfit_interval_censored_uses_turnbull():
    data = {"l": [1, 2, None, 4], "r": [3, 4, 2, None]}
    fit = r.survfit("Surv(l, r, type = 'interval2') ~ 1", data)

    assert isinstance(fit, r.SurvfitResult)
    assert fit.type == "interval"
    assert fit.n == [4]
    assert fit.strata is None
    _close(fit.time, [1.5, 2.5, 4.0])
    _close(fit.surv, [0.625, 0.25, 0.25])
    assert fit.engine is None
    with pytest.raises(NotImplementedError, match="interval-censored"):
        r.survfit0(fit)


# ---------------------------------------------------------------------------
# survfit0, summary.survfit, quantile.survfit
# ---------------------------------------------------------------------------


def test_survfit0_inserts_the_starting_row_per_curve():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data())
    fit0 = r.survfit0(fit)

    assert fit0.time0 is True
    assert fit0.strata == {"group=A": 5, "group=B": 5}
    _close(fit0.time, [0, 1, 2, 3, 4, 0, 5, 6, 7, 8])
    _close(fit0.surv, [1, 0.75, 0.5, 0.5, 0, 1, 1, 2 / 3, 1 / 3, 1 / 3])
    assert fit0.call == fit.call
    assert fit0.model is fit.model
    assert r.survfit0(fit0) is fit0
    with pytest.raises(TypeError, match="function requires a survfit object"):
        r.survfit0([1, 2])


def test_summary_survfit_table_matches_r():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data())
    summary = r.summary_survfit(fit)

    assert isinstance(summary, r.SummarySurvfitResult)
    assert summary.table.rownames == ["group=A", "group=B"]
    assert summary.table.colnames == [
        "records",
        "n.max",
        "n.start",
        "events",
        "rmean",
        "se(rmean)",
        "median",
        "0.95LCL",
        "0.95UCL",
    ]
    _close(summary.table.values[0], [4, 4, 4, 3, 2.75, 0.649519052838329, 3, 1, math.nan], 1e-6)
    _close(summary.table.values[1], [4, 4, 4, 2, 7.0, 0.471404520791032, 7, 6, math.nan], 1e-6)
    # event rows only, std.err on the survival scale
    _close(summary.time, [1, 2, 4, 6, 7])
    assert summary.strata == ["group=A"] * 3 + ["group=B"] * 2
    _close(summary.std_err[:2], [0.75 * 0.288675134594813, 0.25], rel=1e-6)
    _close(summary.rmean_endtime, [8.0, 8.0])


def test_summary_survfit_times_censored_scale_and_rmean_match_r():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data())
    at_times = r.summary_survfit(fit, times=[0, 2.5, 6])

    _close(at_times.time, [0, 2.5, 0, 2.5, 6])
    _close(at_times.n_risk, [4, 2, 4, 4, 3])
    _close(at_times.n_event, [0, 2, 0, 0, 1])
    _close(at_times.surv, [1, 0.5, 1, 1, 2 / 3])
    _close(at_times.std_err, [0, 0.25, 0, 0, 0.272165526975909], rel=1e-6)
    assert at_times.strata == ["group=A", "group=A", "group=B", "group=B", "group=B"]
    extended = r.summary_survfit(fit, times=[0, 2.5, 6], extend=True)
    assert len(extended.time) == 6

    scaled = r.summary_survfit(fit, censored=True, scale=2, rmean=6)
    _close(scaled.time, [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    _close(scaled.table.values[0], [4, 4, 4, 3, 1.375, 0.324759526419165, 1.5, 0.5, math.nan], 1e-6)
    _close(scaled.table.values[1][4:6], [3.0, 0.0], rel=1e-6)
    assert r.summary_survfit(fit, rmean="none").table.colnames == [
        "records",
        "n.max",
        "n.start",
        "events",
        "median",
        "0.95LCL",
        "0.95UCL",
    ]
    with pytest.raises(ValueError, match="Truncation point for the mean time in state"):
        r.summary_survfit(fit, rmean=0.5)
    with pytest.raises(ValueError, match="no values in times vector"):
        r.summary_survfit(fit, times=[])
    with pytest.raises(ValueError, match="Invalid value for rmean option"):
        r.summary_survfit(fit, rmean="bogus")
    with pytest.raises(NotImplementedError, match="summary.survfitms"):
        r.summary_survfit(r.survfit("Surv(time, ev) ~ 1", _mstate_data()))


def test_quantile_survfit_matches_r():
    fit = r.survfit("Surv(time, status) ~ group", _toy_data())
    quantiles = r.quantile_survfit(fit)

    assert quantiles.probs == [0.25, 0.5, 0.75]
    assert quantiles.strata == ["group=A", "group=B"]
    _close(quantiles.quantile[0], [1.5, 3, 4])
    _close(quantiles.quantile[1], [6, 7, math.nan])
    _close(quantiles.lower[0], [1, 1, 2])
    _close(quantiles.lower[1], [6, 6, 7])
    _close(quantiles.upper[0], [math.nan] * 3)
    without = r.quantile_survfit(fit, probs=[0.5], conf_int=False, scale=2)
    assert without.lower is None
    assert without.quantile == [[1.5], [3.5]]
    single = r.quantile_survfit(r.survfit("Surv(time, status) ~ 1", _toy_data()))
    assert single.strata is None
    assert len(single.quantile) == 1
    with pytest.raises(ValueError, match="Invalid probability"):
        r.quantile_survfit(fit, probs=[1.5])
    with pytest.raises(ValueError, match="quantiles are not a well defined quantity"):
        r.quantile_survfit(r.survfit("Surv(time, ev) ~ 1", _mstate_data()))


# ---------------------------------------------------------------------------
# aggregate.survfit, survfit_confint, the bridge influence helpers
# ---------------------------------------------------------------------------


def test_aggregate_survfit_averages_the_data_margin():
    from survival.r._types import CoxSurvfitResult

    curves = CoxSurvfitResult(
        n=[3],
        time=[1.0, 2.0],
        n_risk=[3.0, 2.0],
        n_event=[1.0, 1.0],
        n_censor=[0.0, 0.0],
        surv=[[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]],
        cumhaz=[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        type="right",
        std_err=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    )
    mean = r.aggregate_survfit(curves)
    assert isinstance(mean, CoxSurvfitResult)
    _close(mean.surv[0], [0.7])
    _close(mean.surv[1], [0.6])
    assert mean.std_err is None
    assert mean.time == curves.time
    grouped = r.aggregate_survfit(curves, by=[1, 2, 2], FUN="max")
    _close(grouped.surv[0], [0.9, 0.7])
    named = r.aggregate_survfit(curves, by={"g": ["x", "y", "y"]}, FUN="median")
    _close(named.surv[1], [0.8, 0.5])
    bridge = r.aggregate_survfit_result(curves, groups=[1, 2, 2])
    assert bridge.surv == r.aggregate_survfit(curves, by=[1, 2, 2]).surv
    with pytest.raises(ValueError, match="arguments must have the same length"):
        r.aggregate_survfit(curves, by=[1, 2])
    with pytest.raises(ValueError, match="does not have a 'data' margin"):
        r.aggregate_survfit(r.survfit("Surv(time, status) ~ 1", _toy_data()))
    with pytest.raises(ValueError, match="FUN must be one of"):
        r.aggregate_survfit(curves, FUN="sum")


def test_survfit_confint_matches_r():
    bands = r.survfit_confint([0.9, 0.5, 0.0], [0.1, 0.3, 0.2], conf_type="log-log")

    _close(bands.lower, [0.508152223312532, 0.198107119912392, math.nan], rel=1e-8)
    _close(bands.upper, [0.983735984244167, 0.743215850977201, math.nan], rel=1e-8)
    dotted = r.survfit_confint([0.9], [0.1], **{"conf.type": "plain", "conf.int": 0.9})
    _close(dotted.lower, [0.9 - 0.1 * 0.9 * 1.6448536269514722])
    with pytest.raises(TypeError, match="conf.type"):
        r.survfit_confint([0.9], [0.1])
    with pytest.raises(ValueError, match="invalid conf.int type"):
        r.survfit_confint([0.9], [0.1], conf_type="none")
    with pytest.raises(ValueError, match="confidence intervals must be between 0 and 1"):
        r.survfit_confint([0.9], [0.1], conf_type="log", conf_int=2)


def test_survfitkm_influence_helpers_return_cluster_by_time_matrices():
    data = _toy_data()
    cluster = [1, 1, 2, 2, 3, 3, 4, 4]
    influence = r.survfitkm_influence(data["time"], data["status"], cluster=cluster)
    fit = r.survfit("Surv(time, status) ~ 1", data, cluster=cluster, influence=True)

    assert influence.influence_surv == fit.influence_surv[0].values
    assert len(influence.influence_chaz) == 4
    counting = _counting_data()
    with_curve = r.survfitkm_counting_influence(
        counting["start"],
        counting["stop"],
        counting["status"],
        cluster=counting["id"],
        curve_time=[1.0],
        curve_estimate=[1.0],
    )
    assert len(with_curve.influence_surv) == 4


def test_survfit_dispatches_cox_fits_to_the_cox_module():
    data = _toy_data()
    try:
        fit = r.coxph("Surv(time, status) ~ x1", data)
    except TypeError as exc:  # pragma: no cover - the Cox module is ported separately
        pytest.skip(f"coxph is not on the new engine yet: {exc}")
    curves = r.survfit(fit)

    assert hasattr(curves, "surv")
    assert len(curves.time) > 0
    with pytest.raises(ValueError, match="clogit"):
        r.survfit(r.clogit("status ~ x1 + strata(group)", data))


def test_survfit_never_warns_on_a_plain_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.survfit("Surv(time, status) ~ group", _toy_data())
