import math
from bisect import bisect_right

import pytest

from .helpers import setup_survival_import
from .r_api_support import (
    _backtick_data,
    _factor_data,
    _manual_fh_from_km,
    _manual_fh_std_chaz_from_km,
    _plain_confidence_interval,
    _take,
    _toy_data,
)

survival = setup_survival_import()


def test_survfit_multistate_matches_r_aalen_johansen_fixture():
    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ["censor", "ill", "death", "ill", "censor", "death"],
        type="mstate",
    )

    fit = survival.survfit(response)

    assert isinstance(fit, survival.SurvfitMultiStateResult)
    assert fit.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert fit.states == ("(s0)", "death", "ill")
    assert fit.transitions == ((0, 1), (0, 2))
    assert fit.transition_labels == (("(s0)", "death"), ("(s0)", "ill"))
    assert fit.p0 == pytest.approx([1.0, 0.0, 0.0])
    assert fit.t0 == 0.0
    assert fit.n == fit.n_id == 6
    expected_risk = [[value, 0.0, 0.0] for value in [6, 5, 4, 3, 2, 1]]
    expected_pstate = [
        [1.0, 0.0, 0.0],
        [0.8, 0.0, 0.2],
        [0.6, 0.2, 0.2],
        [0.4, 0.2, 0.4],
        [0.4, 0.2, 0.4],
        [0.0, 0.6, 0.4],
    ]
    expected_cumhaz = [
        [0.0, 0.0],
        [0.0, 0.2],
        [0.25, 0.2],
        [0.25, 0.5333333333333333],
        [0.25, 0.5333333333333333],
        [1.25, 0.5333333333333333],
    ]
    for actual, expected in zip(fit.n_risk, expected_risk, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.pstate, expected_pstate, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.cumhaz, expected_cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.n_event == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert fit.std_err is not None
    assert fit.std_err[1] == pytest.approx([0.1788854382, 0.0, 0.1788854382])
    assert fit.std_chaz is not None
    assert fit.std_chaz[-1] == pytest.approx([0.2165063509, 0.3256901504])
    assert fit.conf_lower is not None
    assert fit.conf_lower[1] == pytest.approx([0.5161257603, 0.0, 0.0346491185])
    assert fit.conf_upper is not None
    assert fit.conf_upper[1] == pytest.approx([1.0, 0.0, 1.0])
    assert fit.n_risk_count == fit.n_risk
    assert fit.n_transition_count == fit.n_transition

    time0_fit = survival.survfit(response, time0=True)
    assert time0_fit.time == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert time0_fit.n_risk[0] == pytest.approx([0.0, 0.0, 0.0])
    assert time0_fit.pstate[0] == pytest.approx([1.0, 0.0, 0.0])

    conditional = survival.survfit(response, start_time=3.0)
    assert conditional.time == pytest.approx([3.0, 4.0, 5.0, 6.0])
    assert conditional.n_risk[0] == pytest.approx([4.0, 0.0, 0.0])
    assert conditional.pstate[0] == pytest.approx([0.75, 0.25, 0.0])

    without_se = survival.survfit(response, se_fit=False)
    assert without_se.std_err is None
    assert without_se.std_chaz is None
    assert without_se.conf_lower is None
    assert without_se.conf_upper is None

    zero_weight = survival.survfit(response, weights=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert zero_weight.n_risk[0] == pytest.approx([5.0, 0.0, 0.0])
    assert zero_weight.n_event[1] == pytest.approx([0.0, 0.0, 0.0])


def test_survfit_multistate_supports_formula_groups_weights_and_etype():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "event": ["censor", "ill", "death", "ill", "censor", "death"],
        "status": [0, 1, 1, 1, 0, 1],
        "kind": ["none", "ill", "death", "ill", "none", "death"],
        "arm": ["b", "a", "b", "a", "b", "a"],
        "weight": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    }

    grouped = survival.survfit(
        "Surv(time, event, type='mstate') ~ arm",
        data=data,
        weights=data["weight"],
    )

    assert list(grouped) == ["a", "b"]
    assert grouped["a"].time == pytest.approx([2.0, 4.0, 6.0])
    assert grouped["a"].n_risk == [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert grouped["a"].pstate[0] == pytest.approx([0.5, 0.0, 0.5])
    assert grouped["a"].pstate[-1] == pytest.approx([0.0, 0.25, 0.75])
    assert grouped["b"].time == pytest.approx([1.0, 3.0, 5.0])
    assert grouped["b"].pstate[-1] == pytest.approx([0.5, 0.5, 0.0])

    etype_fit = survival.survfit(
        "Surv(time, status) ~ 1",
        data=data,
        etype="kind",
    )
    direct_etype = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        etype=data["kind"],
    )
    assert etype_fit.states == direct_etype.states == ("(s0)", "death", "ill")
    for actual, expected in zip(etype_fit.pstate, direct_etype.pstate, strict=True):
        assert actual == pytest.approx(expected)


def test_survfit_multistate_supports_p0_and_survfit0():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        Factor(
            ["ill", "death", "censor", "death"],
            ["censor", "ill", "death"],
        ),
        type="mstate",
    )
    p0 = [0.25, 0.5, 0.25]

    fit = survival.survfit(response, p0=p0)

    assert fit.p0 == pytest.approx(p0)
    expected_pstate = [
        [0.1875, 0.5625, 0.25],
        [0.125, 0.5625, 0.3125],
        [0.125, 0.5625, 0.3125],
        [0.0, 0.5625, 0.4375],
    ]
    for actual, expected in zip(fit.pstate, expected_pstate, strict=True):
        assert actual == pytest.approx(expected)

    inserted = survival.survfit0(fit)
    assert inserted.time == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert inserted.pstate[0] == pytest.approx(p0)
    assert inserted.n_risk[0] == pytest.approx([4.0, 0.0, 0.0])
    assert inserted.n_event[0] == pytest.approx([0.0, 0.0, 0.0])
    assert inserted.n_censor[0] == pytest.approx([0.0, 0.0, 0.0])
    assert inserted.n_transition[0] == pytest.approx([0.0, 0.0])
    assert inserted.cumhaz[0] == pytest.approx([0.0, 0.0])
    assert inserted.std_err is not None
    assert inserted.std_err[0] == pytest.approx([0.0, 0.0, 0.0])
    assert inserted.conf_lower is not None
    assert inserted.conf_lower[0] == pytest.approx([0.0, 0.0, 0.0])
    assert inserted.conf_upper is not None
    assert inserted.conf_upper[0] == pytest.approx([0.0, 0.0, 0.0])
    assert survival.survfit0(inserted) is inserted

    direct_time0 = survival.survfit(response, p0=p0, time0=True)
    assert direct_time0.time == pytest.approx(inserted.time)
    for actual, expected in zip(direct_time0.pstate, inserted.pstate, strict=True):
        assert actual == pytest.approx(expected)

    grouped = survival.survfit(response, group=["a", "a", "b", "b"], p0=p0)
    grouped_inserted = survival.survfit0(grouped)
    assert list(grouped_inserted) == ["a", "b"]
    for curve in grouped_inserted.values():
        assert curve.time[0] == pytest.approx(0.0)
        assert curve.pstate[0] == pytest.approx(p0)


def test_survfit_multistate_validates_p0():
    response = survival.Surv(
        [1.0, 2.0, 3.0],
        ["censor", "ill", "death"],
        type="mstate",
    )

    assert survival.survfit(response, p0=[]).p0 == pytest.approx([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="one probability per"):
        survival.survfit(response, p0=[0.5, 0.5])
    with pytest.raises(ValueError, match="sum to 1"):
        survival.survfit(response, p0=[0.5, 0.4, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        survival.survfit(response, p0=[0.5, 0.6, -0.1])
    with pytest.raises(ValueError, match="finite"):
        survival.survfit(response, p0=[0.5, 0.5, math.nan])
    with pytest.raises(TypeError, match="numeric vector"):
        survival.survfit(response, p0="bad")
    with pytest.raises(TypeError, match="numeric vector"):
        survival.survfit(response, p0=[True, False, False])
    with pytest.raises(ValueError, match="only supported for multi-state"):
        survival.survfit(survival.Surv([1.0, 2.0], [0, 1]), p0=[1.0])


def test_survfit_multistate_validates_unsupported_options():
    response = survival.Surv(
        [1.0, 2.0, 3.0],
        ["censor", "ill", "death"],
        type="mstate",
    )
    with pytest.raises(ValueError, match="robust variance"):
        survival.survfit(response, robust=False)
    with pytest.raises(ValueError, match="Aalen-Johansen"):
        survival.survfit(response, type="fh")
    with pytest.raises(ValueError, match="reverse"):
        survival.survfit(response, reverse=True)
    with pytest.raises(ValueError, match="both istate and etype"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1]),
            istate=["entry", "entry"],
            etype=["none", "death"],
        )


def test_survfit_counting_multistate_matches_r_subject_history_fixture():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    subject_id = [1, 1, 2, 3, 3]
    response = survival.Surv(
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 3.0, 2.0, 1.0, 3.0],
        Factor(
            ["ill", "censor", "death", "censor", "death"],
            ["censor", "ill", "death"],
        ),
        type="mstate",
    )

    fit = survival.survfit(response, id=subject_id)

    assert fit.time == pytest.approx([1.0, 2.0, 3.0])
    assert fit.states == ("(s0)", "ill", "death")
    assert fit.transitions == ((0, 1), (0, 2))
    assert fit.n == 5
    assert fit.n_id == 3
    assert fit.p0 == pytest.approx([1.0, 0.0, 0.0])
    assert fit.n_risk == [
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert fit.n_event == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    assert fit.n_censor == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    expected_pstate = [
        [2 / 3, 1 / 3, 0.0],
        [1 / 3, 1 / 3, 1 / 3],
        [0.0, 1 / 3, 2 / 3],
    ]
    expected_cumhaz = [
        [1 / 3, 0.0],
        [1 / 3, 0.5],
        [1 / 3, 1.5],
    ]
    for actual, expected in zip(fit.pstate, expected_pstate, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.cumhaz, expected_cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.std_err is not None
    expected_std_err = [
        [0.272165527, 0.272165527, 0.0],
        [0.272165527, 0.272165527, 0.272165527],
        [0.0, 0.272165527, 0.272165527],
    ]
    for actual, expected in zip(fit.std_err, expected_std_err, strict=True):
        assert actual == pytest.approx(expected)

    with_entry = survival.survfit(response, id=subject_id, entry=True)
    assert with_entry.time == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert with_entry.n_enter is not None
    assert with_entry.n_enter[0] == pytest.approx([3.0, 0.0, 0.0])
    assert with_entry.n_enter_count is not None
    assert with_entry.n_enter_count[0] == pytest.approx([3.0, 0.0, 0.0])

    independent_rows = survival.survfit(response)
    assert independent_rows.n_id == 5
    assert independent_rows.n_risk == [
        [3.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]


def test_survfit_counting_multistate_supports_istate_formula_and_etype():
    class Factor(list):
        def __init__(self, values, levels):
            super().__init__(values)
            self.categories = levels

    data = {
        "id": [1, 1, 2, 3, 3],
        "start": [0.0, 1.0, 0.0, 0.0, 1.0],
        "stop": [1.0, 3.0, 2.0, 1.0, 3.0],
        "event": Factor(
            ["ill", "censor", "death", "censor", "death"],
            ["censor", "ill", "death"],
        ),
        "status": [1, 0, 1, 0, 1],
        "kind": ["ill", "none", "death", "none", "death"],
        "state": ["entry", "ill", "entry", "entry", "entry"],
    }

    fit = survival.survfit(
        "Surv(start, stop, event) ~ 1",
        data=data,
        id="id",
        istate="state",
    )

    assert fit.states == ("entry", "ill", "death")
    assert fit.p0 == pytest.approx([1.0, 0.0, 0.0])
    assert fit.pstate[-1] == pytest.approx([0.0, 1 / 3, 2 / 3])

    etype_fit = survival.survfit(
        "Surv(start, stop, status) ~ 1",
        data=data,
        id="id",
        etype="kind",
    )
    assert etype_fit.states == ("(s0)", "death", "ill")
    assert etype_fit.pstate[-1] == pytest.approx([0.0, 2 / 3, 1 / 3])

    invalid_state = list(data["state"])
    invalid_state[1] = "entry"
    with pytest.raises(ValueError, match="istate is inconsistent"):
        survival.survfit(
            survival.Surv(data["start"], data["stop"], data["event"], type="mstate"),
            id=data["id"],
            istate=invalid_state,
        )

    heterogeneous = survival.Surv(
        [0.0, 0.0],
        [2.0, 2.0],
        Factor(["death", "death"], ["censor", "death"]),
        type="mstate",
    )
    heterogeneous_fit = survival.survfit(
        heterogeneous,
        id=[1, 2],
        istate=Factor(["entry", "ill"], ["entry", "ill", "death"]),
    )
    assert heterogeneous_fit.time == pytest.approx([2.0])
    assert heterogeneous_fit.states == ("entry", "ill", "death")
    assert heterogeneous_fit.p0 == pytest.approx([0.5, 0.5, 0.0])
    assert heterogeneous_fit.pstate[0] == pytest.approx([0.0, 0.0, 1.0])
    assert heterogeneous_fit.std_err0 == pytest.approx([0.3535533906, 0.3535533906, 0.0])

    heterogeneous_inserted = survival.survfit0(heterogeneous_fit)
    assert heterogeneous_inserted.time == pytest.approx([0.0, 2.0])
    assert heterogeneous_inserted.pstate[0] == pytest.approx([0.5, 0.5, 0.0])
    assert heterogeneous_inserted.std_err is not None
    assert heterogeneous_inserted.std_err[0] == pytest.approx([0.3535533906, 0.3535533906, 0.0])

    heterogeneous_time0 = survival.survfit(
        heterogeneous,
        id=[1, 2],
        istate=Factor(["entry", "ill"], ["entry", "ill", "death"]),
        time0=True,
    )
    assert heterogeneous_time0.time == pytest.approx([2.0])
    assert heterogeneous_time0.std_err0 is None

    delayed_start = survival.Surv(
        [0.0, 1.0, 0.0],
        [1.0, 3.0, 3.0],
        Factor(["ill", "censor", "censor"], ["censor", "ill"]),
        type="mstate",
    )
    delayed_fit = survival.survfit(
        delayed_start,
        id=[1, 1, 2],
        start_time=2.0,
    )
    assert delayed_fit.time == pytest.approx([3.0])
    assert delayed_fit.p0 == pytest.approx([0.5, 0.5])
    assert delayed_fit.pstate[0] == pytest.approx([0.5, 0.5])
    assert delayed_fit.std_err0 == pytest.approx([0.3535533906, 0.3535533906])


def test_survfit_matches_low_level_kaplan_meier():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    high_level = survival.survfit(response)
    low_level = survival.survfitkm(data["time"], data["status"])

    assert high_level.time == pytest.approx(low_level.time)
    assert high_level.estimate == pytest.approx(low_level.estimate)
    assert high_level.cumhaz == pytest.approx(low_level.cumhaz)
    assert high_level.std_chaz == pytest.approx(low_level.std_chaz)
    assert high_level.cumulative_hazard == pytest.approx(low_level.cumhaz)
    assert high_level.cumulative_hazard_std_err == pytest.approx(low_level.std_chaz)


def test_survfit_formula_groups_use_r_level_order():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 0, 1, 1, 0, 1],
        "group": ["treated", "treated", "control", "control", "treated", "control"],
    }
    response = survival.Surv(data["time"], data["status"])

    formula = survival.survfit("Surv(time, status) ~ group", data=data, se_fit=False)
    direct = survival.survfit(response, group=data["group"], se_fit=False)
    formula_frame = survival.as_data_frame(formula)

    assert list(formula) == ["control", "treated"]
    assert formula_frame["strata"] == [
        "control",
        "control",
        "control",
        "treated",
        "treated",
        "treated",
    ]
    assert list(direct) == ["treated", "control"]


def test_survfit_cluster_uses_robust_km_variance():
    response = survival.Surv([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 0, 1, 0, 1])
    subject = [1, 1, 2, 2, 3, 3]

    clustered = survival.survfit(response, cluster=subject)
    robust_id = survival.survfit(response, id=subject, robust=True)
    row_robust = survival.survfit(response, robust=True)
    weighted = survival.survfit(
        response,
        weights=[1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        cluster=subject,
    )
    fractional = survival.survfit(
        response,
        weights=[1.0, 1.5, 1.0, 1.0, 1.0, 1.0],
    )
    fractional_plain = survival.survfit(
        response,
        weights=[1.0, 1.5, 1.0, 1.0, 1.0, 1.0],
        robust=False,
    )
    plain = survival.survfit(response)
    with pytest.warns(RuntimeWarning, match="ignored"):
        ignored = survival.survfit(response, cluster=subject, robust=False)

    assert clustered.time == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert clustered.std_err == pytest.approx(
        [0.1360828, 0.2721655, 0.2721655, 0.2771598, 0.2771598, 0.0]
    )
    assert clustered.std_chaz == pytest.approx(
        [0.1360828, 0.3320419, 0.3320419, 0.4571841, 0.4571841, 0.4571841]
    )
    assert robust_id.std_err == pytest.approx(clustered.std_err)
    assert row_robust.std_err == pytest.approx(
        [0.1521452, 0.1924501, 0.1924501, 0.2222222, 0.2222222, 0.0]
    )
    assert weighted.std_err == pytest.approx(
        [0.09997917, 0.29993752, 0.29993752, 0.26876249, 0.26876249, 0.0]
    )
    assert fractional.std_err == pytest.approx(
        [0.1429946, 0.2076915, 0.2076915, 0.2173089, 0.2173089, 0.0]
    )
    assert fractional_plain.std_err != pytest.approx(fractional.std_err)
    assert ignored.std_err == pytest.approx(plain.std_err)
    assert clustered.std_err != pytest.approx(plain.std_err)


def test_survfitkm_influence_matches_r_right_censored_fixture():
    influence = survival.survfitkm_influence(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 0],
    )
    clustered = survival.survfitkm_influence(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 0],
        cluster=["z", "z", "a", "b"],
    )
    fh = survival.survfitkm_influence(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 0],
        stype=2,
    )

    assert influence.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    for actual, expected in zip(
        influence.influence_surv,
        [
            [-0.1875, -0.1875, -0.09375, -0.09375],
            [0.0625, 0.0625, 0.03125, 0.03125],
            [0.0625, 0.0625, -0.15625, -0.15625],
            [0.0625, 0.0625, 0.21875, 0.21875],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        influence.influence_chaz,
        [
            [0.1875, 0.1875, 0.1875, 0.1875],
            [-0.0625, -0.0625, -0.0625, -0.0625],
            [-0.0625, -0.0625, 0.1875, 0.1875],
            [-0.0625, -0.0625, -0.3125, -0.3125],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        clustered.influence_surv,
        [
            [-0.125, -0.125, -0.0625, -0.0625],
            [0.0625, 0.0625, -0.15625, -0.15625],
            [0.0625, 0.0625, 0.21875, 0.21875],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fh.influence_chaz, influence.influence_chaz, strict=True):
        assert actual == pytest.approx(expected)
    assert fh.influence_surv[0][0] == pytest.approx(-0.1460251467)
    assert fh.influence_surv[2][2] == pytest.approx(-0.0885687307)
    assert fh.influence_surv[3][3] == pytest.approx(0.1476145512)

    tied = survival.survfitkm_influence(
        [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        [1, 1, 0, 1, 1, 0],
        ctype=2,
    )
    fh2 = survival.survfitkm_influence(
        [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        [1, 1, 0, 1, 1, 0],
        stype=2,
        ctype=2,
    )
    assert tied.time == pytest.approx([1.0, 2.0, 3.0])
    assert tied.influence_surv[0] == pytest.approx([-0.1111111111, -0.0370370370, -0.0370370370])
    assert tied.influence_chaz[0] == pytest.approx([0.1355555556, 0.1355555556, 0.1355555556])
    assert tied.influence_chaz[5] == pytest.approx([-0.0677777778, -0.4288888889, -0.4288888889])
    assert fh2.influence_surv[0] == pytest.approx([-0.0939455051, -0.0408285470, -0.0408285470])
    assert fh2.influence_surv[5] == pytest.approx([0.0469727526, 0.1291788505, 0.1291788505])


def test_survfit_non_km_right_censored_curves_support_robust_variance():
    time = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    status = [1, 1, 0, 1, 1, 0]
    response = survival.Surv(time, status)
    cluster = list(range(len(time)))

    fh = survival.survfit(response, cluster=cluster, type="fleming-harrington")
    fh2 = survival.survfit(response, cluster=cluster, type="fh2")
    reverse_fh2 = survival.survfit(response, cluster=cluster, type="fh2", reverse=True)
    reverse_influence = survival.survfitkm_influence(
        time,
        status,
        cluster,
        reverse=True,
        stype=2,
        ctype=2,
    )
    km_survival_fh2_hazard = survival.survfit(response, cluster=cluster, stype=1, ctype=2)
    grouped = survival.survfit(
        response,
        group=["a", "a", "a", "b", "b", "b"],
        cluster=cluster,
        type="fh2",
    )
    direct_a = survival.survfit(
        survival.Surv(time[:3], status[:3]),
        cluster=cluster[:3],
        type="fh2",
    )
    direct_b = survival.survfit(
        survival.Surv(time[3:], status[3:]),
        cluster=cluster[3:],
        type="fh2",
    )

    assert fh.std_err == pytest.approx([0.1378965150, 0.1226264804, 0.1226264804])
    assert fh.std_chaz == pytest.approx([0.1924500897, 0.3333333333, 0.3333333333])
    assert fh.conf_lower == pytest.approx([0.4913843938, 0.1914131059, 0.1914131059])
    assert fh2.std_err == pytest.approx([0.1627183900, 0.1508161491, 0.1508161491])
    assert fh2.std_chaz == pytest.approx([0.2347891095, 0.5007272489, 0.5007272489])
    assert fh2.conf_lower == pytest.approx([0.4374272533, 0.1128825508, 0.1128825508])
    assert reverse_fh2.std_err == pytest.approx(
        [
            math.sqrt(sum(row[column] ** 2 for row in reverse_influence.influence_surv))
            for column in range(len(reverse_influence.time))
        ]
    )
    assert reverse_fh2.std_chaz == pytest.approx(
        [
            math.sqrt(sum(row[column] ** 2 for row in reverse_influence.influence_chaz))
            for column in range(len(reverse_influence.time))
        ]
    )
    assert km_survival_fh2_hazard.estimate == pytest.approx([2.0 / 3.0, 2.0 / 9.0, 2.0 / 9.0])
    assert km_survival_fh2_hazard.std_err == pytest.approx(
        [0.1924500897, 0.1924500897, 0.1924500897]
    )
    assert km_survival_fh2_hazard.std_chaz == pytest.approx(
        [0.2347891095, 0.5007272489, 0.5007272489]
    )
    assert grouped["a"].std_err == pytest.approx(direct_a.std_err)
    assert grouped["a"].std_chaz == pytest.approx(direct_a.std_chaz)
    assert grouped["b"].std_err == pytest.approx(direct_b.std_err)
    assert grouped["b"].std_chaz == pytest.approx(direct_b.std_chaz)


def test_survfitkm_counting_influence_matches_r_fixture():
    start = [0.0, 10.0, 25.0, 0.0, 5.0]
    stop = [10.0, 20.0, 30.0, 15.0, 25.0]
    status = [0, 0, 1, 1, 0]
    cluster = ["a", "a", "a", "b", "c"]
    curve_time = [0.0, 5.0, 15.0, 20.0, 25.0, 30.0]
    km_estimate = [1.0, 1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]
    fh_estimate = [1.0, 1.0, 0.7165313, 0.7165313, 0.7165313, 0.2635971]

    km = survival.survfitkm_counting_influence(
        start,
        stop,
        status,
        curve_time,
        km_estimate,
        cluster=cluster,
    )
    fh = survival.survfitkm_counting_influence(
        start,
        stop,
        status,
        curve_time,
        fh_estimate,
        cluster=cluster,
        stype=2,
    )

    assert km.time == pytest.approx(curve_time)
    assert km.influence_surv[0] == pytest.approx([0.0, 0.0, 0.1111111, 0.1111111, 0.1111111, 0.0])
    assert km.influence_surv[1] == pytest.approx(
        [0.0, 0.0, -0.2222222, -0.2222222, -0.2222222, 0.0]
    )
    assert km.influence_chaz[0] == pytest.approx(
        [0.0, 0.0, -0.1111111, -0.1111111, -0.1111111, -0.1111111]
    )
    assert km.influence_chaz[1] == pytest.approx(
        [0.0, 0.0, 0.2222222, 0.2222222, 0.2222222, 0.2222222]
    )
    assert fh.influence_surv[0] == pytest.approx(
        [0.0, 0.0, 0.0796146, 0.0796146, 0.0796146, 0.0292885667]
    )
    assert fh.influence_surv[1] == pytest.approx(
        [0.0, 0.0, -0.1592292, -0.1592292, -0.1592292, -0.0585771333]
    )
    assert fh.influence_chaz[0] == pytest.approx(km.influence_chaz[0])


def test_survfit_formula_cluster_terms_supply_robust_variance():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 0, 1],
        "id": [1, 1, 2, 2, 3, 3],
        "group": ["a", "a", "b", "b", "b", "a"],
    }

    cluster_only = survival.survfit("Surv(time, status) ~ cluster(id)", data=data)
    grouped = survival.survfit(
        "Surv(time, status) ~ group + cluster(id)",
        data=data,
        model=True,
    )
    external = survival.survfit("Surv(time, status) ~ group", data=data, cluster="id")

    assert isinstance(cluster_only, survival.r_api.SurvfitResult)
    assert cluster_only.std_err == pytest.approx(
        [0.1360828, 0.2721655, 0.2721655, 0.2771598, 0.2771598, 0.0]
    )
    assert list(grouped) == ["a", "b"]
    assert grouped["a"].time == pytest.approx([1.0, 2.0, 6.0])
    assert grouped["a"].std_err == pytest.approx([0.1571348, 0.3142697, 0.0])
    assert grouped["b"].time == pytest.approx([3.0, 4.0, 5.0])
    assert grouped["b"].std_err == pytest.approx([0.0, 0.3535534, 0.3535534])
    for label in grouped:
        assert external[label].std_err == pytest.approx(grouped[label].std_err)
        assert grouped[label].model["(cluster)"] == data["id"]

    with pytest.raises(ValueError, match="formula cluster"):
        survival.survfit(
            "Surv(time, status) ~ cluster(id)",
            data=data,
            cluster=data["id"],
        )


def test_survfit_honors_non_default_conf_level():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    high_level = survival.survfit(response, conf_level=0.9)
    conf_int_alias = survival.survfit(response, conf_int=0.9)
    dotted_conf_int_alias = survival.survfit(response, **{"conf.int": 0.9})
    formula_alias = survival.survfit("Surv(time, status) ~ 1", data=data, conf_int=0.9)
    low_level = survival.survfitkm(data["time"], data["status"], conf_level=0.9)
    default = survival.survfit(response)

    assert high_level.time == pytest.approx(low_level.time)
    assert high_level.conf_lower == pytest.approx(low_level.conf_lower)
    assert high_level.conf_upper == pytest.approx(low_level.conf_upper)
    assert conf_int_alias.conf_lower == pytest.approx(high_level.conf_lower)
    assert conf_int_alias.conf_upper == pytest.approx(high_level.conf_upper)
    assert dotted_conf_int_alias.conf_lower == pytest.approx(high_level.conf_lower)
    assert dotted_conf_int_alias.conf_upper == pytest.approx(high_level.conf_upper)
    assert formula_alias.conf_lower == pytest.approx(high_level.conf_lower)
    assert formula_alias.conf_upper == pytest.approx(high_level.conf_upper)
    assert high_level.conf_lower != pytest.approx(default.conf_lower)


def test_survfit_accepts_se_fit_false_for_km_outputs():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    default = survival.survfit(response)
    direct = survival.survfit(response, se_fit=False)
    dotted = survival.survfit(response, **{"se.fit": False})
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, se_fit=False)
    time0 = survival.survfit(response, se_fit=False, time0=True)
    fh = survival.survfit(response, se_fit=False, type="fleming-harrington")
    fh_default = survival.survfit(response, type="fleming-harrington")

    assert direct.time == pytest.approx(default.time)
    assert direct.n_risk == pytest.approx(default.n_risk)
    assert direct.n_event == pytest.approx(default.n_event)
    assert direct.n_censor == pytest.approx(default.n_censor)
    assert direct.estimate == pytest.approx(default.estimate)
    assert direct.cumhaz == pytest.approx(default.cumhaz)
    assert direct.std_err == []
    assert direct.std_chaz == []
    assert direct.conf_lower == []
    assert direct.conf_upper == []
    assert direct.cumulative_hazard_std_err == []
    direct_frame = survival.as_data_frame(direct)
    assert {"std.err", "lower", "upper", "std.chaz"}.isdisjoint(direct_frame)
    assert {len(values) for values in direct_frame.values()} == {len(direct.time)}

    assert dotted.estimate == pytest.approx(direct.estimate)
    assert dotted.std_err == []
    assert formula.estimate == pytest.approx(direct.estimate)
    assert formula.std_chaz == []

    assert time0.time == pytest.approx([0.0, *default.time])
    assert time0.estimate[0] == pytest.approx(1.0)
    assert time0.std_err == []
    assert time0.std_chaz == []
    assert time0.conf_lower == []
    assert time0.conf_upper == []

    assert fh.estimate == pytest.approx(fh_default.estimate)
    assert fh.cumhaz == pytest.approx(fh_default.cumhaz)
    assert fh.std_err == []
    assert fh.std_chaz == []
    assert fh.conf_lower == []
    assert fh.conf_upper == []


def test_survfit0_adds_initial_row_to_existing_km_outputs():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    base = survival.survfit(response)
    direct_time0 = survival.survfit(response, time0=True)
    no_se = survival.survfit(response, se_fit=False)

    inserted = survival.survfit0(base)
    inserted_no_se = survival.survfit0(no_se)
    already_inserted = survival.survfit0(direct_time0)

    assert inserted.time == pytest.approx(direct_time0.time)
    assert inserted.n_risk == pytest.approx(direct_time0.n_risk)
    assert inserted.n_event == pytest.approx(direct_time0.n_event)
    assert inserted.n_censor == pytest.approx(direct_time0.n_censor)
    assert inserted.estimate == pytest.approx(direct_time0.estimate)
    assert inserted.std_err == pytest.approx(direct_time0.std_err)
    assert inserted.conf_lower == pytest.approx(direct_time0.conf_lower)
    assert inserted.conf_upper == pytest.approx(direct_time0.conf_upper)
    assert inserted.cumhaz == pytest.approx(direct_time0.cumhaz)
    assert inserted.std_chaz == pytest.approx(direct_time0.std_chaz)

    assert inserted_no_se.time == pytest.approx(direct_time0.time)
    assert inserted_no_se.estimate[0] == pytest.approx(1.0)
    assert inserted_no_se.std_err == []
    assert inserted_no_se.std_chaz == []
    assert inserted_no_se.conf_lower == []
    assert inserted_no_se.conf_upper == []
    assert already_inserted is direct_time0

    with pytest.raises(TypeError, match="survfit result"):
        survival.survfit0(object())
    with pytest.raises(TypeError, match="unexpected"):
        survival.survfit0(base, "extra")


def test_survfit_coxph_accepts_conf_int_alias():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)

    alias = survival.survfit(fit, newdata={"x1": [0.5], "x2": [0.8]}, conf_int=0.9)
    direct = survival.survfit(fit, newdata={"x1": [0.5], "x2": [0.8]}, conf_level=0.9)
    default = survival.survfit(fit, newdata={"x1": [0.5], "x2": [0.8]})

    assert alias.time == pytest.approx(direct.time)
    assert alias.surv[0] == pytest.approx(direct.surv[0])
    assert alias.conf_lower[0] == pytest.approx(direct.conf_lower[0])
    assert alias.conf_upper[0] == pytest.approx(direct.conf_upper[0])
    assert alias.conf_lower[0] != pytest.approx(default.conf_lower[0])


def test_survfit_honors_conf_type():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    direct = survival.survfit(response, conf_type="plain")
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, conf_type="plain")
    dotted = survival.survfit(response, **{"conf.type": "plain"})
    low_level = survival.survfitkm(data["time"], data["status"], conf_type="plain")
    default = survival.survfit(response)
    none = survival.survfit(response, conf_type="none")
    plain_prefix = survival.survfit(response, conf_type="p")
    none_prefix = survival.survfit(response, conf_type="n")
    arcsin = survival.survfit(response, conf_type="arcsin")
    arcsin_prefix = survival.survfit(response, conf_type="a")
    log_log = survival.survfit(response, conf_type="log-log")
    log_log_prefix = survival.survfit(response, conf_type="log-")
    logit = survival.survfit(response, conf_type="logit")
    logit_prefix = survival.survfit(response, conf_type="logi")

    assert direct.conf_lower == pytest.approx(low_level.conf_lower)
    assert direct.conf_upper == pytest.approx(low_level.conf_upper)
    assert formula.conf_lower == pytest.approx(direct.conf_lower)
    assert dotted.conf_lower == pytest.approx(direct.conf_lower)
    assert dotted.conf_upper == pytest.approx(direct.conf_upper)
    assert direct.conf_lower != pytest.approx(default.conf_lower)
    assert none.conf_lower == []
    assert none.conf_upper == []
    assert plain_prefix.conf_lower == pytest.approx(direct.conf_lower)
    assert plain_prefix.conf_upper == pytest.approx(direct.conf_upper)
    assert none_prefix.conf_lower == []
    assert none_prefix.conf_upper == []
    assert arcsin_prefix.conf_lower == pytest.approx(arcsin.conf_lower)
    assert arcsin_prefix.conf_upper == pytest.approx(arcsin.conf_upper)
    assert log_log_prefix.conf_lower == pytest.approx(log_log.conf_lower)
    assert log_log_prefix.conf_upper == pytest.approx(log_log.conf_upper)
    assert logit_prefix.conf_lower == pytest.approx(logit.conf_lower)
    assert logit_prefix.conf_upper == pytest.approx(logit.conf_upper)


def test_survfit_confint_matches_r_exported_helper():
    plain = survival.survfit_confint([0.2, 0.5, 0.9], 0.1, conf_type="plain")
    log = survival.survfit_confint([0.2, 0.5, 0.9], 0.1, conf_type="log")
    log_log = survival.survfit_confint([0.2, 0.5, 0.9], 0.1, conf_type="log-log")
    logit = survival.survfit_confint([0.2, 0.5, 0.9], 0.1, conf_type="logit")
    arcsin = survival.survfit_confint([0.2, 0.5, 0.9], 0.1, conf_type="arcsin")
    boundary = survival.survfit_confint([0.0, 1.0, math.nan], [0.1, 0.1, 0.1], conf_type="log-log")
    recycled = survival.survfit_confint([0.2, 0.5], [0.1, 0.2, 0.3], conf_type="plain")
    scaled = survival.survfit_confint(
        0.5,
        0.1,
        logse=False,
        conf_type="plain",
        selow=0.05,
        ulimit=False,
    )
    empty_p_log = survival.survfit_confint([], 0.1, conf_type="log")
    empty_se_log = survival.survfit_confint(0.5, [], conf_type="log")
    empty_selow_log = survival.survfit_confint(
        [0.2, 0.5],
        [0.0, 0.1],
        conf_type="log",
        selow=[],
    )
    recycled_logse = survival.survfit_confint(
        [0.2, 0.5],
        0.1,
        logse=False,
        conf_type="plain",
    )

    assert plain.lower == pytest.approx([0.16080072, 0.4020018, 0.7236032])
    assert plain.upper == pytest.approx([0.23919928, 0.5979982, 1.0])
    assert log.lower == pytest.approx([0.164403])
    assert log.upper == pytest.approx([0.2433045])
    assert log_log.lower == pytest.approx([0.1623716])
    assert log_log.upper == pytest.approx([0.2405312])
    assert logit.lower == pytest.approx([0.1636537])
    assert logit.upper == pytest.approx([0.242082])
    assert arcsin.lower == pytest.approx([0.1623028, 0.4026280, 0.6664164])
    assert arcsin.upper == pytest.approx([0.2405760, 0.5973720, 0.9992298])
    assert all(math.isnan(value) for value in boundary.lower)
    assert all(math.isnan(value) for value in boundary.upper)
    assert recycled.lower == pytest.approx([0.16080072, 0.30400360, 0.08240216])
    assert recycled.upper == pytest.approx([0.23919928, 0.69599640, 0.31759784])
    assert scaled.lower == pytest.approx([0.4020018])
    assert scaled.upper == pytest.approx([0.6959964])
    assert empty_p_log.lower == empty_p_log.upper == []
    assert empty_se_log.lower == empty_se_log.upper == []
    assert empty_selow_log.lower == []
    assert len(empty_selow_log.upper) == 2
    assert recycled_logse.lower == pytest.approx([0.004003602, 0.010009004])
    assert recycled_logse.upper == pytest.approx([0.3959964, 0.9899910])
    assert tuple(scaled) == (scaled.lower, scaled.upper)
    dotted_conf = survival.survfit_confint(
        0.5,
        0.1,
        conf_type="plain",
        **{"conf.int": 0.9},
    )
    assert dotted_conf.lower[0] > plain.lower[1]

    with pytest.raises(ValueError, match="invalid conf.int type"):
        survival.survfit_confint(0.5, 0.1, conf_type="p")
    with pytest.raises(ValueError, match="conf_int"):
        survival.survfit_confint(0.5, 0.1, conf_type="plain", conf_int=1.0)


def test_survfit_start_time_conditions_right_censored_curve():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    keep = [idx for idx, time in enumerate(data["time"]) if time >= 4.0]
    expected = survival.survfitkm(
        [data["time"][idx] for idx in keep],
        [data["status"][idx] for idx in keep],
    )

    direct = survival.survfit(response, start_time=4.0)
    dotted = survival.survfit(response, **{"start.time": 4.0})
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, start_time=4.0)
    fh = survival.survfit(response, start_time=4.0, type="fleming-harrington")
    cumhaz, estimate = _manual_fh_from_km(expected)

    assert direct.time == pytest.approx(expected.time)
    assert direct.n_risk == pytest.approx(expected.n_risk)
    assert direct.estimate == pytest.approx(expected.estimate)
    assert dotted.time == pytest.approx(direct.time)
    assert dotted.estimate == pytest.approx(direct.estimate)
    assert formula.estimate == pytest.approx(direct.estimate)
    assert fh.cumhaz == pytest.approx(cumhaz)
    assert fh.estimate == pytest.approx(estimate)


def test_survfit_time0_adds_starting_row():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    low_level = survival.survfitkm(data["time"], data["status"])

    direct = survival.survfit(response, time0=True)
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, time0=True)
    fh = survival.survfit(response, time0=True, type="fleming-harrington")

    assert isinstance(direct, survival.r_api.SurvfitResult)
    assert direct.time == pytest.approx([0.0, *low_level.time])
    assert direct.n_risk == pytest.approx([low_level.n_risk[0], *low_level.n_risk])
    assert direct.n_event == pytest.approx([0.0, *low_level.n_event])
    assert direct.n_censor == pytest.approx([0.0, *low_level.n_censor])
    assert direct.estimate == pytest.approx([1.0, *low_level.estimate])
    assert direct.std_err == pytest.approx([0.0, *low_level.std_err])
    assert direct.conf_lower == pytest.approx([1.0, *low_level.conf_lower])
    assert direct.conf_upper == pytest.approx([1.0, *low_level.conf_upper])
    assert direct.cumhaz == pytest.approx([0.0, *low_level.cumhaz])
    assert direct.std_chaz == pytest.approx([0.0, *low_level.std_chaz])
    assert formula.estimate == pytest.approx(direct.estimate)
    assert fh.time[0] == pytest.approx(0.0)
    assert fh.estimate[0] == pytest.approx(1.0)
    assert fh.cumhaz[0] == pytest.approx(0.0)


def test_survfit_time0_uses_explicit_start_time():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    keep = [idx for idx, time in enumerate(data["time"]) if time >= 3.5]
    expected = survival.survfitkm(
        [data["time"][idx] for idx in keep],
        [data["status"][idx] for idx in keep],
    )

    direct = survival.survfit(response, start_time=3.5, time0=True)
    no_insert = survival.survfit(response, start_time=4.0, time0=True)

    assert direct.time == pytest.approx([3.5, *expected.time])
    assert direct.n_risk == pytest.approx([expected.n_risk[0], *expected.n_risk])
    assert direct.estimate == pytest.approx([1.0, *expected.estimate])
    assert no_insert.time[0] == pytest.approx(4.0)
    assert no_insert.estimate[0] != pytest.approx(1.0)


def test_survfit0_adds_initial_row_to_turnbull_outputs():
    result = survival.survfit(survival.Surv([1.0, 2.0, 3.0], [0, 1, 0], type="left"))

    inserted = survival.survfit0(result)
    already_inserted = survival.survfit0(inserted)

    assert isinstance(inserted, survival.r_api.TurnbullSurvfitResult)
    assert inserted.time_points == pytest.approx([0.0, *result.time_points])
    assert inserted.survival == pytest.approx([1.0, *result.survival])
    assert inserted.survival_lower == pytest.approx([1.0, *result.survival_lower])
    assert inserted.survival_upper == pytest.approx([1.0, *result.survival_upper])
    assert inserted.n_iter == result.n_iter
    assert inserted.converged == result.converged
    assert already_inserted is inserted


def test_survfit_bool_options_accept_numpy_bool_scalars():
    np = pytest.importorskip("numpy")
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    direct_time0 = survival.survfit(response, time0=True)
    numpy_time0 = survival.survfit(response, time0=np.bool_(True))
    assert numpy_time0.time == pytest.approx(direct_time0.time)
    assert numpy_time0.estimate == pytest.approx(direct_time0.estimate)

    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    direct_events = survival.survfit(fit, censor=False)
    numpy_events = survival.survfit(fit, censor=np.bool_(False))
    assert numpy_events.time == pytest.approx(direct_events.time)
    for actual, expected in zip(numpy_events.cumhaz, direct_events.cumhaz, strict=True):
        assert actual == pytest.approx(expected)


def test_survfit_fleming_harrington_type_matches_low_level_counts():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    km = survival.survfitkm(data["time"], data["status"])

    default = survival.survfit(response)
    kaplan_prefix = survival.survfit(response, type="kap")
    direct = survival.survfit(response, type="fleming-harrington")
    prefix = survival.survfit(response, type="fleming")
    spaced = survival.survfit(response, type=" fleming-harrington ")
    formula = survival.survfit(
        "Surv(time, status) ~ 1",
        data=data,
        type="fleming-harrington",
    )
    cumhaz, estimate = _manual_fh_from_km(km)
    std_chaz = _manual_fh_std_chaz_from_km(km)

    assert isinstance(direct, survival.r_api.SurvfitResult)
    assert kaplan_prefix.estimate == pytest.approx(default.estimate)
    assert kaplan_prefix.cumhaz == pytest.approx(default.cumhaz)
    assert direct.time == pytest.approx(km.time)
    assert direct.n_risk == pytest.approx(km.n_risk)
    assert direct.n_event == pytest.approx(km.n_event)
    assert direct.n_censor == pytest.approx(km.n_censor)
    assert direct.cumhaz == pytest.approx(cumhaz)
    assert direct.cumulative_hazard == pytest.approx(cumhaz)
    assert direct.std_chaz == pytest.approx(std_chaz)
    assert direct.cumulative_hazard_std_err == pytest.approx(std_chaz)
    assert direct.estimate == pytest.approx(estimate)
    assert direct.surv == pytest.approx(estimate)
    assert prefix.cumhaz == pytest.approx(direct.cumhaz)
    assert spaced.cumhaz == pytest.approx(direct.cumhaz)
    assert formula.cumhaz == pytest.approx(direct.cumhaz)
    assert formula.std_chaz == pytest.approx(direct.std_chaz)
    assert formula.estimate == pytest.approx(direct.estimate)


def test_survfit_fleming_harrington_conf_type_uses_transformed_estimate():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    plain = survival.survfit(response, type="fleming-harrington", conf_type="plain")
    none = survival.survfit(response, type="fleming-harrington", conf_type="none")
    expected_lower, expected_upper = _plain_confidence_interval(
        plain.estimate[0],
        plain.std_err[0],
    )

    assert plain.conf_lower[0] == pytest.approx(expected_lower)
    assert plain.conf_upper[0] == pytest.approx(expected_upper)
    assert none.conf_lower == []
    assert none.conf_upper == []


def test_survfit_fh2_type_corrects_tied_event_cumulative_hazard():
    time = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    status = [1, 1, 0, 1, 1, 0]
    response = survival.Surv(time, status)
    km = survival.survfitkm(time, status)

    simple = survival.survfit(response, type="fleming-harrington")
    fh2 = survival.survfit(response, type="fh2")
    abbreviation = survival.survfit(response, type="fh")
    modern = survival.survfit(response, stype=2, ctype=2)
    km_survival_with_fh2_hazard = survival.survfit(response, stype=1, ctype=2)
    simple_cumhaz, _simple_estimate = _manual_fh_from_km(km, ctype=1)
    corrected_cumhaz, corrected_estimate = _manual_fh_from_km(km, ctype=2)
    corrected_std_chaz = _manual_fh_std_chaz_from_km(km, ctype=2)

    assert simple.cumhaz == pytest.approx(simple_cumhaz)
    assert fh2.cumhaz == pytest.approx(corrected_cumhaz)
    assert fh2.std_chaz == pytest.approx(corrected_std_chaz)
    assert fh2.cumhaz != pytest.approx(simple.cumhaz)
    assert fh2.estimate == pytest.approx(corrected_estimate)
    assert abbreviation.cumhaz == pytest.approx(fh2.cumhaz)
    assert modern.estimate == pytest.approx(fh2.estimate)
    assert km_survival_with_fh2_hazard.estimate == pytest.approx(km.estimate)
    assert km_survival_with_fh2_hazard.cumhaz == pytest.approx(fh2.cumhaz)


def test_survfit_fh2_weighted_ties_use_unweighted_event_count():
    time = [1.0, 1.0, 2.0]
    status = [1, 1, 1]
    weights = [2.0, 1.0, 1.0]
    response = survival.Surv(time, status)
    km = survival.survfitkm(time, status, weights=weights)

    fh2 = survival.survfit(response, weights=weights, type="fh2")
    expected_cumhaz, expected_estimate = _manual_fh_from_km(
        km,
        ctype=2,
        event_counts=km.n_event_count,
    )
    expected_std_chaz = _manual_fh_std_chaz_from_km(
        km,
        ctype=2,
        event_counts=km.n_event_count,
    )

    assert km.n_event == pytest.approx([3.0, 1.0])
    assert km.n_risk_count == pytest.approx([3.0, 1.0])
    assert km.n_event_count == pytest.approx([2.0, 1.0])
    assert km.n_censor_count == pytest.approx([0.0, 0.0])
    assert fh2.cumhaz == pytest.approx(expected_cumhaz)
    assert fh2.std_chaz == pytest.approx(expected_std_chaz)
    assert fh2.estimate == pytest.approx(expected_estimate)
    assert fh2.cumhaz[0] == pytest.approx(3.0 / (2.0 * 4.0) + 3.0 / (2.0 * 2.5))


def test_survfit_reverse_matches_low_level_censoring_distribution():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    direct = survival.survfit(response, reverse=True)
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, reverse=True)
    low_level = survival.survfitkm(data["time"], data["status"], reverse=True)

    assert direct.time == pytest.approx(low_level.time)
    assert direct.n_event == pytest.approx(low_level.n_event)
    assert direct.n_censor == pytest.approx(low_level.n_censor)
    assert direct.estimate == pytest.approx(low_level.estimate)
    assert direct.cumhaz == pytest.approx(low_level.cumhaz)
    assert direct.std_chaz == pytest.approx(low_level.std_chaz)
    assert formula.estimate == pytest.approx(direct.estimate)


def test_survfit_accepts_r_style_formula_defaults():
    data = _toy_data()
    default = survival.survfit("Surv(time, status) ~ group", data=data)
    explicit = survival.survfit(
        "Surv(time, status) ~ group",
        data=data,
        id=None,
        cluster=None,
        robust=None,
        istate=None,
        etype=None,
        model=False,
        error=None,
        entry=False,
        se_fit=True,
    )
    explicit_false_robust = survival.survfit(
        "Surv(time, status) ~ group",
        data=data,
        robust=False,
    )

    assert set(explicit) == set(default)
    assert set(explicit_false_robust) == set(default)
    for label in default:
        assert explicit[label].time == pytest.approx(default[label].time)
        assert explicit[label].estimate == pytest.approx(default[label].estimate)
        assert explicit_false_robust[label].estimate == pytest.approx(default[label].estimate)

    dotted_se = survival.survfit(
        "Surv(time, status) ~ group",
        data=data,
        **{"se.fit": True},
    )
    for label in default:
        assert dotted_se[label].std_err == pytest.approx(default[label].std_err)


def test_survfit_stores_direct_and_formula_model_frames():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    weights = [1.0 + 0.1 * idx for idx in range(len(data["time"]))]

    direct = survival.survfit(response)
    grouped = survival.survfit(response, group=data["group"], weights=weights)
    formula = survival.survfit("Surv(time, status) ~ group", data=data)

    assert isinstance(direct, survival.r_api.SurvfitResult)
    assert direct.model["response"].time == pytest.approx(data["time"])
    assert direct.model["response"].event == tuple(data["status"])

    assert set(grouped) == {"A", "B"}
    for curve in grouped.values():
        assert curve.model["response"].time == pytest.approx(data["time"])
        assert curve.model["response"].event == tuple(data["status"])
        assert curve.model["group"] == data["group"]
        assert curve.model["(weights)"] == pytest.approx(weights)

    for curve in formula.values():
        assert curve.model["Surv(time, status)"].time == pytest.approx(data["time"])
        assert curve.model["Surv(time, status)"].event == tuple(data["status"])
        assert curve.model["time"] == pytest.approx(data["time"])
        assert curve.model["status"] == data["status"]
        assert curve.model["group"] == data["group"]

    direct_frame = survival.model_frame(direct)
    assert direct_frame["time"] == pytest.approx(data["time"])
    assert direct_frame["status"] == list(data["status"])

    grouped_frame = survival.model_frame(grouped)
    assert grouped_frame["time"] == pytest.approx(data["time"])
    assert grouped_frame["status"] == list(data["status"])
    assert grouped_frame["group"] == data["group"]
    assert grouped_frame["(weights)"] == pytest.approx(weights)

    formula_frame = survival.model_frame(formula)
    assert formula_frame["time"] == pytest.approx(data["time"])
    assert formula_frame["status"] == data["status"]
    assert formula_frame["group"] == data["group"]

    default_frame = survival.model_frame(survival.survfit(response))
    assert default_frame["time"] == pytest.approx(data["time"])
    assert default_frame["status"] == list(data["status"])


def test_survfit_error_argument_is_accepted_as_noop():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])

    default = survival.survfit(response)
    direct = survival.survfit(response, error="tsiatis")
    formula = survival.survfit("Surv(time, status) ~ 1", data=data, error="greenwood")
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    cox_default = survival.survfit(fit, newdata={"x1": [0.5], "x2": [0.8]})
    cox_error = survival.survfit(fit, newdata={"x1": [0.5], "x2": [0.8]}, error="unused")

    assert direct.time == pytest.approx(default.time)
    assert direct.estimate == pytest.approx(default.estimate)
    assert direct.std_err == pytest.approx(default.std_err)
    assert formula.estimate == pytest.approx(default.estimate)
    assert cox_error.time == pytest.approx(cox_default.time)
    for actual, expected in zip(cox_error.surv, cox_default.surv, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(cox_error.std_err, cox_default.std_err, strict=True):
        assert actual == pytest.approx(expected)


def test_aggregate_survfit_result_averages_cox_prediction_curves():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    curves = survival.survfit(
        fit,
        newdata={
            "x1": [0.2, 0.8, 1.2],
            "x2": [1.0, 0.7, 0.3],
        },
        se_fit=False,
    )

    aggregated = survival.r_api.aggregate_survfit_result(curves)
    expected = [sum(values) / len(values) for values in zip(*curves.surv, strict=True)]

    assert isinstance(aggregated, survival.r_api.CoxSurvfitResult)
    assert aggregated.time == pytest.approx(curves.time)
    assert len(aggregated.surv) == 1
    assert aggregated.surv[0] == pytest.approx(expected)
    assert aggregated.cumhaz[0] == pytest.approx(
        [math.inf if value <= 0.0 else -math.log(value) for value in expected]
    )
    assert aggregated.linear_predictors[0] == pytest.approx(
        sum(curves.linear_predictors) / len(curves.linear_predictors)
    )

    grouped = survival.r_api.aggregate_survfit_result(curves, groups=[2, 1, 2])
    expected_group = [
        (first + third) / 2.0 for first, third in zip(curves.surv[0], curves.surv[2], strict=True)
    ]
    assert len(grouped.surv) == 2
    assert grouped.surv[0] == pytest.approx(curves.surv[1])
    assert grouped.surv[1] == pytest.approx(expected_group)

    uncertain = survival.r_api.CoxSurvfitResult(
        time=[1.0, 2.0],
        surv=[[0.9, 0.8], [0.8, 0.6], [0.7, 0.4]],
        cumhaz=[[], [], []],
        linear_predictors=[1.0, 2.0, 3.0],
        std_err=[[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]],
        std_chaz=[[0.0], [0.0], [0.0]],
        conf_lower=[[0.0], [0.0], [0.0]],
        conf_upper=[[1.0], [1.0], [1.0]],
    )
    weighted = survival.r_api.aggregate_survfit_result(
        uncertain,
        groups=[2, 1, 2],
        weights=[1.0, 2.0, 3.0],
    )
    assert weighted.surv[0] == pytest.approx([0.8, 0.6])
    assert weighted.surv[1] == pytest.approx([0.75, 0.5])
    assert weighted.std_err[0] == pytest.approx([0.3, 0.4])
    assert weighted.std_err[1] == pytest.approx(
        [math.hypot(0.25 * 0.1, 0.75 * 0.2), math.hypot(0.25 * 0.2, 0.75 * 0.1)]
    )
    assert weighted.linear_predictors == pytest.approx([2.0, 2.5])
    assert len(weighted.conf_lower) == len(weighted.conf_upper) == 2

    with pytest.raises(TypeError, match="data.*margin"):
        survival.r_api.aggregate_survfit_result(object())
    with pytest.raises(ValueError, match="same length"):
        survival.r_api.aggregate_survfit_result(curves, groups=[1])
    with pytest.raises(ValueError, match="positive integer"):
        survival.r_api.aggregate_survfit_result(curves, groups=[0, 1, 1])


def test_survfit_counting_id_reports_entry_counts_without_artificial_censors():
    response = survival.Surv(
        [0.0, 10.0, 25.0, 0.0, 5.0],
        [10.0, 20.0, 30.0, 15.0, 25.0],
        [0, 0, 1, 1, 0],
    )
    subject = ["a", "a", "a", "b", "c"]

    fit = survival.survfit(response, id=subject, entry=True)
    weighted = survival.survfit(
        response,
        id=subject,
        weights=[2.0, 2.0, 2.0, 1.0, 3.0],
        entry=True,
    )
    no_entry = survival.survfit(response, id=subject)

    assert fit.time == pytest.approx([0.0, 5.0, 15.0, 20.0, 25.0, 30.0])
    assert fit.n_risk == pytest.approx([0.0, 2.0, 3.0, 2.0, 1.0, 1.0])
    assert fit.n_event == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert fit.n_censor == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    assert fit.n_enter == pytest.approx([2.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert fit.estimate == pytest.approx([1.0, 1.0, 2 / 3, 2 / 3, 2 / 3, 0.0])

    assert no_entry.n_enter is None
    assert no_entry.time == pytest.approx([10.0, 15.0, 20.0, 25.0, 30.0])
    assert no_entry.n_censor == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.0])
    assert no_entry.estimate == pytest.approx([1.0, 2 / 3, 2 / 3, 2 / 3, 0.0])

    assert weighted.n_risk == pytest.approx([0.0, 3.0, 6.0, 5.0, 3.0, 2.0])
    assert weighted.n_event == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 2.0])
    assert weighted.n_censor == pytest.approx([0.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    assert weighted.n_enter == pytest.approx([3.0, 3.0, 0.0, 0.0, 2.0, 0.0])
    assert weighted.estimate == pytest.approx([1.0, 1.0, 5 / 6, 5 / 6, 5 / 6, 0.0])


def test_survfit_counting_process_id_uses_robust_variance():
    data = {
        "start": [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        "stop": [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        "status": [0, 1, 1, 0, 0, 1],
        "id": [1, 1, 2, 2, 3, 3],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    robust = survival.survfit(response, id=data["id"], robust=True)
    plain = survival.survfit(response, id=data["id"])
    entry = survival.survfit(response, id=data["id"], robust=True, entry=True)

    assert robust.time == pytest.approx([2.0, 3.0, 5.0, 6.0, 7.0])
    assert robust.n_risk == pytest.approx([3.0, 3.0, 3.0, 2.0, 1.0])
    assert robust.n_event == pytest.approx([0.0, 1.0, 1.0, 0.0, 1.0])
    assert robust.n_censor == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0])
    assert robust.std_err == pytest.approx([0.0, 0.2721655, 0.1814437, 0.1814437, 0.0])
    assert robust.std_chaz == pytest.approx([0.0, 0.2721655, 0.2721655, 0.2721655, 0.2721655])
    assert robust.std_err != pytest.approx(plain.std_err)

    assert entry.time == pytest.approx([0.0, 3.0, 5.0, 6.0, 7.0])
    assert entry.n_enter == pytest.approx([3.0, 0.0, 0.0, 0.0, 0.0])
    assert entry.std_err == pytest.approx([0.0, 0.2721655, 0.1814437, 0.1814437, 0.0])
    assert entry.std_chaz == pytest.approx([0.0, 0.2721655, 0.2721655, 0.2721655, 0.2721655])


def test_survfit_counting_process_cluster_uses_robust_variance():
    data = {
        "start": [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        "stop": [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        "status": [0, 1, 1, 0, 0, 1],
        "id": [1, 1, 2, 2, 3, 3],
        "group": ["a", "a", "a", "b", "b", "b"],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    clustered = survival.survfit(response, cluster=data["id"])
    grouped = survival.survfit(
        "Surv(start, stop, status) ~ group + cluster(id)",
        data=data,
        model=True,
    )
    external = survival.survfit(
        "Surv(start, stop, status) ~ group",
        data=data,
        cluster="id",
    )

    assert clustered.time == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert clustered.n_risk == pytest.approx([3.0, 3.0, 3.0, 3.0, 2.0, 1.0])
    assert clustered.n_event == pytest.approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert clustered.n_censor == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert clustered.std_err == pytest.approx(
        [0.0, 0.2721655, 0.2721655, 0.1814437, 0.1814437, 0.0]
    )
    assert clustered.std_chaz == pytest.approx(
        [0.0, 0.2721655, 0.2721655, 0.2721655, 0.2721655, 0.2721655]
    )

    assert list(grouped) == ["a", "b"]
    assert grouped["a"].time == pytest.approx([2.0, 3.0, 5.0])
    assert grouped["a"].std_err == pytest.approx([0.0, 0.3535534, 0.0])
    assert grouped["a"].std_chaz == pytest.approx([0.0, 0.3535534, 0.3535534])
    assert grouped["b"].time == pytest.approx([4.0, 6.0, 7.0])
    assert grouped["b"].std_err == pytest.approx([0.0, 0.0, 0.0])
    assert grouped["b"].std_chaz == pytest.approx([0.0, 0.0, 0.0])
    for label in grouped:
        assert external[label].std_err == pytest.approx(grouped[label].std_err)
        assert grouped[label].model["(cluster)"] == data["id"]


def test_survfit_non_km_counting_process_curves_support_robust_variance():
    data = {
        "start": [0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0],
        "stop": [2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 3.0, 3.0],
        "status": [0, 1, 1, 0, 0, 1, 1, 1],
        "id": [1, 1, 2, 2, 3, 3, 4, 5],
        "group": ["a", "a", "a", "b", "b", "b", "a", "b"],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    fh = survival.survfit(response, id=data["id"], robust=True, type="fleming-harrington")
    fh2 = survival.survfit(response, id=data["id"], robust=True, type="fh2")
    km_survival_fh2_hazard = survival.survfit(
        response,
        id=data["id"],
        robust=True,
        stype=1,
        ctype=2,
    )
    grouped = survival.survfit(
        response,
        group=data["group"],
        cluster=data["id"],
        type="fh2",
    )
    direct_a = survival.survfit(
        survival.Surv(
            [data["start"][idx] for idx in (0, 1, 2, 6)],
            [data["stop"][idx] for idx in (0, 1, 2, 6)],
            [data["status"][idx] for idx in (0, 1, 2, 6)],
        ),
        cluster=[data["id"][idx] for idx in (0, 1, 2, 6)],
        type="fh2",
    )
    direct_b = survival.survfit(
        survival.Surv(
            [data["start"][idx] for idx in (3, 4, 5, 7)],
            [data["stop"][idx] for idx in (3, 4, 5, 7)],
            [data["status"][idx] for idx in (3, 4, 5, 7)],
        ),
        cluster=[data["id"][idx] for idx in (3, 4, 5, 7)],
        type="fh2",
    )

    assert fh.time == pytest.approx([2.0, 3.0, 5.0, 6.0, 7.0])
    assert fh.estimate == pytest.approx(
        [1.0, 0.5488116361, 0.3932407209, 0.3932407209, 0.1446651766]
    )
    assert fh.std_err == pytest.approx(
        [0.0, 0.1202386052, 0.1095651003, 0.1095651003, 0.0403067479]
    )
    assert fh.std_chaz == pytest.approx(
        [0.0, 0.2190890230, 0.2786209426, 0.2786209426, 0.2786209426]
    )
    assert fh2.estimate == pytest.approx(
        [1.0, 0.4568805351, 0.3273692086, 0.3273692086, 0.1204324015]
    )
    assert fh2.std_err == pytest.approx(
        [0.0, 0.1781828362, 0.1255399539, 0.1255399539, 0.0461835681]
    )
    assert fh2.std_chaz == pytest.approx(
        [0.0, 0.3899987470, 0.3834812517, 0.3834812517, 0.3834812517]
    )
    assert fh2.conf_lower == pytest.approx(
        [1.0, 0.2127331254, 0.1543895834, 0.1543895834, 0.0567967537]
    )
    assert km_survival_fh2_hazard.estimate == pytest.approx(
        [1.0, 0.4, 0.2666666667, 0.2666666667, 0.0]
    )
    assert km_survival_fh2_hazard.std_err == pytest.approx(
        [0.0, 0.2190890230, 0.1460593487, 0.1460593487, 0.0]
    )
    assert km_survival_fh2_hazard.std_chaz == pytest.approx(
        [0.0, 0.3899987470, 0.3834812517, 0.3834812517, 0.3834812517]
    )
    assert grouped["a"].std_err == pytest.approx(direct_a.std_err)
    assert grouped["a"].std_chaz == pytest.approx(direct_a.std_chaz)
    assert grouped["b"].std_err == pytest.approx(direct_b.std_err)
    assert grouped["b"].std_chaz == pytest.approx(direct_b.std_chaz)


def test_survfit_formula_counting_id_aligns_groups_and_model_frame():
    data = {
        "start": [0.0, 10.0, 25.0, 0.0, 5.0, 5.0],
        "stop": [10.0, 20.0, 30.0, 15.0, 25.0, 18.0],
        "status": [0, 0, 1, 1, 0, 1],
        "subject": ["a", "a", "a", "b", "c", "d"],
        "arm": ["A", "A", "A", "A", "B", "B"],
    }

    grouped = survival.survfit(
        "Surv(start, stop, status) ~ arm",
        data=data,
        id="subject",
        entry=True,
        model=True,
    )
    direct_a = survival.survfit(
        survival.Surv([0.0, 10.0, 25.0, 0.0], [10.0, 20.0, 30.0, 15.0], [0, 0, 1, 1]),
        id=["a", "a", "a", "b"],
        entry=True,
    )
    direct_b = survival.survfit(
        survival.Surv([5.0, 5.0], [25.0, 18.0], [0, 1]),
        id=["c", "d"],
        entry=True,
    )

    assert list(grouped) == ["A", "B"]
    assert grouped["A"].time == pytest.approx(direct_a.time)
    assert grouped["A"].n_enter == pytest.approx(direct_a.n_enter)
    assert grouped["A"].n_censor == pytest.approx(direct_a.n_censor)
    assert grouped["A"].estimate == pytest.approx(direct_a.estimate)
    assert grouped["B"].time == pytest.approx(direct_b.time)
    assert grouped["B"].n_enter == pytest.approx(direct_b.n_enter)
    assert grouped["B"].estimate == pytest.approx(direct_b.estimate)

    for curve in grouped.values():
        assert curve.model["(id)"] == data["subject"]
        assert curve.model["subject"] == data["subject"]


def test_survfit_grouped_formula_accepts_se_fit_false():
    data = _toy_data()

    default = survival.survfit("Surv(time, status) ~ group", data=data)
    grouped = survival.survfit("Surv(time, status) ~ group", data=data, se_fit=False)
    dotted = survival.survfit("Surv(time, status) ~ group", data=data, **{"se.fit": False})

    assert set(grouped) == set(default)
    assert set(dotted) == set(default)
    for label in default:
        assert grouped[label].time == pytest.approx(default[label].time)
        assert grouped[label].estimate == pytest.approx(default[label].estimate)
        assert grouped[label].cumhaz == pytest.approx(default[label].cumhaz)
        assert grouped[label].std_err == []
        assert grouped[label].std_chaz == []
        assert grouped[label].conf_lower == []
        assert grouped[label].conf_upper == []
        assert dotted[label].estimate == pytest.approx(grouped[label].estimate)
        assert dotted[label].std_err == []
    grouped_frame = survival.as_data_frame(grouped)
    assert {"std.err", "lower", "upper", "std.chaz"}.isdisjoint(grouped_frame)
    assert {len(values) for values in grouped_frame.values()} == {len(grouped_frame["time"])}


def test_survfit_timefix_false_uses_exact_event_times():
    times = [1.0, 1.0 + 5e-10, 2.0]
    status = [1, 1, 0]
    response = survival.Surv(times, status)

    default = survival.survfit(response)
    exact = survival.survfit(response, timefix=False)
    exact_dotted = survival.survfit(
        "Surv(time, status) ~ 1",
        data={"time": times, "status": status},
        **{"time.fix": False},
    )
    exact_fh2 = survival.survfit(response, timefix=False, type="fh2")
    low_level_exact = survival.survfitkm(times, status, timefix=False)

    assert default.time == pytest.approx([1.0, 2.0])
    assert default.n_risk == pytest.approx([3.0, 1.0])
    assert default.n_event == pytest.approx([2.0, 0.0])
    assert default.estimate == pytest.approx([1.0 / 3.0, 1.0 / 3.0])

    assert exact.time == pytest.approx(times)
    assert exact.n_risk == pytest.approx([3.0, 2.0, 1.0])
    assert exact.n_event == pytest.approx([1.0, 1.0, 0.0])
    assert exact.estimate == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert exact.std_err == pytest.approx(
        [
            (2.0 / 3.0) * math.sqrt(1.0 / 6.0),
            (1.0 / 3.0) * math.sqrt(2.0 / 3.0),
            (1.0 / 3.0) * math.sqrt(2.0 / 3.0),
        ]
    )
    assert exact_dotted.time == pytest.approx(exact.time)
    assert exact_dotted.estimate == pytest.approx(exact.estimate)
    assert low_level_exact.time == pytest.approx(exact.time)
    assert low_level_exact.estimate == pytest.approx(exact.estimate)
    assert exact_fh2.cumhaz == pytest.approx([1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0])


def test_survfit_counting_process_uses_delayed_entry():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0, 3.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 0, 1, 1, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    abbreviated = survival.Surv(data["start"], data["stop"], data["status"], type="count")

    direct = survival.survfit(response)
    formula = survival.survfit("Surv(start, stop, status) ~ 1", data=data)
    named_formula = survival.survfit(
        "Surv(time=start, time2=stop, event=status) ~ 1",
        data=data,
    )
    low_level = survival.survfitkm(
        data["stop"],
        data["status"],
        entry_times=data["start"],
    )

    assert abbreviated.type == "counting"
    assert abbreviated.start == pytest.approx(response.start)
    assert direct.time == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert direct.n_risk == pytest.approx([3.0, 3.0, 3.0, 2.0])
    assert direct.n_censor == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert direct.estimate == pytest.approx([2.0 / 3.0, 4.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0])
    assert direct.cumhaz == pytest.approx(low_level.cumhaz)
    assert direct.std_chaz == pytest.approx(low_level.std_chaz)
    assert formula.estimate == pytest.approx(direct.estimate)
    assert named_formula.estimate == pytest.approx(direct.estimate)
    assert named_formula.n_risk == pytest.approx(direct.n_risk)
    assert low_level.estimate == pytest.approx(direct.estimate)


def test_survfit_counting_process_timefix_false_uses_exact_risk_sets():
    data = {
        "start": [0.0, 0.0, 1.0, 1.0 + 5e-10],
        "stop": [1.0, 1.0 + 5e-10, 2.0, 2.0],
        "status": [1, 1, 0, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    default = survival.survfit(response)
    exact = survival.survfit(response, timefix=False)
    exact_formula = survival.survfit(
        "Surv(start, stop, status) ~ 1",
        data=data,
        timefix=False,
    )
    low_level_exact = survival.survfitkm(
        data["stop"],
        data["status"],
        entry_times=data["start"],
        timefix=False,
    )

    assert default.time == pytest.approx([1.0, 2.0])
    assert default.n_risk == pytest.approx([2.0, 2.0])
    assert default.n_event == pytest.approx([2.0, 0.0])
    assert default.estimate == pytest.approx([0.0, 0.0])

    assert exact.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])
    assert exact.n_risk == pytest.approx([2.0, 2.0, 2.0])
    assert exact.n_event == pytest.approx([1.0, 1.0, 0.0])
    assert exact.n_censor == pytest.approx([0.0, 0.0, 2.0])
    assert low_level_exact.time == pytest.approx(exact.time)
    assert low_level_exact.n_risk == pytest.approx(exact.n_risk)
    assert low_level_exact.estimate == pytest.approx(exact.estimate)
    assert exact.estimate == pytest.approx([0.5, 0.25, 0.25])
    assert exact_formula.estimate == pytest.approx(exact.estimate)


def test_survfit_start_time_conditions_counting_process_curve():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0, 3.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 0, 1, 1, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    keep = [idx for idx, stop in enumerate(data["stop"]) if stop >= 3.0]
    expected = survival.survfitkm(
        [data["stop"][idx] for idx in keep],
        [data["status"][idx] for idx in keep],
        entry_times=[data["start"][idx] for idx in keep],
    )

    direct = survival.survfit(response, start_time=3.0)
    formula = survival.survfit("Surv(start, stop, status) ~ 1", data=data, start_time=3.0)

    assert direct.time == pytest.approx(expected.time)
    assert direct.n_risk == pytest.approx(expected.n_risk)
    assert direct.n_event == pytest.approx(expected.n_event)
    assert direct.estimate == pytest.approx(expected.estimate)
    assert formula.estimate == pytest.approx(direct.estimate)


def test_survfit_time0_conditions_counting_process_curve():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0, 3.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 0, 1, 1, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    keep = [idx for idx, stop in enumerate(data["stop"]) if stop >= 2.5]
    expected = survival.survfitkm(
        [data["stop"][idx] for idx in keep],
        [data["status"][idx] for idx in keep],
        entry_times=[data["start"][idx] for idx in keep],
    )

    direct = survival.survfit(response, start_time=2.5, time0=True)
    formula = survival.survfit(
        "Surv(start, stop, status) ~ 1",
        data=data,
        start_time=2.5,
        time0=True,
    )

    assert direct.time == pytest.approx([2.5, *expected.time])
    assert direct.n_risk == pytest.approx([expected.n_risk[0], *expected.n_risk])
    assert direct.n_event == pytest.approx([0.0, *expected.n_event])
    assert direct.estimate == pytest.approx([1.0, *expected.estimate])
    assert formula.estimate == pytest.approx(direct.estimate)


def test_survfit_reverse_counting_process_uses_delayed_entry():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0, 3.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 0, 1, 1, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    direct = survival.survfit(response, reverse=True)
    low_level = survival.survfitkm(
        data["stop"],
        data["status"],
        entry_times=data["start"],
        reverse=True,
    )

    assert direct.time == pytest.approx(low_level.time)
    assert direct.n_risk == pytest.approx(low_level.n_risk)
    assert direct.estimate == pytest.approx(low_level.estimate)


def test_survfit_fh_counting_process_uses_delayed_entry():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0, 3.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 5.0],
        "status": [1, 0, 1, 1, 0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    km = survival.survfitkm(
        data["stop"],
        data["status"],
        entry_times=data["start"],
    )

    direct = survival.survfit(response, type="nelson-aalen")
    formula = survival.survfit(
        "Surv(start, stop, status) ~ 1",
        data=data,
        type="fleming-harrington",
    )
    cumhaz, estimate = _manual_fh_from_km(km)
    std_chaz = _manual_fh_std_chaz_from_km(km)

    assert direct.time == pytest.approx(km.time)
    assert direct.n_risk == pytest.approx(km.n_risk)
    assert direct.cumhaz == pytest.approx(cumhaz)
    assert direct.std_chaz == pytest.approx(std_chaz)
    assert direct.estimate == pytest.approx(estimate)
    assert formula.cumhaz == pytest.approx(direct.cumhaz)
    assert formula.std_chaz == pytest.approx(direct.std_chaz)
    assert formula.estimate == pytest.approx(direct.estimate)


def test_survfit_left_censored_response_uses_turnbull_estimator():
    time = [1.0, 2.0, 3.0, 4.0]
    status = [0, 1, 0, 1]
    response = survival.Surv(time, status, type="left")

    high_level = survival.survfit(response)
    low_level = survival.turnbull_estimator(
        [0.0, 2.0, 0.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    )

    assert response.type == "left"
    assert high_level.time_points == pytest.approx(low_level.time_points)
    assert high_level.survival == pytest.approx(low_level.survival)

    with_model = survival.survfit(response, model=True)
    assert isinstance(with_model, survival.r_api.TurnbullSurvfitResult)
    assert with_model.time_points == pytest.approx(low_level.time_points)
    assert with_model.survival == pytest.approx(low_level.survival)
    assert with_model.model["response"].type == "left"
    assert with_model.model["response"].status == response.status


def test_turnbull_weights_match_replicated_rows():
    weighted = survival.turnbull_estimator(
        [0.0, 1.0, 2.0],
        [1.0, 3.0, float("inf")],
        weights=[2.0, 1.0, 3.0],
    )
    replicated = survival.turnbull_estimator(
        [0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 3.0, float("inf"), float("inf"), float("inf")],
    )

    assert weighted.time_points == pytest.approx(replicated.time_points)
    assert weighted.survival == pytest.approx(replicated.survival)


def test_survfit_interval_weights_use_weighted_turnbull_estimator():
    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, 6.0],
        [2, 3, 1, 0],
        type="interval",
    )
    weights = [2.0, 1.0, 3.0, 2.0]

    high_level = survival.survfit(response, weights=weights)
    low_level = survival.turnbull_estimator(
        [0.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, float("inf")],
        weights=weights,
    )

    assert high_level.time_points == pytest.approx(low_level.time_points)
    assert high_level.survival == pytest.approx(low_level.survival)

    with_model = survival.survfit(response, weights=weights, model=True)
    assert isinstance(with_model, survival.r_api.TurnbullSurvfitResult)
    assert with_model.time_points == pytest.approx(low_level.time_points)
    assert with_model.survival == pytest.approx(low_level.survival)
    assert with_model.model["response"].type == "interval"
    assert with_model.model["response"].status == response.status
    assert with_model.model["(weights)"] == pytest.approx(weights)


def test_survfit_formula_accepts_left_censored_surv_type():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [0, 1, 0, 1],
        "arm": ["A", "A", "B", "B"],
    }

    grouped = survival.survfit("Surv(time, status, type='left') ~ arm", data=data)
    abbreviated = survival.survfit("Surv(time, status, type='l') ~ arm", data=data)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"], type="left"),
        group=data["arm"],
    )

    assert list(grouped) == ["A", "B"]
    for label in direct:
        assert grouped[label].time_points == pytest.approx(direct[label].time_points)
        assert grouped[label].survival == pytest.approx(direct[label].survival)
        assert abbreviated[label].time_points == pytest.approx(grouped[label].time_points)
        assert abbreviated[label].survival == pytest.approx(grouped[label].survival)


def test_survfit_formula_accepts_named_surv_response_arguments():
    data = _toy_data()

    positional = survival.survfit("Surv(time, status) ~ group", data=data)
    named = survival.survfit("Surv(time=time, event=status) ~ group", data=data)
    reversed_named = survival.survfit("Surv(status=status, time=time) ~ group", data=data)

    assert list(named) == list(positional)
    assert list(reversed_named) == list(positional)
    for label in positional:
        assert named[label].estimate == pytest.approx(positional[label].estimate)
        assert reversed_named[label].estimate == pytest.approx(positional[label].estimate)

    with pytest.raises(ValueError, match="multiple time="):
        survival.survfit("Surv(time=time, start=time, event=status) ~ group", data=data)
    with pytest.raises(ValueError, match="must not mix"):
        survival.survfit("Surv(time, event=status) ~ group", data=data)


def test_survfit_formula_splits_interval_weights_by_group():
    data = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0],
        "right": [1.0, 5.0, 3.0, 6.0, float("inf")],
        "status": [2, 3, 1, 3, 0],
        "arm": ["A", "A", "A", "B", "B"],
        "weights": [2.0, 1.0, 3.0, 4.0, 2.0],
    }

    grouped = survival.survfit(
        "Surv(left, right, status, type='interval') ~ arm",
        data=data,
        weights=data["weights"],
    )
    expected_a = survival.turnbull_estimator(
        [0.0, 2.0, 3.0],
        [1.0, 5.0, 3.0],
        weights=[2.0, 1.0, 3.0],
    )
    expected_b = survival.turnbull_estimator(
        [4.0, 5.0],
        [6.0, float("inf")],
        weights=[4.0, 2.0],
    )

    assert list(grouped) == ["A", "B"]
    assert grouped["A"].time_points == pytest.approx(expected_a.time_points)
    assert grouped["A"].survival == pytest.approx(expected_a.survival)
    assert grouped["B"].time_points == pytest.approx(expected_b.time_points)
    assert grouped["B"].survival == pytest.approx(expected_b.survival)

    grouped_with_model = survival.survfit(
        "Surv(left, right, status, type='interval') ~ arm",
        data=data,
        weights=data["weights"],
        model=True,
    )
    assert grouped_with_model["A"].survival == pytest.approx(expected_a.survival)
    assert grouped_with_model["B"].survival == pytest.approx(expected_b.survival)
    assert grouped_with_model["A"].model["(weights)"] == pytest.approx(data["weights"])
    assert grouped_with_model["A"].model is grouped_with_model["B"].model


def test_survfit_grouped_turnbull_preserves_first_seen_label_order():
    left = [0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 4.0, 3.0]
    right = [1.0, 3.0, float("inf"), 2.0, 2.0, 5.0, 4.0, float("inf")]
    groups = ["later", "first", "later", "first", "later", "first", "later", "first"]
    weights = [1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 2.5, 1.0]
    response = survival.Surv(left, right, type="interval2")

    grouped = survival.survfit(response, group=groups, weights=weights)

    assert list(grouped) == ["later", "first"]
    for label in grouped:
        indices = [idx for idx, value in enumerate(groups) if value == label]
        expected = survival.survfit(
            survival.Surv(
                [left[idx] for idx in indices],
                [right[idx] for idx in indices],
                type="interval2",
            ),
            weights=[weights[idx] for idx in indices],
        )
        assert isinstance(grouped[label], survival.r_api.TurnbullSurvfitResult)
        assert grouped[label].time_points == pytest.approx(expected.time_points)
        assert grouped[label].survival == pytest.approx(expected.survival)
        assert grouped[label].survival_lower == pytest.approx(expected.survival_lower)
        assert grouped[label].survival_upper == pytest.approx(expected.survival_upper)
        assert grouped[label].n_iter == expected.n_iter
        assert grouped[label].converged == expected.converged


def test_survfit_formula_accepts_named_interval2_response_arguments():
    data = {
        "left": [float("-inf"), 2.0, 3.0, 4.0],
        "right": [1.0, 5.0, 3.0, float("inf")],
        "arm": ["A", "A", "B", "B"],
    }

    named = survival.survfit("Surv(time=left, time2=right, type='interval2') ~ arm", data=data)
    positional = survival.survfit("Surv(left, right, type='interval2') ~ arm", data=data)

    assert list(named) == list(positional)
    for label in positional:
        assert named[label].time_points == pytest.approx(positional[label].time_points)
        assert named[label].survival == pytest.approx(positional[label].survival)


def test_survfit_interval_response_uses_turnbull_estimator():
    response = survival.Surv(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, 6.0],
        [2, 3, 1, 0],
        type="interval",
    )

    high_level = survival.survfit(response)
    low_level = survival.turnbull_estimator(
        [0.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, float("inf")],
    )

    assert response.type == "interval"
    assert response.status == (2, 3, 1, 0)
    assert high_level.time_points == pytest.approx(low_level.time_points)
    assert high_level.survival == pytest.approx(low_level.survival)

    with pytest.raises(ValueError, match="0/1/2/3 interval censoring codes"):
        survival.Surv([1.0, 2.0], [1.0, 3.0], [1.5, 0.0], type="interval")


def test_survfit_interval2_response_derives_censoring_codes():
    response = survival.Surv(
        [float("-inf"), 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, float("inf")],
        type="interval2",
    )

    high_level = survival.survfit(response)
    low_level = survival.turnbull_estimator(
        [0.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 3.0, float("inf")],
    )

    assert response.type == "interval2"
    assert response.status == (2, 3, 1, 0)
    assert high_level.time_points == pytest.approx(low_level.time_points)
    assert high_level.survival == pytest.approx(low_level.survival)


def test_survfit_formula_accepts_intercept_only_rhs():
    data = _toy_data()
    high_level = survival.survfit("Surv(time, status) ~ 1", data=data)
    low_level = survival.survfitkm(data["time"], data["status"])
    all_observed = survival.survfit("Surv(time) ~ 1", data=data)
    low_level_all_observed = survival.survfitkm(data["time"], [1] * len(data["time"]))
    shifted = survival.survfit("Surv(time, status, origin=0.5) ~ 1", data=data)
    low_level_shifted = survival.survfitkm(
        [time - 0.5 for time in data["time"]],
        data["status"],
    )

    assert high_level.time == pytest.approx(low_level.time)
    assert high_level.estimate == pytest.approx(low_level.estimate)
    assert all_observed.time == pytest.approx(low_level_all_observed.time)
    assert all_observed.estimate == pytest.approx(low_level_all_observed.estimate)
    assert shifted.time == pytest.approx(low_level_shifted.time)
    assert shifted.estimate == pytest.approx(low_level_shifted.estimate)

    with pytest.raises(ValueError, match="one-argument Surv"):
        survival.survfit("Surv(time, type='right') ~ 1", data=data)


def test_survfit_formula_response_accepts_event_comparisons():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 2, 1, 2],
        "event code": [2, 2, 2, 2],
        "event label": ["censored", "death", "censored", "death"],
        "group": ["A", "A", "B", "B"],
    }
    grouped = survival.survfit("Surv(time, status == 2) ~ group", data=data)
    direct = survival.survfit(
        survival.Surv(data["time"], [status == 2 for status in data["status"]]),
        group=data["group"],
    )
    reversed_comparison = survival.survfit("Surv(time, 2 == status) ~ group", data=data)
    identity_comparison = survival.survfit("Surv(time, I(status == 2)) ~ group", data=data)
    identity_operand = survival.survfit("Surv(time, identity(status) == 2) ~ group", data=data)
    string_status = survival.survfit("Surv(time, 'death' == `event label`) ~ 1", data=data)
    direct_string_status = survival.survfit(
        survival.Surv(data["time"], [label == "death" for label in data["event label"]])
    )
    column_comparison = survival.survfit(
        "Surv(time, status == `event code`) ~ .",
        data={key: data[key] for key in ("time", "status", "event code", "group")},
    )
    omitted = survival.survfit(
        "Surv(time, status == 2) ~ 1",
        data={**data, "status": [1, None, 1, 2]},
        na_action="omit",
    )
    direct_omitted = survival.survfit(survival.Surv([1.0, 3.0, 4.0], [False, False, True]))

    with pytest.raises(ValueError, match="event must use"):
        survival.survfit(
            "Surv(time, status == 2) ~ 1",
            data={**data, "status": [1, None, 1, 2]},
            na_action="pass",
        )

    assert list(grouped) == list(direct)
    for label in grouped:
        assert grouped[label].time == pytest.approx(direct[label].time)
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)
        assert reversed_comparison[label].time == pytest.approx(direct[label].time)
        assert reversed_comparison[label].estimate == pytest.approx(direct[label].estimate)
        assert identity_comparison[label].time == pytest.approx(direct[label].time)
        assert identity_comparison[label].estimate == pytest.approx(direct[label].estimate)
        assert identity_operand[label].time == pytest.approx(direct[label].time)
        assert identity_operand[label].estimate == pytest.approx(direct[label].estimate)
        assert column_comparison[label].time == pytest.approx(direct[label].time)
        assert column_comparison[label].estimate == pytest.approx(direct[label].estimate)
    assert string_status.time == pytest.approx(direct_string_status.time)
    assert string_status.estimate == pytest.approx(direct_string_status.estimate)
    assert omitted.time == pytest.approx(direct_omitted.time)
    assert omitted.estimate == pytest.approx(direct_omitted.estimate)


def test_survfit_formula_groups_curves_by_label():
    grouped = survival.survfit("Surv(time, status) ~ group", data=_toy_data())

    assert list(grouped) == ["A", "B"]
    assert grouped["A"].time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert grouped["B"].time == pytest.approx([5.0, 6.0, 7.0, 8.0])


def test_survfit_formula_start_time_filters_groups():
    data = _toy_data()
    grouped = survival.survfit("Surv(time, status) ~ group", data=data, start_time=4.0)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=data["group"],
        start_time=4.0,
    )

    assert list(grouped) == list(direct)
    for label in grouped:
        assert grouped[label].time == pytest.approx(direct[label].time)
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_formula_time0_groups_curves_by_label():
    data = _toy_data()
    grouped = survival.survfit("Surv(time, status) ~ group", data=data, time0=True)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=data["group"],
        time0=True,
    )

    assert list(grouped) == list(direct)
    for label in grouped:
        assert grouped[label].time == pytest.approx(direct[label].time)
        assert grouped[label].time[0] == pytest.approx(0.0)
        assert grouped[label].estimate[0] == pytest.approx(1.0)
        assert grouped[label].cumhaz[0] == pytest.approx(0.0)
        assert grouped[label].std_chaz[0] == pytest.approx(0.0)


def test_survfit0_adds_initial_row_to_grouped_curves():
    data = _toy_data()
    grouped = survival.survfit("Surv(time, status) ~ group", data=data)
    grouped_time0 = survival.survfit("Surv(time, status) ~ group", data=data, time0=True)

    inserted = survival.survfit0(grouped)

    assert list(inserted) == list(grouped_time0)
    for label in inserted:
        assert inserted[label].time == pytest.approx(grouped_time0[label].time)
        assert inserted[label].estimate == pytest.approx(grouped_time0[label].estimate)
        assert inserted[label].cumhaz == pytest.approx(grouped_time0[label].cumhaz)


def test_survfit_formula_reverse_groups_curves_by_label():
    data = _toy_data()
    grouped = survival.survfit("Surv(time, status) ~ group", data=data, reverse=True)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=data["group"],
        reverse=True,
    )

    assert list(grouped) == ["A", "B"]
    for label in grouped:
        assert grouped[label].time == pytest.approx(direct[label].time)
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_formula_fh_groups_curves_by_label():
    data = _toy_data()
    grouped = survival.survfit("Surv(time, status) ~ group", data=data, type="fh")
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=data["group"],
        type="fh",
    )

    assert list(grouped) == ["A", "B"]
    for label in grouped:
        assert grouped[label].time == pytest.approx(direct[label].time)
        assert grouped[label].cumhaz == pytest.approx(direct[label].cumhaz)
        assert grouped[label].std_chaz == pytest.approx(direct[label].std_chaz)
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_grouped_batch_matches_weighted_delayed_entry_curves():
    start = [0.0, 0.5, 1.0, 0.0, 1.5, 2.0, 2.5, 3.0]
    stop = [2.0, 3.0, 4.0, 2.5, 4.5, 5.0, 6.0, 7.0]
    status = [1, 0, 1, 0, 1, 1, 0, 1]
    groups = ["later", "first", "later", "first", "later", "first", "later", "first"]
    weights = [1.0, 0.75, 1.5, 1.25, 0.5, 2.0, 1.75, 0.8]
    response = survival.Surv(start, stop, status)

    grouped = survival.survfit(
        response,
        group=groups,
        weights=weights,
        type="fh2",
        conf_type="log-log",
        robust=False,
    )

    assert list(grouped) == ["later", "first"]
    for label in grouped:
        indices = [idx for idx, value in enumerate(groups) if value == label]
        expected = survival.survfit(
            survival.Surv(
                [start[idx] for idx in indices],
                [stop[idx] for idx in indices],
                [status[idx] for idx in indices],
            ),
            weights=[weights[idx] for idx in indices],
            type="fh2",
            conf_type="log-log",
            robust=False,
        )
        assert grouped[label].time == pytest.approx(expected.time)
        assert grouped[label].n_risk == pytest.approx(expected.n_risk)
        assert grouped[label].estimate == pytest.approx(expected.estimate)
        assert grouped[label].std_err == pytest.approx(expected.std_err)
        assert grouped[label].cumhaz == pytest.approx(expected.cumhaz)
        assert grouped[label].std_chaz == pytest.approx(expected.std_chaz)
        assert grouped[label].conf_lower == pytest.approx(expected.conf_lower)
        assert grouped[label].conf_upper == pytest.approx(expected.conf_upper)


def test_survfit_formula_accepts_backtick_column_names():
    grouped = survival.survfit(
        "Surv(`follow-up`, `event status`) ~ `treatment arm`",
        data=_backtick_data(),
    )

    assert list(grouped) == ["A", "B"]
    assert grouped["A"].time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert grouped["B"].time == pytest.approx([5.0, 6.0, 7.0, 8.0])


def test_survfit_formula_accepts_factor_wrapper_for_numeric_groups():
    data = _factor_data()
    grouped = survival.survfit("Surv(time, status) ~ factor(dose)", data=data)
    direct = survival.survfit(survival.Surv(data["time"], data["status"]), group=data["dose"])

    assert list(grouped) == list(direct)
    for label in grouped:
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_formula_groups_by_numeric_transform():
    data = _factor_data()
    grouped = survival.survfit("Surv(time, status) ~ sqrt(dose)", data=data)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=[math.sqrt(value) for value in data["dose"]],
    )

    assert list(grouped) == list(direct)
    for label in grouped:
        assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_formula_accepts_identity_wrappers_for_numeric_groups():
    data = _factor_data()
    for wrapper in ("I", "identity", "as.numeric"):
        grouped = survival.survfit(f"Surv(time, status) ~ {wrapper}(dose)", data=data)
        direct = survival.survfit(survival.Surv(data["time"], data["status"]), group=data["dose"])

        assert list(grouped) == list(direct)
        for label in grouped:
            assert grouped[label].estimate == pytest.approx(direct[label].estimate)


def test_survfit_formula_dot_groups_by_remaining_column():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", "control", "treated", "treated"],
    }
    grouped = survival.survfit("Surv(time, status) ~ .", data=data)

    assert list(grouped) == ["control", "treated"]
    assert grouped["control"].time == pytest.approx([1.0, 2.0])
    assert grouped["treated"].time == pytest.approx([3.0, 4.0])


def test_survfit_formula_dot_can_exclude_identifier_columns():
    data = {
        "id": [101, 102, 103, 104],
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", "control", "treated", "treated"],
    }
    direct = survival.survfit("Surv(time, status) ~ arm", data=data)
    expanded = survival.survfit("Surv(time, status) ~ . - id", data=data)

    assert list(expanded) == list(direct)
    assert expanded["control"].estimate == pytest.approx(direct["control"].estimate)
    assert expanded["treated"].estimate == pytest.approx(direct["treated"].estimate)


def test_survfit_formula_applies_subset_before_grouping():
    data = _toy_data()
    indices = [0, 1, 2, 3, 5, 6]
    fit = survival.survfit("Surv(time, status) ~ group", data=data, subset=indices)
    direct = survival.survfit("Surv(time, status) ~ group", data=_take(data, indices))

    assert list(fit) == list(direct)
    assert fit["A"].estimate == pytest.approx(direct["A"].estimate)
    assert fit["B"].estimate == pytest.approx(direct["B"].estimate)


def test_survfit_formula_na_action_omit_drops_missing_rows():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", None, "treated", "treated"],
    }
    fit = survival.survfit("Surv(time, status) ~ arm", data=data, na_action="omit")
    dotted = survival.survfit("Surv(time, status) ~ arm", data=data, **{"na.action": "omit"})
    direct = survival.survfit(
        "Surv(time, status) ~ arm",
        data={
            "time": [1.0, 3.0, 4.0],
            "status": [1, 1, 1],
            "arm": ["control", "treated", "treated"],
        },
    )

    assert list(fit) == list(direct)
    assert list(dotted) == list(direct)
    assert fit["control"].estimate == pytest.approx(direct["control"].estimate)
    assert fit["treated"].estimate == pytest.approx(direct["treated"].estimate)
    assert dotted["control"].estimate == pytest.approx(direct["control"].estimate)
    assert dotted["treated"].estimate == pytest.approx(direct["treated"].estimate)


def test_survfit_formula_filters_external_weights_with_subset_and_na_action():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 0, 1, 1, 0],
        "arm": ["control", "control", "treated", "treated", "control"],
    }
    fit = survival.survfit(
        "Surv(time, status) ~ arm",
        data=data,
        weights=[1.0, None, 2.0, 1.5, 3.0],
        subset=[0, 1, 2, 3],
        na_action="omit",
    )
    direct = survival.survfit(
        survival.Surv([1.0, 3.0, 4.0], [1, 1, 1]),
        group=["control", "treated", "treated"],
        weights=[1.0, 2.0, 1.5],
    )

    assert list(fit) == list(direct)
    assert fit["control"].estimate == pytest.approx(direct["control"].estimate)
    assert fit["treated"].estimate == pytest.approx(direct["treated"].estimate)


def test_survfit_direct_na_action_omit_filters_response_and_group():
    response = survival.Surv([1.0, float("nan"), 3.0, 4.0], [1, 0, 1, 1])
    fit = survival.survfit(response, group=["A", "A", "B", "B"], na_action="omit")
    direct = survival.survfit(survival.Surv([1.0, 3.0, 4.0], [1, 1, 1]), group=["A", "B", "B"])

    assert list(fit) == list(direct)
    assert fit["A"].estimate == pytest.approx(direct["A"].estimate)
    assert fit["B"].estimate == pytest.approx(direct["B"].estimate)


def test_survfit_formula_accepts_strata_wrapper():
    grouped = survival.survfit("Surv(time, status) ~ strata(group)", data=_toy_data())

    assert list(grouped) == ["A", "B"]
    assert grouped["A"].estimate == pytest.approx([0.75, 0.5, 0.5, 0.0])
    assert grouped["B"].estimate == pytest.approx([1.0, 2 / 3, 1 / 3, 1 / 3])


def test_survfit_formula_accepts_interaction_groups():
    data = _factor_data()
    interaction = survival.survfit("Surv(time, status) ~ factor(dose):sqrt(x1)", data=data)
    direct = survival.survfit(
        survival.Surv(data["time"], data["status"]),
        group=[(data["dose"][idx], math.sqrt(data["x1"][idx])) for idx in range(len(data["time"]))],
    )
    expected_order = [
        (0, math.sqrt(0.2)),
        (0, math.sqrt(0.4)),
        (0, math.sqrt(0.6)),
        (1, math.sqrt(0.1)),
        (1, math.sqrt(0.8)),
        (1, math.sqrt(1.4)),
        (2, math.sqrt(1.0)),
        (2, math.sqrt(1.2)),
    ]

    assert list(interaction) == expected_order
    for key in direct:
        assert interaction[key].estimate == pytest.approx(direct[key].estimate)


def test_survfit_accepts_simple_coxph_model():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    result = survival.survfit(fit, newdata=[[0.5, 0.8]])
    event_only = survival.survfit(fit, newdata=[[0.5, 0.8]], censor=False)
    no_conf = survival.survfit(fit, newdata=[[0.5, 0.8]], conf_type="none")
    no_se = survival.survfit(fit, newdata=[[0.5, 0.8]], se_fit=False)
    with_model = survival.survfit(fit, newdata=[[0.5, 0.8]], model=True)
    plain = survival.survfit(fit, newdata=[[0.5, 0.8]], conf_type="plain")
    direct_times, direct_curves = fit.survival_curve([[0.5, 0.8]])
    times, curves = result
    expected_prediction = survival.predict(
        fit,
        {
            "time": result.time,
            "status": [0] * len(result.time),
            "x1": [0.5] * len(result.time),
            "x2": [0.8] * len(result.time),
        },
        type="expected",
        se_fit=True,
    )

    assert isinstance(result, survival.r_api.CoxSurvfitResult)
    assert result.time == pytest.approx(times)
    assert result.time == pytest.approx(data["time"])
    assert event_only.time == pytest.approx(direct_times)
    assert event_only.surv[0] == pytest.approx(direct_curves[0])
    assert event_only.cumhaz[0] == pytest.approx(
        [
            hazard
            for hazard, status in zip(result.cumhaz[0], data["status"], strict=True)
            if status == 1
        ]
    )
    assert result.surv[0] == pytest.approx(curves[0])
    assert no_conf.time == pytest.approx(result.time)
    assert no_conf.surv[0] == pytest.approx(result.surv[0])
    assert no_conf.cumhaz[0] == pytest.approx(result.cumhaz[0])
    assert no_conf.conf_lower == []
    assert no_conf.conf_upper == []
    assert no_se.time == pytest.approx(result.time)
    assert no_se.surv[0] == pytest.approx(result.surv[0])
    assert no_se.cumhaz[0] == pytest.approx(result.cumhaz[0])
    assert no_se.std_err == []
    assert no_se.std_chaz == []
    assert no_se.conf_lower == []
    assert no_se.conf_upper == []
    assert no_se.strata is None
    no_se_frame = survival.as_data_frame(no_se)
    assert "strata" not in no_se_frame
    assert with_model.time == pytest.approx(result.time)
    assert with_model.surv[0] == pytest.approx(result.surv[0])
    assert with_model.model["fit"] is fit
    assert with_model.model["newdata"] == [[0.5, 0.8]]
    assert result.curves[0] == pytest.approx(curves[0])
    assert result.estimate[0] == pytest.approx(curves[0])
    assert result.cumulative_hazard[0] == pytest.approx(result.cumhaz[0])
    assert result.cumulative_hazard_std_err[0] == pytest.approx(expected_prediction.se_fit)
    assert result.cumhaz[0] == pytest.approx(expected_prediction.fit)
    assert result.std_chaz[0] == pytest.approx(expected_prediction.se_fit)
    assert result.std_err[0] == pytest.approx(
        [surv * se for surv, se in zip(result.surv[0], expected_prediction.se_fit, strict=True)]
    )
    assert len(result.conf_lower) == 1
    assert len(result.conf_upper) == 1
    assert len(result.conf_lower[0]) == len(result.time)
    assert len(result.conf_upper[0]) == len(result.time)
    expected_plain_first = _plain_confidence_interval(
        plain.surv[0][0],
        plain.std_err[0][0],
    )
    assert plain.conf_lower[0][0] == pytest.approx(expected_plain_first[0])
    assert plain.conf_upper[0][0] == pytest.approx(expected_plain_first[1])
    assert len(times) > 0
    assert len(curves) == 1
    assert all(0.0 <= value <= 1.0 for value in curves[0])


def test_survfit_coxph_start_time_conditions_curve():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    full = survival.survfit(fit, newdata=[[0.5, 0.8]])
    conditioned = survival.survfit(fit, newdata=[[0.5, 0.8]], start_time=4.5)
    with_time0 = survival.survfit(
        fit,
        newdata=[[0.5, 0.8]],
        start_time=4.5,
        time0=True,
    )
    at_event = survival.survfit(
        fit,
        newdata=[[0.5, 0.8]],
        start_time=4.0,
        time0=True,
    )

    start_pos = sum(1 for time in full.time if time < 4.5)
    start_hazard = full.cumhaz[0][start_pos - 1]
    expected_cumhaz = [value - start_hazard for value in full.cumhaz[0][start_pos:]]
    expected_surv = [math.exp(-value) for value in expected_cumhaz]

    assert conditioned.start_time == pytest.approx(4.5)
    assert conditioned.time == pytest.approx(full.time[start_pos:])
    assert conditioned.cumhaz[0] == pytest.approx(expected_cumhaz)
    assert conditioned.surv[0] == pytest.approx(expected_surv)
    assert with_time0.time == pytest.approx([4.5, *conditioned.time])
    assert with_time0.cumhaz[0] == pytest.approx([0.0, *conditioned.cumhaz[0]])
    assert with_time0.surv[0] == pytest.approx([1.0, *conditioned.surv[0]])
    assert at_event.time[0] == pytest.approx(4.0)
    assert at_event.surv[0][0] != pytest.approx(1.0)


def test_survfit0_adds_initial_row_to_existing_cox_curves():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    base = survival.survfit(fit, newdata=[[0.5, 0.8]])
    direct_time0 = survival.survfit(fit, newdata=[[0.5, 0.8]], time0=True)
    conditioned = survival.survfit(fit, newdata=[[0.5, 0.8]], start_time=4.5)
    conditioned_time0 = survival.survfit(
        fit,
        newdata=[[0.5, 0.8]],
        start_time=4.5,
        time0=True,
    )

    inserted = survival.survfit0(base)
    inserted_conditioned = survival.survfit0(conditioned)
    already_inserted = survival.survfit0(direct_time0)

    assert inserted.time == pytest.approx(direct_time0.time)
    assert inserted.surv[0] == pytest.approx(direct_time0.surv[0])
    assert inserted.cumhaz[0] == pytest.approx(direct_time0.cumhaz[0])
    assert inserted.std_err[0] == pytest.approx(direct_time0.std_err[0])
    assert inserted.std_chaz[0] == pytest.approx(direct_time0.std_chaz[0])
    assert inserted.conf_lower[0] == pytest.approx(direct_time0.conf_lower[0])
    assert inserted.conf_upper[0] == pytest.approx(direct_time0.conf_upper[0])
    assert inserted_conditioned.time == pytest.approx(conditioned_time0.time)
    assert inserted_conditioned.surv[0] == pytest.approx(conditioned_time0.surv[0])
    assert inserted_conditioned.cumhaz[0] == pytest.approx(conditioned_time0.cumhaz[0])
    assert already_inserted is direct_time0


def test_survfit_coxph_start_time_conditions_stratified_curves():
    data = {
        "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["A", "A", "A", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 1.0, 1.2, 0.8],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )

    conditioned = survival.survfit(fit, start_time=2.5, time0=True)

    assert conditioned.start_time == pytest.approx(2.5)
    assert conditioned.time == pytest.approx([2.5, 3.0, 4.0])
    assert conditioned.strata == [0, 1]
    assert conditioned.cumhaz[0] == pytest.approx([0.0, 0.0, 0.0])
    assert conditioned.cumhaz[1] == pytest.approx([0.0, 0.5, 1.5])
    assert conditioned.surv[0] == pytest.approx([1.0, 1.0, 1.0])
    assert conditioned.surv[1] == pytest.approx([1.0, math.exp(-0.5), math.exp(-1.5)])


def test_survfit_coxph_formula_accepts_newdata_mapping():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    newdata = {"x1": [0.5], "x2": [0.8]}

    result = survival.survfit(fit, newdata=newdata, censor=False)
    times, curves = result
    direct_times, direct_curves = fit.survival_curve([[0.5, 0.8]])

    assert times == pytest.approx(direct_times)
    assert result.time == pytest.approx(direct_times)
    assert curves[0] == pytest.approx(direct_curves[0])
    assert result.surv[0] == pytest.approx(direct_curves[0])


def test_survfit_coxph_formula_offset_accepts_newdata_mapping():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "offset": offsets}
    linear_predictors = [
        value + offset for value, offset in zip(fit.predict(rows), offsets, strict=True)
    ]
    center = sum(fit.linear_predictors) / len(fit.linear_predictors)
    baseline_times, hazards = fit.basehaz(True)
    expected_curves = [
        [math.exp(-hazard * math.exp(lp - center)) for hazard in hazards]
        for lp in linear_predictors
    ]

    result = survival.survfit(fit, newdata=newdata, censor=False)
    times, curves = result

    assert times == pytest.approx(baseline_times)
    for actual, expected in zip(curves, expected_curves, strict=True):
        assert actual == pytest.approx(expected)
    for actual, linear_predictor in zip(result.cumhaz, linear_predictors, strict=True):
        expected_hazard = [hazard * math.exp(linear_predictor - center) for hazard in hazards]
        assert actual == pytest.approx(expected_hazard)


def test_survfit_coxph_formula_offset_uses_stratified_baseline_steps():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset) + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "offset": offsets, "group": ["A", "B"]}
    linear_predictors = [
        value + offset for value, offset in zip(fit.predict(rows), offsets, strict=True)
    ]
    center = sum(fit.linear_predictors) / len(fit.linear_predictors)
    base_times, base_hazards, base_strata = fit.basehaz_with_strata(True)
    expected_times = sorted(set(base_times))

    result = survival.survfit(fit, newdata=newdata, censor=False, se_fit=False)
    default_result = survival.survfit(fit, censor=False, se_fit=False)

    assert result.time == pytest.approx(expected_times)
    assert result.strata == [0, 1]
    assert result.strata_labels == [1, 2]
    result_frame = survival.as_data_frame(result)
    assert set(result_frame["strata"]) == {1, 2}
    assert default_result.strata == [0, 1]
    assert default_result.strata_labels == ["A", "B"]
    default_frame = survival.as_data_frame(default_result)
    assert set(default_frame["strata"]) == {"A", "B"}
    for curve_idx, stratum in enumerate([0, 1]):
        stratum_times = [
            time for time, label in zip(base_times, base_strata, strict=True) if label == stratum
        ]
        stratum_hazards = [
            hazard
            for hazard, label in zip(base_hazards, base_strata, strict=True)
            if label == stratum
        ]
        risk = math.exp(linear_predictors[curve_idx] - center)
        expected_hazard = [
            (0.0 if (pos := bisect_right(stratum_times, time)) == 0 else stratum_hazards[pos - 1])
            * risk
            for time in expected_times
        ]
        assert result.cumhaz[curve_idx] == pytest.approx(expected_hazard)
        assert result.surv[curve_idx] == pytest.approx(
            [math.exp(-hazard) for hazard in expected_hazard]
        )


def test_survfit_accepts_optimizer_coxph_fit():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.5], "group": ["B"]}
    times, curves = survival.survfit(fit, newdata=newdata)
    event_times, event_curves = survival.survfit(fit, newdata=newdata, censor=False)
    direct_times, direct_curves = fit.survival_curve_with_strata([[0.5]], [1])
    hazard_times, hazards, hazard_strata = fit.basehaz_with_strata()

    assert times == pytest.approx([5.0, 6.0, 7.0, 8.0])
    assert event_times == pytest.approx(direct_times)
    assert event_curves[0] == pytest.approx(direct_curves[0])
    assert curves[0][0] == pytest.approx(1.0)
    assert curves[0][1:3] == pytest.approx(direct_curves[0])
    assert curves[0][3] == pytest.approx(direct_curves[0][-1])
    assert set(event_times) <= set(hazard_times)
    for stratum in set(hazard_strata):
        stratum_hazards = [
            hazard for hazard, label in zip(hazards, hazard_strata, strict=True) if label == stratum
        ]
        assert all(
            later >= earlier
            for earlier, later in zip(stratum_hazards[:-1], stratum_hazards[1:], strict=True)
        )
    assert all(0.0 <= value <= 1.0 for value in curves[0])
    with pytest.raises(ValueError, match="newdata strata are required"):
        survival.survfit(fit, newdata=[[0.5]])


def test_survfit_coxph_stratified_default_returns_one_curve_per_stratum():
    data = {
        "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["A", "A", "A", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 1.0, 1.2, 0.8],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    result = survival.survfit(fit)
    direct_times, direct_curves = fit.survival_curve_with_strata([fit.means, fit.means], [0, 1])

    assert result.time == pytest.approx(direct_times)
    assert result.strata == [0, 1]
    assert len(result.surv) == 2
    assert result.linear_predictors == pytest.approx([0.0, 0.0])
    for actual, expected in zip(result.surv, direct_curves, strict=True):
        assert actual == pytest.approx(expected)
        assert all(later <= earlier for earlier, later in zip(actual[:-1], actual[1:], strict=True))
    assert result.cumhaz[0] == pytest.approx([1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0])
    assert result.cumhaz[1] == pytest.approx([0.0, 0.0, 0.5, 1.5])


def test_survfit_optimizer_coxph_defaults_to_fitted_means():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, eps=1e-5)
    result = survival.survfit(fit)
    event_only = survival.survfit(fit, censor=False)
    times, curves = result
    direct_times, direct_curves = fit.survival_curve()
    expected_lp = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )

    assert times == pytest.approx(data["time"])
    assert result.time == pytest.approx(data["time"])
    assert event_only.time == pytest.approx(direct_times)
    assert event_only.surv[0] == pytest.approx(direct_curves[0])
    assert len(curves) == 1
    assert [
        value for value, status in zip(curves[0], data["status"], strict=True) if status == 1
    ] == pytest.approx(direct_curves[0])
    assert result.linear_predictors == pytest.approx([expected_lp])
    assert result.surv[0] == pytest.approx(curves[0])
