import importlib
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r_pyears = importlib.import_module("survival.r._pyears")


def test_r_style_ratetable_helpers_delegate_to_population_core():
    table = survival.survexp_us()
    rural = survival.survexp_usr()
    mn = survival.survexp_mn()

    assert isinstance(table, survival.RateTable)
    assert survival.is_ratetable(table) is True
    assert survival.is_ratetable(table.ndim(), True, True) is True
    assert survival.is_ratetable(object()) is False
    assert table.ndim() == 3
    assert mn.ndim() == 3
    assert rural.summary == table.summary

    scaled = survival.survexp(
        time=[365.25, 730.5],
        age=[18262.5, 21915.0],
        year=[2000.0, 2000.0],
        sex=[0, 1],
        times=[365.25, 730.5],
        method="ederer",
        scale=365.25,
    )
    conditional = survival.survexp(
        time=[365.25, 730.5],
        age=[18262.5, 21915.0],
        year=[2000.0, 2000.0],
        sex=[0, 1],
        conditional=True,
    )
    direct_individual = survival.survexp_individual(
        time=[365.25, 730.5],
        age=[18262.5, 21915.0],
        year=[2000.0, 2000.0],
        sex=[0, 1],
    )
    individual_surv = survival.survexp(
        time=[365.25, 730.5],
        age=[18262.5, 21915.0],
        year=[2000.0, 2000.0],
        sex=[0, 1],
        cohort=False,
    )
    individual_hazard = survival.survexp(
        time=[365.25, 730.5],
        age=[18262.5, 21915.0],
        year=[2000.0, 2000.0],
        sex=[0, 1],
        method="individual.h",
    )

    assert isinstance(scaled, survival.SurvExpResult)
    assert scaled.time == pytest.approx([1.0, 2.0])
    assert scaled.method == "ederer"
    assert conditional.method == "conditional"
    assert individual_surv == pytest.approx(direct_individual)
    assert individual_hazard == pytest.approx([-math.log(value) for value in direct_individual])

    with pytest.warns(RuntimeWarning, match="se_fit value ignored"):
        survival.survexp(
            time=[365.25],
            age=[18262.5],
            year=[2000.0],
            se_fit=True,
        )
    with pytest.raises(TypeError, match="ratetable"):
        survival.survexp([365.25], [18262.5], [2000.0], ratetable=object())
    with pytest.raises(ValueError, match="method must"):
        survival.survexp([365.25], [18262.5], [2000.0], method="fancy")
    with pytest.raises(ValueError, match="scale must be positive"):
        survival.survexp([365.25], [18262.5], [2000.0], scale=0)

    assert survival.ratetableDate(2000, 2, 29) == pytest.approx(11016.0)
    assert survival.ratetableDate("2000-02-29") == pytest.approx(11016.0)
    assert survival.ratetableDate("1940-01-01") == pytest.approx(-10958.0)
    assert survival.ratetableDate(["1940-01-01", "2000-02-29", "2001-01-01"]) == pytest.approx(
        [-10958.0, 11016.0, 11323.0]
    )
    assert survival.ratetableDate(11016.0) == pytest.approx(11016.0)
    with pytest.raises(ValueError, match="day is invalid"):
        survival.ratetableDate(2001, 2, 29)
    with pytest.raises(TypeError, match="month and day"):
        survival.ratetableDate(2000, month=2)


def test_survexp_ederer_keeps_the_full_reference_cohort():
    ages = [14610.0, 25567.5]
    years = [2000.0, 2000.0]
    sexes = [0, 1]
    eval_times = [365.25, 730.5]
    follow_up = [180.0, 1095.0]

    ederer = survival.survexp(
        follow_up,
        ages,
        years,
        sex=sexes,
        times=eval_times,
        method="ederer",
    )
    hakulinen = survival.survexp(
        follow_up,
        ages,
        years,
        sex=sexes,
        times=eval_times,
        method="hakulinen",
    )

    assert isinstance(ederer, survival.SurvExpResult)
    assert isinstance(hakulinen, survival.SurvExpResult)
    assert ederer.n_risk == pytest.approx([2.0, 2.0])
    assert hakulinen.n_risk == pytest.approx([1.0, 1.0])
    expected_ederer = [
        sum(
            survival.survexp_individual(
                time=[eval_time, eval_time],
                age=ages,
                year=years,
                sex=sexes,
            )
        )
        / 2.0
        for eval_time in eval_times
    ]
    assert ederer.surv == pytest.approx(expected_ederer)


def test_r_style_pyears_tabulates_surv_inputs():
    data = {
        "time": [10.0, 20.0, 30.0],
        "status": [1, 0, 1],
        "group": ["a", "a", "b"],
    }
    formula_result = survival.pyears("Surv(time, status) ~ group", data=data, scale=1)
    direct_result = survival.pyears(
        survival.Surv(data["time"], data["status"]),
        group=data["group"],
        weights=[2.0, 1.0, 1.0],
        scale=1,
    )
    counting_result = survival.pyears(
        survival.Surv([0.0, 5.0], [10.0, 15.0], [1, 0]),
        scale=1,
    )
    no_event_result = survival.pyears([10.0, 20.0, 30.0], group=data["group"], scale=1)
    tcut_result = survival.pyears(
        survival.Surv([25.0, 8.0], [1, 0]),
        group=survival.tcut([0.0, 5.0], [0.0, 10.0, 20.0, 30.0]),
        scale=1,
    )
    tcut_subset = survival.pyears(
        survival.Surv([25.0, 8.0], [1, 0]),
        group=survival.tcut([0.0, 5.0], [0.0, 10.0, 20.0, 30.0]),
        subset=[True, False],
        scale=1,
    )
    frame = survival.as_data_frame(formula_result)

    assert isinstance(formula_result, survival.PyearsResult)
    assert formula_result.group == ["a", "b"]
    assert formula_result.pyears == pytest.approx([30.0, 30.0])
    assert formula_result.n == pytest.approx([2.0, 1.0])
    assert formula_result.event == pytest.approx([1.0, 1.0])
    assert formula_result.observations == 3
    assert direct_result.pyears == pytest.approx([40.0, 30.0])
    assert direct_result.event == pytest.approx([2.0, 1.0])
    assert counting_result.pyears == pytest.approx([20.0])
    assert counting_result.event == pytest.approx([1.0])
    assert no_event_result.event is None
    assert no_event_result.pyears == pytest.approx([30.0, 30.0])
    assert tcut_result.group == ["0+ thru 10", "10+ thru 20", "20+ thru 30"]
    assert tcut_result.pyears == pytest.approx([15.0, 13.0, 5.0])
    assert tcut_result.n == pytest.approx([2.0, 2.0, 1.0])
    assert tcut_result.event == pytest.approx([0.0, 0.0, 1.0])
    assert tcut_result.tcut is True
    assert tcut_subset.pyears == pytest.approx([10.0, 10.0, 5.0])
    assert frame == {
        "group": ["a", "b"],
        "pyears": pytest.approx([30.0, 30.0]),
        "n": pytest.approx([2.0, 1.0]),
        "event": pytest.approx([1.0, 1.0]),
    }
    assert survival.pyears(
        "Surv(time, status) ~ group",
        data=data,
        subset=[True, False, True],
        scale=1,
    ).pyears == pytest.approx([10.0, 30.0])

    order_data = {
        "time": [10.0, 20.0, 30.0, 40.0],
        "status": [1, 0, 1, 1],
        "group": ["treated", "treated", "control", "control"],
        "id": [1, 2, 3, 4],
        "off": [0.1, 0.2, 0.3, 0.4],
    }
    ordered_formula = survival.pyears("Surv(time, status) ~ group", data=order_data, scale=1)
    offset_only = survival.pyears("Surv(time, status) ~ offset(off)", data=order_data, scale=1)
    offset_group = survival.pyears(
        "Surv(time, status) ~ group + offset(off)",
        data=order_data,
        scale=1,
    )
    cluster_group = survival.pyears(
        "Surv(time, status) ~ group + cluster(id)",
        data=order_data,
        scale=1,
    )
    ordered_direct = survival.pyears(
        survival.Surv(order_data["time"], order_data["status"]),
        group=order_data["group"],
        scale=1,
    )
    assert ordered_formula.group == ["control", "treated"]
    assert ordered_formula.pyears == pytest.approx([70.0, 30.0])
    assert ordered_formula.event == pytest.approx([2.0, 1.0])
    assert offset_only.group == ["(all)"]
    assert offset_only.pyears == pytest.approx([100.0])
    assert offset_only.event == pytest.approx([3.0])
    assert offset_group.group == ordered_formula.group
    assert offset_group.pyears == pytest.approx(ordered_formula.pyears)
    assert offset_group.event == pytest.approx(ordered_formula.event)
    assert cluster_group.pyears == pytest.approx([0.0, 10.0, 0.0, 20.0, 30.0, 0.0, 40.0, 0.0])
    assert cluster_group.n == pytest.approx([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    assert cluster_group.event == pytest.approx([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert ordered_direct.group == ["treated", "control"]
    assert ordered_direct.pyears == pytest.approx([30.0, 70.0])
    with pytest.raises(ValueError, match="interaction"):
        survival.pyears("Surv(time, status) ~ group:status", data=data, scale=1)
    with pytest.raises(ValueError, match="same length"):
        survival.pyears([1.0, 2.0], event=[1], scale=1)
    with pytest.raises(ValueError, match="scale must be positive"):
        survival.pyears([1.0], scale=0)


def test_pyears_normalizes_direct_inputs_once_without_subsetting(monkeypatch):
    original = r_pyears._pyears_response_from_direct
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(r_pyears, "_pyears_response_from_direct", tracked)

    result = survival.pyears(
        [10.0, 20.0, 30.0],
        event=[1, 0, 1],
        group=["a", "b", "a"],
        scale=1,
    )

    assert result.pyears == pytest.approx([40.0, 20.0])
    assert calls == 1
