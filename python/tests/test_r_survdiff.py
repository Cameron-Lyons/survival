import math

import pytest

from .helpers import setup_survival_import
from .r_api_support import _backtick_data, _factor_data, _take, _toy_data

survival = setup_survival_import()


def test_survdiff_formula_uses_logrank_binding():
    result = survival.survdiff("Surv(time, status) ~ group", data=_toy_data())

    assert result.df == 1
    assert len(result.observed) == 2
    assert result.weight_type == "LogRank"


def test_logrank_binding_validates_status_and_groups_near_ties():
    exact = survival.logrank_test([1.0, 1.0, 2.0, 3.0], [1, 1, 0, 1], [0, 1, 0, 1])
    near = survival.logrank_test(
        [1.0, 1.0 + 1e-13, 2.0, 3.0],
        [1, 1, 0, 1],
        [0, 1, 0, 1],
    )

    assert near.observed == pytest.approx(exact.observed)
    assert near.expected == pytest.approx(exact.expected)
    assert near.variance == pytest.approx(exact.variance)
    assert near.statistic == pytest.approx(exact.statistic)
    assert near.p_value == pytest.approx(exact.p_value)

    weighted = survival.logrank_test(
        [1.0, 1.0, 2.0, 3.0],
        [1, 1, 0, 1],
        [0, 1, 0, 1],
        weight_type="Tarone_Ware",
    )
    assert weighted.weight_type == "TaroneWare"

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.logrank_test([1.0, 2.0], [1, 2], [0, 1])

    with pytest.raises(ValueError, match="weight_type must be one of"):
        survival.logrank_test([1.0, 2.0], [1, 0], [0, 1], weight_type="bogus")

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.logrank_trend([1.0, 2.0], [1, 2], [0, 1])


def test_logrank_multigroup_matches_r_survdiff_chisquare():
    data = {
        "time": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 5.0, 7.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 1, 0],
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    }
    group_codes = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    direct = survival.logrank_test(data["time"], data["status"], group_codes)
    formula = survival.survdiff("Surv(time, status) ~ group", data=data)

    for result in (direct, formula):
        assert result.df == 2
        assert result.observed == pytest.approx([2.0, 2.0, 2.0])
        assert result.expected == pytest.approx([1.0, 2.25, 2.75])
        assert result.statistic == pytest.approx(1.5105257668985863)
        assert result.p_value == pytest.approx(0.4698870729581883)


def test_survdiff_formula_accepts_general_rho():
    data = _toy_data()
    result = survival.survdiff("Surv(time, status) ~ group", data=data, rho=0.5)
    low_level = survival.fleming_harrington_test(
        data["time"],
        data["status"],
        [0 if value == "A" else 1 for value in data["group"]],
        0.5,
        0.0,
    )

    assert result.weight_type == "FlemingHarrington(p=0.5, q=0)"
    assert result.statistic == pytest.approx(low_level.statistic)
    assert result.p_value == pytest.approx(low_level.p_value)
    assert result.observed == pytest.approx(low_level.observed)
    assert result.expected == pytest.approx(low_level.expected)


def test_survdiff_formula_accepts_offset_only_model():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "expected": [0.9, 0.8, 0.7, 0.6],
    }

    def manual(rho=0.0):
        observed = float(sum(data["status"]))
        expected = sum(-math.log(value) for value in data["expected"])
        if rho == 0.0:
            variance = expected
            numerator = observed - variance
        else:
            inverse_rho = 1.0 / rho
            numerator = sum(
                inverse_rho - ((inverse_rho + event) * (offset**rho))
                for offset, event in zip(data["expected"], data["status"], strict=True)
            )
            variance = sum(
                (1.0 - offset ** (2.0 * rho)) / (2.0 * rho) for offset in data["expected"]
            )
        statistic = numerator * numerator / variance
        return observed, expected, variance, statistic, math.erfc(math.sqrt(statistic / 2.0))

    result = survival.survdiff("Surv(time, status) ~ offset(expected)", data=data)
    weighted = survival.survdiff("Surv(time, status) ~ offset(expected)", data=data, rho=0.5)
    observed, expected, variance, statistic, p_value = manual()
    weighted_observed, weighted_expected, weighted_variance, weighted_statistic, weighted_p = (
        manual(0.5)
    )

    assert result.observed == pytest.approx([observed])
    assert result.expected == pytest.approx([expected])
    assert result.variance == pytest.approx(variance)
    assert result.statistic == pytest.approx(statistic)
    assert result.p_value == pytest.approx(p_value)
    assert weighted.observed == pytest.approx([weighted_observed])
    assert weighted.expected == pytest.approx([weighted_expected])
    assert weighted.variance == pytest.approx(weighted_variance)
    assert weighted.statistic == pytest.approx(weighted_statistic)
    assert weighted.p_value == pytest.approx(weighted_p)
    assert survival.as_data_frame(result)["observed"] == pytest.approx([observed])
    with pytest.raises(ValueError, match="Cannot have both an offset and groups"):
        survival.survdiff(
            "Surv(time, status) ~ group + offset(expected)",
            data={**data, "group": ["a"] * 4},
        )
    with pytest.raises(ValueError, match="survival probability"):
        survival.survdiff(
            "Surv(time, status) ~ offset(expected)",
            data={**data, "expected": [1.1, 0.8, 0.7, 0.6]},
        )


def test_survdiff_counting_process_uses_delayed_entry():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0],
        "stop": [2.0, 4.0, 3.0, 5.0],
        "status": [1, 0, 1, 1],
        "group": ["treated", "control", "treated", "control"],
    }
    formula_groups = [1 if value == "treated" else 0 for value in data["group"]]

    direct = survival.survdiff(
        survival.Surv(data["start"], data["stop"], data["status"]),
        group=data["group"],
    )
    formula = survival.survdiff("Surv(start, stop, status) ~ group", data=data)
    low_level = survival.logrank_test(
        data["stop"],
        data["status"],
        formula_groups,
        entry_times=data["start"],
    )
    fh = survival.survdiff("Surv(start, stop, status) ~ group", data=data, rho=0.5)
    fh_low_level = survival.fleming_harrington_test(
        data["stop"],
        data["status"],
        formula_groups,
        0.5,
        0.0,
        entry_times=data["start"],
    )

    assert direct.statistic == pytest.approx(2.25)
    assert direct.observed == pytest.approx([2.0, 1.0])
    assert direct.expected == pytest.approx([1.0, 2.0])
    assert direct.variance == pytest.approx(4.0 / 9.0)
    assert formula.statistic == pytest.approx(direct.statistic)
    assert formula.observed == pytest.approx(low_level.observed)
    assert formula.expected == pytest.approx(low_level.expected)
    assert low_level.statistic == pytest.approx(direct.statistic)
    assert fh.statistic == pytest.approx(fh_low_level.statistic)


def test_survdiff_counting_process_timefix_false_uses_exact_event_times():
    start = [0.0, 0.0, 0.0, 0.0]
    stop = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [1, 1, 0, 1]
    group = ["control", "treated", "control", "treated"]
    group_codes = [0, 1, 0, 1]
    response = survival.Surv(start, stop, status)

    def manual_exact(rho=0.0):
        observed = [0.0, 0.0]
        expected = [0.0, 0.0]
        variance = 0.0
        km_survival = 1.0
        for event_time in sorted({time for time, event in zip(stop, status, strict=True) if event}):
            at_risk = [0.0, 0.0]
            events = [0.0, 0.0]
            total_events = 0.0
            for row_idx, stop_time in enumerate(stop):
                group_idx = group_codes[row_idx]
                if start[row_idx] < event_time <= stop_time:
                    at_risk[group_idx] += 1.0
                if status[row_idx] == 1 and stop_time == event_time:
                    events[group_idx] += 1.0
                    total_events += 1.0
            total_at_risk = sum(at_risk)
            weight = km_survival**rho
            for group_idx in range(2):
                observed[group_idx] += weight * events[group_idx]
                expected[group_idx] += weight * total_events * at_risk[group_idx] / total_at_risk
            if total_at_risk > 1.0:
                variance += (
                    weight
                    * weight
                    * total_events
                    * (total_at_risk - total_events)
                    / (total_at_risk * total_at_risk * (total_at_risk - 1.0))
                    * at_risk[0]
                    * at_risk[1]
                )
            km_survival *= 1.0 - total_events / total_at_risk
        statistic = (observed[0] - expected[0]) ** 2 / variance
        return observed, expected, variance, statistic

    default = survival.survdiff(response, group=group)
    exact = survival.survdiff(response, group=group, timefix=False)
    exact_dotted = survival.survdiff(response, group=group, **{"time.fix": False})
    fh_exact = survival.survdiff(response, group=group, rho=0.5, timefix=False)
    low_level = survival.surv_analysis.compute_counting_logrank_components(
        stop,
        status,
        [code + 1 for code in group_codes],
        start,
        None,
        0.0,
        False,
    )
    observed, expected, variance, statistic = manual_exact()
    fh_observed, fh_expected, fh_variance, fh_statistic = manual_exact(0.5)

    assert exact.observed == pytest.approx(observed)
    assert exact.expected == pytest.approx(expected)
    assert exact.variance == pytest.approx(variance)
    assert exact.statistic == pytest.approx(statistic)
    assert exact.statistic == pytest.approx(low_level.chi_squared)
    assert exact.variance == pytest.approx(low_level.variance[0][0])
    assert exact_dotted.observed == pytest.approx(exact.observed)
    assert exact_dotted.expected == pytest.approx(exact.expected)
    assert exact_dotted.statistic == pytest.approx(exact.statistic)
    assert exact.statistic != pytest.approx(default.statistic)
    assert fh_exact.observed == pytest.approx(fh_observed)
    assert fh_exact.expected == pytest.approx(fh_expected)
    assert fh_exact.variance == pytest.approx(fh_variance)
    assert fh_exact.statistic == pytest.approx(fh_statistic)


def test_survdiff_counting_process_formula_strata_combines_delayed_entry_components():
    data = {
        "start": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0],
        "stop": [2.0, 2.5, 4.0, 4.5, 3.0, 5.0],
        "status": [1, 1, 0, 0, 1, 1],
        "group": ["treated", "treated", "control", "control", "treated", "control"],
        "site": ["x", "y", "x", "y", "x", "x"],
    }
    group_codes = [0 if value == "control" else 1 for value in data["group"]]
    strata_codes = [0 if value == "x" else 1 for value in data["site"]]

    def combine(rho=0.0):
        observed = [0.0, 0.0]
        expected = [0.0, 0.0]
        variance = 0.0
        for site in ("x", "y"):
            indices = [idx for idx, value in enumerate(data["site"]) if value == site]
            if rho == 0.0:
                result = survival.logrank_test(
                    [data["stop"][idx] for idx in indices],
                    [data["status"][idx] for idx in indices],
                    [group_codes[idx] for idx in indices],
                    entry_times=[data["start"][idx] for idx in indices],
                )
            else:
                result = survival.fleming_harrington_test(
                    [data["stop"][idx] for idx in indices],
                    [data["status"][idx] for idx in indices],
                    [group_codes[idx] for idx in indices],
                    rho,
                    0.0,
                    entry_times=[data["start"][idx] for idx in indices],
                )
            for group_idx in range(2):
                observed[group_idx] += result.observed[group_idx]
                expected[group_idx] += result.expected[group_idx]
            variance += result.variance
        statistic = (observed[0] - expected[0]) ** 2 / variance
        return observed, expected, variance, statistic

    observed, expected, variance, statistic = combine()
    fh_observed, fh_expected, fh_variance, fh_statistic = combine(0.5)

    result = survival.survdiff("Surv(start, stop, status) ~ group + strata(site)", data=data)
    fh = survival.survdiff(
        "Surv(start, stop, status) ~ group + strata(site)",
        data=data,
        rho=0.5,
    )
    exact = survival.survdiff(
        "Surv(start, stop, status) ~ group + strata(site)",
        data=data,
        timefix=False,
    )
    low_level = survival.surv_analysis.stratified_counting_logrank_components(
        data["stop"],
        data["status"],
        [code + 1 for code in group_codes],
        data["start"],
        strata_codes,
        0.0,
        False,
    )

    assert result.observed == pytest.approx(observed)
    assert result.expected == pytest.approx(expected)
    assert result.variance == pytest.approx(variance)
    assert result.statistic == pytest.approx(statistic)
    assert fh.observed == pytest.approx(fh_observed)
    assert fh.expected == pytest.approx(fh_expected)
    assert fh.variance == pytest.approx(fh_variance)
    assert fh.statistic == pytest.approx(fh_statistic)
    assert exact.observed == pytest.approx(observed)
    assert exact.expected == pytest.approx(expected)
    assert exact.variance == pytest.approx(variance)
    assert exact.statistic == pytest.approx(statistic)
    assert exact.statistic == pytest.approx(low_level.chi_squared)
    assert exact.variance == pytest.approx(low_level.variance[0][0])


def test_survdiff_timefix_false_uses_exact_event_times():
    times = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [1, 1, 0, 0]
    groups = ["control", "treated", "control", "treated"]
    response = survival.Surv(times, status)

    default = survival.survdiff(response, group=groups)
    exact = survival.survdiff(response, group=groups, timefix=False)
    exact_formula = survival.survdiff(
        "Surv(time, status) ~ group",
        data={"time": times, "status": status, "group": groups},
        **{"time.fix": False},
    )
    low_level = survival.survdiff2(times, status, [1, 2, 1, 2], None, None, False)

    assert default.statistic == pytest.approx(0.0)
    assert exact.statistic == pytest.approx(low_level.chi_squared)
    assert exact.statistic == pytest.approx(0.05882352941176476)
    assert exact.statistic != pytest.approx(default.statistic)
    assert exact_formula.statistic == pytest.approx(exact.statistic)
    assert exact.observed == pytest.approx(low_level.observed)
    assert exact.expected == pytest.approx(low_level.expected)


def test_survdiff_formula_accepts_dotted_na_action_alias():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "group": ["control", None, "treated", "treated"],
    }

    named = survival.survdiff("Surv(time, status) ~ group", data=data, na_action="omit")
    dotted = survival.survdiff(
        "Surv(time, status) ~ group",
        data=data,
        **{"na.action": "omit"},
    )

    assert dotted.statistic == pytest.approx(named.statistic)
    assert dotted.observed == pytest.approx(named.observed)
    assert dotted.expected == pytest.approx(named.expected)


def test_survdiff_formula_strata_requires_comparison_group():
    with pytest.raises(ValueError, match="no groups"):
        survival.survdiff("Surv(time, status) ~ strata(group)", data=_toy_data())


def test_survdiff_formula_supports_stratified_groups():
    data = {
        "time": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "status": [1, 0, 0, 1, 1, 1],
        "group": ["treated", "treated", "control", "control", "treated", "control"],
        "site": ["north", "south", "north", "south", "north", "south"],
    }
    group_codes = {"control": 1, "treated": 2}
    strata_codes = [0 if site == "north" else 1 for site in data["site"]]
    encoded_groups = [group_codes[value] for value in data["group"]]

    result = survival.survdiff("Surv(time, status) ~ group + strata(site)", data=data)
    low_level = survival.surv_analysis.stratified_logrank_components(
        data["time"],
        data["status"],
        encoded_groups,
        strata_codes,
        None,
        True,
    )
    expected_p = survival.lrt_test(
        low_level.chi_squared / 2.0,
        0.0,
        low_level.degrees_of_freedom,
    ).p_value

    assert result.statistic == pytest.approx(low_level.chi_squared)
    assert result.df == low_level.degrees_of_freedom
    assert result.p_value == pytest.approx(expected_p)
    assert result.observed == pytest.approx(low_level.observed)
    assert result.expected == pytest.approx(low_level.expected)
    assert result.weight_type == "LogRank"

    weighted = survival.survdiff("Surv(time, status) ~ group + strata(site)", data=data, rho=0.5)
    weighted_low_level = survival.surv_analysis.stratified_logrank_components(
        data["time"],
        data["status"],
        encoded_groups,
        strata_codes,
        0.5,
        True,
    )
    assert weighted.statistic == pytest.approx(weighted_low_level.chi_squared)
    assert weighted.observed == pytest.approx(weighted_low_level.observed)
    assert weighted.expected == pytest.approx(weighted_low_level.expected)
    assert weighted.weight_type == "FlemingHarrington(p=0.5, q=0)"


def test_survdiff_formula_strata_honors_timefix():
    data = {
        "time": [1.0, 1.0 + 5e-10, 2.0, 3.0, 1.0, 1.0 + 5e-10, 2.0, 3.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "group": ["control", "treated", "control", "treated"] * 2,
        "site": ["north"] * 4 + ["south"] * 4,
    }
    group_codes = {"control": 1, "treated": 2}
    strata_codes = [0 if site == "north" else 1 for site in data["site"]]
    encoded_groups = [group_codes[value] for value in data["group"]]

    default = survival.survdiff("Surv(time, status) ~ group + strata(site)", data=data)
    exact = survival.survdiff(
        "Surv(time, status) ~ group + strata(site)",
        data=data,
        timefix=False,
    )
    default_low_level = survival.surv_analysis.stratified_logrank_components(
        data["time"],
        data["status"],
        encoded_groups,
        strata_codes,
        None,
        True,
    )
    exact_low_level = survival.surv_analysis.stratified_logrank_components(
        data["time"],
        data["status"],
        encoded_groups,
        strata_codes,
        None,
        False,
    )

    assert default.statistic == pytest.approx(default_low_level.chi_squared)
    assert exact.statistic == pytest.approx(exact_low_level.chi_squared)
    assert exact.statistic != pytest.approx(default.statistic)
    assert default.observed == pytest.approx(default_low_level.observed)
    assert exact.expected == pytest.approx(exact_low_level.expected)


def test_survdiff_formula_dot_uses_remaining_group_column():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", "control", "treated", "treated"],
    }
    direct = survival.survdiff("Surv(time, status) ~ arm", data=data)
    expanded = survival.survdiff("Surv(time, status) ~ .", data=data)

    assert expanded.statistic == pytest.approx(direct.statistic)
    assert expanded.observed == pytest.approx(direct.observed)


def test_survdiff_formula_dot_can_exclude_identifier_columns():
    data = {
        "id": [101, 102, 103, 104],
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", "control", "treated", "treated"],
    }
    direct = survival.survdiff("Surv(time, status) ~ arm", data=data)
    expanded = survival.survdiff("Surv(time, status) ~ . - id", data=data)

    assert expanded.statistic == pytest.approx(direct.statistic)
    assert expanded.observed == pytest.approx(direct.observed)


def test_survdiff_formula_accepts_backtick_column_names():
    data = _backtick_data()
    result = survival.survdiff(
        "Surv(`follow-up`, `event status`) ~ `treatment arm`",
        data=data,
    )
    direct = survival.survdiff(
        survival.Surv(data["follow-up"], data["event status"]),
        group=data["treatment arm"],
    )

    assert result.statistic == pytest.approx(direct.statistic)
    assert result.observed == pytest.approx(direct.observed)


def test_survdiff_direct_surv_applies_subset_to_group_labels():
    data = _toy_data()
    indices = [0, 1, 2, 3, 5, 6]
    response = survival.Surv(data["time"], data["status"])
    result = survival.survdiff(response, group=data["group"], subset=indices)
    filtered = _take(data, indices)
    direct = survival.survdiff(
        survival.Surv(filtered["time"], filtered["status"]),
        group=filtered["group"],
    )

    assert result.statistic == pytest.approx(direct.statistic)
    assert result.observed == pytest.approx(direct.observed)


def test_survdiff_formula_accepts_interaction_groups():
    data = _factor_data()
    group_values = [
        (data["dose"][idx], math.sqrt(data["x1"][idx])) for idx in range(len(data["time"]))
    ]
    result = survival.survdiff("Surv(time, status) ~ factor(dose):sqrt(x1)", data=data)
    direct = survival.survdiff(
        survival.Surv(data["time"], data["status"]),
        group=group_values,
    )
    direct_levels = list(dict.fromkeys(group_values))
    formula_levels = sorted(set(group_values))
    direct_order = [direct_levels.index(level) for level in formula_levels]

    assert result.statistic == pytest.approx(direct.statistic)
    assert result.observed == pytest.approx([direct.observed[idx] for idx in direct_order])
    assert result.expected == pytest.approx([direct.expected[idx] for idx in direct_order])


def test_survdiff_formula_accepts_identity_wrappers_for_groups():
    data = _factor_data()
    for wrapper in ("I", "identity", "as.numeric"):
        result = survival.survdiff(f"Surv(time, status) ~ {wrapper}(dose)", data=data)
        direct = survival.survdiff(
            survival.Surv(data["time"], data["status"]),
            group=data["dose"],
        )

        assert result.statistic == pytest.approx(direct.statistic)
        assert result.observed == pytest.approx(direct.observed)
