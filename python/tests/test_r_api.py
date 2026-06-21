import math
from bisect import bisect_right
from itertools import combinations
from statistics import NormalDist

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _toy_data():
    return {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "status": [1, 1, 0, 1, 0, 1, 1, 0],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0, 1.2, 0.6, 1.4],
        "x2": [1.0, 0.9, 1.1, 0.7, 0.4, 0.3, 0.6, 0.2],
        "offset": [0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0],
    }


def _numeric_data():
    data = _toy_data()
    return {
        "time": data["time"],
        "status": data["status"],
        "x1": data["x1"],
        "x2": data["x2"],
    }


def _numeric_data_with_id():
    data = _numeric_data()
    return {"id": list(range(1, len(data["time"]) + 1)), **data}


def _backtick_data():
    data = _toy_data()
    return {
        "follow-up": data["time"],
        "event status": data["status"],
        "treatment arm": data["group"],
        "age-years": data["x1"],
        "marker/value": data["x2"],
        "log exposure": data["offset"],
    }


def _factor_data():
    data = _toy_data()
    return {
        "time": data["time"],
        "status": data["status"],
        "x1": data["x1"],
        "x2": data["x2"],
        "dose": [0, 0, 1, 1, 2, 2, 0, 1],
    }


def _tied_cox_data():
    return {
        "time": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        "status": [1, 1, 0, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5],
        "strata": ["A", "A", "A", "B", "B", "B", "A", "B"],
    }


def _counting_cox_data():
    return {
        "start": [0.0, 0.0, 1.5, 2.5, 0.0, 3.0],
        "stop": [2.0, 2.0, 4.0, 5.0, 5.0, 6.0],
        "status": [1, 1, 1, 0, 1, 0],
        "x1": [0.2, 0.8, 0.4, 1.1, 0.7, 0.3],
    }


def _take(data, indices):
    return {key: [values[idx] for idx in indices] for key, values in data.items()}


def _with_intercept(rows):
    return [[1.0, *row] for row in rows]


def _weibull_saturated_center_loglik(time, time2, status, scale):
    rows = []
    for idx, event in enumerate(status):
        y = math.log(time[idx])
        if event == 3:
            if time2 is None:
                raise AssertionError("interval rows require time2")
            width = (math.log(time2[idx]) - y) / scale
            log_temp = math.log(width) - math.log(math.expm1(width))
            temp = math.exp(log_temp)
            tail = 0.0 if width > 40.0 else math.log1p(-math.exp(-math.exp(width)))
            rows.append((y - log_temp, -temp + tail))
        elif event == 1:
            rows.append((y, -(1.0 + math.log(scale))))
        else:
            rows.append((y, 0.0))
    return rows


def _survreg_deviance_from_matrix(matrix, saturated_rows):
    working = [0.0 if abs(row[2]) <= 1e-12 else -row[1] / row[2] for row in matrix]
    result = []
    for row, (_, saturated_loglik), work in zip(matrix, saturated_rows, working, strict=True):
        magnitude = math.sqrt(max(0.0, 2.0 * (saturated_loglik - row[0])))
        result.append(magnitude if work > 0.0 else -magnitude if work < 0.0 else 0.0)
    return result


def _fit_event_times(fit):
    strata = fit.strata if hasattr(fit, "strata") else [0] * len(fit.status)
    order = sorted(
        range(len(fit.status)),
        key=lambda idx: (strata[idx], fit.event_times[idx], idx),
    )
    return [float(fit.event_times[idx]) for idx in order if fit.status[idx] == 1]


def _manual_cox_loglik_at_zero(time, status, method, entry_times=None):
    loglik = 0.0
    event_times = sorted({time[idx] for idx, event in enumerate(status) if event == 1})
    for event_time in event_times:
        deaths = sum(
            1 for idx, event in enumerate(status) if event == 1 and time[idx] == event_time
        )
        if entry_times is None:
            risk = sum(1 for value in time if value >= event_time)
        else:
            risk = sum(
                1
                for start, stop in zip(entry_times, time, strict=True)
                if start < event_time <= stop
            )
        if method == "breslow":
            loglik -= deaths * math.log(risk)
        elif method == "exact":
            loglik -= math.log(math.comb(risk, deaths))
        else:
            loglik -= sum(math.log(risk - step) for step in range(deaths))
    return loglik


def _manual_cox_robust_variance(fit, cluster):
    score = fit.score_residuals()
    nvar = len(fit.coefficients[0])
    naive = fit.information_matrix
    weights = fit.weights if hasattr(fit, "weights") else [1.0] * len(score)
    cluster_scores = {}
    for row_idx, label in enumerate(cluster):
        row = cluster_scores.setdefault(label, [0.0] * nvar)
        for col_idx in range(nvar):
            row[col_idx] += weights[row_idx] * score[row_idx][col_idx]

    meat = [[0.0 for _ in range(nvar)] for _ in range(nvar)]
    for row in cluster_scores.values():
        for left in range(nvar):
            for right in range(nvar):
                meat[left][right] += row[left] * row[right]

    return [
        [
            sum(
                naive[left][inner_left] * meat[inner_left][inner_right] * naive[inner_right][right]
                for inner_left in range(nvar)
                for inner_right in range(nvar)
            )
            for right in range(nvar)
        ]
        for left in range(nvar)
    ]


def _manual_survreg_robust_variance(fit, cluster):
    dfbeta = survival.r_api.residuals(
        fit,
        type="dfbeta",
        weighted=True,
        collapse=cluster,
        rsigma=True,
    )
    width = len(dfbeta[0]) if dfbeta else len(fit.variance_matrix)
    robust = [[0.0 for _ in range(width)] for _ in range(width)]
    for row in dfbeta:
        for left in range(width):
            for right in range(width):
                robust[left][right] += row[left] * row[right]
    return robust


def _manual_cox_loglik(
    time,
    status,
    covariates,
    beta,
    method,
    *,
    entry_times=None,
    weights=None,
    offset=None,
    strata=None,
):
    weights = [1.0] * len(time) if weights is None else weights
    offset = [0.0] * len(time) if offset is None else offset
    strata = [0] * len(time) if strata is None else strata
    linear_predictors = [
        offset[idx]
        + sum(value * coefficient for value, coefficient in zip(covariates[idx], beta, strict=True))
        for idx in range(len(time))
    ]
    risks = [weights[idx] * math.exp(linear_predictors[idx]) for idx in range(len(time))]
    loglik = 0.0
    for stratum in sorted(set(strata)):
        event_times = sorted(
            {time[idx] for idx, event in enumerate(status) if event == 1 and strata[idx] == stratum}
        )
        for event_time in event_times:
            deaths = [
                idx
                for idx, event in enumerate(status)
                if event == 1 and strata[idx] == stratum and time[idx] == event_time
            ]
            at_risk = [
                idx
                for idx in range(len(time))
                if strata[idx] == stratum
                and time[idx] >= event_time
                and (entry_times is None or entry_times[idx] < event_time)
            ]
            death_count = len(deaths)
            deadwt = sum(weights[idx] for idx in deaths)
            denom = sum(risks[idx] for idx in at_risk)
            denom2 = sum(risks[idx] for idx in deaths)
            loglik += sum(weights[idx] * linear_predictors[idx] for idx in deaths)
            if method == "breslow" or death_count == 1:
                loglik -= deadwt * math.log(denom)
            elif method == "exact":
                exact_denom = 0.0
                for combo in combinations(at_risk, death_count):
                    exact_denom += math.exp(sum(linear_predictors[idx] for idx in combo))
                loglik -= math.log(exact_denom)
            else:
                weight_average = deadwt / death_count
                for step in range(death_count):
                    loglik -= weight_average * math.log(denom - (step / death_count) * denom2)
    return loglik


def _manual_counting_concordance(start, stop, status, scores, weights=None, timewt="n"):
    concordant, comparable = _manual_counting_concordance_counts(
        start,
        stop,
        status,
        scores,
        weights,
        timewt,
    )
    return concordant / comparable if comparable else 0.5


def _manual_concordance_time_multiplier(timewt, total_weight, survival, censoring_survival, nrisk):
    if nrisk <= 0.0:
        return 0.0
    if timewt == "S":
        return total_weight * survival / nrisk
    if timewt == "S/G":
        return (
            total_weight * survival / (censoring_survival * nrisk)
            if censoring_survival > 0.0
            else 0.0
        )
    if timewt == "n/G2":
        return 1.0 / (censoring_survival * censoring_survival) if censoring_survival > 0.0 else 0.0
    if timewt == "I":
        return 1.0 / nrisk
    return 1.0


def _manual_right_time_multipliers(time, status, weights, timewt):
    if timewt == "n":
        return dict.fromkeys(
            {time[idx] for idx, event in enumerate(status) if event == 1},
            1.0,
        )
    total_weight = float(len(time)) if weights is None else sum(weights)
    survival = 1.0
    censoring_survival = 1.0
    multipliers = {}
    for event_time in sorted(set(time)):
        indices = [idx for idx, value in enumerate(time) if value == event_time]
        nrisk = sum(
            (1.0 if weights is None else weights[idx])
            for idx, value in enumerate(time)
            if value >= event_time
        )
        death_weight = sum(
            (1.0 if weights is None else weights[idx]) for idx in indices if status[idx] == 1
        )
        censor_weight = sum(
            (1.0 if weights is None else weights[idx]) for idx in indices if status[idx] != 1
        )
        if death_weight > 0.0:
            multipliers[event_time] = _manual_concordance_time_multiplier(
                timewt,
                total_weight,
                survival,
                censoring_survival,
                nrisk,
            )
            if nrisk > 0.0:
                survival *= max((nrisk - death_weight) / nrisk, 0.0)
        if censor_weight > 0.0 and nrisk > 0.0:
            censoring_survival *= max((nrisk - censor_weight) / nrisk, 0.0)
    return multipliers


def _manual_counting_time_multipliers(start, stop, status, weights, timewt):
    if timewt == "n":
        return dict.fromkeys(
            {stop[idx] for idx, event in enumerate(status) if event == 1},
            1.0,
        )
    total_weight = float(len(stop)) if weights is None else sum(weights)
    survival = 1.0
    multipliers = {}
    for event_time in sorted({stop[idx] for idx, event in enumerate(status) if event == 1}):
        nrisk = sum(
            (1.0 if weights is None else weights[idx])
            for idx, (entry, exit_time) in enumerate(zip(start, stop, strict=True))
            if entry < event_time <= exit_time
        )
        multipliers[event_time] = _manual_concordance_time_multiplier(
            timewt,
            total_weight,
            survival,
            1.0,
            nrisk,
        )
        death_weight = sum(
            (1.0 if weights is None else weights[idx])
            for idx, event in enumerate(status)
            if event == 1 and stop[idx] == event_time
        )
        if nrisk > 0.0:
            survival *= max((nrisk - death_weight) / nrisk, 0.0)
    return multipliers


def _manual_concordance_bounded_times_and_status(time, status, ymin=None, ymax=None):
    bounded_time = [max(value, ymin) for value in time] if ymin is not None else list(time)
    bounded_status = [
        0 if ymax is not None and event == 1 and bounded_time[idx] > ymax else event
        for idx, event in enumerate(status)
    ]
    return bounded_time, bounded_status


def _manual_counting_concordance_counts(start, stop, status, scores, weights=None, timewt="n"):
    comparable = 0.0
    concordant = 0.0
    multipliers = _manual_counting_time_multipliers(start, stop, status, weights, timewt)
    for event_idx, event in enumerate(status):
        if event != 1:
            continue
        event_time = stop[event_idx]
        multiplier = multipliers.get(event_time, 0.0)
        if multiplier <= 0.0:
            continue
        for risk_idx in range(len(stop)):
            if risk_idx == event_idx:
                continue
            if start[risk_idx] < event_time and stop[risk_idx] > event_time:
                pair_weight = (
                    1.0 if weights is None else weights[event_idx] * weights[risk_idx]
                ) * multiplier
                comparable += pair_weight
                diff = scores[event_idx] - scores[risk_idx]
                if diff > 0.0:
                    concordant += pair_weight
                elif abs(diff) < 1e-12:
                    concordant += 0.5 * pair_weight
    return concordant, comparable


def _manual_right_concordance_counts(time, status, scores, weights=None, timewt="n"):
    comparable = 0.0
    concordant = 0.0
    multipliers = _manual_right_time_multipliers(time, status, weights, timewt)
    for left in range(len(time)):
        for right in range(left + 1, len(time)):
            if status[left] == 1 and time[left] < time[right]:
                event_idx, risk_idx = left, right
            elif status[right] == 1 and time[right] < time[left]:
                event_idx, risk_idx = right, left
            else:
                continue
            multiplier = multipliers.get(time[event_idx], 0.0)
            pair_weight = (
                1.0 if weights is None else weights[event_idx] * weights[risk_idx]
            ) * multiplier
            comparable += pair_weight
            diff = scores[event_idx] - scores[risk_idx]
            if diff > 0.0:
                concordant += pair_weight
            elif abs(diff) < 1e-12:
                concordant += 0.5 * pair_weight
    return concordant, comparable


def _manual_fh_from_km(km, ctype=1, event_counts=None):
    hazard = 0.0
    cumhaz = []
    estimate = []
    if event_counts is None:
        event_counts = km.n_event
    for risk, events, unweighted_events in zip(
        km.n_risk,
        km.n_event,
        event_counts,
        strict=True,
    ):
        if risk > 0.0 and events > 0.0 and unweighted_events > 0.0:
            if ctype == 1:
                hazard += events / risk
            else:
                for step in range(int(unweighted_events)):
                    hazard += events / (
                        unweighted_events * (risk - step * events / unweighted_events)
                    )
        cumhaz.append(hazard)
        estimate.append(math.exp(-hazard))
    return cumhaz, estimate


def _manual_fh_std_chaz_from_km(km, ctype=1, event_counts=None):
    variance = 0.0
    std_chaz = []
    if event_counts is None:
        event_counts = km.n_event
    for risk, events, unweighted_events in zip(
        km.n_risk,
        km.n_event,
        event_counts,
        strict=True,
    ):
        if risk > 0.0 and events > 0.0 and unweighted_events > 0.0:
            if ctype == 1:
                variance += events / (risk * risk)
            else:
                for step in range(int(unweighted_events)):
                    denominator = risk - step * events / unweighted_events
                    if denominator > 0.0:
                        variance += events / (unweighted_events * denominator * denominator)
        std_chaz.append(math.sqrt(max(variance, 0.0)))
    return std_chaz


def _plain_confidence_interval(estimate, std_err, conf_level=0.95):
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf_level) / 2.0)
    return max(estimate - z * std_err, 0.0), min(estimate + z * std_err, 1.0)


def test_surv_right_censored_response():
    response = survival.Surv([1, 2, 3], [1, 0, 1])
    all_observed = survival.Surv([1, 2, 3])
    right_abbrev = survival.Surv([1, 2, 3], [1, 0, 1], type="r")

    assert len(response) == 3
    assert response.type == "right"
    assert response.time == pytest.approx((1.0, 2.0, 3.0))
    assert response.status == (1, 0, 1)
    assert all_observed.type == "right"
    assert all_observed.status == (1, 1, 1)
    assert right_abbrev.type == "right"
    assert survival.r_api.is_surv(response) is True
    assert survival.r_api.is_surv({"time": [1, 2, 3]}) is False

    for explicit_type in ("right", "left"):
        with pytest.raises(ValueError, match="one-argument Surv"):
            survival.Surv([1, 2, 3], type=explicit_type)
    with pytest.raises(ValueError, match="ambiguous"):
        survival.Surv([1, 2], [1, 0], type="i")


def test_surv_accepts_r_one_two_event_coding():
    response = survival.Surv([1, 2, 3], [1, 2, 1])

    assert response.status == (0, 1, 0)

    with pytest.raises(ValueError, match="0/1 or 1/2"):
        survival.Surv([1, 2], [0, 2])
    with pytest.raises(ValueError, match="0/1 or 1/2"):
        survival.Surv([1, 2], [0.5, 1.0])


def test_surv_origin_shifts_supported_time_columns():
    all_observed = survival.Surv([11.0, 12.0], origin=10.0)
    right = survival.Surv([11.0, 12.0], [1, 0], origin=10.0)
    counting = survival.Surv([10.0, 11.0], [12.0, 14.0], [1, 0], origin=10.0)
    interval = survival.Surv([11.0, 12.0], [11.0, 15.0], [1, 3], type="interval", origin=10.0)
    interval2 = survival.Surv(
        [float("-inf"), 12.0, 13.0],
        [11.0, float("inf"), 15.0],
        type="interval2",
        origin=10.0,
    )

    assert all_observed.time == pytest.approx((1.0, 2.0))
    assert all_observed.status == (1, 1)
    assert right.time == pytest.approx((1.0, 2.0))
    assert counting.start == pytest.approx((0.0, 1.0))
    assert counting.time == pytest.approx((2.0, 4.0))
    assert interval.time == pytest.approx((1.0, 2.0))
    assert interval.time2 == pytest.approx((1.0, 5.0))
    assert interval2.time[0] == float("-inf")
    assert interval2.time[1:] == pytest.approx((2.0, 3.0))
    assert interval2.time2[:2] == pytest.approx((1.0, float("inf")))
    assert interval2.time2[2] == pytest.approx(5.0)

    with pytest.raises(TypeError, match="origin must be numeric"):
        survival.Surv([1.0, 2.0], origin="baseline")
    with pytest.raises(ValueError, match="origin must be finite"):
        survival.Surv([1.0, 2.0], origin=float("nan"))


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


def test_r_api_bool_options_accept_numpy_bool_scalars():
    np = pytest.importorskip("numpy")
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    rows = [[0.5, 0.8]]
    scores = [8.0 - value for value in data["time"]]

    direct_reverse = survival.survfit(response, reverse=True)
    numpy_reverse = survival.survfit(response, reverse=np.bool_(True))
    assert numpy_reverse.time == pytest.approx(direct_reverse.time)
    assert numpy_reverse.estimate == pytest.approx(direct_reverse.estimate)

    direct_concordance = survival.concordance(response, scores=scores, reverse=True)
    numpy_concordance = survival.concordance(response, scores=scores, reverse=np.bool_(True))
    numpy_timefix = survival.concordance(response, scores=scores, timefix=np.bool_(False))
    assert numpy_concordance.concordance == pytest.approx(direct_concordance.concordance)
    assert numpy_concordance.reverse is True
    assert numpy_timefix.concordance == pytest.approx(
        survival.concordance(response, scores=scores, timefix=False).concordance
    )

    direct_basehaz = survival.basehaz(fit, centered=False)
    numpy_basehaz = survival.basehaz(fit, centered=np.bool_(False))
    assert numpy_basehaz.cumhaz == pytest.approx(direct_basehaz.cumhaz)
    assert numpy_basehaz.centered is False

    zph = survival.cox_zph(
        fit,
        terms=np.bool_(False),
        singledf=np.bool_(False),
        global_test=np.bool_(False),
    )
    assert zph.variable_names == ["x1", "x2"]
    assert zph.table[-1]["name"] != "GLOBAL"

    prediction = survival.predict(fit, rows, se_fit=np.bool_(True))
    assert isinstance(prediction, survival.r_api.PredictResult)
    assert prediction.fit == pytest.approx(survival.predict(fit, rows))

    uncentered = survival.predict(fit, rows, centered=np.bool_(False))
    assert uncentered == pytest.approx(survival.predict(fit, rows, reference="zero"))

    residual_values = survival.r_api.residuals(fit, weighted=np.bool_(False))
    assert residual_values == pytest.approx(survival.r_api.residuals(fit, weighted=False))

    detail = survival.coxph_detail(fit, riskmat=np.bool_(True))
    assert detail.riskmat is not None


def test_r_api_bool_options_reject_python_truthiness():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    rows = [[0.5, 0.8]]
    scores = [8.0 - value for value in data["time"]]

    with pytest.raises(TypeError, match="reverse"):
        survival.survfit(response, reverse=1)
    with pytest.raises(TypeError, match="reverse"):
        survival.concordance(response, scores=scores, reverse="yes")
    with pytest.raises(TypeError, match="timefix"):
        survival.concordance(response, scores=scores, timefix=1)
    with pytest.raises(TypeError, match="centered"):
        survival.basehaz(fit, centered=1)
    with pytest.raises(TypeError, match="terms"):
        survival.cox_zph(fit, terms=1)
    with pytest.raises(TypeError, match="singledf"):
        survival.cox_zph(fit, singledf="yes")
    with pytest.raises(TypeError, match="global"):
        survival.cox_zph(fit, global_test=1)
    with pytest.raises(ValueError, match="global_test or global"):
        survival.cox_zph(fit, global_test=False, **{"global": True})
    with pytest.raises(TypeError, match="se_fit"):
        survival.predict(fit, rows, se_fit=1)
    with pytest.raises(TypeError, match="centered"):
        survival.predict(fit, rows, centered=1)
    with pytest.raises(TypeError, match="weighted"):
        survival.r_api.residuals(fit, weighted=1)
    with pytest.raises(TypeError, match="riskmat"):
        survival.coxph_detail(fit, riskmat=1)


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
        event_counts=[2.0, 1.0],
    )
    expected_std_chaz = _manual_fh_std_chaz_from_km(
        km,
        ctype=2,
        event_counts=[2.0, 1.0],
    )

    assert km.n_event == pytest.approx([3.0, 1.0])
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


def test_survfit_model_true_stores_direct_and_formula_inputs():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    weights = [1.0 + 0.1 * idx for idx in range(len(data["time"]))]

    direct = survival.survfit(response, model=True)
    grouped = survival.survfit(response, group=data["group"], weights=weights, model=True)
    formula = survival.survfit("Surv(time, status) ~ group", data=data, model=True)

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

    assert default.time == pytest.approx([1.0, 2.0])
    assert default.n_risk == pytest.approx([2.0, 2.0])
    assert default.n_event == pytest.approx([2.0, 0.0])
    assert default.estimate == pytest.approx([0.0, 0.0])

    assert exact.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])
    assert exact.n_risk == pytest.approx([2.0, 2.0, 2.0])
    assert exact.n_event == pytest.approx([1.0, 1.0, 0.0])
    assert exact.n_censor == pytest.approx([0.0, 0.0, 2.0])
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


def test_na_action_accepts_r_style_names_and_rejects_non_strings():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 0, 1, 1],
        "arm": ["control", None, "treated", "treated"],
    }
    direct = survival.survfit(
        survival.Surv([1.0, 3.0, 4.0], [1, 1, 1]),
        group=["control", "treated", "treated"],
    )

    for na_action in ("na.omit", " na.exclude "):
        fit = survival.survfit("Surv(time, status) ~ arm", data=data, na_action=na_action)
        assert list(fit) == list(direct)
        assert fit["control"].estimate == pytest.approx(direct["control"].estimate)
        assert fit["treated"].estimate == pytest.approx(direct["treated"].estimate)

    passthrough = survival.survfit(
        "Surv(time, status) ~ group",
        data=_toy_data(),
        na_action="na.pass",
    )
    default = survival.survfit("Surv(time, status) ~ group", data=_toy_data())
    assert list(passthrough) == list(default)
    assert passthrough["A"].estimate == pytest.approx(default["A"].estimate)
    assert passthrough["B"].estimate == pytest.approx(default["B"].estimate)

    with pytest.raises(TypeError, match="na_action"):
        survival.survfit("Surv(time, status) ~ group", data=_toy_data(), na_action=1)


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

    assert list(interaction) == list(direct)
    for key in direct:
        assert interaction[key].estimate == pytest.approx(direct[key].estimate)


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

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.logrank_test([1.0, 2.0], [1, 2], [0, 1])

    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.logrank_trend([1.0, 2.0], [1, 2], [0, 1])


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


def test_concordance_direct_surv_uses_rust_c_index():
    data = _toy_data()
    scores = [8.0 - value for value in data["time"]]
    result = survival.concordance(survival.Surv(data["time"], data["status"]), scores=scores)
    low_level = survival.concordance_index(data["time"], data["status"], scores)
    summary = survival.core.concordance_summary(data["time"], data["status"], scores)

    assert result.concordance == pytest.approx(low_level)
    assert result.c_index == pytest.approx(result.concordance)
    assert result.n == len(data["time"])
    assert result.n_event == sum(data["status"])
    assert result.reverse is False
    assert result.concordant == pytest.approx(summary["concordant"])
    assert result.comparable == pytest.approx(summary["comparable"])

    reversed_result = survival.concordance(
        survival.Surv(data["time"], data["status"]),
        scores=scores,
        reverse=True,
    )
    assert reversed_result.concordance == pytest.approx(
        survival.concordance_index(data["time"], data["status"], [-value for value in scores])
    )
    assert reversed_result.reverse is True


def test_concordance_timefix_false_uses_exact_event_times():
    times = [1.0, 1.0 + 5e-10, 2.0]
    status = [1, 1, 0]
    scores = [0.9, 0.1, 0.5]
    response = survival.Surv(times, status)

    default = survival.concordance(response, scores=scores)
    exact = survival.concordance(response, scores=scores, timefix=False)
    exact_dotted = survival.concordance(
        "Surv(time, status) ~ score",
        data={"time": times, "status": status, "score": scores},
        **{"time.fix": False},
    )
    fixed_times = [1.0, 1.0, 2.0]
    fixed_concordant, fixed_comparable = _manual_right_concordance_counts(
        fixed_times,
        status,
        scores,
    )
    exact_concordant, exact_comparable = _manual_right_concordance_counts(
        times,
        status,
        scores,
    )

    assert default.concordance == pytest.approx(fixed_concordant / fixed_comparable)
    assert exact.concordance == pytest.approx(exact_concordant / exact_comparable)
    assert exact_dotted.concordance == pytest.approx(exact.concordance)
    assert default.concordant == pytest.approx(fixed_concordant)
    assert default.comparable == pytest.approx(fixed_comparable)
    assert exact.concordant == pytest.approx(exact_concordant)
    assert exact.comparable == pytest.approx(exact_comparable)
    assert default.concordance != pytest.approx(exact.concordance)
    assert default.n_event == exact.n_event == sum(status)


def test_concordance_accepts_r_style_defaults():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
        "score": [0.8, 0.6, 0.2, 0.1],
    }

    default = survival.concordance("Surv(time, status) ~ score", data=data)
    explicit = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights=None,
        subset=None,
        cluster=None,
        ymin=None,
        ymax=None,
        timewt="n",
        influence=0,
        ranks=False,
        reverse=False,
        timefix=True,
        keepstrata=10,
    )
    r_default_timewt = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        timewt=("n", "S", "S/G", "n/G2", "I"),
        influence=None,
        ranks=None,
        keepstrata=None,
    )

    assert explicit.concordance == pytest.approx(default.concordance)
    assert explicit.concordant == pytest.approx(default.concordant)
    assert explicit.comparable == pytest.approx(default.comparable)
    assert r_default_timewt.concordance == pytest.approx(default.concordance)


def test_concordance_ranks_return_weighted_event_contributions():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.6, 0.4, 0.1],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])

    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    ranked = survival.concordance(response, scores=data["score"], weights=data["wt"], ranks=True)
    formula_ranked = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        ranks=True,
    )
    reversed_ranked = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ranks=True,
        reverse=True,
    )

    assert default.ranks is None
    assert ranked.ranks is not None
    assert [row["time"] for row in ranked.ranks] == pytest.approx([1.0, 2.0, 3.0])
    assert [row["casewt"] for row in ranked.ranks] == pytest.approx([2.0, 1.0, 3.0])
    assert [row["timewt"] for row in ranked.ranks] == pytest.approx([7.0, 5.0, 4.0])
    assert [row["rank"] for row in ranked.ranks] == pytest.approx([5.0 / 7.0, 4.0 / 5.0, 0.25])
    assert ranked.concordance == pytest.approx(default.concordance)
    assert formula_ranked.ranks is not None
    for formula_row, direct_row in zip(formula_ranked.ranks, ranked.ranks, strict=True):
        assert formula_row.keys() == direct_row.keys()
        for key in direct_row:
            assert formula_row[key] == pytest.approx(direct_row[key])
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in ranked.ranks) == (
        pytest.approx(2.0 * ranked.concordant - ranked.comparable)
    )
    assert sum(
        row["rank"] * row["timewt"] * row["casewt"] for row in reversed_ranked.ranks
    ) == pytest.approx(2.0 * reversed_ranked.concordant - reversed_ranked.comparable)
    assert reversed_ranked.concordance == pytest.approx(1.0 - ranked.concordance)


def test_concordance_influence_modes_return_r_style_diagnostics():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])

    dfbeta_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=1,
    )
    influence_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=2,
    )
    both = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
    )
    bool_alias = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=True,
    )
    reversed_both = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
        reverse=True,
    )

    assert dfbeta_only.dfbeta is not None
    assert dfbeta_only.influence is None
    assert influence_only.dfbeta is None
    assert influence_only.influence is not None
    assert both.dfbeta == pytest.approx(dfbeta_only.dfbeta)
    assert bool_alias.dfbeta == pytest.approx(dfbeta_only.dfbeta)
    assert both.influence is not None
    assert influence_only.influence is not None
    for both_row, influence_row in zip(both.influence, influence_only.influence, strict=True):
        assert both_row == pytest.approx(influence_row)
    assert both.variance == pytest.approx(sum(value * value for value in both.dfbeta))
    assert both.var == pytest.approx(both.variance)
    assert [sum(row[col] for row in both.influence) for col in range(5)] == pytest.approx(
        [both.concordant, both.comparable - both.concordant, 0.0, 0.0, 0.0]
    )
    assert sum(both.dfbeta) == pytest.approx(0.0)
    assert reversed_both.concordance == pytest.approx(1.0 - both.concordance)
    assert reversed_both.dfbeta == pytest.approx([-value for value in both.dfbeta])
    assert [sum(row[col] for row in reversed_both.influence) for col in range(5)] == pytest.approx(
        [both.comparable - both.concordant, both.concordant, 0.0, 0.0, 0.0]
    )


def test_concordance_cluster_collapses_dfbeta_for_robust_variance():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
        "wt": [2.0, 1.0, 3.0, 1.0],
        "cluster": ["a", "a", "b", "c"],
    }
    response = survival.Surv(data["time"], data["status"])

    unclustered = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        influence=3,
    )
    clustered = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        cluster=data["cluster"],
        influence=3,
    )
    formula_clustered = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        cluster="cluster",
        influence=1,
    )
    formula_term_clustered = survival.concordance(
        "Surv(time, status) ~ score + cluster(cluster)",
        data=data,
        weights="wt",
        influence=1,
    )
    variance_only = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        cluster=data["cluster"],
    )

    expected_dfbeta = [
        unclustered.dfbeta[0] + unclustered.dfbeta[1],
        unclustered.dfbeta[2],
        unclustered.dfbeta[3],
    ]
    assert clustered.dfbeta == pytest.approx(expected_dfbeta)
    assert formula_clustered.dfbeta == pytest.approx(expected_dfbeta)
    assert formula_term_clustered.dfbeta == pytest.approx(expected_dfbeta)
    assert clustered.variance == pytest.approx(sum(value * value for value in expected_dfbeta))
    assert formula_clustered.variance == pytest.approx(clustered.variance)
    assert formula_term_clustered.variance == pytest.approx(clustered.variance)
    assert variance_only.dfbeta is None
    assert variance_only.influence is None
    assert variance_only.variance == pytest.approx(clustered.variance)
    for clustered_row, unclustered_row in zip(
        clustered.influence,
        unclustered.influence,
        strict=True,
    ):
        assert clustered_row == pytest.approx(unclustered_row)


def test_concordance_accepts_case_weights():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.4, 0.9, 0.2, 0.1],
        "wt": [5.0, 1.0, 2.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    expected_concordant, expected_comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
    )

    direct = survival.concordance(response, scores=data["score"], weights=data["wt"])
    formula = survival.concordance("Surv(time, status) ~ score", data=data, weights="wt")
    formula_vector = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights=data["wt"],
    )
    low_level = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
    )

    assert direct.concordance == pytest.approx(expected_concordant / expected_comparable)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert formula_vector.concordance == pytest.approx(direct.concordance)
    assert direct.concordant == pytest.approx(expected_concordant)
    assert direct.comparable == pytest.approx(expected_comparable)
    assert low_level["concordance"] == pytest.approx(direct.concordance)
    assert low_level["concordant"] == pytest.approx(expected_concordant)
    assert low_level["comparable"] == pytest.approx(expected_comparable)
    assert survival.concordance(response, scores=data["score"]).concordance != pytest.approx(
        direct.concordance
    )
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
    ) == pytest.approx(direct.concordance)


def test_concordance_accepts_identity_time_weight():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="I",
    )
    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        timewt="I",
    )
    summary = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    ) == pytest.approx(direct.concordance)
    assert direct.concordance != pytest.approx(default.concordance)


@pytest.mark.parametrize("timewt", ["S", "S/G", "n/G2"])
def test_concordance_accepts_km_time_weights(timewt):
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 0, 1, 0, 1, 1],
        "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.4],
        "wt": [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
    }
    response = survival.Surv(data["time"], data["status"])
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        data["score"],
        data["wt"],
        timewt=timewt,
    )

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt=timewt,
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        timewt=timewt,
    )
    summary = survival.core.concordance_summary(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt=timewt,
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.concordance_index(
        data["time"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt=timewt,
    ) == pytest.approx(direct.concordance)


def test_concordance_accepts_time_window_restrictions():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 1, 0],
        "score": [0.2, 0.9, 0.1, 0.8, 0.3, 0.4],
        "wt": [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
    }
    ymin = 2.5
    ymax = 4.0
    bounded_time, bounded_status = _manual_concordance_bounded_times_and_status(
        data["time"],
        data["status"],
        ymin=ymin,
        ymax=ymax,
    )
    concordant, comparable = _manual_right_concordance_counts(
        bounded_time,
        bounded_status,
        data["score"],
        data["wt"],
        timewt="S",
    )
    response = survival.Surv(data["time"], data["status"])

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )
    formula = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        weights="wt",
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert direct.n_event == sum(bounded_status)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert formula.n_event == direct.n_event


def test_concordance_formula_applies_subset_and_na_action():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 0, 1, 0],
        "score": [5.0, None, 3.0, 2.0, 1.0],
    }
    result = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        subset=[0, 1, 2, 3],
        na_action="omit",
    )
    dotted = survival.concordance(
        "Surv(time, status) ~ score",
        data=data,
        subset=[0, 1, 2, 3],
        **{"na.action": "omit"},
    )

    assert result.concordance == pytest.approx(
        survival.concordance_index([1.0, 3.0, 4.0], [1, 0, 1], [5.0, 3.0, 2.0])
    )
    assert dotted.concordance == pytest.approx(result.concordance)
    assert result.n == 3
    assert result.n_event == 2
    assert dotted.n == result.n
    assert dotted.n_event == result.n_event


def test_concordance_formula_adds_offset_to_risk_score():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.2, 0.3, 0.4, 0.5],
        "offset": [1.0, 0.0, 0.0, -1.0],
        "bonus": [0.1, -0.2, 0.3, -0.4],
    }
    expected_scores = [
        score + offset for score, offset in zip(data["score"], data["offset"], strict=True)
    ]
    expected_arithmetic_scores = [
        score + offset + bonus
        for score, offset, bonus in zip(data["score"], data["offset"], data["bonus"], strict=True)
    ]

    result = survival.concordance("Surv(time, status) ~ score + offset(offset)", data=data)
    arithmetic_result = survival.concordance(
        "Surv(time, status) ~ score + offset(offset + bonus)",
        data=data,
    )

    assert result.concordance == pytest.approx(
        survival.concordance_index(data["time"], data["status"], expected_scores)
    )
    assert arithmetic_result.concordance == pytest.approx(
        survival.concordance_index(data["time"], data["status"], expected_arithmetic_scores)
    )
    assert result.n_event == sum(data["status"])
    assert arithmetic_result.n_event == result.n_event


def test_concordance_formula_accepts_offset_only_predictor():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
        "offset": [1.0, 0.5, 0.2, 0.0],
    }

    result = survival.concordance("Surv(time, status) ~ offset(offset)", data=data)

    assert result.concordance == pytest.approx(
        survival.concordance_index(data["time"], data["status"], data["offset"])
    )
    assert result.n == len(data["time"])


def test_concordance_formula_returns_one_result_per_score_column():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 0, 1, 1],
        "x1": [0.8, 0.6, 0.4, 0.2, 0.1],
        "x2": [0.1, 0.5, 0.2, 0.7, 0.3],
    }
    result = survival.concordance("Surv(time, status) ~ x1 + x2", data=data)
    x1 = survival.core.concordance_summary(data["time"], data["status"], data["x1"])
    x2 = survival.core.concordance_summary(data["time"], data["status"], data["x2"])

    assert result.score_names == ["x1", "x2"]
    assert result.concordance == pytest.approx([x1["concordance"], x2["concordance"]])
    assert result.concordant == pytest.approx([x1["concordant"], x2["concordant"]])
    assert result.comparable == pytest.approx([x1["comparable"], x2["comparable"]])
    assert result.n == len(data["time"])
    assert result.n_event == sum(data["status"])


def test_concordance_matrix_scores_return_parallel_cluster_variances():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 1, 0],
        "x1": [0.8, 0.6, 0.4, 0.2, 0.1, 0.3],
        "x2": [0.1, 0.5, 0.2, 0.7, 0.3, 0.9],
        "wt": [1.0, 2.0, 1.0, 0.5, 1.5, 1.0],
        "cluster": ["a", "a", "b", "b", "c", "c"],
    }
    response = survival.Surv(data["time"], data["status"])
    scores = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]

    result = survival.concordance(
        response,
        scores=scores,
        weights=data["wt"],
        cluster=data["cluster"],
    )
    x1 = survival.concordance(
        response,
        scores=data["x1"],
        weights=data["wt"],
        cluster=data["cluster"],
    )
    x2 = survival.concordance(
        response,
        scores=data["x2"],
        weights=data["wt"],
        cluster=data["cluster"],
    )

    assert result.score_names == ["score1", "score2"]
    assert result.concordance == pytest.approx([x1.concordance, x2.concordance])
    assert result.concordant == pytest.approx([x1.concordant, x2.concordant])
    assert result.comparable == pytest.approx([x1.comparable, x2.comparable])
    assert result.variance == pytest.approx([x1.variance, x2.variance])


def test_concordance_summary_low_level_reports_pair_counts():
    data = _toy_data()
    scores = [8.0 - value for value in data["time"]]
    concordant, comparable = _manual_right_concordance_counts(
        data["time"],
        data["status"],
        scores,
    )

    summary = survival.core.concordance_summary(data["time"], data["status"], scores)

    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert summary["concordance"] == pytest.approx(concordant / comparable)
    assert summary["concordance"] == pytest.approx(
        survival.core.concordance_index(data["time"], data["status"], scores)
    )


def test_concordance_formula_accepts_strata_wrapper():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 1],
        "score": [2.0, 1.0, 100.0, 99.0],
        "group": ["A", "A", "B", "B"],
    }

    stratified = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
    )
    collapsed = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=False,
    )
    thresholded = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=0,
    )
    retained = survival.concordance(
        "Surv(time, status) ~ score + strata(group)",
        data=data,
        keepstrata=True,
    )
    unstratified = survival.concordance("Surv(time, status) ~ score", data=data)

    assert stratified.concordance == pytest.approx(1.0)
    assert collapsed.concordance == pytest.approx(stratified.concordance)
    assert thresholded.concordance == pytest.approx(stratified.concordance)
    assert retained.concordance == pytest.approx(stratified.concordance)
    assert unstratified.concordance < stratified.concordance
    assert stratified.n == len(data["time"])
    assert stratified.n_event == sum(data["status"])


def test_concordance_counting_process_uses_delayed_entry_risk_sets():
    data = _counting_cox_data()
    scores = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    response = survival.Surv(data["start"], data["stop"], data["status"])

    result = survival.concordance(response, scores=scores)
    expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    reversed_result = survival.concordance(response, scores=scores, reverse=True)
    reversed_expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        [-value for value in scores],
    )

    assert result.concordance == pytest.approx(expected)
    assert result.n == len(data["stop"])
    assert result.n_event == sum(data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    assert result.concordant == pytest.approx(concordant)
    assert result.comparable == pytest.approx(comparable)
    assert reversed_result.concordance == pytest.approx(reversed_expected)
    assert reversed_result.reverse is True


def test_concordance_counting_process_ranks_use_delayed_entry_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.5, 1.5],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
    }
    scores = [0.9, 0.7, 0.4, 0.1]
    response = survival.Surv(data["start"], data["stop"], data["status"])

    ranked = survival.concordance(response, scores=scores, ranks=True)
    exact_ranked = survival.concordance(response, scores=scores, ranks=True, timefix=False)
    formula_ranked = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data={**data, "score": scores},
        ranks=True,
    )

    assert ranked.ranks is not None
    assert exact_ranked.ranks is not None
    assert formula_ranked.ranks is not None
    assert [row["time"] for row in ranked.ranks] == pytest.approx(
        sorted(data["stop"][idx] for idx, event in enumerate(data["status"]) if event == 1)
    )
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in ranked.ranks) == (
        pytest.approx(2.0 * ranked.concordant - ranked.comparable)
    )
    assert sum(row["rank"] * row["timewt"] * row["casewt"] for row in exact_ranked.ranks) == (
        pytest.approx(2.0 * exact_ranked.concordant - exact_ranked.comparable)
    )
    for formula_row, direct_row in zip(formula_ranked.ranks, ranked.ranks, strict=True):
        assert formula_row.keys() == direct_row.keys()
        for key in direct_row:
            assert formula_row[key] == pytest.approx(direct_row[key])


def test_concordance_counting_process_influence_uses_delayed_entry_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.5, 1.5],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.9, 0.1, 0.4, 0.2],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    result = survival.concordance(response, scores=data["score"], influence=3)
    exact = survival.concordance(response, scores=data["score"], influence=3, timefix=False)
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        influence=3,
    )

    assert result.dfbeta is not None
    assert result.influence is not None
    assert exact.dfbeta is not None
    assert exact.influence is not None
    assert formula.dfbeta == pytest.approx(result.dfbeta)
    assert formula.influence is not None
    assert result.influence is not None
    for formula_row, result_row in zip(formula.influence, result.influence, strict=True):
        assert formula_row == pytest.approx(result_row)
    assert [sum(row[col] for row in result.influence) for col in range(5)] == pytest.approx(
        [result.concordant, result.comparable - result.concordant, 0.0, 0.0, 0.0]
    )
    assert result.variance == pytest.approx(sum(value * value for value in result.dfbeta))
    assert exact.variance == pytest.approx(sum(value * value for value in exact.dfbeta))


def test_concordance_counting_process_timefix_false_uses_exact_risk_sets():
    data = {
        "start": [0.0, 0.0, 0.0],
        "stop": [1.0, 1.0 + 5e-10, 2.0],
        "status": [1, 0, 0],
        "score": [0.5, 0.9, 0.1],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])

    default = survival.concordance(response, scores=data["score"])
    exact = survival.concordance(response, scores=data["score"], timefix=False)
    exact_formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        timefix=False,
    )
    fixed_expected = _manual_counting_concordance(
        data["start"],
        [1.0, 1.0, 2.0],
        data["status"],
        data["score"],
    )
    exact_expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
    )

    assert default.concordance == pytest.approx(fixed_expected)
    assert exact.concordance == pytest.approx(exact_expected)
    assert exact_formula.concordance == pytest.approx(exact.concordance)
    assert default.concordance != pytest.approx(exact.concordance)


def test_counting_concordance_low_level_matches_manual_risk_sets():
    data = _counting_cox_data()
    scores = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )

    result = survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        scores,
    )
    expected = concordant / comparable

    assert result == pytest.approx(expected)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert summary["concordance"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="same length"):
        survival.counting_concordance_index([0.0], [1.0, 2.0], [1], [0.5])


def test_counting_concordance_accepts_case_weights():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    data["wt"] = [2.0, 1.0, 3.0, 1.5, 0.5, 4.0]
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
    )
    expected = concordant / comparable

    direct = survival.concordance(response, scores=data["score"], weights=data["wt"])
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
    )
    low_level = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
    )

    assert direct.concordance == pytest.approx(expected)
    assert formula.concordance == pytest.approx(expected)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert low_level["concordance"] == pytest.approx(expected)
    assert low_level["concordant"] == pytest.approx(concordant)
    assert low_level["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
    ) == pytest.approx(expected)


def test_counting_concordance_accepts_identity_time_weight():
    data = {
        "start": [0.0, 0.0, 0.0, 1.0],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="I",
    )
    default = survival.concordance(response, scores=data["score"], weights=data["wt"])
    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="I",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        timewt="I",
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="I",
    ) == pytest.approx(direct.concordance)
    assert direct.concordance != pytest.approx(default.concordance)


def test_counting_concordance_accepts_survival_time_weight():
    data = {
        "start": [0.0, 0.0, 0.0, 1.0],
        "stop": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0],
        "wt": [2.0, 1.0, 3.0, 1.0],
    }
    response = survival.Surv(data["start"], data["stop"], data["status"])
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        data["wt"],
        timewt="S",
    )

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        timewt="S",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        timewt="S",
    )
    summary = survival.core.counting_concordance_summary(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert summary["concordance"] == pytest.approx(direct.concordance)
    assert summary["concordant"] == pytest.approx(concordant)
    assert summary["comparable"] == pytest.approx(comparable)
    assert survival.counting_concordance_index(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
        weights=data["wt"],
        timewt="S",
    ) == pytest.approx(direct.concordance)


def test_counting_concordance_accepts_time_window_restrictions():
    data = {
        "start": [0.0, 0.0, 0.5, 1.0, 2.5],
        "stop": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [1, 1, 1, 1, 0],
        "score": [0.1, 0.9, 0.8, 0.0, 0.4],
        "wt": [2.0, 1.0, 3.0, 1.0, 0.5],
    }
    ymin = 1.5
    ymax = 3.0
    bounded_stop, bounded_status = _manual_concordance_bounded_times_and_status(
        data["stop"],
        data["status"],
        ymin=ymin,
        ymax=ymax,
    )
    concordant, comparable = _manual_counting_concordance_counts(
        data["start"],
        bounded_stop,
        bounded_status,
        data["score"],
        data["wt"],
        timewt="S",
    )
    response = survival.Surv(data["start"], data["stop"], data["status"])

    direct = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )
    exact = survival.concordance(
        response,
        scores=data["score"],
        weights=data["wt"],
        ymin=ymin,
        ymax=ymax,
        timewt="S",
        timefix=False,
    )
    formula = survival.concordance(
        "Surv(start, stop, status) ~ score",
        data=data,
        weights="wt",
        ymin=ymin,
        ymax=ymax,
        timewt="S",
    )

    assert direct.concordance == pytest.approx(concordant / comparable)
    assert direct.concordant == pytest.approx(concordant)
    assert direct.comparable == pytest.approx(comparable)
    assert direct.n_event == sum(bounded_status)
    assert exact.concordance == pytest.approx(direct.concordance)
    assert formula.concordance == pytest.approx(direct.concordance)
    assert formula.n_event == direct.n_event


def test_concordance_formula_strata_supports_counting_process_response():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    data["group"] = ["A", "A", "A", "B", "B", "B"]

    result = survival.concordance(
        "Surv(start, stop, status) ~ score + strata(group)",
        data=data,
    )
    total_concordant = 0.0
    total_comparable = 0.0
    for group in ("A", "B"):
        indices = [idx for idx, value in enumerate(data["group"]) if value == group]
        concordant, comparable = _manual_counting_concordance_counts(
            [data["start"][idx] for idx in indices],
            [data["stop"][idx] for idx in indices],
            [data["status"][idx] for idx in indices],
            [data["score"][idx] for idx in indices],
        )
        total_concordant += concordant
        total_comparable += comparable

    assert result.concordance == pytest.approx(total_concordant / total_comparable)
    assert result.concordant == pytest.approx(total_concordant)
    assert result.comparable == pytest.approx(total_comparable)
    assert result.n_event == sum(data["status"])


def test_concordance_formula_accepts_counting_process_response():
    data = _counting_cox_data()
    data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]

    result = survival.concordance("Surv(start, stop, status) ~ score", data=data)
    expected = _manual_counting_concordance(
        data["start"],
        data["stop"],
        data["status"],
        data["score"],
    )

    assert result.concordance == pytest.approx(expected)
    assert result.n_event == sum(data["status"])


def test_low_level_concordance_remains_available_from_core_module():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = [1, 2, 1, 2, 1]
    wt = [1.0, 1.0, 1.0, 1.0, 1.0]
    timewt = [1.0, 1.0, 1.0, 1.0, 1.0]
    sortstart = None
    sortstop = [0, 1, 2, 3, 4]

    result = survival.core.concordance(y, x, wt, timewt, sortstart, sortstop)
    assert isinstance(result, dict)
    assert "count" in result


def test_survdiff_counting_process_uses_delayed_entry():
    data = {
        "start": [0.0, 0.0, 1.0, 2.0],
        "stop": [2.0, 4.0, 3.0, 5.0],
        "status": [1, 0, 1, 1],
        "group": ["treated", "control", "treated", "control"],
    }
    groups = [0, 1, 0, 1]

    direct = survival.survdiff(
        survival.Surv(data["start"], data["stop"], data["status"]),
        group=data["group"],
    )
    formula = survival.survdiff("Surv(start, stop, status) ~ group", data=data)
    low_level = survival.logrank_test(
        data["stop"],
        data["status"],
        groups,
        entry_times=data["start"],
    )
    fh = survival.survdiff("Surv(start, stop, status) ~ group", data=data, rho=0.5)
    fh_low_level = survival.fleming_harrington_test(
        data["stop"],
        data["status"],
        groups,
        0.5,
        0.0,
        entry_times=data["start"],
    )

    assert direct.statistic == pytest.approx(2.25)
    assert direct.observed == pytest.approx([2.0, 1.0])
    assert direct.expected == pytest.approx([1.0, 2.0])
    assert direct.variance == pytest.approx(4.0 / 9.0)
    assert formula.statistic == pytest.approx(direct.statistic)
    assert formula.observed == pytest.approx(direct.observed)
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
    observed, expected, variance, statistic = manual_exact()
    fh_observed, fh_expected, fh_variance, fh_statistic = manual_exact(0.5)

    assert exact.observed == pytest.approx(observed)
    assert exact.expected == pytest.approx(expected)
    assert exact.variance == pytest.approx(variance)
    assert exact.statistic == pytest.approx(statistic)
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
        "start": [0.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        "stop": [2.0, 4.0, 3.0, 5.0, 2.5, 4.5],
        "status": [1, 0, 1, 1, 1, 0],
        "group": ["treated", "control", "treated", "control", "treated", "control"],
        "site": ["x", "x", "x", "x", "y", "y"],
    }
    group_codes = [0 if value == "treated" else 1 for value in data["group"]]

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


def test_survdiff_timefix_false_uses_exact_event_times():
    times = [1.0, 1.0 + 5e-10, 2.0, 3.0]
    status = [0, 0, 1, 1]
    groups = ["control", "control", "control", "treated"]
    response = survival.Surv(times, status)

    default = survival.survdiff(response, group=groups)
    exact = survival.survdiff(response, group=groups, timefix=False)
    exact_formula = survival.survdiff(
        "Surv(time, status) ~ group",
        data={"time": times, "status": status, "group": groups},
        **{"time.fix": False},
    )
    low_level = survival.survdiff2(times, status, [1, 1, 1, 2], None, None)

    assert default.statistic == pytest.approx(1.0)
    assert exact.statistic == pytest.approx(low_level.chi_squared)
    assert exact.statistic == pytest.approx(0.5)
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
        "time": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "status": [1, 0, 1, 0, 1, 1],
        "group": ["treated", "control", "treated", "treated", "control", "control"],
        "site": ["north", "north", "north", "south", "south", "south"],
    }
    group_codes = {"treated": 1, "control": 2}
    order = sorted(
        range(len(data["time"])), key=lambda idx: (data["site"][idx], data["time"][idx], idx)
    )
    markers = [
        int(idx + 1 == len(order) or data["site"][order[idx + 1]] != data["site"][order[idx]])
        for idx in range(len(order))
    ]

    result = survival.survdiff("Surv(time, status) ~ group + strata(site)", data=data)
    low_level = survival.survdiff2(
        [data["time"][idx] for idx in order],
        [data["status"][idx] for idx in order],
        [group_codes[data["group"][idx]] for idx in order],
        markers,
        None,
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
    weighted_low_level = survival.survdiff2(
        [data["time"][idx] for idx in order],
        [data["status"][idx] for idx in order],
        [group_codes[data["group"][idx]] for idx in order],
        markers,
        0.5,
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
    fixed_times = [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0]
    order = sorted(
        range(len(data["time"])),
        key=lambda idx: (data["site"][idx], data["time"][idx], idx),
    )
    markers = [
        int(idx + 1 == len(order) or data["site"][order[idx + 1]] != data["site"][order[idx]])
        for idx in range(len(order))
    ]

    default = survival.survdiff("Surv(time, status) ~ group + strata(site)", data=data)
    exact = survival.survdiff(
        "Surv(time, status) ~ group + strata(site)",
        data=data,
        timefix=False,
    )
    default_low_level = survival.survdiff2(
        [fixed_times[idx] for idx in order],
        [data["status"][idx] for idx in order],
        [group_codes[data["group"][idx]] for idx in order],
        markers,
        None,
    )
    exact_low_level = survival.survdiff2(
        [data["time"][idx] for idx in order],
        [data["status"][idx] for idx in order],
        [group_codes[data["group"][idx]] for idx in order],
        markers,
        None,
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
    result = survival.survdiff("Surv(time, status) ~ factor(dose):sqrt(x1)", data=data)
    direct = survival.survdiff(
        survival.Surv(data["time"], data["status"]),
        group=[(data["dose"][idx], math.sqrt(data["x1"][idx])) for idx in range(len(data["time"]))],
    )

    assert result.statistic == pytest.approx(direct.statistic)
    assert result.observed == pytest.approx(direct.observed)


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


def test_coxph_formula_returns_fitted_cox_model():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10)
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        max_iter=10,
    )

    assert sum(len(row) for row in fit.coefficients) == 2
    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert len(fit.risk_scores) == 8
    assert len(fit.predict([[0.5, 0.8]])) == 1


def test_coxph_formula_cluster_computes_robust_variance():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    clustered = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    plain = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    explicit_cluster = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    id_cluster = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        id=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    id_model = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        id=data["subject"],
        model=True,
        max_iter=10,
        eps=1e-5,
    )
    matrix_id = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        id=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    expected_robust = _manual_cox_robust_variance(plain, data["subject"])

    assert clustered.robust is True
    assert clustered.cluster == data["subject"]
    assert id_cluster.robust is True
    assert id_cluster.id == data["subject"]
    assert id_cluster.cluster == data["subject"]
    assert id_model.model["(id)"] == data["subject"]
    assert clustered.coefficients[0] == pytest.approx(plain.coefficients[0])
    assert id_cluster.coefficients[0] == pytest.approx(plain.coefficients[0])
    assert len(clustered.covariates[0]) == 2
    for actual, expected in zip(
        clustered.naive_information_matrix, plain.information_matrix, strict=True
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.naive_var, plain.information_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(explicit_cluster.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(id_cluster.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(matrix_id.information_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)

    score = clustered.score_residuals()
    expected_dfbeta = [
        [
            sum(expected_robust[col_idx][inner_idx] * row[inner_idx] for inner_idx in range(2))
            for col_idx in range(2)
        ]
        for row in score
    ]
    expected_dfbetas = [
        [
            row[col_idx]
            / max(
                math.sqrt(abs(expected_robust[col_idx][col_idx])),
                survival.r_api._COX_DFBETAS_SCALE_FLOOR,
            )
            for col_idx in range(2)
        ]
        for row in expected_dfbeta
    ]
    for actual, expected in zip(clustered.dfbeta(), expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(clustered.dfbetas(), expected_dfbetas, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.r_api.residuals(clustered, type="dfbeta", weighted=False),
        expected_dfbeta,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.r_api.residuals(clustered, type="dfbetas", weighted=False),
        expected_dfbetas,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert clustered.dfbeta()[3] != pytest.approx(clustered.fit.dfbeta()[3])

    row = [0.5, 0.8]
    robust_prediction = survival.predict(clustered, [row], reference="zero", se_fit=True)
    expected_se = math.sqrt(
        max(
            sum(
                row[left] * expected_robust[left][right] * row[right]
                for left in range(2)
                for right in range(2)
            ),
            0.0,
        )
    )

    assert robust_prediction.fit == pytest.approx(plain.predict([row]))
    assert robust_prediction.se_fit == pytest.approx([expected_se])


def test_model_generic_helpers_report_core_fit_metadata():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    fixed_scale_aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        scale=1.0,
        max_iter=10,
        eps=1e-5,
    )
    matrix_cox = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[value] for value in data["x1"]],
        max_iter=0,
    )
    weighted_matrix_cox = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[value] for value in data["x1"]],
        weights=[1.0] * len(data["time"]),
        max_iter=0,
    )
    weighted_cox = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        weights=data["x2"],
        max_iter=0,
    )
    weighted_aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        weights=data["x2"],
        max_iter=1,
    )

    assert survival.coef(cox) == pytest.approx(cox.coefficients[0])
    for actual, expected in zip(survival.vcov(cox), cox.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.loglik(cox) == pytest.approx(cox.log_likelihood[-1])
    assert survival.nobs(cox) == len(data["time"])
    assert survival.degrees_freedom(cox) == len(cox.coefficients[0])
    assert survival.aic(cox) == pytest.approx(
        -2.0 * survival.loglik(cox) + 2.0 * survival.degrees_freedom(cox)
    )
    assert survival.aic(cox, k=4) == pytest.approx(
        -2.0 * survival.loglik(cox) + 4.0 * survival.degrees_freedom(cox)
    )
    assert survival.bic(cox) == pytest.approx(
        -2.0 * survival.loglik(cox) + math.log(survival.nobs(cox)) * survival.degrees_freedom(cox)
    )
    assert survival.extract_aic(cox) == pytest.approx(
        [survival.degrees_freedom(cox), survival.aic(cox)]
    )
    assert survival.model_formula(cox) == "Surv(time, status) ~ x1 + x2 + cluster(group)"
    assert survival.model_weights(cox) is None
    assert survival.model_weights(weighted_matrix_cox) == pytest.approx([1.0] * len(data["time"]))
    assert survival.model_weights(weighted_cox) == pytest.approx(data["x2"])
    cox_matrix = survival.model_matrix(cox)
    assert cox_matrix["columns"] == ["x1", "x2"]
    for actual, expected in zip(cox_matrix["data"], cox.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.fitted(cox) == pytest.approx(survival.predict(cox))
    assert survival.fitted(cox, type="risk") == pytest.approx(survival.predict(cox, type="risk"))
    cox_fitted_with_se = survival.fitted(cox, **{"se.fit": True})
    assert cox_fitted_with_se.fit == pytest.approx(survival.fitted(cox))
    assert len(cox_fitted_with_se.se_fit) == len(data["time"])

    assert survival.coef(aft) == pytest.approx(aft.location_coefficients)
    for actual, expected in zip(survival.vcov(aft), aft.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        survival.vcov(aft, complete=False),
        [row[: aft.n_covariates] for row in aft.variance_matrix[: aft.n_covariates]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_aft_loglik = aft.log_likelihood - sum(
        math.log(time)
        for time, event in zip(data["time"], data["status"], strict=True)
        if event == 1
    )
    assert survival.loglik(aft) == pytest.approx(expected_aft_loglik)
    assert survival.nobs(aft) == len(data["time"])
    assert survival.degrees_freedom(aft) == len(aft.coefficients)
    assert survival.df_residual(aft) == survival.nobs(aft) - survival.degrees_freedom(aft)
    assert survival.degrees_freedom(fixed_scale_aft) == len(fixed_scale_aft.location_coefficients)
    assert survival.aic(aft) == pytest.approx(
        -2.0 * survival.loglik(aft) + 2.0 * survival.degrees_freedom(aft)
    )
    assert survival.extract_aic(aft, k=3) == pytest.approx(
        [survival.degrees_freedom(aft), survival.aic(aft, k=3)]
    )
    assert survival.model_formula(aft) == "Surv(time, status) ~ x1 + x2"
    assert survival.model_weights(aft) is None
    assert survival.model_weights(weighted_aft) == pytest.approx(data["x2"])
    aft_matrix = survival.model_matrix(aft)
    assert aft_matrix["columns"] == ["(Intercept)", "x1", "x2"]
    for actual, expected in zip(aft_matrix["data"], aft.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.fitted(aft) == pytest.approx(survival.predict(aft))
    assert survival.fitted(aft, type="lp") == pytest.approx(survival.predict(aft, type="lp"))

    with pytest.raises(TypeError, match="fitted coxph or survreg"):
        survival.coef(survival.Surv([1.0, 2.0], [1, 0]))
    with pytest.raises(TypeError, match="unexpected keyword"):
        survival.fitted(cox, newdata={"x1": [0.2], "x2": [0.8]})
    with pytest.raises(ValueError, match="finite"):
        survival.aic(cox, k=float("nan"))
    with pytest.raises(TypeError, match="formula-based"):
        survival.model_formula(matrix_cox)
    with pytest.raises(TypeError, match="fitted coxph or survreg"):
        survival.model_weights(survival.Surv([1.0, 2.0], [1, 0]))
    with pytest.raises(TypeError, match="survreg"):
        survival.df_residual(cox)
    with pytest.raises(TypeError, match="model=True"):
        survival.model_frame(cox)


def test_model_frame_returns_stored_formula_columns():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + offset(offset) + strata(group)",
        data=data,
        model=True,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        model=True,
        max_iter=10,
        eps=1e-5,
    )

    cox_frame = survival.model_frame(cox)
    assert {"time", "status", "x1", "x2", "offset", "group", "(offset)", "(strata)"} <= set(
        cox_frame
    )
    assert cox_frame["time"] == pytest.approx(data["time"])
    assert cox_frame["status"] == list(data["status"])
    assert cox_frame["x1"] == pytest.approx(data["x1"])
    assert cox_frame["(strata)"] == data["group"]

    aft_frame = survival.model_frame(aft)
    assert {"time", "status", "x1", "x2"} <= set(aft_frame)
    assert aft_frame["x2"] == pytest.approx(data["x2"])


def test_formula_fit_iteration_counts_accept_integer_valued_floats():
    data = _toy_data()

    cox = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=0.0)
    aft = survival.survreg("Surv(time, status) ~ x1", data=data, max_iter=1.0)

    assert len(cox.coefficients[0]) == 1
    assert len(aft.location_coefficients) == 2
    with pytest.raises(ValueError, match="integer"):
        survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=1.5)


def test_model_generic_helpers_report_formula_coefficient_names():
    data = _factor_data()
    data["group"] = ["A", "A", "B", "B", "C", "C", "A", "B"]
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + factor(group) + x1:x2",
        data=data,
        max_iter=0,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert survival.coef_names(cox) == ["x1", "groupB", "groupC", "x1:x2"]
    assert survival.coef_names(aft) == ["(Intercept)", "x1", "x2"]
    assert survival.coef_names(aft, complete=True) == [
        "(Intercept)",
        "x1",
        "x2",
        "Log(scale:A)",
        "Log(scale:B)",
        "Log(scale:C)",
    ]
    assert len(survival.coef_names(aft, complete=True)) == len(aft.coefficients)
    assert len(survival.coef_names(aft)) == len(survival.coef(aft))


def test_model_summary_reports_named_coefficient_table():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    cox_summary = survival.model_summary(cox)
    aft_summary = survival.model_summary(aft)

    assert cox_summary["model_type"] == "coxph"
    assert cox_summary["robust"] is True
    assert cox_summary["n"] == len(data["time"])
    assert cox_summary["n_event"] == sum(data["status"])
    assert cox_summary["df"] == 2
    assert cox_summary["loglik"] == pytest.approx(cox.log_likelihood[-1])
    assert cox_summary["null_loglik"] == pytest.approx(cox.log_likelihood[0])
    assert [row["name"] for row in cox_summary["coefficients"]] == ["x1", "x2"]
    for idx, (row, coefficient, variance_row) in enumerate(
        zip(
            cox_summary["coefficients"],
            survival.coef(cox),
            survival.vcov(cox),
            strict=True,
        )
    ):
        assert row["coef"] == pytest.approx(coefficient)
        assert row["se"] == pytest.approx(math.sqrt(max(variance_row[idx], 0.0)))
        assert row["statistic"] == pytest.approx(row["coef"] / row["se"])
        assert 0.0 <= row["p"] <= 1.0

    assert aft_summary["model_type"] == "survreg"
    assert aft_summary["n"] == len(data["time"])
    assert aft_summary["df"] == len(aft.coefficients)
    assert aft_summary["scale"] == pytest.approx(aft.scale)
    assert aft_summary["scales"] == pytest.approx(aft.scales)
    assert [row["name"] for row in aft_summary["coefficients"]] == ["(Intercept)", "x1", "x2"]
    for row, coefficient in zip(
        aft_summary["coefficients"], aft.location_coefficients, strict=True
    ):
        assert row["coef"] == pytest.approx(coefficient)
        assert 0.0 <= row["p"] <= 1.0


def test_model_confint_reports_named_coefficient_intervals():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    aft = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    z = NormalDist().inv_cdf(0.95)

    intervals = survival.confint(cox, level=0.9)

    assert [row["name"] for row in intervals] == ["x1", "x2"]
    for idx, row in enumerate(intervals):
        coefficient = survival.coef(cox)[idx]
        se = math.sqrt(max(survival.vcov(cox)[idx][idx], 0.0))
        assert row["lower"] == pytest.approx(coefficient - z * se)
        assert row["upper"] == pytest.approx(coefficient + z * se)

    assert survival.confint(cox, parm="x2")[0]["name"] == "x2"
    assert [row["name"] for row in survival.confint(aft, parm=[1, "x2"])] == [
        "(Intercept)",
        "x2",
    ]
    with pytest.raises(ValueError, match="level"):
        survival.confint(cox, level=1.5)
    with pytest.raises(ValueError, match="unknown coefficient"):
        survival.confint(cox, parm="missing")
    with pytest.raises(IndexError, match="parm index"):
        survival.confint(cox, parm=0)
    with pytest.raises(TypeError, match="parm"):
        survival.confint(cox, parm=1.5)


def test_as_data_frame_returns_r_friendly_result_tables():
    data = _toy_data()
    cox = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    surv_table = survival.as_data_frame(survival.Surv(data["time"], data["status"]))
    assert set(surv_table) == {"time", "status", "type"}
    assert surv_table["time"] == pytest.approx(data["time"])
    assert surv_table["status"] == data["status"]
    assert surv_table["type"] == ["right"] * len(data["time"])

    counting_table = survival.as_data_frame(survival.Surv([0.0, 1.0], [2.0, 3.0], [1, 0]))
    assert set(counting_table) == {"start", "stop", "status", "type"}
    assert counting_table["start"] == pytest.approx([0.0, 1.0])
    assert counting_table["stop"] == pytest.approx([2.0, 3.0])
    assert counting_table["type"] == ["counting", "counting"]

    interval_table = survival.as_data_frame(
        survival.Surv([1.0, 2.0], [2.0, 4.0], [3, 0], type="interval")
    )
    assert set(interval_table) == {"time", "status", "time2", "type"}
    assert interval_table["time2"] == pytest.approx([2.0, 4.0])
    assert interval_table["type"] == ["interval", "interval"]

    km_table = survival.as_data_frame(survival.survfit("Surv(time, status) ~ group", data=data))
    assert {"strata", "time", "n.risk", "n.event", "surv", "cumhaz"} <= set(km_table)
    assert len(km_table["strata"]) == len(km_table["time"])
    assert set(km_table["strata"]) == {"A", "B"}

    zero_survival_table = survival.as_data_frame(
        survival.survfit(survival.Surv([1.0, 2.0], [1, 1]))
    )
    assert zero_survival_table["surv"][-1] == pytest.approx(0.0)
    assert math.isnan(zero_survival_table["std.err"][-1])
    assert math.isnan(zero_survival_table["lower"][-1])
    assert math.isnan(zero_survival_table["upper"][-1])

    cox_surv_table = survival.as_data_frame(
        survival.survfit(cox, newdata={"x1": [0.2, 1.0], "x2": [1.0, 0.4]})
    )
    assert {"curve", "time", "surv", "cumhaz", "linear.predictor"} <= set(cox_surv_table)
    assert set(cox_surv_table["curve"]) == {1, 2}
    assert len(cox_surv_table["surv"]) == 2 * len(set(data["time"]))

    basehaz_table = survival.as_data_frame(survival.basehaz(cox))
    assert set(basehaz_table) == {"time", "cumhaz"}
    assert basehaz_table["time"] == sorted(set(data["time"]))

    survdiff_table = survival.as_data_frame(
        survival.survdiff("Surv(time, status) ~ group", data=data)
    )
    assert set(survdiff_table) == {"group", "observed", "expected", "variance"}
    assert survdiff_table["group"] == [1, 2]

    zph_table = survival.as_data_frame(survival.cox_zph(cox))
    assert {"name", "chisq", "df", "p"} <= set(zph_table)
    assert zph_table["name"][-1] == "GLOBAL"

    detail_table = survival.as_data_frame(survival.coxph_detail(cox))
    assert {"time", "n.event", "n.risk", "hazard", "cumhaz"} <= set(detail_table)
    assert len(detail_table["time"]) == sum(data["status"])

    anova_table = survival.as_data_frame(survival.anova(cox))
    assert set(anova_table) == {"model", "loglik", "df", "chisq", "p"}
    assert anova_table["model"][0] == "NULL"


def test_anova_coxph_single_formula_model_refits_terms_sequentially():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)
    first_term = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
    )

    result = survival.anova(fit)

    assert result.test_type == "Chisq"
    assert [row.model_name for row in result.rows] == ["NULL", "x1", "x2"]
    assert [row.df for row in result.rows] == [0, 1, 2]
    assert result.rows[0].loglik == pytest.approx(fit.log_likelihood[0])
    assert result.rows[1].loglik == pytest.approx(first_term.log_likelihood[-1])
    assert result.rows[2].loglik == pytest.approx(fit.log_likelihood[-1])
    assert result.rows[1].chisq == pytest.approx(
        2.0 * (result.rows[1].loglik - result.rows[0].loglik)
    )
    assert result.rows[2].chisq == pytest.approx(
        2.0 * (result.rows[2].loglik - result.rows[1].loglik)
    )
    assert 0.0 <= result.rows[1].p_value <= 1.0


def test_anova_coxph_single_formula_model_preserves_offsets():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2 + offset(offset)", data=data)
    first_term = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=data["offset"],
    )

    result = survival.anova(fit)

    assert [row.model_name for row in result.rows] == ["NULL", "x1", "x2"]
    assert result.rows[1].loglik == pytest.approx(first_term.log_likelihood[-1])
    assert result.rows[2].loglik == pytest.approx(fit.log_likelihood[-1])


def test_anova_coxph_compares_nested_models():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)

    result = survival.anova(fit_x1, fit_full)

    assert [row.model_name for row in result.rows] == ["Model 1", "Model 2"]
    assert [row.df for row in result.rows] == [1, 2]
    assert result.rows[0].loglik == pytest.approx(fit_x1.log_likelihood[-1])
    assert result.rows[1].loglik == pytest.approx(fit_full.log_likelihood[-1])
    assert result.rows[1].chisq == pytest.approx(
        2.0 * (fit_full.log_likelihood[-1] - fit_x1.log_likelihood[-1])
    )


def test_anova_coxph_accepts_r_style_test_aliases_and_prefixes():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)

    chisq = survival.anova(fit_x1, fit_full, test="ch")
    lrt = survival.anova(fit_x1, fit_full, test="likelihood-ratio")
    none = survival.anova(fit_x1, fit_full, test="n")

    assert chisq.test_type == "Chisq"
    assert chisq.rows[1].chisq == pytest.approx(
        2.0 * (fit_full.log_likelihood[-1] - fit_x1.log_likelihood[-1])
    )
    assert lrt.test_type == "LRT"
    assert lrt.rows[1].chisq == pytest.approx(chisq.rows[1].chisq)
    assert none.test_type == "none"
    assert none.rows[1].chisq is None
    assert none.rows[1].p_value is None

    with pytest.raises(ValueError, match="anova test"):
        survival.anova(fit_x1, fit_full, test="wald")


def test_anova_coxph_can_omit_tests_and_rejects_non_cox_models():
    data = _toy_data()
    fit_x1 = survival.coxph("Surv(time, status) ~ x1", data=data)
    fit_full = survival.coxph("Surv(time, status) ~ x1 + x2", data=data)
    without_tests = survival.anova(fit_x1, fit_full, test=None)

    assert without_tests.test_type == "none"
    assert without_tests.rows[0].chisq is None
    assert without_tests.rows[1].p_value is None

    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
    )
    with pytest.raises(TypeError, match="anova requires fitted Cox model objects"):
        survival.anova(aft)


def test_coxph_detail_exposes_r_style_event_contributions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        init=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )

    detail = survival.coxph_detail(fit)
    baseline_times, baseline_cumhaz = fit.basehaz(True)
    event_indices = [idx for idx, event in enumerate(data["status"]) if event == 1]

    assert isinstance(detail, survival.r_api.CoxPHDetailResult)
    assert detail.time == pytest.approx([data["time"][idx] for idx in event_indices])
    assert detail.times() == pytest.approx(detail.time)
    assert detail.nevent == [1] * len(event_indices)
    assert detail.n_event == detail.nevent
    assert detail.nrisk == [len(data["time"]) - idx for idx in event_indices]
    assert detail.n_risk_at_times() == detail.nrisk
    assert detail.cumulative_hazard == pytest.approx(baseline_cumhaz)
    assert detail.cumulative_hazards() == pytest.approx(baseline_cumhaz)
    assert detail.hazards() == pytest.approx(detail.hazard)
    expected_x = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]
    expected_y = [
        [time, float(status)] for time, status in zip(data["time"], data["status"], strict=True)
    ]
    assert detail.x == expected_x
    assert detail.y == expected_y
    for actual, expected in zip(detail.score, fit.schoenfeld_residuals(), strict=True):
        assert actual == pytest.approx(expected)
    assert [sum(row[col_idx] for row in detail.score) for col_idx in range(2)] == pytest.approx(
        fit.score_vector
    )
    assert baseline_times == pytest.approx(detail.time)
    assert all(value > 0.0 for value in detail.varhaz)


def test_coxph_detail_riskmat_honors_counting_entry_and_strata():
    data = _counting_cox_data() | {"group": ["A", "A", "A", "A", "B", "B"]}
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1 + strata(group)",
        data=data,
        init=[0.0],
        max_iter=0,
        method="breslow",
    )

    detail = survival.coxph_detail(fit, riskmat=True)

    assert detail.time == pytest.approx([2.0, 4.0, 5.0])
    assert detail.strata == {0: 2, 1: 1}
    assert detail.y[0] == pytest.approx([0.0, 2.0, 1.0])
    assert detail.riskmat is not None
    assert [row[0] for row in detail.riskmat] == [1, 1, 1, 0, 0, 0]
    assert [row[1] for row in detail.riskmat] == [0, 0, 1, 1, 0, 0]
    assert [row[2] for row in detail.riskmat] == [0, 0, 0, 0, 1, 1]

    time_order = survival.coxph_detail(fit, riskmat=True, rorder="time")
    time_prefix = survival.coxph_detail(fit, riskmat=True, rorder="t")
    data_prefix = survival.coxph_detail(fit, riskmat=True, rorder="d")
    assert time_order.sortorder == [0, 1, 2, 3, 4, 5]
    assert time_order.riskmat == detail.riskmat
    assert time_prefix.sortorder == time_order.sortorder
    assert time_prefix.riskmat == time_order.riskmat
    assert data_prefix.sortorder is None
    assert data_prefix.riskmat == detail.riskmat

    with pytest.raises(ValueError, match="rorder"):
        survival.coxph_detail(fit, rorder="event")


def test_coxph_detail_rejects_exact_ties_like_r():
    data = _tied_cox_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        init=[0.0],
        max_iter=0,
        method="exact",
    )

    with pytest.raises(ValueError, match="exact method"):
        survival.coxph_detail(fit)


def test_coxph_formula_accepts_counting_process_response():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    low_level = survival.regression.coxph_fit(
        data["stop"],
        data["status"],
        [[value] for value in data["x1"]],
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
        entry_times=data["start"],
    )

    assert fit.entry_times == pytest.approx(data["start"])
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.score_vector == pytest.approx(low_level.score_vector)

    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    raw_times, raw_hazard = survival.basehaz(
        data["stop"],
        data["status"],
        fit.linear_predictors,
        False,
        entry_times=data["start"],
    )
    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(raw_times, time)) == 0 else raw_hazard[pos - 1]
        for time in sorted(set(data["stop"]))
    ]
    assert fitted_times == pytest.approx(sorted(set(data["stop"])))
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)


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


def test_predict_coxph_r_style_generic_types():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[0.5, 0.8], [1.1, 0.3], [0.7, 0.9]]
    collapse = ["A", "A", "B"]

    lp = survival.predict(fit, rows)
    centered_lp = survival.predict(fit, rows, centered=True)
    risk = survival.predict(fit, rows, type="risk")
    prefix_lp = survival.predict(fit, rows, type="l")
    prefix_risk = survival.predict(fit, rows, type="r")
    uncentered_lp = survival.predict(fit, rows, reference="zero")
    terms = survival.predict(fit, rows, type="terms")
    prefix_terms = survival.predict(fit, rows, type="t")
    uncentered_terms = survival.predict(fit, rows, type="terms", reference="zero")
    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )

    assert lp == pytest.approx([value - center for value in direct_lp])
    assert prefix_lp == pytest.approx(lp)
    assert centered_lp == pytest.approx([value - center for value in direct_lp])
    assert uncentered_lp == pytest.approx(direct_lp)
    assert risk == pytest.approx([math.exp(value - center) for value in direct_lp])
    assert prefix_risk == pytest.approx(risk)
    expected_terms = [
        [
            (rows[row_idx][col_idx] - fit.means[col_idx]) * fit.coefficients[0][col_idx]
            for col_idx in range(2)
        ]
        for row_idx in range(len(rows))
    ]
    expected_uncentered_terms = [
        [rows[row_idx][col_idx] * fit.coefficients[0][col_idx] for col_idx in range(2)]
        for row_idx in range(len(rows))
    ]
    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(uncentered_terms, expected_uncentered_terms, strict=True):
        assert actual == pytest.approx(expected)
    assert survival.predict(fit, rows, reference="zero", collapse=collapse) == pytest.approx(
        [direct_lp[0] + direct_lp[1], direct_lp[2]]
    )
    variance = fit.information_matrix
    centered_rows = [[row[col_idx] - fit.means[col_idx] for col_idx in range(2)] for row in rows]
    expected_lp_se = [
        math.sqrt(
            max(
                sum(
                    centered_row[row_idx] * variance[row_idx][col_idx] * centered_row[col_idx]
                    for row_idx in range(2)
                    for col_idx in range(2)
                ),
                0.0,
            )
        )
        for centered_row in centered_rows
    ]
    lp_with_se = survival.predict(fit, rows, se_fit=True)
    dotted_lp_with_se = survival.predict(fit, rows, **{"se.fit": True})
    unpacked_lp, unpacked_se = lp_with_se
    assert isinstance(lp_with_se, survival.r_api.PredictResult)
    assert lp_with_se.fit == pytest.approx(lp)
    assert lp_with_se.predictions == pytest.approx(lp)
    assert lp_with_se.se_fit == pytest.approx(expected_lp_se)
    assert lp_with_se.se == pytest.approx(expected_lp_se)
    assert isinstance(dotted_lp_with_se, survival.r_api.PredictResult)
    assert dotted_lp_with_se.fit == pytest.approx(lp_with_se.fit)
    assert dotted_lp_with_se.se_fit == pytest.approx(lp_with_se.se_fit)
    assert unpacked_lp == pytest.approx(lp)
    assert unpacked_se == pytest.approx(expected_lp_se)

    risk_with_se = survival.predict(fit, rows, type="risk", se_fit=True)
    assert risk_with_se.fit == pytest.approx(risk)
    assert risk_with_se.se_fit == pytest.approx(
        [se * value for se, value in zip(expected_lp_se, risk, strict=True)]
    )
    zero_se = [
        math.sqrt(
            max(
                sum(
                    row[row_idx] * variance[row_idx][col_idx] * row[col_idx]
                    for row_idx in range(2)
                    for col_idx in range(2)
                ),
                0.0,
            )
        )
        for row in rows
    ]
    collapsed_lp_with_se = survival.predict(
        fit,
        rows,
        reference="zero",
        collapse=collapse,
        se_fit=True,
    )
    assert collapsed_lp_with_se.fit == pytest.approx([direct_lp[0] + direct_lp[1], direct_lp[2]])
    assert collapsed_lp_with_se.se_fit == pytest.approx(
        [math.sqrt(zero_se[0] ** 2 + zero_se[1] ** 2), zero_se[2]]
    )
    assert survival.predict(
        fit,
        rows,
        type="risk",
        reference="zero",
        collapse=collapse,
    ) == pytest.approx([math.exp(direct_lp[0]) + math.exp(direct_lp[1]), math.exp(direct_lp[2])])
    collapsed_terms = survival.predict(
        fit,
        rows,
        type="terms",
        reference="zero",
        collapse=collapse,
    )
    expected_collapsed_terms = [
        [
            expected_uncentered_terms[0][col_idx] + expected_uncentered_terms[1][col_idx]
            for col_idx in range(2)
        ],
        expected_uncentered_terms[2],
    ]
    for actual, expected in zip(collapsed_terms, expected_collapsed_terms, strict=True):
        assert actual == pytest.approx(expected)
    terms_with_se = survival.predict(
        fit,
        rows,
        type="terms",
        reference="zero",
        collapse=collapse,
        se_fit=True,
    )
    expected_term_se = [
        [
            abs(row[col_idx]) * math.sqrt(max(variance[col_idx][col_idx], 0.0))
            for col_idx in range(2)
        ]
        for row in rows
    ]
    expected_collapsed_term_se = [
        [
            math.sqrt(expected_term_se[0][col_idx] ** 2 + expected_term_se[1][col_idx] ** 2)
            for col_idx in range(2)
        ],
        expected_term_se[2],
    ]
    for actual, expected in zip(terms_with_se.fit, expected_collapsed_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(terms_with_se.se_fit, expected_collapsed_term_se, strict=True):
        assert actual == pytest.approx(expected)

    full_times, full_curves = fit.survival_curve(rows, True)
    full_survfit = survival.survfit(fit, newdata=rows, censor=False)
    requested_times = [0.5, full_times[0], full_times[-1]]
    times, curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    assert curves[0][0] == pytest.approx(1.0)
    assert curves[0][1] == pytest.approx(full_curves[0][0])
    assert curves[0][-1] == pytest.approx(full_curves[0][-1])
    collapsed_times, collapsed_curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        collapse=collapse,
    )
    expected_collapsed_curves = [
        [curves[0][idx] + curves[1][idx] for idx in range(len(requested_times))],
        curves[2],
    ]
    assert collapsed_times == pytest.approx(requested_times)
    for actual, expected in zip(collapsed_curves, expected_collapsed_curves, strict=True):
        assert actual == pytest.approx(expected)
    all_times, all_curves = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        collapse=collapse,
    )
    expected_all_curves = [
        [full_curves[0][idx] + full_curves[1][idx] for idx in range(len(full_times))],
        full_curves[2],
    ]
    assert all_times == pytest.approx(full_times)
    for actual, expected in zip(all_curves, expected_all_curves, strict=True):
        assert actual == pytest.approx(expected)
    survival_with_se = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        se_fit=True,
    )
    (fit_times, fit_curves), (se_times, se_curves) = survival_with_se
    expected_se_curves = []
    for std_err_curve in full_survfit.std_err:
        expected_se = []
        for time in requested_times:
            pos = bisect_right(full_survfit.time, time)
            expected_se.append(0.0 if pos == 0 else std_err_curve[pos - 1])
        expected_se_curves.append(expected_se)
    assert fit_times == pytest.approx(requested_times)
    assert se_times == pytest.approx(requested_times)
    for actual, expected in zip(fit_curves, curves, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(se_curves, expected_se_curves, strict=True):
        assert actual == pytest.approx(expected)
    collapsed_with_se = survival.predict(
        fit,
        rows,
        type="survival",
        centered=True,
        times=requested_times,
        collapse=collapse,
        se_fit=True,
    )
    collapsed_fit, collapsed_se = collapsed_with_se
    assert collapsed_fit[0] == pytest.approx(requested_times)
    assert collapsed_se[0] == pytest.approx(requested_times)
    for actual, expected in zip(collapsed_fit[1], expected_collapsed_curves, strict=True):
        assert actual == pytest.approx(expected)
    expected_collapsed_se = [
        [
            math.sqrt(expected_se_curves[0][idx] ** 2 + expected_se_curves[1][idx] ** 2)
            for idx in range(len(requested_times))
        ],
        expected_se_curves[2],
    ]
    for actual, expected in zip(collapsed_se[1], expected_collapsed_se, strict=True):
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="same length as predictions"):
        survival.predict(fit, rows, collapse=["A"])


def test_predict_coxph_formula_accepts_newdata_mapping():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[0.5, 0.8], [1.1, 0.3]]
    newdata = {"x1": [0.5, 1.1], "x2": [0.8, 0.3]}

    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )
    assert survival.predict(fit, newdata) == pytest.approx([value - center for value in direct_lp])
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(direct_lp)
    assert survival.predict(fit, newdata, type="risk") == pytest.approx(
        [math.exp(value - center) for value in direct_lp]
    )
    terms = survival.predict(fit, newdata, type="terms")
    expected_terms = [
        [(row[col_idx] - fit.means[col_idx]) * fit.coefficients[0][col_idx] for col_idx in range(2)]
        for row in rows
    ]
    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)


def test_predict_coxph_formula_newdata_mapping_uses_training_factor_levels():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ group + x1", data=data, max_iter=10, eps=1e-5)
    newdata = {"group": ["B", "A"], "x1": [0.5, 0.8]}
    rows = [[1.0, 0.5], [0.0, 0.8]]
    beta = fit.coefficients[0]
    center = fit.means[1] * beta[1]

    assert survival.predict(fit, newdata) == pytest.approx(
        [value - center for value in fit.predict(rows)]
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(fit.predict(rows))
    with pytest.raises(ValueError, match="unknown level"):
        survival.predict(fit, {"group": ["C"], "x1": [0.5]})


def test_predict_coxph_terms_groups_formula_terms_and_selects_by_name_or_index():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"dose": [0, 1, 2], "x1": [0.5, 0.8, 1.1]}
    beta = fit.coefficients[0]
    expected_terms = [
        [0.0, (0.5 - fit.means[2]) * beta[2]],
        [beta[0], (0.8 - fit.means[2]) * beta[2]],
        [beta[1], (1.1 - fit.means[2]) * beta[2]],
    ]

    terms = survival.predict(fit, newdata, type="terms")
    factor_terms = survival.predict(fit, newdata, type="terms", terms="factor(dose)")
    x1_terms = survival.predict(fit, newdata, type="terms", terms=[2])

    for actual, expected in zip(terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(factor_terms, [[row[0]] for row in expected_terms], strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(x1_terms, [[row[1]] for row in expected_terms], strict=True):
        assert actual == pytest.approx(expected)

    with pytest.raises(ValueError, match="unknown model term"):
        survival.predict(fit, newdata, type="terms", terms=["missing"])


def test_predict_coxph_terms_selects_matrix_fit_columns_with_one_based_indices():
    data = _toy_data()
    fit = survival.coxph(
        survival.Surv(data["time"], data["status"]),
        x=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    beta = fit.coefficients[0]

    selected_x2 = survival.predict(fit, rows, type="terms", terms=[2])
    selected_x1 = survival.predict(fit, rows, type="terms", terms="x1")
    for actual, expected in zip(
        selected_x2,
        [[(0.8 - fit.means[1]) * beta[1]], [(0.3 - fit.means[1]) * beta[1]]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        selected_x1,
        [[(0.5 - fit.means[0]) * beta[0]], [(1.1 - fit.means[0]) * beta[0]]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_predict_coxph_formula_newdata_mapping_rebuilds_transforms_and_interactions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ log(x1) + x1:x2 + group:x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.5, 1.0], "x2": [0.25, 0.8], "group": ["B", "A"]}
    rows = [
        [math.log(0.5), 0.5 * 0.25, 1.0 * 0.25],
        [math.log(1.0), 1.0 * 0.8, 0.0 * 0.8],
    ]

    direct_lp = fit.predict(rows)
    center = sum(
        value * coefficient
        for value, coefficient in zip(fit.means, fit.coefficients[0], strict=True)
    )
    assert survival.predict(fit, newdata) == pytest.approx([value - center for value in direct_lp])
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(direct_lp)


def test_predict_coxph_formula_offset_newdata_mapping_survival():
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
    full_curves = [
        [math.exp(-hazard * math.exp(lp - center)) for hazard in hazards]
        for lp in linear_predictors
    ]
    requested_times = [0.5, baseline_times[0], baseline_times[-1]]

    times, curves = survival.predict(
        fit,
        newdata,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    for actual, expected_curve in zip(curves, full_curves, strict=True):
        assert actual == pytest.approx([1.0, expected_curve[0], expected_curve[-1]])


def test_predict_coxph_formula_rebuilds_transformed_offsets_from_newdata():
    data = _toy_data()
    data["exposure"] = [math.exp(value) for value in data["offset"]]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "exposure": [math.exp(value) for value in offsets]}
    expected = [value + offset for value, offset in zip(fit.predict(rows), offsets, strict=True)]

    assert survival.predict(fit, newdata, reference="zero") == pytest.approx(expected)


def test_predict_expected_and_residuals_share_martingale_identity():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)

    expected = survival.predict(fit, type="expected")
    martingale = survival.r_api.residuals(fit)
    deviance = survival.r_api.residuals(fit, type="deviance")

    assert expected == pytest.approx(fit.expected_events())
    assert martingale == pytest.approx(fit.martingale_residuals())
    assert martingale == pytest.approx(
        [status - value for status, value in zip(fit.status, expected, strict=True)]
    )
    assert len(deviance) == len(data["time"])
    assert all(math.isfinite(value) for value in deviance)


def test_cox_zph_rank_transform_matches_low_level_ph_test():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10, eps=1e-5)
    raw = fit.schoenfeld_residuals()
    scaled = fit.scaled_schoenfeld_residuals()
    ranks = list(range(1, len(raw) + 1))
    low_level = survival.ph_test(scaled, ranks, None)
    raw_level = survival.ph_test(raw, ranks, None)

    result = survival.cox_zph(fit, transform="rank", terms=False)

    assert isinstance(result, survival.r_api.CoxZPHResult)
    assert result.variable_names == ["x1", "x2"]
    assert result.x == pytest.approx(ranks)
    assert result.time == pytest.approx(_fit_event_times(fit))
    assert result.chi2_values == pytest.approx(low_level.chi2_values)
    assert result.p_values == pytest.approx(low_level.p_values)
    assert result.global_chi2 == pytest.approx(low_level.global_chi2)
    assert result.global_df == low_level.global_df
    assert result.global_p_value == pytest.approx(low_level.global_p_value)
    assert result.global_chi2 != pytest.approx(raw_level.global_chi2)
    for actual, expected in zip(result.y, scaled, strict=True):
        assert actual == pytest.approx(expected)
    assert result.table[-1]["name"] == "GLOBAL"


def test_cox_zph_clustered_fit_uses_robust_scaled_schoenfeld_residuals():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    raw = fit.schoenfeld_residuals()
    robust_scaled = survival.r_api.residuals(fit, type="scaledsch")
    naive_scaled = fit.fit.scaled_schoenfeld_residuals()
    ranks = list(range(1, len(raw) + 1))
    low_level = survival.ph_test(robust_scaled, ranks, None)
    naive_level = survival.ph_test(naive_scaled, ranks, None)

    result = survival.cox_zph(fit, transform="rank", terms=False)

    assert result.variable_names == ["x1", "x2"]
    assert result.x == pytest.approx(ranks)
    assert result.chi2_values == pytest.approx(low_level.chi2_values)
    assert result.p_values == pytest.approx(low_level.p_values)
    assert result.global_chi2 == pytest.approx(low_level.global_chi2)
    assert result.global_df == low_level.global_df
    assert result.global_p_value == pytest.approx(low_level.global_p_value)
    assert result.global_chi2 != pytest.approx(naive_level.global_chi2)
    for actual, expected in zip(fit.scaled_schoenfeld_residuals(), robust_scaled, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.scaled_schoenfeld_residuals()[0] != pytest.approx(naive_scaled[0])
    for actual, expected in zip(result.y, robust_scaled, strict=True):
        assert actual == pytest.approx(expected)


def test_cox_zph_identity_and_km_transforms_expose_r_style_time_axes():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=10, eps=1e-5)
    event_times = _fit_event_times(fit)

    identity = survival.cox_zph(fit, transform="identity")
    ranked = survival.cox_zph(fit, transform="rank")
    km = survival.cox_zph(fit)
    identity_prefix = survival.cox_zph(fit, transform="i")
    ranked_prefix = survival.cox_zph(fit, transform="r")
    km_prefix = survival.cox_zph(fit, transform="k")
    km_alias = survival.cox_zph(fit, transform="kaplan_meier")
    logged_prefix = survival.cox_zph(fit, transform="l")
    low_level = survival.survfitkm(data["time"], data["status"], conf_type="none")
    expected_km = []
    cursor = 0
    for event_time in event_times:
        while cursor < len(low_level.time) and low_level.time[cursor] < event_time - 1e-9:
            cursor += 1
        previous_survival = low_level.estimate[cursor - 1] if cursor else 1.0
        expected_km.append(1.0 - previous_survival)

    assert identity.transform == "identity"
    assert identity.x == pytest.approx(event_times)
    assert identity_prefix.transform == "identity"
    assert identity_prefix.x == pytest.approx(identity.x)
    assert ranked.transform == "rank"
    assert ranked.x == pytest.approx(list(range(1, len(event_times) + 1)))
    assert ranked_prefix.transform == "rank"
    assert ranked_prefix.x == pytest.approx(ranked.x)
    assert km.transform == "km"
    assert km.x == pytest.approx(expected_km)
    assert km_prefix.transform == "km"
    assert km_prefix.x == pytest.approx(km.x)
    assert km_alias.transform == "km"
    assert km_alias.x == pytest.approx(km.x)
    assert logged_prefix.transform == "log"
    assert logged_prefix.x == pytest.approx([math.log(time) for time in event_times])
    assert km.x != pytest.approx(identity.x)


def test_cox_zph_rejects_unknown_transform():
    fit = survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=10, eps=1e-5)

    with pytest.raises(ValueError, match="transform"):
        survival.cox_zph(fit, transform="weird")


def test_cox_zph_formula_terms_group_multi_column_factors():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    by_term = survival.cox_zph(fit, transform="rank")
    by_column = survival.cox_zph(fit, transform="rank", terms=False)
    single_df = survival.cox_zph(fit, transform="rank", singledf=True)

    assert by_term.variable_names == ["factor(dose)", "x1"]
    assert by_term.df == [2, 1]
    assert by_column.variable_names == ["dose1", "dose2", "x1"]
    assert by_column.df == [1, 1, 1]
    assert single_df.df == [1, 1]
    assert len(by_term.y[0]) == 2
    assert len(by_column.y[0]) == 3
    assert by_term.table[-1]["name"] == "GLOBAL"
    assert survival.cox_zph(fit, global_test=False).table[-1]["name"] != "GLOBAL"
    assert survival.cox_zph(fit, **{"global": False}).table[-1]["name"] != "GLOBAL"


def test_coxph_partial_residuals_add_terms_to_martingales():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10, eps=1e-5)
    martingale = fit.martingale_residuals()
    beta = fit.coefficients[0]
    expected = [
        [
            martingale[row_idx] + fit.covariates[row_idx][col_idx] * beta[col_idx]
            for col_idx in range(2)
        ]
        for row_idx in range(len(martingale))
    ]

    partial = fit.partial_residuals()
    for actual, expected_row in zip(partial, expected, strict=True):
        assert actual == pytest.approx(expected_row)
    for alias in ("partial", "partials", "p"):
        for actual, expected_row in zip(
            survival.r_api.residuals(fit, type=alias),
            expected,
            strict=True,
        ):
            assert actual == pytest.approx(expected_row)

    selected_x2 = survival.r_api.residuals(fit, type="partial", terms="x2")
    selected_by_index = survival.r_api.residuals(fit, type="partial", terms=[2, 1])
    for row_idx, actual in enumerate(selected_x2):
        assert actual == pytest.approx([expected[row_idx][1]])
    for row_idx, actual in enumerate(selected_by_index):
        assert actual == pytest.approx([expected[row_idx][1], expected[row_idx][0]])

    with pytest.raises(ValueError, match="unknown model term"):
        survival.r_api.residuals(fit, type="partial", terms="missing")


def test_coxph_counting_process_partial_residuals_keep_training_row_order():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        method="breslow",
        initial_beta=[0.25],
        max_iter=0,
    )
    martingale = fit.martingale_residuals()
    beta = fit.coefficients[0]
    expected = [
        [martingale[row_idx] + fit.covariates[row_idx][0] * beta[0]]
        for row_idx in range(len(martingale))
    ]

    partial = survival.r_api.residuals(fit, type="partial")
    assert len(partial) == len(data["stop"])
    for actual, expected_row in zip(partial, expected, strict=True):
        assert actual == pytest.approx(expected_row)


def test_coxph_residuals_honor_case_weighting_rules():
    data = _toy_data()
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.75, 2.5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.1, -0.2],
        max_iter=0,
    )

    martingale = fit.martingale_residuals()
    assert survival.r_api.residuals(fit, type="martingale", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(martingale)]
    )
    assert survival.r_api.residuals(fit, type="m") == pytest.approx(martingale)

    score = fit.score_residuals()
    weighted_score = survival.r_api.residuals(fit, type="score", weighted=True)
    for row_idx, (actual, expected_row) in enumerate(zip(weighted_score, score, strict=True)):
        assert actual == pytest.approx([value * weights[row_idx] for value in expected_row])

    dfbeta = fit.dfbeta()
    default_dfbeta = survival.r_api.residuals(fit, type="dfbeta")
    unweighted_dfbeta = survival.r_api.residuals(fit, type="dfbeta", weighted=False)
    for row_idx, (default_row, raw_row, unweighted_row) in enumerate(
        zip(default_dfbeta, dfbeta, unweighted_dfbeta, strict=True)
    ):
        assert unweighted_row == pytest.approx(raw_row)
        assert default_row == pytest.approx([value * weights[row_idx] for value in raw_row])

    terms = [
        [fit.covariates[row_idx][col_idx] * fit.coefficients[0][col_idx] for col_idx in range(2)]
        for row_idx in range(len(martingale))
    ]
    partial = survival.r_api.residuals(fit, type="partial", weighted=True)
    for row_idx, actual in enumerate(partial):
        assert actual == pytest.approx(
            [term + martingale[row_idx] * weights[row_idx] for term in terms[row_idx]]
        )


def test_coxph_residuals_collapse_training_rows_by_label():
    data = _counting_cox_data()
    collapse = ["A", "A", "B", "B", "C", "C"]
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        method="breslow",
        initial_beta=[0.25],
        max_iter=0,
    )

    martingale = fit.martingale_residuals()
    expected_martingale = [
        sum(martingale[idx] for idx, label in enumerate(collapse) if label == group)
        for group in ("A", "B", "C")
    ]
    assert survival.r_api.residuals(
        fit,
        type="martingale",
        collapse=collapse,
    ) == pytest.approx(expected_martingale)

    score = fit.score_residuals()
    collapsed_score = survival.r_api.residuals(fit, type="score", collapse=collapse)
    expected_score = [
        [sum(score[idx][0] for idx, label in enumerate(collapse) if label == group)]
        for group in ("A", "B", "C")
    ]
    for actual, expected_row in zip(collapsed_score, expected_score, strict=True):
        assert actual == pytest.approx(expected_row)

    partial = fit.partial_residuals()
    collapsed_partial = survival.r_api.residuals(fit, type="partial", collapse=collapse)
    expected_partial = [
        [sum(partial[idx][0] for idx, label in enumerate(collapse) if label == group)]
        for group in ("A", "B", "C")
    ]
    for actual, expected_row in zip(collapsed_partial, expected_partial, strict=True):
        assert actual == pytest.approx(expected_row)

    collapsed_status = [
        sum(fit.status[idx] for idx, label in enumerate(collapse) if label == group)
        for group in ("A", "B", "C")
    ]
    expected_deviance = []
    for residual, status in zip(expected_martingale, collapsed_status, strict=True):
        log_term = status * math.log(max(status - residual, 1e-12)) if status > 0 else 0.0
        magnitude = math.sqrt(max(-2.0 * (residual + log_term), 0.0))
        expected_deviance.append(magnitude if residual >= 0.0 else -magnitude)
    assert survival.r_api.residuals(
        fit,
        type="deviance",
        collapse=collapse,
    ) == pytest.approx(expected_deviance)


def test_coxph_event_residuals_support_weighted_schoenfeld_output():
    data = _toy_data()
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.75, 2.5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.1, -0.2],
        max_iter=0,
    )
    raw = fit.schoenfeld_residuals()
    event_weights = [weights[idx] for idx, status in enumerate(data["status"]) if status == 1]
    weighted_raw = [
        [value * event_weights[row_idx] for value in row] for row_idx, row in enumerate(raw)
    ]

    for actual, expected_row in zip(
        survival.r_api.residuals(fit, type="schoenfeld", weighted=True),
        weighted_raw,
        strict=True,
    ):
        assert actual == pytest.approx(expected_row)

    beta = fit.coefficients[0]
    variance = fit.information_matrix
    expected_scaled = [
        [
            beta[col_idx]
            + len(weighted_raw)
            * sum(row[inner_idx] * variance[inner_idx][col_idx] for inner_idx in range(2))
            for col_idx in range(2)
        ]
        for row in weighted_raw
    ]
    for actual, expected_row in zip(
        survival.r_api.residuals(fit, type="scaledsch", weighted=True),
        expected_scaled,
        strict=True,
    ):
        assert actual == pytest.approx(expected_row)


def test_coxph_schoenfeld_residuals_match_event_risk_set_means():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    rows = [[x1, x2] for x1, x2 in zip(data["x1"], data["x2"], strict=True)]
    risks = [math.exp(value) for value in fit.linear_predictors]
    expected = []
    for event_idx, status in enumerate(data["status"]):
        if status != 1:
            continue
        at_risk = [idx for idx, time in enumerate(data["time"]) if time >= data["time"][event_idx]]
        denom = sum(risks[idx] for idx in at_risk)
        means = [
            sum(risks[idx] * rows[idx][col_idx] for idx in at_risk) / denom for col_idx in range(2)
        ]
        expected.append([rows[event_idx][col_idx] - means[col_idx] for col_idx in range(2)])

    assert fit.method == "efron"
    for actual, expected_row in zip(fit.covariates, rows, strict=True):
        assert actual == pytest.approx(expected_row)
    for residuals in (
        fit.schoenfeld_residuals(),
        survival.r_api.residuals(fit, type="schoenfeld"),
        survival.r_api.residuals(fit, type="sch"),
        survival.r_api.residuals(fit, type="scho"),
    ):
        assert len(residuals) == len(expected)
        for actual, expected_row in zip(residuals, expected, strict=True):
            assert actual == pytest.approx(expected_row)


def test_coxph_scaled_schoenfeld_residuals_use_r_scaling():
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_toy_data(),
        max_iter=10,
        eps=1e-5,
    )
    raw = fit.schoenfeld_residuals()
    beta = fit.coefficients[0]
    variance = fit.information_matrix
    event_count = len(raw)
    expected = [
        [
            beta[col_idx]
            + event_count
            * sum(row[inner_idx] * variance[inner_idx][col_idx] for inner_idx in range(2))
            for col_idx in range(2)
        ]
        for row in raw
    ]

    scaled = fit.scaled_schoenfeld_residuals()
    for actual, expected_row in zip(scaled, expected, strict=True):
        assert actual == pytest.approx(expected_row)
    for alias in ("scaledsch", "scaledschoenfeld", "scaled_schoenfeld", "sca"):
        for actual, expected_row in zip(
            survival.r_api.residuals(fit, type=alias),
            expected,
            strict=True,
        ):
            assert actual == pytest.approx(expected_row)


def test_coxph_score_and_dfbeta_residuals_use_fitted_information():
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_toy_data(),
        ties="efron",
        max_iter=10,
        eps=1e-5,
    )
    score = fit.score_residuals()
    dfbeta = fit.dfbeta()
    dfbetas = fit.dfbetas()

    for actual_matrix, expected_matrix in (
        (survival.r_api.residuals(fit, type="score"), score),
        (survival.r_api.residuals(fit, type="sco"), score),
        (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
        (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
    ):
        for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
            assert actual == pytest.approx(expected)
    assert len(score) == len(fit.event_times)
    assert len(dfbeta) == len(fit.event_times)
    assert len(dfbetas) == len(fit.event_times)
    for row_idx, (score_row, dfbeta_row, dfbetas_row) in enumerate(
        zip(score, dfbeta, dfbetas, strict=True)
    ):
        assert len(score_row) == 2
        assert len(dfbeta_row) == 2
        assert len(dfbetas_row) == 2
        for col_idx in range(2):
            expected_dfbeta = sum(
                fit.information_matrix[col_idx][inner_idx] * score_row[inner_idx]
                for inner_idx in range(2)
            )
            scale = math.sqrt(abs(fit.information_matrix[col_idx][col_idx]))
            assert dfbeta_row[col_idx] == pytest.approx(expected_dfbeta)
            assert dfbetas_row[col_idx] == pytest.approx(dfbeta[row_idx][col_idx] / scale)


def test_coxph_counting_process_score_and_dfbeta_residuals_sum_to_score_vector():
    data = _counting_cox_data()
    for method in ("breslow", "efron", "exact"):
        fit = survival.coxph(
            "Surv(start, stop, status) ~ x1",
            data=data,
            method=method,
            initial_beta=[0.0],
            max_iter=0,
        )
        score = fit.score_residuals()
        dfbeta = fit.dfbeta()
        dfbetas = fit.dfbetas()

        assert len(score) == len(data["stop"])
        assert [sum(row[col_idx] for row in score) for col_idx in range(1)] == pytest.approx(
            fit.score_vector
        )
        for actual_matrix, expected_matrix in (
            (survival.r_api.residuals(fit, type="score"), score),
            (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
            (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
        ):
            for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
                assert actual == pytest.approx(expected)
        for row_idx, score_row in enumerate(score):
            expected_dfbeta = fit.information_matrix[0][0] * score_row[0]
            scale = math.sqrt(abs(fit.information_matrix[0][0]))
            assert dfbeta[row_idx][0] == pytest.approx(expected_dfbeta)
            assert dfbetas[row_idx][0] == pytest.approx(expected_dfbeta / scale)


def test_coxph_exact_tie_score_and_dfbeta_residuals_sum_to_score_vector():
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=_tied_cox_data(),
        ties="exact",
        initial_beta=[0.0],
        max_iter=0,
    )
    score = fit.score_residuals()
    dfbeta = fit.dfbeta()
    dfbetas = fit.dfbetas()

    assert [sum(row[col_idx] for row in score) for col_idx in range(1)] == pytest.approx(
        fit.score_vector
    )
    for actual_matrix, expected_matrix in (
        (survival.r_api.residuals(fit, type="score"), score),
        (survival.r_api.residuals(fit, type="dfbeta"), dfbeta),
        (survival.r_api.residuals(fit, type="dfbetas"), dfbetas),
    ):
        for actual, expected in zip(actual_matrix, expected_matrix, strict=True):
            assert actual == pytest.approx(expected)
    for row_idx, score_row in enumerate(score):
        expected_dfbeta = fit.information_matrix[0][0] * score_row[0]
        scale = math.sqrt(abs(fit.information_matrix[0][0]))
        assert dfbeta[row_idx][0] == pytest.approx(expected_dfbeta)
        assert dfbetas[row_idx][0] == pytest.approx(expected_dfbeta / scale)


def test_predict_expected_uses_counting_process_entry_intervals():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )

    hazard_times, hazards = fit.basehaz(False)
    expected = survival.predict(fit, type="expected")
    martingale = survival.r_api.residuals(fit, type="martingale")

    assert hazard_times == pytest.approx([2.0, 4.0, 5.0])
    assert expected[3] == pytest.approx(hazards[-1] - hazards[0])
    assert martingale[3] == pytest.approx(-expected[3])


def test_coxph_formula_treatment_codes_categorical_covariates():
    fit = survival.coxph("Surv(time, status) ~ group + x1", data=_toy_data(), max_iter=10)

    assert sum(len(row) for row in fit.coefficients) == 2
    assert len(fit.predict([[1.0, 0.5]])) == 1


def test_coxph_formula_factor_treatment_codes_numeric_covariates():
    data = _factor_data()
    fit = survival.coxph(
        "Surv(time, status) ~ factor(dose) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [
                1.0 if data["dose"][idx] == 1 else 0.0,
                1.0 if data["dose"][idx] == 2 else 0.0,
                data["x1"][idx],
            ]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_numeric_transforms():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ log(x2) + x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [math.log(data["x2"][idx]), data["x1"][idx]] for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_identity_wrappers_for_numeric_terms():
    data = _toy_data()
    direct = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)

    for wrapper in ("I", "identity", "as.numeric"):
        fit = survival.coxph(
            f"Surv(time, status) ~ {wrapper}(x1) + x2",
            data=data,
            max_iter=10,
            eps=1e-5,
        )

        assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
        assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_accepts_identity_arithmetic_terms():
    data = _numeric_data()
    fit = survival.coxph(
        "Surv(time, status) ~ I(-x1) + I(x1 + x2) + I((x1 + x2)^2)",
        data=data,
        max_iter=0,
    )
    expected_rows = [
        [
            -data["x1"][idx],
            data["x1"][idx] + data["x2"][idx],
            (data["x1"][idx] + data["x2"][idx]) ** 2,
        ]
        for idx in range(len(data["time"]))
    ]
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=expected_rows,
        max_iter=0,
    )

    for actual, expected in zip(fit.covariates, expected_rows, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)

    data_with_missing = _numeric_data()
    data_with_missing["x2"][1] = None
    filtered = survival.coxph(
        "Surv(time, status) ~ I(x1 + x2)",
        data=data_with_missing,
        na_action="omit",
        max_iter=0,
    )
    expected_indices = [
        idx for idx, value in enumerate(data_with_missing["x2"]) if value is not None
    ]
    assert filtered.event_times == pytest.approx(
        [data_with_missing["time"][idx] for idx in expected_indices]
    )
    expected_filtered_rows = [
        [data_with_missing["x1"][idx] + data_with_missing["x2"][idx]] for idx in expected_indices
    ]
    for actual, expected in zip(filtered.covariates, expected_filtered_rows, strict=True):
        assert actual == pytest.approx(expected)


def test_coxph_formula_accepts_numeric_interactions():
    data = _numeric_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [data["x1"][idx], data["x2"][idx], data["x1"][idx] * data["x2"][idx]]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_accepts_categorical_interactions():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ group * x1",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[
            [
                1.0 if data["group"][idx] == "B" else 0.0,
                data["x1"][idx],
                data["x1"][idx] if data["group"][idx] == "B" else 0.0,
            ]
            for idx in range(len(data["time"]))
        ],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_defaults_to_efron_ties():
    data = _tied_cox_data()
    default = survival.coxph("Surv(time, status) ~ x1", data=data, max_iter=15, eps=1e-5)
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    breslow = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert default.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert efron.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert efron.risk_scores == pytest.approx(low_level.risk_scores)
    assert efron.coefficients[0][0] != pytest.approx(breslow.coefficients[0][0])


def test_coxph_accepts_r_style_ties_alias():
    data = _tied_cox_data()
    by_ties = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="breslow",
        max_iter=15,
        eps=1e-5,
    )
    by_method = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
    )
    partial_ties = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="br",
        max_iter=15,
        eps=1e-5,
    )
    matching_aliases = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="br",
        ties="breslow",
        max_iter=15,
        eps=1e-5,
    )

    assert by_ties.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert by_ties.log_likelihood == pytest.approx(by_method.log_likelihood)
    assert partial_ties.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert partial_ties.log_likelihood == pytest.approx(by_method.log_likelihood)
    assert matching_aliases.coefficients[0] == pytest.approx(by_method.coefficients[0])
    assert matching_aliases.log_likelihood == pytest.approx(by_method.log_likelihood)


def test_coxph_accepts_unused_tt_argument_without_time_transform_terms():
    data = _toy_data()
    baseline = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    unused_tt = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        tt=lambda value, *_args: value,
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(unused_tt.coefficients, baseline.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert unused_tt.log_likelihood == pytest.approx(baseline.log_likelihood)


def test_coxph_accepts_r_style_control_mapping():
    data = _tied_cox_data()
    explicit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        max_iter=15,
        eps=1e-5,
        toler=1e-8,
    )
    controlled = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        control={
            "iter.max": 15,
            "eps": 1e-5,
            "toler.chol": 1e-8,
            "toler.inf": math.sqrt(1e-5),
            "outer.max": 10,
            "timefix": True,
        },
    )

    assert controlled.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert controlled.log_likelihood == pytest.approx(explicit.log_likelihood)
    assert controlled.risk_scores == pytest.approx(explicit.risk_scores)

    nondefault_ignored = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        method="breslow",
        control={
            "iter.max": 15,
            "eps": 1e-5,
            "toler.chol": 1e-8,
            "toler.inf": 0.25,
            "outer.max": 2,
        },
    )

    assert nondefault_ignored.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert nondefault_ignored.log_likelihood == pytest.approx(explicit.log_likelihood)
    assert nondefault_ignored.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_accepts_r_style_formula_storage_flags():
    data = _toy_data()
    default = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    explicit_defaults = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        x=False,
        y=True,
        model=False,
        tt=None,
        id=None,
        istate=None,
        statedata=None,
        singular_ok=True,
        nocenter=[-1, 0, 1],
        max_iter=10,
        eps=1e-5,
    )
    strict_singular = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        singular_ok=False,
        max_iter=10,
        eps=1e-5,
    )
    with_model = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + offset(offset) + strata(group)",
        data=data,
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(explicit_defaults.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(strict_singular.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert explicit_defaults.log_likelihood == pytest.approx(default.log_likelihood)
    assert strict_singular.log_likelihood == pytest.approx(default.log_likelihood)
    assert explicit_defaults.y.time == pytest.approx(data["time"])
    assert explicit_defaults.y.event == tuple(data["status"])
    assert not hasattr(explicit_defaults, "x")
    assert not hasattr(explicit_defaults, "model")

    model_frame = with_model.model
    assert model_frame["Surv(time, status)"].time == pytest.approx(data["time"])
    assert model_frame["Surv(time, status)"].event == tuple(data["status"])
    assert model_frame["time"] == pytest.approx(data["time"])
    assert model_frame["status"] == data["status"]
    assert model_frame["x1"] == pytest.approx(data["x1"])
    assert model_frame["x2"] == pytest.approx(data["x2"])
    assert model_frame["offset"] == pytest.approx(data["offset"])
    assert model_frame["group"] == data["group"]
    assert model_frame["(offset)"] == pytest.approx(data["offset"])
    assert model_frame["(strata)"] == data["group"]

    with_x = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        x=True,
        y=False,
        max_iter=10,
        eps=1e-5,
    )
    expected_x = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    for actual, expected in zip(with_x.x, expected_x, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(with_x, "y")

    dotted_singular = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
        **{"singular.ok": True},
    )
    for actual, expected in zip(dotted_singular.coefficients, default.coefficients, strict=True):
        assert actual == pytest.approx(expected)
    assert dotted_singular.log_likelihood == pytest.approx(default.log_likelihood)


def test_coxph_nocenter_controls_r_style_column_centering():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 0, 1],
        "dummy": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "x": [0.2, 1.3, 0.7, 1.8, 1.1, 2.2],
    }

    default = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        max_iter=20,
        eps=1e-9,
    )
    center_all = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        nocenter=[],
        max_iter=20,
        eps=1e-9,
    )
    scalar = survival.coxph(
        "Surv(time, status) ~ dummy + x",
        data=data,
        nocenter=0,
        max_iter=20,
        eps=1e-9,
    )

    assert default.nocenter == pytest.approx([-1.0, 0.0, 1.0])
    assert center_all.nocenter == []
    assert scalar.nocenter == pytest.approx([0.0])
    assert default.coefficients[0] == pytest.approx(center_all.coefficients[0])
    assert default.log_likelihood == pytest.approx(center_all.log_likelihood)
    assert default.means[0] == pytest.approx(0.0)
    assert center_all.means[0] == pytest.approx(sum(data["dummy"]) / len(data["dummy"]))
    assert scalar.means[0] == pytest.approx(center_all.means[0])

    default_lp = survival.predict(default, reference="sample")
    center_all_lp = survival.predict(center_all, reference="sample")
    shift = center_all.means[0] * center_all.coefficients[0][0]
    assert center_all_lp == pytest.approx([value - shift for value in default_lp])
    assert survival.predict(default, reference="zero") == pytest.approx(
        survival.predict(center_all, reference="zero")
    )


def test_coxph_model_true_stores_matrix_inputs():
    data = _toy_data()
    response = survival.Surv(data["time"], data["status"])
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]

    fit = survival.coxph(response, x=rows, model=True, y=False, max_iter=10, eps=1e-5)

    assert fit.model["response"].time == pytest.approx(data["time"])
    assert fit.model["response"].event == tuple(data["status"])
    for actual, expected in zip(fit.model["x"], rows, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(fit, "y")


def test_coxph_singular_ok_false_rejects_dependent_designs():
    data = _toy_data()
    singular_data = {
        **data,
        "constant": [1.0] * len(data["time"]),
        "x_duplicate": [2.0 * value + 1.0 for value in data["x1"]],
    }
    response = survival.Surv(data["time"], data["status"])

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x_duplicate",
            data=singular_data,
            singular_ok=False,
        )

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            "Surv(time, status) ~ constant",
            data=singular_data,
            singular_ok=False,
        )

    with pytest.raises(ValueError, match="singular.*singular_ok=True"):
        survival.coxph(
            response,
            x=[[value, 2.0 * value + 1.0] for value in data["x1"]],
            singular_ok=False,
        )

    intercept_only = survival.coxph(
        "Surv(time, status) ~ 1",
        data=data,
        singular_ok=False,
        max_iter=0,
    )
    assert intercept_only.coefficients == [[]]


def test_coxph_control_timefix_matches_r_near_tie_behavior():
    data = {
        "time": [1.0, 1.0 + 5e-10, 2.0, 3.0],
        "status": [1, 1, 0, 1],
        "x1": [0.0, 1.0, 0.5, 1.5],
    }
    fixed_times = [1.0, 1.0, 2.0, 3.0]
    rows = [[value] for value in data["x1"]]
    fixed_low_level = survival.regression.coxph_fit(
        fixed_times,
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    exact_low_level = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )

    default = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    explicit_true = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": True},
    )
    exact = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": False},
    )
    exact_dotted = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"time.fix": False},
    )

    assert default.coefficients[0] == pytest.approx(fixed_low_level.coefficients[0])
    assert default.log_likelihood == pytest.approx(fixed_low_level.log_likelihood)
    assert explicit_true.coefficients[0] == pytest.approx(default.coefficients[0])
    assert exact.coefficients[0] == pytest.approx(exact_low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(exact_low_level.log_likelihood)
    assert exact_dotted.coefficients[0] == pytest.approx(exact.coefficients[0])
    assert exact.coefficients[0] != pytest.approx(default.coefficients[0])

    with pytest.raises(ValueError, match=r"control\.timefix or control\.time\.fix"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=data,
            control={"timefix": False, "time.fix": True},
        )


def test_coxph_control_timefix_applies_to_counting_process_endpoints():
    start = [0.0, 0.0, 0.5, 1.0 + 5e-10, 0.0]
    stop = [1.0, 1.0 + 5e-10, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1, 0]
    rows = [[value] for value in [0.0, 1.0, 0.5, 1.5, 0.2]]
    fixed_start, fixed_stop = survival.r_api._timefix_vectors(
        [float(value) for value in start],
        [float(value) for value in stop],
    )
    fixed_low_level = survival.regression.coxph_fit(
        fixed_stop,
        status,
        rows,
        entry_times=fixed_start,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    exact_low_level = survival.regression.coxph_fit(
        stop,
        status,
        rows,
        entry_times=start,
        max_iter=20,
        eps=1e-9,
        method="efron",
    )
    response = survival.Surv(start, stop, status)

    default = survival.coxph(response, x=rows, max_iter=20, eps=1e-9, method="efron")
    exact = survival.coxph(
        response,
        x=rows,
        max_iter=20,
        eps=1e-9,
        method="efron",
        control={"timefix": False},
    )

    assert default.coefficients[0] == pytest.approx(fixed_low_level.coefficients[0])
    assert default.log_likelihood == pytest.approx(fixed_low_level.log_likelihood)
    assert exact.coefficients[0] == pytest.approx(exact_low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(exact_low_level.log_likelihood)
    assert exact.coefficients[0] != pytest.approx(default.coefficients[0])


def test_coxph_exact_ties_alias_accepts_data_without_tied_events():
    data = _toy_data()
    exact = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="exact",
        max_iter=10,
        eps=1e-5,
    )
    partial_exact = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="ex",
        max_iter=10,
        eps=1e-5,
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        ties="efron",
        max_iter=10,
        eps=1e-5,
    )

    assert exact.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert exact.log_likelihood == pytest.approx(efron.log_likelihood)
    assert exact.risk_scores == pytest.approx(efron.risk_scores)
    assert partial_exact.coefficients[0] == pytest.approx(efron.coefficients[0])
    assert partial_exact.log_likelihood == pytest.approx(efron.log_likelihood)
    assert partial_exact.risk_scores == pytest.approx(efron.risk_scores)


def test_coxph_exact_ties_handles_tied_events():
    data = _tied_cox_data()
    exact = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="exact",
        max_iter=15,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="exact",
        max_iter=15,
        eps=1e-5,
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        ties="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert exact.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert exact.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert exact.risk_scores == pytest.approx(low_level.risk_scores)
    assert exact.coefficients[0][0] != pytest.approx(efron.coefficients[0][0])


def test_low_level_coxph_tie_methods_match_hand_likelihood_at_initial_beta():
    data = _tied_cox_data()
    for method in ("efron", "breslow", "exact"):
        fit = survival.regression.coxph_fit(
            time=data["time"],
            status=data["status"],
            covariates=[[value] for value in data["x1"]],
            initial_beta=[0.0],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik_at_zero(data["time"], data["status"], method)

        assert fit.log_likelihood == pytest.approx([expected, expected])


def test_low_level_coxph_accepts_zero_column_null_model():
    data = _toy_data()
    fit = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        max_iter=0,
    )
    expected = _manual_cox_loglik_at_zero(data["time"], data["status"], "efron")

    assert fit.coefficients == [[]]
    assert fit.means == []
    assert fit.score_vector == []
    assert fit.information_matrix == []
    assert fit.linear_predictors == pytest.approx([0.0] * len(data["time"]))
    assert fit.risk_scores == pytest.approx([1.0] * len(data["time"]))
    assert fit.log_likelihood == pytest.approx([expected, expected])
    assert fit.predict([[], []]) == pytest.approx([0.0, 0.0])
    times, curves = fit.survival_curve([[]])
    assert times
    assert len(curves) == 1


def test_low_level_coxph_nocenter_disables_column_scaling():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "status": [1, 1, 0, 1, 0, 1],
        "dummy": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "x": [0.2, 1.3, 0.7, 1.8, 1.1, 2.2],
    }
    rows = [[dummy, x] for dummy, x in zip(data["dummy"], data["x"], strict=True)]

    centered = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
    )
    default = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        nocenter=[-1.0, 0.0, 1.0],
    )
    zero_only = survival.regression.coxph_fit(
        data["time"],
        data["status"],
        rows,
        max_iter=20,
        eps=1e-9,
        nocenter=[0.0],
    )

    assert centered.nocenter == []
    assert default.nocenter == pytest.approx([-1.0, 0.0, 1.0])
    assert centered.coefficients[0] == pytest.approx(default.coefficients[0])
    assert centered.log_likelihood == pytest.approx(default.log_likelihood)
    assert centered.means[0] == pytest.approx(sum(data["dummy"]) / len(data["dummy"]))
    assert default.means[0] == pytest.approx(0.0)
    assert default.means[1] != pytest.approx(0.0)
    assert zero_only.means[0] == pytest.approx(centered.means[0])


def test_low_level_coxph_rejects_invalid_numeric_inputs():
    data = _toy_data()
    kwargs = {
        "time": data["time"],
        "status": data["status"],
        "covariates": [[value] for value in data["x1"]],
        "initial_beta": [0.0],
        "max_iter": 0,
    }

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "time": [1.0, float("nan"), *data["time"][2:]]})

    bad_status = [*data["status"]]
    bad_status[0] = 2
    with pytest.raises(ValueError, match="status must contain only 0/1"):
        survival.regression.coxph_fit(**{**kwargs, "status": bad_status})

    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        survival.regression.coxph_fit(
            **{**kwargs, "covariates": [[float("inf")], *kwargs["covariates"][1:]]}
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.regression.coxph_fit(**{**kwargs, "weights": [1.0, -1.0, *([1.0] * 6)]})

    with pytest.raises(ValueError, match="at least one positive"):
        survival.regression.coxph_fit(**{**kwargs, "weights": [0.0] * len(data["time"])})

    with pytest.raises(ValueError, match="offset contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "offset": [0.0, float("inf"), *([0.0] * 6)]})

    with pytest.raises(ValueError, match="entry_times contains non-finite"):
        survival.regression.coxph_fit(
            **{**kwargs, "entry_times": [0.0, float("nan"), *([0.0] * 6)]}
        )

    with pytest.raises(ValueError, match="initial_beta contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "initial_beta": [float("nan")]})

    with pytest.raises(ValueError, match="nocenter contains non-finite"):
        survival.regression.coxph_fit(**{**kwargs, "nocenter": [0.0, float("nan")]})

    with pytest.raises(ValueError, match="eps must be a finite positive value"):
        survival.regression.coxph_fit(**{**kwargs, "eps": 0.0})

    fit = survival.regression.coxph_fit(**kwargs)
    with pytest.raises(ValueError, match="covariates row contains non-finite"):
        fit.predict([[float("nan")]])
    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        fit.survival_curve([[float("inf")]])
    with pytest.raises(ValueError, match=r"covariates\[0\] contains non-finite"):
        fit.survival_curve_with_strata([[float("nan")]], [0])


def test_low_level_coxph_counting_process_uses_entry_times():
    data = _counting_cox_data()
    for method in ("efron", "breslow"):
        fit = survival.regression.coxph_fit(
            time=data["stop"],
            status=data["status"],
            covariates=[[value] for value in data["x1"]],
            entry_times=data["start"],
            initial_beta=[0.0],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik_at_zero(
            data["stop"],
            data["status"],
            method,
            entry_times=data["start"],
        )

        assert fit.entry_times == pytest.approx(data["start"])
        assert fit.log_likelihood == pytest.approx([expected, expected])


def test_low_level_coxph_counting_process_matches_weighted_hand_likelihood():
    data = _counting_cox_data()
    data["strata"] = ["A", "A", "A", "B", "B", "B"]
    weights = [1.5, 0.75, 2.0, 1.0, 1.25, 0.5]
    offset = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2]
    covariates = [[value] for value in data["x1"]]
    strata = [0 if value == "A" else 1 for value in data["strata"]]
    for method in ("efron", "breslow"):
        fit = survival.regression.coxph_fit(
            time=data["stop"],
            status=data["status"],
            covariates=covariates,
            strata=strata,
            weights=weights,
            offset=offset,
            entry_times=data["start"],
            initial_beta=[0.35],
            method=method,
            max_iter=0,
        )
        expected = _manual_cox_loglik(
            data["stop"],
            data["status"],
            covariates,
            [0.35],
            method,
            entry_times=data["start"],
            weights=weights,
            offset=offset,
            strata=strata,
        )

        assert fit.log_likelihood == pytest.approx([expected, expected])


def test_low_level_coxph_zero_entry_matches_right_censored_fit():
    data = _tied_cox_data()
    right = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    counting = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        method="efron",
        entry_times=[0.0] * len(data["time"]),
        max_iter=15,
        eps=1e-5,
    )

    assert counting.coefficients[0] == pytest.approx(right.coefficients[0])
    assert counting.log_likelihood == pytest.approx(right.log_likelihood)


def test_low_level_coxph_sorts_counting_process_inputs_with_strata():
    data = _counting_cox_data()
    data["strata"] = ["A", "A", "A", "B", "B", "B"]
    sorted_indices = sorted(
        range(len(data["stop"])),
        key=lambda idx: (data["strata"][idx], data["stop"][idx]),
    )
    shuffled_indices = [3, 0, 5, 1, 4, 2]
    sorted_data = _take(data, sorted_indices)
    shuffled_data = _take(data, shuffled_indices)
    sorted_fit = survival.regression.coxph_fit(
        time=sorted_data["stop"],
        status=sorted_data["status"],
        covariates=[[value] for value in sorted_data["x1"]],
        strata=[0 if value == "A" else 1 for value in sorted_data["strata"]],
        entry_times=sorted_data["start"],
        method="breslow",
        initial_beta=[0.0],
        max_iter=0,
    )
    shuffled_fit = survival.regression.coxph_fit(
        time=shuffled_data["stop"],
        status=shuffled_data["status"],
        covariates=[[value] for value in shuffled_data["x1"]],
        strata=[0 if value == "A" else 1 for value in shuffled_data["strata"]],
        entry_times=shuffled_data["start"],
        method="breslow",
        initial_beta=[0.0],
        max_iter=0,
    )

    assert shuffled_fit.log_likelihood == pytest.approx(sorted_fit.log_likelihood)
    assert shuffled_fit.score_vector == pytest.approx(sorted_fit.score_vector)


def test_low_level_coxph_sorts_rows_with_strata_before_fitting():
    data = _tied_cox_data()
    sorted_indices = sorted(
        range(len(data["time"])),
        key=lambda idx: (data["strata"][idx], data["time"][idx]),
    )
    shuffled_indices = [3, 0, 7, 1, 5, 2, 6, 4]
    sorted_data = _take(data, sorted_indices)
    shuffled_data = _take(data, shuffled_indices)
    sorted_fit = survival.regression.coxph_fit(
        time=sorted_data["time"],
        status=sorted_data["status"],
        covariates=[[value] for value in sorted_data["x1"]],
        strata=[0 if value == "A" else 1 for value in sorted_data["strata"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )
    shuffled_fit = survival.regression.coxph_fit(
        time=shuffled_data["time"],
        status=shuffled_data["status"],
        covariates=[[value] for value in shuffled_data["x1"]],
        strata=[0 if value == "A" else 1 for value in shuffled_data["strata"]],
        method="efron",
        max_iter=15,
        eps=1e-5,
    )

    assert shuffled_fit.coefficients[0] == pytest.approx(sorted_fit.coefficients[0])
    assert shuffled_fit.log_likelihood == pytest.approx(sorted_fit.log_likelihood)


def test_coxph_formula_dot_expands_remaining_covariates():
    data = _numeric_data()
    fit = survival.coxph("Surv(time, status) ~ .", data=data, max_iter=10, eps=1e-5)
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert fit.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_formula_dot_can_exclude_identifier_columns():
    data = _numeric_data_with_id()
    fit = survival.coxph("Surv(time, status) ~ . - id", data=data, max_iter=10, eps=1e-5)
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(explicit.coefficients[0])
    assert fit.risk_scores == pytest.approx(explicit.risk_scores)


def test_coxph_formula_accepts_backticks_for_covariates_and_offsets():
    data = _backtick_data()
    fit = survival.coxph(
        "Surv(`follow-up`, `event status`) ~ `age-years` + `marker/value` + offset(`log exposure`)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["follow-up"],
        status=data["event status"],
        covariates=[
            [data["age-years"][idx], data["marker/value"][idx]]
            for idx in range(len(data["follow-up"]))
        ],
        offset=data["log exposure"],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_applies_subset_before_design_matrix():
    data = _numeric_data()
    indices = [0, 1, 3, 5, 6]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        subset=[idx in indices for idx in range(len(data["time"]))],
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_take(data, indices),
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_na_action_omit_matches_filtered_data():
    data = _numeric_data()
    data = {**data, "x1": [0.2, 0.4, float("nan"), 0.8, 1.0, 1.2, 0.6, 1.4]}
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        na_action="omit",
        max_iter=10,
        eps=1e-5,
    )
    dotted = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        max_iter=10,
        eps=1e-5,
        **{"na.action": "omit"},
    )
    direct = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=_take(data, indices),
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)
    assert dotted.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert dotted.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_filters_external_weights_and_offset_with_subset_and_na_action():
    data = _numeric_data()
    indices = [0, 2, 3, 4, 5]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        offset=[0.1, 0.0, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1],
        subset=[0, 1, 2, 3, 4, 5],
        na_action="omit",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.regression.coxph_fit(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in indices],
        weights=[1.0, 1.0, 0.8, 1.2, 1.1],
        offset=[0.1, 0.0, -0.1, 0.2, 0.0],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(direct.coefficients[0])
    assert fit.risk_scores == pytest.approx(direct.risk_scores)


def test_coxph_formula_passes_strata_to_rust_optimizer():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        strata=[0, 0, 0, 0, 1, 1, 1, 1],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert len(fit.risk_scores) == 8
    assert len(fit.predict([[0.5]])) == 1


def test_predict_coxph_reference_strata_uses_training_stratum_means():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    beta = fit.coefficients[0][0]
    strata_means = {"A": 0.375, "B": 1.05}
    sample_mean = sum(data["x1"]) / len(data["x1"])

    default_lp = survival.predict(fit)
    sample_lp = survival.predict(fit, reference="sample")
    sample_prefix_lp = survival.predict(fit, reference="sa")
    zero_lp = survival.predict(fit, reference="zero")
    zero_prefix_lp = survival.predict(fit, reference="z")
    strata_prefix_lp = survival.predict(fit, reference="st")

    assert zero_lp == pytest.approx(fit.linear_predictors)
    assert zero_prefix_lp == pytest.approx(zero_lp)
    assert sample_lp == pytest.approx([beta * (value - sample_mean) for value in data["x1"]])
    assert sample_prefix_lp == pytest.approx(sample_lp)
    assert default_lp == pytest.approx(
        [
            beta * (value - strata_means[group])
            for value, group in zip(data["x1"], data["group"], strict=True)
        ]
    )
    assert strata_prefix_lp == pytest.approx(default_lp)


def test_predict_coxph_reference_strata_uses_formula_newdata_strata():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    beta = fit.coefficients[0][0]
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}

    assert survival.predict(fit, newdata) == pytest.approx(
        [beta * (0.5 - 0.375), beta * (0.5 - 1.05)]
    )
    assert survival.predict(fit, newdata, reference="sample") == pytest.approx(
        [beta * (0.5 - sum(data["x1"]) / len(data["x1"]))] * 2
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx([beta * 0.5] * 2)
    with_se = survival.predict(fit, newdata, se_fit=True)
    var = fit.information_matrix[0][0]
    assert with_se.fit == pytest.approx([beta * (0.5 - 0.375), beta * (0.5 - 1.05)])
    assert with_se.se_fit == pytest.approx(
        [abs(0.5 - 0.375) * math.sqrt(var), abs(0.5 - 1.05) * math.sqrt(var)]
    )
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(fit, {"x1": [0.5], "group": ["C"]})
    with pytest.raises(ValueError, match="newdata strata are required"):
        survival.predict(fit, [[0.5]])


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


def test_predict_coxph_survival_uses_formula_newdata_strata():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + strata(group)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}
    rows = [[0.5], [0.5]]
    strata = [0, 1]
    full_times, full_curves = fit.survival_curve_with_strata(rows, strata, True)
    requested_times = [0.5, *full_times[:2], full_times[-1]]

    times, curves = survival.predict(
        fit,
        newdata,
        type="survival",
        centered=True,
        times=requested_times,
    )

    assert times == pytest.approx(requested_times)
    for actual, expected_curve in zip(curves, full_curves, strict=True):
        expected = []
        for time in requested_times:
            pos = bisect_right(full_times, time)
            expected.append(1.0 if pos == 0 else expected_curve[pos - 1])
        assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(fit, {"x1": [0.5], "group": ["C"]}, type="survival")


def test_predict_coxph_expected_accepts_formula_newdata_response():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 4.5, 8.0],
        "status": [0, 0, 1],
        "x1": [0.2, 0.8, 1.4],
        "x2": [1.0, 0.7, 0.2],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = [
        0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1]
        for time in newdata["time"]
    ]
    collapse = ["A", "A", "B"]

    actual = survival.predict(fit, newdata, type="expected")

    assert actual == pytest.approx(expected)
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    assert survival.predict(fit, newdata, type="expected", collapse=collapse) == pytest.approx(
        [expected[0] + expected[1], expected[2]]
    )
    assert survival.predict(fit, newdata, type="survival", collapse=collapse) == pytest.approx(
        [math.exp(-expected[0]) + math.exp(-expected[1]), math.exp(-expected[2])]
    )
    newdata_with_different_status = {**newdata, "status": [1, 1, 0]}
    assert survival.predict(fit, newdata_with_different_status, type="expected") == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="same length as predictions"):
        survival.predict(fit, newdata, type="expected", collapse=["A"])
    expected_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    survival_with_se = survival.predict(fit, newdata, type="survival", se_fit=True)
    assert expected_with_se.fit == pytest.approx(expected)
    assert len(expected_with_se.se_fit) == len(expected)
    assert all(math.isfinite(value) and value >= 0.0 for value in expected_with_se.se_fit)
    assert survival_with_se.fit == pytest.approx([math.exp(-value) for value in expected])
    assert survival_with_se.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(expected_with_se.se_fit, expected, strict=True)]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(fit, {"x1": [0.2], "x2": [1.0]}, type="expected")


def test_predict_coxph_expected_accepts_formula_response_comparison_newdata():
    data = _toy_data()
    data["r_status"] = [2 if status == 1 else 1 for status in data["status"]]
    fit = survival.coxph(
        "Surv(time, r_status == 2) ~ x1 + x2",
        data=data,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 4.5, 8.0],
        "r_status": [1, 1, 2],
        "x1": [0.2, 0.8, 1.4],
        "x2": [1.0, 0.7, 0.2],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = [
        0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1]
        for time in newdata["time"]
    ]

    assert survival.predict(fit, newdata, type="expected") == pytest.approx(expected)
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(fit, {"time": [1.0], "x1": [0.2], "x2": [1.0]}, type="expected")


def test_predict_coxph_expected_and_survival_se_use_baseline_variance():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0],
        "status": [1, 1, 0, 1],
    }
    fit = survival.coxph(
        "Surv(time, status) ~ 1",
        data=data,
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "time": [0.5, 2.5, 4.0],
        "status": [0, 0, 1],
    }
    expected = [0.0, 1.0 / 4.0 + 1.0 / 3.0, 1.0 / 4.0 + 1.0 / 3.0 + 1.0]
    expected_var = [0.0, 1.0 / 16.0 + 1.0 / 9.0, 1.0 / 16.0 + 1.0 / 9.0 + 1.0]
    expected_se = [math.sqrt(value) for value in expected_var]

    expected_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    survival_with_se = survival.predict(fit, newdata, type="survival", se_fit=True)
    collapsed = survival.predict(
        fit,
        newdata,
        type="expected",
        collapse=["A", "A", "B"],
        se_fit=True,
    )
    training = survival.predict(fit, type="expected", se_fit=True)
    training_survival = survival.predict(fit, type="survival", se_fit=True)

    assert expected_with_se.fit == pytest.approx(expected)
    assert expected_with_se.se_fit == pytest.approx(expected_se)
    assert survival_with_se.fit == pytest.approx([math.exp(-value) for value in expected])
    assert survival_with_se.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(expected_se, expected, strict=True)]
    )
    assert collapsed.fit == pytest.approx([expected[0] + expected[1], expected[2]])
    assert collapsed.se_fit == pytest.approx(
        [math.sqrt(expected_se[0] ** 2 + expected_se[1] ** 2), expected_se[2]]
    )
    assert training.fit == pytest.approx(
        [expected[1] - 1.0 / 3.0, expected[1], expected[1], expected[2]]
    )
    assert training.se_fit == pytest.approx(
        [
            math.sqrt(expected_var[1] - 1.0 / 9.0),
            expected_se[1],
            expected_se[1],
            expected_se[2],
        ]
    )
    assert training_survival.fit == pytest.approx([math.exp(-value) for value in training.fit])
    assert training_survival.se_fit == pytest.approx(
        [se * math.exp(-value) for se, value in zip(training.se_fit, training.fit, strict=True)]
    )


def test_predict_coxph_expected_accepts_counting_newdata_response():
    data = _counting_cox_data()
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    newdata = {
        "start": [0.0, 2.5],
        "stop": [4.0, 6.0],
        "status": [0, 1],
        "x1": [0.2, 1.1],
    }
    base_times, base_hazards = fit.basehaz(False)
    expected = []
    for start, stop in zip(newdata["start"], newdata["stop"], strict=True):
        start_pos = bisect_right(base_times, start)
        stop_pos = bisect_right(base_times, stop)
        start_hazard = 0.0 if start_pos == 0 else base_hazards[start_pos - 1]
        stop_hazard = 0.0 if stop_pos == 0 else base_hazards[stop_pos - 1]
        expected.append(stop_hazard - start_hazard)

    actual = survival.predict(fit, newdata, type="expected")
    actual_with_se = survival.predict(fit, newdata, type="expected", se_fit=True)
    baseline = survival.r_api._cox_expected_baseline_by_stratum(fit)[0]
    variance = fit.information_matrix
    means = fit.means
    linear_predictors = survival.predict(
        fit,
        [[value] for value in newdata["x1"]],
        reference="zero",
    )
    expected_se = []
    for start, stop, x1, linear_predictor in zip(
        newdata["start"],
        newdata["stop"],
        newdata["x1"],
        linear_predictors,
        strict=True,
    ):
        start_hazard, start_varhaz, start_xbar = survival.r_api._cox_expected_baseline_at(
            baseline,
            start,
            1,
        )
        stop_hazard, stop_varhaz, stop_xbar = survival.r_api._cox_expected_baseline_at(
            baseline,
            stop,
            1,
        )
        centered_x = x1 - means[0]
        start_delta = start_hazard * centered_x - start_xbar[0]
        stop_delta = stop_hazard * centered_x - stop_xbar[0]
        interval_delta = stop_delta - start_delta
        expected_var = stop_varhaz - start_varhaz + interval_delta * variance[0][0] * interval_delta
        expected_se.append(math.sqrt(max(expected_var, 0.0)) * math.exp(linear_predictor))
    conditioned = survival.survfit(
        fit,
        newdata={"x1": [newdata["x1"][1]]},
        start_time=newdata["start"][1],
        conf_type="none",
    )

    assert actual == pytest.approx(expected)
    assert actual_with_se.fit == pytest.approx(expected)
    assert actual_with_se.se_fit == pytest.approx(expected_se)
    assert conditioned.std_chaz[0][-1] == pytest.approx(expected_se[1])
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-value) for value in expected]
    )
    with pytest.raises(ValueError, match="response columns"):
        survival.predict(
            fit,
            {"stop": [4.0], "status": [1], "x1": [0.2]},
            type="expected",
        )


def test_predict_coxph_expected_uses_formula_newdata_strata():
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
    newdata = {
        "time": [2.5, 4.0],
        "status": [0, 1],
        "group": ["A", "B"],
        "x1": [0.5, 0.5],
    }

    actual = survival.predict(fit, newdata, type="expected")

    assert actual == pytest.approx([5.0 / 6.0, 1.5])
    assert survival.predict(fit, newdata, type="survival") == pytest.approx(
        [math.exp(-5.0 / 6.0), math.exp(-1.5)]
    )
    with pytest.raises(ValueError, match="partial formula response"):
        survival.predict(
            fit,
            {"time": [2.0], "group": ["A"], "x1": [0.5]},
            type="survival",
        )
    with pytest.raises(ValueError, match="unknown strata level"):
        survival.predict(
            fit,
            {"time": [2.0], "status": [1], "group": ["C"], "x1": [0.5]},
            type="expected",
        )


def test_basehaz_accepts_fitted_coxph_model_and_raw_inputs():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=data, max_iter=10, eps=1e-5)
    r_status = [2 if event else 1 for event in data["status"]]
    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    method_times, method_hazard = fit.basehaz(False)
    raw_times, raw_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
    )
    raw_r_times, raw_r_hazard = survival.basehaz(
        data["time"],
        r_status,
        fit.linear_predictors,
        False,
    )
    keyword_times, keyword_hazard = survival.basehaz(
        time=data["time"],
        status=data["status"],
        linear_predictors=fit.linear_predictors,
        centered=False,
    )
    keyword_r_times, keyword_r_hazard = survival.basehaz(
        time=data["time"],
        status=r_status,
        linear_predictors=fit.linear_predictors,
        centered=False,
    )

    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(method_times, time)) == 0 else method_hazard[pos - 1]
        for time in data["time"]
    ]
    assert fitted_times == pytest.approx(data["time"])
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)
    assert raw_times == pytest.approx(method_times)
    assert raw_hazard == pytest.approx(method_hazard)
    assert keyword_times == pytest.approx(method_times)
    assert keyword_hazard == pytest.approx(method_hazard)
    assert raw_r_times == pytest.approx(method_times)
    assert raw_r_hazard == pytest.approx(method_hazard)
    assert keyword_r_times == pytest.approx(method_times)
    assert keyword_r_hazard == pytest.approx(method_hazard)


def test_fitted_basehaz_uses_scaled_risk_scores_for_large_linear_predictors():
    fit = survival.regression.coxph_fit(
        time=[1.0, 2.0, 3.0],
        status=[1, 1, 1],
        covariates=[[1.0], [709.0 / 710.0], [708.0 / 710.0]],
        initial_beta=[710.0],
        max_iter=0,
        method="breslow",
    )

    times, hazard = fit.basehaz(False)
    expected_first = math.exp(-710.0) / (1.0 + math.exp(-1.0) + math.exp(-2.0))

    assert fit.linear_predictors == pytest.approx([710.0, 709.0, 708.0])
    assert times == pytest.approx([1.0, 2.0, 3.0])
    assert hazard[0] == pytest.approx(expected_first, rel=1e-12, abs=0.0)
    assert 0.0 < hazard[0] < hazard[1] < hazard[2]


def test_basehaz_accepts_fitted_coxph_newdata():
    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=10)
    newdata = {"x1": [0.5, 1.0], "x2": [0.8, 0.2]}

    result = survival.basehaz(fit, newdata=newdata, centered=False)
    positional = survival.basehaz(fit, newdata, centered=False)
    centered = survival.basehaz(fit, newdata=newdata, centered=True)
    survfit = survival.survfit(fit, newdata=newdata)
    unpacked_times, unpacked_hazards = result
    single = survival.basehaz(fit, newdata={"x1": [0.5], "x2": [0.8]}, centered=False)
    base_times, base_hazards = fit.basehaz(False)
    linear_predictors = survival.predict(fit, newdata, reference="zero")
    expected_uncentered = [
        [
            (0.0 if (pos := bisect_right(base_times, time)) == 0 else base_hazards[pos - 1])
            * math.exp(linear_predictor)
            for time in result.time
        ]
        for linear_predictor in linear_predictors
    ]

    assert isinstance(result, survival.r_api.CoxBaseHazardResult)
    assert result.centered is True
    assert positional.centered is True
    assert centered.centered is True
    assert result.time == pytest.approx(survfit.time)
    assert positional.time == pytest.approx(result.time)
    assert centered.time == pytest.approx(survfit.time)
    assert unpacked_times == pytest.approx(result.time)
    assert result.hazard == result.cumhaz
    assert result.cumulative_hazard == result.cumhaz
    assert len(result.cumhaz) == 2
    for actual, positional_curve, expected in zip(
        result.cumhaz,
        positional.cumhaz,
        expected_uncentered,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
        assert positional_curve == pytest.approx(expected)
    for actual, expected in zip(centered.cumhaz, survfit.cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    assert result.cumhaz[0] == pytest.approx(centered.cumhaz[0])
    assert unpacked_hazards == result.cumhaz
    assert single.centered is True
    assert single.time == pytest.approx(result.time)
    assert single.cumhaz == pytest.approx(expected_uncentered[0])
    with pytest.raises(ValueError, match="positional newdata"):
        survival.basehaz(fit, newdata, newdata=newdata)


def test_basehaz_uses_fitted_coxph_weights():
    data = _toy_data()
    weights = [2.0, 1.0, 0.5, 1.5, 1.0, 2.0, 0.75, 1.25]
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        initial_beta=[0.0, 0.0],
        max_iter=0,
        method="breslow",
    )
    fitted_times, fitted_hazard = survival.basehaz(fit, centered=False)
    raw_times, raw_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
        weights=weights,
    )
    unweighted_times, unweighted_hazard = survival.basehaz(
        data["time"],
        data["status"],
        fit.linear_predictors,
        False,
    )

    assert fit.weights == pytest.approx(weights)
    expected_fitted_hazard = [
        0.0 if (pos := bisect_right(raw_times, time)) == 0 else raw_hazard[pos - 1]
        for time in data["time"]
    ]
    assert fitted_times == pytest.approx(data["time"])
    assert fitted_hazard == pytest.approx(expected_fitted_hazard)
    assert unweighted_times == pytest.approx(raw_times)
    assert fitted_hazard[0] == pytest.approx(weights[0] / sum(weights))
    assert raw_hazard != pytest.approx(unweighted_hazard)


def test_basehaz_uses_counting_process_weights():
    data = _counting_cox_data()
    weights = [2.0, 1.0, 3.0, 1.0, 4.0, 1.0]
    fit = survival.coxph(
        "Surv(start, stop, status) ~ x1",
        data=data,
        weights=weights,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    fitted_times, fitted_hazard = fit.basehaz(False)
    raw_times, raw_hazard = survival.basehaz(
        data["stop"],
        data["status"],
        fit.linear_predictors,
        False,
        entry_times=data["start"],
        weights=weights,
    )

    assert fit.weights == pytest.approx(weights)
    assert fitted_times == pytest.approx(raw_times)
    assert fitted_hazard == pytest.approx(raw_hazard)
    assert fitted_hazard[0] == pytest.approx((weights[0] + weights[1]) / 10.0)


def test_basehaz_uses_fitted_coxph_strata():
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

    times, hazards, strata = fit.basehaz_with_strata(False)
    result = survival.basehaz(fit, centered=False)
    unpacked_times, unpacked_hazards = result
    expected_event_times = [1.0, 2.0, 3.0, 4.0]
    expected_event_hazards = [1.0 / 3.0, 1.0 / 3.0 + 1.0 / 2.0, 1.0 / 2.0, 1.5]
    expected_times = [1.0, 2.0, 4.0, 1.0, 3.0, 4.0]
    expected_hazards = [1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0, 0.0, 0.5, 1.5]

    assert times == pytest.approx(expected_event_times)
    assert hazards == pytest.approx(expected_event_hazards)
    assert strata == [0, 0, 1, 1]
    assert unpacked_times == pytest.approx(expected_times)
    assert unpacked_hazards == pytest.approx(expected_hazards)
    assert result.time == pytest.approx(expected_times)
    assert result.cumhaz == pytest.approx(expected_hazards)
    assert result.hazard == pytest.approx(expected_hazards)
    assert result.cumulative_hazard == pytest.approx(expected_hazards)
    assert result.strata == [0, 0, 0, 1, 1, 1]

    expected = survival.predict(fit, type="expected")
    assert expected == pytest.approx([1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0, 0.0, 0.5, 1.5])


def test_basehaz_newdata_uses_fitted_coxph_strata():
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
    newdata = {"x1": [0.5, 0.5], "group": ["A", "B"]}

    result = survival.basehaz(fit, newdata=newdata, centered=False)
    survfit = survival.survfit(fit, newdata=newdata)
    single = survival.basehaz(fit, newdata={"x1": [0.5], "group": ["B"]})

    assert result.centered is True
    assert result.curve_strata == [0, 1]
    assert result.time == pytest.approx(survfit.time)
    for actual, expected in zip(result.cumhaz, survfit.cumhaz, strict=True):
        assert actual == pytest.approx(expected)
    assert single.time == pytest.approx([1.0, 3.0, 4.0])
    assert single.cumhaz == pytest.approx([0.0, 0.5, 1.5])
    assert single.strata == [1, 1, 1]
    assert single.curve_strata == [1]


def test_basehaz_uses_efron_tie_increments_for_fitted_coxph():
    data = {
        "time": [1.0, 1.0, 2.0, 3.0],
        "status": [1, 1, 0, 1],
        "x1": [0.2, 0.8, 0.4, 1.1],
    }
    breslow = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="breslow",
    )
    efron = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        initial_beta=[0.0],
        max_iter=0,
        method="efron",
    )

    breslow_times, breslow_hazard = breslow.basehaz(False)
    efron_times, efron_hazard = efron.basehaz(False)
    detail = survival.coxph_detail(efron)

    assert breslow_times == pytest.approx([1.0, 3.0])
    assert efron_times == pytest.approx(breslow_times)
    assert breslow_hazard == pytest.approx([2.0 / 4.0, 2.0 / 4.0 + 1.0 / 1.0])
    assert efron_hazard == pytest.approx([1.0 / 4.0 + 1.0 / 3.0, 1.0 / 4.0 + 1.0 / 3.0 + 1.0])
    assert efron_hazard[0] > breslow_hazard[0]
    assert efron_hazard == pytest.approx(detail.cumulative_hazard)
    surv = survival.survfit(efron)
    event_only = survival.survfit(efron, censor=False)
    assert surv.time == pytest.approx([1.0, 2.0, 3.0])
    assert surv.cumhaz[0] == pytest.approx([efron_hazard[0], efron_hazard[0], efron_hazard[1]])
    assert event_only.cumhaz[0] == pytest.approx(efron_hazard)


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


def test_coxph_advanced_inputs_use_rust_optimizer():
    data = _toy_data()
    weights = [1.0, 1.5, 1.0, 0.8, 1.2, 1.0, 0.9, 1.1]
    offset = [0.1, 0.1, 0.0, -0.1, 0.0, -0.1, 0.1, 0.0]
    fit = survival.coxph(
        "Surv(time, status) ~ x1",
        data=data,
        weights=weights,
        offset=offset,
        init=[0.0],
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        weights=weights,
        offset=offset,
        initial_beta=[0.0],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)


def test_coxph_formula_offset_uses_rust_optimizer():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=data["offset"],
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert fit.risk_scores == pytest.approx(low_level.risk_scores)

    transformed = {**data, "exposure": [math.exp(value) for value in data["offset"]]}
    transformed_fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=transformed,
        max_iter=10,
        eps=1e-5,
    )
    arithmetic_offsets = [
        offset + x2 for offset, x2 in zip(data["offset"], data["x2"], strict=True)
    ]
    arithmetic_fit = survival.coxph(
        "Surv(time, status) ~ x1 + offset(offset + x2)",
        data=data,
        max_iter=10,
        eps=1e-5,
    )
    arithmetic_low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[value] for value in data["x1"]],
        offset=arithmetic_offsets,
        max_iter=10,
        eps=1e-5,
    )

    assert transformed_fit.coefficients[0] == pytest.approx(low_level.coefficients[0])
    assert transformed_fit.risk_scores == pytest.approx(low_level.risk_scores)
    assert arithmetic_fit.coefficients[0] == pytest.approx(arithmetic_low_level.coefficients[0])
    assert arithmetic_fit.risk_scores == pytest.approx(arithmetic_low_level.risk_scores)


def test_coxph_formula_accepts_intercept_only_rhs():
    data = _toy_data()
    fit = survival.coxph("Surv(time, status) ~ 1", data=data, max_iter=0)
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        max_iter=0,
    )

    assert fit.coefficients == [[]]
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert survival.predict(fit, {"row": [0, 1]}) == pytest.approx([0.0, 0.0])
    assert survival.predict(fit, {"row": [0, 1]}, type="risk") == pytest.approx([1.0, 1.0])
    times, curves = survival.survfit(fit, newdata={"row": [0]})
    assert times
    assert len(curves) == 1


def test_coxph_formula_accepts_offset_only_rhs():
    data = _toy_data()
    fit = survival.coxph(
        "Surv(time, status) ~ offset(offset)",
        data=data,
        max_iter=0,
    )
    low_level = survival.regression.coxph_fit(
        time=data["time"],
        status=data["status"],
        covariates=[[] for _ in data["time"]],
        offset=data["offset"],
        max_iter=0,
    )

    assert fit.coefficients == [[]]
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.linear_predictors == pytest.approx(data["offset"])
    assert fit.risk_scores == pytest.approx([math.exp(value) for value in data["offset"]])

    newdata = {"offset": [0.2, -0.1]}
    offset_center = sum(data["offset"]) / len(data["offset"])
    assert survival.predict(fit, newdata) == pytest.approx(
        [0.2 - offset_center, -0.1 - offset_center]
    )
    assert survival.predict(fit, newdata, reference="zero") == pytest.approx([0.2, -0.1])
    assert survival.predict(fit, newdata, type="risk") == pytest.approx(
        [math.exp(0.2 - offset_center), math.exp(-0.1 - offset_center)]
    )


def test_survreg_formula_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_fixed_scale_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        scale=1.25,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        fixed_scale=1.25,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.location_coefficients == pytest.approx(low_level.location_coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.scale == pytest.approx(1.25)
    assert fit.scales == pytest.approx([1.25])
    assert len(fit.variance_matrix) == len(fit.coefficients)
    assert all(len(row) == len(fit.coefficients) for row in fit.variance_matrix)
    assert len(fit.score_vector) == len(fit.coefficients)


def test_survreg_score_true_exposes_score_vector_alias():
    data = _toy_data()
    formula_fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        score=True,
        max_iter=10,
        eps=1e-5,
    )
    formula_low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert formula_fit.score == pytest.approx(formula_low_level.score_vector)
    assert formula_fit.score_vector == pytest.approx(formula_low_level.score_vector)

    no_score = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        score=False,
        max_iter=10,
        eps=1e-5,
    )
    assert not hasattr(no_score, "score")

    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    matrix_fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        score=True,
        max_iter=10,
        eps=1e-5,
    )
    matrix_low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=rows,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert matrix_fit.score == pytest.approx(matrix_low_level.score_vector)
    assert matrix_fit.score_vector == pytest.approx(matrix_low_level.score_vector)


def test_survreg_cluster_computes_robust_variance():
    data = {**_toy_data(), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"]}
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    plain = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    formula_clustered = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + cluster(subject)",
        data=data,
        dist="weibull",
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )
    explicit_cluster = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    matrix_cluster = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=_with_intercept(rows),
        distribution="weibull",
        cluster=data["subject"],
        max_iter=10,
        eps=1e-5,
    )
    singleton_robust = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        robust=True,
        max_iter=10,
        eps=1e-5,
    )
    nonrobust_cluster = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        cluster=data["subject"],
        robust=False,
        max_iter=10,
        eps=1e-5,
    )

    expected_robust = _manual_survreg_robust_variance(plain, data["subject"])
    singleton_expected = _manual_survreg_robust_variance(plain, list(range(len(data["time"]))))

    assert formula_clustered.robust is True
    assert formula_clustered.cluster == data["subject"]
    assert formula_clustered.coefficients == pytest.approx(plain.coefficients)
    assert explicit_cluster.coefficients == pytest.approx(plain.coefficients)
    assert matrix_cluster.coefficients == pytest.approx(plain.coefficients)
    assert formula_clustered.model["subject"] == data["subject"]
    assert formula_clustered.model["(cluster)"] == data["subject"]
    for actual, expected in zip(formula_clustered.x, _with_intercept(rows), strict=True):
        assert actual == pytest.approx(expected)

    for actual, expected in zip(
        formula_clustered.naive_variance,
        plain.variance_matrix,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(formula_clustered.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(explicit_cluster.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(matrix_cluster.variance_matrix, expected_robust, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(singleton_robust.variance_matrix, singleton_expected, strict=True):
        assert actual == pytest.approx(expected)

    assert nonrobust_cluster.robust is False
    for actual, expected in zip(
        nonrobust_cluster.variance_matrix,
        plain.variance_matrix,
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_survreg_accepts_r_style_control_mapping():
    data = _toy_data()
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
        tol_chol=1e-8,
    )
    controlled = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        scale=0,
        parms=None,
        control={
            "maxiter": 10,
            "rel.tolerance": 1e-5,
            "toler.chol": 1e-8,
            "debug": 0,
            "outer.max": 10,
        },
    )

    assert controlled.coefficients == pytest.approx(explicit.coefficients)
    assert controlled.log_likelihood == pytest.approx(explicit.log_likelihood)

    nondefault_ignored = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        scale=0,
        control={
            "maxiter": 10,
            "rel.tolerance": 1e-5,
            "toler.chol": 1e-8,
            "debug": 1,
            "outer.max": 2,
        },
    )

    assert nondefault_ignored.coefficients == pytest.approx(explicit.coefficients)
    assert nondefault_ignored.log_likelihood == pytest.approx(explicit.log_likelihood)

    matrix_controlled = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        distribution="weibull",
        scale=0.0,
        control={"maxiter": 10, "rel.tolerance": 1e-5, "toler.chol": 1e-8},
    )
    matrix_explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=[[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
        tol_chol=1e-8,
    )

    assert matrix_controlled.coefficients == pytest.approx(matrix_explicit.coefficients)
    assert matrix_controlled.log_likelihood == pytest.approx(matrix_explicit.log_likelihood)


def test_survreg_accepts_r_style_formula_storage_flags():
    data = _toy_data()
    default = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit_defaults = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        model=False,
        x=False,
        y=True,
        robust=False,
        cluster=None,
        score=False,
        max_iter=10,
        eps=1e-5,
    )
    with_model = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + offset(offset)",
        data=data,
        dist="weibull",
        model=True,
        x=True,
        max_iter=10,
        eps=1e-5,
    )

    assert explicit_defaults.coefficients == pytest.approx(default.coefficients)
    assert explicit_defaults.log_likelihood == pytest.approx(default.log_likelihood)
    assert explicit_defaults.y.time == pytest.approx(data["time"])
    assert explicit_defaults.y.event == tuple(data["status"])
    assert not hasattr(explicit_defaults, "x")
    assert not hasattr(explicit_defaults, "model")
    assert not hasattr(explicit_defaults, "score")

    model_frame = with_model.model
    assert model_frame["Surv(time, status)"].time == pytest.approx(data["time"])
    assert model_frame["Surv(time, status)"].event == tuple(data["status"])
    assert model_frame["time"] == pytest.approx(data["time"])
    assert model_frame["status"] == data["status"]
    assert model_frame["x1"] == pytest.approx(data["x1"])
    assert model_frame["x2"] == pytest.approx(data["x2"])
    assert model_frame["offset"] == pytest.approx(data["offset"])
    assert model_frame["(offset)"] == pytest.approx(data["offset"])

    with_x = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        x=True,
        y=False,
        max_iter=10,
        eps=1e-5,
    )
    expected_x = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    for actual, expected in zip(with_x.x, expected_x, strict=True):
        assert actual == pytest.approx(expected)
    assert not hasattr(with_x, "y")


def test_survreg_model_true_stores_matrix_inputs():
    data = _toy_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]

    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        model=True,
        max_iter=10,
        eps=1e-5,
    )

    assert fit.model["time"] == pytest.approx(data["time"])
    assert fit.model["status"] == pytest.approx([float(value) for value in data["status"]])
    for actual, expected in zip(fit.model["x"], rows, strict=True):
        assert actual == pytest.approx(expected)


def test_survreg_accepts_r_style_init_alias():
    data = _toy_data()
    initial = [0.15, -0.1, 0.05, 0.0]
    alias = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        init=initial,
        max_iter=0,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        initial_beta=initial,
        max_iter=0,
    )

    assert alias.coefficients == pytest.approx(explicit.coefficients)
    assert alias.log_likelihood == pytest.approx(explicit.log_likelihood)

    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    matrix_alias = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        init=initial[1:],
        max_iter=0,
    )
    matrix_explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        initial=initial[1:],
        max_iter=0,
    )

    assert matrix_alias.coefficients == pytest.approx(matrix_explicit.coefficients)
    assert matrix_alias.log_likelihood == pytest.approx(matrix_explicit.log_likelihood)


def test_survreg_distribution_accepts_r_style_prefixes_and_aliases():
    data = _toy_data()
    full = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    abbreviated = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="wei",
        max_iter=10,
        eps=1e-5,
    )

    assert abbreviated.distribution == "weibull"
    assert abbreviated.coefficients == pytest.approx(full.coefficients)
    assert abbreviated.log_likelihood == pytest.approx(full.log_likelihood)

    expected = {
        "exp": "exponential",
        "ext": "extreme_value",
        "extreme": "extreme_value",
        "extreme value": "extreme_value",
        "extreme_value": "extreme_value",
        "gauss": "gaussian",
        "normal": "gaussian",
        "logi": "logistic",
        "logn": "lognormal",
        "logl": "loglogistic",
        "log-normal": "lognormal",
        "log-logistic": "loglogistic",
    }
    for dist, distribution in expected.items():
        fit = survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist=dist,
            max_iter=10,
            eps=1e-5,
        )
        assert fit.distribution == distribution

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist="log",
            max_iter=10,
            eps=1e-5,
        )

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=data,
            dist="ex",
            max_iter=10,
            eps=1e-5,
        )


def test_survreg_loglik_and_response_transform_follow_r_distribution_scale():
    data = _toy_data()

    weibull = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=80,
        eps=1e-8,
    )
    expected_loglik = weibull.log_likelihood - sum(
        math.log(time)
        for time, event in zip(data["time"], data["status"], strict=True)
        if event == 1
    )
    assert survival.loglik(weibull) == pytest.approx(expected_loglik)

    lognormal = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="lognormal",
        max_iter=80,
        eps=1e-8,
    )
    lognormal_lp = survival.predict(lognormal, type="lp")
    assert survival.predict(lognormal, type="response") == pytest.approx(
        [math.exp(value) for value in lognormal_lp]
    )

    gaussian = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=80,
        eps=1e-8,
    )
    assert survival.loglik(gaussian) == pytest.approx(gaussian.log_likelihood)
    assert survival.predict(gaussian, type="response") == pytest.approx(
        survival.predict(gaussian, type="lp")
    )


def test_low_level_survreg_rejects_invalid_numeric_inputs():
    kwargs = {
        "time": [1.0, 2.0, 3.0],
        "status": [1.0, 0.0, 1.0],
        "covariates": [[0.2, 1.0], [0.4, 0.9], [0.1, 1.1]],
        "distribution": "weibull",
        "max_iter": 1,
    }

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.regression.survreg(**{**kwargs, "time": [1.0, float("nan"), 3.0]})

    with pytest.raises(ValueError, match="must be positive"):
        survival.regression.survreg(**{**kwargs, "time": [1.0, 0.0, 3.0]})

    with pytest.raises(ValueError, match="status must contain only 0/1/2/3"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 4.0, 0.0]})

    with pytest.raises(ValueError, match=r"covariates\[1\]\[0\] contains non-finite"):
        survival.regression.survreg(
            **{**kwargs, "covariates": [[0.2, 1.0], [float("inf"), 0.9], [0.1, 1.1]]}
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.regression.survreg(**{**kwargs, "weights": [1.0, -1.0, 1.0]})

    with pytest.raises(ValueError, match="at least one positive"):
        survival.regression.survreg(**{**kwargs, "weights": [0.0, 0.0, 0.0]})

    with pytest.raises(ValueError, match="offsets contains non-finite"):
        survival.regression.survreg(**{**kwargs, "offsets": [0.0, float("nan"), 0.0]})

    with pytest.raises(ValueError, match="initial_beta contains non-finite"):
        survival.regression.survreg(**{**kwargs, "initial_beta": [0.0, 0.0, float("nan")]})

    with pytest.raises(ValueError, match="fixed_scale must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "fixed_scale": 0.0})

    with pytest.raises(ValueError, match="fixed_scale must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "fixed_scale": float("nan")})

    with pytest.raises(ValueError, match="cannot have both a fixed scale and strata"):
        survival.regression.survreg(**{**kwargs, "strata": [0, 1, 1], "fixed_scale": 1.0})

    with pytest.raises(ValueError, match="initial_beta has 3 values but model expects 2"):
        survival.regression.survreg(
            **{**kwargs, "initial_beta": [0.0, 0.0, 0.0], "fixed_scale": 1.0}
        )

    with pytest.raises(ValueError, match="eps must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "eps": 0.0})

    with pytest.raises(ValueError, match="tol_chol must be a finite positive value"):
        survival.regression.survreg(**{**kwargs, "tol_chol": float("nan")})

    with pytest.raises(ValueError, match="distribution must be one of"):
        survival.regression.survreg(**{**kwargs, "distribution": "mystery"})

    with pytest.raises(ValueError, match="time2 is required"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 3.0, 0.0]})

    with pytest.raises(ValueError, match="time2 has 2"):
        survival.regression.survreg(**{**kwargs, "status": [1.0, 3.0, 0.0], "time2": [1.0, 2.5]})

    with pytest.raises(ValueError, match="non-finite interval endpoint"):
        survival.regression.survreg(
            **{
                **kwargs,
                "status": [1.0, 3.0, 0.0],
                "time2": [1.0, float("inf"), 3.0],
            }
        )

    with pytest.raises(ValueError, match="greater than time"):
        survival.regression.survreg(
            **{**kwargs, "status": [1.0, 3.0, 0.0], "time2": [1.0, 2.0, 3.0]}
        )


def test_low_level_survreg_accepts_left_and_interval_censoring():
    fit = survival.regression.survreg(
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        time2=[1.0, 2.0, 3.0, 4.5, 5.0],
        status=[1.0, 2.0, 0.0, 3.0, 1.0],
        covariates=[[0.2], [0.4], [0.1], [0.8], [1.0]],
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.status == [1, 2, 0, 3, 1]
    assert fit.time2 == pytest.approx([1.0, 2.0, 3.0, 4.5, 5.0])
    assert math.isfinite(fit.log_likelihood)
    assert len(fit.coefficients) == 2


def test_survreg_left_censored_formula_matches_low_level_binding():
    data = {
        "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "status": [0, 1, 0, 1, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(time, status, type='left') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[2.0, 1.0, 2.0, 1.0, 1.0],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.status == [2, 1, 2, 1, 1]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval_formula_matches_low_level_binding():
    data = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0],
        "right": [1.0, 2.0, 3.0, 4.5, 5.0],
        "status": [1, 2, 0, 3, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(left, right, status, type='interval') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["left"],
        time2=data["right"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.time2 == pytest.approx(low_level.time2)
    assert fit.status == data["status"]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval2_formula_derives_low_level_status_codes():
    data = {
        "left": [float("-inf"), 2.0, 3.0, 4.0],
        "right": [1.0, 5.0, 3.0, float("inf")],
        "x1": [0.2, 0.4, 0.1, 0.8],
    }
    fit = survival.survreg(
        "Surv(left, right, type='interval2') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=[1.0, 2.0, 3.0, 4.0],
        time2=[1.0, 5.0, 3.0, float("inf")],
        status=[2.0, 3.0, 1.0, 0.0],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        distribution="weibull",
        max_iter=5,
        eps=1e-5,
    )

    assert fit.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert fit.time2 == pytest.approx(low_level.time2)
    assert fit.status == [2, 3, 1, 0]
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_interval_residuals_support_scalar_and_influence_types():
    data = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0],
        "right": [1.0, 2.0, 3.0, 4.5, 5.0],
        "status": [1, 2, 0, 3, 1],
        "x1": [0.2, 0.4, 0.1, 0.8, 1.0],
    }
    fit = survival.survreg(
        "Surv(left, right, status, type='interval') ~ x1",
        data=data,
        dist="weibull",
        max_iter=5,
        eps=1e-5,
    )
    low_level = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="ldcase",
        time2=fit.time2,
    )
    low_response = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="response",
        time2=fit.time2,
    )
    low_deviance = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="deviance",
        time2=fit.time2,
    )
    low_working = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="working",
        time2=fit.time2,
    )
    matrix = survival.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=fit.time2,
    )
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    expected_ldcase = survival.survreg_influence_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldcase",
        True,
    )
    expected_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        False,
    )
    saturated = _weibull_saturated_center_loglik(fit.time, fit.time2, fit.status, fit.scale)
    expected_response = [
        math.exp(center) - math.exp(linear_predictor)
        for (center, _), linear_predictor in zip(saturated, fit.linear_predictors, strict=True)
    ]
    expected_deviance = _survreg_deviance_from_matrix(matrix, saturated)
    expected_working = [0.0 if abs(row[2]) <= 1e-12 else -row[1] / row[2] for row in matrix]
    expected_location_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        False,
        False,
    )
    low_location_dfbeta = survival.dfbeta_survreg(
        fit.time,
        fit.status,
        fit.covariates,
        fit.linear_predictors,
        fit.scale,
        location_vcov,
        fit.distribution,
        time2=fit.time2,
    )

    assert survival.r_api.residuals(fit, type="ldcase") == pytest.approx(expected_ldcase)
    assert survival.r_api.residuals(fit, type="ldc") == pytest.approx(expected_ldcase)
    assert low_response.residuals == pytest.approx(expected_response)
    assert fit.residuals("response").residuals == pytest.approx(expected_response)
    assert survival.r_api.residuals(fit, type="response") == pytest.approx(expected_response)
    assert low_deviance.residuals == pytest.approx(expected_deviance)
    assert fit.residuals("deviance").residuals == pytest.approx(expected_deviance)
    assert survival.r_api.residuals(fit, type="deviance") == pytest.approx(expected_deviance)
    assert low_working.residuals == pytest.approx(expected_working)
    assert fit.residuals("working").residuals == pytest.approx(expected_working)
    assert survival.r_api.residuals(fit, type="working") == pytest.approx(expected_working)
    for actual, expected in zip(
        survival.r_api.residuals(fit, type="dfbeta"),
        expected_dfbeta,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.dfbeta(), expected_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(low_location_dfbeta, expected_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    assert all(math.isfinite(value) for value in low_level.residuals)
    assert all(math.isfinite(value) for value in expected_ldcase)

    with pytest.raises(ValueError, match="ambiguous"):
        survival.r_api.residuals(fit, type="ld")


def test_low_level_survreg_residuals_handle_interval_ldcase_and_working():
    time = [1.0, 1.0, 1.0, 1.0]
    time2 = [1.0, 1.0, 2.0, 1.0]
    status = [1, 2, 3, 0]
    linear_pred = [0.0, 0.0, 0.0, 0.0]

    ldcase = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="ldcase",
        time2=time2,
    )

    assert ldcase.residuals[0] == pytest.approx(-1.0)
    assert ldcase.residuals[1] == pytest.approx(math.log(1.0 - math.exp(-1.0)))
    assert ldcase.residuals[2] == pytest.approx(math.log(math.exp(-1.0) - math.exp(-2.0)))
    assert ldcase.residuals[3] == pytest.approx(-1.0)

    covariates = [[1.0], [1.0], [1.0], [1.0]]
    matrix = survival.survreg_residual_matrix(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        time2=time2,
    )
    dfbeta = survival.dfbeta_survreg(
        time,
        status,
        covariates,
        linear_pred,
        1.0,
        [[1.0]],
        "weibull",
        time2=time2,
    )
    expected_dfbeta = survival.survreg_dfbeta_residuals(
        matrix,
        covariates,
        [1.0],
        [0, 0, 0, 0],
        [[1.0]],
        False,
        False,
    )
    for actual, expected in zip(dfbeta, expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)

    saturated = _weibull_saturated_center_loglik(time, time2, status, 1.0)
    response = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="response",
        time2=time2,
    )
    deviance = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="deviance",
        time2=time2,
    )
    assert response.residuals == pytest.approx(
        [
            math.exp(center) - math.exp(lp)
            for (center, _), lp in zip(saturated, linear_pred, strict=True)
        ]
    )
    assert deviance.residuals == pytest.approx(_survreg_deviance_from_matrix(matrix, saturated))

    working = survival.residuals_survreg(
        time,
        status,
        linear_pred,
        1.0,
        "weibull",
        residual_type="working",
        time2=time2,
    )
    expected_working = [0.0 if abs(row[2]) <= 1e-12 else -row[1] / row[2] for row in matrix]
    assert working.residuals == pytest.approx(expected_working)

    with pytest.raises(ValueError, match="time2 is required"):
        survival.residuals_survreg(
            time,
            status,
            linear_pred,
            1.0,
            "weibull",
            residual_type="ldcase",
        )


def test_survreg_fit_exposes_prediction_metadata_and_methods():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)
    expected_lp = [
        sum(
            value * coefficient
            for value, coefficient in zip(row, fit.location_coefficients, strict=True)
        )
        for row in design_rows
    ]
    training_rows = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    training_lp = [
        sum(
            value * coefficient
            for value, coefficient in zip(row, fit.location_coefficients, strict=True)
        )
        for row in training_rows
    ]

    assert fit.n_covariates == 3
    assert fit.n_strata == 1
    assert fit.distribution == "weibull"
    assert fit.scale > 0.0
    assert fit.scales == pytest.approx([fit.scale])
    assert fit.location_coefficients == pytest.approx(fit.coefficients[:3])
    assert fit.linear_predictors == pytest.approx(training_lp)
    for actual, expected in zip(fit.information_matrix, fit.fit.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(fit.variance_matrix, fit.fit.variance_matrix, strict=True):
        assert actual == pytest.approx(expected)

    lp = fit.predict(design_rows, "lp")
    response = fit.predict(design_rows)
    quantiles = fit.predict_quantile(design_rows, [0.25, 0.5])

    assert lp.predictions == pytest.approx(expected_lp)
    assert response.predictions == pytest.approx([math.exp(value) for value in expected_lp])
    assert quantiles.quantiles == pytest.approx([0.25, 0.5])
    assert len(quantiles.predictions) == 2
    assert all(len(row) == 2 for row in quantiles.predictions)


def test_predict_survreg_r_style_generic_types():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)

    lp = survival.predict(fit, rows, type="lp")
    prefix_lp = survival.predict(fit, rows, type="l")
    response = survival.predict(fit, rows)
    prefix_response = survival.predict(fit, rows, type="r")
    response_with_se = survival.predict(fit, rows, se_fit=True)
    dotted_response_with_se = survival.predict(fit, rows, **{"se.fit": True})
    terms = survival.predict(fit, rows, type="terms")
    prefix_terms = survival.predict(fit, rows, type="t")
    terms_with_se = survival.predict(fit, rows, type="terms", se_fit=True)
    x2_with_se = survival.predict(fit, rows, type="terms", terms="x2", se_fit=True)
    training_terms = survival.predict(fit, type="terms")
    training_terms_with_se = survival.predict(fit, type="terms", se_fit=True)
    training_x1_with_se = survival.predict(fit, type="terms", terms="x1", se_fit=True)
    median = survival.predict(fit, rows, type="quantile", p=0.5)
    prefix_median = survival.predict(fit, rows, type="q", p=0.5)
    median_with_se = survival.predict(fit, rows, type="quantile", p=0.5, se_fit=True)
    uquantile_with_se = survival.predict(fit, rows, type="uquantile", p=0.5, se_fit=True)
    prefix_uquantile_with_se = survival.predict(fit, rows, type="u", p=0.5, se_fit=True)
    default_bands = survival.predict(fit, rows, type="quantile")
    bands = survival.predict(fit, rows, type="quantile", quantiles=[0.25, 0.75])
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]
    full_vcov = [
        row[: fit.n_covariates + len(fit.scales)]
        for row in fit.variance_matrix[: fit.n_covariates + len(fit.scales)]
    ]
    training_rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    training_design_rows = _with_intercept(training_rows)
    means = [
        sum(row[col_idx] for row in training_design_rows) / len(training_design_rows)
        for col_idx in range(fit.n_covariates)
    ]

    assert lp == pytest.approx(fit.predict(design_rows, "lp").predictions)
    assert prefix_lp == pytest.approx(lp)
    assert response == pytest.approx(fit.predict(design_rows).predictions)
    assert prefix_response == pytest.approx(response)
    expected_linear_se = [
        math.sqrt(
            max(
                sum(
                    design_row[left] * location_vcov[left][right] * design_row[right]
                    for left in range(fit.n_covariates)
                    for right in range(fit.n_covariates)
                ),
                0.0,
            )
        )
        for design_row in design_rows
    ]
    assert response_with_se.fit == pytest.approx(response)
    assert response_with_se.se_fit == pytest.approx(
        [se * prediction for se, prediction in zip(expected_linear_se, response, strict=True)]
    )
    assert isinstance(dotted_response_with_se, survival.r_api.PredictResult)
    assert dotted_response_with_se.fit == pytest.approx(response_with_se.fit)
    assert dotted_response_with_se.se_fit == pytest.approx(response_with_se.se_fit)
    expected_terms = [
        [
            (row[col_idx] - means[col_idx]) * fit.location_coefficients[col_idx]
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in design_rows
    ]
    for actual, expected in zip(
        terms,
        expected_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_terms, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(terms_with_se.fit, expected_terms, strict=True):
        assert actual == pytest.approx(expected)
    expected_terms_se = [
        [
            abs(row[col_idx] - means[col_idx])
            * math.sqrt(max(location_vcov[col_idx][col_idx], 0.0))
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in design_rows
    ]
    for actual, expected in zip(terms_with_se.se_fit, expected_terms_se, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        x2_with_se.fit,
        [[row[1]] for row in expected_terms],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        x2_with_se.se_fit,
        [[row[1]] for row in expected_terms_se],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_training_terms = [
        [
            (row[col_idx] - means[col_idx]) * fit.location_coefficients[col_idx]
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in training_design_rows
    ]
    expected_training_terms_se = [
        [
            abs(row[col_idx] - means[col_idx])
            * math.sqrt(max(location_vcov[col_idx][col_idx], 0.0))
            for col_idx in range(1, fit.n_covariates)
        ]
        for row in training_design_rows
    ]
    for actual, expected in zip(
        training_terms,
        expected_training_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_terms_with_se.fit,
        expected_training_terms,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_terms_with_se.se_fit,
        expected_training_terms_se,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_x1_with_se.fit,
        [[row[0]] for row in expected_training_terms],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        training_x1_with_se.se_fit,
        [[row[0]] for row in expected_training_terms_se],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    expected_median = [row[0] for row in fit.predict_quantile(design_rows, [0.5]).predictions]
    median_score = math.log(-math.log1p(-0.5))
    expected_uquantile = [lp_value + median_score * fit.scale for lp_value in lp]
    expected_quantile_se = []
    expected_uquantile_se = []
    for row, prediction in zip(design_rows, expected_median, strict=True):
        design = [*row, median_score * fit.scale]
        variance = sum(
            design[left] * full_vcov[left][right] * design[right]
            for left in range(len(design))
            for right in range(len(design))
        )
        linear_se = math.sqrt(max(variance, 0.0))
        expected_uquantile_se.append(linear_se)
        expected_quantile_se.append(linear_se * prediction)
    assert median == pytest.approx(expected_median)
    assert prefix_median == pytest.approx(expected_median)
    assert median_with_se.fit == pytest.approx(expected_median)
    assert median_with_se.se_fit == pytest.approx(expected_quantile_se)
    assert uquantile_with_se.fit == pytest.approx(expected_uquantile)
    assert uquantile_with_se.se_fit == pytest.approx(expected_uquantile_se)
    assert prefix_uquantile_with_se.fit == pytest.approx(expected_uquantile)
    assert prefix_uquantile_with_se.se_fit == pytest.approx(expected_uquantile_se)
    assert len(default_bands) == 2
    assert all(len(row) == 2 for row in default_bands)
    assert len(bands) == 2
    assert all(len(row) == 2 for row in bands)
    with pytest.raises(ValueError, match="collapse"):
        survival.predict(fit, rows, collapse=["A", "B"])


def test_predict_survreg_formula_accepts_newdata_mapping():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]
    design_rows = _with_intercept(rows)
    newdata = {"x1": [0.5, 1.1], "x2": [0.8, 0.3]}

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp").predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(fit.predict(design_rows).predictions)
    assert survival.predict(fit, newdata, type="quantile", p=0.5) == pytest.approx(
        [row[0] for row in fit.predict_quantile(design_rows, [0.5]).predictions]
    )


def test_predict_survreg_gaussian_response_uses_identity_transform():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5, 0.8], [1.1, 0.3]]

    lp_with_se = survival.predict(fit, rows, type="lp", se_fit=True)
    response_with_se = survival.predict(fit, rows, se_fit=True)
    quantile_with_se = survival.predict(fit, rows, type="quantile", p=0.9, se_fit=True)
    uquantile_with_se = survival.predict(fit, rows, type="uquantile", p=0.9, se_fit=True)

    assert response_with_se.fit == pytest.approx(lp_with_se.fit)
    assert response_with_se.se_fit == pytest.approx(lp_with_se.se_fit)
    assert quantile_with_se.fit == pytest.approx(uquantile_with_se.fit)
    assert quantile_with_se.se_fit == pytest.approx(uquantile_with_se.se_fit)


def test_survreg_gaussian_residuals_use_identity_response_scale():
    normal = NormalDist()
    low_level_response = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="response",
    )
    low_level_deviance = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="deviance",
    )
    low_level_working = survival.residuals_survreg(
        [1.0, 2.0],
        [1, 0],
        [0.5, 1.5],
        1.0,
        "gaussian",
        residual_type="working",
    )
    z = 0.5
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    survivor = 1.0 - normal.cdf(z)

    assert low_level_response.residuals == pytest.approx([0.5, 0.5])
    assert low_level_deviance.residuals == pytest.approx(
        [
            z,
            math.sqrt(-2.0 * math.log(survivor)),
        ]
    )
    assert low_level_working.residuals == pytest.approx([z, density / survivor])

    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="gaussian",
        max_iter=10,
        eps=1e-5,
    )
    residuals = fit.residuals("response")
    linear_predictors = survival.predict(fit, type="lp")

    assert residuals.residual_type == "response"
    assert residuals.residuals == pytest.approx(
        [
            time - linear_predictor
            for time, linear_predictor in zip(data["time"], linear_predictors, strict=True)
        ]
    )


def test_predict_survreg_formula_newdata_mapping_rebuilds_transforms_and_interactions():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ sqrt(x1) + group:x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.25, 1.0], "x2": [0.25, 0.8], "group": ["B", "A"]}
    rows = [
        [math.sqrt(0.25), 1.0 * 0.25],
        [math.sqrt(1.0), 0.0 * 0.8],
    ]
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp").predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(fit.predict(design_rows).predictions)


def test_formula_identity_arithmetic_terms_rebuild_for_newdata():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ I(x1 + x2) + I(x1 * x2) + I(x1^2)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    expected_rows = [
        [
            data["x1"][idx] + data["x2"][idx],
            data["x1"][idx] * data["x2"][idx],
            data["x1"][idx] ** 2,
        ]
        for idx in range(len(data["time"]))
    ]
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(expected_rows),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    newdata = {"x1": [0.25, 1.0], "x2": [0.5, 0.8]}
    new_rows = [[0.75, 0.125, 0.0625], [1.8, 0.8, 1.0]]

    for actual, expected in zip(fit.covariates, low_level.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(_with_intercept(new_rows), "lp").predictions
    )


def test_predict_survreg_uses_training_rows_and_offsets():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    expected_lp = [
        fit.location_coefficients[0]
        + data["x1"][idx] * fit.location_coefficients[1]
        + data["offset"][idx]
        for idx in range(len(data["time"]))
    ]
    expected_se = [
        math.sqrt(
            max(
                fit.variance_matrix[0][0]
                + 2.0 * data["x1"][idx] * fit.variance_matrix[0][1]
                + data["x1"][idx] ** 2 * fit.variance_matrix[1][1],
                0.0,
            )
        )
        for idx in range(len(data["time"]))
    ]
    lp_with_se = survival.predict(fit, type="lp", se_fit=True)
    response_with_se = survival.predict(fit, se_fit=True)

    assert survival.predict(fit, type="lp") == pytest.approx(expected_lp)
    assert survival.predict(fit) == pytest.approx([math.exp(value) for value in expected_lp])
    assert lp_with_se.fit == pytest.approx(expected_lp)
    assert lp_with_se.se_fit == pytest.approx(expected_se)
    assert response_with_se.fit == pytest.approx([math.exp(value) for value in expected_lp])
    assert response_with_se.se_fit == pytest.approx(
        [
            se * math.exp(linear_predictor)
            for se, linear_predictor in zip(expected_se, expected_lp, strict=True)
        ]
    )
    assert fit.predict([[1.0, 0.5]], "lp", [0.2]).predictions == pytest.approx(
        [fit.location_coefficients[0] + 0.5 * fit.location_coefficients[1] + 0.2]
    )


def test_predict_survreg_formula_newdata_mapping_uses_offsets():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "offset": offsets}
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(
        fit.predict(design_rows, "response", offsets).predictions
    )


def test_predict_survreg_formula_rebuilds_transformed_offsets_from_newdata():
    data = _toy_data()
    data["exposure"] = [math.exp(value) for value in data["offset"]]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.2, -0.1]
    newdata = {"x1": [0.5, 1.0], "exposure": [math.exp(value) for value in offsets]}
    design_rows = _with_intercept(rows)

    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(fit, newdata) == pytest.approx(
        fit.predict(design_rows, "response", offsets).predictions
    )


def test_predict_survreg_formula_rebuilds_identity_arithmetic_offsets_from_newdata():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(I(offset + x2))",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    bare_fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset + x2)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        offsets=[offset + x2 for offset, x2 in zip(data["offset"], data["x2"], strict=True)],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    rows = [[0.5], [1.0]]
    offsets = [0.5, 0.3]
    newdata = {"x1": [0.5, 1.0], "offset": [0.2, -0.1], "x2": [0.3, 0.4]}
    design_rows = _with_intercept(rows)

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert bare_fit.coefficients == pytest.approx(low_level.coefficients)
    assert survival.predict(fit, newdata, type="lp") == pytest.approx(
        fit.predict(design_rows, "lp", offsets).predictions
    )
    assert survival.predict(bare_fit, newdata, type="lp") == pytest.approx(
        bare_fit.predict(design_rows, "lp", offsets).predictions
    )


def test_survreg_fit_residuals_match_low_level_apis():
    data = _toy_data()
    weights = [1.0, 1.5, 0.75, 2.0, 1.25, 0.5, 1.75, 1.0]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=weights,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    location_vcov = [row[: fit.n_covariates] for row in fit.variance_matrix[: fit.n_covariates]]

    response = fit.residuals("response")
    deviance = fit.residuals()
    working = survival.r_api.residuals(fit, type="working")
    dfbeta = survival.r_api.residuals(fit, type="dfbeta")
    dfbetas = survival.r_api.residuals(fit, type="dfbetas")
    dfbeta_without_scale = survival.r_api.residuals(fit, type="dfbeta", rsigma=False)
    matrix_residuals = survival.r_api.residuals(fit, type="matrix")
    ldcase = survival.r_api.residuals(fit, type="ldcase")
    ldresp = survival.r_api.residuals(fit, type="ldresp")
    ldshape = survival.r_api.residuals(fit, type="ldshape")
    ldcase_without_scale = survival.r_api.residuals(fit, type="ldcase", rsigma=False)
    prefix_response = survival.r_api.residuals(fit, type="r")
    prefix_working = survival.r_api.residuals(fit, type="w")
    prefix_dfbeta = survival.r_api.residuals(fit, type="dfb")
    prefix_matrix = survival.r_api.residuals(fit, type="mat")
    prefix_ldcase = survival.r_api.residuals(fit, type="ldc")
    low_response = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="response",
    )
    low_working = survival.residuals_survreg(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        residual_type="working",
    )
    low_location_dfbeta = survival.dfbeta_survreg(
        fit.time,
        fit.status,
        fit.covariates,
        fit.linear_predictors,
        fit.scale,
        location_vcov,
        fit.distribution,
    )
    low_matrix = survival.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=fit.time2,
    )
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    low_dfbeta = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        False,
    )
    low_dfbetas = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        True,
        True,
    )
    low_dfbeta_without_scale = survival.survreg_dfbeta_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        False,
        False,
    )
    low_ldcase = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldcase",
        True,
    )
    low_ldresp = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldresp",
        True,
    )
    low_ldshape = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        full_vcov,
        "ldshape",
        True,
    )
    low_ldcase_without_scale = survival.survreg_influence_residuals(
        low_matrix,
        fit.covariates,
        fit.scales,
        fit.strata,
        location_vcov,
        "ldcase",
        False,
    )

    assert fit.time == pytest.approx(data["time"])
    assert fit.status == data["status"]
    assert fit.weights == pytest.approx(weights)
    expected_covariates = _with_intercept(
        [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    )
    for actual, expected in zip(fit.covariates, expected_covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert response.residual_type == "response"
    assert response.residuals == pytest.approx(low_response.residuals)
    assert prefix_response == pytest.approx(low_response.residuals)
    assert response.residuals == pytest.approx(
        [
            time - math.exp(linear_predictor)
            for time, linear_predictor in zip(fit.time, fit.linear_predictors, strict=True)
        ]
    )
    assert deviance.residual_type == "deviance"
    assert len(deviance.residuals) == len(data["time"])
    assert working == pytest.approx(low_working.residuals)
    assert prefix_working == pytest.approx(low_working.residuals)
    assert len(dfbeta) == len(data["time"])
    for actual, expected in zip(fit.dfbeta(), low_location_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for dfbeta_matrix in (dfbeta, prefix_dfbeta):
        for actual, expected in zip(dfbeta_matrix, low_dfbeta, strict=True):
            assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("dfbeta")
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("dfbetas")
    with pytest.raises(ValueError, match="matrix-valued"):
        fit.residuals("matrix")
    assert len(dfbetas) == len(data["time"])
    for actual, expected in zip(dfbeta_without_scale, low_dfbeta_without_scale, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(dfbetas, low_dfbetas, strict=True):
        assert actual == pytest.approx(expected)
    for dfbeta_row, dfbetas_row in zip(dfbeta, dfbetas, strict=True):
        for col_idx, value in enumerate(dfbeta_row):
            scale = max(math.sqrt(abs(full_vcov[col_idx][col_idx])), 1e-12)
            assert dfbetas_row[col_idx] == pytest.approx(value / scale)
    assert len(matrix_residuals) == len(data["time"])
    assert all(len(row) == 6 for row in matrix_residuals)
    for actual, expected in zip(matrix_residuals, low_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(prefix_matrix, low_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert ldcase == pytest.approx(low_ldcase)
    assert prefix_ldcase == pytest.approx(low_ldcase)
    assert ldresp == pytest.approx(low_ldresp)
    assert ldshape == pytest.approx(low_ldshape)
    assert ldcase_without_scale == pytest.approx(low_ldcase_without_scale)

    collapse = ["A", "A", "B", "B", "C", "C", "D", "D"]
    collapsed_response = survival.r_api.residuals(fit, type="response", collapse=collapse)
    collapsed_matrix = survival.r_api.residuals(fit, type="matrix", collapse=collapse)
    collapsed_ldcase = survival.r_api.residuals(fit, type="ldcase", collapse=collapse)
    expected_response = [
        sum(
            residual
            for residual, label in zip(low_response.residuals, collapse, strict=True)
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    collapsed_dfbeta = survival.r_api.residuals(fit, type="dfbeta", collapse=collapse)
    collapsed_dfbetas = survival.r_api.residuals(fit, type="dfbetas", collapse=collapse)
    collapsed_weighted_response = survival.r_api.residuals(
        fit,
        type="response",
        collapse=collapse,
        weighted=True,
    )
    collapsed_weighted_matrix = survival.r_api.residuals(
        fit,
        type="matrix",
        collapse=collapse,
        weighted=True,
    )
    collapsed_weighted_ldcase = survival.r_api.residuals(
        fit,
        type="ldcase",
        collapse=collapse,
        weighted=True,
    )
    expected_dfbeta = [
        [
            sum(
                row[col_idx]
                for row, label in zip(low_dfbeta, collapse, strict=True)
                if label == group
            )
            for col_idx in range(len(low_dfbeta[0]))
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_response = [
        sum(
            residual * weights[idx]
            for idx, (residual, label) in enumerate(
                zip(low_response.residuals, collapse, strict=True)
            )
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_matrix = [
        [
            sum(
                row[col_idx]
                for row, label in zip(low_matrix, collapse, strict=True)
                if label == group
            )
            for col_idx in range(6)
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_matrix = [
        [
            sum(
                row[col_idx] * weights[idx]
                for idx, (row, label) in enumerate(zip(low_matrix, collapse, strict=True))
                if label == group
            )
            for col_idx in range(6)
        ]
        for group in ("A", "B", "C", "D")
    ]
    expected_ldcase = [
        sum(
            residual for residual, label in zip(low_ldcase, collapse, strict=True) if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_weighted_ldcase = [
        sum(
            residual * weights[idx]
            for idx, (residual, label) in enumerate(zip(low_ldcase, collapse, strict=True))
            if label == group
        )
        for group in ("A", "B", "C", "D")
    ]
    expected_dfbetas = [
        [
            sum(
                row[col_idx] for row, label in zip(dfbetas, collapse, strict=True) if label == group
            )
            for col_idx in range(len(dfbetas[0]))
        ]
        for group in ("A", "B", "C", "D")
    ]

    assert survival.r_api.residuals(fit, type="working", weighted=False) == pytest.approx(
        low_working.residuals
    )
    assert survival.r_api.residuals(fit, type="working", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(low_working.residuals)]
    )
    weighted_matrix = survival.r_api.residuals(fit, type="matrix", weighted=True)
    for row_idx, actual in enumerate(weighted_matrix):
        assert actual == pytest.approx([value * weights[row_idx] for value in low_matrix[row_idx]])
    assert survival.r_api.residuals(fit, type="ldcase", weighted=True) == pytest.approx(
        [value * weights[idx] for idx, value in enumerate(low_ldcase)]
    )
    weighted_dfbeta = survival.r_api.residuals(fit, type="dfbeta", weighted=True)
    weighted_dfbetas = survival.r_api.residuals(fit, type="dfbetas", weighted=True)
    for row_idx, (actual_dfbeta, actual_dfbetas) in enumerate(
        zip(weighted_dfbeta, weighted_dfbetas, strict=True)
    ):
        assert actual_dfbeta == pytest.approx(
            [value * weights[row_idx] for value in dfbeta[row_idx]]
        )
        assert actual_dfbetas == pytest.approx(
            [value * weights[row_idx] for value in dfbetas[row_idx]]
        )
    assert collapsed_response == pytest.approx(expected_response)
    assert collapsed_weighted_response == pytest.approx(expected_weighted_response)
    for actual, expected in zip(collapsed_matrix, expected_matrix, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(collapsed_weighted_matrix, expected_weighted_matrix, strict=True):
        assert actual == pytest.approx(expected)
    assert collapsed_ldcase == pytest.approx(expected_ldcase)
    assert collapsed_weighted_ldcase == pytest.approx(expected_weighted_ldcase)
    for actual, expected in zip(collapsed_dfbeta, expected_dfbeta, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(collapsed_dfbetas, expected_dfbetas, strict=True):
        assert actual == pytest.approx(expected)


def test_survreg_formula_accepts_intercept_only_rhs():
    data = _toy_data()
    fit = survival.survreg("Surv(time, status) ~ 1", data=data, max_iter=10, eps=1e-5)
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=[[1.0] for _ in data["time"]],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.n_covariates == 1
    assert fit.covariates == [[1.0] for _ in data["time"]]


def test_survreg_formula_accepts_numeric_interactions():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [data["x1"][idx], data["x2"][idx], data["x1"][idx] * data["x2"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_formula_slash_expands_nested_terms_like_r():
    data = _numeric_data()
    nested = survival.survreg(
        "Surv(time, status) ~ x1 / x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x1:x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [data["x1"][idx], data["x1"][idx] * data["x2"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(nested.covariates, low_level.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert nested.coefficients == pytest.approx(explicit.coefficients)
    assert nested.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_formula_in_operator_expands_nested_terms_like_r():
    data = _numeric_data()
    nested = survival.survreg(
        "Surv(time, status) ~ x1 + x2 %in% x1",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x1:x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    slash = survival.survreg(
        "Surv(time, status) ~ x1 / x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(nested.covariates, slash.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert nested.coefficients == pytest.approx(explicit.coefficients)
    assert nested.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_formula_power_expands_crossing_degree_like_r():
    data = _numeric_data()
    power = survival.survreg(
        "Surv(time, status) ~ (x1 + x2)^2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    crossed = survival.survreg(
        "Surv(time, status) ~ x1 * x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    for actual, expected in zip(power.covariates, crossed.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert power.coefficients == pytest.approx(crossed.coefficients)
    assert power.log_likelihood == pytest.approx(crossed.log_likelihood)

    grouped_data = {
        **data,
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    grouped = survival.survfit(
        "Surv(time, status) ~ (x1 + x2 + x3)^2",
        data=grouped_data,
    )
    explicit = survival.survfit(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3",
        data=grouped_data,
    )

    assert list(grouped) == list(explicit)
    for key in explicit:
        assert grouped[key].estimate == pytest.approx(explicit[key].estimate)


def test_formula_parenthesized_crossing_expands_like_r():
    data = {
        **_numeric_data(),
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    grouped = survival.coxph(
        "Surv(time, status) ~ (x1 + x2) * x3",
        data=data,
        max_iter=0,
    )
    explicit = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x3 + x2:x3",
        data=data,
        max_iter=0,
    )
    expected_rows = [
        [
            data["x1"][idx],
            data["x2"][idx],
            data["x3"][idx],
            data["x1"][idx] * data["x3"][idx],
            data["x2"][idx] * data["x3"][idx],
        ]
        for idx in range(len(data["time"]))
    ]

    for actual, expected in zip(grouped.covariates, expected_rows, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(grouped.covariates, explicit.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert grouped.log_likelihood == pytest.approx(explicit.log_likelihood)

    grouped_curves = survival.survfit("Surv(time, status) ~ (x1 + x2) * x3", data=data)
    explicit_curves = survival.survfit(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x3 + x2:x3",
        data=data,
    )
    assert list(grouped_curves) == list(explicit_curves)
    for key in explicit_curves:
        assert grouped_curves[key].estimate == pytest.approx(explicit_curves[key].estimate)


def test_formula_dot_expands_inside_compound_expressions_like_r():
    data = {
        **_numeric_data(),
        "x3": [0.3, 0.6, 0.2, 0.7, 1.1, 0.5, 0.9, 0.4],
    }
    crossed = survival.coxph(
        "Surv(time, status) ~ x1 * .",
        data=data,
        max_iter=0,
    )
    explicit_crossed = survival.coxph(
        "Surv(time, status) ~ x1 + x2 + x3 + x1:x2 + x1:x3",
        data=data,
        max_iter=0,
    )
    expected_crossed_rows = [
        [
            data["x1"][idx],
            data["x2"][idx],
            data["x3"][idx],
            data["x1"][idx] * data["x2"][idx],
            data["x1"][idx] * data["x3"][idx],
        ]
        for idx in range(len(data["time"]))
    ]

    for actual, expected in zip(crossed.covariates, expected_crossed_rows, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(crossed.covariates, explicit_crossed.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert crossed.log_likelihood == pytest.approx(explicit_crossed.log_likelihood)

    interaction = survival.coxph(
        "Surv(time, status) ~ x1:.",
        data=data,
        max_iter=0,
    )
    explicit_interaction = survival.coxph(
        "Surv(time, status) ~ x1 + x1:x2 + x1:x3",
        data=data,
        max_iter=0,
    )

    for actual, expected in zip(
        interaction.covariates, explicit_interaction.covariates, strict=True
    ):
        assert actual == pytest.approx(expected)
    assert interaction.log_likelihood == pytest.approx(explicit_interaction.log_likelihood)

    powered = survival.survreg(
        "Surv(time, status) ~ (. - x2)^2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit_powered = survival.survreg(
        "Surv(time, status) ~ x1 + x3 + x1:x3",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for actual, expected in zip(powered.covariates, explicit_powered.covariates, strict=True):
        assert actual == pytest.approx(expected)
    assert powered.coefficients == pytest.approx(explicit_powered.coefficients)
    assert powered.log_likelihood == pytest.approx(explicit_powered.log_likelihood)


def test_survreg_formula_dot_expands_remaining_covariates():
    data = _numeric_data()
    fit = survival.survreg(
        "Surv(time, status) ~ .",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(explicit.coefficients)
    assert fit.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_formula_dot_can_exclude_identifier_columns():
    data = _numeric_data_with_id()
    fit = survival.survreg(
        "Surv(time, status) ~ . - id",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(explicit.coefficients)
    assert fit.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_formula_accepts_backtick_column_names():
    data = _backtick_data()
    fit = survival.survreg(
        "Surv(`follow-up`, `event status`) ~ `age-years` + `marker/value`",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["follow-up"],
        status=[float(value) for value in data["event status"]],
        covariates=_with_intercept(
            [
                [data["age-years"][idx], data["marker/value"][idx]]
                for idx in range(len(data["follow-up"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_accepts_backtick_numeric_transforms():
    data = _backtick_data()
    fit = survival.survreg(
        "Surv(`follow-up`, `event status`) ~ sqrt(`marker/value`) + `age-years`",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["follow-up"],
        status=[float(value) for value in data["event status"]],
        covariates=_with_intercept(
            [
                [math.sqrt(data["marker/value"][idx]), data["age-years"][idx]]
                for idx in range(len(data["follow-up"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_accepts_identity_wrappers_for_numeric_terms():
    data = _toy_data()
    direct = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    for wrapper in ("I", "identity", "as.numeric"):
        fit = survival.survreg(
            f"Surv(time, status) ~ {wrapper}(x1) + x2",
            data=data,
            dist="weibull",
            max_iter=10,
            eps=1e-5,
        )

        assert fit.coefficients == pytest.approx(direct.coefficients)
        assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_formula_filters_external_weights_with_subset_and_na_action():
    data = _numeric_data()
    indices = [0, 2, 3, 4, 5]
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        subset=[0, 1, 2, 3, 4, 5],
        na_action="omit",
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    dotted = survival.survreg(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        weights=[1.0, None, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
        subset=[0, 1, 2, 3, 4, 5],
        dist="weibull",
        max_iter=10,
        eps=1e-5,
        **{"na.action": "omit"},
    )
    low_level = survival.regression.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[float(data["status"][idx]) for idx in indices],
        covariates=_with_intercept([[data["x1"][idx], data["x2"][idx]] for idx in indices]),
        weights=[1.0, 1.0, 0.8, 1.2, 1.1],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert dotted.coefficients == pytest.approx(low_level.coefficients)
    assert dotted.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_formula_as_factor_treatment_codes_numeric_covariates():
    data = _factor_data()
    fit = survival.survreg(
        "Surv(time, status) ~ as.factor(dose) + x2",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [
                    1.0 if data["dose"][idx] == 1 else 0.0,
                    1.0 if data["dose"][idx] == 2 else 0.0,
                    data["x2"][idx],
                ]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_survreg_matrix_input_applies_subset_to_row_aligned_arrays():
    data = _numeric_data()
    indices = [0, 1, 3, 5, 6]
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        subset=indices,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_matrix_input_defaults_to_weibull_distribution():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    default = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        max_iter=10,
        eps=1e-5,
    )
    explicit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert default.distribution == "weibull"
    assert default.coefficients == pytest.approx(explicit.coefficients)
    assert default.log_likelihood == pytest.approx(explicit.log_likelihood)


def test_survreg_matrix_na_action_omit_filters_covariate_rows():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    rows[2][0] = float("nan")
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.survreg(
        time=data["time"],
        status=data["status"],
        covariates=rows,
        na_action="omit",
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_matrix_input_rejects_fractional_status_codes():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    status = list(data["status"])
    status[2] = 1.5

    with pytest.raises(ValueError, match="0/1/2/3 censoring codes"):
        survival.survreg(
            time=data["time"],
            status=status,
            covariates=rows,
            distribution="weibull",
            max_iter=10,
            eps=1e-5,
        )


def test_survreg_matrix_na_action_omit_filters_status_before_code_validation():
    data = _numeric_data()
    rows = [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
    status = list(data["status"])
    status[2] = float("nan")
    indices = [0, 1, 3, 4, 5, 6, 7]
    fit = survival.survreg(
        time=data["time"],
        status=status,
        covariates=rows,
        na_action="omit",
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )
    direct = survival.survreg(
        time=[data["time"][idx] for idx in indices],
        status=[data["status"][idx] for idx in indices],
        covariates=[rows[idx] for idx in indices],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(direct.coefficients)
    assert fit.log_likelihood == pytest.approx(direct.log_likelihood)


def test_survreg_formula_treatment_codes_categorical_covariates():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ group + x1",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [
                [1.0 if data["group"][idx] == "B" else 0.0, data["x1"][idx]]
                for idx in range(len(data["time"]))
            ]
        ),
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    newdata = {"group": ["B", "A"], "x1": [0.5, 0.8]}
    term_se = survival.predict(fit, newdata, type="terms", terms="factor(group)", se_fit=True)
    group_var = fit.variance_matrix[1][1]
    group_mean = sum(1.0 if value == "B" else 0.0 for value in data["group"]) / len(data["group"])
    for actual, expected in zip(
        term_se.fit,
        [
            [(1.0 - group_mean) * fit.location_coefficients[1]],
            [(0.0 - group_mean) * fit.location_coefficients[1]],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        term_se.se_fit,
        [
            [abs(1.0 - group_mean) * math.sqrt(max(group_var, 0.0))],
            [abs(0.0 - group_mean) * math.sqrt(max(group_var, 0.0))],
        ],
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_survreg_formula_passes_strata_to_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + x2 + strata(group)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept(
            [[data["x1"][idx], data["x2"][idx]] for idx in range(len(data["time"]))]
        ),
        strata=[0, 0, 0, 0, 1, 1, 1, 1],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)
    assert fit.strata == [0, 0, 0, 0, 1, 1, 1, 1]

    score = math.log(-math.log1p(-0.5))
    quantile_with_se = survival.predict(fit, type="uquantile", p=0.5, se_fit=True)
    newdata = {"x1": [0.5, 0.8], "x2": [0.4, 0.6], "group": ["B", "A"]}
    newdata_quantile = survival.predict(fit, newdata, type="uquantile", p=0.5)
    full_width = fit.n_covariates + len(fit.scales)
    full_vcov = [row[:full_width] for row in fit.variance_matrix[:full_width]]
    expected_fit = []
    expected_se = []
    for idx, (x1, x2) in enumerate(zip(data["x1"], data["x2"], strict=True)):
        stratum = fit.strata[idx]
        expected_fit.append(fit.linear_predictors[idx] + score * fit.scales[stratum])
        design = [1.0, x1, x2, *([0.0] * len(fit.scales))]
        design[fit.n_covariates + stratum] = score * fit.scales[stratum]
        variance = sum(
            design[left] * full_vcov[left][right] * design[right]
            for left in range(full_width)
            for right in range(full_width)
        )
        expected_se.append(math.sqrt(max(variance, 0.0)))

    assert quantile_with_se.fit == pytest.approx(expected_fit)
    assert quantile_with_se.se_fit == pytest.approx(expected_se)
    assert newdata_quantile == pytest.approx(
        [
            (
                fit.location_coefficients[0]
                + 0.5 * fit.location_coefficients[1]
                + 0.4 * fit.location_coefficients[2]
                + score * fit.scales[1]
            ),
            (
                fit.location_coefficients[0]
                + 0.8 * fit.location_coefficients[1]
                + 0.6 * fit.location_coefficients[2]
                + score * fit.scales[0]
            ),
        ]
    )


def test_survreg_formula_offset_matches_low_level_binding():
    data = _toy_data()
    fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(offset)",
        data=data,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )
    low_level = survival.regression.survreg(
        time=data["time"],
        status=[float(value) for value in data["status"]],
        covariates=_with_intercept([[value] for value in data["x1"]]),
        offsets=data["offset"],
        distribution="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert fit.coefficients == pytest.approx(low_level.coefficients)
    assert fit.log_likelihood == pytest.approx(low_level.log_likelihood)

    transformed = {**data, "exposure": [math.exp(value) for value in data["offset"]]}
    transformed_fit = survival.survreg(
        "Surv(time, status) ~ x1 + offset(log(exposure))",
        data=transformed,
        dist="weibull",
        max_iter=10,
        eps=1e-5,
    )

    assert transformed_fit.coefficients == pytest.approx(low_level.coefficients)
    assert transformed_fit.log_likelihood == pytest.approx(low_level.log_likelihood)


def test_r_api_rejects_unsupported_formula_features():
    with pytest.raises(ValueError, match="unsupported formula"):
        survival.coxph("Surv(time, status) ~ x1(x2)", data=_toy_data())

    with pytest.raises(ValueError, match="unterminated backtick"):
        survival.survfit("Surv(time, status) ~ `group", data=_toy_data())

    with pytest.raises(ValueError, match=r"factor\(\) requires exactly one column"):
        survival.coxph("Surv(time, status) ~ factor(x1, x2)", data=_toy_data())

    with pytest.raises(ValueError, match=r"log\(\) requires exactly one column"):
        survival.coxph("Surv(time, status) ~ log(x1, x2)", data=_toy_data())

    data_with_zero = _numeric_data()
    data_with_zero["x2"][0] = 0.0
    with pytest.raises(ValueError, match="requires positive values"):
        survival.coxph("Surv(time, status) ~ log(x2)", data=data_with_zero)

    for function in (
        survival.survfit,
        survival.survdiff,
    ):
        with pytest.raises(ValueError, match=r"cluster\(\)"):
            function("Surv(time, status) ~ x1 + cluster(group)", data=_toy_data())

    with pytest.raises(ValueError, match="method or ties"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            method="efron",
            ties="breslow",
        )

    with pytest.raises(ValueError, match="ambiguous"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), ties="e")

    with pytest.raises(TypeError, match="y"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), y=1)

    with pytest.raises(TypeError, match="x"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), x=1)

    with pytest.raises(ValueError, match=r"singular_ok or singular\.ok"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            singular_ok=False,
            **{"singular.ok": True},
        )

    with pytest.raises(NotImplementedError, match="tt"):
        survival.coxph(
            "Surv(time, status) ~ x1 + tt(x2)",
            data=_toy_data(),
            tt=lambda value, *_args: value,
        )

    with pytest.raises(ValueError, match="id must have length"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), id=["a"])

    with pytest.raises(ValueError, match="cluster or id"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            id=_toy_data()["group"],
            robust=False,
        )

    with pytest.raises(NotImplementedError, match="istate"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), istate=_toy_data()["status"])

    with pytest.raises(NotImplementedError, match="statedata"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), statedata={"states": [1]})

    with pytest.raises(ValueError, match="nocenter"):
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), nocenter=[0.0, float("nan")])

    with pytest.raises(ValueError, match="fixed scale and strata"):
        survival.survreg("Surv(time, status) ~ x1 + strata(group)", data=_toy_data(), scale=1.0)

    with pytest.raises(NotImplementedError, match="parms"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), parms=[1.0])

    concordance_data = _toy_data()
    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            weights=[1.0, -1.0, *([1.0] * (len(concordance_data["time"]) - 2))],
        )

    with pytest.raises(ValueError, match="cluster must have"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            cluster=["a"],
        )

    with pytest.raises(ValueError, match="formula cluster"):
        survival.concordance(
            "Surv(time, status) ~ x1 + cluster(group)",
            data=concordance_data,
            cluster=concordance_data["group"],
        )

    with pytest.raises(TypeError, match="ymin"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            ymin=object(),
        )

    counting_concordance_data = _counting_cox_data()
    counting_concordance_data["score"] = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4]
    with pytest.raises(ValueError, match="counting-process"):
        survival.concordance(
            "Surv(start, stop, status) ~ score",
            data=counting_concordance_data,
            timewt="S/G",
        )

    with pytest.raises(ValueError, match="influence"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            influence=4,
        )

    with pytest.raises(TypeError, match="ranks"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            ranks=1,
        )

    with pytest.raises(TypeError, match="keepstrata"):
        survival.concordance(
            "Surv(time, status) ~ x1",
            data=concordance_data,
            keepstrata=object(),
        )

    with pytest.raises(TypeError, match="y"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), y=1)

    with pytest.raises(TypeError, match="x"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), x=1)

    with pytest.raises(ValueError, match="cluster must have"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), cluster=["a"])

    with pytest.raises(ValueError, match="scale must be non-negative"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), scale=-1.0)

    with pytest.raises(ValueError, match="scale must be finite"):
        survival.survreg("Surv(time, status) ~ x1", data=_toy_data(), scale=float("nan"))

    with pytest.raises(ValueError, match="time="):
        survival.basehaz(survival.coxph("Surv(time, status) ~ x1", data=_toy_data()), time=[1.0])

    with pytest.raises(ValueError, match="weights"):
        survival.basehaz(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data()),
            weights=[1.0] * len(_toy_data()["time"]),
        )

    with pytest.raises(ValueError, match="status and linear_predictors"):
        survival.basehaz([1.0, 2.0], status=[1, 0])

    with pytest.raises(ValueError, match="newdata"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            newdata=[[0.0]],
        )

    with pytest.raises(ValueError, match="status must use 0/1 or 1/2"):
        survival.basehaz([1.0, 2.0], status=[0, 2], linear_predictors=[0.0, 0.0], centered=False)

    with pytest.raises(ValueError, match="linear_predictors contains non-finite"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, float("nan")],
            centered=False,
        )

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            weights=[1.0, -1.0],
        )

    with pytest.raises(ValueError, match="at least one positive"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            weights=[0.0, 0.0],
        )

    with pytest.raises(ValueError, match="entry_times contains non-finite"):
        survival.basehaz(
            [1.0, 2.0],
            status=[1, 0],
            linear_predictors=[0.0, 0.0],
            centered=False,
            entry_times=[0.0, float("nan")],
        )

    fit = survival.coxph("Surv(time, status) ~ x1 + x2", data=_toy_data(), max_iter=1)
    with pytest.raises(ValueError, match="predict type"):
        survival.predict(fit, [[0.5, 0.8]], type="score")

    with pytest.raises(ValueError, match="expected"):
        survival.predict(fit, [[0.5, 0.8]], type="expected")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.predict(fit, [[0.5, 0.8]], reference="s")

    matrix_fit = survival.coxph(
        survival.Surv(_toy_data()["time"], _toy_data()["status"]),
        x=[[x1, x2] for x1, x2 in zip(_toy_data()["x1"], _toy_data()["x2"], strict=True)],
        max_iter=1,
    )
    with pytest.raises(TypeError, match="design matrix"):
        survival.predict(matrix_fit, {"x1": [0.5], "x2": [0.8]})

    with pytest.raises(ValueError, match="residuals type"):
        survival.r_api.residuals(fit, type="unknown")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.r_api.residuals(fit, type="d")

    with pytest.raises(ValueError, match="Cox partial residuals"):
        survival.r_api.residuals(fit, type="score", terms="x1")

    data = _numeric_data()
    data["x1"][2] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        survival.coxph("Surv(time, status) ~ x1", data=data)

    with pytest.raises(ValueError, match="conf_level"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_level=1.2)

    with pytest.raises(ValueError, match="conf_int"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_int=1.2)

    with pytest.raises(ValueError, match="conf_level or conf_int"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_level=0.9,
            conf_int=0.8,
        )

    with pytest.raises(ValueError, match=r"conf_int or conf\.int"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_int=0.9,
            **{"conf.int": 0.8},
        )

    with pytest.raises(ValueError, match=r"conf_type or conf\.type"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            conf_type="plain",
            **{"conf.type": "logit"},
        )

    dotted_timefix = survival.survfit(
        survival.Surv([1.0, 1.0 + 5e-10, 2.0], [1, 1, 0]),
        **{"time.fix": False},
    )
    assert dotted_timefix.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])

    with pytest.raises(TypeError, match="timefix"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), timefix=1)

    with pytest.raises(ValueError, match=r"se_fit or se\.fit"):
        survival.predict(fit, [[0.5, 0.8]], se_fit=True, **{"se.fit": False})

    with pytest.raises(TypeError, match="unexpected keyword"):
        survival.predict(fit, [[0.5, 0.8]], **{"na.action": "omit"})

    with pytest.raises(ValueError, match=r"max_iter or control\.iter\.max"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            max_iter=5,
            control={"iter.max": 10},
        )

    with pytest.raises(ValueError, match="outer\\.max"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"outer.max": 0},
        )

    with pytest.raises(ValueError, match="toler\\.inf"):
        survival.coxph(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"toler.inf": 0},
        )

    with pytest.raises(ValueError, match=r"eps or control\.rel\.tolerance"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            eps=1e-5,
            control={"rel.tolerance": 1e-6},
        )

    with pytest.raises(ValueError, match="debug"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"debug": math.nan},
        )

    with pytest.raises(ValueError, match="outer\\.max"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            control={"outer.max": 0},
        )

    with pytest.raises(ValueError, match="only one of init"):
        survival.survreg(
            "Surv(time, status) ~ x1 + x2",
            data=_toy_data(),
            init=[0.0, 0.0, 0.0],
            initial=[0.0, 0.0, 0.0],
        )

    aft = survival.survreg(
        "Surv(time, status) ~ x1",
        data=_toy_data(),
        max_iter=5,
        eps=1e-5,
    )
    with pytest.raises(ValueError, match="Cox partial residuals"):
        survival.r_api.residuals(aft, type="response", terms="x1")

    with pytest.raises(ValueError, match="conf_level"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            conf_level=1.2,
        )

    with pytest.raises(ValueError, match="conf_type"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_type="weird")

    with pytest.raises(ValueError, match="ambiguous"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), conf_type="l")

    with pytest.raises(ValueError, match="start_time"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), start_time=float("nan"))

    with pytest.raises(TypeError, match="time0"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), time0=1)

    with pytest.raises(TypeError, match="censor"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), censor=1)

    with pytest.raises(ValueError, match="censor"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), censor=False)

    with pytest.raises(ValueError, match="all observations"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), start_time=3.0)

    with pytest.raises(ValueError, match="survfit type"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), type="mystery")

    with pytest.raises(ValueError, match="stype"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), stype=3)

    with pytest.raises(ValueError, match="ctype"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), ctype=0)

    left_model = survival.survfit(survival.Surv([1.0, 2.0], [0, 1], type="left"), model=True)
    assert isinstance(left_model, survival.r_api.TurnbullSurvfitResult)
    assert left_model.model["response"].type == "left"

    with pytest.raises(ValueError, match="entry=TRUE"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), entry=True)

    with pytest.raises(ValueError, match="entry=TRUE"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), id=["a", "b"], entry=True)

    with pytest.raises(ValueError, match=r"se_fit or se\.fit"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [1, 0]),
            se_fit=False,
            **{"se.fit": True},
        )

    with pytest.raises(ValueError, match="id must have"):
        survival.survfit(
            survival.Surv([0.0, 0.0], [1.0, 2.0], [1, 0]),
            id=["a"],
        )

    with pytest.raises(NotImplementedError, match="cluster"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), cluster=["a", "b"])

    with pytest.raises(NotImplementedError, match="robust variance"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), robust=True)

    with pytest.raises(NotImplementedError, match="istate"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), istate=[0, 1])

    with pytest.raises(NotImplementedError, match="etype"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), etype=[1, 2])

    with pytest.raises(TypeError, match="entry"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), entry=1)

    with pytest.raises(ValueError, match=r"start\[0\] must be less than stop\[0\]"):
        survival.Surv([1.0, 1.0], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="time contains non-finite"):
        survival.Surv([1.0, float("inf")], [1, 0])

    with pytest.raises(ValueError, match="start contains non-finite"):
        survival.Surv([0.0, float("-inf")], [1.0, 2.0], [1, 0])

    with pytest.raises(ValueError, match="stop contains non-finite"):
        survival.Surv([0.0, 1.0], [1.0, float("inf")], [1, 0])

    with pytest.raises(ValueError, match="time2 contains non-finite"):
        survival.Surv([1.0], [float("inf")], [3], type="interval")

    with pytest.raises(ValueError, match="weights must be non-negative"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            weights=[1.0, -1.0],
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            reverse=True,
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            type="fh",
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            conf_type="plain",
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            start_time=1.0,
        )

    with pytest.raises(ValueError, match="right-censored"):
        survival.survfit(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            time0=True,
        )

    with pytest.raises(ValueError, match="Surv or formula"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            reverse=True,
        )

    cox_plain = survival.survfit(
        survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
        conf_type="plain",
    )
    assert cox_plain.conf_lower
    assert cox_plain.conf_upper

    with pytest.raises(ValueError, match="removed all endpoints"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            start_time=99.0,
        )

    with pytest.raises(ValueError, match="Surv or formula"):
        survival.survfit(
            survival.coxph("Surv(time, status) ~ x1", data=_toy_data(), max_iter=1),
            type="fh",
        )

    with pytest.raises(NotImplementedError, match="right-censored and counting"):
        survival.survdiff(
            survival.Surv([1.0, 2.0], [0, 1], type="left"),
            group=["A", "B"],
        )

    with pytest.raises(TypeError, match="timefix"):
        survival.survdiff(
            survival.Surv([1.0, 2.0], [1, 0]),
            group=["A", "B"],
            timefix=1,
        )

    with pytest.raises(ValueError, match=r"timefix or time\.fix"):
        survival.survdiff(
            survival.Surv([1.0, 1.0 + 5e-10, 2.0], [1, 1, 0]),
            group=["A", "A", "B"],
            timefix=False,
            **{"time.fix": True},
        )

    with pytest.raises(ValueError, match="right endpoint"):
        survival.Surv([2.0], [1.0], type="interval2")

    with pytest.raises(ValueError, match="one-dimensional"):
        survival.survfit(survival.Surv([1.0, 2.0, 3.0], [1, 0, 1]), group=["A", ["B"], "C"])

    with pytest.raises(ValueError, match="selects no rows"):
        survival.survfit(survival.Surv([1.0, 2.0], [1, 0]), subset=[False, False])

    with pytest.raises(NotImplementedError, match="right, left, interval, and interval2"):
        survival.survreg(survival.Surv([0.0, 1.0], [1.0, 2.0], [1, 0]), x=[[1.0], [2.0]])

    with pytest.raises(ValueError, match="only one of init"):
        survival.coxph(
            "Surv(time, status) ~ x1",
            data=_toy_data(),
            init=[0.0],
            initial_beta=[0.0],
        )

    with pytest.raises(ValueError, match="formula offset"):
        survival.coxph(
            "Surv(time, status) ~ x1 + offset(offset)",
            data=_toy_data(),
            offset=[0.0] * 8,
        )

    with pytest.raises(ValueError, match="formula offset"):
        survival.survreg(
            "Surv(time, status) ~ x1 + offset(offset)",
            data=_toy_data(),
            offset=[0.0] * 8,
        )
