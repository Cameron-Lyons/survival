"""Shared data builders and manual reference implementations for the test_r_*.py suite."""

import math
from itertools import combinations
from statistics import NormalDist

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


def _interaction_contrast_data():
    return {
        "time": [float(idx) for idx in range(1, 13)],
        "status": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        "g": ["A", "A", "B", "B", "C", "C"] * 2,
        "h": ["L", "H", "L", "H", "L", "H"] * 2,
        "x": [float(idx) for idx in range(1, 13)],
    }


def _interaction_contrast_rows(data, columns):
    rows = []
    for row_idx in range(len(data["time"])):
        row = []
        for column in columns:
            value = 1.0
            for factor in column.split(":"):
                if factor == "(Intercept)":
                    continue
                if factor == "x":
                    value *= data["x"][row_idx]
                elif factor.startswith("g"):
                    value *= float(data["g"][row_idx] == factor[1:])
                elif factor.startswith("h"):
                    value *= float(data["h"][row_idx] == factor[1:])
                else:
                    raise AssertionError(f"unknown test design column {factor!r}")
            row.append(value)
        rows.append(row)
    return rows


class _NamedMatrix:
    def __init__(self, columns, rows):
        self.columns = columns
        self._rows = rows

    def to_numpy(self):
        return self

    def tolist(self):
        return [list(row) for row in self._rows]


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


def _clogit_data():
    return {
        "case": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        "set": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        "x": [0.2, 0.8, 1.1, 0.4, 0.5, 1.2, 0.9, 0.1, 0.7, 1.0, 0.3, 1.3, 0.6, 0.2, 1.4, 0.9],
        "z": [1.0, 0.3, 0.8, 1.4, 0.2, 1.1, 0.6, 0.9, 0.5, 1.2, 0.1, 0.7, 1.3, 0.4, 0.8, 0.2],
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


def _formula_predictor_scores(scores):
    return [-value for value in scores]


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


def _cch_parity_data() -> dict[str, list[object]]:
    return {
        "start": [0, 2, 1, 5, 4, 0, 10, 3, 12, 1, 5, 9, 0, 6, 2, 4, 7, 2, 11, 13],
        "stop": [5, 12, 3, 18, 9, 1, 15, 7, 20, 4, 11, 16, 2, 14, 6, 10, 13, 8, 17, 19],
        "status": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "x": [
            -1.2,
            0.4,
            0.9,
            -0.3,
            1.4,
            -0.8,
            0.2,
            1.1,
            -0.5,
            0.7,
            -1.0,
            0.1,
            1.7,
            -0.6,
            0.5,
            -1.5,
            1.0,
            -0.1,
            0.8,
            -0.9,
        ],
        "z": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        "group": ["a", "b", "a", "b"] * 5,
        "id": list(range(1, 21)),
        "subcohort": [1] * 14 + [0] * 6,
    }
