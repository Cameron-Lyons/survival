"""AJ time-in-state uncertainty and infinite endpoints against stock R."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aj_mean_infinite_reference.json").read_text()
)


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, str) and value in ("Inf", "-Inf"):
        return float(value)
    return value


def _fit(case, api):
    source = _decode(case["input"])
    options = case["options"]
    if api == "native":
        return survival.surv_analysis.survfitaj(**source, **options)
    data = {
        "time": source["time"],
        "event": r._coerce._r_factor(
            [["censor", *source["states"]][code] for code in source["state"]],
            ["censor", *source["states"]],
        ),
    }
    if source["start"] is not None:
        data["start"] = source["start"]
    arguments = {**options}
    for name in ("weights", "id", "cluster"):
        if source[name] is not None:
            data[name] = source[name]
            arguments[name] = name
    if source["istate"] is not None:
        data["initial"] = r._coerce._r_factor(source["istate"], source["istate_levels"])
        arguments["istate"] = "initial"
    if source["strata"] is not None:
        data["g"] = source["strata"]
    response = "Surv(time,event)" if source["start"] is None else "Surv(start,time,event)"
    formula = f"{response} ~ " + ("1" if source["strata"] is None else "g")
    return r.survfit(formula, data, **arguments)


def _assert_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(_decode(expected), dtype=float),
        rtol=2e-12,
        atol=2e-14,
    )


@pytest.mark.parametrize("api", ["native", "formula"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_aj_endpoints_and_influence_mean_tables_match_r(case, api):
    fit = _fit(case, api)
    expected = case["fit"]
    assert fit.states == expected["states"]
    for field in (
        "time",
        "p0",
        "pstate",
        "cumhaz",
        "n_risk",
        "n_event",
        "n_censor",
        "n_enter",
        "n_transition",
        "std_err",
        "std_chaz",
        "std_auc",
    ):
        if expected[field] is None:
            assert getattr(fit, field) is None, field
        else:
            _assert_close(getattr(fit, field), expected[field])
    _assert_close([fit.t0], expected["t0"])
    if expected["influence_pstate"] is not None:
        for actual, values in zip(fit.influence_pstate, expected["influence_pstate"], strict=True):
            _assert_close(actual.values.reshape(-1), np.asarray(values, dtype=float).reshape(-1))
    for query in case["tables"]:
        option = _decode(query["rmean"])
        if isinstance(option, list):
            option = option[0]
        expected_table = query["expected"]
        if "error" in expected_table:
            if api == "formula" and fit.time:
                with pytest.raises(
                    ValueError, match="Truncation|empty|min\\(|non-NA|missing value"
                ):
                    r.summary_survfit(fit, rmean=option)
            continue
        for source in (fit, survival.surv_analysis.survfit0_aj(fit)) if api == "native" else (fit,):
            if api == "native":
                table, ends, columns = source.mean_table(rmean=str(option))
            else:
                result = r.summary_survfit(source, rmean=option)
                table, ends, columns = (
                    result.table.values,
                    result.rmean_endtime,
                    result.table.colnames,
                )
            assert columns == expected_table["columns"]
            _assert_close(table, expected_table["values"])
            _assert_close(ends or [], expected_table["ends"] or [])


def test_retained_influences_follow_curve_and_state_subsets_and_scale():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "weighted_groups_TRUE_FALSE")
    fit = _fit(case, "native")
    table = np.asarray(fit.mean_table(rmean="4")[0]).reshape(3, 2, 4)
    selected = fit.select_states([2, 0]).select_curves([1, 0, 1])
    expected = table[[2, 0]][:, [1, 0, 1]].reshape(6, 4).copy()
    expected[:, 2:] /= 2
    _assert_close(selected.mean_table(rmean="4", scale=2)[0], expected)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_aj_start_and_summary_query_times_still_require_finite_values(value):
    case = next(case for case in REFERENCE["cases"] if case["name"] == "positive_event_TRUE_FALSE")
    with pytest.raises(ValueError, match="start.time"):
        survival.surv_analysis.survfitaj(**_decode(case["input"]), start_time=value)
    fit = _fit(case, "formula")
    with pytest.raises(ValueError, match="times"):
        r.summary_survfit(fit, times=[value], rmean="none")
