"""Rate-table requested-time selection against stock R and exact curve invariants."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
population = survival.population
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures/survexp_grid_reference.json").read_text())


def decode_times(values):
    return None if values is None else [float.fromhex(value) for value in values]


@pytest.fixture(scope="module")
def table():
    spec = REFERENCE["table"]
    return population.RateTable(
        spec["dims"],
        [spec["dimid"]],
        spec["dimnames"],
        spec["cutpoints"],
        spec["types"],
        spec["rates"],
    )


@pytest.mark.parametrize("api", ["native", "formula"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_requested_rate_table_grid_matches_r(case, api, table):
    data = REFERENCE["data"]
    times = decode_times(case["times"])
    kwargs = {
        "times": times,
        "method": case["method"],
        "conditional": case["conditional"],
        "scale": case["scale"],
    }
    if api == "native":
        result = population.survexp(
            table,
            [[age] for age in data["age"]],
            y=None if case["formula"].startswith("~") else data["time"],
            group=[1 if label == "b" else 0 for label in data["g"]],
            **kwargs,
        )
    else:
        result = r.survexp(case["formula"], data, ratetable=table, **kwargs)
        assert result.strata == case["expected"]["labels"]

    expected = case["expected"]
    # Ordinary numerical equality would miss a changed sign of zero.
    assert [value.hex() for value in result.time] == [
        value.hex() for value in decode_times(expected["time"])
    ]
    assert result.method == expected["method"]
    np.testing.assert_array_equal(result.n_risk, expected["n_risk"])
    np.testing.assert_allclose(result.surv, expected["surv"], rtol=3e-14, atol=2e-15)
    if times is not None:
        for index in range(1, len(times)):
            if times[index] == times[index - 1]:
                assert result.surv[index] == result.surv[index - 1]
                assert result.n_risk[index] == result.n_risk[index - 1]
        if len(times) > 4:
            # The subject followed through exactly 2 contributes at 2, and
            # leaves before the adjacent representable request after 2.
            at_two = times.index(2.0)
            near_two = times.index(np.nextafter(2.0, np.inf))
            if case["formula"].startswith("time"):
                assert result.n_risk[at_two][1] == 2
                assert result.n_risk[near_two][1] == 1


@pytest.mark.parametrize("api", ["native", "formula"])
def test_no_response_duplicate_grid_matches_constant_rate_mixture(api):
    table = population.RateTable([2], ["sex"], [["male", "female"]], [None], [1], [0.01, 0.03])
    times = [-0.0, 0.0, -0.0, 2.0, 2.0, np.nextafter(2.0, np.inf), 4.0, 4.0, 8.0]
    if api == "native":
        result = population.survexp(
            table, [[1], [2], [1], [2]], group=[0, 0, 1, 1], times=times, scale=2.5
        )
    else:
        result = r.survexp(
            "~ g",
            {"sex": ["male", "female", "male", "female"], "g": ["a", "a", "b", "b"]},
            ratetable=table,
            times=times,
            scale=2.5,
        )
    expected = np.exp(-np.asarray(times)[:, None] * [0.01, 0.03]).mean(axis=1)
    assert [value.hex() for value in result.time] == [float(value / 2.5).hex() for value in times]
    np.testing.assert_allclose(result.surv, np.repeat(expected[:, None], 2, axis=1), rtol=2e-14)
    np.testing.assert_array_equal(result.n_risk, np.full((len(times), 2), 2))
    assert result.method == "Ederer"
    for index in [1, 2, 4, 7]:
        assert result.surv[index] == result.surv[index - 1]


@pytest.mark.parametrize("api", ["native", "formula"])
def test_all_zero_requests_preserve_rows_and_empty_followup(api):
    table = population.RateTable([1], ["age"], [["0"]], [[0]], [2], [0.01])
    times = [0.0, -0.0, 0.0]
    if api == "native":
        result = population.survexp(table, [[0]], times=times)
        surv, n_risk = result.surv, result.n_risk
    else:
        result = r.survexp("~ 1", {"age": [0]}, ratetable=table, times=times)
        surv, n_risk = [[value] for value in result.surv], [[value] for value in result.n_risk]
    assert [value.hex() for value in result.time] == [value.hex() for value in times]
    assert surv == [[0.0], [0.0], [0.0]]
    assert n_risk == [[0.0], [0.0], [0.0]]
