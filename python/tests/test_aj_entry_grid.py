"""Entry curves use stock survival 3.8-12's subject-boundary reporting grid."""

import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads((Path(__file__).parent / "fixtures/aj_entry_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("api", ["formula", "native"])
def test_entry_grid_counts_and_estimates_match_current_stock_r(case, api):
    data = {
        name: [values[row - 1] for row in case["rows"]]
        for name, values in REFERENCE["data"].items()
    }
    options = {"entry": case["entry"], "time0": case["time0"], "influence": True}
    if case["start_time"] is not None:
        options["start_time"] = case["start_time"]
    if api == "formula":
        data["event"] = r._r_factor(data["event"], REFERENCE["event_levels"])
        fit = r.survfit(
            "Surv(start, stop, event) ~ " + ("group" if case["grouped"] else "1"),
            data,
            id="id",
            weights="weight" if case["weighted"] else None,
            **options,
        )
    else:
        fit = survival.surv_analysis.survfitaj(
            data["stop"],
            [REFERENCE["event_levels"].index(event) for event in data["event"]],
            REFERENCE["event_levels"][1:],
            start=data["start"],
            id=data["id"],
            weights=data["weight"] if case["weighted"] else None,
            strata=[0 if label == "one" else 1 for label in data["group"]]
            if case["grouped"]
            else None,
            **options,
        )
    expected = case["expected"]
    assert fit.states == expected["states"]
    assert fit.n == expected["n"]
    assert fit.n_id == expected["n_id"]
    strata = list(fit.strata.values()) if isinstance(fit.strata, dict) else fit.strata
    assert strata == expected["strata"]
    for name, values in expected.items():
        if name in {"states", "n", "n_id", "strata"}:
            continue
        actual = getattr(fit, name)
        if name == "influence_pstate":
            assert len(actual) == len(values)
            for influence, reference in zip(actual, values, strict=True):
                np.testing.assert_allclose(influence.values, reference, rtol=2e-12, atol=1e-14)
            continue
        if name == "counts" and values is not None:
            for field, reference in values.items():
                if reference is None:
                    assert getattr(actual, field) is None
                else:
                    np.testing.assert_equal(getattr(actual, field), reference)
            continue
        if values is None:
            assert actual is None, name
        else:
            np.testing.assert_allclose(
                actual,
                np.asarray(values, dtype=float),
                rtol=2e-12,
                atol=1e-14,
                err_msg=f"{case['name']}: {name}",
            )


@pytest.mark.parametrize("api", ["formula", "native"])
@pytest.mark.parametrize("entry", [False, True])
@pytest.mark.parametrize("timefix", [False, True])
def test_subject_gaps_are_rejected_before_constructing_an_entry_grid(api, entry, timefix):
    """Stock survival rejects this gap in survcheck before choosing grid rows."""

    def fit(second_start):
        if api == "native":
            return survival.surv_analysis.survfitaj(
                [2.0, 5.0, 4.0],
                [0, 0, 1],
                ["a"],
                start=[0.0, second_start, 0.0],
                id=[2, 2, 3],
                entry=entry,
                timefix=timefix,
            )
        return r.survfit(
            "Surv(start, stop, event) ~ 1",
            {
                "start": [0.0, second_start, 0.0],
                "stop": [2.0, 5.0, 4.0],
                "event": r._r_factor(["censor", "censor", "a"], ["censor", "a"]),
                "id": [2, 2, 3],
            },
            id="id",
            entry=entry,
            timefix=timefix,
        )

    with pytest.raises(ValueError, match="survcheck"):
        fit(2.75)
    fit(2.0)
