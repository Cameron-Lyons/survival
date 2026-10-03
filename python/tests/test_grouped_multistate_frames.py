"""Grouped multi-state tables match stock matrices by state-column position.

The fixture retains raw stock fit[, states] matrices, including repeated names.
Stock [.survfitms leaves n.censor unsliced, which can break summary(...,
data.frame=TRUE). The reference selects censor counts from the original fit's
matrix and assembles the remaining raw selected columns without that summary.
"""

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/grouped_multistate_frame_reference.json").read_text()
)


@lru_cache(None)
def source_fit(se_fit, initial_row):
    data = dict(REFERENCE["data"])
    for name, levels in REFERENCE["levels"].items():
        data[name] = r._r_factor(data[name], levels)
    fit = r.survfit("Surv(time,event) ~ group", data, weights="weight", se_fit=se_fit)
    return r.survfit0(fit) if initial_row else fit


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_grouped_state_columns_match_stock_selected_matrices(case):
    selected = r._subset_survfit_multistate(
        source_fit(case["se_fit"], case["initial_row"]), case["selection"]
    )
    curves = r._survfit_strata_curves(selected)
    assert list(curves) == [group["label"] for group in case["groups"]]
    for group in case["groups"]:
        curve = curves[group["label"]]
        raw = group["raw_selected"]
        assert curve.states == raw["states"]
        # n.censor in raw stock subsets retains all original columns. Its
        # correctly selected values are checked in the complete table below.
        for name, values in raw.items():
            if name not in {"states", "n.censor"}:
                np.testing.assert_allclose(
                    getattr(curve, name.replace(".", "_")),
                    np.asarray(values, dtype=float),
                    rtol=2e-12,
                    atol=1e-14,
                )
    frame = r.as_data_frame(curves)
    expected = case["expected"]
    assert list(frame) == list(expected)
    assert len(frame["time"]) == len(selected.time) * len(selected.states)
    for name, values in frame.items():
        if name in {"strata", "state"}:
            assert values == expected[name]
        else:
            np.testing.assert_allclose(values, expected[name], rtol=2e-12, atol=1e-14)
    # Exported table columns are independent snapshots of the selected curves.
    frame["pstate"][0] = 999
    np.testing.assert_allclose(
        r.as_data_frame(curves)["pstate"], expected["pstate"], rtol=2e-12, atol=1e-14
    )


def test_grouped_state_tables_reject_incompatible_columns():
    curves = r._survfit_strata_curves(source_fit(True, False))
    first = next(iter(curves))
    bad = {
        **curves,
        first: dataclasses.replace(curves[first], states=list(reversed(curves[first].states))),
    }
    with pytest.raises(ValueError, match="share state columns"):
        r.as_data_frame(bad)
    bad = {**curves, first: dataclasses.replace(curves[first], std_err=None)}
    with pytest.raises(ValueError, match="share tabular columns"):
        r.as_data_frame(bad)
